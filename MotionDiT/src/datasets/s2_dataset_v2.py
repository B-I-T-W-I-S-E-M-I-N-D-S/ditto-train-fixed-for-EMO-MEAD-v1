# emo, eye, canonical keypoints
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import random
import traceback
from collections import Counter
from typing import Iterable, Optional, Union, Set, Dict, Any, List, Tuple

from ..utils.utils import load_json, load_pkl, dump_pkl


# ---------------------------------------------------------------------------
# Emotion class registry – matches MEAD folder names (sorted → int IDs).
# ---------------------------------------------------------------------------
EMOTION_CLASSES: List[str] = [
    "Angry",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised",
]
EMOTION_TO_INT: Dict[str, int] = {e: i for i, e in enumerate(EMOTION_CLASSES)}
INT_TO_EMOTION: Dict[int, str] = {i: e for i, e in enumerate(EMOTION_CLASSES)}


FPS = 25


def norm_by_mean_var(arr, v_mean_var):
    mean = np.broadcast_to(v_mean_var[0], arr.shape)
    var = np.broadcast_to(v_mean_var[1], arr.shape)
    norm_arr = (arr - mean) / np.sqrt(var)
    return norm_arr


def norm_by_mean_std(arr, v_mean_std):
    mean = np.broadcast_to(v_mean_std[0], arr.shape)
    std = np.broadcast_to(v_mean_std[1], arr.shape)
    norm_arr = (arr - mean) / std
    return norm_arr


def denorm_by_mean_var(arr, v_mean_var):
    mean = np.broadcast_to(v_mean_var[0], arr.shape)
    var = np.broadcast_to(v_mean_var[1], arr.shape)
    denorm_arr = arr * np.sqrt(var) + mean
    return denorm_arr


def _read_split_txt(split_txt: str) -> Set[str]:
    """
    Returns a set of sample basenames (without extension).
    Supports lines like:
      - foo.mp4
      - foo
      - subdir/foo.mp4
    Ignores empty lines and comments (# ...).
    """
    names: Set[str] = set()
    with open(split_txt, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            base = os.path.basename(line)
            if "." in base:
                base = base.rsplit(".", 1)[0]
            if base:
                names.add(base)
    return names


def _infer_sample_name_from_item(item: Dict[str, Any]) -> str:
    # Prefer motion feature path; fallback to audio path.
    p = item.get("mtn") or item.get("aud") or ""
    base = os.path.basename(p)
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base


def _filter_data_list_by_split(
    data_list: List[Dict[str, Any]],
    split_names: Set[str],
    *,
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filters data_list to only those whose inferred basename is in split_names.
    If strict=True, raises if any split_names are not found.
    Returns (filtered_list, stats).
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    dup = 0
    for item in data_list:
        name = _infer_sample_name_from_item(item)
        if not name:
            continue
        if name in by_name:
            dup += 1
            continue
        by_name[name] = item

    filtered: List[Dict[str, Any]] = []
    missing_in_json = 0
    for name in split_names:
        if name in by_name:
            filtered.append(by_name[name])
        else:
            missing_in_json += 1

    if strict and missing_in_json:
        missing_examples = sorted(list(split_names - set(by_name.keys())))[:20]
        raise FileNotFoundError(
            f"Split has {missing_in_json} names not present in data_list_json. "
            f"Examples: {missing_examples}"
        )

    stats = {
        "json_total": len(data_list),
        "json_unique_names": len(by_name),
        "json_duplicate_names_skipped": dup,
        "split_total_names": len(split_names),
        "split_missing_in_json": missing_in_json,
        "kept": len(filtered),
    }
    return filtered, stats

class Stage2Dataset(Dataset):
    def __init__(
            self, 
            data_list_json,
            seq_len=int(3.2 * FPS), 
            preload=False, 
            cache=False, 
            preload_pkl="", 
            motion_feat_dim=100, 
            motion_feat_start=0,
            motion_feat_offset_dim_se=None,
            use_eye_open=False,
            use_eye_ball=False,
            use_emo=False,
            use_sc=False,    # source canonical keypoints
            use_last_frame=False,    # last frame as cond frame
            use_lmk=False,   # mediapipe lmk (0, 1)
            use_cond_end=False,    # cond by clip end
            mtn_mean_var_npy="",
            reprepare_idx_map=False,
            split_txt: str = "",
            split_strict: bool = False,
            **kwargs):
        super().__init__()

        """
        data_list_json:
            [
                {
                    'frame_num': frame_num,
                    'aud': aud_npy,
                    'mtn': mtn_npy,
                    'emo': emo_npy,
                    'eye_open': eye_open_npy,
                    'eye_ball': eye_ball_npy,
                }
            ]
        motion_feat_offset_dim_se: for D-S
        """

        self.is_train = True
        self.preload = preload
        self.preload_pkl = preload_pkl
        self.cache = cache
        self.seq_len = seq_len
        self.motion_feat_dim = motion_feat_dim
        self.motion_feat_start = motion_feat_start
        self.motion_feat_offset_dim_se = motion_feat_offset_dim_se

        self.use_eye_open = use_eye_open
        self.use_eye_ball = use_eye_ball
        self.use_emo = use_emo
        self.use_sc = use_sc
        self.use_last_frame = use_last_frame
        self.use_lmk = use_lmk
        self.use_cond_end = use_cond_end

        self.data_list_json = data_list_json
        self.split_txt = split_txt
        self.split_strict = split_strict

        if preload and preload_pkl and os.path.isfile(preload_pkl):
            print('load data from preload_pkl:', preload_pkl)
            self.v_list, self.idx_map = load_pkl(preload_pkl)
            self.num_v = len(self.v_list)
            if reprepare_idx_map:
                print('reprepare_idx_map...')
                self.idx_map = self._prepare_idx_map()
            self.num_seq = len(self.idx_map)
        else:
            self.v_list = self._load_data(data_list_json, split_txt=split_txt, split_strict=split_strict)
            self.num_v = len(self.v_list)

            self.idx_map = self._prepare_idx_map()
            self.num_seq = len(self.idx_map)

        if preload and preload_pkl and not os.path.isfile(preload_pkl):
            print('save data to preload_pkl:', preload_pkl)
            dump_pkl([self.v_list, self.idx_map], preload_pkl)

        self.cache_dict = {}
        print(f'load [num_v: {self.num_v}, num_seq: {self.num_seq}]')

        self.mtn_mean_var = None
        if mtn_mean_var_npy:
            self.mtn_mean_var = np.load(mtn_mean_var_npy)    # [2, dim]
            self.mtn_mean_var[1][self.mtn_mean_var[1] == 0] = 1e-8
            self.mtn_mean_std = self.mtn_mean_var.copy()
            self.mtn_mean_std[1] = np.sqrt(self.mtn_mean_std[1])

            # process preload
            if preload:
                print("norm for preload")
                for v_idx in trange(len(self.v_list)):
                    self.v_list[v_idx]['mtn'] = norm_by_mean_std(self.v_list[v_idx]['mtn'], self.mtn_mean_std)

    def __len__(self):
        return self.num_seq
    
    def _load_one(self, data):
        ss = self.motion_feat_start
        ee = ss + self.motion_feat_dim

        frame_num = data['frame_num']
        mtn_arr = np.load(data['mtn'])[:frame_num]
        mtn = mtn_arr[:frame_num, ss:ee]   # [n, dim_mtn]
        aud = np.load(data['aud'])[:frame_num]    # [n, dim_aud]

        arr_data = {
            'frame_num': frame_num,
            'mtn': mtn,
            'aud': aud,
        }

        # Emotion integer label (MEAD datasets only; -1 if absent)
        arr_data['label'] = int(data.get('label', -1))
        if self.use_sc:
            sc = mtn_arr[:, 265:]    # [n, 63]
            arr_data['sc'] = sc

        if self.use_emo:
            emo = np.load(data['emo'])[:frame_num]   # [n, 8]
            arr_data['emo'] = emo

        if self.use_eye_open:
            eye_open = np.load(data['eye_open'])[:frame_num]    # [n, 2]
            arr_data['eye_open'] = eye_open

        if self.use_eye_ball:
            eye_ball = np.load(data['eye_ball'])[:frame_num]   # [n, 3, 2]
            eye_ball = eye_ball.reshape(frame_num, -1)    # [n, 6]
            arr_data['eye_ball'] = eye_ball
        
        if self.use_lmk:
            lmk = np.load(data['lmk'])[:frame_num]  # [n, 478, 3]
            lmk = lmk.reshape(frame_num, -1)
            arr_data['lmk'] = lmk

        return arr_data
    
    #def _load_data(self, data_list_json):
    def _load_data(self, data_list_json, split_txt: str = "", split_strict: bool = False):
        # List[Dict]: [{'frame_num', 'aud', 'mtn', ...}, ...]
        if isinstance(data_list_json, str):
            data_list = load_json(data_list_json)
        else:
            # allow passing an already-loaded list for flexibility
            data_list = data_list_json

        if split_txt:
            if not os.path.isfile(split_txt):
                raise FileNotFoundError(f"split_txt not found: {split_txt}")
            split_names = _read_split_txt(split_txt)
            data_list, stats = _filter_data_list_by_split(data_list, split_names, strict=split_strict)
            print(f"[Stage2Dataset] split_txt={split_txt} stats={stats}")

        # ── Emotion label statistics ────────────────────────────────────────
        label_counter: Counter = Counter()
        for item in data_list:
            lbl = item.get("label")
            if lbl is not None:
                label_counter[INT_TO_EMOTION.get(int(lbl), str(lbl))] += 1
        if label_counter:
            print(f"[Stage2Dataset] Emotion distribution ({sum(label_counter.values())} samples):")
            for emo in sorted(label_counter):
                print(f"  {emo:<12} : {label_counter[emo]}")
        else:
            print(f"[Stage2Dataset] No 'label' field found in data (non-MEAD dataset).")
        # ───────────────────────────────────────────────────────────────────

        if not self.preload:
            return data_list
        
        arr_data_list = []
        skipped = 0
        for data in tqdm(data_list):
            try:
                arr_data = self._load_one(data)
                arr_data_list.append(arr_data)
            except Exception:
                skipped += 1
                traceback.print_exc()
        if skipped:
            print(f"[Stage2Dataset] WARNING: skipped {skipped} corrupted/missing samples during preload.")

        return arr_data_list
    
    def _prepare_idx_map(self):
        # idx -> v_idx, f_idx
        num_v = self.num_v
        seq_len = self.seq_len

        idx_map = []
        for v_idx in range(num_v):
            num_f = self.v_list[v_idx]['frame_num']
            for f_idx in range(1, num_f - seq_len + 1):
                idx_map.append([v_idx, f_idx])
        return idx_map
    
    def getitem(self, idx):
        """
        return:
            kp_seq      # (B, L, kp_dim)
            kp_cond     # (B, kp_dim)
            aud_cond    # (B, L, aud_dim)
        """
        seq_len = self.seq_len

        v_idx, f_idx = self.idx_map[idx]

        if self.preload:
            arr_data = self.v_list[v_idx]
        else:
            if v_idx in self.cache_dict:
                arr_data = self.cache_dict[v_idx]
            else:
                data = self.v_list[v_idx]
                arr_data = self._load_one(data)
                if self.mtn_mean_var is not None:
                    arr_data['mtn'] = norm_by_mean_std(arr_data['mtn'], self.mtn_mean_std)
                if self.cache:
                    self.cache_dict[v_idx] = arr_data
        
        v_mtn = arr_data['mtn']
        v_aud = arr_data['aud']

        # if self.mtn_mean_var is not None:
        #     # v_mtn = norm_by_mean_var(v_mtn.copy(), self.mtn_mean_var)
        #     v_mtn = norm_by_mean_std(v_mtn, self.mtn_mean_std)

        if self.use_last_frame:
            kp_cond = v_mtn[f_idx - 1]
        else:
            kp_cond = v_mtn[random.randint(0, len(v_mtn) - 1)]

        if self.use_cond_end:
            if f_idx + seq_len < len(v_mtn):
                kp_cond_end = v_mtn[f_idx + seq_len]
            else:
                kp_cond_end = v_mtn[f_idx + seq_len - 1]
        else:
            kp_cond_end = None

        kp_seq = v_mtn[f_idx: f_idx + seq_len]
        aud_cond = v_aud[f_idx: f_idx + seq_len]

        if self.motion_feat_offset_dim_se:
            _s, _e = self.motion_feat_offset_dim_se
            kp_seq = kp_seq.copy()
            kp_seq[:, _s:_e] = kp_seq[:, _s:_e] - kp_cond[_s:_e][None]

        # other cond: emo, eye
        more_cond = []
        if self.use_emo:
            v_emo = arr_data['emo']
            emo_seq = v_emo[f_idx: f_idx + seq_len]   # [n, 8]
            emo_avg = np.mean(emo_seq, 0)   # [8]
            emo_avg_seq = np.stack([emo_avg] * seq_len, 0)   # [n, 8]
            more_cond.append(emo_avg_seq)
        
        if self.use_eye_open:
            v_eye_open = arr_data['eye_open']
            eye_open_seq = v_eye_open[f_idx: f_idx + seq_len]   # [n, 2]
            more_cond.append(eye_open_seq)

        if self.use_eye_ball:
            v_eye_ball = arr_data['eye_ball']
            eye_ball_seq = v_eye_ball[f_idx: f_idx + seq_len]   # [n, 6]
            more_cond.append(eye_ball_seq)

        rand_f_idx = random.randint(0, len(v_mtn) - 1)
        if self.use_sc:
            v_sc = arr_data['sc']
            sc = v_sc[rand_f_idx]    # [63]
            sc_seq = np.stack([sc] * seq_len, 0)    # [n, 63]
            more_cond.append(sc_seq)

        if self.use_lmk:
            v_lmk = arr_data['lmk']
            lmk = v_lmk[rand_f_idx]   # 478x3
            lmk_seq = np.stack([lmk] * seq_len, 0)  # [n, dim]
            more_cond.append(lmk_seq)
            
        if more_cond:
            cond_seq = np.concatenate([aud_cond] + more_cond, -1)    # [n, dim_cond]
        else:
            cond_seq = aud_cond

        data_dict = {
            'kp_seq': kp_seq,
            'kp_cond': kp_cond,
            'aud_cond': cond_seq,
            # Raw HuBERT features (first 1024 dims) for the EmotionEncoder.
            # The EmotionEncoder slices cond_embed internally, but we also
            # expose aud_raw here for convenience (e.g. offline pre-extraction).
            'aud_raw': aud_cond[:, :1024].copy(),   # [T, 1024]
            'idx': f'{idx}_{v_idx}_{f_idx}',
            'label': arr_data.get('label', -1),  # emotion class int (-1 if unknown)
        }

        if self.use_cond_end:
            data_dict['kp_cond_end'] = kp_cond_end
        
        return data_dict
    
    def __getitem__(self, idx):
        while True:
            try:
                return self.getitem(idx)
            except:
                traceback.print_exc()
                idx = random.randint(0, self.num_seq-1)
