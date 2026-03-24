from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR)))
DEFAULT_EXPERIMENT_DIR = os.path.join(PROJECT_DIR, 'experiments', 's2')


class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __repr__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)
    

@dataclass(repr=False)  # use repr from PrintableConfig
class TrainOptions(PrintableConfig):
    ########## experiment ##########
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR    # experiment_dir
    experiment_name: str = ""   # experiment_name

    ########## dataset ##########
    data_list_json: str = ""    # train data list: [[kps_npy, aud_npy, frame_num],]
    data_preload: bool = False  # data_preload flag
    data_preload_pkl: str = ""  # save to data_preload_pkl
    reprepare_idx_map: bool = False    # reprepare_idx_map flag for dataset
    data_cache: bool = False    # data_cache flag

    # Optional split filtering (HDTF-style .txt lists of video filenames).
    # Lines can be "name.mp4" or "name" (basename only).
    train_split_txt: str = ""   # e.g. splits/train.txt
    val_split_txt: str = ""     # e.g. splits/validation.txt
    test_split_txt: str = ""    # e.g. splits/test.txt
    split_strict: bool = False  # if True, error when split names not found in data_list_json

    # MEAD hierarchical dataset: per-split data_list JSONs.
    # When these are set the trainer uses them directly (no split_txt filtering needed).
    # Leave as "" to fall back to the classic data_list_json + split_txt approach.
    mead_train_data_list_json: str = ""   # e.g. example/MEAD/train/data_list.json
    mead_val_data_list_json: str = ""     # e.g. example/MEAD/val/data_list.json
    mead_test_data_list_json: str = ""    # e.g. example/MEAD/test/data_list.json

    mtn_mean_var_npy: str = ""    # for mtn norm

    motion_feat_start: int = 0  # motion feat start dim
    motion_feat_offset_dim_se: tuple[int, ...] = ()    # cal offset dim range for part D-S

    use_emo: bool = False    # use_emo flag (static clip-averaged emotion label in cond)
    use_eye_open: bool = False    # use_eye_open flag
    use_eye_ball: bool = False    # use_eye_ball flag
    use_sc: bool = False    # use source canonical keypoints flag
    use_last_frame: bool = False    # use last frame as cond frame flag
    use_lmk: bool = False    # use mediapipe lmk as cond flag
    use_cond_end: bool = False    # use clip start and end frame as cond flag

    # ── Emotion-aware audio module (MEMO-inspired) ────────────────────────
    use_emotion: bool = False   # enable EmotionEncoder + EmoAdaLN in DiT
    emo_dim: int = 128          # emotion embedding dimension
    hubert_dim: int = 1024      # HuBERT feature dim (first N dims of audio cond)
    lambda_emo: float = 0.1     # weight for emotion consistency loss
    # Stage-2 training: freeze pre-trained DiT backbone, only train emotion modules
    freeze_backbone: bool = False   # freeze DiT backbone during emotion fine-tuning

    dataset_version: str = "v2"    # dataset version: [v1, v2]

    ########## model ##########
    motion_feat_dim: int = 265          # motion_feat_dim
    audio_feat_dim: int = 1103     # audio_feat_dim (1024 + 63 + 8 + 2 + 6)
    seq_frames: int = int(3.2 * 25)     # clip length

    ########## train ##########
    use_accelerate: bool = True         # use_accelerate flag for multi gpu
    epochs: int = 1000      # epochs
    batch_size: int = 512   # batch_size
    num_workers: int = 0    # num_workers
    lr: float = 1e-4        # lr
    part_w_dict_json: str = ""    # part loss weights dict json
    use_last_frame_loss: bool = False    # use_last_frame_loss flag
    use_reg_loss: bool = False    # use_reg_loss flag
    dim_ws_npy: str = ""    # dim_ws npy

    checkpoint: str = ""        # checkpoint for load

    save_ckpt_freq: int = 50    # save ckpt freq (epoch)

    # Validation (optional)
    val_freq: int = 1           # run validation every N epochs (if val_split_txt set)
    val_batch_size: int = 0     # 0 => use batch_size
    val_num_workers: int = 0    # 0 => use num_workers


def check_train_opt(opt: TrainOptions):
    assert opt.experiment_dir, opt.experiment_dir
    assert opt.experiment_name, opt.experiment_name
    # Accept either classic data_list_json or MEAD per-split JSONs
    has_data = opt.data_list_json or opt.mead_train_data_list_json
    assert has_data, (
        "Must set either --data_list_json or --mead_train_data_list_json"
    )
