"""
Generate data_info_{split}.json files for the MEAD hierarchical emotion dataset.

Expected directory layout:
    example/MEAD/
        train/
            Angry/  *.mp4
            Happy/  *.mp4
            ...
        val/
            ...
        test/
            ...

Outputs (one per split):
    example/MEAD/train/data_info.json
    example/MEAD/val/data_info.json
    example/MEAD/test/data_info.json

Each data_info.json is a dict-of-lists with the same schema as the original
trainset_example/data_info.json, plus two extra lists:
    - emotion_label_str_list  : e.g. ["Angry", "Angry", "Happy", ...]
    - emotion_label_int_list  : e.g. [0, 0, 3, ...]  (sorted class index)

Usage:
    python example/get_data_info_json_for_MEAD.py

    # or override root:
    python example/get_data_info_json_for_MEAD.py --mead_root example/MEAD
"""

import os
import json
import argparse
from collections import Counter


# ---------------------------------------------------------------------------
# Canonical emotion class list (sorted alphabetically → consistent int IDs).
# You can extend this list if new emotions are added to the dataset.
# ---------------------------------------------------------------------------
EMOTION_CLASSES = [
    "Angry",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised",
]

# Build a case-insensitive lookup so folder names like 'angry' still match.
_EMO_TO_INT = {e.lower(): i for i, e in enumerate(EMOTION_CLASSES)}
_EMO_CANONICAL = {e.lower(): e for e in EMOTION_CLASSES}


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_emotion_id(folder_name: str):
    """Return (canonical_str, int_id) for a given emotion folder name."""
    key = folder_name.lower()
    if key not in _EMO_TO_INT:
        return None, None
    return _EMO_CANONICAL[key], _EMO_TO_INT[key]


def scan_split(split_dir: str, save_dir: str):
    """
    Walk split_dir (e.g. .../MEAD/train) and collect paths for all .mp4 videos
    found under emotion sub-folders.

    save_dir is where extracted features will be written
    (mirrors the flat trainset_example layout but nested under split_dir).

    Returns a data_info dict-of-lists.
    """

    fps25_video_list = []
    video_list = []
    wav_list = []
    hubert_aud_npy_list = []
    LP_pkl_list = []
    LP_npy_list = []
    MP_lmk_npy_list = []
    eye_open_npy_list = []
    eye_ball_npy_list = []
    emo_npy_list = []
    emotion_label_str_list = []
    emotion_label_int_list = []

    skipped = []
    emotion_counter = Counter()

    emotion_folders = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )

    if not emotion_folders:
        print(f"  [WARN] No sub-folders found in {split_dir}")

    for emo_folder in emotion_folders:
        emo_str, emo_int = get_emotion_id(emo_folder)
        if emo_str is None:
            print(f"  [WARN] Unknown emotion folder '{emo_folder}' — skipping.")
            skipped.append(emo_folder)
            continue

        emo_dir = os.path.join(split_dir, emo_folder)
        mp4_files = sorted(
            f for f in os.listdir(emo_dir)
            if f.lower().endswith(".mp4")
        )

        if not mp4_files:
            print(f"  [WARN] No .mp4 files in {emo_dir}")
            continue

        for video_name in mp4_files:
            # Use emotion/name as the unique identifier to avoid cross-emotion
            # collisions (e.g. two subjects with the same filename in different
            # emotion folders).
            stem = video_name.rsplit(".", 1)[0]
            unique_name = f"{emo_str}_{stem}"

            fps25_video = os.path.join(emo_dir, video_name)

            fps25_video_list.append(fps25_video)
            video_list.append(f"{save_dir}/video/{unique_name}.mp4")
            wav_list.append(f"{save_dir}/wav/{unique_name}.wav")
            hubert_aud_npy_list.append(f"{save_dir}/hubert_aud_npy/{unique_name}.npy")
            LP_pkl_list.append(f"{save_dir}/LP_pkl/{unique_name}.pkl")
            LP_npy_list.append(f"{save_dir}/LP_npy/{unique_name}.npy")
            MP_lmk_npy_list.append(f"{save_dir}/MP_lmk_npy/{unique_name}.npy")
            eye_open_npy_list.append(f"{save_dir}/eye_open_npy/{unique_name}.npy")
            eye_ball_npy_list.append(f"{save_dir}/eye_ball_npy/{unique_name}.npy")
            emo_npy_list.append(f"{save_dir}/emo_npy/{unique_name}.npy")
            emotion_label_str_list.append(emo_str)
            emotion_label_int_list.append(emo_int)

            emotion_counter[emo_str] += 1

    data_info = {
        "fps25_video_list": fps25_video_list,
        "video_list": video_list,
        "wav_list": wav_list,
        "hubert_aud_npy_list": hubert_aud_npy_list,
        "LP_pkl_list": LP_pkl_list,
        "LP_npy_list": LP_npy_list,
        "MP_lmk_npy_list": MP_lmk_npy_list,
        "eye_open_npy_list": eye_open_npy_list,
        "eye_ball_npy_list": eye_ball_npy_list,
        "emo_npy_list": emo_npy_list,
        "emotion_label_str_list": emotion_label_str_list,
        "emotion_label_int_list": emotion_label_int_list,
    }

    total = len(fps25_video_list)
    print(f"  Total videos : {total}")
    for emo in sorted(emotion_counter):
        print(f"    {emo:<12} : {emotion_counter[emo]}")
    if skipped:
        print(f"  Skipped unknown emotion folders: {skipped}")

    return data_info


def main():
    parser = argparse.ArgumentParser(
        description="Generate data_info.json files for the MEAD dataset."
    )
    parser.add_argument(
        "--mead_root",
        default=None,
        help="Path to the MEAD root directory. "
             "Defaults to <this_script_dir>/MEAD",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to process (default: train val test)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mead_root = args.mead_root or os.path.join(script_dir, "MEAD")

    if not os.path.isdir(mead_root):
        raise FileNotFoundError(f"MEAD root not found: {mead_root}")

    print(f"MEAD root : {mead_root}")
    print(f"Splits    : {args.splits}")
    print(f"Emotion classes ({len(EMOTION_CLASSES)}): {EMOTION_CLASSES}")
    print()

    for split in args.splits:
        split_dir = os.path.join(mead_root, split)
        if not os.path.isdir(split_dir):
            print(f"[SKIP] Split directory not found: {split_dir}")
            continue

        # Feature files land alongside the split directory so the folder stays
        # self-contained:  MEAD/train/video/, MEAD/train/hubert_aud_npy/, …
        save_dir = split_dir

        print(f"[{split.upper()}] Scanning {split_dir} …")
        data_info = scan_split(split_dir, save_dir)

        out_path = os.path.join(split_dir, "data_info.json")
        dump_json(data_info, out_path)
        print(f"  Saved → {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
