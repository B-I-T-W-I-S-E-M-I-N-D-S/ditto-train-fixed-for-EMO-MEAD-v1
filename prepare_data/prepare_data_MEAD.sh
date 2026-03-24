#!/usr/bin/env bash
# prepare_data_MEAD.sh
#
# Runs the full feature-extraction pipeline for each split of the MEAD dataset.
#
# Usage:
#   bash prepare_data/prepare_data_MEAD.sh
#
# Outputs per split (e.g. example/MEAD/train/):
#   data_info.json
#   data_list.json
#   data_preload.pkl     (optional, comment out if not needed)
#
# Requires:
#   prepare_data/prepare_data.sh  (the original flat-dataset script)
#   example/get_data_info_json_for_MEAD.py

set -euo pipefail

DITTO_ROOT_DIR="C:\\Users\\USERAS\\Downloads\\sazzad\\ditto\\ditto-talkinghead-train"

DITTO_PYTORCH_PATH="${DITTO_ROOT_DIR}\\checkpoints\\ditto-pytorch"
HUBERT_ONNX="${DITTO_PYTORCH_PATH}\\aux_models\\hubert_streaming_fix_kv.onnx"
MP_FACE_LMK_TASK="${DITTO_PYTORCH_PATH}\\aux_models\\face_landmarker.task"

MEAD_ROOT="${DITTO_ROOT_DIR}\\example\\MEAD"
PREPARE_SCRIPTS="${DITTO_ROOT_DIR}\\prepare_data\\scripts"

SECONDS=0

# ─── Step 1: Generate data_info.json for every split ───────────────────────
echo "=== Generating data_info.json for all MEAD splits ==="
python "${DITTO_ROOT_DIR}\\example\\get_data_info_json_for_MEAD.py" \
    --mead_root "${MEAD_ROOT}" \
    --splits train val test


# ─── Step 2: Run preprocessing per split ────────────────────────────────────

for SPLIT in train val test; do
    SPLIT_DIR="${MEAD_ROOT}\\${SPLIT}"
    DATA_INFO_JSON="${SPLIT_DIR}\\data_info.json"
    DATA_LIST_JSON="${SPLIT_DIR}\\data_list.json"
    DATA_PRELOAD_PKL="${SPLIT_DIR}\\data_preload.pkl"

    if [ ! -f "${DATA_INFO_JSON}" ]; then
        echo "[SKIP] No data_info.json found for split=${SPLIT}"
        continue
    fi

    echo ""
    echo "=== Processing split: ${SPLIT} ==="
    echo "    data_info_json : ${DATA_INFO_JSON}"

    cd "${DITTO_ROOT_DIR}\\prepare_data"

    # check ckpt
    python scripts/check_ckpt_path.py --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"

    ### VIDEO ###
    python scripts/crop_video_by_LP.py -i "${DATA_INFO_JSON}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"

    ### AUDIO ###
    python scripts/extract_audio_from_video.py -i "${DATA_INFO_JSON}"

    ### FEATURES ###
    # audio feat
    python scripts/extract_audio_feat_by_Hubert.py -i "${DATA_INFO_JSON}" \
        --Hubert_onnx "${HUBERT_ONNX}"

    # motion feat (normal + flipped for train only)
    python scripts/extract_motion_feat_by_LP.py -i "${DATA_INFO_JSON}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"
    if [ "${SPLIT}" = "train" ]; then
        python scripts/extract_motion_feat_by_LP.py -i "${DATA_INFO_JSON}" \
            --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" --flip_flag
    fi

    # eye feat (normal + flipped for train only)
    python scripts/extract_eye_ratio_from_video.py -i "${DATA_INFO_JSON}" \
        --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}"
    if [ "${SPLIT}" = "train" ]; then
        python scripts/extract_eye_ratio_from_video.py -i "${DATA_INFO_JSON}" \
            --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" --flip_lmk_flag
    fi

    # emo feat
    python scripts/extract_emo_feat_from_video.py -i "${DATA_INFO_JSON}"

    ### DATA LIST ###
    FLIP_FLAG=""
    if [ "${SPLIT}" = "train" ]; then
        FLIP_FLAG="--with_flip"
    fi

    python scripts/gather_data_list_json_for_train.py \
        -i "${DATA_INFO_JSON}" \
        -o "${DATA_LIST_JSON}" \
        --use_emo \
        --use_eye_open \
        --use_eye_ball \
        ${FLIP_FLAG}

    echo "    data_list_json : ${DATA_LIST_JSON}"

    ### PRELOAD PKL (optional – comment out to skip) ###
    python scripts/preload_train_data_to_pkl.py \
        --data_list_json  "${DATA_LIST_JSON}" \
        --data_preload_pkl "${DATA_PRELOAD_PKL}" \
        --use_sc \
        --use_emo \
        --use_eye_open \
        --use_eye_ball \
        --motion_feat_dim 265

    echo "    data_preload_pkl : ${DATA_PRELOAD_PKL}"

    cd "${DITTO_ROOT_DIR}"
done


echo ""
echo "=== prepare_data_MEAD.sh DONE ==="
echo "Results:"
for SPLIT in train val test; do
    echo "  ${SPLIT} → ${MEAD_ROOT}\\${SPLIT}\\data_list.json"
done
echo "Elapsed time: ${SECONDS}s"
