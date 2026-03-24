<h2 align='center'>Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis</h2>

<div align='center'>
    <a href=""><strong>Tianqi Li</strong></a>
    ·
    <a href=""><strong>Ruobing Zheng</strong></a><sup>†</sup>
    ·
    <a href=""><strong>Minghui Yang</strong></a>
    ·
    <a href=""><strong>Jingdong Chen</strong></a>
    ·
    <a href=""><strong>Ming Yang</strong></a>
</div>
<div align='center'>
Ant Group
</div>
<br>
<div align='center'>
    <a href='https://arxiv.org/abs/2411.19509'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
    <a href='https://digital-avatar.github.io/ai/Ditto/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://huggingface.co/digital-avatar/ditto-talkinghead'><img src='https://img.shields.io/badge/Model-HuggingFace-yellow'></a>
    <a href='https://github.com/antgroup/ditto-talkinghead'><img src='https://img.shields.io/badge/Code-GitHub-purple'></a>
    <!-- <a href='https://github.com/antgroup/ditto-talkinghead'><img src='https://img.shields.io/github/stars/antgroup/ditto-talkinghead?style=social'></a> -->
    <a href='https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing'><img src='https://img.shields.io/badge/Demo-Colab-orange'></a>
</div>
<br>
<div align="center">
    <video style="width: 95%; object-fit: cover;" controls loop src="https://github.com/user-attachments/assets/ef1a0b08-bff3-4997-a6dd-62a7f51cdb40" muted="false"></video>
    <p>
    ✨  For more results, visit our <a href="https://digital-avatar.github.io/ai/Ditto/"><strong>Project Page</strong></a> ✨ 
    </p>
</div>


## 📌 Updates
* [2025.11.12] 🔥🔥 We noticed the community's enthusiasm for open-source training code. [Training code](https://github.com/antgroup/ditto-talkinghead/tree/train) is now available, since there have been multiple versions and limited time to organize, it may differ slightly from the paper version.
* [2025.07.11] 🔥 The [PyTorch model](#-pytorch-model) is now available.
* [2025.07.07] 🔥 Ditto is accepted by ACM MM 2025.
* [2025.01.21] 🔥 We update the [Colab](https://colab.research.google.com/drive/19SUi1TiO32IS-Crmsu9wrkNspWE8tFbs?usp=sharing) demo, welcome to try it. 
* [2025.01.10] 🔥 We release our inference [codes](https://github.com/antgroup/ditto-talkinghead) and [models](https://huggingface.co/digital-avatar/ditto-talkinghead).
* [2024.11.29] 🔥 Our [paper](https://arxiv.org/abs/2411.19509) is in public on arxiv.

 

## 🔍 Overview
This is the **train branch**, containing code for **training the model**. For inference code, please switch to the [`main`](https://github.com/antgroup/ditto-talkinghead) branch.

<!-- This is the **inference branch**. For training code, please switch to the `train` branch. -->


## 🛠️ Environment

Tested Environment
- System: Centos 7.2  
- GPU: A100  
- Python: 3.10  


Clone the codes from [GitHub](https://github.com/antgroup/ditto-talkinghead) and switch to `train` branch :  
```bash
git clone https://github.com/antgroup/ditto-talkinghead
cd ditto-talkinghead

git checkout train
```

Create `conda` environment:
```bash
conda env create -f environment.yaml -n ditto_train
conda activate ditto_train
```

***If you have trouble setting up the environment with Conda, feel free to use your preferred method based on the dependencies listed in [environment.yaml](environment.yaml).***  


## 📥 Checkpoints Preparation

To begin with, acquire the model used for data processing. This requirement is similar to the inference setup in the main branch. Please download the necessary checkpoints from [HuggingFace](https://huggingface.co/digital-avatar/ditto-talkinghead). 

```bash
git lfs install
git clone https://huggingface.co/digital-avatar/ditto-talkinghead checkpoints
```

For the preprocessing of training data, only the model located in the `ditto_pytorch` directory is required, as illustrated below:

```text
./checkpoints/
├── ...
└── ditto_pytorch
    ├── aux_models
    │   ├── 2d106det.onnx
    │   ├── det_10g.onnx
    │   ├── face_landmarker.task
    │   ├── hubert_streaming_fix_kv.onnx
    │   └── landmark203.onnx
    └── models
        ├── appearance_extractor.pth
        ├── decoder.pth
        ├── motion_extractor.pth
        ├── stitch_network.pth
        ├── warp_network.pth
        └── ...

```


## ⭕ Quick Start

To quickly get started, we have provided a few example videos under the `example/trainset_example` directory as training data. Before diving into the detailed steps, we will first use these example datasets to walk through the entire data processing and training pipeline.  

**Note: This quick start guide only uses a small amount of data and limited training steps to demonstrate the full workflow. To achieve reasonable generation performance, more high-quality training data and longer training are required.**  


```shell
DITTO_PATH="<your-ditto-talkinghead-absolute-path>"


## Prepare data_info.json
python example/get_data_info_json_for_trainset_example.py
# you will get `example/trainset_example/data_info.json`


## Process Videos into Training Features
DATA_INFO_JSON="${DITTO_PATH}/example/trainset_example/data_info.json"
DATA_LIST_JSON="${DITTO_PATH}/example/trainset_example/data_list.json"
DATA_PRELOAD_PKL="${DITTO_PATH}/example/trainset_example/data_preload.pkl"

bash prepare_data/prepare_data.sh ${DATA_INFO_JSON} ${DATA_LIST_JSON} ${DATA_PRELOAD_PKL}
# results in `example/trainset_example/`


## MotionDiT Training
accelerate config

accelerate launch MotionDiT/train.py \
  --experiment_dir experiments/s2 \
  --experiment_name mead_run1 \
  --data_list_json example/MEAD/train/data_list.json \
  --mead_train_data_list_json example/MEAD/train/data_list.json \
  --mead_val_data_list_json example/MEAD/val/data_list.json \
  --use_sc \
  --use_last_frame \
  --use_last_frame_loss \
  --use_emo \
  --use_emotion \
  --use_eye_open \
  --use_eye_ball \
  --audio_feat_dim 1103 \
  --motion_feat_dim 265 \
  --emo_dim 128 \
  --hubert_dim 1024 \
  --lambda_emo 0.1 \
  --batch_size 100 \
  --num_workers 8 \
  --epochs 3 \
  --save_ckpt_freq 1 \
  --data_preload \
  --data_preload_pkl example/MEAD/train/data_preload.pkl

# training outputs in `example/exp_dir/exp_trainset_example`
```


## 📁 Data Preparation

### Process Training Data

> **Before proceeding, ensure that your video data has been preprocessed. This includes cleaning, filtering, shot detection, and frame rate normalization to 25fps. The final videos should be in MP4 format with synchronized audio and video, and each frame should clearly show the target face.**  


**step 1: Prepare `data_info.json`**  
Based on your video data, manually create a corresponding `data_info.json` file (you can refer to the example: [example/get_data_info_json_for_trainset_example.py](example/get_data_info_json_for_trainset_example.py)).
The structure of the JSON file is as follows:

```python
# data_info.json

# Used to specify the mapping between all video files and their corresponding feature storage paths. All paths in this file must be absolute paths.

data_info = {
    'fps25_video_list': fps25_video_list,           # [*.mp4, ...], your video data
    'video_list': video_list,                       # [*.mp4, ...], cropped video
    'wav_list': wav_list,                           # [*.wav, ...], audio
    'hubert_aud_npy_list': hubert_aud_npy_list,     # [*.npy, ...], audio feat
    'LP_pkl_list': LP_pkl_list,                     # [*.pkl, ...], LP motion
    'LP_npy_list': LP_npy_list,                     # [*.npy, ...], LP motion
    'MP_lmk_npy_list': MP_lmk_npy_list,             # [*.npy, ...], MP lmk
    'eye_open_npy_list': eye_open_npy_list,         # [*.npy, ...], eye open state
    'eye_ball_npy_list': eye_ball_npy_list,         # [*.npy, ...], eye ball state
    'emo_npy_list': emo_npy_list,                   # [*.npy, ...], emo label
}

```

**Step 2: Process Videos into Training Features**  
Based on the `data_info.json`, process the raw videos into the feature format required for training.


```bash
DATA_INFO_JSON="<path-to-data-info-json>"       # input:  data_info.json
DATA_LIST_JSON="<path-to-data-list-json>"       # output: data_list.json    (for train)
DATA_PRELOAD_PKL="<path-to-data-preload-pkl>"   # output: data_preload.pkl  (for train)

bash prepare_data/prepare_data.sh ${DATA_INFO_JSON} ${DATA_LIST_JSON} ${DATA_PRELOAD_PKL}

```


## 🏋️ Model Training

Don't forget to run accelerate config to set up the default configuration for Accelerate, or include the appropriate accelerate arguments in the training command shown below.

```shell

cd MotionDiT

EXP_DIR="<path-to-experiment-dir>"
EXP_NAME="<experiment-name>"

DATA_LIST_JSON="<path-to-data-list-json>"
DATA_PRELOAD_PKL="<path-to-data-preload-pkl>"


accelerate launch train.py \
--experiment_dir ${EXP_DIR} \
--experiment_name ${EXP_NAME} \
--use_sc \
--use_last_frame \
--use_last_frame_loss \
--use_emo \
--use_eye_open \
--use_eye_ball \
--audio_feat_dim 1103 \
--motion_feat_dim 265 \
--batch_size 1024 \
--num_workers 8 \
--epochs 500 \
--save_ckpt_freq 1 \
--data_list_json ${DATA_LIST_JSON} \
--data_preload \
--data_preload_pkl ${DATA_PRELOAD_PKL} \

# You can find the training outputs in `${EXP_DIR}/${EXP_NAME}`.

```



## ⚖️ License
This repository is released under the Apache-2.0 license as found in the [LICENSE](LICENSE) file.

## 📚 Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@article{li2024ditto,
    title={Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis},
    author={Li, Tianqi and Zheng, Ruobing and Yang, Minghui and Chen, Jingdong and Yang, Ming},
    journal={arXiv preprint arXiv:2411.19509},
    year={2024}
}
```


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/ditto-talkinghead&type=Date)](https://www.star-history.com/#antgroup/ditto-talkinghead&Date)


