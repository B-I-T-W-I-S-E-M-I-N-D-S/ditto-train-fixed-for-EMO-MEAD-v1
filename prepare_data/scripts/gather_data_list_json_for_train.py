import os
from tqdm.contrib import tzip
from tqdm import trange
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback
import numpy as np

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import load_json, dump_json


"""
[
    [motion_npy, aud_npy, frame_num]
]
"""


def check_one_v2(data, N_thre=81):
    ns = []
    for k, v in data.items():

        if not os.path.isfile(v):
            return False, None
        
        n = np.load(v).shape[0]
        ns.append(n)

    N = min(ns)
    if N < N_thre:
        return False, None
    return True, N


def gather_and_filter_data_list_for_s2_v2(
    data_list_dict, 
    save_json='',
    use_emo=False,
    use_eye_open=False,
    use_eye_ball=False,
    use_lmk=False,
    flip=False,
    aud_feat_name='hubert_aud_npy_list',
):
    """
    frame_num, mtn, aud, emo, eye_open, eye_ball
    LP_npy_list, hubert_aud_npy_list, emo_npy_list, eye_open_npy_list, eye_ball_npy_list
    Also carries emotion_label_int_list -> 'label' if present (MEAD datasets).
    """
    lst = []
    num_v = len(data_list_dict[aud_feat_name])

    # Optional MEAD emotion labels
    emotion_label_int_list = data_list_dict.get('emotion_label_int_list', [])
    has_labels = len(emotion_label_int_list) == num_v

    for i in trange(num_v):
        try:
            data = {
                'mtn': data_list_dict['LP_npy_list'][i],
                'aud': data_list_dict[aud_feat_name][i],
            }
            if use_emo:
                data['emo'] = data_list_dict['emo_npy_list'][i]
            if use_eye_open:
                data['eye_open'] = data_list_dict['eye_open_npy_list'][i]
            if use_eye_ball:
                data['eye_ball'] = data_list_dict['eye_ball_npy_list'][i]
            if use_lmk:
                data['lmk'] = data_list_dict['MP_lmk_npy_list'][i]

            if flip:
                for k in ['mtn', 'eye_open', 'eye_ball', 'lmk']:
                    if k in data:
                        data[k] = flip_path(data[k])

            flag, N = check_one_v2(data)
            if not flag:
                continue

            data['frame_num'] = N

            # Propagate MEAD emotion label if available
            if has_labels:
                data['label'] = int(emotion_label_int_list[i])

            lst.append(data)
        except:
            traceback.print_exc()

    print(len(lst))
    if save_json:
        dump_json(lst, save_json)
    return lst


def flip_path(p):
    items = p.split('/')
    items[-2] = items[-2] + '_flip'
    p = '/'.join(items)
    return p


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data info json
    output_data_json: Annotated[str, tyro.conf.arg(aliases=["-o"])] = ""  # s2 data list json: [[motion_npy, aud_npy, frame_num]]

    use_emo: bool = False    # use_emo flag
    use_eye_open: bool = False    # use_eye_open flag
    use_eye_ball: bool = False    # use_eye_ball flag

    use_lmk: bool = False    # use_lmk flag

    dataset_version: str = "v2"    # dataset version: [v1, v2]

    with_flip: bool = False    # with flip flag

    aud_feat_name: str = "hubert_aud_npy_list"    # aud_feat_key: ['hubert_aud_npy_list']



def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json
    assert opt.output_data_json

    data_info = load_json(opt.input_data_json)

    if opt.dataset_version in ['v2']:
        if opt.with_flip:
            lst = gather_and_filter_data_list_for_s2_v2(
                data_info, 
                save_json='',
                use_emo=opt.use_emo,
                use_eye_open=opt.use_eye_open,
                use_eye_ball=opt.use_eye_ball,
                use_lmk=opt.use_lmk,
                aud_feat_name=opt.aud_feat_name,
            )
            flip_lst = gather_and_filter_data_list_for_s2_v2(
                data_info, 
                save_json='',
                use_emo=opt.use_emo,
                use_eye_open=opt.use_eye_open,
                use_eye_ball=opt.use_eye_ball,
                use_lmk=opt.use_lmk,
                flip=True,
                aud_feat_name=opt.aud_feat_name,
            )
            lst = lst + flip_lst
            dump_json(lst, opt.output_data_json)

        else:
            gather_and_filter_data_list_for_s2_v2(
                data_info, 
                save_json=opt.output_data_json,
                use_emo=opt.use_emo,
                use_eye_open=opt.use_eye_open,
                use_eye_ball=opt.use_eye_ball,
                use_lmk=opt.use_lmk,
                aud_feat_name=opt.aud_feat_name,
            )
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()

    """
    data_info = {
        'fps25_video_list': fps25_video_list,
        'video_list': video_list,
        'wav_list': wav_list,
        'hubert_aud_npy_list': hubert_aud_npy_list,
        'LP_pkl_list': LP_pkl_list,
        'LP_npy_list': LP_npy_list,
        'MP_lmk_npy_list': MP_lmk_npy_list,
        'eye_open_npy_list': eye_open_npy_list,
        'eye_ball_npy_list': eye_ball_npy_list,
        'emo_npy_list': emo_npy_list,
    }
    """
