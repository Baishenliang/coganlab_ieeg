import os
from ieeg.io import get_data, DataLoader
import numpy as np
from ieeg.arrays.label import LabeledArray, combine

# Sampling rate: 100hz
# Go event: onset at 50 sample
# Resp event: onset at 100 sample
def set_confusion_matrix(set_A, set_B):
    import numpy as np
    shared = len(set_A & set_B)  # 交集
    only_A = len(set_A - set_B)  # 仅在 A 中
    only_B = len(set_B - set_A)  # 仅在 B 中
    neither = 0  # 不涉及全集的话可以忽略

    matrix = np.array([[shared, only_A], [only_B, neither]])
    import pandas as pd

    # 使用 pandas DataFrame 添加标签
    df = pd.DataFrame(matrix,
                      index=["In Aaron's SM", "Not in Aaron's SM"],
                      columns=["In Baishen's SM", "Not in Baishen's SM"])

    return df

def group_elecs(all_sig: dict[str, np.ndarray] | LabeledArray, names: list[str],
                conds: tuple[str], wide: bool = False, resp_onset_thres_sec: float = 0
                ) -> (set[int], set[int], set[int], set[int],set[int]):

    resp_onset_thres=int(100+100*resp_onset_thres_sec)
    # threshold to separate pre articulation and post articulation windows for retrieving Sensorimotor electrodes

    sig_chans = set()
    AUD = set()
    SM = set()
    PROD = set()
    SM_baishen = set()
    for i, name in enumerate(names):
        for cond in conds:
            idx = i
            if wide:
                t_idx = slice(None)
            elif np.squeeze(all_sig).ndim == 2:
                t_idx = None
            elif cond in ["aud_ls", "aud_lm", "aud_jl"]:
                t_idx = slice(50, 100)
            else:
                t_idx = slice(75, 125)

            if np.any(all_sig[cond, idx, ..., t_idx] == 1):
                sig_chans |= {i}
                break

        if all_sig.ndim == 2:
            aud_slice = None
            go_slice = None
        elif all_sig.shape[2] == 1:
            aud_slice = slice(None)
            go_slice = slice(None)
        elif wide:
            aud_slice = slice(50, 175)
            go_slice = slice(50, None)
        else:
            aud_slice = slice(50, 100)
            go_slice = slice(75, 125)
            go_slice_baishen = slice(50, 150)
            resp_slice_baishen = slice(1, resp_onset_thres)

        audls_is = np.any(all_sig['aud_ls', idx, ..., aud_slice] == 1)
        audlm_is = np.any(all_sig['aud_lm', idx, ..., aud_slice] == 1)
        audjl_is = np.any(all_sig['aud_jl', idx, ..., aud_slice] == 1)
        mime_is = np.any(all_sig['go_lm', idx, ..., go_slice] == 1)
        speak_is = np.any(all_sig['go_ls', idx, ..., go_slice] == 1)
        speak_is_baishen = np.any(all_sig['go_ls', idx, ..., go_slice] == 1)
        resp_is = np.any(all_sig['resp', idx, ..., go_slice] == 1)
        resp_is_baishen = np.any(all_sig['resp', idx, ..., resp_slice_baishen] == 1)

        if audls_is and audlm_is and mime_is and (speak_is or resp_is):
            SM |= {i}
        elif audls_is and audlm_is and audjl_is:
            AUD |= {i}
        elif mime_is and (speak_is or resp_is):
            PROD |= {i}

        if audls_is and audlm_is and (speak_is_baishen and resp_is_baishen):
            SM_baishen |= {i}

    return AUD, SM, PROD, sig_chans, SM_baishen

fpath = os.path.expanduser("~/Box/CoganLab")
layout = get_data('SentenceRep', root=fpath)

conds = {"resp": (-1, 1), "aud_ls": (-0.5, 1.5),
                 "aud_lm": (-0.5, 1.5), "aud_jl": (-0.5, 1.5),
                 "go_ls": (-0.5, 1.5), "go_lm": (-0.5, 1.5),
                 "go_jl": (-0.5, 1.5)}
loader = DataLoader(layout, conds, 'significance', True, 'stats',
                   '.fif')
sigs = LabeledArray.from_dict(combine(loader.load_dict(
    dtype=bool, n_jobs=-1), (0, 2)), dtype=bool)
loader = DataLoader(layout, conds, 'zscore', False, 'stats',
                   '.fif')
zscores = LabeledArray.from_dict(combine(loader.load_dict(
    dtype=float, n_jobs=-1), (0, 2)), dtype=float)

ch_names = sigs.labels[1]
for thres in range(-19,0,1):
    AUD, SM, PROD, sig_chans, SM_baishen = group_elecs(sigs, sigs.labels[1], sigs.labels[0],resp_onset_thres_sec=0.01*thres)
    df=set_confusion_matrix(SM, SM_baishen)
    print(f'Pre-articulation end time aligned to motor onset: {np.round(0.01*thres,2)}s')
    print(df)
    print('\n')