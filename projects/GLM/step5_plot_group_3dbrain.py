#%% Import everything
import os
# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

import sys
import pandas as pd
import glm_utils as glm
import numpy as np
import itertools
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle


#%% Set parameters
with open('glm_config.json', 'r') as f:
    config = json.load(f)

# Extract parameters from config
Acoustic_col = config['Acoustic_col']
Phonemic_col = config['Phonemic_col']
Lexical_col = config['Lexical_col']

stat = "zscore"
glm_feas = ["Acoustic","Phonemic","Lexical"]
wordnesses = ["ALL"]#["ALL", "Word", "Nonword"]
cluster_twin=0.011
mean_word_len=0.62
auditory_decay=0.4
delay_len=0.5
# motor_prep_win=[-0.5,-0.1]
# motor_resp_win=[0.25,0.75]
Waveplot_wth=10 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots

#%% Plot brain
for wordness in wordnesses:
    subjs, _, _, chs, times = glm.fifread("Auditory", 'zscore', 'Repeat',wordness)
    with open(os.path.join('data', 'sig_idx.npy'), "rb") as f:
        sig_idx = pickle.load(f)

    chs_all = np.concatenate(chs, axis=0)

    for TypeLabel, chs_ov, base_sig, spec_sig in zip(
            ('Auditory', 'Delay', 'Response'),
            ([100, 10, 1], [100, 10, 1], [100, 10, 1]),
            (gp.set2arr(sig_idx[f"Auditory/Repeat/{wordness}/Acoustic/aud"] | sig_idx[f"Auditory/Repeat/{wordness}/Phonemic/aud"] | sig_idx[f"Auditory/Repeat/{wordness}/Lexical/aud"],len(chs_all)),
            gp.set2arr(sig_idx[f"Auditory/Repeat/{wordness}/Acoustic/del"] | sig_idx[f"Auditory/Repeat/{wordness}/Phonemic/del"] | sig_idx[f"Auditory/Repeat/{wordness}/Lexical/del"],len(chs_all)),
            gp.set2arr(sig_idx[f"Resp/Repeat/{wordness}/Acoustic/resp"] | sig_idx[f"Resp/Repeat/{wordness}/Phonemic/resp"] | sig_idx[f"Resp/Repeat/{wordness}/Lexical/resp"],len(chs_all))),
            ([sig_idx[f"Auditory/Repeat/{wordness}/Acoustic/aud"],sig_idx[f"Auditory/Repeat/{wordness}/Phonemic/aud"],sig_idx[f"Auditory/Repeat/{wordness}/Lexical/aud"]],
             [sig_idx[f"Auditory/Repeat/{wordness}/Acoustic/del"],sig_idx[f"Auditory/Repeat/{wordness}/Phonemic/del"],sig_idx[f"Auditory/Repeat/{wordness}/Lexical/del"]],
             [sig_idx[f"Resp/Repeat/{wordness}/Acoustic/resp"],sig_idx[f"Resp/Repeat/{wordness}/Phonemic/resp"],sig_idx[f"Resp/Repeat/{wordness}/Lexical/resp"]])
    ):

        color_map = {
            100: Acoustic_col,
            10: Phonemic_col,
            1: Lexical_col,
            110: [1,1,1],
            101: [1,1,1],
            111: [1,1,1],
            11: [1,1,1]
        }

        chs_col_idx = [
            int(chs_ov[0] * gp.set2arr(spec_sig[0],len(chs_all))[i] + chs_ov[1] * gp.set2arr(spec_sig[1],len(chs_all))[i] + chs_ov[2] * gp.set2arr(spec_sig[2],len(chs_all))[i])
            for i in range(len(chs_all))]
        picks = [i for i in range(len(chs_all)) if base_sig[i] == 1]
        pick_labels = [str(chs_all[i]) for i in range(len(chs_all)) if base_sig[i] == 1]
        # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
        chs_cols = [color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(chs_all))]
        chs_cols_picked = [chs_cols[i] for i in picks]

        # TRY also to plot valid (white?) vs. invalid electrodes (dark grey)
        gp.plot_brain(subjs, pick_labels, chs_cols_picked, None,
                   os.path.join('plot', f'GLM electrode loc {TypeLabel}.jpg'))