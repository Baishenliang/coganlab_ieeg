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

subjs, _, _, chs, times = glm.fifread("Auditory_inRep", 'zscore', 'Repeat', wordnesses[0])
with open(os.path.join('data', 'sig_idx.npy'), "rb") as f:
    LexDelay_glm_idxes = pickle.load(f)

#%% Plot brain
for wordness in wordnesses:
    # Just get the electrodes
    masks, _, _ = glm.load_stats('Auditory_inRep', 'mask', 'Repeat', 'cluster_mask', 'Acoustic', subjs, chs, times, wordness)
    chs_all = masks.labels[0]
    chs_coor = gp.get_coor(chs_all, 'group')
    ch_labels_roi, _ = gp.chs2atlas(subjs,chs_all)
    hickok_roi_labels = gp.hickok_roi_sphere(chs_coor)
    if wordness == 'ALL':
        keys_of_interest = [
            "Auditory_inRep/Repeat/ALL/Acoustic/aud",
            "Auditory_inRep/Repeat/ALL/Phonemic/aud",
            "Auditory_inRep/Repeat/ALL/Lexical/aud",
            "Auditory_inRep/Repeat/ALL/Acoustic/del",
            "Auditory_inRep/Repeat/ALL/Phonemic/del",
            "Auditory_inRep/Repeat/ALL/Lexical/del",
            "Resp_inRep/Repeat/ALL/Acoustic/resp",
            "Resp_inRep/Repeat/ALL/Phonemic/resp",
            "Resp_inRep/Repeat/ALL/Lexical/resp"
        ]
    else:
        keys_of_interest = [
            f"Auditory_inRep/Repeat/{wordness}/Acoustic/aud",
            f"Auditory_inRep/Repeat/{wordness}/Phonemic/aud",
            f"Auditory_inRep/Repeat/{wordness}/Acoustic/del",
            f"Auditory_inRep/Repeat/{wordness}/Phonemic/del",
            f"Resp_inRep/Repeat/{wordness}/Acoustic/resp",
            f"Resp_inRep/Repeat/{wordness}/Phonemic/resp"
         ]

    for TypeLabel in keys_of_interest:
        chs_ov=[100,10,1]
        sig=LexDelay_glm_idxes[TypeLabel]
        if 'Acoustic' in TypeLabel:
            col = Acoustic_col
        elif 'Phonemic' in TypeLabel:
            col = Phonemic_col
        elif 'Lexical' in TypeLabel:
            col = Lexical_col
        chs_sel=chs_all[list(sig)].tolist()
        cols=[col]*len(chs_sel)
        gp.plot_brain(subjs, chs_sel, cols, None,
                   os.path.join('plot', f'GLM electrode loc {TypeLabel}.jpg'))
        gp.atlas2_hist(ch_labels_roi,chs_sel,col,os.path.join('plot',f'Atlas histogram {TypeLabel.replace('/', ' ')}.tif'))
        gp.plot_sig_roi_counts(hickok_roi_labels, col, sig, os.path.join('plot',f'Hickok ROI histogram {TypeLabel.replace('/', ' ')}.tif'))

#%% ovelapped plot
overlap_Plot=False
for wordness in wordnesses:
    if not overlap_Plot:
        continue

    # Just get the electrodes
    masks,_,_=glm.load_stats('Auditory','mask','Repeat','cluster_mask','Acoustic',subjs,chs,times,wordness)
    chs_all = masks.labels[0]

    for TypeLabel, chs_ov, base_sig, spec_sig in zip(
            ('Auditory', 'Delay', 'Response'),
            ([100, 10, 1], [100, 10, 1], [100, 10, 1]),
            (gp.set2arr(LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Acoustic/aud"] | LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Phonemic/aud"] | LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Lexical/aud"],len(chs_all)),
            gp.set2arr(LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Acoustic/del"] | LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Phonemic/del"] | LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Lexical/del"],len(chs_all)),
            gp.set2arr(LexDelay_glm_idxes[f"Resp_inRep/Repeat/{wordness}/Acoustic/resp"] | LexDelay_glm_idxes[f"Resp_inRep/Repeat/{wordness}/Phonemic/resp"] | LexDelay_glm_idxes[f"Resp_inRep/Repeat/{wordness}/Lexical/resp"],len(chs_all))),
            ([LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Acoustic/aud"],LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Phonemic/aud"],LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Lexical/aud"]],
             [LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Acoustic/del"],LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Phonemic/del"],LexDelay_glm_idxes[f"Auditory_inRep/Repeat/{wordness}/Lexical/del"]],
             [LexDelay_glm_idxes[f"Resp_inRep/Repeat/{wordness}/Acoustic/resp"],LexDelay_glm_idxes[f"Resp_inRep/Repeat/{wordness}/Phonemic/resp"],LexDelay_glm_idxes[f"Resp_inRep/Repeat/{wordness}/Lexical/resp"]])
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