#%% Import everything
import os
from matplotlib_venn import venn3
from array_api_compat.dask.array import astype
from sqlalchemy import false

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

subjs, _, _, chs, times = glm.fifread("Auditory", 'zscore', 'Repeat', wordnesses[0])

#%% Get confusion between time-windowed selected electrodes and glm electrods

with open(os.path.join('data', 'sig_idx.npy'), "rb") as f:
    LexDelay_glm_idxes = pickle.load(f)
with open(os.path.join('data', 'LexDelay_twin_idxes.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)

win_aud=LexDelay_twin_idxes['LexDelay_Aud_NoMotor_sig_idx']
win_sm=LexDelay_twin_idxes['LexDelay_Sensorimotor_sig_idx']
win_mtr=LexDelay_twin_idxes['LexDelay_Motor_sig_idx']
win_delo=LexDelay_twin_idxes['LexDelay_DelayOnly_sig_idx']
win_mtrprep=LexDelay_twin_idxes['LexDelay_Motorprep_Only_sig_idx']

for ph,ph_Tag in zip(['aud','del','resp'],['Auditory',"Delay","Response"]):
    if ph !='resp':
        glm_aco=LexDelay_glm_idxes[f'Auditory/Repeat/ALL/Acoustic/{ph}']
        glm_pho=LexDelay_glm_idxes[f'Auditory/Repeat/ALL/Phonemic/{ph}']
        glm_lex=LexDelay_glm_idxes[f'Auditory/Repeat/ALL/Lexical/{ph}']
    else:
        glm_aco=LexDelay_glm_idxes[f'Resp/Repeat/ALL/Acoustic/{ph}']
        glm_pho=LexDelay_glm_idxes[f'Resp/Repeat/ALL/Phonemic/{ph}']
        glm_lex=LexDelay_glm_idxes[f'Resp/Repeat/ALL/Lexical/{ph}']

    all_win_electrodes = win_aud | win_sm | win_mtr | win_mtrprep | win_delo

    # Confusion matrix in percentage
    data = {
        "Auditory": [len(glm_aco & win_aud)/len(glm_aco)*100, len(glm_pho & win_aud)/len(glm_pho)*100, len(glm_lex & win_aud)/len(glm_lex)*100],
        "Sensorimotor": [len(glm_aco & win_sm)/len(glm_aco)*100, len(glm_pho & win_sm)/len(glm_pho)*100, len(glm_lex & win_sm)/len(glm_lex)*100],
        "Motor": [len(glm_aco & win_mtr)/len(glm_aco)*100, len(glm_pho & win_mtr)/len(glm_pho)*100, len(glm_lex & win_mtr)/len(glm_lex)*100],
        "Motor_prep":[len(glm_aco & win_mtrprep)/len(glm_aco)*100, len(glm_pho & win_mtrprep)/len(glm_pho)*100, len(glm_lex & win_mtrprep)/len(glm_lex)*100],
        "Delay_only":[len(glm_aco & win_delo) / len(glm_aco) * 100,len(glm_pho & win_delo) / len(glm_pho) * 100,len(glm_lex & win_delo) / len(glm_lex) * 100],
        "Others (not sig to bsl)":[len(glm_aco - all_win_electrodes)/len(glm_aco)*100,len(glm_pho - all_win_electrodes)/len(glm_pho)*100,len(glm_lex - all_win_electrodes)/len(glm_lex)*100]
    }

    df_cm = pd.DataFrame(data, index=["Acoustic", "Phonemic", "Lexical"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="rocket_r", annot_kws={"size": 14}, vmin=0, vmax=100, cbar=False)
    plt.title(f"EGL electrodes for {ph_Tag}")
    plt.ylabel("Significant elec in GLM")
    plt.xlabel("SIgnificant elec by time window")
    plt.tight_layout()
    plt.savefig(os.path.join('plot', f'Confusion Matrix for {ph.upper()}.tif'), dpi=300)
    plt.close()

    # Venn diagrams
    for ele_set,set_Tag in zip([win_aud, win_sm, win_mtr],['Auditory','Sensory-motor','Motor']):
        aco = ele_set & glm_aco
        pho = ele_set & glm_pho
        lex = ele_set & glm_lex
        plt.figure(figsize=(6, 6))
        venn3([aco, pho, lex], ('Acoustic', 'Phonemic', 'Lexical'))
        # Show the plot
        plt.title(f"{set_Tag} electrodes in {ph_Tag} phase (GLM elec.: {np.round(100*len(aco | pho | lex)/len(ele_set),3)}%)")
        plt.tight_layout()
        plt.savefig(os.path.join('plot', f"{set_Tag}_{ph_Tag}_venn.tif"), dpi=300)
        plt.close()


#%% Make Atlas histograms
from ieeg.viz.mri import subject_to_info,gen_labels
subjs_s = ['D' + subj[1:].lstrip('0') for subj in subjs]
ch_labels = dict()
for subj in subjs_s:
    info_i = subject_to_info(subj)
    ch_labels_k = gen_labels(info_i, subj, atlas='.BN_atlas')
    for key, value in ch_labels_k.items():
        ch_labels[f'{subj}-{key}'] = value

# Extract relevant columns and create a mapping dictionary
# Load the CSV file
df = pd.read_csv('atlas.csv')
# Create the dictionary
mapping_dict = {}
for index, row in df.iterrows():
    key = str(row['Anatomical and modified Cyto-architectonic descriptions']).split(',')[0]
    value = str(row['Left and Right Hemisphere']).split('_')[0]
    mapping_dict[key] = value
mapping_dict['TE1.0/TE1.2']='STG'
ch_labels_roi=dict()
for key,value in ch_labels.items():
    try:
        ch_labels_roi[key] = mapping_dict[value.split("_")[0]]
    except KeyError as e:
        ch_labels_roi[key] = 'unknown'

#%% Plot brain
for wordness in wordnesses:
    # Just get the electrodes
    masks, _, _ = glm.load_stats('Auditory', 'mask', 'Repeat', 'cluster_mask', 'Acoustic', subjs, chs, times, wordness)
    chs_all = masks.labels[0]
    if wordness == 'ALL':
        keys_of_interest = [
            "Auditory/Repeat/ALL/Acoustic/aud",
            "Auditory/Repeat/ALL/Phonemic/aud",
            "Auditory/Repeat/ALL/Lexical/aud",
            "Auditory/Repeat/ALL/Acoustic/del",
            "Auditory/Repeat/ALL/Phonemic/del",
            "Auditory/Repeat/ALL/Lexical/del",
            "Resp/Repeat/ALL/Acoustic/resp",
            "Resp/Repeat/ALL/Phonemic/resp",
            "Resp/Repeat/ALL/Lexical/resp"
        ]
    else:
        keys_of_interest = [
            f"Auditory/Repeat/{wordness}/Acoustic/aud",
            f"Auditory/Repeat/{wordness}/Phonemic/aud",
            f"Auditory/Repeat/{wordness}/Acoustic/del",
            f"Auditory/Repeat/{wordness}/Phonemic/del",
            f"Resp/Repeat/{wordness}/Acoustic/resp",
            f"Resp/Repeat/{wordness}/Phonemic/resp"
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
            (gp.set2arr(LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Acoustic/aud"] | LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Phonemic/aud"] | LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Lexical/aud"],len(chs_all)),
            gp.set2arr(LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Acoustic/del"] | LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Phonemic/del"] | LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Lexical/del"],len(chs_all)),
            gp.set2arr(LexDelay_glm_idxes[f"Resp/Repeat/{wordness}/Acoustic/resp"] | LexDelay_glm_idxes[f"Resp/Repeat/{wordness}/Phonemic/resp"] | LexDelay_glm_idxes[f"Resp/Repeat/{wordness}/Lexical/resp"],len(chs_all))),
            ([LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Acoustic/aud"],LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Phonemic/aud"],LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Lexical/aud"]],
             [LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Acoustic/del"],LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Phonemic/del"],LexDelay_glm_idxes[f"Auditory/Repeat/{wordness}/Lexical/del"]],
             [LexDelay_glm_idxes[f"Resp/Repeat/{wordness}/Acoustic/resp"],LexDelay_glm_idxes[f"Resp/Repeat/{wordness}/Phonemic/resp"],LexDelay_glm_idxes[f"Resp/Repeat/{wordness}/Lexical/resp"]])
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