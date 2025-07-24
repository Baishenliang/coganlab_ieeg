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
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
import json
import pickle
from matplotlib_venn import venn3,venn2
from matplotlib import pyplot as plt


#%% Set parameters
with open('glm_config.json', 'r') as f:
    config = json.load(f)
with open(os.path.join('data', f'Lex_twin_idxes_hg.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)

# Extract parameters from config
Acoustic_col = config['Acoustic_col']
Phonemic_col = config['Phonemic_col']
Lexical_col = config['Lexical_col']

stat = "zscore"
glm_feas = ["Del_selective"]
wordnesses = ["ALL"]#["ALL", "Word", "Nonword"]
cluster_twin=0.011
mean_word_len=0.62
auditory_decay=0.4
delay_len=0.5
# motor_prep_win=[-0.5,-0.1]
# motor_resp_win=[0.25,0.75]
Waveplot_wth=10 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots
mask_corr_type='cluster_mask' #cluster_mask: mask from glm time perm cluster; # org_mask: mask from permutation (original R2 ranked in null distribution) # fdr_mask: after fdr correction.
event_suffix='inRep'
if event_suffix=='inYN':
    task_Tag='Yes_No'
elif event_suffix=='inRep':
    task_Tag='Repeat'

subjs, _, _, chs, times = glm.fifread(f'Cue_{event_suffix}', stat, task_Tag, 'ALL', Comp_task=glm_feas[0],
                                      preonset_bsl_correct=False)
with open(os.path.join('data', f'sig_idx_{glm_feas[0]}.npy'), "rb") as f:
    Del_selective_idxes = pickle.load(f)
with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data','Lex_twin_idxes_hg.npy'), "rb") as f:
    Lex_twin_idxes_hg = pickle.load(f)



#%% Plot brain
hickok_roi_all = pd.DataFrame()
# Just get the electrodes
masks, _, _ = glm.load_stats(f'Cue_{event_suffix}', 'mask', task_Tag, 'cluster_mask', glm_feas[0], subjs, chs,
                             times, 'ALL')
chs_all = masks.labels[0]
chs_coor = gp.get_coor(chs_all, 'group')
ch_labels_roi, _ = gp.chs2atlas(subjs, chs_all)
hickok_roi_labels, _ = gp.hickok_roi_sphere(chs_coor)

for sig,tag,hist_ylim in zip(
        (Del_selective_idxes & Lex_twin_idxes_hg['LexDelay_all_sig_idx'],
        Del_selective_idxes & Lex_twin_idxes_hg['LexDelay_all_sig_idx'] & Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx']),
        ('DelSelinALL','DelSelinMtrDel'),
        ([0,30],[0,10])
):

    chs_ov=[100,10,1]
    col = Acoustic_col
    chs_sel=chs_all[list(sig)].tolist()
    # cols = [gp.adjust_saturation(np.array(col),val) for val in avg]
    cols = [col for i in range(0,len(sig))]
    gp.plot_brain(subjs, chs_sel, cols, None,dotsize=0.3,
               fig_save_dir_f=os.path.join('plot', f'GLM electrode loc {glm_feas[0]}.jpg'))
    gp.atlas2_hist(ch_labels_roi,chs_sel,col,os.path.join('plot',f'Atlas histogram {glm_feas[0]} {tag}.tif'),ylim=hist_ylim)
    gp.plot_sig_roi_counts(hickok_roi_labels, col, sig, os.path.join('plot',f'Hickok ROI histogram {glm_feas[0]} {tag}.tif'))

#%% Overlap between GLM Selective electrodes and Delay electrodes
pre_stim_task_sig=Del_selective_idxes & Lex_twin_idxes_hg['LexDelay_all_sig_idx']

## Within all significant HG electrodes in Delay repeat tasks, Delay electrodes are more likely to be pre-stim delay selective than the others
print(len(Lex_twin_idxes_hg['LexDelay_Delay_sig_idx']))
print(len(Lex_twin_idxes_hg['LexDelay_all_sig_idx']))
print(len(pre_stim_task_sig & Lex_twin_idxes_hg['LexDelay_Delay_sig_idx']))
print(len(pre_stim_task_sig))

## Most Delay-selective Motor delay electrodes are silent in NoDelay repeat
print(len(pre_stim_task_sig & (Lex_twin_idxes_hg['LexDelay_Auditory_in_Delay_sig_idx'] | LexDelay_twin_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx'])))
print(len(pre_stim_task_sig & (Lex_twin_idxes_hg['LexDelay_Auditory_in_Delay_sig_idx'] | LexDelay_twin_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx'])
          & (Lex_twin_idxes_hg['LexNoDelay_Aud_NoMotor_sig_idx'] | LexDelay_twin_idxes['LexNoDelay_Sensorimotor_sig_idx'])))
print(len(pre_stim_task_sig & Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx']))
print(len(pre_stim_task_sig & Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx'] & Lex_twin_idxes_hg['LexNoDelay_Motor_sig_idx']))

## Delay-selective Motor delay electrodes explain additional Motor delay activity in Delay repeat tasks
# Venn plot: Delay electrodes in NoDelay
plt.figure(figsize=(6, 6))
plt.title(f'Motor delay electrodes only (N={len(Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx'])}, other not shown)')
venn3([pre_stim_task_sig & Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx'],
       Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx'],
       Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx'] - Lex_twin_idxes_hg['LexNoDelay_Motor_sig_idx']],
      ('Pre-stim Delay-selective',
       'Motor delay electrodes',
       'Not Motor electrodes in NoDelay'))
plt.tight_layout()
plt.savefig(os.path.join('plot',  f'Motor delay electrodes only.tif'),
            dpi=300)
plt.close()
# Remove the Delay-selective motor delay electrodes
Del_selective_Motor_in_Delay=pre_stim_task_sig & Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx']
print(len(Lex_twin_idxes_hg['LexDelay_Auditory_in_Delay_sig_idx'] | LexDelay_twin_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx']
          - Del_selective_Motor_in_Delay))
print(len(Lex_twin_idxes_hg['LexDelay_Delay_sig_idx'] & (Lex_twin_idxes_hg['LexNoDelay_Aud_NoMotor_sig_idx'] | LexDelay_twin_idxes['LexNoDelay_Sensorimotor_sig_idx'])
          - Del_selective_Motor_in_Delay))
print(len(Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx']
          - Del_selective_Motor_in_Delay))
print(len(Lex_twin_idxes_hg['LexDelay_Delay_sig_idx'] & Lex_twin_idxes_hg['LexNoDelay_Motor_sig_idx']
          - Del_selective_Motor_in_Delay))

# Venn: Motor delay Aud NoDelay, Delay-selective Motor delay
# Venn plot: Delay electrodes in NoDelay
plt.figure(figsize=(6, 6))
plt.title(f'Motor delay electrodes only (N={len(Lex_twin_idxes_hg['LexDelay_Motor_in_Delay_sig_idx'])}, other not shown)')
venn2([LexDelay_twin_idxes['MtrDel_AudNoDel_sig'],
       Del_selective_Motor_in_Delay],
      ('Auditory in NoDelay',
       'Pre-onset Delay selective'))
plt.tight_layout()
plt.savefig(os.path.join('plot',  f'Motor delay Aud NoDelay Delay-selective Motor delay.tif'),
            dpi=300)
plt.close()