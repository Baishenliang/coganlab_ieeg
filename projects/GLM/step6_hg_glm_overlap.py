#%% Import everything
import os
import pickle
from scipy import stats as st
import numpy as np
import matplotlib.pyplot as plt
import pyreadstat

# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

import sys
import pandas as pd
import glm_utils as glm
import itertools
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
import seaborn as sns
import json
import pickle

#%% Set parameters: HG
groupsTag="LexDelay"

stat_type='mask'
contrast='ave' # average, not contrasting different conditions

# For lexical delay task, whether run the data only with repeat tasks
#Delayseleted=''
Delayseleted = '_inRep'

# Parameters from the lexical delay task
mean_word_len=0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0.4 # a short period of time that we may assume auditory decay takes
delay_len=0.5 # from task script
motor_prep_win=[-0.5,-0.1] # get windows for motor preparation (0.1s to avoid high gamma filter leakage)
motor_resp_win=[-0.1,0.75] # get windows for motor response (0.75s to avoid too much auditory feedback)
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

#%% Set parameters: glm
with open('glm_config.json', 'r') as f:
    config = json.load(f)

# Extract parameters from config
Acoustic_col = config['Acoustic_col']
Phonemic_col = config['Phonemic_col']
Lexical_col = config['Lexical_col']

events = ["Auditory","Resp"]
stat = "zscore"
task_Tags = ["Repeat"]#,"Yes_No"]
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

#%% get hg masks
if groupsTag=="LexDelay":

    hgmask_aud,_=gp.load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_delay,stats_root_delay)
    hgmask_resp, _ = gp.load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_delay, stats_root_delay)

#%% hg glm r^2 plots
subjs, _, _, chs, times = glm.fifread("Auditory", 'zscore', 'Repeat', wordnesses[0])
with open(os.path.join('data', 'LexDelay_twin_idxes.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)

twin_sets=[LexDelay_twin_idxes['LexDelay_Aud_NoMotor_sig_idx'],
           LexDelay_twin_idxes['LexDelay_Sensorimotor_sig_idx'],
           LexDelay_twin_idxes['LexDelay_Motor_sig_idx']]
twin_labels=[np.repeat('Auditory',len(LexDelay_twin_idxes['LexDelay_Aud_NoMotor_sig_idx'])).tolist(),
             np.repeat('Sensory-motor',len(LexDelay_twin_idxes['LexDelay_Sensorimotor_sig_idx'])).tolist(),
             np.repeat('Motor',len(LexDelay_twin_idxes['LexDelay_Motor_sig_idx'])).tolist()]
twin_labels=np.concatenate(twin_labels).tolist()

glm_avgs=dict()
stass=dict()
for event, task_Tag, wordness in itertools.product(events,task_Tags,wordnesses):
    subjs, _, _, chs, times = glm.fifread(event, 'zscore', task_Tag,wordness)
    for glm_fea in glm_feas:
        # if task_Tag == "Yes_No" and ((event=='Auditory' and glm_fea=='Acoustic') or glm_fea=='Acoustic'):# (event != "Resp" or glm_fea != "Lexical"):
        #     continue
        if wordness != "ALL" and glm_fea == "Lexical":
            continue
        else:
            _,stats,_=glm.load_stats(event,stat,task_Tag,'cluster_mask',glm_fea,subjs,chs,times,wordness)
            stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}']=stats
            if event=='Auditory':
                # auditory window
                _,glm_masked,_,_ = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,[-0.1,mean_word_len+auditory_decay])
                _,glm_avg,_=gp.time_avg_select(glm_masked, twin_sets)
                glm_avgs[f'{task_Tag}/{wordness}/{glm_fea}/aud']=glm_avg
                # delay window
                _,glm_masked,_,_ = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
                _,glm_avg,_=gp.time_avg_select(glm_masked, twin_sets)
                glm_avgs[f'{task_Tag}/{wordness}/{glm_fea}/del']=glm_avg
            elif event=="Resp":
                # response window
                _,glm_masked,_,_ = gp.sort_chs_by_actonset(hgmask_resp, stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin, [-0.1, 5])
                _,glm_avg,_=gp.time_avg_select(glm_masked, twin_sets)
                glm_avgs[f'{task_Tag}/{wordness}/{glm_fea}/resp']=glm_avg
#%% Average and select and do 3d plots
phs=['aud','del','resp']
Sensorimotor_col = [1, 0, 0]  # Sensorimotor (Red)
Auditory_col = [0, 1, 0]  # Auditory (Green)
Motor_col = [0, 0, 1]  # Motor (Blue)
group_colors = {
    'Auditory': 'g',
    'Motor': 'b',
    'Sensory-motor': 'r'
}
from statsmodels.multivariate.manova import MANOVA
def glm_normalize(data,meth):
    clean_data = data[~np.isnan(data)]
    z_full = np.full_like(data, np.nan)
    if meth=='zscore':
        z_full[~np.isnan(data)] = stats.zscore(clean_data)
    elif meth=='one':
        z_full[~np.isnan(data)] = (clean_data-np.nanmean(clean_data))/(np.max(clean_data)-np.min(clean_data))
    elif meth=='mean':
        z_full[~np.isnan(data)] = np.nanmean(clean_data)
    return z_full
def rm_out(data, threshold=3):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    z_scores = (data - mean) / std
    data[np.abs(z_scores) > threshold] = np.nan
    return data

for task_Tag, wordness,ph in itertools.product(task_Tags,wordnesses,phs):
        w=dict()
        w['group']=twin_labels
        for glm_fea in glm_feas:
            w[glm_fea]=rm_out(glm_avgs[f'{task_Tag}/{wordness}/{glm_fea}/{ph}'])
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        w=pd.DataFrame(w)
        for group, color in group_colors.items():
            subset = w[w['group'] == group]
            ax.scatter(glm_normalize(subset[glm_feas[0]],'mean'),
                       glm_normalize(subset[glm_feas[1]],'mean'),
                       glm_normalize(subset[glm_feas[2]],'mean'),
                c=color,label=group,alpha=1,s=5  # size of the point
            )
        ax.set_xlabel(glm_feas[0])
        ax.set_ylabel(glm_feas[1])
        ax.set_zlabel(glm_feas[2])
        ax.set_xlim([0,0.2])
        ax.set_ylim([0,0.2])
        ax.set_zlim([0,0.2])
        ax.set_title(f'{task_Tag}/{wordness}/{ph}')
        ax.legend()
        plt.tight_layout()
        # ax.view_init(elev=90, azim=-90)
        plt.savefig(os.path.join('plot',f'glm hg plot {task_Tag} {wordness} {ph}.tif'),dpi=300)
        plt.close()
        pyreadstat.write_sav(w, os.path.join('data',f'glm hg {task_Tag} {wordness} {ph}.sav'))

        A=w[w['group']=='Auditory']['Lexical']
        SM=w[w['group']=='Sensory-motor']['Lexical']
        M=w[w['group']=='Motor']['Lexical']

        print(f'A-SM:{st.ttest_ind(A,SM,nan_policy='omit')}')
        print(f'M-SM:{st.ttest_ind(M,SM,nan_policy='omit')}')
        print(f'A-M:{st.ttest_ind(A,M,nan_policy='omit')}')

#%% Get confusion between time-windowed selected electrodes and glm electrods
if 1==0:
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

