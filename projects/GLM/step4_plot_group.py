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
mask_type='glm' #hg: used high-gamma permutation time-cluster masks; glm: use glm permutation time-cluster masks
with open('glm_config.json', 'r') as f:
    config = json.load(f)

# Extract parameters from config
Acoustic_col = config['Acoustic_col']
Phonemic_col = config['Phonemic_col']
Lexical_col = config['Lexical_col']

events = ["Auditory_inRep"]#,"Resp"]
stat = "zscore"
task_Tags = ["Repeat"]#,"Yes_No"]
wordnesses = ["ALL"]#, "Word", "Nonword"]
glm_feas = ["Acoustic","Phonemic","Lexical"]
cluster_twin=0.011
mean_word_len=0.62
auditory_decay=0.4
delay_len=0.5
# motor_prep_win=[-0.5,-0.1]
# motor_resp_win=[0.25,0.75]
Waveplot_wth=18 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots

# %% get hg masks
if mask_type == 'hg':

    # % Set parameters: HG
    groupsTag = "LexDelay"
    stat_type = 'mask'
    contrast = 'ave'  # average, not contrasting different conditions

    # For lexical delay task, whether run the data only with repeat tasks
    # Delayseleted=''
    Delayseleted = '_inRep'
    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "Box", "CoganLab")

    stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
    stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

    if groupsTag == "LexDelay":
        hgmask_aud, _ = gp.load_stats(stat_type, 'Auditory' + Delayseleted, contrast, stats_root_delay, stats_root_delay)
        hgmask_resp, _ = gp.load_stats(stat_type, 'Resp' + Delayseleted, contrast, stats_root_delay, stats_root_delay)

#%% Load masks, sort, getting sort index, and plot the ranks
# sig_idx_arr=np.empty((len(events), len(task_Tags),len(glm_feas),4), dtype=object)
# sig_idx_lab=tuple(tuple(item) for item in (events,task_Tags,glm_feas,('all','aud','del','resp')))
# sig_idx=LabeledArray(sig_idx_arr,sig_idx_lab)
sig_idx=dict()
stass=dict()
for event, task_Tag, wordness in itertools.product(events,task_Tags,wordnesses):
    subjs, _, _, chs, times = glm.fifread(event, 'zscore', task_Tag,wordness)
    for glm_fea in glm_feas:
        # if task_Tag == "Yes_No" and ((event=='Auditory' and glm_fea=='Acoustic') or glm_fea=='Acoustic'):# (event != "Resp" or glm_fea != "Lexical"):
        #     continue
        if wordness != "ALL" and glm_fea == "Lexical":
            continue
        else:
            masks,stats,_=glm.load_stats(event,stat,task_Tag,'cluster_mask',glm_fea,subjs,chs,times,wordness)
            if mask_type == 'glm':
                if event=='Auditory':
                    hgmask_aud=masks
                elif event=='Resp':
                    hgmask_resp=masks
            del masks

            clean_chs_idx = gp.get_notmuscle_electrodes(stats)
            stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}']=stats
            if event=='Auditory':
                # whole trial
                all_masks_sorted,_,_,all_masks_sig = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'],cluster_twin,[-0.1,5])
                all_masks_sig = all_masks_sig & clean_chs_idx
                gp.plot_chs(all_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{wordness}_{glm_fea}_all.jpg'),f"N chs = {len(all_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/all"] = all_masks_sig
                # auditory window
                aud_masks_sorted,_,_,aud_masks_sig = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,[-0.1,mean_word_len+auditory_decay])
                aud_masks_sig = aud_masks_sig & clean_chs_idx
                gp.plot_chs(aud_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{wordness}_{glm_fea}_aud.jpg'),f"N chs = {len(aud_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/aud"] = aud_masks_sig
                # delay window
                del_masks_sorted,_,_,del_masks_sig = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
                del_masks_sig = del_masks_sig & clean_chs_idx
                gp.plot_chs(del_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{wordness}_{glm_fea}_del.jpg'),f"N chs = {len(del_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/del"] = del_masks_sig

            elif event=="Resp":
                # response window
                resp_masks_sorted, _, _, resp_masks_sig = gp.sort_chs_by_actonset(hgmask_resp, stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin, [-0.1, 5])
                resp_masks_sig = resp_masks_sig & clean_chs_idx
                gp.plot_chs(resp_masks_sorted, os.path.join('plot', f'{event}_{task_Tag}_{wordness}_{glm_fea}_resp.jpg'),
                            f"N chs = {len(resp_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/resp"] = resp_masks_sig

#%% plot significant electrodes
for wordness in wordnesses[:2]:
    for md,md_Tag in zip(['all','aud','del'],['whole trial','auditory window','delay window']):
        if md=='all':
            wid_scale=1
        elif md=='aud':
            xlim_l=-0.1
            xlim_r=1.5#mean_word_len + auditory_decay
            wid_scale=(xlim_r-xlim_l)*100/350
        elif md=='del':
            xlim_l=0.5
            xlim_r=1.5
            wid_scale = (xlim_r - xlim_l)*100/350
        plt.figure(figsize=(Waveplot_wth*wid_scale, Waveplot_hgt))
        if wordness == 'ALL':
            gp.plot_wave(stass[f'Auditory/Repeat/{wordness}/Acoustic'], sig_idx[f"Auditory/Repeat/{wordness}/Acoustic/{md}"],
                         'Acoustic rep', Acoustic_col, '-',True)
            gp.plot_wave(stass[f'Auditory/Repeat/{wordness}/Phonemic'], sig_idx[f"Auditory/Repeat/{wordness}/Phonemic/{md}"],
                         'Phonemic rep', Phonemic_col, '-',True)
            gp.plot_wave(stass[f'Auditory/Repeat/{wordness}/Lexical'], sig_idx[f"Auditory/Repeat/{wordness}/Lexical/{md}"],
                         'Lexical status rep', Lexical_col, '-',True)
            if 'Yes_No' in task_Tags:
                gp.plot_wave(stass[f'Auditory/Yes_No/{wordness}/Acoustic'], sig_idx[f"Auditory/Yes_No/{wordness}/Acoustic/{md}"],
                         'Acoustic YN', Acoustic_col, '--',True)
                gp.plot_wave(stass[f'Auditory/Yes_No/{wordness}/Phonemic'], sig_idx[f"Auditory/Yes_No/{wordness}/Phonemic/{md}"],
                         'Phonemic YN', Phonemic_col, '--',True)
                gp.plot_wave(stass[f'Auditory/Yes_No/{wordness}/Lexical'], sig_idx[f"Auditory/Yes_No/{wordness}/Lexical/{md}"],
                         'Lexical status YN', Lexical_col, '--',True)
        elif wordness == 'Word':
            gp.plot_wave(stass[f'Auditory/Repeat/Word/Acoustic'], sig_idx[f"Auditory/Repeat/Word/Acoustic/{md}"],
                         'Acoustic_Word', Acoustic_col, '-',True)
            gp.plot_wave(stass[f'Auditory/Repeat/Word/Phonemic'], sig_idx[f"Auditory/Repeat/Word/Phonemic/{md}"],
                         'Phonemic_Word', Phonemic_col, '-',True)
            gp.plot_wave(stass[f'Auditory/Repeat/Nonword/Acoustic'], sig_idx[f"Auditory/Repeat/Nonword/Acoustic/{md}"],
                         'Acoustic_Nonword', Acoustic_col, '--',True)
            gp.plot_wave(stass[f'Auditory/Repeat/Nonword/Phonemic'], sig_idx[f"Auditory/Repeat/Nonword/Phonemic/{md}"],
                         'Phonemic_Nonword', Phonemic_col, '--',True)
        plt.axvline(x=0, linestyle='--', color='k')
        plt.axhline(y=0, linestyle='--', color='k')
        if wordness == 'ALL':
            wordness_Tag = 'Word & Nonword'
        else:
            wordness_Tag = 'Word or Nonword'
        plt.title(f'GLM:  {wordness_Tag} in {md_Tag}')
        plt.ylabel(r'GLM R^2 bsl corrected (normalized)')
        plt.xlabel('Time from auditory onset (s)')
        plt.gca().spines[['top', 'right']].set_visible(False)
        if md == 'aud' or md=='del':
            plt.xlim(xlim_l, xlim_r)
        plt.tight_layout()
        if md=='all' or md == 'aud':
            plt.legend()
        plt.savefig(os.path.join('plot',f'wave auditory onset {wordness} {md}.tif'),dpi=300)
        plt.close()

    xlim_l = -0.2
    xlim_r = mean_word_len
    wid_scale = (xlim_r - xlim_l)*100/350
    plt.figure(figsize=(Waveplot_wth*wid_scale, Waveplot_hgt))
    if wordness == 'ALL':
        gp.plot_wave(stass[f'Resp/Repeat/{wordness}/Acoustic'], sig_idx[f"Resp/Repeat/{wordness}/Acoustic/resp"],
                     'Acoustic', Acoustic_col, '-',False)
        gp.plot_wave(stass[f'Resp/Repeat/{wordness}/Phonemic'], sig_idx[f"Resp/Repeat/{wordness}/Phonemic/resp"],
                     'Phonemic', Phonemic_col, '-',False)
        gp.plot_wave(stass[f'Resp/Repeat/{wordness}/Lexical'], sig_idx[f"Resp/Repeat/{wordness}/Lexical/resp"], 'Lexical status', Lexical_col,'-',True)
        gp.plot_wave(stass[f'Resp/Yes_No/{wordness}/Phonemic'], sig_idx[f"Resp/Yes_No/{wordness}/Phonemic/resp"], 'Phonemic in Decision', Phonemic_col,'--',False)
        gp.plot_wave(stass[f'Resp/Yes_No/{wordness}/Lexical'], sig_idx[f"Resp/Yes_No/{wordness}/Lexical/resp"], 'Lexical status in Decision', Lexical_col,'--',False)
    elif wordness == 'Word':
        gp.plot_wave(stass[f'Resp/Repeat/Word/Acoustic'], sig_idx[f"Resp/Repeat/Word/Acoustic/resp"],
                     'Acoustic_Word', Acoustic_col, '-',True)
        gp.plot_wave(stass[f'Resp/Repeat/Word/Phonemic'], sig_idx[f"Resp/Repeat/Word/Phonemic/resp"],
                     'Phonemic_Word', Phonemic_col, '-',True)
        gp.plot_wave(stass[f'Resp/Repeat/Nonword/Acoustic'], sig_idx[f"Resp/Repeat/Nonword/Acoustic/resp"],
                     'Acoustic_Nonword', Acoustic_col, '--',True)
        gp.plot_wave(stass[f'Resp/Repeat/Nonword/Phonemic'], sig_idx[f"Resp/Repeat/Nonword/Phonemic/resp"],
                     'Phonemic_Nonword', Phonemic_col, '--',True)

    if wordness == 'ALL':
        wordness_Tag = 'Word & Nonword'
    else:
        wordness_Tag = wordness

    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.title(f'GLM:  {wordness_Tag} in resp window')
    plt.ylabel('GLM R^2 bsl corrected (-min)')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.legend()
    plt.xlim(xlim_l, xlim_r)
    plt.xlabel('Time from motor onset (s)')
    plt.tight_layout()
    plt.savefig(os.path.join('plot', f'wave motor onset in {wordness}.tif'),dpi=300)
    plt.close()

    with open(os.path.join('data', 'sig_idx.npy'), "wb") as f:
        pickle.dump(sig_idx, f)

#%% Get confusion matrix of glm sig electrode sets
for wordness in wordnesses:

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

    filtered_sets = {key: sig_idx[key] for key in keys_of_interest}
    short_names = {key: "\n".join(key.split("/")[-2:]).replace('del', 'Delay').replace('aud', 'Auditory').replace('resp', 'Response') for key in keys_of_interest}
    conf_matrix = pd.DataFrame(index=short_names.values(), columns=short_names.values())

    for key1 in keys_of_interest:
        for key2 in keys_of_interest:
            conf_matrix.loc[short_names[key1], short_names[key2]] = (len(filtered_sets[key1] & filtered_sets[key2]) / len(filtered_sets[key1])) * 100

    conf_matrix = conf_matrix.astype(float)

    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt=".2f",cmap="rocket_r", xticklabels=short_names.values(),
                    yticklabels=short_names.values(), annot_kws={"size": 14},vmin=0, vmax=100,cbar=False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title("Shared encoding electrodes across features and phase (%)",fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join('plot', f'GLM electrode sharing in {wordness}.tif'),dpi=300)
    plt.close()

