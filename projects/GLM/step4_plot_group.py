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


#%% Set parameters
events = ["Auditory","Resp"]
stat = "zscore"
task_Tags = ["Repeat","Yes_No"]
glm_feas = ["Acoustic","Phonemic","Lexical"]
cluster_twin=0.011
mean_word_len=0.62
auditory_decay=0.4
delay_len=0.5
# motor_prep_win=[-0.5,-0.1]
# motor_resp_win=[0.25,0.75]
Waveplot_wth=10 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots
Acoustic_col = [255/255, 140/255, 0/255]    # (Dark Orange)
Phonemic_col = [138/255, 43/255, 226/255]   # (Violet)
Lexical_col = [50/255, 205/255, 50/255]     # (Lime Green)

#%% Load masks, sort, getting sort index, and plot the ranks
# sig_idx_arr=np.empty((len(events), len(task_Tags),len(glm_feas),4), dtype=object)
# sig_idx_lab=tuple(tuple(item) for item in (events,task_Tags,glm_feas,('all','aud','del','resp')))
# sig_idx=LabeledArray(sig_idx_arr,sig_idx_lab)
sig_idx=dict()
stass=dict()
for event, task_Tag in itertools.product(events,task_Tags):
    subjs, _, _, chs, times = glm.fifread(event, 'zscore', task_Tag)
    for glm_fea in glm_feas:
        if task_Tag == "Yes_No" and (event != "Resp" or glm_fea != "Lexical"):
            continue
        else:
            masks,stats,_=glm.load_stats(event,stat,task_Tag,'cluster_mask',glm_fea,subjs,chs,times)
            stass[f'{event}/{task_Tag}/{glm_fea}']=stats
            if event=='Auditory':
                # whole trial
                all_masks_sorted,_,all_masks_sig = gp.sort_chs_by_actonset(masks,cluster_twin,[-0.1,5])
                gp.plot_chs(all_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{glm_fea}_all.jpg'),f"N chs = {len(all_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{glm_fea}/all"] = all_masks_sig
                # auditory window
                aud_masks_sorted,_,aud_masks_sig = gp.sort_chs_by_actonset(masks,cluster_twin,[-0.1,mean_word_len+auditory_decay])
                gp.plot_chs(aud_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{glm_fea}_aud.jpg'),f"N chs = {len(aud_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{glm_fea}/aud"] = aud_masks_sig
                # delay window
                del_masks_sorted,_,del_masks_sig = gp.sort_chs_by_actonset(masks,cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
                gp.plot_chs(del_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{glm_fea}_del.jpg'),f"N chs = {len(del_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{glm_fea}/del"] = del_masks_sig

            elif event=="Resp":
                # response window
                resp_masks_sorted, _, resp_masks_sig = gp.sort_chs_by_actonset(masks, cluster_twin, [-0.1, 5])
                gp.plot_chs(resp_masks_sorted, os.path.join('plot', f'{event}_{task_Tag}_{glm_fea}_resp.jpg'),
                            f"N chs = {len(resp_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{glm_fea}/resp"] = resp_masks_sig

#%% plot significant electrodes
for md in ['all','aud','del']:
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    gp.plot_wave(stass['Auditory/Repeat/Acoustic'], sig_idx[f"Auditory/Repeat/Acoustic/{md}"], 'Acoustic',Acoustic_col)
    gp.plot_wave(stass['Auditory/Repeat/Phonemic'], sig_idx[f"Auditory/Repeat/Phonemic/{md}"], 'Phonemic',Phonemic_col)
    gp.plot_wave(stass['Auditory/Repeat/Lexical'], sig_idx[f"Auditory/Repeat/Lexical/{md}"], 'Lexical', Lexical_col)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.title(f'Lexical Repeat Delay {md}')
    plt.ylabel('GLM R^2 bsl corrected (-min)')
    plt.legend()
    plt.savefig(os.path.join('plot',f'wave auditory onset {md}.jpg'))

plt.figure(figsize=(Waveplot_wth * (150 / 350), Waveplot_hgt))
gp.plot_wave(stass['Resp/Repeat/Acoustic'], sig_idx[f"Resp/Repeat/Acoustic/resp"], 'Acoustic', Acoustic_col)
gp.plot_wave(stass['Resp/Repeat/Phonemic'], sig_idx[f"Resp/Repeat/Phonemic/resp"], 'Phonemic', Phonemic_col)
gp.plot_wave(stass['Resp/Repeat/Lexical'], sig_idx[f"Resp/Repeat/Lexical/resp"], 'Lexical', Lexical_col)
plt.axvline(x=0, linestyle='--', color='k')
plt.title('Lexical Repeat Delay resp')
plt.ylabel('GLM R^2 bsl corrected (-min)')
plt.legend()
plt.savefig(os.path.join('plot', 'wave motor onset.jpg'))

#%% Get confusion matrix of glm sig electrode sets
keys_of_interest = [
    "Auditory/Repeat/Acoustic/aud",
    "Auditory/Repeat/Phonemic/aud",
    "Auditory/Repeat/Lexical/aud",
    "Auditory/Repeat/Acoustic/del",
    "Auditory/Repeat/Phonemic/del",
    "Auditory/Repeat/Lexical/del",
    "Resp/Repeat/Acoustic/resp",
    "Resp/Repeat/Phonemic/resp",
    "Resp/Repeat/Lexical/resp"
]

filtered_sets = {key: sig_idx[key] for key in keys_of_interest}
short_names = {key: "/".join(key.split("/")[-2:]) for key in keys_of_interest}
conf_matrix = pd.DataFrame(index=short_names.values(), columns=short_names.values())

for key1 in keys_of_interest:
    for key2 in keys_of_interest:
        conf_matrix.loc[short_names[key1], short_names[key2]] = (len(filtered_sets[key1] & filtered_sets[key2]) / len(filtered_sets[key1])) * 100

conf_matrix = conf_matrix.astype(float)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".2f",cmap="rocket_r", xticklabels=short_names.values(),
                yticklabels=short_names.values(), annot_kws={"size": 12},vmin=0, vmax=100)

plt.title("Shared encoding electrodes across features and phase (%)")

plt.savefig(os.path.join('plot', f'GLM electrode sharing.jpg'))

#%% Plot brain
chs_all = np.concatenate(chs, axis=0)

for TypeLabel, chs_ov, base_sig, spec_sig in zip(
        ('Auditory', 'Delay', 'Response'),
        ([100, 10, 1], [100, 10, 1], [100, 10, 1]),
        (np.ones(len(chs_all),dtype=int),np.ones(len(chs_all),dtype=int),np.ones(len(chs_all),dtype=int)),
        ([sig_idx["Auditory/Repeat/Acoustic/aud"],sig_idx["Auditory/Repeat/Phonemic/aud"],sig_idx["Auditory/Repeat/Lexical/aud"]],
         [sig_idx["Auditory/Repeat/Acoustic/del"],sig_idx["Auditory/Repeat/Phonemic/del"],sig_idx["Auditory/Repeat/Lexical/del"]],
         [sig_idx["Resp/Repeat/Acoustic/resp"],sig_idx["Resp/Repeat/Phonemic/resp"],sig_idx["Resp/Repeat/Lexical/resp"]])
):

    color_map = {
        100: Acoustic_col,
        10: Phonemic_col,
        1: Lexical_col
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
    gp.plot_brain(subjs, pick_labels, chs_cols_picked, 0,
               os.path.join('plot', f'GLM electrode loc {TypeLabel}.jpg'))