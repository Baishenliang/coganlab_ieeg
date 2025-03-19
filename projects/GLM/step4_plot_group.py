#%% Import everything
import os
# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

import sys
import numpy as np
import glm_utils as glm
import itertools
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
import matplotlib.pyplot as plt
from ieeg.arrays.label import LabeledArray



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

#%% Load masks, sort, getting sort index, and plot
sig_idx_arr=np.empty((len(events), len(task_Tags),len(glm_feas),4), dtype=object)
sig_idx_lab=tuple(tuple(item) for item in (events,task_Tags,glm_feas,('all','aud','del','resp')))
sig_idx=LabeledArray(sig_idx_arr,sig_idx_lab)
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
                sig_idx[event, task_Tag, glm_fea, 'all'] = np.array(all_masks_sig)
                # auditory window
                aud_masks_sorted,_,aud_masks_sig = gp.sort_chs_by_actonset(masks,cluster_twin,[-0.1,mean_word_len+auditory_decay])
                gp.plot_chs(aud_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{glm_fea}_aud.jpg'),f"N chs = {len(aud_masks_sig)}")
                sig_idx[event, task_Tag, glm_fea, 'aud'] = np.array(aud_masks_sig)
                # delay window
                del_masks_sorted,_,del_masks_sig = gp.sort_chs_by_actonset(masks,cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
                gp.plot_chs(del_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{glm_fea}_del.jpg'),f"N chs = {len(del_masks_sig)}")
                sig_idx[event, task_Tag, glm_fea, 'del'] = np.array(del_masks_sig)

            elif event=="Resp":
                # response window
                resp_masks_sorted, _, resp_masks_sig = gp.sort_chs_by_actonset(masks, cluster_twin, [-0.1, 5])
                gp.plot_chs(resp_masks_sorted, os.path.join('plot', f'{event}_{task_Tag}_{glm_fea}_resp.jpg'),
                            f"N chs = {len(resp_masks_sig)}")
                sig_idx[event, task_Tag, glm_fea, 'resp'] = np.array(resp_masks_sig)

for md in ['all','aud','del']:
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    gp.plot_wave(stass['Auditory/Repeat/Acoustic'], set(sig_idx['Auditory', 'Repeat', 'Acoustic', md]), 'Acoustic','r')
    gp.plot_wave(stass['Auditory/Repeat/Phonemic'], set(sig_idx['Auditory', 'Repeat', 'Phonemic', md]), 'Phonemic','g')
    gp.plot_wave(stass['Auditory/Repeat/Lexical'], set(sig_idx['Auditory', 'Repeat', 'Lexical', md]), 'Lexical', 'b')
    plt.axvline(x=0, linestyle='--', color='k')
    plt.title(f'Lexical Repeat Delay {md}')
    plt.ylabel('GLM R^2 bsl corrected (-min)')
    plt.legend()
    plt.savefig(os.path.join('plot',f'wave auditory onset {md}.jpg'))

plt.figure(figsize=(Waveplot_wth * (150 / 350), Waveplot_hgt))
gp.plot_wave(stass['Resp/Repeat/Acoustic'], set(sig_idx['Resp', 'Repeat', 'Acoustic', 'resp']), 'Acoustic', 'r')
gp.plot_wave(stass['Resp/Repeat/Phonemic'], set(sig_idx['Resp', 'Repeat', 'Phonemic', 'resp']), 'Phonemic', 'g')
gp.plot_wave(stass['Resp/Repeat/Lexical'], set(sig_idx['Resp', 'Repeat', 'Lexical', 'resp']), 'Lexical', 'b')
plt.axvline(x=0, linestyle='--', color='k')
plt.title('Lexical Repeat Delay resp')
plt.ylabel('GLM R^2 bsl corrected (-min)')
plt.legend()
plt.savefig(os.path.join('plot', 'wave motor onset.jpg'))