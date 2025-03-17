#%% Import everything
import sys
import os
import numpy as np
import json
import glm_utils as glm
import itertools
import glm_validate_plot as glmp
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp

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

#%% Load masks, sort, getting sort index, and plot
for event, task_Tag in itertools.product(events,task_Tags):
    subjs, _, _, chs, times = glm.fifread(event, stat, task_Tag)
    for glm_fea in glm_feas:
        if task_Tag == "Yes_No" and (event != "Resp" or glm_fea != "Lexical"):
            continue
        else:
            masks,_=glm.load_stats(event,stat,task_Tag,'cluster_mask',glm_fea,subjs,chs,times)
            if event=='Auditory':
                # whole trial
                all_masks_sorted,_,all_masks_sig = gp.sort_chs_by_actonset(masks,cluster_twin,[-0.1,5])
                gp.plot_chs(all_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{glm_fea}_all.jpg'),f"N chs = {len(all_masks_sig)}")
                # auditory window
                aud_masks_sorted,_,aud_masks_sig = gp.sort_chs_by_actonset(masks,cluster_twin,[-0.1,mean_word_len+auditory_decay])
                gp.plot_chs(aud_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{glm_fea}_aud.jpg'),f"N chs = {len(aud_masks_sig)}")
                # delay window
                aud_masks_sorted,_,del_masks_sig = gp.sort_chs_by_actonset(masks,cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
                gp.plot_chs(aud_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{glm_fea}_del.jpg'),f"N chs = {len(del_masks_sig)}")
            elif event=="Resp":
                # response window
                resp_masks_sorted, _, resp_masks_sig = gp.sort_chs_by_actonset(masks, cluster_twin, [-0.1, 5])
                gp.plot_chs(resp_masks_sorted, os.path.join('plot', f'{event}_{task_Tag}_{glm_fea}_resp.jpg'),
                            f"N chs = {len(resp_masks_sig)}")
