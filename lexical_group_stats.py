# %% groups of patients
from pickle import FALSE
from matplotlib_venn import venn3

datasource='hg' # 'glm_(Feature)' or 'hg'
#groupsTag="LexDelay"
#groupsTag="LexNoDelay"
groupsTag="LexDelay&LexNoDelay"

# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions
#contrast='ave_YN_Rep' # contrasting yesno to repetition
#contrast='ave_Rep_YN' # contrasting repetition to yesno
#contrast='ave_W_NW' # contrasting word to nonword trials only in repetition
#contrast='ave_NW_W' # contrasting nonword to word trials only in repetition

# For lexical delay task, whether run the data only with repeat tasks
#Delayseleted=''
Delayseleted = '_inRep'
trial_labels='CORRECT'

# Parameters from the lexical delay task
mean_word_len=0.65#0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0 # a short period of time that we may assume auditory decay takes
delay_len=1 # from task script
motor_prep_win=[-0.25,-0.1] # get windows for motor preparation (0.1s to avoid high gamma filter leakage)
motor_resp_win=[-0.1,0.75] # get windows for motor response (0.75s to avoid too much auditory feedback)
pre_stimonset_win=[-0.5,0]
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

# %% Sort data and get significant electrode lists
import os
import pickle
import numpy as np
import pandas as pd
from utils.group import load_stats, sort_chs_by_actonset, plot_chs, plot_brain, plot_wave,set2arr, chs2atlas, atlas2_hist, plot_sig_roi_counts, get_sig_elecs_keyword, get_coor, hickok_roi_sphere, get_sig_roi_counts, plot_roi_counts_comparison, sort_chs_by_actonset_combined, select_electrodes
import matplotlib.pyplot as plt
import projects.GLM.glm_utils as glm

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

fig_save_dir = os.path.join(LAB_root, 'D_Data','LexicalDecRepDelay','Baishen_Figs','LexicalDecRepDelay','group')
if not os.path.exists(os.path.join(fig_save_dir)):
    os.mkdir(os.path.join(fig_save_dir))

stats_save_root = os.path.join(stats_root_delay,'group')
if not os.path.exists(os.path.join(stats_save_root)):
    os.mkdir(os.path.join(stats_save_root))

if groupsTag=="LexDelay":

    if datasource=='hg':

        data_LexDelay_Aud,subjs=load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_delay,stats_root_delay)
        if Delayseleted=='_inRep':
            data_LexDelay_Go, _ = load_stats(stat_type, 'Go'+Delayseleted, contrast, stats_root_delay, stats_root_delay)
        data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_delay, stats_root_delay)

        # Get the ROI of labels
        ch_labels_roi,ch_labels=chs2atlas(subjs,data_LexDelay_Aud.labels[0])

        epoc_LexDelay_Aud,_=load_stats('zscore','Auditory'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels)
        if Delayseleted=='_inRep':
            epoc_LexDelay_Go,_=load_stats('zscore','Go'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels)
        epoc_LexDelay_Resp,_=load_stats('zscore','Resp'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels)

        if trial_labels == 'Word':
            epoc_LexDelay_Aud_nonword,_=load_stats('zscore','Auditory'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels='Nonword')
            epoc_LexDelay_Resp_nonword,_=load_stats('zscore','Resp'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels='Nonword')

    elif datasource.split('_')[0]=='glm':
        subjs, _, _, chs, times = glm.fifread('Auditory_inRep', 'zscore', 'Repeat','ALL')
        data_LexDelay_Aud,epoc_LexDelay_Aud,_=glm.load_stats('Auditory'+Delayseleted,'zscore','Repeat','cluster_mask',datasource.split('_')[1],subjs,chs,times,'ALL')
        _, _, _, _, times = glm.fifread('Resp_inRep', 'zscore', 'Repeat','ALL')
        data_LexDelay_Resp,epoc_LexDelay_Resp,_=glm.load_stats('Resp'+Delayseleted,'zscore','Repeat','cluster_mask',datasource.split('_')[1],subjs,chs,times,'ALL')
        ch_labels_roi, ch_labels = chs2atlas(subjs, data_LexDelay_Aud.labels[0])

elif groupsTag=="LexNoDelay":

    data_LexNoDelay_Aud,subjs=load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_nodelay)
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

    data_LexNoDelay_Silence_Aud,_=load_stats(stat_type,'Auditory_inSilence',contrast,stats_root_nodelay,stats_root_nodelay)

    # Get the ROI of labels
    ch_labels_roi,ch_labels=chs2atlas(subjs,data_LexNoDelay_Aud.labels[0])

    epoc_LexNoDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)
    epoc_LexNoDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

    epoc_LexNoDelay_Silence_Aud,_=load_stats('zscore','Auditory_inSilence','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

    epoc_LexNoDelay_Aud_nonword, _ = load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_nodelay,
                                              trial_labels='Nonword')
    epoc_LexNoDelay_Resp_nonword, _ = load_stats('zscore', 'Resp_inRep', 'epo', stats_root_nodelay, stats_root_nodelay,
                                               trial_labels='Nonword')

    if trial_labels=='Word':
        epoc_LexNoDelay_Aud_nonword,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels='Nonword')
        epoc_LexNoDelay_Resp_nonword,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels='Nonword')

elif groupsTag=="LexDelay&LexNoDelay":

    # first get the patient inform from no delay tasks and then extract the corresponding
    data_LexDelay_Aud,subjs=load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_nodelay,stats_root_delay)
    data_LexNoDelay_Aud,_=load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_nodelay)
    data_LexNoDelay_Silence_Aud,_=load_stats(stat_type,'Auditory_inSilence',contrast,stats_root_nodelay,stats_root_nodelay)

    # Get the ROI of labels
    ch_labels_roi,ch_labels=chs2atlas(subjs,data_LexDelay_Aud.labels[0])

    data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_nodelay, stats_root_delay)
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

    epoc_LexDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_delay,trial_labels=trial_labels)
    epoc_LexNoDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)
    epoc_LexNoDelay_Silence_Aud,_=load_stats('zscore','Auditory_inSilence','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

    if Delayseleted == '_inRep':
        data_LexDelay_Go, _ = load_stats(stat_type, 'Go' + Delayseleted, contrast, stats_root_nodelay, stats_root_delay)
        epoc_LexDelay_Go, _ = load_stats('zscore', 'Go' + Delayseleted, 'epo', stats_root_nodelay, stats_root_delay,
                                         trial_labels=trial_labels)

    epoc_LexDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_delay,trial_labels=trial_labels)
    epoc_LexNoDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

if groupsTag=="LexNoDelay":
    chs_coor=get_coor(data_LexNoDelay_Aud.labels[0],'group')
else:
    chs_coor=get_coor(data_LexDelay_Aud.labels[0],'group')
hickok_roi_labels, hickok_roi_sig_idx=hickok_roi_sphere(chs_coor)

#%% Get sorted electrodes
Lex_idxes = dict()
if "LexDelay" in groupsTag:

    # sort the data according to the onset within a time range (Full)
    data_LexDelay_sorted,_,_,LexDelay_sig_idx = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[-10,10])
    # plot the data
    plot_chs(data_LexDelay_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_sig_idx)}")

    # Get pre-onset activations:
    data_LexDelay_sorted_preonset,_,_,LexDelay_sig_idx_preonset = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,pre_stimonset_win)
    plot_chs(data_LexDelay_sorted_preonset,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-{stat_type}-{contrast}_preonset.jpg'),f"N chs = {len(LexDelay_sig_idx_preonset)}")

    # (Auditory)
    data_LexDelay_Aud_sorted,_,_,LexDelay_Aud_sig_idx = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[-0.1,mean_word_len+auditory_decay])
    plot_chs(data_LexDelay_Aud_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-{'Auditory'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Aud_sig_idx)}")

    # (Delay)
    data_LexDelay_Delay_sorted,_,_,LexDelay_Delay_sig_idx=sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
    plot_chs(data_LexDelay_Delay_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-Delay_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Delay_sig_idx)}")

    # (Go)
    data_LexDelay_Go_sorted, _,_, LexDelay_Go_sig_idx = sort_chs_by_actonset(data_LexDelay_Go, epoc_LexDelay_Go, cluster_twin, [0.25, 0.75])
    plot_chs(data_LexDelay_Go_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-Go_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Go_sig_idx)}")

    # (Motor prepare)
    # !!!!!!!!!!In the future use **set** to replace the list indexing!!!!!!!!!!!!!!!!!

    data_LexDelay_Motor_Prep_sorted, _, _, LexDelay_Motor_Prep_sig_idx = sort_chs_by_actonset(data_LexDelay_Resp,epoc_LexDelay_Resp, cluster_twin, motor_prep_win)
    LexDelay_Motor_Prep_sig_idx = LexDelay_Go_sig_idx & LexDelay_Motor_Prep_sig_idx
    plot_chs(data_LexDelay_Motor_Prep_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Motor_Prep'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Motor_Prep_sig_idx)}")

    # (Motor response)
    data_LexDelay_Motor_Resp_sorted, _, _, LexDelay_Motor_Resp_sig_idx = sort_chs_by_actonset(data_LexDelay_Resp, epoc_LexDelay_Resp, cluster_twin, motor_resp_win)
    plot_chs(data_LexDelay_Motor_Resp_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Motor_Resp'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Motor_Resp_sig_idx)}")

    # Channel selection: Auditory nomotor electrodes (auditory window:1, motor prep: 0)
    LexDelay_Aud_NoMotor_sig_idx = LexDelay_Aud_sig_idx - LexDelay_Motor_Prep_sig_idx

    # Channel selection: Sensorimotor electrodes (auditory window:1, motor prep: 1)
    LexDelay_Sensorimotor_sig_idx = LexDelay_Aud_sig_idx & LexDelay_Motor_Prep_sig_idx

    # Channel selection: Sensory OR motor electrodes (auditory window:1 or motor prep: 1 or motor resp: 1)
    LexDelay_Sensory_OR_Motor_sig_idx = LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx

    # Channel selection: All sig electrodes (auditory window:1 or delay: 1 or motor prep: 1 or motor resp: 1)
    LexDelay_all_sig_idx = LexDelay_Aud_sig_idx | LexDelay_Delay_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx

    # Channel selection: Motor electrodes (auditory window:0, motor resp: 1)
    LexDelay_Motor_sig_idx = LexDelay_Motor_Resp_sig_idx - LexDelay_Aud_sig_idx

    # Channel selection: Delay only electrodes (delay electrodes ,with: auditory window:0, motor prep: 0, motor resp: 0)
    LexDelay_DelayOnly_sig_idx = LexDelay_Delay_sig_idx - (LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx)
    data_LexDelay_DelayOnly = select_electrodes(data_LexDelay_Aud, LexDelay_DelayOnly_sig_idx)
    epoc_LexDelay_DelayOnly = select_electrodes(epoc_LexDelay_Aud, LexDelay_DelayOnly_sig_idx)
    data_LexDelay_DelayOnly_sorted, _, _, _ = sort_chs_by_actonset(data_LexDelay_DelayOnly, epoc_LexDelay_DelayOnly,
                                                              cluster_twin, [-0.1, 1.6],mask_data=True)
    plot_chs(data_LexDelay_DelayOnly_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Delay_only'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_DelayOnly_sig_idx)}")

    # Channel selection: Auditory electrodes in Delay electrodes
    LexDelay_Auditory_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Aud_NoMotor_sig_idx
    data_LexDelay_Auditory_in_Delay = select_electrodes(data_LexDelay_Aud, LexDelay_Auditory_in_Delay_sig_idx)
    epoc_LexDelay_Auditory_in_Delay = select_electrodes(epoc_LexDelay_Aud, LexDelay_Auditory_in_Delay_sig_idx)
    LexDelay_Auditory_in_Delay_sorted, _, _, _ = sort_chs_by_actonset(data_LexDelay_Auditory_in_Delay, epoc_LexDelay_Auditory_in_Delay,
                                                              cluster_twin, [-0.1, 1.6],mask_data=True)
    plot_chs(LexDelay_Auditory_in_Delay_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Auditory_in_Delay'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Auditory_in_Delay_sig_idx)}")

    # Channel selection: Sensorimotor electrodes in Delay electrodes
    LexDelay_Sensorimotor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Sensorimotor_sig_idx
    data_LexDelay_Sensorimotor_in_Delay = select_electrodes(data_LexDelay_Aud, LexDelay_Sensorimotor_in_Delay_sig_idx)
    epoc_LexDelay_Sensorimotor_in_Delay = select_electrodes(epoc_LexDelay_Aud, LexDelay_Sensorimotor_in_Delay_sig_idx)
    LexDelay_Sensorimotor_in_Delay_sorted, _, _, _ = sort_chs_by_actonset(data_LexDelay_Sensorimotor_in_Delay, epoc_LexDelay_Sensorimotor_in_Delay,
                                                              cluster_twin, [-0.1, 1.6],mask_data=True)
    plot_chs(LexDelay_Sensorimotor_in_Delay_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Sensorimotor_in_Delay_'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Sensorimotor_in_Delay_sig_idx)}")


    # Channel selection: Motor electrodes in Delay electrodes
    LexDelay_Motor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Motor_sig_idx
    # Special plot: motor Delay electrodes in Delay:
    data_LexDelay_Mtr_Delay = select_electrodes(data_LexDelay_Aud, LexDelay_Motor_in_Delay_sig_idx)
    epoc_LexDelay_Mtr_Delay = select_electrodes(epoc_LexDelay_Aud, LexDelay_Motor_in_Delay_sig_idx)
    LexDelay_Mtr_Delay_sorted, _, _, _ = sort_chs_by_actonset(data_LexDelay_Mtr_Delay, epoc_LexDelay_Mtr_Delay,
                                                              cluster_twin, [-0.1, 1.6],mask_data=True)
    plot_chs(LexDelay_Mtr_Delay_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Motor_in_Delay_'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Motor_in_Delay_sig_idx)}")


    # Motor_prep only
    LexDelay_Motorprep_Only_sig_idx = (LexDelay_Motor_Prep_sig_idx - (LexDelay_Aud_NoMotor_sig_idx | LexDelay_Sensorimotor_sig_idx | LexDelay_Motor_sig_idx | LexDelay_DelayOnly_sig_idx))

    # Others
    # LexDelay_Other_sig_idx = ((LexDelay_Aud_sig_idx | LexDelay_Delay_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx)
    #                           - (LexDelay_Aud_NoMotor_sig_idx | LexDelay_Sensorimotor_sig_idx | LexDelay_Motor_sig_idx | LexDelay_DelayOnly_sig_idx))

    Lex_idxes['LexDelay_Aud_NoMotor_sig_idx']=LexDelay_Aud_NoMotor_sig_idx
    Lex_idxes['LexDelay_Sensorimotor_sig_idx']=LexDelay_Sensorimotor_sig_idx
    Lex_idxes['LexDelay_Motor_sig_idx']=LexDelay_Motor_sig_idx
    Lex_idxes['LexDelay_Delay_sig_idx']=LexDelay_Delay_sig_idx
    Lex_idxes['LexDelay_DelayOnly_sig_idx']=LexDelay_DelayOnly_sig_idx
    Lex_idxes['LexDelay_Auditory_in_Delay_sig_idx']=LexDelay_Auditory_in_Delay_sig_idx
    Lex_idxes['LexDelay_all_sig_idx']=LexDelay_all_sig_idx
    Lex_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx']=LexDelay_Sensorimotor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Motor_in_Delay_sig_idx']=LexDelay_Motor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Motorprep_Only_sig_idx']=LexDelay_Motorprep_Only_sig_idx

    del data_LexDelay_sorted, data_LexDelay_Aud_sorted, data_LexDelay_Delay_sorted, data_LexDelay_Motor_Prep_sorted, data_LexDelay_Motor_Resp_sorted

    # Get the labels of all the motor electrodes in the temporal lobe
    Mot_elec_in_tmp=get_sig_elecs_keyword(data_LexDelay_Aud,LexDelay_Motor_sig_idx,'T')
if "LexNoDelay" in groupsTag:

    # (Auditory)
    data_LexNoDelay_Aud_sorted,_,_,LexNoDelay_Aud_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Aud,epoc_LexNoDelay_Aud, cluster_twin,[-0.1,mean_word_len+auditory_decay])
    plot_chs(data_LexNoDelay_Aud_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexNoDelay-{'Auditory_inRep'}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexNoDelay_Aud_sig_idx)}")

    # (Go)
    # data_LexNoDelay_Go_sorted, _, LexNoDelay_Go_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Go, cluster_twin, [0.25, 0.75])
    # plot_chs(data_LexNoDelay_Go_sorted, os.path.join(fig_save_dir, f'{'Go_inRep'}_{stat_type}-{contrast}.jpg'))

    # (Motor prepare)
    data_LexNoDelay_Motor_Prep_sorted,_,_,LexNoDelay_Motor_Prep_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, motor_prep_win)
    plot_chs(data_LexNoDelay_Motor_Prep_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexNoDelay-{'Motor_Prep'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexNoDelay_Motor_Prep_sig_idx)}")

    # (Motor response)
    data_LexNoDelay_Motor_Resp_sorted,_,_,LexNoDelay_Motor_Resp_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, motor_resp_win)
    plot_chs(data_LexNoDelay_Motor_Resp_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexNoDelay-{'Motor_Resp'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexNoDelay_Motor_Resp_sig_idx)}")

    # (NoDelay Silence trials Whole win: Encoding)
    data_LexNoDelay_Silence_Encode_sorted,_,_,LexNoDelay_Silence_Encode_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Silence_Aud,epoc_LexNoDelay_Silence_Aud, cluster_twin,[-0.1,mean_word_len+auditory_decay])
    plot_chs(data_LexNoDelay_Silence_Encode_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexNoDelay-{'Auditory_inSilence_Encode'}.jpg'),f"N chs = {len(LexNoDelay_Silence_Encode_sig_idx)}")

    # (NoDelay Silence trials Whole win: Delay)
    data_LexNoDelay_Silence_Del_sorted,_,_,LexNoDelay_Silence_Del_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Silence_Aud,epoc_LexNoDelay_Silence_Aud, cluster_twin,[mean_word_len+auditory_decay,10])
    plot_chs(data_LexNoDelay_Silence_Del_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexNoDelay-{'Auditory_inSilence_Delay'}.jpg'),f"N chs = {len(LexNoDelay_Silence_Del_sig_idx)}")

    # (Encoding electrodes without Delay)
    LexNoDelay_Silence_Encode_Only_sig_idx = LexNoDelay_Silence_Encode_sig_idx.difference(LexNoDelay_Silence_Del_sig_idx)

    # Channel selection: Auditory nomotor electrodes (auditory window:1, motor prep: 0)
    LexNoDelay_Aud_NoMotor_sig_idx = LexNoDelay_Aud_sig_idx - LexNoDelay_Motor_Prep_sig_idx

    # Channel selection: Sensorimotor electrodes (auditory window:1, motor prep: 1)
    LexNoDelay_Sensorimotor_sig_idx = LexNoDelay_Aud_sig_idx & LexNoDelay_Motor_Prep_sig_idx

    # Channel selection: Sensory OR motor electrodes (auditory window:1 or motor prep: 1 or motor resp: 1)
    LexNoDelay_Sensory_OR_Motor_sig_idx = LexNoDelay_Aud_sig_idx | LexNoDelay_Motor_Prep_sig_idx | LexNoDelay_Motor_Resp_sig_idx

    # Channel selection: All sig electrodes (auditory window:1 or delay: 1 or motor prep: 1 or motor resp: 1)
    LexNoDelay_all_sig_idx = LexNoDelay_Aud_sig_idx | LexNoDelay_Motor_Prep_sig_idx | LexNoDelay_Motor_Resp_sig_idx

    # Channel selection: Motor electrodes (auditory window:0, motor resp: 1)
    LexNoDelay_Motor_sig_idx = LexNoDelay_Motor_Resp_sig_idx - LexNoDelay_Aud_sig_idx

    # (Motor prep only)
    LexNoDelay_Motor_Prep_Only_sig_idx = LexNoDelay_Motor_Prep_sig_idx - (LexNoDelay_Aud_sig_idx | LexNoDelay_Motor_sig_idx)

    Lex_idxes['LexNoDelay_Aud_NoMotor_sig_idx'] = LexNoDelay_Aud_NoMotor_sig_idx
    Lex_idxes['LexNoDelay_Sensorimotor_sig_idx'] = LexNoDelay_Sensorimotor_sig_idx
    Lex_idxes['LexNoDelay_Sensory_OR_Motor_sig_idx'] = LexNoDelay_Sensory_OR_Motor_sig_idx
    Lex_idxes['LexNoDelay_all_sig_idx'] = LexNoDelay_all_sig_idx
    Lex_idxes['LexNoDelay_Motor_sig_idx'] = LexNoDelay_Motor_sig_idx
    Lex_idxes['LexNoDelay_Silence_Encode_sig_idx'] = LexNoDelay_Silence_Encode_sig_idx
    Lex_idxes['LexNoDelay_Silence_Encode_Only_sig_idx'] = LexNoDelay_Silence_Encode_Only_sig_idx
    Lex_idxes['LexNoDelay_Silence_Del_sig_idx'] = LexNoDelay_Silence_Del_sig_idx

    if "LexDelay" in groupsTag:
        # With Nodelay Repeat: Overlapped electrodes
        data_LexNoDelay_Repeat_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Aud, cluster_twin,[-0.05,mean_word_len+auditory_decay],sortonset_base=1)
        plot_chs(data_LexNoDelay_Repeat_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_Rep_Shared.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay Rep only'])

        # With Nodelay Repeat: Delay only sig electrodes
        data_LexNoDelay_Repeat_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Aud, cluster_twin,[-0.05,mean_word_len+auditory_decay],sortonset_base=2)
        plot_chs(data_LexNoDelay_Repeat_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_Rep_Delay.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay Rep only'])

        # With Nodelay Repeat: Nodelay only sig electrodes
        data_LexNoDelay_Repeat_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Aud, cluster_twin,[-0.05,mean_word_len+auditory_decay],sortonset_base=3)
        plot_chs(data_LexNoDelay_Repeat_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_Rep_NoDelay.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay Rep only'])

        # With Nodelay JL: Overlapped electrodes
        data_LexNoDelay_Silence_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Silence_Aud, cluster_twin,[-0.05,mean_word_len+auditory_decay],sortonset_base=1)
        plot_chs(data_LexNoDelay_Silence_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_JL_Shared.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay JL only'])

        # With Nodelay JL: Delay only sig electrodes
        data_LexNoDelay_Silence_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Silence_Aud, cluster_twin,[-0.05,mean_word_len+auditory_decay],sortonset_base=2)
        plot_chs(data_LexNoDelay_Silence_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_JL_Delay.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay JL only'])

        # With Nodelay JL: Nodelay only sig electrodes
        data_LexNoDelay_Silence_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Silence_Aud, cluster_twin,[-0.05,mean_word_len+auditory_decay],sortonset_base=3)
        plot_chs(data_LexNoDelay_Silence_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_JL_NoDelay.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay JL only'])

        # With Nodelay JL: Overlapped electrodes (Delay)
        data_LexNoDelay_Silence_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Silence_Aud, cluster_twin,[mean_word_len+auditory_decay,mean_word_len+auditory_decay+delay_len],sortonset_base=1)
        plot_chs(data_LexNoDelay_Silence_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_JL_Shared_Delay.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay JL only'])

        # With Nodelay JL: Delay only sig electrodes (Delay)
        data_LexNoDelay_Silence_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Silence_Aud, cluster_twin,[mean_word_len+auditory_decay,mean_word_len+auditory_decay+delay_len],sortonset_base=2)
        plot_chs(data_LexNoDelay_Silence_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_JL_Delay_Delay.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay JL only'])

        # With Nodelay JL: Nodelay only sig electrodes (Delay)
        data_LexNoDelay_Silence_LexDelay_sorted,_,_,LexNoDelay_Repeat_LexDelay_sig_idx = sort_chs_by_actonset_combined(data_LexDelay_Aud,data_LexNoDelay_Silence_Aud, cluster_twin,[mean_word_len+auditory_decay,mean_word_len+auditory_decay+delay_len],sortonset_base=3)
        plot_chs(data_LexNoDelay_Silence_LexDelay_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'NoDelay_JL_NoDelay_Delay.jpg'),f"N chs = {len(LexNoDelay_Repeat_LexDelay_sig_idx)}",discrete_y=True,discrete_y_lables=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay JL only'])

        # Get all significant electrodes in Delay Repeat tasks:
        print(f'Auditory resp elec. in all sig electrodes in Delay Rep {len(LexDelay_Aud_sig_idx)}, {len(LexDelay_Aud_sig_idx)/len(LexDelay_all_sig_idx)}')
        print(f'MotorPrep only resp elec. in all sig electrodes in Delay Rep {len(LexDelay_Motorprep_Only_sig_idx)}, {len(LexDelay_Motorprep_Only_sig_idx)/len(LexDelay_all_sig_idx)}')
        print(f'Motor resp elec. in all sig electrodes in Delay Rep {len(LexDelay_Motor_sig_idx)}, {len(LexDelay_Motor_sig_idx)/len(LexDelay_all_sig_idx)}')

        # Get all significant electrodes in NoDelay Repeat tasks:
        print(f'Auditory resp elec. in all sig electrodes in NoDelay Rep {len(LexNoDelay_Aud_sig_idx)}, {len(LexNoDelay_Aud_sig_idx)/len(LexNoDelay_all_sig_idx)}')
        print(f'MotorPrep only resp elec. in all sig electrodes in NoDelay Rep {len(LexNoDelay_Motor_Prep_Only_sig_idx)}, {len(LexNoDelay_Motor_Prep_Only_sig_idx)/len(LexNoDelay_all_sig_idx)}')
        print(f'Motor resp elec. in all sig electrodes in NoDelay Rep {len(LexNoDelay_Motor_sig_idx)}, {len(LexNoDelay_Motor_sig_idx)/len(LexNoDelay_all_sig_idx)}')

        # Get Delay electrodes in Delay Repeat tasks (as baseline):
        print(f'Auditory resp elec. in Delay Rep {len(LexDelay_Aud_sig_idx & LexDelay_Delay_sig_idx)}, {len(LexDelay_Aud_sig_idx & LexDelay_Delay_sig_idx) / len(LexDelay_Delay_sig_idx)}')
        print(f'MotorPrep resp elec. in Delay Rep {len(LexDelay_Motorprep_Only_sig_idx & LexDelay_Delay_sig_idx)}, {len(LexDelay_Motorprep_Only_sig_idx & LexDelay_Delay_sig_idx) / len(LexDelay_Delay_sig_idx)}')
        print(f'Motor resp elec. Delay Rep {len(LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx)}, {len(LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx) / len(LexDelay_Delay_sig_idx)}')

        # Get Delay electrodes in No Delay Repeat tasks:
        data_LexNoDelay_Aud_DelDel=select_electrodes(data_LexNoDelay_Aud,LexDelay_Delay_sig_idx)
        data_LexNoDelay_Resp_DelDel=select_electrodes(data_LexNoDelay_Resp,LexDelay_Delay_sig_idx)
        epoc_LexNoDelay_Aud_DelDel=select_electrodes(epoc_LexNoDelay_Aud,LexDelay_Delay_sig_idx)
        epoc_LexNoDelay_Resp_DelDel=select_electrodes(epoc_LexNoDelay_Resp,LexDelay_Delay_sig_idx)
        _, _, _, LexNoDelay_Aud_DelDel_all = sort_chs_by_actonset(data_LexNoDelay_Aud_DelDel,epoc_LexNoDelay_Aud_DelDel,cluster_twin, [-0.1,10])
        _, _, _, LexNoDelay_Aud_DelDel_aud = sort_chs_by_actonset(data_LexNoDelay_Aud_DelDel,epoc_LexNoDelay_Aud_DelDel,cluster_twin, [-0.1,mean_word_len + auditory_decay])
        _, _, _, LexNoDelay_Aud_DelDel_mtrprep = sort_chs_by_actonset(data_LexNoDelay_Resp_DelDel, epoc_LexNoDelay_Resp_DelDel, cluster_twin, motor_prep_win)
        _, _, _, LexNoDelay_Aud_DelDel_mtr = sort_chs_by_actonset(data_LexNoDelay_Resp_DelDel, epoc_LexNoDelay_Resp_DelDel, cluster_twin, motor_resp_win)
        delsm_full=set(range(1, len(LexDelay_Delay_sig_idx)+1))
        print(f'Auditory resp elec. in NoDelay Rep {len(LexNoDelay_Aud_DelDel_aud)}, {len(LexNoDelay_Aud_DelDel_aud) / len(delsm_full)}')
        print(f'Prob. Motorprep resp elec. NoDelay Rep {len(LexNoDelay_Aud_DelDel_mtrprep - (LexNoDelay_Aud_DelDel_aud | LexNoDelay_Aud_DelDel_mtr))}, {len(LexNoDelay_Aud_DelDel_mtrprep - (LexNoDelay_Aud_DelDel_aud | LexNoDelay_Aud_DelDel_mtr)) / len(delsm_full)}')
        print(f'Motor resp elec. NoDelay Rep {len(LexNoDelay_Aud_DelDel_mtr - LexNoDelay_Aud_DelDel_aud)}, {len(LexNoDelay_Aud_DelDel_mtr - LexNoDelay_Aud_DelDel_aud) / len(delsm_full)}')
        print(f'Silent elec. in NoDelay Rep {len(delsm_full-LexNoDelay_Aud_DelDel_all)}, {len(delsm_full-LexNoDelay_Aud_DelDel_all) / len(delsm_full)}')

        # Aud/Mtr delay responses in NoDelay
        len(LexDelay_Aud_sig_idx & LexDelay_Delay_sig_idx & LexNoDelay_Aud_sig_idx)
        len((LexDelay_Aud_sig_idx & LexDelay_Delay_sig_idx) - LexNoDelay_Aud_sig_idx)
        len(LexDelay_Motor_Prep_sig_idx & LexDelay_Delay_sig_idx & LexNoDelay_Motor_Prep_sig_idx)
        len((LexDelay_Motor_Prep_sig_idx & LexDelay_Delay_sig_idx) - LexNoDelay_Motor_Prep_sig_idx)
        len(LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx & LexNoDelay_Motor_sig_idx)
        len((LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx) - LexNoDelay_Motor_sig_idx)
        len((LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx) - LexNoDelay_all_sig_idx)
        len((LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx) & LexNoDelay_Aud_sig_idx)
        len((LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx) & (LexNoDelay_Motor_Prep_sig_idx - (LexNoDelay_Aud_sig_idx | LexNoDelay_Motor_sig_idx)))

        # Venn plot: Delay electrodes in Delay
        plt.figure(figsize=(6, 6))
        venn3([LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx,LexDelay_Motor_Prep_sig_idx & LexDelay_Delay_sig_idx,LexDelay_Aud_sig_idx & LexDelay_Delay_sig_idx],
              (f'Delay_Motor', f'Delay_MotorPrep', f'Delay_Auditory'))
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'pie_Delelectrodes_inDelay.tif'),
                    dpi=300)
        plt.close()

        # Venn plot: Delay electrodes in NoDelay
        plt.figure(figsize=(6, 6))
        venn3([LexNoDelay_Aud_DelDel_mtr - LexNoDelay_Aud_DelDel_aud,LexNoDelay_Aud_DelDel_mtrprep,LexNoDelay_Aud_DelDel_aud],
              (f'NoDelay_Motor', f'NoDelay_MotorPrep', f'NoDelay_Auditory'))
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'pie_Delelectrodes_inNoDelay.tif'),
                    dpi=300)
        plt.close()

       # Get Delay electrodes in No Delay JL tasks:
        data_LexNoDelay_JL_Aud_DelDel=select_electrodes(data_LexNoDelay_Silence_Aud,LexDelay_Delay_sig_idx)
        epoc_LexNoDelay_JL_Aud_DelDel=select_electrodes(epoc_LexNoDelay_Silence_Aud,LexDelay_Delay_sig_idx)
        _, _, _, LexNoDelay_JL_Aud_DelDel_aud = sort_chs_by_actonset(data_LexNoDelay_JL_Aud_DelDel,epoc_LexNoDelay_JL_Aud_DelDel,cluster_twin, [-0.1,mean_word_len + auditory_decay])
        _, _, _, LexNoDelay_JL_Aud_DelDel_del = sort_chs_by_actonset(data_LexNoDelay_JL_Aud_DelDel,epoc_LexNoDelay_JL_Aud_DelDel,cluster_twin, [mean_word_len + auditory_decay,mean_word_len + auditory_decay + delay_len])
        delsm_full=set(range(1, len(LexDelay_Delay_sig_idx)+1))
        print(f'Prob. Auditory resp elec. in NoDelay JL {len(LexNoDelay_JL_Aud_DelDel_aud) / len(delsm_full)}')
        print(f'Prob. Delay-only resp elec. in NoDelay JL {len(LexNoDelay_JL_Aud_DelDel_del - LexNoDelay_JL_Aud_DelDel_aud) / len(delsm_full)}')
        print(f'Prob. Silent elec. in NoDelay JL {len(delsm_full-(LexNoDelay_JL_Aud_DelDel_aud | LexNoDelay_JL_Aud_DelDel_del)) / len(delsm_full)}')

       # Motor delay electrodes in Delay that are Auditory electrodes in NoDelay
        MtrDel_AudNoDel_sig=(LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx) & LexNoDelay_Aud_sig_idx
        data_LexNoDelay_MtrDel_AudNoDel=select_electrodes(data_LexNoDelay_Aud,MtrDel_AudNoDel_sig)
        data_LexDelay_MtrDel_AudNoDel=select_electrodes(data_LexDelay_Aud,MtrDel_AudNoDel_sig)
        epoc_LexNoDelay_MtrDel_AudNoDel=select_electrodes(epoc_LexNoDelay_Aud,MtrDel_AudNoDel_sig)
        epoc_LexDelay_MtrDel_AudNoDel=select_electrodes(epoc_LexDelay_Aud,MtrDel_AudNoDel_sig)
        LexNoDelay_MtrDel_AudNoDel_sorted, _, _, _ = sort_chs_by_actonset(data_LexNoDelay_MtrDel_AudNoDel, epoc_LexNoDelay_MtrDel_AudNoDel,
                                                                  cluster_twin, [-0.1, 1.6], mask_data=True)
        plot_chs(LexNoDelay_MtrDel_AudNoDel_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'LexNoDelay_MtrDel_AudNoDel.jpg'),f"N chs = {len(MtrDel_AudNoDel_sig)}")
        LexDelay_MtrDel_AudNoDel_sorted, _, _, _ = sort_chs_by_actonset(data_LexDelay_MtrDel_AudNoDel, epoc_LexDelay_MtrDel_AudNoDel,
                                                                  cluster_twin, [-0.1, 1.6], mask_data=True)
        plot_chs(LexDelay_MtrDel_AudNoDel_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'LexDelay_MtrDel_AudNoDel.jpg'),f"N chs = {len(MtrDel_AudNoDel_sig)}")


        data_LexNoDelay_MtrDel_AudNoDel=select_electrodes(data_LexNoDelay_Resp,MtrDel_AudNoDel_sig)
        data_LexDelay_MtrDel_AudNoDel=select_electrodes(data_LexDelay_Resp,MtrDel_AudNoDel_sig)
        epoc_LexNoDelay_MtrDel_AudNoDel=select_electrodes(epoc_LexNoDelay_Resp,MtrDel_AudNoDel_sig)
        epoc_LexDelay_MtrDel_AudNoDel=select_electrodes(epoc_LexDelay_Resp,MtrDel_AudNoDel_sig)
        LexNoDelay_MtrDel_AudNoDel_sorted, _, _, _ = sort_chs_by_actonset(data_LexNoDelay_MtrDel_AudNoDel, epoc_LexNoDelay_MtrDel_AudNoDel,
                                                                  cluster_twin, [-0.1, 1.6], mask_data=True)
        plot_chs(LexNoDelay_MtrDel_AudNoDel_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'LexNoDelay_MtrDel_AudNoDel_Resp.jpg'),f"N chs = {len(MtrDel_AudNoDel_sig)}")
        LexDelay_MtrDel_AudNoDel_sorted, _, _, _ = sort_chs_by_actonset(data_LexDelay_MtrDel_AudNoDel, epoc_LexDelay_MtrDel_AudNoDel,
                                                                  cluster_twin, [-0.1, 1.6], mask_data=True)
        plot_chs(LexDelay_MtrDel_AudNoDel_sorted,os.path.join(fig_save_dir,'del_ndel_overlap',f'LexDelay_MtrDel_AudNoDel_Resp.jpg'),f"N chs = {len(MtrDel_AudNoDel_sig)}")


        sig=MtrDel_AudNoDel_sig
        col = [0,1,0]
        chs_sel = data_LexDelay_Aud.labels[0][list(sig)].tolist()
        # cols = [gp.adjust_saturation(np.array(col),val) for val in avg]
        cols = [col for i in range(0, len(sig))]
        plot_brain(subjs, chs_sel, cols, None, dotsize=0.3,
                      fig_save_dir_f=os.path.join('plot', 'xx'))
        atlas2_hist(ch_labels_roi, chs_sel, col, os.path.join(fig_save_dir,'del_ndel_overlap',f'MtrDel_AudNoDel.jpg'),
                       ylim=[0,20])

        Lex_idxes['MtrDel_AudNoDel_sig'] = MtrDel_AudNoDel_sig

    # May do in_Silence electrodes later
Lex_idxes['groupsTag']=groupsTag
with open(os.path.join('projects','GLM','data', f'Lex_twin_idxes_{datasource}.npy'), "wb") as f:
    pickle.dump(Lex_idxes, f)

# %% reassign electrode indices by conditions
MotorPrep_col = [1.0, 0.0784, 0.5765] # Motor prepare
Sensorimotor_col = [1, 0, 0]  # Sensorimotor
Auditory_col = [0, 1, 0]  # Auditory
Delay_col = [1, 0.65, 0]  # Delay
Motor_col = [0, 0, 1]  # Motor
Sensorimotor_Delay_col = Sensorimotor_col#[1, 0, 1]  # Sensorimotor-Delay
Auditory_Delay_col = Auditory_col#[1, 1, 0]  # Auditory-Delay
Delay_Motor_col = Motor_col#[0, 1, 1]  # Delay-Motor
Waveplot_wth=10 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots

if groupsTag == "LexDelay":
    ## Original electrode groups
    # Plot the spatial locations of original groups of electrodes
    len_d=len(data_LexDelay_Aud.labels[0])

    for roi_idx,roi_idx_tag in zip(
            (LexDelay_sig_idx,LexDelay_Delay_sig_idx),
            ('All','Delay')):

        # Overlap between Delay and other types of electrodes
        plt.figure(figsize=(6, 6))
        print(f'pie_original_all_in{roi_idx_tag}.tif')
        venn3([LexDelay_Motor_sig_idx & roi_idx,LexDelay_Motor_Prep_sig_idx & roi_idx,LexDelay_Aud_sig_idx & roi_idx],
              (f'Motor_in{roi_idx_tag}', f'Motorprep_all_in{roi_idx_tag}', f'Auditory_all_in{roi_idx_tag}'))
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'pie_original_all_in{roi_idx_tag}.tif'),
                    dpi=300)
        plt.close()
        print(
            f"Delay only electrodes: {len(LexDelay_Delay_sig_idx - (LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_sig_idx))}")


        for TypeLabel, sig, atlas_hist_ylim,col in zip(
            ('Auditory_all', 'Delay_all', 'Motorprep_all','Motor_noAud_all'),
            (LexDelay_Aud_sig_idx & roi_idx,
             LexDelay_Delay_sig_idx & roi_idx,
             LexDelay_Motor_Prep_sig_idx & roi_idx,
             LexDelay_Motor_sig_idx & roi_idx),
            ([0, 250], [0, 250], [0, 250], [0, 250]),
            (Auditory_col,Delay_col,MotorPrep_col,Motor_col)
        ):
            chs_sel = data_LexDelay_Aud.labels[0][list(sig)].tolist()
            cols = [col for i in range(0, len(sig))]
            plot_brain(subjs, chs_sel, cols, None, dotsize=0.3,
                          fig_save_dir_f=os.path.join('plot', 'x'))
            atlas2_hist(ch_labels_roi, chs_sel, col, os.path.join(fig_save_dir, f'Atlas histogram {TypeLabel} {roi_idx_tag}.tif'),
                           ylim=atlas_hist_ylim)
            plot_sig_roi_counts(hickok_roi_labels, col, sig, os.path.join(fig_save_dir, f'Hickok ROI histogram {TypeLabel} {roi_idx_tag}.tif'))

        # Waves for Auditory, Delay, Motor_Prep, and Motor electrodes
        # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)
        plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
        plt.title('Z-scores in lexical delay repeat tasks (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = False
        plot_wave(epoc_LexDelay_Aud, LexDelay_Aud_sig_idx & roi_idx, f'Auditory_all in {roi_idx_tag} n={len(LexDelay_Aud_sig_idx & roi_idx)}',
                  Auditory_col, '-', wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Delay_sig_idx & roi_idx, f'Delay_all in {roi_idx_tag} n={len(LexDelay_Delay_sig_idx & roi_idx)}',Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Motor_Prep_sig_idx & roi_idx, f'Motorprep_all in {roi_idx_tag} n={len(LexDelay_Motor_Prep_sig_idx & roi_idx)}',MotorPrep_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Motor_sig_idx & roi_idx, f'Motor_noAud_all in {roi_idx_tag} n={len(LexDelay_Motor_sig_idx & roi_idx)}', Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plt.axvline(x=0, linestyle='--', color='k')
        plt.legend(loc='upper right',fontsize=15)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.xlim([-0.25, 1.6])
        plt.savefig(os.path.join(fig_save_dir,f'LexDelay_sig_zscore_org_cat_Aud_{roi_idx_tag}.tif'),dpi=300)
        plt.close()

        # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to Go onset)
        plt.figure(figsize=(Waveplot_wth*(100/350), Waveplot_hgt))
        wav_bsl_corr = False
        plot_wave(epoc_LexDelay_Go, LexDelay_Aud_sig_idx,
                  f'Auditory_all n={len(LexDelay_Aud_sig_idx & roi_idx)}', Auditory_col, '-', False,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Go, LexDelay_Delay_sig_idx & roi_idx, f'Motorprep_all n={len(LexDelay_Delay_sig_idx & roi_idx)}', Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Go, LexDelay_Motor_Prep_sig_idx & roi_idx, f'Delay only n={len(LexDelay_Motor_Prep_sig_idx & roi_idx)}', MotorPrep_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Go, LexDelay_Motor_sig_idx & roi_idx, f'Motor_noAud_all n={len(LexDelay_Motor_sig_idx & roi_idx)}', Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plt.axvline(x=0, linestyle='--', color='k')
        plt.title('(Go aligned)',fontsize=20)
        plt.legend().set_visible(False)
        plt.xlim([-0.25,1])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_org_cat_Go_{roi_idx_tag}.tif'), dpi=300)
        plt.close()

        # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to motor onset)
        plt.figure(figsize=(Waveplot_wth*(100/350), Waveplot_hgt))
        wav_bsl_corr = False
        plot_wave(epoc_LexDelay_Resp, LexDelay_Aud_sig_idx & roi_idx,
                  f'Auditory_all n={len(LexDelay_Aud_sig_idx & roi_idx)}', Auditory_col, '-', False,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Delay_sig_idx & roi_idx, f'Motorprep_all n={len(LexDelay_Delay_sig_idx & roi_idx)}', Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Motor_Prep_sig_idx & roi_idx, f'Delay only n={len(LexDelay_Motor_Prep_sig_idx & roi_idx)}', MotorPrep_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Motor_sig_idx & roi_idx, f'Motor_noAud_all n={len(LexDelay_Motor_sig_idx & roi_idx)}', Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plt.axvline(x=0, linestyle='--', color='k')
        plt.title('(Motor alinged)',fontsize=20)
        plt.legend().set_visible(False)
        plt.xlim([-0.25,1])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_org_cat_Resp_{roi_idx_tag}.tif'), dpi=300)
        plt.close()

    ## Plot electrodes categorized in Aud, Mtr, and SM
    hickok_roi_all = pd.DataFrame()
    # Location plot for different types of electrodes
    for TypeLabel,chs_ov,pick_sig_idx,atlas_hist_ylim in zip(
            ('Sensory-motor','Auditory','Delay','Delay_overlapped','Delay_only','MotorPrep_only','Motor','Sensory_OR_Motor','Hickok_ROI_SM','Hickok_ROI_Delay'),
            ([0,1000,0,0,0],[0,0,100,0,0],[0,0,0,10,0],[10000,1000,100,10,1],[10000,1000,100,10,1],[10000,0,0,0,0],[0,0,0,0,1],[0,1000,100,0,1],[10000,1000,100,10,1],[10000,1000,100,10,1],[10000,1000,100,10,1]),
            (set2arr(LexDelay_Sensorimotor_sig_idx & LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Aud_NoMotor_sig_idx & LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_DelayOnly_sig_idx,len_d),
             set2arr(LexDelay_Motorprep_Only_sig_idx,len_d),
             set2arr(LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Sensory_OR_Motor_sig_idx,len_d),
             set2arr(LexDelay_Sensory_OR_Motor_sig_idx & (hickok_roi_sig_idx['Spt'] | hickok_roi_sig_idx['lPMC'] | hickok_roi_sig_idx['lIPL'] | hickok_roi_sig_idx['lIFG']),len_d),
             set2arr(LexDelay_Delay_sig_idx & (hickok_roi_sig_idx['Spt'] | hickok_roi_sig_idx['lPMC'] | hickok_roi_sig_idx['lIPL'] | hickok_roi_sig_idx['lIFG']),len_d)),
            ([0,200], [0,200], [0,250], [0,250], [0,30], [0,5],[0,200],
             [0,200], [0,40], [0,40], [0,40])
    ):

        # Elecorde selection and color assigning

        color_map = {
           10000: MotorPrep_col, # Motor prepare
            1000: Sensorimotor_col, # Sensorimotor (Orange)
             100: Auditory_col,  # Auditory (Red)
              10: Delay_col,  # Delay (Green)
               1: Motor_col,  # Motor (Blue)
            1010: Sensorimotor_Delay_col,  # Sensorimotor-Delay (Purple)
             110: Auditory_Delay_col, # Auditory-Delay (Yellow)
              11: Delay_Motor_col # Delay-Motor (Greenblue)
        }

        chs_col_idx=[chs_ov[0]*set2arr(LexDelay_Motor_Prep_sig_idx,len_d)[i]
                     +chs_ov[1]*set2arr(LexDelay_Sensorimotor_sig_idx,len_d)[i]
                     +chs_ov[2]*set2arr(LexDelay_Aud_NoMotor_sig_idx,len_d)[i]
                     +chs_ov[3]*set2arr(LexDelay_Delay_sig_idx,len_d)[i]
                     +chs_ov[4]*set2arr(LexDelay_Motor_sig_idx,len_d)[i] for i in range(len_d)]
        picks = [i for i in range(len_d) if pick_sig_idx[i] == 1]
        pick_labels = [data_LexDelay_Aud.labels[0][i] for i in range(len_d) if pick_sig_idx[i] == 1]        # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
        chs_cols =[color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len_d)]
        chs_cols_picked=[chs_cols[i] for i in picks]

        # Plot (cannot plot D107,D042)
        # if TypeLabel=='Motor' or TypeLabel=='Auditory':
        #     label_every=1
        # else:
        #     label_every=None

        # TRY also to plot valid (white?) vs. invalid electrodes (dark grey)
        plot_brain(subjs, pick_labels,chs_cols_picked,None,os.path.join(fig_save_dir,f'{TypeLabel}_{stat_type}-{contrast}.jpg'))
        atlas2_hist(ch_labels_roi,pick_labels,chs_cols_picked[0],os.path.join(fig_save_dir,f'Atlas histogram {TypeLabel.replace('/', ' ')}.tif'),ylim=atlas_hist_ylim)
        plot_sig_roi_counts(hickok_roi_labels, chs_cols_picked[0], pick_sig_idx, os.path.join(fig_save_dir,f'Hickok ROI histogram {TypeLabel.replace('/', ' ')}.tif'))
        if TypeLabel=='Sensory-motor' or TypeLabel=='Auditory' or TypeLabel=='Motor':
            hickok_roi_all[TypeLabel] = get_sig_roi_counts(hickok_roi_labels, {i for i, val in enumerate(pick_sig_idx) if val == 1})

    plot_roi_counts_comparison(hickok_roi_all, os.path.join(fig_save_dir, f'Hickok ROI his {TypeLabel.replace('/', ' ')}'))

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)
    for ROI_idx,ROI_tag in zip(
    (LexDelay_all_sig_idx,hickok_roi_sig_idx['Spt'],hickok_roi_sig_idx['lPMC'], hickok_roi_sig_idx['lIFG']),
            ('All','Spt','lPMC','lIFG')):
        plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
        if datasource == 'hg':
            plt.title('Z-scores in lexical delay repeat tasks (aligned to stim onset)',fontsize=20)
            wav_bsl_corr = False
        elif datasource.split('_')[0]=='glm':
            plt.title('GLM Sum|β| in lexical delay repeat (aligned to stim onset)',fontsize=20)
            wav_bsl_corr = False
        plot_wave(epoc_LexDelay_Aud, LexDelay_Sensorimotor_sig_idx & ROI_idx, f'Sensory-motor n={len(LexDelay_Sensorimotor_sig_idx & ROI_idx)}',
                  Sensorimotor_col, '-', wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Aud_NoMotor_sig_idx & ROI_idx, f'Auditory n={len(LexDelay_Aud_NoMotor_sig_idx & ROI_idx)}',Auditory_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Motor_sig_idx & ROI_idx, f'Motor n={len(LexDelay_Motor_sig_idx & ROI_idx)}',Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_DelayOnly_sig_idx & ROI_idx, f'Delay only n={len(LexDelay_DelayOnly_sig_idx & ROI_idx)}', Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Motorprep_Only_sig_idx & ROI_idx, f'MotorPrep only n={len(LexDelay_Motorprep_Only_sig_idx & ROI_idx)}', MotorPrep_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        if trial_labels=='Word':
            plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_Sensorimotor_sig_idx & ROI_idx, f'Sensory-motor (nonword) n={len(LexDelay_Sensorimotor_sig_idx & ROI_idx)}',
                      Sensorimotor_col, '--', wav_bsl_corr)
            plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_Aud_NoMotor_sig_idx & ROI_idx, f'Auditory (nonword) n={len(LexDelay_Aud_NoMotor_sig_idx & ROI_idx)}',Auditory_col,'--',wav_bsl_corr)
            plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_Motor_sig_idx & ROI_idx, f'Motor (nonword) n={len(LexDelay_Motor_sig_idx & ROI_idx)}',Motor_col,'--',wav_bsl_corr)
            plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_DelayOnly_sig_idx & ROI_idx, f'Delay only (nonword) n={len(LexDelay_DelayOnly_sig_idx & ROI_idx)}', Delay_col,'--',wav_bsl_corr)
        plt.axvline(x=0, linestyle='--', color='k')
        # plt.axhline(y=0, linestyle='--', color='k')
        plt.legend(loc='upper right',fontsize=15)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.xlim([-0.25, 1.6])
        plt.savefig(os.path.join(fig_save_dir,f'LexDelay_sig_zscore_Aud_{ROI_tag}.tif'),dpi=300)
        plt.close()

        # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to motor onset)
        plt.figure(figsize=(Waveplot_wth*(150/350), Waveplot_hgt))
        if datasource.split('_')[0] != 'glm':
            wav_bsl_corr = False
        else:
            wav_bsl_corr = False
        plot_wave(epoc_LexDelay_Resp, LexDelay_Sensorimotor_sig_idx & ROI_idx,
                  f'Sensory-motor n={len(LexDelay_Sensorimotor_sig_idx & ROI_idx)}', Sensorimotor_col, '-', False,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Aud_NoMotor_sig_idx & ROI_idx, f'Auditory n={len(LexDelay_Aud_NoMotor_sig_idx & ROI_idx)}',
                  Auditory_col,'-',False,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Motor_sig_idx & ROI_idx, f'Motor n={len(LexDelay_Motor_sig_idx & ROI_idx)}', Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_DelayOnly_sig_idx & ROI_idx, f'Delay only n={len(LexDelay_DelayOnly_sig_idx & ROI_idx)}', Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Motorprep_Only_sig_idx & ROI_idx, f'MotorPrep only n={len(LexDelay_Motorprep_Only_sig_idx & ROI_idx)}', MotorPrep_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        if trial_labels == 'Word':
            plot_wave(epoc_LexDelay_Resp_nonword, LexDelay_Sensorimotor_sig_idx & ROI_idx,
                      f'Sensory-motor (nonword) n={len(LexDelay_Sensorimotor_sig_idx & ROI_idx)}', Sensorimotor_col, '--',
                      False)
            plot_wave(epoc_LexDelay_Resp_nonword, LexDelay_Aud_NoMotor_sig_idx & ROI_idx,
                      f'Auditory (nonword) n={len(LexDelay_Aud_NoMotor_sig_idx & ROI_idx)}',
                      Auditory_col, '--', False)
            plot_wave(epoc_LexDelay_Resp_nonword, LexDelay_Motor_sig_idx & ROI_idx,
                      f'Motor (nonword) n={len(LexDelay_Motor_sig_idx & ROI_idx)}', Motor_col, '--', wav_bsl_corr)
            plot_wave(epoc_LexDelay_Resp_nonword, LexDelay_DelayOnly_sig_idx & ROI_idx,
                      f'Delay only (nonword) n={len(LexDelay_DelayOnly_sig_idx & ROI_idx)}', Delay_col, '--', wav_bsl_corr)
        plt.axvline(x=0, linestyle='--', color='k')
        # plt.axhline(y=0, linestyle='--', color='k')
        plt.title('(Aligned to motor onset)',fontsize=20)
        plt.legend().set_visible(False)
        plt.xlim([-0.25,1])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_Resp_{ROI_tag}.tif'), dpi=300)
        plt.close()

        # Plot Delay electrodes
        plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
        num_delay_elec=len(LexDelay_Delay_sig_idx & ROI_idx)
        if datasource == 'hg':
            plt.title('(aligned to stim onset)',fontsize=20)
            wav_bsl_corr = False
        elif datasource.split('_')[0]=='glm':
            plt.title('(aligned to stim onset)',fontsize=20)
            wav_bsl_corr = False
        plot_wave(epoc_LexDelay_Aud, LexDelay_DelayOnly_sig_idx & ROI_idx,
                  f'Delay Only n={len(LexDelay_DelayOnly_sig_idx & ROI_idx)} '
                  f'({np.round(100*len(LexDelay_DelayOnly_sig_idx & ROI_idx)/num_delay_elec,3)}%)', Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Motorprep_Only_sig_idx & ROI_idx,
                  f'MotorPrep Only n={len(LexDelay_Motorprep_Only_sig_idx & ROI_idx,)} '
                  f'({np.round(100*len(LexDelay_Motorprep_Only_sig_idx & ROI_idx,)/num_delay_elec,3)}%)', MotorPrep_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Auditory_in_Delay_sig_idx & ROI_idx,
                  f'Auditory in Delay n={len(LexDelay_Auditory_in_Delay_sig_idx & ROI_idx)} '
                  f'({np.round(100*len(LexDelay_Auditory_in_Delay_sig_idx & ROI_idx)/num_delay_elec,3)}%)', Auditory_Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx,
                  f'Sensory-motor in Delay n={len(LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx)} '
                  f'({np.round(100*len(LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx)/num_delay_elec,3)}%)', Sensorimotor_Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Motor_in_Delay_sig_idx & ROI_idx,
                  f'Motor in Delay n={len(LexDelay_Motor_in_Delay_sig_idx & ROI_idx)} '
                  f'({np.round(100*len(LexDelay_Motor_in_Delay_sig_idx & ROI_idx)/num_delay_elec,3)}%)',Delay_Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        if trial_labels == 'Word':
            plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_DelayOnly_sig_idx & ROI_idx,
                      f'Delay Only (nonword) n={len(LexDelay_DelayOnly_sig_idx & ROI_idx)} '
                      f'({np.round(100*len(LexDelay_DelayOnly_sig_idx & ROI_idx)/num_delay_elec,3)}%)', Delay_col,'--',wav_bsl_corr)
            plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_Auditory_in_Delay_sig_idx & ROI_idx,
                      f'Auditory in Delay (nonword) n={len(LexDelay_Auditory_in_Delay_sig_idx & ROI_idx)} '
                      f'({np.round(100*len(LexDelay_Auditory_in_Delay_sig_idx & ROI_idx)/num_delay_elec,3)}%)', Auditory_Delay_col,'--',wav_bsl_corr)
            plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx,
                      f'Sensory-motor in Delay (nonword) n={len(LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx)} '
                      f'({np.round(100*len(LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx)/num_delay_elec,3)}%)', Sensorimotor_Delay_col,'--',wav_bsl_corr)
            plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_Motor_in_Delay_sig_idx & ROI_idx,
                      f'Motor in Delay (nonword) n={len(LexDelay_Motor_in_Delay_sig_idx & ROI_idx)} '
                      f'({np.round(100*len(LexDelay_Motor_in_Delay_sig_idx & ROI_idx)/num_delay_elec,3)}%)',Delay_Motor_col,'--',wav_bsl_corr)
        plt.axvline(x=0, linestyle='--', color='k')
        # plt.axhline(y=0, linestyle='--', color='k')
        plt.legend(fontsize=10)
        # plt.xlim([-0.25, 1.6])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'LexDelay_Delay_sig_zscore_{ROI_tag}.tif'), dpi=300)
        plt.close()

        # Plot Delay electrodes (Aligned to motor onset)
        plt.figure(figsize=(Waveplot_wth*(150/350), Waveplot_hgt))
        plot_wave(epoc_LexDelay_Resp, LexDelay_DelayOnly_sig_idx & ROI_idx, f'Delay Only n={len(LexDelay_DelayOnly_sig_idx & ROI_idx)}',
                  Delay_col,'-',False,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Motorprep_Only_sig_idx & ROI_idx,
                  f'MotorPrep Only n={len(LexDelay_Motorprep_Only_sig_idx & ROI_idx,)} '
                  f'({np.round(100*len(LexDelay_Motorprep_Only_sig_idx & ROI_idx,)/num_delay_elec,3)}%)', MotorPrep_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Auditory_in_Delay_sig_idx & ROI_idx, f'Auditory in Delay n={len(LexDelay_Auditory_in_Delay_sig_idx & ROI_idx)}', Auditory_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx, f'Sensory-motor in Delay n={len(LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx)}', Sensorimotor_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Motor_in_Delay_sig_idx & ROI_idx, f'Motor in Delay n={len(LexDelay_Motor_in_Delay_sig_idx & ROI_idx)}', Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.5])
        if trial_labels == 'Word':
            plot_wave(epoc_LexDelay_Resp_nonword, LexDelay_DelayOnly_sig_idx & ROI_idx, f'Delay Only (nonword) n={len(LexDelay_DelayOnly_sig_idx & ROI_idx)}',
                      Delay_col,'--',False)
            plot_wave(epoc_LexDelay_Resp_nonword, LexDelay_Auditory_in_Delay_sig_idx & ROI_idx, f'Auditory in Delay (nonword) n={len(LexDelay_Auditory_in_Delay_sig_idx & ROI_idx)}', Auditory_col,'--',wav_bsl_corr)
            plot_wave(epoc_LexDelay_Resp_nonword, LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx, f'Sensory-motor in Delay (nonword) n={len(LexDelay_Sensorimotor_in_Delay_sig_idx & ROI_idx)}', Sensorimotor_col,'--',wav_bsl_corr)
            plot_wave(epoc_LexDelay_Resp_nonword, LexDelay_Motor_in_Delay_sig_idx & ROI_idx, f'Motor in Delay (nonword) n={len(LexDelay_Motor_in_Delay_sig_idx & ROI_idx)}', Motor_col,'--',wav_bsl_corr)

        plt.axvline(x=0, linestyle='--', color='k')
        # plt.axhline(y=0, linestyle='--', color='k')
        plt.title('(Aligned to motor onset)',fontsize=20)
        plt.legend().set_visible(False)
        plt.xlim([-0.25,1])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'Le2xDelay_Delay_sig_zscore_Resp_{ROI_tag}.tif'), dpi=300)
        plt.close()

    TypeLabel='Hickok_ROIs'
    chs_ov=[1000,100,10,1]
    pick_sig_idx=set2arr(LexDelay_all_sig_idx & (hickok_roi_sig_idx['Spt'] | hickok_roi_sig_idx['lPMC'] | hickok_roi_sig_idx['lIPL'] | hickok_roi_sig_idx['lIFG']),len_d)
    color_map = {
        1000: Auditory_col,
         100: Sensorimotor_col,
          10: Delay_col,
           1: Motor_col
    }

    chs_col_idx=[chs_ov[0]*set2arr(hickok_roi_sig_idx['Spt'],len_d)[i]
                 +chs_ov[1]*set2arr(hickok_roi_sig_idx['lPMC'],len_d)[i]
                 +chs_ov[2]*set2arr(hickok_roi_sig_idx['lIPL'],len_d)[i]
                 +chs_ov[3]*set2arr(hickok_roi_sig_idx['lIFG'],len_d)[i] for i in range(len_d)]
    picks = [i for i in range(len_d) if pick_sig_idx[i] == 1]
    pick_labels = [data_LexDelay_Aud.labels[0][i] for i in range(len_d) if pick_sig_idx[i] == 1]        # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
    chs_cols =[color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len_d)]
    chs_cols_picked=[chs_cols[i] for i in picks]
    plot_brain(subjs, pick_labels,chs_cols_picked,None,os.path.join(fig_save_dir,f'{TypeLabel}_{stat_type}-{contrast}.jpg'))

    # Plot the Hickok ROI traces
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    if datasource == 'hg':
        plt.title('Z-scores in lexical delay repeat tasks (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = False
        plt.xlim([-0.25, 1.6])
    elif datasource.split('_')[0]=='glm':
        plt.title('GLM Sum|β| in lexical delay repeat (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = False
        plt.xlim([-0.25, 2.5])
    plot_wave(epoc_LexDelay_Aud, LexDelay_all_sig_idx & hickok_roi_sig_idx['Spt'], f'Spt n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['Spt'])}',
              Auditory_col, '-', wav_bsl_corr)
    plot_wave(epoc_LexDelay_Aud, LexDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'], f'lPMC n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'])}',Sensorimotor_col,'-',wav_bsl_corr)
    plot_wave(epoc_LexDelay_Aud, LexDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'], f'lIFG n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'])}',Motor_col,'-',wav_bsl_corr)
    plot_wave(epoc_LexDelay_Aud, LexDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'], f'lIPL n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'])}',Delay_col,'-',wav_bsl_corr)
    if trial_labels=='Word':
        plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_all_sig_idx & hickok_roi_sig_idx['Spt'],
                  f'Spt (Nonword) n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['Spt'])}',
                  Auditory_col, '--', wav_bsl_corr)
        plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'],
                  f'lPMC (Nonword) n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'])}', Sensorimotor_col, '--',
                  wav_bsl_corr)
        plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'],
                  f'lIFG (Nonword) n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'])}', Motor_col, '--',
                  wav_bsl_corr)
        plot_wave(epoc_LexDelay_Aud_nonword, LexDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'], f'lIPL (Nonword) n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'])}',Delay_col,'--',wav_bsl_corr)
    plt.axvline(x=0, linestyle='--', color='k')
    # plt.axhline(y=0, linestyle='--', color='k')
    plt.legend(loc='upper right',fontsize=15)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir,f'LexDelay_sig_zscore_Aud_Hikcok_ROI.tif'),dpi=300)
    plt.close()

    # Plot the Hickok ROIs (Aligned to motor onset)
    plt.figure(figsize=(Waveplot_wth * (150 / 350), Waveplot_hgt))
    plot_wave(epoc_LexDelay_Resp, LexDelay_all_sig_idx & hickok_roi_sig_idx['Spt'],
              f'Spt n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['Spt'])}', Auditory_col, '-', False)
    plot_wave(epoc_LexDelay_Resp, LexDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'],
              f'lPMC n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'])}',
              Sensorimotor_col, '-', False)
    plot_wave(epoc_LexDelay_Resp, LexDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'],
              f'lIPL n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'])}',
              Delay_col, '-', False)
    plot_wave(epoc_LexDelay_Resp, LexDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'], f'lIFG n={len(LexDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'])}',
              Motor_col, '-', False)
    plt.axvline(x=0, linestyle='--', color='k')
    # plt.axhline(y=0, linestyle='--', color='k')
    plt.title('(Aligned to motor onset)', fontsize=20)
    plt.legend().set_visible(True)
    plt.xlim([-0.25, 1])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_Resp_Hikcok_ROI.tif'), dpi=300)
    plt.close()

    # Pie chart for Delay electrodes
    plt.figure()
    plt.title(f'Electrode categories in Delay')
    DLREP_DEL_inDLREP = np.array([len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Motorprep_Only_sig_idx']),
                                  len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Aud_NoMotor_sig_idx']),
                                  len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Sensorimotor_sig_idx']),
                                  len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Motor_sig_idx']),
                                  len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_DelayOnly_sig_idx'])
                                  ])

    DLREP_DEL_inDLREP_labels = [f"Motorprep Only N = {len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Motorprep_Only_sig_idx'])}",
                                f"Auditory N = {len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Aud_NoMotor_sig_idx'])}",
                                f"Sensory-motor N = {len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Sensorimotor_sig_idx'])}",
                                f"Motor N = {len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Motor_sig_idx'])}",
                                f"Delay Only N = {len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_DelayOnly_sig_idx'])}"]
    DLREP_DEL_inDLREP_colors = [MotorPrep_col,Auditory_col, Sensorimotor_col, Motor_col, Delay_col]
    plt.pie(DLREP_DEL_inDLREP, labels=DLREP_DEL_inDLREP_labels, colors=DLREP_DEL_inDLREP_colors, startangle=90,
            autopct='%1.2f%%')
    plt.show()


elif groupsTag == "LexNoDelay":
    len_d = len(data_LexNoDelay_Aud.labels[0])
    hickok_roi_all = pd.DataFrame()
    for TypeLabel, chs_ov, pick_sig_idx in zip(
            ('Sensory-motor', 'Auditory', 'Motor', 'Sensory_OR_Motor'),
            ([1000, 0, 0, 0], [0, 100, 0, 0], [0, 0, 0, 1],[1000, 100, 0, 1]),
            (set2arr(LexNoDelay_Sensorimotor_sig_idx,len_d),
             set2arr(LexNoDelay_Aud_NoMotor_sig_idx,len_d),
             set2arr(LexNoDelay_Motor_sig_idx,len_d),
             set2arr(LexNoDelay_Sensory_OR_Motor_sig_idx,len_d))):

        # Elecorde selection and color assigning

        color_map = {
            1000: Sensorimotor_col,  # Sensorimotor (Orange)
            100: Auditory_col,  # Auditory (Red)
            1: Motor_col  # Motor (Blue)
        }

        chs_col_idx = [
            chs_ov[0] * set2arr(LexNoDelay_Sensorimotor_sig_idx,len_d)[i] +
            chs_ov[1] * set2arr(LexNoDelay_Aud_NoMotor_sig_idx,len_d)[i] +
            chs_ov[3] * set2arr(LexNoDelay_Motor_sig_idx,len_d)[i]
            for i in
            range(len(data_LexNoDelay_Aud.labels[0]))]
        picks = [i for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[i] == 1]
        pick_labels = [data_LexNoDelay_Aud.labels[0][i] for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[
            i] == 1]  # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
        chs_cols = [color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(data_LexNoDelay_Aud.labels[0]))]
        chs_cols_picked = [chs_cols[i] for i in picks]

        # TRY also to plot valid (white?) vs. invalid electrodes (dark grey)
        plot_brain(subjs, pick_labels, chs_cols_picked, None,
                   os.path.join(fig_save_dir, f'{TypeLabel}_{stat_type}-{contrast}.jpg'))

        if TypeLabel == 'Sensory-motor' or TypeLabel == 'Auditory' or TypeLabel == 'Motor':
            hickok_roi_all[TypeLabel] = get_sig_roi_counts(hickok_roi_labels,
                                                           {i for i, val in enumerate(pick_sig_idx) if val == 1})

    plot_roi_counts_comparison(hickok_roi_all, os.path.join(fig_save_dir, f'Hickok ROI his {TypeLabel.replace('/', ' ')}'))

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_Sensorimotor_sig_idx, f'Sensorimotor n={len(LexNoDelay_Sensorimotor_sig_idx)}', Sensorimotor_col,'-',False)
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_Aud_NoMotor_sig_idx, f'Auditory n={len(LexNoDelay_Aud_NoMotor_sig_idx)}',Auditory_col,'-',False)
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_Motor_sig_idx, f'Motor n={len(LexNoDelay_Motor_sig_idx)}',Motor_col,'-',False)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    if datasource == 'hg':
        plt.title('Z-scores in lexical nodelay repeat (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = False
    elif datasource.split('_')[0]=='glm':
        plt.title('GLM Sum|β| in lexical nodelay repeat (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = True
    if datasource=='glm_Phonemic':
        plt.xlim([2.5, 3.5])
    plt.legend(loc='upper right',fontsize=15)
    plt.xlim([-0.25,1.6])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir,'LexNoDelay_sig_zscore_Aud.tif'),dpi=300)
    plt.close()

    # Plot NoDelay Silence trials electrodes (Encoding + Delay)
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plot_wave(epoc_LexNoDelay_Silence_Aud, LexNoDelay_Silence_Encode_sig_idx & LexNoDelay_Silence_Del_sig_idx, f'Nodelay Silence n={len(LexNoDelay_Silence_Encode_sig_idx & LexNoDelay_Silence_Del_sig_idx)}', Sensorimotor_col,'-',False)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    if datasource == 'hg':
        plt.title('Z-scores in lexical nodelay silent (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = False
    elif datasource.split('_')[0]=='glm':
        plt.title('GLM Sum|β| in lexical nodelay silent (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = True
    if datasource=='glm_Phonemic':
        plt.xlim([2.5, 3.5])
    plt.legend(loc='upper right',fontsize=15)
    plt.xlim([-0.25,1.6])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir,'LexNoDelay_sig_zscore_Aud_silence.tif'),dpi=300)
    plt.close()

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to motor onset)
    plt.figure(figsize=(Waveplot_wth*(150/350), Waveplot_hgt))
    plot_wave(epoc_LexNoDelay_Resp, LexNoDelay_Sensorimotor_sig_idx, f'Sensorimotor n={len(LexNoDelay_Sensorimotor_sig_idx)}', Sensorimotor_col,'-',wav_bsl_corr)
    plot_wave(epoc_LexNoDelay_Resp, LexNoDelay_Aud_NoMotor_sig_idx, f'Auditory n={len(LexNoDelay_Aud_NoMotor_sig_idx)}',Auditory_col,'-',wav_bsl_corr)
    plot_wave(epoc_LexNoDelay_Resp, LexNoDelay_Motor_sig_idx, f'Motor n={len(LexNoDelay_Motor_sig_idx)}',Motor_col,'-',wav_bsl_corr)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.title('(aligned to motor onset)',fontsize=20)
    plt.legend().set_visible(False)
    plt.xlim([-0.25,1])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir,'LexNoDelay_sig_zscore_Resp.tif'),dpi=300)
    plt.close()

    TypeLabel='Hickok_ROIs'
    chs_ov=[1000,100,10,1]
    pick_sig_idx=set2arr(LexNoDelay_all_sig_idx & (hickok_roi_sig_idx['Spt'] | hickok_roi_sig_idx['lPMC'] | hickok_roi_sig_idx['lIPL'] | hickok_roi_sig_idx['lIFG']),len_d)
    color_map = {
        1000: Auditory_col,
         100: Sensorimotor_col,
          10: Delay_col,
           1: Motor_col
    }

    chs_col_idx=[chs_ov[0]*set2arr(hickok_roi_sig_idx['Spt'],len_d)[i]
                 +chs_ov[1]*set2arr(hickok_roi_sig_idx['lPMC'],len_d)[i]
                 +chs_ov[2]*set2arr(hickok_roi_sig_idx['lIPL'],len_d)[i]
                 +chs_ov[3]*set2arr(hickok_roi_sig_idx['lIFG'],len_d)[i] for i in range(len_d)]
    picks = [i for i in range(len_d) if pick_sig_idx[i] == 1]
    pick_labels = [data_LexNoDelay_Aud.labels[0][i] for i in range(len_d) if pick_sig_idx[i] == 1]        # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
    chs_cols =[color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len_d)]
    chs_cols_picked=[chs_cols[i] for i in picks]
    plot_brain(subjs, pick_labels,chs_cols_picked,None,os.path.join(fig_save_dir,f'{TypeLabel}_{stat_type}-{contrast}.jpg'))

    # Plot the Hickok ROI traces: LexNoDelay
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plt.title('Z-scores in lexical no delay repeat tasks (aligned to stim onset)',fontsize=20)
    wav_bsl_corr = False
    plt.xlim([-0.25, 1.6])
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_all_sig_idx & hickok_roi_sig_idx['Spt'], f'Spt n={len(LexNoDelay_all_sig_idx & hickok_roi_sig_idx['Spt'])}',
              Auditory_col, '-', wav_bsl_corr)
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'], f'lPMC n={len(LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'])}',Sensorimotor_col,'-',wav_bsl_corr)
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'], f'lIFG n={len(LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'])}',Motor_col,'-',wav_bsl_corr)
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'], f'lIPL n={len(LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'])}',Delay_col,'-',wav_bsl_corr)
    if trial_labels=='Word':
        plot_wave(epoc_LexNoDelay_Aud_nonword,LexNoDelay_all_sig_idx & hickok_roi_sig_idx['Spt'], f'Spt (Nonword) n={len(LexNoDelay_all_sig_idx & hickok_roi_sig_idx['Spt'])}',
                  Auditory_col, '--', wav_bsl_corr)
        plot_wave(epoc_LexNoDelay_Aud_nonword, LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'], f'lPMC (Nonword) n={len(LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lPMC'])}',Sensorimotor_col,'--',wav_bsl_corr)
        plot_wave(epoc_LexNoDelay_Aud_nonword, LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'], f'lIFG (Nonword) n={len(LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lIFG'])}',Motor_col,'--',wav_bsl_corr)
        plot_wave(epoc_LexNoDelay_Aud_nonword, LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'], f'lIPL (Nonword) n={len(LexNoDelay_all_sig_idx & hickok_roi_sig_idx['lIPL'])}',Delay_col,'--',wav_bsl_corr)
    plt.axvline(x=0, linestyle='--', color='k')
    # plt.axhline(y=0, linestyle='--', color='k')
    plt.legend(loc='upper right',fontsize=15)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir,f'LexNoDelay_sig_zscore_Aud_Hikcok_ROI.tif'),dpi=300)
    plt.close()

elif groupsTag=="LexDelay&LexNoDelay":

    tag1='DL'
    tag2='NDL'
    del_aud = Lex_idxes['LexDelay_Aud_NoMotor_sig_idx']
    del_sm = Lex_idxes['LexDelay_Sensorimotor_sig_idx']
    del_del = Lex_idxes['LexDelay_Delay_sig_idx']
    del_mtr = Lex_idxes['LexDelay_Motor_sig_idx']
    del_delol = Lex_idxes['LexDelay_DelayOnly_sig_idx']
    Ndel_aud = Lex_idxes['LexNoDelay_Aud_NoMotor_sig_idx']
    Ndel_sm = Lex_idxes['LexNoDelay_Sensorimotor_sig_idx']
    Ndel_mtr = Lex_idxes['LexNoDelay_Motor_sig_idx']
    del_all = (del_aud | del_sm | del_mtr | Lex_idxes['LexDelay_Motorprep_Only_sig_idx'] | Lex_idxes['LexDelay_DelayOnly_sig_idx'])
    Ndel_all = (Ndel_aud | Ndel_sm | Ndel_mtr)
    # Do the same things but for the Silent trials in lexical No Delay tasks (i.e., just listen)
    Ndel_S_encode_only=Lex_idxes['LexNoDelay_Silence_Encode_Only_sig_idx']
    Ndel_S_del=Lex_idxes['LexNoDelay_Silence_Del_sig_idx']
    Ndel_S_all=(Ndel_S_encode_only | Ndel_S_del)

    import pandas as pd
    import seaborn as sns
    data = {
        "Auditory": [len(del_del & del_aud & Ndel_aud)/len(del_del & del_aud)*100,
                       len(del_del & del_aud & Ndel_sm)/len(del_del & del_aud)*100,
                       len(del_del & del_aud & Ndel_mtr)/len(del_del & del_aud)*100,
                       len((del_del & del_aud).difference(Ndel_all))/len(del_del & del_aud)*100],
        "Sensory-motor": [len(del_del & del_sm & Ndel_aud)/len(del_del & del_sm)*100,
                        len(del_del & del_sm & Ndel_sm)/len(del_del & del_sm)*100,
                        len(del_del & del_sm & Ndel_mtr)/len(del_del & del_sm)*100,
                        len((del_del & del_sm).difference(Ndel_all))/len(del_del & del_sm)*100],
        "Motor": [len(del_del & del_mtr & Ndel_aud)/len(del_del & del_mtr)*100,
                       len(del_del & del_mtr & Ndel_sm)/len(del_del & del_mtr)*100,
                       len(del_del & del_mtr & Ndel_mtr)/len(del_del & del_mtr)*100,
                       len((del_del & del_mtr).difference(Ndel_all))/len(del_del & del_mtr)*100],
        "Delay Only": [len(del_delol & Ndel_aud) / len(del_delol) * 100,
                       len(del_delol & Ndel_sm) / len(del_delol) * 100,
                       len(del_delol & Ndel_mtr) / len(del_delol) * 100,
                       len(del_delol.difference(Ndel_all)) / len(del_delol) * 100]
    }
    df_cm = pd.DataFrame(data, index=["Auditory", "Sensory-motor", "Motor", "Silent"]).transpose()
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="rocket_r", annot_kws={"size": 14}, vmin=0, vmax=80, cbar=False)
    plt.title(f"NoDelay Repeat (% in Delay)")
    plt.ylabel("Delay electrodes in LexDelay Repeat")
    plt.xlabel("LexNoDelay Repeat")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'Confuse_DelNoDeloverlap_rep.tif'), dpi=300)
    plt.close()

    # Plot NoDelay Silence trials electrodes (Encoding + Delay)
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plot_wave(epoc_LexDelay_Aud,del_aud, f'Delay auditory n={len(del_aud)}', Auditory_col,'-',True)
    plot_wave(epoc_LexDelay_Aud,del_sm, f'Delay sensory-motor n={len(del_sm)}', Sensorimotor_col,'-',True)
    plot_wave(epoc_LexDelay_Aud,del_mtr, f'Delay motor n={len(del_mtr)}', Motor_col,'-',True)
    plot_wave(epoc_LexDelay_Aud,del_delol, f'Delay only n={len(del_delol)}', Delay_col,'-',True)
    plot_wave(epoc_LexNoDelay_Silence_Aud, Ndel_S_all, f'NoDelay Silence all n={len(Ndel_S_all)}', [0.5,0.5,0.5],'--',True)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.ylim([-0.1,0.9])
    if datasource == 'hg':
        plt.title('Z-scores in lexical nodelay silent (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = False
    elif datasource.split('_')[0]=='glm':
        plt.title('GLM Sum|β| in lexical nodelay silent (aligned to stim onset)',fontsize=20)
        wav_bsl_corr = True
    if datasource=='glm_Phonemic':
        plt.xlim([2.5, 3.5])
    plt.legend(loc='upper right',fontsize=15)
    plt.xlim([-0.25,1.6])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir,'LexNoDelay_sig_zscore_Aud_delay_silence.tif'),dpi=300)
    plt.close()

    data = {
        "Aud responses": [len(del_del & (del_aud | del_sm) & (Ndel_aud | Ndel_sm))/len(del_del & (del_aud | del_sm))*100,
                       len(del_del & (del_aud | del_sm) & Ndel_mtr)/len(del_del & (del_aud | del_sm))*100,
                       len((del_del & (del_aud | del_sm)).difference(Ndel_all))/len(del_del & (del_aud | del_sm))*100],
        "Mot responses": [len(del_del & (del_mtr | del_sm) & Ndel_aud)/len(del_del & (del_mtr | del_sm))*100,
                       len(del_del & (del_mtr | del_sm) & (Ndel_mtr | Ndel_sm))/len(del_del & (del_mtr | del_sm))*100,
                       len((del_del & (del_mtr | del_sm)).difference(Ndel_all))/len(del_del & (del_mtr | del_sm))*100],
        "Delay Only": [len(del_delol & (Ndel_aud | Ndel_sm)) / len(del_delol) * 100,
                       len(del_delol & (Ndel_mtr | Ndel_sm)) / len(del_delol) * 100,
                       len(del_delol.difference(Ndel_all)) / len(del_delol) * 100]
    }
    df_cm = pd.DataFrame(data, index=["Aud responses", "Mot responses", "Silent"]).transpose()
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="rocket_r", annot_kws={"size": 14}, vmin=0, vmax=80, cbar=False)
    plt.title(f"NoDelay Repeat (% in Delay)")
    plt.ylabel("Delay electrodes in LexDelay Repeat")
    plt.xlabel("LexNoDelay Repeat")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'Confuse_DelNoDeloverlap_rep_A_M.tif'), dpi=300)
    plt.close()

    # Auditory, SM, Motor, and Delay-only electrodes in NoDelay-Silent
    data = {
        "Aud responses": [len(del_del & (del_aud | del_sm) & Ndel_S_all)/len(del_del & (del_aud | del_sm))*100,
                       len((del_del & (del_aud | del_sm)).difference(Ndel_S_all))/len(del_del & (del_aud | del_sm))*100],
        "Mot responses": [len(del_del & (del_mtr | del_sm) & Ndel_S_all)/len(del_del & (del_mtr | del_sm))*100,
                        len((del_del & (del_mtr | del_sm)).difference(Ndel_S_all))/len(del_del & (del_mtr | del_sm))*100],
        "Delay Only": [len(del_delol & Ndel_S_all) / len(del_delol) * 100,
                       len(del_delol.difference(Ndel_S_all)) / len(del_delol) * 100]
    }
    df_cm = pd.DataFrame(data, index=["Active", "Silent"]).transpose()
    plt.figure(figsize=(4, 5))
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="rocket_r", annot_kws={"size": 14}, vmin=0, vmax=80, cbar=False)
    plt.title(f"NoDelay Just Listen (% in Delay)")
    plt.ylabel("Delay electrodes in LexDelay Silent")
    plt.xlabel("LexNoDelay Just Listen")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'Confuse_DelNoDeloverlap_rep_A_M_JL.tif'), dpi=300)
    plt.close()

    # Auditory and Motor responses Delay-only electrodes in NoDelay-Silent
    data = {
        "Auditory": [len(del_del & del_aud & Ndel_S_all)/len(del_del & del_aud)*100,
                       len((del_del & del_aud).difference(Ndel_S_all))/len(del_del & del_aud)*100],
        "Sensory-motor": [len(del_del & del_sm & Ndel_S_all)/len(del_del & del_sm)*100,
                        len((del_del & del_sm).difference(Ndel_S_all))/len(del_del & del_sm)*100],
        "Motor": [len(del_del & del_mtr & Ndel_S_all)/len(del_del & del_mtr)*100,
                       len((del_del & del_mtr).difference(Ndel_S_all))/len(del_del & del_mtr)*100],
        "Delay Only": [len(del_delol & Ndel_S_all) / len(del_delol) * 100,
                       len(del_delol.difference(Ndel_S_all)) / len(del_delol) * 100]
    }
    df_cm = pd.DataFrame(data, index=["Active", "Silent"]).transpose()
    plt.figure(figsize=(4, 5))
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="rocket_r", annot_kws={"size": 14}, vmin=0, vmax=80, cbar=False)
    plt.title(f"NoDelay Just Listen (% in Delay)")
    plt.ylabel("Delay electrodes in LexDelay Repeat")
    plt.xlabel("LexNoDelay Just Listen")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'Confuse_DelNoDeloverlap_jl.tif'), dpi=300)
    plt.close()

    for D_sig,D_tag in zip(
            (del_sm,del_del,del_delol),
            ('SM_DEL','DEL_DEL','DELOL_DEL')):

        if D_tag!='DEL_DEL':
            continue

        # Pie chart (How the Lexical Delay Repeat SM electrodes are distributed )
        plt.figure()
        plt.title(f'{D_tag} in DelRep')
        DLREP_SM_inDLREP = np.array([len(D_sig & del_aud), len(D_sig & del_sm),
                                      len(D_sig & del_mtr), len(D_sig & del_delol)])
        DLREP_SM_inDLREP_labels = [f"Auditory {len(D_sig & del_aud)}",
                                    f"Sensory-motor {len(D_sig & del_sm)}",
                                    f"Motor {len(D_sig & del_mtr)}",
                                    f"Delay-only {len(D_sig & del_delol)}"]
        DLREP_SM_inDLREP_colors = [Auditory_col, Sensorimotor_col, Motor_col, Delay_col]
        plt.pie(DLREP_SM_inDLREP,labels=DLREP_SM_inDLREP_labels,colors=DLREP_SM_inDLREP_colors,startangle=90,autopct='%1.2f%%')
        plt.show()

        # Pie chart
        plt.figure()
        plt.title(f'{D_tag} in NoDelRep')
        DLREP_SM_inNDLREP = np.array([len(D_sig & Ndel_aud), len(D_sig & Ndel_sm),
                                      len(D_sig & Ndel_mtr), len(D_sig - Ndel_all)])
        DLREP_SM_inNDLREP_labels = [f"Auditory {len(D_sig & Ndel_aud)}",
                                    f"Sensory-motor {len(D_sig & Ndel_sm)}",
                                    f"Motor {len(D_sig & Ndel_mtr)}",
                                    f"Silent {len(D_sig - Ndel_all)}"]
        DLREP_SM_inNDLREP_colors = [Auditory_col, Sensorimotor_col, Motor_col, [0.5,0.5,0.5]]
        plt.pie(DLREP_SM_inNDLREP,labels=DLREP_SM_inNDLREP_labels,colors=DLREP_SM_inNDLREP_colors,startangle=90,autopct='%1.2f%%')
        plt.show()

        plt.figure()
        plt.title(f'{D_tag} in NoDelJ')
        DLREP_SM_inNDLJL = np.array([len(D_sig & Ndel_S_encode_only), len(D_sig & Ndel_S_del),
                                      len(D_sig - Ndel_S_all)])
        DLREP_SM_inNDLJL_labels = [f"Encode_Only{len(D_sig & Ndel_S_encode_only)}",
                                   f"Delay{len(D_sig & Ndel_S_del)}",
                                   f"Silent{len(D_sig - Ndel_S_all)}"]
        DLREP_SM_inNDLJL_colors = [Auditory_col, Delay_col, [0.5,0.5,0.5]]
        plt.pie(DLREP_SM_inNDLJL,labels=DLREP_SM_inNDLJL_labels,colors=DLREP_SM_inNDLJL_colors,startangle=90,autopct='%1.2f%%')
        plt.show()

        # Brain plot for SM electrodes in Delay (separated by functions in No Delay)
        len_d=len(data_LexDelay_Aud.labels[0])
        TypeLabel=f'{D_tag}_Delay_Rep in NoDelay Rep'
        cols = np.full((len_d, 3), 0.5)
        cols[list(D_sig & Ndel_aud),:] = Auditory_col
        cols[list(D_sig & Ndel_sm),:] = Sensorimotor_col
        cols[list(D_sig & Ndel_mtr),:] = Motor_col
        cols[list(D_sig - Ndel_all),:] = (1,1,1)
        cols_lst=cols[list(D_sig)].tolist()
        pick_labels=list(data_LexDelay_Aud.labels[0][list(D_sig)])
        plot_brain(subjs, pick_labels,cols_lst,None,os.path.join(fig_save_dir,f'{TypeLabel}_brain.tif'),0.5,0.2)
        # atlas2_hist(ch_labels_roi,list(data_LexDelay_Aud.labels[0][list(del_sm - (Ndel_aud | Ndel_sm))]),[0.5,0.5,0.5],os.path.join(fig_save_dir,f'Atlas histogram silent SM in NDL rep.tif'))

        TypeLabel = f'{D_tag}_Delay_Rep in NoDelay JL'
        cols = np.full((len_d, 3), 0.5)
        cols[list(D_sig & Ndel_S_encode_only), :] = Auditory_col
        cols[list(D_sig & Ndel_S_del), :] = Delay_col
        cols[list(D_sig - Ndel_S_all), :] = (1, 1, 1)
        cols_lst = cols[list(D_sig)].tolist()
        pick_labels = list(data_LexDelay_Aud.labels[0][list(D_sig)])
        plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'{TypeLabel}_brain.tif'), 0.5, 0.2)

        for ele_cat, ele_cat_tag,spec_col in zip(
                ([D_sig & Ndel_aud,D_sig & Ndel_sm,D_sig & Ndel_mtr,D_sig.difference(Ndel_all)],
                 [D_sig & del_aud,D_sig & del_sm,D_sig & del_mtr,del_delol]),
                (['NDL_Aud','NDL_SM','NDL_M','NDL_Silent'],
                ['DL_Aud','DL_SM','DL_M','Delay_Only']),
                 ([0.5,0.5,0.5],Delay_col)
        ):
            for epch_data,epoch_taf,figwid,xlim in zip(
                    (epoc_LexDelay_Aud,epoc_LexNoDelay_Aud,epoc_LexDelay_Resp,epoc_LexNoDelay_Resp),
                    ('Del_Aud','NoDel_Aud','Del_Resp','NoDel_Resp'),
                    (1,1,(150/350),(150/350)),
                    ([-0.5,3],[-0.5,3],[-0.5,0.75],[-0.5,0.75])):
                plt.figure(figsize=(Waveplot_wth*figwid, Waveplot_hgt))
                plot_wave(epch_data, ele_cat[0], ele_cat_tag[0], Auditory_col, '-', False)
                plot_wave(epch_data, ele_cat[1], ele_cat_tag[1], Sensorimotor_col, '-', False)
                plot_wave(epch_data, ele_cat[2], ele_cat_tag[2], Motor_col, '-', False)
                plot_wave(epch_data, ele_cat[3], ele_cat_tag[3],spec_col, '-', False)
                plt.axvline(x=0, linestyle='--', color='k')
                plt.axhline(y=0, linestyle='--', color='k')
                plt.legend(loc='upper right', fontsize=8)
                plt.gca().spines[['top', 'right']].set_visible(False)
                plt.tight_layout()
                plt.xlim(xlim)
                plt.savefig(os.path.join(fig_save_dir, f'{D_tag}_Delay_Rep in NoDelay Rep {epoch_taf} Traces {ele_cat_tag[0]}.tif'), dpi=300)
                plt.close()

        # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)
        for epch_data,epoch_taf,figwid,xlim in zip(
                (epoc_LexDelay_Aud,epoc_LexNoDelay_Silence_Aud,epoc_LexDelay_Resp),
                ('Del_Aud','NoDel_Aud','Del_Resp'),
                (1,1,(150/350)),
                ([-0.5,3],[-0.5,3],[-0.5,0.75])):
            plt.figure(figsize=(Waveplot_wth*figwid, Waveplot_hgt))
            plot_wave(epch_data, D_sig & Ndel_S_encode_only, f'Encode Only', Auditory_col, '-', False)
            plot_wave(epch_data, D_sig & Ndel_S_del, f'Delay', Delay_col, '-', False)
            plot_wave(epch_data, D_sig - Ndel_all, f'Silent', [0.5,0.5,0.5], '-', False)
            plt.axvline(x=0, linestyle='--', color='k')
            plt.axhline(y=0, linestyle='--', color='k')
            plt.legend(loc='upper right', fontsize=8)
            plt.gca().spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.xlim(xlim)
            plt.savefig(os.path.join(fig_save_dir, f'{D_tag}_Delay_Rep in NoDelay JL {epoch_taf} Traces.tif'), dpi=300)
            plt.close()