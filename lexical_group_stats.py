# %% groups of patients
from pickle import FALSE
from matplotlib_venn import venn3

datasource='hg' # 'glm_(Feature)' or 'hg'
groupsTag="LexDelay"
#groupsTag="LexNoDelay"
#groupsTag="LexDelay&LexNoDelay"

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

# Strong Auditory response set
mean_word_len=0.65#0.65 # from utils/lexdelay_get_stim_length.m
auditory_decay=0 # a short period of time that we may assume auditory decay takes
delay_len=1.125 # average length from sound offset to Go onset

# mean_word_len=0.15#0.62 # from utils/lexdelay_get_stim_length.m
# auditory_decay=0 # a short period of time that we may assume auditory decay takes
# delay_len=1.525 # average length from sound offset to Go onset

motor_prep_win=[-0.75,0] # get windows for motor preparation (0.1s to avoid high gamma filter leakage)
motor_resp_win=[0,1] # get windows for motor response (0.75s to avoid too much auditory feedback)
go_resp_win=[0, 0.75] # speech preparation signals from Go onset. 0.25s cut
pre_stimonset_win=[-0.5,0]
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

MotorPrep_col = [1.0, 0.0784, 0.5765] # Motor prepare
Sensorimotor_col = [1, 0, 0]  # Sensorimotor
Auditory_col = [0, 1, 0]  # Auditory
Delay_col = [1, 0.65, 0]  # Delay
Motor_col = [0, 0, 1]  # Motor
Sensorimotor_Delay_col = Sensorimotor_col#[1, 0, 1]  # Sensorimotor-Delay
Auditory_Delay_col = Auditory_col#[1, 1, 0]  # Auditory-Delay
Delay_Motor_col = Motor_col#[0, 1, 1]  # Delay-Motor

# %% Sort data and get significant electrode lists
import os
import pickle
import numpy as np
import pandas as pd
from utils.group import load_stats, sort_chs_by_actonset, plot_chs, plot_brain, plot_wave,set2arr, chs2atlas, atlas2_hist, plot_sig_roi_counts, get_sig_elecs_keyword, get_coor, hickok_roi_sphere, get_sig_roi_counts, plot_roi_counts_comparison, sort_chs_by_actonset_combined, select_electrodes,onsets2col,elegroup_strip, create_gradient
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
        data_LexDelay_Cue, _ = load_stats(stat_type, 'Cue'+Delayseleted, contrast, stats_root_delay, stats_root_delay)
        data_LexDelay_Go, _ = load_stats(stat_type, 'Go'+Delayseleted, contrast, stats_root_delay, stats_root_delay)
        data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_delay, stats_root_delay)

        # Get the ROI of labels
        ch_labels_roi,ch_labels=chs2atlas(subjs,data_LexDelay_Aud.labels[0])

        epoc_LexDelay_Aud,_=load_stats('zscore','Auditory'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels)
        epoc_LexDelay_Cue,_=load_stats('zscore','Cue'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels)
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
    data_LexNoDelay_Cue, _ = load_stats(stat_type, 'Cue_inRep', contrast, stats_root_nodelay, stats_root_nodelay)
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

    # data_LexNoDelay_Silence_Aud,_=load_stats(stat_type,'Auditory_inSilence',contrast,stats_root_nodelay,stats_root_nodelay)

    # Get the ROI of labels
    ch_labels_roi,ch_labels=chs2atlas(subjs,data_LexNoDelay_Aud.labels[0])

    epoc_LexNoDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)
    epoc_LexNoDelay_Cue,_=load_stats('zscore','Cue_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)
    epoc_LexNoDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

    # epoc_LexNoDelay_Silence_Aud,_=load_stats('zscore','Auditory_inSilence','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

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
    # data_LexNoDelay_Silence_Aud,_=load_stats(stat_type,'Auditory_inSilence',contrast,stats_root_nodelay,stats_root_nodelay)

    # Get the ROI of labels
    ch_labels_roi,ch_labels=chs2atlas(subjs,data_LexDelay_Aud.labels[0])

    data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_nodelay, stats_root_delay)
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

    epoc_LexDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_delay,trial_labels=trial_labels)
    epoc_LexNoDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)
    # epoc_LexNoDelay_Silence_Aud,_=load_stats('zscore','Auditory_inSilence','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

    data_LexDelay_Go, _ = load_stats(stat_type, 'Go' + Delayseleted, contrast, stats_root_nodelay, stats_root_delay)
    epoc_LexDelay_Go, _ = load_stats('zscore', 'Go' + Delayseleted, 'epo', stats_root_nodelay, stats_root_delay,
                                     trial_labels=trial_labels)

    data_LexDelay_Cue, _ = load_stats(stat_type, 'Cue' + Delayseleted, contrast, stats_root_nodelay, stats_root_delay)
    epoc_LexDelay_Cue, _ = load_stats('zscore', 'Cue' + Delayseleted, 'epo', stats_root_nodelay, stats_root_delay,
                                      trial_labels =trial_labels)

    data_LexNoDelay_Cue, _ = load_stats(stat_type, 'Cue_inRep', contrast, stats_root_nodelay, stats_root_nodelay)
    epoc_LexNoDelay_Cue,_=load_stats('zscore','Cue_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

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
    data_LexDelay_sorted,_,_,LexDelay_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[-10,10])

    # Get pre-onset activations:
    data_LexDelay_sorted_preonset,_,_,LexDelay_sig_idx_preonset,*_ = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,pre_stimonset_win)

    # (Auditory)
    data_LexDelay_Aud_sorted,_,_,LexDelay_Aud_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[0,mean_word_len+auditory_decay])

    # (Delay)
    data_LexDelay_Delay_sorted,_,_,LexDelay_Delay_sig_idx,*_=sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[mean_word_len+auditory_decay,mean_word_len+auditory_decay+delay_len])

    # (Go)
    data_LexDelay_Go_sorted, _,_, LexDelay_Go_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Go, epoc_LexDelay_Go, cluster_twin, go_resp_win)

    # (Motor response)
    data_LexDelay_Motor_Resp_sorted, _, _, LexDelay_Motor_Resp_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Resp, epoc_LexDelay_Resp, cluster_twin, motor_resp_win)

    # (Motor prepare)
    data_LexDelay_Motor_Prep_sorted, _, _, LexDelay_Motor_Prep_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Resp,epoc_LexDelay_Resp, cluster_twin, motor_prep_win)
    LexDelay_Motor_Prep_sig_idx = LexDelay_Go_sig_idx & LexDelay_Motor_Prep_sig_idx

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

    # Channel selection: Auditory electrodes in Delay electrodes
    LexDelay_Auditory_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Aud_NoMotor_sig_idx

    # Channel selection: Motor electrodes in Delay electrodes
    LexDelay_Motor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Motor_sig_idx

    # Motor_prep only
    LexDelay_Motorprep_Only_sig_idx = (LexDelay_Motor_Prep_sig_idx - (LexDelay_Aud_NoMotor_sig_idx | LexDelay_Sensorimotor_sig_idx | LexDelay_Motor_sig_idx | LexDelay_DelayOnly_sig_idx))

    # Channel selection: Sensorimotor electrodes in Delay electrodes
    LexDelay_Sensorimotor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Sensorimotor_sig_idx

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
    data_LexNoDelay_Aud_sorted,_,_,LexNoDelay_Aud_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Aud,epoc_LexNoDelay_Aud, cluster_twin,[0,mean_word_len+auditory_decay])

    # (Motor prepare)
    data_LexNoDelay_Motor_Prep_sorted,_,_,LexNoDelay_Motor_Prep_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, motor_prep_win)

    # (Motor response)
    data_LexNoDelay_Motor_Resp_sorted,_,_,LexNoDelay_Motor_Resp_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, motor_resp_win)

    # (NoDelay Silence trials Whole win: Encoding)
    # data_LexNoDelay_Silence_Encode_sorted,_,_,LexNoDelay_Silence_Encode_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Silence_Aud,epoc_LexNoDelay_Silence_Aud, cluster_twin,[0,mean_word_len+auditory_decay])

    # (NoDelay Silence trials Whole win: Delay)
    # data_LexNoDelay_Silence_Del_sorted,_,_,LexNoDelay_Silence_Del_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Silence_Aud,epoc_LexNoDelay_Silence_Aud, cluster_twin,[mean_word_len+auditory_decay,10])

    # (Encoding electrodes without Delay)
    # LexNoDelay_Silence_Encode_Only_sig_idx = LexNoDelay_Silence_Encode_sig_idx.difference(LexNoDelay_Silence_Del_sig_idx)

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
    Lex_idxes['LexNoDelay_Aud_sig_idx'] = LexNoDelay_Aud_sig_idx
    Lex_idxes['LexNoDelay_Motor_Resp_sig_idx'] = LexNoDelay_Motor_Resp_sig_idx
    # Lex_idxes['LexNoDelay_Silence_Encode_sig_idx'] = LexNoDelay_Silence_Encode_sig_idx
    # Lex_idxes['LexNoDelay_Silence_Encode_Only_sig_idx'] = LexNoDelay_Silence_Encode_Only_sig_idx
    # Lex_idxes['LexNoDelay_Silence_Del_sig_idx'] = LexNoDelay_Silence_Del_sig_idx

    if "LexDelay" in groupsTag:

        # Get Delay electrodes in No Delay Repeat tasks:
        data_LexNoDelay_Aud_DelDel=select_electrodes(data_LexNoDelay_Aud,LexDelay_Delay_sig_idx)
        data_LexNoDelay_Resp_DelDel=select_electrodes(data_LexNoDelay_Resp,LexDelay_Delay_sig_idx)
        epoc_LexNoDelay_Aud_DelDel=select_electrodes(epoc_LexNoDelay_Aud,LexDelay_Delay_sig_idx)
        epoc_LexNoDelay_Resp_DelDel=select_electrodes(epoc_LexNoDelay_Resp,LexDelay_Delay_sig_idx)
        _, _, _, LexNoDelay_Aud_DelDel_all,*_ = sort_chs_by_actonset(data_LexNoDelay_Aud_DelDel,epoc_LexNoDelay_Aud_DelDel,cluster_twin, [-0.1,10])
        _, _, _, LexNoDelay_Aud_DelDel_aud,*_ = sort_chs_by_actonset(data_LexNoDelay_Aud_DelDel,epoc_LexNoDelay_Aud_DelDel,cluster_twin, [0,mean_word_len])
        _, _, _, LexNoDelay_Aud_DelDel_mtrprep,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp_DelDel, epoc_LexNoDelay_Resp_DelDel, cluster_twin, motor_prep_win)
        _, _, _, LexNoDelay_Aud_DelDel_mtr,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp_DelDel, epoc_LexNoDelay_Resp_DelDel, cluster_twin, motor_resp_win)
        delsm_full=set(range(1, len(LexDelay_Delay_sig_idx)+1))
        print(f'Auditory resp elec. in NoDelay Rep {len(LexNoDelay_Aud_DelDel_aud)}, {len(LexNoDelay_Aud_DelDel_aud) / len(delsm_full)}')
        print(f'Prob. Motorprep resp elec. NoDelay Rep {len(LexNoDelay_Aud_DelDel_mtrprep - (LexNoDelay_Aud_DelDel_aud | LexNoDelay_Aud_DelDel_mtr))}, {len(LexNoDelay_Aud_DelDel_mtrprep - (LexNoDelay_Aud_DelDel_aud | LexNoDelay_Aud_DelDel_mtr)) / len(delsm_full)}')
        print(f'Motor resp elec. NoDelay Rep {len(LexNoDelay_Aud_DelDel_mtr - LexNoDelay_Aud_DelDel_aud)}, {len(LexNoDelay_Aud_DelDel_mtr - LexNoDelay_Aud_DelDel_aud) / len(delsm_full)}')
        print(f'Silent elec. in NoDelay Rep {len(delsm_full-LexNoDelay_Aud_DelDel_all)}, {len(delsm_full-LexNoDelay_Aud_DelDel_all) / len(delsm_full)}')

        # Venn plot: Delay electrodes in Delay
        plt.figure(figsize=(6, 6))
        venn3([LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx,LexDelay_Motor_Prep_sig_idx & LexDelay_Delay_sig_idx,LexDelay_Aud_sig_idx & LexDelay_Delay_sig_idx],
              (f'Delay_Motor', f'Delay_MotorPrep', f'Delay_Auditory'))
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'pie_Delelectrodes_inDelay.tif'),
                    dpi=300)
        plt.close()

    # May do in_Silence electrodes later
Lex_idxes['groupsTag']=groupsTag

Spt_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['Spt']
lPMC_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['lPMC']
lIPL_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['lIPL']
lIFG_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['lIFG']

Lex_idxes['Hikock_Spt']=Spt_sig_idx
Lex_idxes['Hikock_lPMC']=lPMC_sig_idx
Lex_idxes['Hikock_lIPL']=lIPL_sig_idx
Lex_idxes['Hikock_lIFG']=lIFG_sig_idx

with open(os.path.join('projects','GLM','data', f'Lex_twin_idxes_{datasource}.npy'), "wb") as f:
    pickle.dump(Lex_idxes, f)

# %% brain plot and electrode plot for HG time traces
DelNoDel_Aud_paras={}
for sig_idx, sig_tag,align_data,align_epoc in zip(
        (LexDelay_all_sig_idx,LexDelay_Delay_sig_idx,LexDelay_DelayOnly_sig_idx, LexDelay_Auditory_in_Delay_sig_idx, LexDelay_Sensorimotor_in_Delay_sig_idx,
         LexDelay_Motor_in_Delay_sig_idx, LexDelay_Motorprep_Only_sig_idx),
        ('all_sig','Delay_all','DelOnly', 'Auditory_Del', 'SM_Del', 'Motor_Del', 'Motorpreponly'),
        (data_LexDelay_Aud,data_LexDelay_Aud,data_LexDelay_Aud,data_LexDelay_Aud,data_LexDelay_Aud,data_LexDelay_Aud,data_LexDelay_Aud),
        (epoc_LexDelay_Aud,epoc_LexDelay_Aud,epoc_LexDelay_Aud, epoc_LexDelay_Aud, epoc_LexDelay_Aud, epoc_LexDelay_Aud, epoc_LexDelay_Aud)
):
    # Debug:
    # DelNoDel_Aud_paras = {}
    # sig_idx=LexDelay_Sensorimotor_in_Delay_sig_idx
    # sig_tag='SM'
    # align_data=data_LexDelay_Aud
    # align_epoc=epoc_LexDelay_Aud


    # Get sorted indexs (sorted by onsets aligned to xx)
    data_LexDelay_in_Delay = select_electrodes(align_data, sig_idx)
    epoc_LexDelay_in_Delay = select_electrodes(align_epoc, sig_idx)
    _, _, LexDelay_in_Delay_sorted_indices, LexDelay_in_Delay_chs_s_all_idx,onsets_aud,*_ = sort_chs_by_actonset(data_LexDelay_in_Delay, epoc_LexDelay_in_Delay,
                                                             cluster_twin, [-0.1, 10], mask_data=True,select_electrodes=False)
    chs_sel = data_LexDelay_Aud.labels[0][list(sig_idx)].tolist()
    # # if sig_tag=='all_sig':
    # cols = onsets2col(onsets_aud, chs_sel)
    # plot_brain(subjs, chs_sel, cols, None, dotsize=0.2)

    # Aud alinged
    data_LexDelay_in_Delay_Aud = select_electrodes(data_LexDelay_Aud, sig_idx)
    epoc_LexDelay_in_Delay_Aud = select_electrodes(epoc_LexDelay_Aud, sig_idx)
    LexDelay_in_Delay_sorted_Aud,*_ = sort_chs_by_actonset(data_LexDelay_in_Delay_Aud, epoc_LexDelay_in_Delay_Aud,
                                                             cluster_twin, [-2, 3], mask_data=True,sorted_indices=LexDelay_in_Delay_sorted_indices,chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,select_electrodes=False)
    plot_chs(LexDelay_in_Delay_sorted_Aud, os.path.join(fig_save_dir,
                                                    f'{groupsTag}-LexDelay-{sig_tag}_in_Delay_Aud_{Delayseleted}_{stat_type}-{contrast}.jpg'),
             f"N chs = {len(sig_idx)}",percentage_vscale=False,vmin=0,vmax=3,is_colbar=False,fig_size=[4,10*(len(sig_idx)/250)])
    _, _, _, _, _, paras,*_ = sort_chs_by_actonset(
        data_LexDelay_in_Delay_Aud, epoc_LexDelay_in_Delay_Aud,
        cluster_twin, [mean_word_len+auditory_decay, mean_word_len+auditory_decay+delay_len], mask_data=True, sorted_indices=LexDelay_in_Delay_sorted_indices,
        chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx, select_electrodes=False)
    DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned']=paras
    # for column_name in ['activity_length', 'peak_location']:
    #     paras_col=paras[column_name]
    #     cols = onsets2col(paras_col, chs_sel)
    #     plot_brain(subjs, chs_sel, cols, None, dotsize=0.3)
    atlas2_hist(ch_labels_roi, chs_sel, [0.5,0.5,0.5,0.4],
                os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{sig_tag}_in_Delay_Aud_{Delayseleted}_{stat_type}-{contrast}_atlas.jpg'),
                ylim=[0,100])

    # Cue alinged
    data_LexDelay_in_Delay_Cue = select_electrodes(data_LexDelay_Cue, sig_idx)
    epoc_LexDelay_in_Delay_Cue = select_electrodes(epoc_LexDelay_Cue, sig_idx)
    LexDelay_in_Delay_sorted_Cue,_,_,_,onsets_cue,*_ = sort_chs_by_actonset(data_LexDelay_in_Delay_Cue, epoc_LexDelay_in_Delay_Cue,
                                                             cluster_twin, [-0.5, 3], mask_data=True,sorted_indices=LexDelay_in_Delay_sorted_indices,chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,select_electrodes=False)
    plot_chs(LexDelay_in_Delay_sorted_Cue, os.path.join(fig_save_dir,
                                                    f'{groupsTag}-LexDelay-{sig_tag}_in_Delay_Cue_{Delayseleted}_{stat_type}-{contrast}.jpg'),
             f"N chs = {len(sig_idx)}",percentage_vscale=False,vmin=0,vmax=3,is_colbar=False,fig_size=[4,10*(len(sig_idx)/250)])

    # if sig_tag=='all_sig':
    #     cols = onsets2col(onsets_cue, chs_sel)
    #     plot_brain(subjs, chs_sel, cols, None, dotsize=0.2)

    # Go aligned
    data_LexDelay_in_Delay_Go = select_electrodes(data_LexDelay_Go, sig_idx)
    epoc_LexDelay_in_Delay_Go = select_electrodes(epoc_LexDelay_Go, sig_idx)
    LexDelay_in_Delay_Go_sorted,_,_,_,onsets_go,*_ = sort_chs_by_actonset(data_LexDelay_in_Delay_Go,
                                                                epoc_LexDelay_in_Delay_Go,
                                                                cluster_twin, [-2, 1], mask_data=True,sorted_indices=LexDelay_in_Delay_sorted_indices,chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,select_electrodes=False)
    plot_chs(LexDelay_in_Delay_Go_sorted, os.path.join(fig_save_dir,
                                                       f'{groupsTag}-LexDelay-{sig_tag}_in_Delay_Go_{Delayseleted}_{stat_type}-{contrast}.jpg'),
             f"N chs = {len(sig_idx)}",percentage_vscale=False,vmin=0,vmax=3,is_colbar=False,fig_size=[4,10*(len(sig_idx)/250)])

    # if sig_tag=='all_sig':
    #     cols = onsets2col(onsets_go, chs_sel)
    #     plot_brain(subjs, chs_sel, cols, None, dotsize=0.2)

    # Motor aligned
    data_LexDelay_in_Delay_Resp = select_electrodes(data_LexDelay_Resp, sig_idx)
    epoc_LexDelay_in_Delay_Resp = select_electrodes(epoc_LexDelay_Resp, sig_idx)
    LexDelay_in_Delay_Resp_sorted,_,_,_,onsets_mot,*_ = sort_chs_by_actonset(data_LexDelay_in_Delay_Resp,
                                                                  epoc_LexDelay_in_Delay_Resp,
                                                                  cluster_twin, [-2, 1], mask_data=True,sorted_indices=LexDelay_in_Delay_sorted_indices,chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,select_electrodes=False)
    plot_chs(LexDelay_in_Delay_Resp_sorted, os.path.join(fig_save_dir,
                                                         f'{groupsTag}-LexDelay-{sig_tag}_in_Delay_Resp_{Delayseleted}_{stat_type}-{contrast}.jpg'),
             f"N chs = {len(sig_idx)}",percentage_vscale=False,vmin=0,vmax=3,is_colbar=False,fig_size=[4,10*(len(sig_idx)/250)])
    _, _, _, _, _, paras,*_ = sort_chs_by_actonset(data_LexDelay_in_Delay_Resp,
                                                                  epoc_LexDelay_in_Delay_Resp,
        cluster_twin, [-1*(delay_len),-0.1], mask_data=True, sorted_indices=LexDelay_in_Delay_sorted_indices,
        chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx, select_electrodes=False)
    DelNoDel_Aud_paras[f'{sig_tag}_Delay_motaligned']=paras
    # for column_name in ['activity_length', 'peak_location']:
    #     paras_col=paras[column_name]
    #     cols = onsets2col(paras_col, chs_sel)
    #     plot_brain(subjs, chs_sel, cols, None, dotsize=0.3)
    atlas2_hist(ch_labels_roi, chs_sel, [0.5,0.5,0.5,0.4],
                os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{sig_tag}_in_Delay_Resp_{Delayseleted}_{stat_type}-{contrast}_atlas.jpg'),
                ylim=[0,100])

    # if sig_tag=='all_sig':
    #     cols = onsets2col(onsets_mot, chs_sel)
    #     plot_brain(subjs, chs_sel, cols, None, dotsize=0.2)

    if "LexNoDelay" in groupsTag:
        # Cue alinged
        data_LexNoDelay_in_Delay_Cue = select_electrodes(data_LexNoDelay_Cue, sig_idx)
        epoc_LexNoDelay_in_Delay_Cue = select_electrodes(epoc_LexNoDelay_Cue, sig_idx)
        LexNoDelay_in_Delay_sorted_Cue, _, _, _, onsets_cue, *_ = sort_chs_by_actonset(data_LexNoDelay_in_Delay_Cue,
                                                                                     epoc_LexNoDelay_in_Delay_Cue,
                                                                                     cluster_twin, [-0.5, 3],
                                                                                     mask_data=True,
                                                                                     sorted_indices=LexDelay_in_Delay_sorted_indices,
                                                                                     chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,
                                                                                     select_electrodes=False)
        plot_chs(LexNoDelay_in_Delay_sorted_Cue, os.path.join(fig_save_dir,
                                                            f'{groupsTag}-LexNoDelay-{sig_tag}_in_Delay_Cue_{Delayseleted}_{stat_type}-{contrast}.jpg'),
                 f"N chs = {len(sig_idx)}", percentage_vscale=False, vmin=0, vmax=3, is_colbar=False,
                 fig_size=[4*(3/5), 10 * (len(sig_idx) / 250)])

        # NoDelayAud
        data_LexNoDelay_in_Delay_Aud = select_electrodes(data_LexNoDelay_Aud, sig_idx)
        epoc_LexNoDelay_in_Delay_Aud = select_electrodes(epoc_LexNoDelay_Aud, sig_idx)
        LexNoDelay_in_Delay_sorted_Aud,*_ = sort_chs_by_actonset(data_LexNoDelay_in_Delay_Aud,
                                                                         epoc_LexNoDelay_in_Delay_Aud,
                                                                         cluster_twin, [-2, 3], mask_data=True,
                                                                         sorted_indices=LexDelay_in_Delay_sorted_indices,
                                                                         chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,
                                                                         select_electrodes=False)
        plot_chs(LexNoDelay_in_Delay_sorted_Aud, os.path.join(fig_save_dir,
                                                            f'{groupsTag}-LexNoDelay-{sig_tag}_in_Delay_Aud_{Delayseleted}_{stat_type}-{contrast}.jpg'),
                 f"N chs = {len(sig_idx)}", percentage_vscale=False, vmin=0, vmax=3, is_colbar=False,
                 fig_size=[4*(4/5), 10 * (len(sig_idx) / 250)])
        _, _, _, _, _, paras, *_ = sort_chs_by_actonset(data_LexNoDelay_in_Delay_Aud,
                                                                         epoc_LexNoDelay_in_Delay_Aud,
                                                                         cluster_twin, [0, mean_word_len+auditory_decay+delay_len], mask_data=True,
                                                                         sorted_indices=LexDelay_in_Delay_sorted_indices,
                                                                         chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,
                                                                         select_electrodes=False)
        DelNoDel_Aud_paras[f'{sig_tag}_NoDelay_stimaligned'] = paras
        # for column_name in ['activity_length', 'peak_location']:
        #     paras_col = paras[column_name]
        #     cols = onsets2col(paras_col, chs_sel)
        #     plot_brain(subjs, chs_sel, cols, None, dotsize=0.3)
        atlas2_hist(ch_labels_roi, chs_sel, [0.5, 0.5, 0.5, 0.4],
                    os.path.join(fig_save_dir,
                                 f'{groupsTag}-LexNoDelay-{sig_tag}_in_Delay_Aud_{Delayseleted}_{stat_type}-{contrast}_atlas.jpg'),
                    ylim=[0, 100])

        # Motor aligned
        data_LexNoDelay_in_Delay_Resp = select_electrodes(data_LexNoDelay_Resp, sig_idx)
        epoc_LexNoDelay_in_Delay_Resp = select_electrodes(epoc_LexNoDelay_Resp, sig_idx)
        LexNoDelay_in_Delay_Resp_sorted, _, _, _, onsets_mot, *_ = sort_chs_by_actonset(data_LexNoDelay_in_Delay_Resp,
                                                                                      epoc_LexNoDelay_in_Delay_Resp,
                                                                                      cluster_twin, [-2, 1],
                                                                                      mask_data=True,
                                                                                      sorted_indices=LexDelay_in_Delay_sorted_indices,
                                                                                      chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,
                                                                                      select_electrodes=False)
        plot_chs(LexNoDelay_in_Delay_Resp_sorted, os.path.join(fig_save_dir,
                                                             f'{groupsTag}-LexNoDelay-{sig_tag}_in_Delay_Resp_{Delayseleted}_{stat_type}-{contrast}.jpg'),
                 f"N chs = {len(sig_idx)}", percentage_vscale=False, vmin=0, vmax=3, is_colbar=False,
                 fig_size=[4*(3/5), 10 * (len(sig_idx) / 250)])
        _, _, _, _, _, paras, *_ = sort_chs_by_actonset(data_LexNoDelay_in_Delay_Resp,
                                                                                      epoc_LexNoDelay_in_Delay_Resp,
                                                                                      cluster_twin, [-1*(mean_word_len+auditory_decay+delay_len),-0.1],
                                                                                      mask_data=True,
                                                                                      sorted_indices=LexDelay_in_Delay_sorted_indices,
                                                                                      chs_s_all_idx=LexDelay_in_Delay_chs_s_all_idx,
                                                                                      select_electrodes=False)
        DelNoDel_Aud_paras[f'{sig_tag}_NoDelay_motaligned'] = paras
        # for column_name in ['activity_length', 'peak_location']:
        #     paras_col = paras[column_name]
        #     cols = onsets2col(paras_col, chs_sel)
        #     plot_brain(subjs, chs_sel, cols, None, dotsize=0.3)
        atlas2_hist(ch_labels_roi, chs_sel, [0.5, 0.5, 0.5, 0.4],
                    os.path.join(fig_save_dir,
                                 f'{groupsTag}-LexNoDelay-{sig_tag}_in_Delay_Resp_{Delayseleted}_{stat_type}-{contrast}_atlas.jpg'),
                    ylim=[0, 100])

# Summing up key electrodes
electrode_activity_length_dfs=[]
electrode_peak_value_dfs=[]
electrode_rms_value_dfs=[]

# electrode_colorss=[]
ele_grps=[]
for sig_idx, sig_tag in zip(
        (LexDelay_DelayOnly_sig_idx, LexDelay_Auditory_in_Delay_sig_idx, LexDelay_Sensorimotor_in_Delay_sig_idx,
         LexDelay_Motor_in_Delay_sig_idx),
        ('DelOnly', 'Auditory_Del', 'SM_Del', 'Motor_Del')
):
    # electrode_latency_df = DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned'][['first_non_nan_location']]
    electrode_latency_df = DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned'][['activity_length']]

    # Calculate 3 standard deviations for outlier removal
    # 这种删数据的方法可能会被attack，以后再回来认真看看吧。
    peak_values = DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned']['peak_value']
    rms_values = DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned']['sum_value']

    peak_3sd_threshold = peak_values.mean() + 3 * peak_values.std()
    rms_3sd_threshold = rms_values.mean() + 3 * rms_values.std()

    # Filter for values > 0 and <= 3SD
    electrode_peak_value_df = DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned'][['peak_value']][
        (DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned']['peak_value'] > 0) &
        (DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned']['peak_value'] <= peak_3sd_threshold)
    ]
    electrode_rms_value_df = DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned'][['rms_value']][
        (DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned']['rms_value'] > 0) &
        (DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned']['rms_value'] <= rms_3sd_threshold)
    ]
    # electrode_colors=onsets2col(DelNoDel_Aud_paras[f'{sig_tag}_Delay_stimaligned']['activity_length'],data_LexDelay_Aud.labels[0][list(sig_idx)].tolist())
    electrode_activity_length_dfs.append(electrode_latency_df)
    electrode_peak_value_dfs.append(electrode_peak_value_df)
    electrode_rms_value_dfs.append(electrode_rms_value_df)

    # electrode_colorss.append(electrode_colors)
    ele_grps.append(sig_tag)

elegroup_strip(electrode_activity_length_dfs, ele_grps,[Delay_col,Auditory_col,Sensorimotor_col,Motor_col],'Accumulated cluster length')
elegroup_strip(electrode_peak_value_dfs, ele_grps,[Delay_col,Auditory_col,Sensorimotor_col,Motor_col],'Peak of z-scores')
elegroup_strip(electrode_rms_value_dfs, ele_grps,[Delay_col,Auditory_col,Sensorimotor_col,Motor_col],'Sum of z-scores')

# %% reassign electrode indices by conditions
Waveplot_wth=10 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots

if groupsTag == "LexDelay":
    ## Original electrode groups

    # Plot the spatial locations of original groups of electrodes
    len_d=len(data_LexDelay_Aud.labels[0])

    #Plot the overlapping []+Delay and [] without Delay electrodes
    # Delay & whether they are still Encoding electrodes in NoDelay
    for elec_idx,elec_col in zip((LexDelay_Aud_NoMotor_sig_idx,LexDelay_Sensorimotor_sig_idx,LexDelay_Motor_sig_idx,LexDelay_DelayOnly_sig_idx),
                                 (Auditory_col,Sensorimotor_col,Motor_col,Delay_col)):
        mode='all' #'all','with delay', 'without_delay'
        cols = np.full((len_d, 3), 0.5)
        cols[list(elec_idx & LexDelay_Delay_sig_idx), :] = elec_col
        if len(elec_idx - LexDelay_Delay_sig_idx)>0:
            cols[list(elec_idx - LexDelay_Delay_sig_idx), :] = create_gradient(elec_col,6)[4]
        print(f'In Delay {len(elec_idx & LexDelay_Delay_sig_idx)} Not in Delay {len(elec_idx - LexDelay_Delay_sig_idx)}')
        if mode=='all':
            cols_lst = cols[list(elec_idx)].tolist()
            pick_labels = list(data_LexDelay_Aud.labels[0][list(elec_idx)])
        elif mode=='with_delay':
            cols_lst = cols[list(elec_idx & LexDelay_Delay_sig_idx)].tolist()
            pick_labels = list(data_LexDelay_Aud.labels[0][list(elec_idx & LexDelay_Delay_sig_idx)])
        elif mode=='without_delay' and len(elec_idx - LexDelay_Delay_sig_idx)>0:
            cols_lst = cols[list(elec_idx - LexDelay_Delay_sig_idx)].tolist()
            pick_labels = list(data_LexDelay_Aud.labels[0][list(elec_idx - LexDelay_Delay_sig_idx)])
        plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'brain.tif'), 0.3,hemi='lh')
        plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'brain.tif'), 0.3,hemi='rh')

    for roi_idx,roi_idx_tag in zip(
            (LexDelay_Delay_sig_idx,LexDelay_all_sig_idx-LexDelay_Delay_sig_idx,),
            ('Delay','Without_Delay',)):

        for TypeLabel, sig, atlas_hist_ylim,col in zip(
            ('Auditory', 'Delay', 'Sensory-motor','Motor'),
            (LexDelay_Aud_NoMotor_sig_idx & roi_idx,
             LexDelay_DelayOnly_sig_idx & roi_idx,
             LexDelay_Sensorimotor_sig_idx & roi_idx,
             LexDelay_Motor_sig_idx & roi_idx),
            ([0, 200], [0, 200], [0, 200], [0, 200]),
            (Auditory_col,Delay_col,Sensorimotor_col,Motor_col)
        ):
            if roi_idx_tag=='Without_Delay' and TypeLabel=='Delay':
                continue
            chs_sel = data_LexDelay_Aud.labels[0][list(sig)].tolist()
            cols = [col for i in range(0, len(sig))]
            plot_brain(subjs, chs_sel, cols, None, dotsize=0.3,
                          fig_save_dir_f=os.path.join('plot', 'x'))
            atlas2_hist(ch_labels_roi, chs_sel, col, os.path.join(fig_save_dir, f'Atlas histogram {TypeLabel} {roi_idx_tag}.tif'),
                           ylim=atlas_hist_ylim,pie_label_col_base=col)
            # atlas2_hist(ch_labels_roi, chs_sel, col, os.path.join(fig_save_dir, f'Atlas histogram {TypeLabel} {roi_idx_tag} Percent.tif'),
            #                ylim=[0,50],is_percentage=True)
            plot_sig_roi_counts(hickok_roi_labels, col, sig, os.path.join(fig_save_dir, f'Hickok ROI histogram {TypeLabel} {roi_idx_tag}.tif'))

        if roi_idx_tag=='Without_Delay':
            continue

        # Waves for Auditory, Delay, Motor_Prep, and Motor electrodes
        # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)
        plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
        plt.title('High gamma z-score traces (Stim aligned)',fontsize=20)
        wav_bsl_corr = True
        plot_wave(epoc_LexDelay_Aud, LexDelay_Aud_NoMotor_sig_idx & roi_idx, f'Auditory Delay n={len(LexDelay_Aud_NoMotor_sig_idx & roi_idx)}',
                  Auditory_col, '-', wav_bsl_corr,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Aud, LexDelay_DelayOnly_sig_idx & roi_idx, f'Delay Only n={len(LexDelay_DelayOnly_sig_idx & roi_idx)}',Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Aud,LexDelay_Sensorimotor_sig_idx & roi_idx, f'Sensory-motor Delay n={len(LexDelay_Sensorimotor_sig_idx & roi_idx)}',Sensorimotor_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Motor_sig_idx & roi_idx, f'Motor Delay n={len(LexDelay_Motor_sig_idx & roi_idx)}', Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plt.axvline(x=0, linestyle='--', color='k')
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.legend(loc='upper right',fontsize=15)
        plt.tick_params(axis='both', labelsize=16)
        plt.xticks(rotation=45)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.xlim([-0.25, 1.6])
        plt.savefig(os.path.join(fig_save_dir,f'LexDelay_sig_zscore_org_cat_Aud_{roi_idx_tag}.tif'),dpi=300)
        plt.close()

        # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to Go onset)
        plt.figure(figsize=(Waveplot_wth*(100/350), Waveplot_hgt))
        wav_bsl_corr = False
        plot_wave(epoc_LexDelay_Go, LexDelay_Aud_NoMotor_sig_idx,
                  f'Auditory_all n={len(LexDelay_Aud_NoMotor_sig_idx & roi_idx)}', Auditory_col, '-', False,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Go, LexDelay_DelayOnly_sig_idx & roi_idx, f'Delay Only n={len(LexDelay_DelayOnly_sig_idx & roi_idx)}', Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Go,LexDelay_Sensorimotor_sig_idx & roi_idx, f'Sensorimotor n={len(LexDelay_Sensorimotor_sig_idx & roi_idx)}',Sensorimotor_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Go, LexDelay_Motor_sig_idx & roi_idx, f'Motor_noAud_all n={len(LexDelay_Motor_sig_idx & roi_idx)}', Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plt.axvline(x=0, linestyle='--', color='k')
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.title('(Go aligned)',fontsize=20)
        plt.legend().set_visible(False)
        plt.tick_params(axis='both', labelsize=16)
        plt.xticks(rotation=45)
        plt.xlim([-0.25,1])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_org_cat_Go_{roi_idx_tag}.tif'), dpi=300)
        plt.close()

        # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to motor onset)
        plt.figure(figsize=(Waveplot_wth*(100/350), Waveplot_hgt))
        wav_bsl_corr = False
        plot_wave(epoc_LexDelay_Resp, LexDelay_Aud_NoMotor_sig_idx & roi_idx,
                  f'Auditory_all n={len(LexDelay_Aud_NoMotor_sig_idx & roi_idx)}', Auditory_col, '-', False,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Resp, LexDelay_DelayOnly_sig_idx & roi_idx, f'Delay Only n={len(LexDelay_DelayOnly_sig_idx & roi_idx)}', Delay_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Resp,LexDelay_Sensorimotor_sig_idx & roi_idx, f'Sensorimotor n={len(LexDelay_Sensorimotor_sig_idx & roi_idx)}',Sensorimotor_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Motor_sig_idx & roi_idx, f'Motor_noAud_all n={len(LexDelay_Motor_sig_idx & roi_idx)}', Motor_col,'-',wav_bsl_corr,ylim=[-0.2,1.3])
        plt.axvline(x=0, linestyle='--', color='k')
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.title('(Motor aligned)',fontsize=20)
        plt.legend().set_visible(False)
        plt.tick_params(axis='both', labelsize=16)
        plt.xticks(rotation=45)
        plt.xlim([-0.25,1])
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_org_cat_Resp_{roi_idx_tag}.tif'), dpi=300)
        plt.close()

    ## Plot Hikcok's ROI and traces:

    roi_idx_tag='Hickok ROI'

    # Waves for Auditory, Delay, Motor_Prep, and Motor electrodes
    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)

    # Brain plot for electrode distribution:
    len_d = len(data_LexDelay_Aud.labels[0])
    TypeLabel = f'Hikock'
    cols = np.full((len_d, 3), 0.5)
    cols[list(Spt_sig_idx), :] = Auditory_col
    cols[list(lPMC_sig_idx), :] = Sensorimotor_col
    cols[list(lIPL_sig_idx), :] = Delay_col
    cols[list(lIFG_sig_idx), :] = Motor_col
    cols_lst = cols[list(Spt_sig_idx | lPMC_sig_idx | lIPL_sig_idx | lIFG_sig_idx)].tolist()
    pick_labels = list(data_LexDelay_Aud.labels[0][list(Spt_sig_idx | lPMC_sig_idx | lIPL_sig_idx | lIFG_sig_idx)])
    plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'{TypeLabel}_brain.tif'), 0.3, 0.2)

    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plt.title('High gamma z-score traces (Stim aligned)', fontsize=20)
    wav_bsl_corr = True
    plot_wave(epoc_LexDelay_Aud, Spt_sig_idx,f'Spt n={len(Spt_sig_idx)}',Auditory_col, '-', wav_bsl_corr, ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Aud, lPMC_sig_idx,f'lPMC n={len(lPMC_sig_idx)}', Sensorimotor_col, '-', wav_bsl_corr,ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Aud, lIPL_sig_idx,f'lIPL n={len(lIPL_sig_idx)}', Delay_col, '-',wav_bsl_corr, ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Aud, lIFG_sig_idx,f'lIFG n={len(lIFG_sig_idx)}', Motor_col, '-', wav_bsl_corr, ylim=[-0.2, 1.6])
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.legend(loc='upper right', fontsize=15)
    plt.tick_params(axis='both', labelsize=16)
    plt.xticks(rotation=45)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.xlim([-0.25, 1.6])
    plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_org_cat_Aud_{roi_idx_tag}.tif'), dpi=300)
    plt.close()

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to Go onset)
    plt.figure(figsize=(Waveplot_wth * (100 / 350), Waveplot_hgt))
    wav_bsl_corr = False
    plot_wave(epoc_LexDelay_Go, Spt_sig_idx,f'Spt n={len(Spt_sig_idx)}',Auditory_col, '-', wav_bsl_corr, ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Go, lPMC_sig_idx,f'lPMC n={len(lPMC_sig_idx)}', Sensorimotor_col, '-', wav_bsl_corr,ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Go, lIPL_sig_idx,f'lIPL n={len(lIPL_sig_idx)}', Delay_col, '-',wav_bsl_corr, ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Go, lIFG_sig_idx,f'lIFG n={len(lIFG_sig_idx)}', Motor_col, '-', wav_bsl_corr, ylim=[-0.2, 1.6])
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.title('(Go aligned)', fontsize=20)
    plt.legend().set_visible(False)
    plt.tick_params(axis='both', labelsize=16)
    plt.xticks(rotation=45)
    plt.xlim([-0.25, 1])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_org_cat_Go_{roi_idx_tag}.tif'), dpi=300)
    plt.close()

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to motor onset)
    plt.figure(figsize=(Waveplot_wth * (100 / 350), Waveplot_hgt))
    wav_bsl_corr = False
    plot_wave(epoc_LexDelay_Resp, Spt_sig_idx,f'lPMC n={len(Spt_sig_idx)}', Sensorimotor_col, '-', wav_bsl_corr,ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Resp, lPMC_sig_idx,f'lPMC n={len(lPMC_sig_idx)}', Sensorimotor_col, '-', wav_bsl_corr,ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Resp, lIPL_sig_idx,f'lIPL n={len(lIPL_sig_idx)}', Delay_col, '-',wav_bsl_corr, ylim=[-0.2, 1.6])
    plot_wave(epoc_LexDelay_Resp, lIFG_sig_idx,f'lIFG n={len(lIFG_sig_idx)}', Motor_col, '-', wav_bsl_corr, ylim=[-0.2, 1.6])
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.title('(Motor aligned)', fontsize=20)
    plt.legend().set_visible(False)
    plt.tick_params(axis='both', labelsize=16)
    plt.xticks(rotation=45)
    plt.xlim([-0.5, 1])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_org_cat_Resp_{roi_idx_tag}.tif'), dpi=300)
    plt.close()

    # Plot Hickok ROI waves by electrodes aligned to Motor onsets
    # sig/nonsig before Motor onset
    for Hickok_roi_gp, col, tag in zip(
            (Spt_sig_idx, lPMC_sig_idx, lIFG_sig_idx),
            (Auditory_col, Sensorimotor_col, Motor_col),
            ('Spt', 'lPMC', 'lIFG')
    ):
        for data_epoch,epoc_epoch,wav_fig_size,wav_x_lim,epoch_tag in zip(
                (data_LexDelay_Resp,data_LexDelay_Aud,data_LexDelay_Go),
                (epoc_LexDelay_Resp,epoc_LexDelay_Aud,epoc_LexDelay_Go),
                ((Waveplot_wth * (6/5)*(100 / 350), Waveplot_hgt),(Waveplot_wth, Waveplot_hgt),(Waveplot_wth * (12/5)*(100 / 350), Waveplot_hgt)),
                ([-0.5, 1],[-0.25, 1.6],[-2, 1]),
                ('Resp','Stim','Go')
        ):

            # Clus plots
            Hickok_ROI_data = select_electrodes(data_epoch, Hickok_roi_gp)
            Hickok_ROI_epoch = select_electrodes(epoc_epoch, Hickok_roi_gp)
            Hickok_ROI_epoch_sort_unmask,*_ = sort_chs_by_actonset(Hickok_ROI_data,
                                                                  Hickok_ROI_epoch,
                                                                  cluster_twin, wav_x_lim,
                                                                  mask_data=False,
                                                                  select_electrodes=False)
            if epoch_tag == 'Resp':
                Hickok_ROI_epoch_sort, _, Hickok_ROI_epoch_sort_idx, *_ = sort_chs_by_actonset(Hickok_ROI_data,
                                                                                              Hickok_ROI_epoch,
                                                                                              cluster_twin, wav_x_lim,
                                                                                              mask_data=True,
                                                                                              select_electrodes=False)
            else:
                Hickok_ROI_epoch_sort, *_ = sort_chs_by_actonset(Hickok_ROI_data,
                                                                  Hickok_ROI_epoch,
                                                                  cluster_twin, wav_x_lim,
                                                                  sorted_indices=Hickok_ROI_epoch_sort_idx,
                                                                  mask_data=True,
                                                                  select_electrodes=False)
            plot_chs(Hickok_ROI_epoch_sort, os.path.join(fig_save_dir,
                                                                 f'Hickok_sig_alg_resp_{tag}_{epoch_tag}.jpg'),
                     tag, percentage_vscale=False, vmin=0, vmax=2, is_colbar=False,
                     fig_size=[4, 20 * (len(Hickok_roi_gp) / 250)])

            # wave
            plt.figure(figsize=wav_fig_size)
            wav_bsl_corr = False
            plot_wave(Hickok_ROI_epoch_sort_unmask, Hickok_ROI_epoch_sort_idx,f'',col, '-', wav_bsl_corr, ylim=[-0.4, 3.5],average_trace=False)
            plt.axvline(x=0, linestyle='--', color='k')
            plt.axhline(y=0, linestyle='--', color='gray')
            plt.title(tag, fontsize=20)
            plt.tick_params(axis='both', labelsize=16)
            plt.xticks(rotation=45)
            plt.xlim(wav_x_lim)
            plt.gca().spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_save_dir, f'Hickok_wave_alg_resp_{tag}_{epoch_tag}.tif'), dpi=300)
            plt.close()

            # brain plot (sig vs. nonsig before onset)
            cols = np.full((len_d, 3), 0.5)
            for col_idx_i in range(len(Hickok_roi_gp)):
                cols[list(Hickok_roi_gp)[Hickok_ROI_epoch_sort_idx[col_idx_i]], :] = create_gradient(col, len(Hickok_roi_gp)+1)[col_idx_i]
            cols_lst = cols[list(Hickok_roi_gp)].tolist()
            pick_labels = list(data_LexDelay_Resp.labels[0][list(Hickok_roi_gp)])
            plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'brain.tif'), 0.3, hemi='both')

    # Count subj electrods in each Hickok region
    Spt_subj_elec = data_LexDelay_Aud.labels[0][list(Lex_idxes['Hikock_Spt'])]
    lPMC_subj_elec = data_LexDelay_Aud.labels[0][list(Lex_idxes['Hikock_lPMC'])]
    lIFG_subj_elec = data_LexDelay_Aud.labels[0][list(Lex_idxes['Hikock_lIFG'])]


    # Percentages of Aud, Mt, SM, and delay electrodes
    def my_autopct(pct):
        val = int(pct * total / 100)
        # return f'{val}({pct:.1f}%)'
        if pct > 5:
            return f'{pct:.1f}%'
        else:
            return None

    for TypeLabel, sig in zip(
            ('Spt', 'lPMC', 'lIPL', 'lIFG'),
            (Spt_sig_idx,
             lPMC_sig_idx,
             lIPL_sig_idx,
             lIFG_sig_idx)
    ):

        # Sorts of electrodes
        plt.figure()
        DLREP_DEL_inDLREP = np.array(
            [len(sig & Lex_idxes['LexDelay_Aud_NoMotor_sig_idx']),
             len(sig & Lex_idxes['LexDelay_Sensorimotor_sig_idx']),
             len(sig & Lex_idxes['LexDelay_Motor_sig_idx']),
             len(sig & Lex_idxes['LexDelay_DelayOnly_sig_idx']),
             len(sig - (
                         Lex_idxes['LexDelay_Aud_NoMotor_sig_idx'] | Lex_idxes['LexDelay_Sensorimotor_sig_idx'] |
                         Lex_idxes['LexDelay_Motor_sig_idx'] | Lex_idxes['LexDelay_DelayOnly_sig_idx']))
             ])
        DLREP_DEL_inDLREP_labels = [f"Auditory", "Sensory-motor", "Motor", "Delay Only", "Others"]
        total = len(sig)

        DLREP_DEL_inDLREP_colors = [Auditory_col, Sensorimotor_col, Motor_col, Delay_col, [0.5, 0.5, 0.5]]
        plt.pie(DLREP_DEL_inDLREP, labels=None, colors=DLREP_DEL_inDLREP_colors, startangle=90,
                autopct=my_autopct, textprops={'fontsize': 15})
        plt.rcParams.update({'font.size': 14})  # Base font size
        plt.show()

        # Delay versus nodelay
        plt.figure()
        DLREP_DEL_inDLREP = np.array(
            [len(sig & LexDelay_Delay_sig_idx),
             len(sig - LexDelay_Delay_sig_idx)
             ])
        DLREP_DEL_inDLREP_labels = [f"Delay", "Not Delay"]
        total = len(sig)

        DLREP_DEL_inDLREP_colors = [Delay_col, [0.5, 0.5, 0.5]]
        plt.pie(DLREP_DEL_inDLREP, labels=None, colors=DLREP_DEL_inDLREP_colors, startangle=90,
                autopct=my_autopct, textprops={'fontsize': 15})
        plt.rcParams.update({'font.size': 14})  # Base font size
        plt.show()


    ## Plot electrodes categorized in Aud, Mtr, and SM
    hickok_roi_all = pd.DataFrame()
    # Location plot for different types of electrodes
    for TypeLabel,chs_ov,pick_sig_idx,atlas_hist_ylim in zip(
            ('Sensory-motor','Auditory','Delay','Delay_overlapped','Delay_only','MotorPrep_only','Motor','Hickok_ROI_SM','Hickok_ROI_Delay'),
            ([0,1000,0,0,0],[0,0,100,0,0],[0,0,0,10,0],[10000,1000,100,10,1],[10000,1000,100,10,1],[10000,0,0,0,0],[0,0,0,0,1],[0,1000,100,0,1],[10000,1000,100,10,1],[10000,1000,100,10,1]),
            (set2arr(LexDelay_Sensorimotor_sig_idx & LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Aud_NoMotor_sig_idx & LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_DelayOnly_sig_idx,len_d),
             set2arr(LexDelay_Motorprep_Only_sig_idx,len_d),
             set2arr(LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Sensory_OR_Motor_sig_idx & (hickok_roi_sig_idx['Spt'] | hickok_roi_sig_idx['lPMC'] | hickok_roi_sig_idx['lIPL'] | hickok_roi_sig_idx['lIFG']),len_d),
             set2arr(LexDelay_Delay_sig_idx & (hickok_roi_sig_idx['Spt'] | hickok_roi_sig_idx['lPMC'] | hickok_roi_sig_idx['lIPL'] | hickok_roi_sig_idx['lIFG']),len_d)),
            ([0,200], [0,200], [0,250], [0,250], [0,30], [0,5],[0,200],
             [0,200], [0,40], [0,40])
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
    DLREP_DEL_inDLREP = np.array([len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Aud_NoMotor_sig_idx']),
                                  len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Sensorimotor_sig_idx']),
                                  len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_Motor_sig_idx']),
                                  len(Lex_idxes['LexDelay_Delay_sig_idx'] & Lex_idxes['LexDelay_DelayOnly_sig_idx']),
                                  len(Lex_idxes['LexDelay_Delay_sig_idx'] - (Lex_idxes['LexDelay_Aud_NoMotor_sig_idx']| Lex_idxes['LexDelay_Sensorimotor_sig_idx'] | Lex_idxes['LexDelay_Motor_sig_idx'] | Lex_idxes['LexDelay_DelayOnly_sig_idx']))
                                  ])

    DLREP_DEL_inDLREP_labels = [f"Auditory","Sensory-motor","Motor","Delay Only","Others"]
    total = len(Lex_idxes['LexDelay_Delay_sig_idx'])
    def my_autopct(pct):
        val = int(pct * total / 100)
        # return f'{val}({pct:.1f}%)'
        return f'{pct:.1f}%'

    DLREP_DEL_inDLREP_colors = [Auditory_col, Sensorimotor_col, Motor_col, Delay_col,[0.5,0.5,0.5]]
    plt.pie(DLREP_DEL_inDLREP, labels=DLREP_DEL_inDLREP_labels, colors=DLREP_DEL_inDLREP_colors, startangle=90,
            autopct=my_autopct,textprops={'fontsize': 15})
    plt.rcParams.update({'font.size': 14})  # Base font size
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
    Ndel_encode = Lex_idxes['LexNoDelay_Aud_sig_idx']
    Ndel_response = Lex_idxes['LexNoDelay_Motor_Resp_sig_idx']

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
            (del_del,),#del_sm,del_delol),
            ('DEL_DEL',)):#'SM_DEL','DELOL_DEL')):

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

        # Brain plot for electrodes in Delay (separated by functions in No Delay)
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

        # Brain plot for electrodes in Delay (separated by functions in No Delay)
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

        # Brain plot for Aud and SM electrodes in
        len_d=len(data_LexDelay_Aud.labels[0])
        TypeLabel=f'{D_tag}_Delay_Rep in NoDelay Rep'
        # Delay & whether they are still Encoding electrodes in NoDelay
        cols = np.full((len_d, 3), 0.5)
        cols[list(D_sig & (del_aud | del_sm) & Ndel_encode),:] = Auditory_col
        cols[list(D_sig & (del_aud | del_sm) - Ndel_encode),:] = [0.5,0.5,0.5]
        cols_lst=cols[list(D_sig & (del_aud | del_sm))].tolist()
        pick_labels=list(data_LexDelay_Aud.labels[0][list(D_sig & (del_aud | del_sm))])
        plot_brain(subjs, pick_labels,cols_lst,None,os.path.join(fig_save_dir,f'{TypeLabel}_brain.tif'),0.3,0.2)
        # Delay & whether they are still Response electrodes in NoDelay
        cols = np.full((len_d, 3), 0.5)
        cols[list(D_sig & (del_mtr | del_sm) & Ndel_response),:] = Motor_col
        cols[list(D_sig & (del_mtr | del_sm) - Ndel_response),:] = [0.5,0.5,0.5]
        cols_lst=cols[list(D_sig & (del_mtr | del_sm))].tolist()
        pick_labels=list(data_LexDelay_Aud.labels[0][list(D_sig & (del_aud | del_sm))])
        plot_brain(subjs, pick_labels,cols_lst,None,os.path.join(fig_save_dir,f'{TypeLabel}_brain.tif'),0.3,0.2)

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