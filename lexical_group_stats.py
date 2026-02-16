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
WGW_p55b_col=[0.74901961, 0.25098039, 0.74901961] # WGW 55b
WGW_a55b_col=[0, 0.5, 0.5] # WGW a55b
Sensorimotor_Delay_col = Sensorimotor_col#[1, 0, 1]  # Sensorimotor-Delay
Auditory_Delay_col = Auditory_col#[1, 1, 0]  # Auditory-Delay
Delay_Motor_col = Motor_col#[0, 1, 1]  # Delay-Motor

# %% Sort data and get significant electrode lists
import os
import pickle
import numpy as np
import pandas as pd
from utils.group import generate_neuro_publication_plot,get_roi_subj_matrix,get_subj_elec_idx, load_stats, sort_chs_by_actonset, plot_chs, plot_brain, plot_wave,set2arr, chs2atlas, atlas2_hist, plot_sig_roi_counts, get_sig_elecs_keyword, get_coor, hickok_roi_sphere, get_sig_roi_counts, plot_roi_counts_comparison, sort_chs_by_actonset_combined, select_electrodes,onsets2col,elegroup_strip, create_gradient
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
    # data_LexDelay_sorted_preonset,_,_,LexDelay_sig_idx_preonset,*_ = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,pre_stimonset_win)

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
    Lex_idxes['LexDelay_Auditory_not_in_Delay_sig_idx']=LexDelay_Aud_NoMotor_sig_idx-LexDelay_Auditory_in_Delay_sig_idx
    Lex_idxes['LexDelay_all_sig_idx']=LexDelay_all_sig_idx
    Lex_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx']=LexDelay_Sensorimotor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Sensorimotor_not_in_Delay_sig_idx']=LexDelay_Sensorimotor_sig_idx-LexDelay_Sensorimotor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Motor_in_Delay_sig_idx']=LexDelay_Motor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Motor_not_in_Delay_sig_idx']=LexDelay_Motor_sig_idx-LexDelay_Motor_in_Delay_sig_idx
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
Wgw_p55b_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['Wgw_p55b']
Wgw_a55b_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['Wgw_a55b']

Lex_idxes['Hikock_Spt']=Spt_sig_idx
Lex_idxes['Hikock_lPMC']=lPMC_sig_idx
Lex_idxes['Hikock_lIPL']=lIPL_sig_idx
Lex_idxes['Hikock_lIFG']=lIFG_sig_idx
Lex_idxes['Wgw_p55b']=Wgw_p55b_sig_idx
Lex_idxes['Wgw_a55b']=Wgw_a55b_sig_idx

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

    #Plot the distribution of different Delay electrodes within one brain
    TypeLabel = f'wholebrain'
    cols = np.full((len_d, 3), 0.5)
    cols[list(LexDelay_Auditory_in_Delay_sig_idx), :] = Auditory_col
    cols[list(LexDelay_Sensorimotor_in_Delay_sig_idx), :] = Sensorimotor_col
    cols[list(LexDelay_DelayOnly_sig_idx), :] = Delay_col
    cols[list(LexDelay_Motor_in_Delay_sig_idx), :] = Motor_col
    cols_lst = cols[list(LexDelay_Auditory_in_Delay_sig_idx | LexDelay_Sensorimotor_in_Delay_sig_idx | LexDelay_DelayOnly_sig_idx | LexDelay_Motor_in_Delay_sig_idx)].tolist()
    pick_labels = list(data_LexDelay_Aud.labels[0][list(LexDelay_Auditory_in_Delay_sig_idx | LexDelay_Sensorimotor_in_Delay_sig_idx | LexDelay_DelayOnly_sig_idx | LexDelay_Motor_in_Delay_sig_idx)])
    plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'{TypeLabel}_brain.tif'), 0.3, 0.2)

    # Plot the distribution of different Delay electrodes in a pie chart
    counts = [
        len(LexDelay_Auditory_in_Delay_sig_idx), 
        len(LexDelay_Sensorimotor_in_Delay_sig_idx),
        len(LexDelay_Motor_in_Delay_sig_idx), 
        len(LexDelay_DelayOnly_sig_idx),
        len(LexDelay_Delay_sig_idx - (
            LexDelay_Auditory_in_Delay_sig_idx | 
            LexDelay_Sensorimotor_in_Delay_sig_idx | 
            LexDelay_DelayOnly_sig_idx | 
            LexDelay_Motor_in_Delay_sig_idx
        ))
    ]
    DLREP_SM_inDLREP = np.array(counts)

    labels = ["Auditory vWM", "Sensory-motor vWM", "Motor vWM", "Delay-only", "Others"]
    colors = [Auditory_col, Sensorimotor_col, Motor_col, Delay_col, [0.5, 0.5, 0.5]]

    for l, c in zip(labels, DLREP_SM_inDLREP):
        print(f"{l}: {c}")

    plot_labels = [l.replace(" ", "\n") if l != "Others" else "" for l in labels]

    plt.figure(figsize=(8, 8))

    patches, texts, autotexts = plt.pie(
        DLREP_SM_inDLREP, 
        labels=plot_labels, 
        colors=colors, 
        startangle=90, 
        autopct='%1.2f%%',
        pctdistance=0.5,
        labeldistance=1.05,
        wedgeprops={'alpha': 0.75, 'edgecolor': 'w', 'linewidth': 1}
    )

    for l, a in zip(labels, autotexts):
        if l == "Others":
            a.set_text("")

    plt.show()


    # Plot the distribution of different Delay electrodes in a surf plot
    import nibabel as nib
    from scipy import stats
    from nilearn import plotting, datasets, surface
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap 
    from IPython.display import display as ipy_display

    # --- 核心修改：调整颜色节点的辅助函数 ---
    def make_narrow_custom_cmap(rgb_list, name="narrow_custom"):
        """
        通过设置节点，使前 95% 的数值映射为底色（白色），
        仅在最后 5% 显示目标颜色的渐变。
        """
        # 颜色序列：底色 (白色), 底色 (白色), 目标颜色 (RGB)
        colors = [[1, 1, 1], [1, 1, 1], rgb_list]
        # 对应位置：0 (起点), 0.95 (渐变起点), 1.0 (终点)
        nodes = [0.0, 0.05, 1.0]
        return LinearSegmentedColormap.from_list(name, list(zip(nodes, colors)), N=256)

    def get_density_niimg(chs_coor, target_indices, bandwidth=8.0):
        """计算 3D 空间概率密度并生成 Nifti 对象"""
        target_df = chs_coor.iloc[list(target_indices)]
        target_coords = target_df[['x', 'y', 'z']].values
        
        if len(target_coords) < 3:
            return None

        res = 2 
        x_range, y_range, z_range = slice(-70, 71, res), slice(-100, 71, res), slice(-60, 81, res)
        x_grid, y_grid, z_grid = np.mgrid[x_range, y_range, z_range]
        grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
        
        kernel = stats.gaussian_kde(target_coords.T)
        kernel.set_bandwidth(bw_method=bandwidth / np.std(target_coords))
        density = kernel(grid_coords).reshape(x_grid.shape)
        
        affine = np.eye(4)
        affine[0, 0], affine[1, 1], affine[2, 2] = res, res, res
        affine[:3, 3] = [-70, -100, -60] 
        return nib.Nifti1Image(density, affine)

    # --- 交互式吹胀皮层视图逻辑 ---

    # 1. 准备数据组 (使用你定义的变量)
    groups_to_plot = ['Auditory', 'Sensorimotor', 'Motor', 'DelayOnly']
    groups = {
        'Auditory': {'idx': LexDelay_Auditory_in_Delay_sig_idx, 'rgb': Auditory_col},
        'Sensorimotor': {'idx': LexDelay_Sensorimotor_in_Delay_sig_idx, 'rgb': Sensorimotor_col},
        'Motor': {'idx': LexDelay_Motor_in_Delay_sig_idx, 'rgb': Motor_col},
        'DelayOnly': {'idx': LexDelay_DelayOnly_sig_idx, 'rgb': Delay_col}
    }

    # 2. 循环生成交互式视图
    for name in groups_to_plot:
        cfg = groups[name]
        ni_img = get_density_niimg(chs_coor, cfg['idx'], bandwidth=8.0)
        
        if ni_img:
            print(f"Generating Narrow-Range Inflated View: {name}")
            
            # 使用量程压缩后的自定义 Colormap
            custom_cmap = make_narrow_custom_cmap(cfg['rgb'], name=name)
            
            # 动态获取最大值以对齐色带
            data = ni_img.get_fdata()
            vmax_val = np.nanmax(data)
            
            view = plotting.view_img_on_surf(
                ni_img, 
                threshold='95%',       # 物理截断：过滤掉 95% 以下的低密度背景
                vmax=vmax_val,         # 锁定上限
                surf_mesh='fsaverage', 
                vol_to_surf_kwargs={'n_samples': 15, 'radius': 1.5},
                cmap=custom_cmap,      
                symmetric_cmap=False,
                #title=f"Density (Top 5% Range): {name}",
                colorbar=False          # 建议开启以直观查看压缩效果
            )
            
            ipy_display(view)

    #Plot the overlapping []+Delay and [] without Delay electrodes
    # Delay & whether they are still Encoding electrodes in NoDelay
    for elec_idx,elec_col in zip((LexDelay_Aud_NoMotor_sig_idx,LexDelay_Sensorimotor_sig_idx,LexDelay_Motor_sig_idx,LexDelay_DelayOnly_sig_idx),
                                 (Auditory_col,Sensorimotor_col,Motor_col,Delay_col)):
        mode='with_delay' #'all','with delay', 'without_delay'
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
        plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'brain1.png'), 0.3,0.2,hemi='lh')
        plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'brain2.png'), 0.3,0.2,hemi='rh')

    # Reassigning electrode indices by conditions for plotting (ROIS)
    import collections
    # Initialize the master dictionary
    elec_roi = {}
    # Iterate through Subgroups with new names
    for roi_idx, roi_idx_tag in zip(
            (LexDelay_Delay_sig_idx, LexDelay_all_sig_idx - LexDelay_Delay_sig_idx),
            ('vWM', 'no_vWM',)): # Renamed from 'Delay' and 'Without_Delay'

        for TypeLabel, sig in zip(
                ('Auditory', 'Delay_only', 'Sensory-motor', 'Motor'), # TypeLabel also updated for consistency
                (LexDelay_Aud_NoMotor_sig_idx & roi_idx,
                LexDelay_DelayOnly_sig_idx & roi_idx,
                LexDelay_Sensorimotor_sig_idx & roi_idx,
                LexDelay_Motor_sig_idx & roi_idx)):
            
            # Skip the logically empty group
            if roi_idx_tag == 'no_vWM' and TypeLabel == 'vWM': 
                continue
            
            # Get electrode labels for the current selection
            chs_sel = data_LexDelay_Aud.labels[0][list(sig)].tolist()
            
            if TypeLabel not in elec_roi:
                elec_roi[TypeLabel] = {}
            
            # 1. Count ALL anatomical ROIs for this subgroup
            temp_counts = collections.Counter([ch_labels_roi.get(ch, 'unknown') for ch in chs_sel])
            
            # 2. Identify the local Top 4
            top4_rois = [r for r, count in temp_counts.most_common(4)]
            
            # 3. Save the counts only for the Top 4 (will be unified during plotting)
            elec_roi[TypeLabel][roi_idx_tag] = {roi: temp_counts[roi] for roi in top4_rois}

    generate_neuro_publication_plot(elec_roi)
    plt.show()

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
        # roi_idx = LexDelay_Delay_sig_idx
        # roi_idx_tag = 'Delay'

        # --- 1. 缩小全局字体以适应更小的画布 ---
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['axes.linewidth'] = 0.6
        plt.rcParams['font.size'] = 9  # 全局基础字号调小

        wav_bsl_corr_val = True
        go_resp_bsl = range(631, 650)

        # 时间跨度比例
        w1, w2 = 1.75, 1.25

        # --- 2. 缩小画布尺寸 (例如改为 6x5 英寸) ---
        fig = plt.figure(figsize=(6, 5), dpi=300)

        # 设置 GridSpec
        # 我们需要让第一行图的绘图区宽度与时间跨度一致
        # 这里通过 width_ratios 稍微微调，确保 ax1 宽度 = ax2 + ax3 的一半以上
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        # --- Row 1: Stimulus Aligned ---
        # legend 占位稍微缩小到 0.35
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :], width_ratios=[w1, 0.45])
        ax1 = fig.add_subplot(gs_top[0])

        # 绘图逻辑保持不变
        plot_wave(epoc_LexDelay_Aud, LexDelay_Aud_NoMotor_sig_idx & roi_idx, f'Auditory vWM n={len(LexDelay_Aud_NoMotor_sig_idx & roi_idx)}',
                Auditory_col, '-', wav_bsl_corr_val, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_DelayOnly_sig_idx & roi_idx, f'Delay Only n={len(LexDelay_DelayOnly_sig_idx & roi_idx)}',
                Delay_col, '-', wav_bsl_corr_val, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Sensorimotor_sig_idx & roi_idx, f'Sensorymotor vWM n={len(LexDelay_Sensorimotor_sig_idx & roi_idx)}',
                Sensorimotor_col, '-', wav_bsl_corr_val, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Aud, LexDelay_Motor_sig_idx & roi_idx, f'Motor vWM n={len(LexDelay_Motor_sig_idx & roi_idx)}',
                Motor_col, '-', wav_bsl_corr_val, ylim=[-0.2, 1.5])

        ax1.set_xlim([-0.25, 1.5])
        # 缩小图例字号
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)

        # --- Row 2 Left & Right ---
        ax2 = fig.add_subplot(gs[1, 0])
        plot_wave(epoc_LexDelay_Go, LexDelay_Aud_NoMotor_sig_idx & roi_idx, '', Auditory_col, '-', False, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Go, LexDelay_DelayOnly_sig_idx & roi_idx, '', Delay_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Go, LexDelay_Sensorimotor_sig_idx & roi_idx, '', Sensorimotor_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Go, LexDelay_Motor_sig_idx & roi_idx, '', Motor_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])
        ax2.set_xlim([-0.25, 1.0])

        ax3 = fig.add_subplot(gs[1, 1])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Aud_NoMotor_sig_idx & roi_idx, '', Auditory_col, '-', False, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_DelayOnly_sig_idx & roi_idx, '', Delay_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Sensorimotor_sig_idx & roi_idx, '', Sensorimotor_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])
        plot_wave(epoc_LexDelay_Resp, LexDelay_Motor_sig_idx & roi_idx, '', Motor_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])
        ax3.set_xlim([-0.25, 1.0])

        # --- 修饰细节 ---
        for i, ax in enumerate([ax1, ax2, ax3]):
            ax.spines[['top', 'right']].set_visible(False)
            # --- 关键：缩小 Offset 距离 ---
            ax.spines['left'].set_position(('outward', 5)) 
            ax.spines['bottom'].set_position(('outward', 5))
            
            ax.axvline(x=0, linestyle='--', color='#444444', linewidth=0.6, dashes=(5, 5), zorder=0)
            ax.axhline(y=0, linestyle='-', color='#DDDDDD', linewidth=0.5, zorder=0)
            
            # 缩小刻度大小
            ax.tick_params(axis='both', which='major', labelsize=8, direction='out', length=3, pad=2)
            ax.set_ylim([-0.2, 1.5])
            
            if i > 0:
                ax.set_xlabel('Time (s)', fontsize=9, labelpad=2)
            
            if ax == ax3:
                ax.set_ylabel('')
                ax.set_yticklabels([])
                ax.spines['left'].set_visible(False)
                ax.tick_params(axis='y', left=False)
            else:
                ax.set_ylabel('High-gamma (z-score)', fontsize=9, labelpad=2)

        # 自动对齐 Y 轴标签（防止左右错位）
        fig.align_ylabels([ax1, ax2])

        # --- 3. 使用 tight_layout 并手动微调边距 ---
        plt.tight_layout()
        # 调整 top/bottom 比例，为顶部的 title 或底部的 label 留出固定空间
        plt.subplots_adjust(hspace=0.4, wspace=0.2, right=0.85) 

        plt.show()
        # plt.savefig(os.path.join(fig_save_dir, f'LexDelay_sig_zscore_org_cat_Resp_{roi_idx_tag}.tif'), dpi=300)
        # plt.close()

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
    cols[list(Wgw_p55b_sig_idx),:] = WGW_p55b_col
    cols[list(Wgw_a55b_sig_idx),:] = WGW_a55b_col
    cols_lst = cols[list(Spt_sig_idx | lPMC_sig_idx | lIPL_sig_idx | lIFG_sig_idx | Wgw_p55b_sig_idx | Wgw_a55b_sig_idx)].tolist()
    pick_labels = list(data_LexDelay_Aud.labels[0][list(Spt_sig_idx | lPMC_sig_idx | lIPL_sig_idx | lIFG_sig_idx | Wgw_p55b_sig_idx | Wgw_a55b_sig_idx)])
    plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'{TypeLabel}_brain.tif'), 0.3, 0.2)

    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    # plt.title('High gamma z-score traces (Stim aligned)', fontsize=20)
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
    # plt.xlabel('(Time in secs')
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

    # Do for each patient
    # s_es=get_subj_elec_idx(subjs, data_LexDelay_Resp.labels[0].tolist())
    # for s,s_e in s_es.items():
    #     for Hickok_roi_gp, col, tag in zip(
    #             (Spt_sig_idx & s_e, lPMC_sig_idx & s_e, lIFG_sig_idx & s_e),
    #             (Auditory_col, Sensorimotor_col, Motor_col),
    #             ('Spt', 'lPMC', 'lIFG')
    #     ):
    def Hickok_aud_resp_corr(x, y, fig_save_dir, tag, col, 
                        xlabel='Onset response duration (s)', 
                        ylabel='Self speech response magnitude (z-score)',
                        fname = 'Hickok_aud_resp_corr'):
        from scipy.stats import pearsonr
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) > 2:  # pearsonr 需要至少2个数据点
            corr_coefficient, p_value = pearsonr(x_valid, y_valid)
            df = len(x_valid) - 2  # 自由度
            print(tag)
            print(f"Pearson correlation: r({df}) = {corr_coefficient:.4f}, p = {p_value:.4f}")
            
            plt.figure(figsize=(8, 6))
            # 使用传入的 col 参数为散点图上色
            plt.scatter(x_valid, y_valid, alpha=0.6, label='Data points', color=col)
            
            m, b = np.polyfit(x_valid, y_valid, 1)
            plt.plot(x_valid, m * x_valid + b, color='red', label='Line of best fit')
            
            plt.title(f'Correlation for {tag}', fontsize=20)
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            
            ax = plt.gca()
            ax.spines[['top', 'right']].set_visible(False)
            
            plt.savefig(os.path.join(fig_save_dir, f'{fname}_{tag}.jpg'), dpi=300)
            plt.close()

    ele_codes = []
    Waveplot_wth=5 # Width of wave plots
    Waveplot_hgt=4 # Height of wave plots
    for Hickok_roi_gp, col, tag in zip(
            (Spt_sig_idx, lPMC_sig_idx, lIFG_sig_idx,Wgw_p55b_sig_idx,Wgw_a55b_sig_idx),
            (Auditory_col, Sensorimotor_col, Motor_col,WGW_p55b_col,WGW_a55b_col),
            ('Spt', 'lPMC (dPCSA)', 'lIFG (vPCSA)','posterior 55b','anterior 55b')
    ):

        if Hickok_roi_gp:

            cluster_paras={}
            cluster_paras_Spt={}
            cluster_paras_lPMC={}
            cluster_paras_lIFG={}


            for data_epoch,epoc_epoch,wav_fig_size,wav_x_lim,epoch_tag in zip(
                    (data_LexDelay_Resp,data_LexDelay_Aud,data_LexDelay_Go,data_LexDelay_Cue),
                    (epoc_LexDelay_Resp,epoc_LexDelay_Aud,epoc_LexDelay_Go,epoc_LexDelay_Cue),
                    ((Waveplot_wth, Waveplot_hgt),(Waveplot_wth, Waveplot_hgt),(Waveplot_wth, Waveplot_hgt),(Waveplot_wth, Waveplot_hgt)),
                    ([-0.5, 1.5],[-0.5, 1.5],[-0.5, 1.5],[-0.5, 1.5]),
                    ('Resp','Stim','Go','Cue')
            ):

                # for testing:
                # Hickok_roi_gp=lPMC_sig_idx
                # col=Auditory_col
                # tag='lPMC (dPCSA)'
                # data_epoch=data_LexDelay_Resp
                # epoc_epoch=epoc_LexDelay_Resp
                # wav_fig_size=(Waveplot_wth, Waveplot_hgt)
                # wav_x_lim=[-0.5, 1.5]
                # epoch_tag='Resp'

                # Clus plots
                Hickok_ROI_data = select_electrodes(data_epoch, Hickok_roi_gp)
                Hickok_ROI_epoch = select_electrodes(epoc_epoch, Hickok_roi_gp)
                if epoch_tag == 'Resp':
                    _, _, Hickok_ROI_epoch_sort_idx, _,_,Hickok_ROI_epoch_sort_paras_tab = sort_chs_by_actonset(Hickok_ROI_data,
                                                                               Hickok_ROI_epoch,
                                                                               cluster_twin, [-1,wav_x_lim[1]],
                                                                               mask_data=True,
                                                                               select_electrodes=False)
                    cluster_paras[epoch_tag] = Hickok_ROI_epoch_sort_paras_tab

                    Hickok_ROI_epoch_sort, *_ = sort_chs_by_actonset(Hickok_ROI_data,
                                                                     Hickok_ROI_epoch,
                                                                     cluster_twin, wav_x_lim,
                                                                     sorted_indices=Hickok_ROI_epoch_sort_idx,
                                                                     mask_data=True,
                                                                     select_electrodes=False)
                else:
                    if epoch_tag == 'Stim':
                        # Get parameters from Auditory responses
                        _, _, _, _,_,Hickok_ROI_epoch_sort_paras_tab = sort_chs_by_actonset(Hickok_ROI_data,
                                                                        Hickok_ROI_epoch,
                                                                        cluster_twin, [0,1],
                                                                        sorted_indices=Hickok_ROI_epoch_sort_idx,
                                                                        mask_data=True,
                                                                        select_electrodes=False)            
                        cluster_paras[epoch_tag] = Hickok_ROI_epoch_sort_paras_tab

                    Hickok_ROI_epoch_sort, *_ = sort_chs_by_actonset(Hickok_ROI_data,
                                                                     Hickok_ROI_epoch,
                                                                     cluster_twin, wav_x_lim,
                                                                     sorted_indices=Hickok_ROI_epoch_sort_idx,
                                                                     mask_data=True,
                                                                     select_electrodes=False)
                Hickok_ROI_data_sort = select_electrodes(Hickok_ROI_data, Hickok_ROI_epoch_sort_idx)
                plot_chs(Hickok_ROI_epoch_sort, os.path.join(fig_save_dir,
                                                             f'Hickok_sig_alg_resp_{tag}_{epoch_tag}.jpg'),
                         f'{tag}', percentage_vscale=False, vmin=0, vmax=2, is_colbar=False,
                         fig_size=[4, 20 * (len(Hickok_roi_gp) / 250)])


            # Correlation between auditory response duration and motor response magnitude

                # wave
                # Hickok_ROI_epoch_sort_unmask,*_ = sort_chs_by_actonset(Hickok_ROI_data,
                #                                                        Hickok_ROI_epoch,
                #                                                        cluster_twin, wav_x_lim,
                #                                                        sorted_indices=Hickok_ROI_epoch_sort_idx,
                #                                                        mask_data=False,
                #                                                        select_electrodes=False)
                plt.figure(figsize=wav_fig_size)
                wav_bsl_corr = False
                plot_wave(Hickok_ROI_epoch, Hickok_ROI_epoch_sort_idx,f'',col, '-', wav_bsl_corr, ylim=[-0.4, 5.5],average_trace=False)
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

                # For Spt only: wave plots for different categories of electrodes
                if tag == 'Spt':
                    # Hickok's manual categorizations
                    # hickok_sub_idx_aud_mtr=np.array([3,4,6,10,11,12,13,14,15])-1
                    # hickok_sub_idx_aud_onset = np.array([20,21,22,23,25,26,27])-1
                    # hickok_sub_idx_aud_contin = np.array([19,28,29,31,32,33,34])-1
                    
                    # New categorizations based on NMF
                    hickok_sub_idx_aud_mtr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 23, 29, 34])
                    hickok_sub_idx_aud_contin = np.array([13, 14, 18, 28, 31, 32, 33])
                    hickok_sub_idx_aud_onset = np.array([19, 20, 21, 22, 24, 25, 26, 27, 30])

                    for hickok_sub_idx,hickok_sub_idx_tag,hickok_sub_idx_col,wave_y_lim in zip(
                            (hickok_sub_idx_aud_mtr,hickok_sub_idx_aud_onset,hickok_sub_idx_aud_contin),
                            ('auditory motor','auditory onset','auditory continuous'),
                            ([0,1,0],[1,165/255,0],[1,0,0]),
                            ([-0.4,3],[-0.4,4],[-0.4,7])):
                        
                        Hickok_ROI_data_sort_sub = select_electrodes(Hickok_ROI_data_sort, hickok_sub_idx)
                        Hickok_ROI_epoch_sort_sub = select_electrodes(Hickok_ROI_epoch_sort, hickok_sub_idx)

                        # save manually coded electrode types
                        ele={'Group':tag,'manual_tag':hickok_sub_idx_tag,'chs':Hickok_ROI_data_sort_sub.labels[0].tolist()}
                        ele_code=pd.DataFrame(data=ele)
                        ele_codes.append(ele_code)

                        plot_chs(Hickok_ROI_epoch_sort_sub, os.path.join(fig_save_dir,
                                                                     f'Hickok_sig_alg_resp_{tag}_{epoch_tag}_{hickok_sub_idx_tag}.jpg'),
                                 f'{tag} {hickok_sub_idx_tag}', percentage_vscale=False, vmin=0, vmax=2, is_colbar=False,
                                 fig_size=[4, 20 * (len(Hickok_ROI_epoch_sort_sub) / 250)])

                        plt.figure(figsize=wav_fig_size)
                        wav_bsl_corr = False
                        plot_wave(Hickok_ROI_epoch, Hickok_ROI_epoch_sort_idx[hickok_sub_idx], f'', hickok_sub_idx_col, '-', wav_bsl_corr,
                                  ylim=wave_y_lim, average_trace=False)
                        plt.axvline(x=0, linestyle='--', color='k')
                        plt.axhline(y=0, linestyle='--', color='gray')
                        plt.title(f'{tag} {hickok_sub_idx_tag}', fontsize=20)
                        plt.tick_params(axis='both', labelsize=16)
                        plt.xticks(rotation=45)
                        plt.xlim(wav_x_lim)
                        plt.gca().spines[['top', 'right']].set_visible(False)
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_save_dir, f'Hickok_wave_alg_resp_{tag}_{epoch_tag}_{hickok_sub_idx_tag}.tif'), dpi=300)
                        plt.close()

                        # Duration-resp correlation:
                        if epoch_tag == 'Stim':
                            Spt_wav_x_lim=[0,1]
                        elif epoch_tag == 'Resp':
                            Spt_wav_x_lim=[-1,1.5]
                        
                        if epoch_tag == 'Resp' or epoch_tag == 'Stim':
                            _, _, _, _,_,Hickok_ROI_epoch_sort_paras_tab = sort_chs_by_actonset(Hickok_ROI_data_sort_sub,
                                                                Hickok_ROI_epoch_sort_sub,
                                                                cluster_twin, Spt_wav_x_lim,
                                                                mask_data=True,
                                                                sorted_indices=np.arange(len(hickok_sub_idx)),
                                                                select_electrodes=False)
                            
                            if hickok_sub_idx_tag not in cluster_paras_Spt:
                                cluster_paras_Spt[hickok_sub_idx_tag] = {}
                            cluster_paras_Spt[hickok_sub_idx_tag][epoch_tag] = Hickok_ROI_epoch_sort_paras_tab

                    # Brain plots of Spt clusters
                    Hickok_roi_gp_arr = np.array(list(Hickok_roi_gp))
                    Hickok_roi_gp_arr_sorted = Hickok_roi_gp_arr[list(Hickok_ROI_epoch_sort_idx)]
                    Hickok_roi_gp_arr_sorted_aud_mtr = set(Hickok_roi_gp_arr_sorted[list(hickok_sub_idx_aud_mtr)])
                    Hickok_roi_gp_arr_sorted_aud_onset = set(Hickok_roi_gp_arr_sorted[list(hickok_sub_idx_aud_onset)])
                    Hickok_roi_gp_arr_sorted_aud_contin = set(Hickok_roi_gp_arr_sorted[list(hickok_sub_idx_aud_contin)])
                    len_d = len(data_LexDelay_Aud.labels[0])
                    TypeLabel = f'Hikock'
                    cols = np.full((len_d, 3), 0.5)
                    cols[list(Hickok_roi_gp_arr_sorted_aud_mtr), :] = create_gradient([0,1,0], len(Hickok_roi_gp_arr_sorted_aud_mtr) + 1)[:-1]
                    cols[list(Hickok_roi_gp_arr_sorted_aud_onset), :] = create_gradient([1,165/255,0], len(Hickok_roi_gp_arr_sorted_aud_onset) + 1)[:-1]
                    cols[list(Hickok_roi_gp_arr_sorted_aud_contin), :] = create_gradient([1,0,0], len(Hickok_roi_gp_arr_sorted_aud_contin) + 1)[:-1]
                    cols_lst = cols[
                        list(Hickok_roi_gp_arr_sorted_aud_mtr | Hickok_roi_gp_arr_sorted_aud_onset | Hickok_roi_gp_arr_sorted_aud_contin)].tolist()
                    pick_labels = list(data_LexDelay_Aud.labels[0][list(Hickok_roi_gp_arr_sorted_aud_mtr | Hickok_roi_gp_arr_sorted_aud_onset | Hickok_roi_gp_arr_sorted_aud_contin)])
                    plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'{TypeLabel}_brain.tif'),
                               0.3, 0.2)

                elif tag == 'lPMC (dPCSA)':
                    # Hicok's manual categorizations
                    # hickok_sub_idx_aud_onset = np.array([8,11,18,24,26,33,35])-1
                    # hickok_sub_idx_aud_contin = np.array([3,4,7,13,14,15,16,17,20,21,22,28,31])-1
                    # New categorizations based on NMF
                    hickok_sub_idx_aud_onset = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 23, 26, 27, 28, 30, 31, 32, 33, 34])
                    hickok_sub_idx_aud_contin = np.array([6, 7, 8, 12, 13, 14, 15, 16, 17, 21, 22, 24, 25, 29])

                    for hickok_sub_idx,hickok_sub_idx_tag,hickok_sub_idx_col,wave_y_lim in zip(
                            (hickok_sub_idx_aud_onset,hickok_sub_idx_aud_contin),
                            ('auditory onset','auditory continuous'),
                            ([1,165/255,0],[1,0,0]),
                            ([-0.4,4],[-0.4,4])):
                        
                        Hickok_ROI_data_sort_sub = select_electrodes(Hickok_ROI_data_sort, hickok_sub_idx)
                        Hickok_ROI_epoch_sort_sub = select_electrodes(Hickok_ROI_epoch_sort, hickok_sub_idx)

                        # save manually coded electrode types
                        ele={'Group':tag,'manual_tag':hickok_sub_idx_tag,'chs':Hickok_ROI_data_sort_sub.labels[0].tolist()}
                        ele_code=pd.DataFrame(data=ele)
                        ele_codes.append(ele_code)

                        plot_chs(Hickok_ROI_epoch_sort_sub, os.path.join(fig_save_dir,
                                                                     f'Hickok_sig_alg_resp_{tag}_{epoch_tag}_{hickok_sub_idx_tag}.jpg'),
                                 f'{tag} {hickok_sub_idx_tag}', percentage_vscale=False, vmin=0, vmax=2, is_colbar=False,
                                 fig_size=[4, 20 * (len(Hickok_ROI_epoch_sort_sub) / 250)])

                        plt.figure(figsize=wav_fig_size)
                        wav_bsl_corr = False
                        plot_wave(Hickok_ROI_epoch, Hickok_ROI_epoch_sort_idx[hickok_sub_idx], f'', hickok_sub_idx_col, '-', wav_bsl_corr,
                                  ylim=wave_y_lim, average_trace=False)
                        plt.axvline(x=0, linestyle='--', color='k')
                        plt.axhline(y=0, linestyle='--', color='gray')
                        plt.title(f'{tag} {hickok_sub_idx_tag}', fontsize=20)
                        plt.tick_params(axis='both', labelsize=16)
                        plt.xticks(rotation=45)
                        plt.xlim(wav_x_lim)
                        plt.gca().spines[['top', 'right']].set_visible(False)
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_save_dir, f'Hickok_wave_alg_resp_{tag}_{epoch_tag}_{hickok_sub_idx_tag}.tif'), dpi=300)
                        plt.close()

                        # Duration-resp correlation:
                        if epoch_tag == 'Stim':
                            Spt_wav_x_lim=[0,1]
                        elif epoch_tag == 'Resp':
                            Spt_wav_x_lim=[-1,1.5]
                        
                        if epoch_tag == 'Resp' or epoch_tag == 'Stim':
                            _, _, _, _,_,Hickok_ROI_epoch_sort_paras_tab = sort_chs_by_actonset(Hickok_ROI_data_sort_sub,
                                                                Hickok_ROI_epoch_sort_sub,
                                                                cluster_twin, Spt_wav_x_lim,
                                                                mask_data=True,
                                                                sorted_indices=np.arange(len(hickok_sub_idx)),
                                                                select_electrodes=False)
                            
                            if hickok_sub_idx_tag not in cluster_paras_lPMC:
                                cluster_paras_lPMC[hickok_sub_idx_tag] = {}
                            cluster_paras_lPMC[hickok_sub_idx_tag][epoch_tag] = Hickok_ROI_epoch_sort_paras_tab
                
                elif tag == 'lIFG (vPCSA)':
                    # Hicok's manual categorizations
                    # hickok_sub_idx_delay = np.array([5,8,17,18])-1
                    # hickok_sub_idx_articulation = np.array([3,6,13,14])-1
                    # hickok_sub_idx_both = np.array([2,4,9,10,11])-1
                    # New categorizations based on NMF
                    hickok_sub_idx_articulation = np.array([14, 15])
                    hickok_sub_idx_both = np.array([5, 8, 9, 10, 12, 13])
                    hickok_sub_idx_delay = np.array([0, 1, 2, 3, 4, 6, 7, 11, 16, 17])

                    for hickok_sub_idx,hickok_sub_idx_tag,hickok_sub_idx_col,wave_y_lim in zip(
                            (hickok_sub_idx_delay,hickok_sub_idx_articulation,hickok_sub_idx_both),
                            ('delay pre-articulation','articulation','both delay and articulation'),
                            ([191/255,191/255,191/255],[0,0,0],[127/255,127/255,127/255]),
                            ([-0.4,2],[-0.4,2],[-0.4,3])):
                        
                        Hickok_ROI_data_sort_sub = select_electrodes(Hickok_ROI_data_sort, hickok_sub_idx)
                        Hickok_ROI_epoch_sort_sub = select_electrodes(Hickok_ROI_epoch_sort, hickok_sub_idx)

                        # save manually coded electrode types
                        ele={'Group':tag,'manual_tag':hickok_sub_idx_tag,'chs':Hickok_ROI_data_sort_sub.labels[0].tolist()}
                        ele_code=pd.DataFrame(data=ele)
                        ele_codes.append(ele_code)

                        plot_chs(Hickok_ROI_epoch_sort_sub, os.path.join(fig_save_dir,
                                                                     f'Hickok_sig_alg_resp_{tag}_{epoch_tag}_{hickok_sub_idx_tag}.jpg'),
                                 f'{tag} {hickok_sub_idx_tag}', percentage_vscale=False, vmin=0, vmax=2, is_colbar=False,
                                 fig_size=[4, 20 * (len(Hickok_ROI_epoch_sort_sub) / 250)])

                        plt.figure(figsize=wav_fig_size)
                        wav_bsl_corr = False
                        plot_wave(Hickok_ROI_epoch, Hickok_ROI_epoch_sort_idx[hickok_sub_idx], f'', hickok_sub_idx_col, '-', wav_bsl_corr,
                                  ylim=wave_y_lim, average_trace=False)
                        plt.axvline(x=0, linestyle='--', color='k')
                        plt.axhline(y=0, linestyle='--', color='gray')
                        plt.title(f'{tag} {hickok_sub_idx_tag}', fontsize=20)
                        plt.tick_params(axis='both', labelsize=16)
                        plt.xticks(rotation=45)
                        plt.xlim(wav_x_lim)
                        plt.gca().spines[['top', 'right']].set_visible(False)
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_save_dir, f'Hickok_wave_alg_resp_{tag}_{epoch_tag}_{hickok_sub_idx_tag}.tif'), dpi=300)
                        plt.close()

                        # Duration-resp correlation:
                        if epoch_tag == 'Stim':
                            Spt_wav_x_lim=[0,1]
                        elif epoch_tag == 'Resp':
                            Spt_wav_x_lim=[-1,1.5]
                        
                        if epoch_tag == 'Resp' or epoch_tag == 'Stim':
                            _, _, _, _,_,Hickok_ROI_epoch_sort_paras_tab = sort_chs_by_actonset(Hickok_ROI_data_sort_sub,
                                                                Hickok_ROI_epoch_sort_sub,
                                                                cluster_twin, Spt_wav_x_lim,
                                                                mask_data=True,
                                                                sorted_indices=np.arange(len(hickok_sub_idx)),
                                                                select_electrodes=False)
                            
                            if hickok_sub_idx_tag not in cluster_paras_lIFG:
                                cluster_paras_lIFG[hickok_sub_idx_tag] = {}
                            cluster_paras_lIFG[hickok_sub_idx_tag][epoch_tag] = Hickok_ROI_epoch_sort_paras_tab

                # brain plot (sig vs. nonsig before onset)
                cols = np.full((len_d, 3), 0.5)
                for col_idx_i in range(len(Hickok_roi_gp)):
                    cols[list(Hickok_roi_gp)[Hickok_ROI_epoch_sort_idx[col_idx_i]], :] = create_gradient(col, len(Hickok_roi_gp)+1)[col_idx_i]
                cols_lst = cols[list(Hickok_roi_gp)].tolist()
                pick_labels = list(data_LexDelay_Resp.labels[0][list(Hickok_roi_gp)])
                plot_brain(subjs, pick_labels, cols_lst, None, os.path.join(fig_save_dir, f'brain.tif'), 0.3, hemi='both')

            x=cluster_paras['Stim']['activity_length']
            y=cluster_paras['Resp']['rms_value']
            Hickok_aud_resp_corr(x, y, fig_save_dir, tag, col, fname = 'Hickok_aud_resp_corr')

            if tag == 'Spt':
                for hickok_sub_idx_tag in ('auditory motor','auditory onset','auditory continuous'):
                    x=cluster_paras_Spt[hickok_sub_idx_tag]['Stim']['activity_length']
                    y=cluster_paras_Spt[hickok_sub_idx_tag]['Resp']['rms_value']
                    Hickok_aud_resp_corr(x, y, fig_save_dir, f"{tag}_{hickok_sub_idx_tag}", col, fname = f'Hickok_aud_resp_corr_Spt_{hickok_sub_idx_tag}')

    if ele_codes:
        ele_codes_df = pd.concat(ele_codes, ignore_index=True)
        ele_codes_df_uni = ele_codes_df.drop_duplicates()
        ele_codes_df_uni.to_csv(os.path.join('projects','Greg_ROIs', 'Hickok_ROI_electrode_manual_coding.csv'), index=False)

    # Count subj electrods in each Hickok region
    Spt_subj_elec = data_LexDelay_Aud.labels[0][list(Lex_idxes['Hikock_Spt'])]
    lPMC_subj_elec = data_LexDelay_Aud.labels[0][list(Lex_idxes['Hikock_lPMC'])]
    lIFG_subj_elec = data_LexDelay_Aud.labels[0][list(Lex_idxes['Hikock_lIFG'])]

    roi_data = {
        'Spt': Spt_subj_elec,
        'lPMC': lPMC_subj_elec,
        'lIFG': lIFG_subj_elec
    }

    roi_subj_table=get_roi_subj_matrix(roi_data)


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

    DLREP_DEL_inDLREP_labels = [f"Auditory\nvWM","Sensory-motor\nvWM","Motor\nvWM","Delay Only",""]
    total = len(Lex_idxes['LexDelay_Delay_sig_idx'])
    def my_autopct(pct):
        val = int(pct * total / 100)
        # return f'{val}({pct:.1f}%)'
        if pct>5:
            return f'{pct:.1f}%'
        else:
            return f''

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
    # Ndel_S_encode_only=Lex_idxes['LexNoDelay_Silence_Encode_Only_sig_idx']
    # Ndel_S_del=Lex_idxes['LexNoDelay_Silence_Del_sig_idx']
    # Ndel_S_all=(Ndel_S_encode_only | Ndel_S_del)

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

    # Combine Auditory and Motor responses
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

    # Combine Auditory and Motor responses but without delay only
    data = {
        "Auditory": [len(del_del & (del_aud | del_sm) & (Ndel_aud | Ndel_sm))/len(del_del & (del_aud | del_sm))*100,
                          len(del_del & (del_aud | del_sm) & Ndel_mtr)/len(del_del & (del_aud | del_sm))*100,
                          len((del_del & (del_aud | del_sm)).difference(Ndel_all))/len(del_del & (del_aud | del_sm))*100],
        "Motor": [len(del_del & (del_mtr | del_sm) & Ndel_aud)/len(del_del & (del_mtr | del_sm))*100,
                          len(del_del & (del_mtr | del_sm) & (Ndel_mtr | Ndel_sm))/len(del_del & (del_mtr | del_sm))*100,
                          len((del_del & (del_mtr | del_sm)).difference(Ndel_all))/len(del_del & (del_mtr | del_sm))*100],
        "Delay Only": [len(del_delol & (Ndel_aud | Ndel_sm)) / len(del_delol) * 100,
                       len(del_delol & (Ndel_mtr | Ndel_sm)) / len(del_delol) * 100,
                       len(del_delol.difference(Ndel_all)) / len(del_delol) * 100]
    }
    df_cm = pd.DataFrame(data, index=["Auditory", "Motor", "Silent"]).transpose()
    plt.figure(figsize=(5, 5))
    # Adjust font size for posters
    NEW_BASE_FONT_SIZE = 20
    plt.rcParams.update({
        'font.size': NEW_BASE_FONT_SIZE * 1.8,
        'axes.labelsize': NEW_BASE_FONT_SIZE,
        'axes.titlesize': NEW_BASE_FONT_SIZE * 1.2,
        'xtick.labelsize': NEW_BASE_FONT_SIZE * 0.7,
        'ytick.labelsize': NEW_BASE_FONT_SIZE * 0.7,
    })
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="rocket_r", annot_kws={"size": 14}, vmin=0, vmax=80, cbar=False)
    # plt.title(f"% of electrodes in $NoDelay$")
    plt.ylabel("vWM electrodes in $Delay$")
    plt.xlabel("Same electrodes in $NoDelay$")
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