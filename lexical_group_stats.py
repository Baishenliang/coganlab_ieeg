# %% groups of patients
from pickle import FALSE
from matplotlib_venn import venn3
import seaborn as sns   
import matplotlib.ticker as mticker

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
from utils.group import bsliang_add_connecting_lines,generate_neuro_publication_plot,get_roi_subj_matrix,get_subj_elec_idx, load_stats, sort_chs_by_actonset, plot_chs, plot_brain, plot_wave,set2arr, chs2atlas, atlas2_hist, plot_sig_roi_counts, get_sig_elecs_keyword, get_coor, hickok_roi_sphere, get_sig_roi_counts, plot_roi_counts_comparison, sort_chs_by_actonset_combined, select_electrodes,onsets2col,elegroup_strip, create_gradient
import matplotlib.pyplot as plt
import projects.GLM.glm_utils as glm

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

fig_save_dir = os.path.join(LAB_root, 'D_Data','LexicalDecRepDelay','Baishen_Figs','LexicalDecRepDelay','group')
manuscript_save_dir = r"D:\lbs\Little_projects\Greg_LexDelay\materials\figs_elements\Fig2"
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
    #LexDelay_Motor_sig_idx = LexDelay_Motor_Resp_sig_idx - LexDelay_Aud_sig_idx
    LexDelay_Motor_sig_idx = LexDelay_Motor_Prep_sig_idx - LexDelay_Aud_sig_idx

    # Channel selection: Delay only electrodes (delay electrodes ,with: auditory window:0, motor prep: 0, motor resp: 0)
    #LexDelay_DelayOnly_sig_idx = LexDelay_Delay_sig_idx - (LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx)
    LexDelay_DelayOnly_sig_idx = LexDelay_Delay_sig_idx - (LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx)
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
    _,_,_,LexNoDelay_Aud_narraw_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Aud,epoc_LexNoDelay_Aud, cluster_twin,[0,50])

    # (Motor prepare)
    data_LexNoDelay_Motor_Prep_sorted,_,_,LexNoDelay_Motor_Prep_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, motor_prep_win)

    # (Motor response)
    data_LexNoDelay_Motor_Resp_sorted,_,_,LexNoDelay_Motor_Resp_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, motor_resp_win)
    _,_,_,LexNoDelay_Motor_narraw_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, [0,50])

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
    Lex_idxes['LexNoDelay_Motor_narraw_sig_idx'] = LexNoDelay_Motor_narraw_sig_idx
    Lex_idxes['LexNoDelay_Aud_narraw_sig_idx'] = LexNoDelay_Aud_narraw_sig_idx
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

    # --- 1. 数据准备 (保持你的原始定义) ---
    counts = [
        len(LexDelay_Auditory_in_Delay_sig_idx), 
        len(LexDelay_Sensorimotor_in_Delay_sig_idx),
        len(LexDelay_Motor_in_Delay_sig_idx), 
        len(LexDelay_DelayOnly_sig_idx)]#,
        # len(LexDelay_Delay_sig_idx - (
        #     LexDelay_Auditory_in_Delay_sig_idx | 
        #     LexDelay_Sensorimotor_in_Delay_sig_idx | 
        #     LexDelay_DelayOnly_sig_idx | 
        #     LexDelay_Motor_in_Delay_sig_idx
        # ))
    DLREP_SM_inDLREP = np.array(counts)
    labels = ["Auditory vWM", "Sensory-motor vWM", "Motor vWM", "Delay-only"]#, "Others"]
    colors = [Auditory_col, Sensorimotor_col, Motor_col, Delay_col]#, [0.8, 0.8, 0.8]] # 灰色稍调浅一点

    # --- 2. 自定义标签函数：电极数 \n (百分比) ---
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            # 按照你的要求：电极数-换行-（百分比）
            return f'{val}\n({pct:.1f}%)'
        return my_autopct

    # --- 3. 绘图 ---
    plt.figure(figsize=(8, 8), dpi=300)
    # 移除了 labels=plot_labels，改用空列表以删掉图外标签
    patches, texts, autotexts = plt.pie(
        DLREP_SM_inDLREP, 
        labels=None,  # 删掉图外的电极组 labels
        colors=colors, 
        startangle=90, 
        autopct=make_autopct(DLREP_SM_inDLREP),
        pctdistance=0.6,
        wedgeprops={'alpha': 0.8, 'edgecolor': 'w', 'linewidth': 2} # 增加加粗白边
    )

    # --- 4. 细节修饰 (字体与隐藏 Others) ---
    for i, (l, a) in enumerate(zip(labels, autotexts)):
        # 统一使用 48 号大字体，确保排版清晰
        a.set_fontsize(36)
        a.set_fontweight('bold')
        a.set_color('black') # 或者用 'white'，取决于你底色的深浅
        
        # 隐藏 Others 的百分比显示 (如果你想保持图面纯净)
        if l == "Others":
            a.set_text("")

    plt.tight_layout()

    # --- 5. 保存 ---
    save_path = os.path.join(manuscript_save_dir, "..","Fig1","vWM_Electrode_Distribution_Pie.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()


    # Plot the distribution of different Delay electrodes in a surf plot
    import numpy as np
    from scipy.spatial.distance import pdist
    import nibabel as nib
    from scipy import stats
    from nilearn import plotting,image
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    from scipy.spatial.distance import cdist
    from IPython.display import display as ipy_display
    from scipy.stats import skew, kurtosis, entropy
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    import nibabel as nib
    pink_rgb = (1.0, 0.2, 0.6) 
    pink_cmap = LinearSegmentedColormap.from_list("pink_density", [ (0.8, 0.8, 0.8), (1.0, 1.0, 1.0)], N=256)


    def analyze_bna_activation(final_mask, atlas_path):
        """
        Maps a custom density mask to the Brainnetome Atlas (BNA) and calculates 
        activation ratios for specific clusters (SFG, MFG, IFG, PreCG).
        
        Parameters:
        -----------
        final_mask : np.ndarray
            3D array containing the binary mask (1.0 for active voxels, NaN/0 for others).
        atlas_path : str
            Path to the 'BN_Atlas_246_1mm.nii.gz' file.
            
        Returns:
        --------
        pd.DataFrame
            A summary table grouped by Cluster and Hemisphere.
        """
        
        # --- Step 1: Create NIfTI object from numpy array ---
        # Define the Affine matrix based on the grid: 
        # x_range: -70 to 71, y_range: -100 to 71, z_range: -60 to 81 with res=2.
        # The translation parameters (-70, -100, -60) are the starting coordinates.
        affine = np.array([
            [2, 0, 0, -70],
            [0, 2, 0, -100],
            [0, 0, 2, -60],
            [0, 0, 0, 1]
        ])
        
        # Replace NaNs with 0 and ensure float32 type for Nifti compatibility
        mask_clean = np.nan_to_num(final_mask).astype(np.float32)
        mask_img = nib.Nifti1Image(mask_clean, affine)

        # --- Step 2: Resample Atlas to match Mask Space ---
        # Load the high-res atlas (1mm) and resample it to the 2mm mask grid.
        # 'nearest' interpolation is crucial to preserve the integer Label IDs.
        atlas_img = image.load_img(atlas_path)
        atlas_resampled = image.resample_to_img(atlas_img, mask_img, interpolation='nearest')
        atlas_data = atlas_resampled.get_fdata()

        # --- Step 3: Define Region Mapping Logic ---
        def get_cluster_info(r_id):
            """Helper to map ID to Cluster name and Hemisphere."""
            if 1 <= r_id <= 14:
                name = 'SFG'
            elif 15 <= r_id <= 28:
                name = 'MFG'
            elif 29 <= r_id <= 40:
                name = 'IFG'
            elif 53 <= r_id <= 64:
                name = 'PreCG'
            else:
                return None, None
                
            # Lateralization: Odd IDs = Left Hemisphere (LH), Even IDs = Right (RH)
            hemi = 'LH' if r_id % 2 != 0 else 'RH'
            return name, hemi

        # Get unique IDs present in the resampled atlas data
        all_ids = np.unique(atlas_data)
        results = []

        # --- Step 4: Iterative Voxel Counting ---
        for r_id in all_ids:
            if r_id == 0: continue # Skip background
            
            cluster, hemi = get_cluster_info(r_id)
            if cluster:
                # Create a boolean mask for the current specific region ID
                region_mask = (atlas_data == r_id)
                
                # Count total voxels in this region and how many are 'active' (1.0)
                total_vox = np.sum(region_mask)
                active_vox = np.sum(mask_clean[region_mask] == 1.0)
                
                results.append({
                    'Cluster': cluster,
                    'Hemi': hemi,
                    'Active_Voxel': active_vox,
                    'Total_Voxel': total_vox
                })

        # --- Step 5: Data Aggregation and Optimization ---
        df_raw = pd.DataFrame(results)
        
        # Group by the specified clusters and hemispheres
        summary = df_raw.groupby(['Cluster', 'Hemi']).agg({
            'Active_Voxel': 'sum',
            'Total_Voxel': 'sum'
        }).reset_index()

        # Calculate the percentage ratio: (Active / Total) * 100
        # Formula: $$ \text{Ratio} = \frac{\sum \text{Active Voxels}}{\sum \text{Total Voxels}} \times 100 $$
        summary['Ratio_Percent'] = (summary['Active_Voxel'] / summary['Total_Voxel']) * 100
        
        # Sort the table for better readability
        sort_mapping = {'SFG': 0, 'MFG': 1, 'IFG': 2, 'PreCG': 3}
        summary['Sort_ID'] = summary['Cluster'].map(sort_mapping)
        summary = summary.sort_values(['Sort_ID', 'Hemi']).drop(columns=['Sort_ID'])

        def convert_table_to_dict(df):
            hemi_dict = {}
            for hemi in ['LH', 'RH']:
                hemi_df = df[df['Hemi'] == hemi]
                hemi_dict[hemi] = dict(zip(hemi_df['Cluster'], hemi_df['Active_Voxel']))
            return hemi_dict
        
        return convert_table_to_dict(summary)

    def analyze_and_plot_density_comparison(density_vwm, density_novwm, mask_vwm, mask_novwm, g_name, base_color, desat_color):

        threshold_val = 1e-6
        vals_vwm = density_vwm[density_vwm > threshold_val].ravel()
        vals_novwm = density_novwm[density_novwm > threshold_val].ravel()
        
        # 对数坐标要求数据必须大于 0，进行简单的正值过滤
        vals_vwm = vals_vwm[vals_vwm > 0]
        vals_novwm = vals_novwm[vals_novwm > 0]
        
        if len(vals_vwm) == 0 or len(vals_novwm) == 0: return

        # --- 统计计算（KL 散度计算基于原始分布） ---
        bins_log = np.logspace(np.log10(min(vals_vwm.min(), vals_novwm.min())), 
                            np.log10(max(vals_vwm.max(), vals_novwm.max())), 100)
        p_vwm, _ = np.histogram(vals_vwm, bins=bins_log, density=True)
        q_novwm, _ = np.histogram(vals_novwm, bins=bins_log, density=True)
        kl_div = entropy(p_vwm + 1e-10, q_novwm + 1e-10)

        # 阈值计算
        vals_vwm_masked = density_vwm[mask_vwm > 0].ravel()
        vals_novwm_masked = density_novwm[mask_novwm > 0].ravel()
        thresh_v = np.min(vals_vwm_masked)
        thresh_n = np.min(vals_novwm_masked)

        # --- 绘图设置 ---
        plt.figure(figsize=(6, 4))
        ax = plt.gca()

        # 设置 X 轴为对数坐标
        ax.set_xscale('log')

        # 绘制直方图和 KDE
        # 注意：在 log 坐标下，log_scale=True 可以让 sns 处理得更好
        sns.histplot(vals_vwm, color=base_color, kde=True, element="step", alpha=0.3, 
                    stat="density", line_kws={'linewidth': 3.5}, ax=ax, log_scale=True)
        sns.histplot(vals_novwm, color=desat_color, kde=True, element="step", alpha=0.3, 
                    stat="density", line_kws={'linewidth': 3.5}, ax=ax, log_scale=True)
        
        # 显著性阈值虚线
        plt.axvline(thresh_v, color=base_color, linestyle="--", linewidth=2.5, alpha=0.8)
        plt.axvline(thresh_n, color=desat_color, linestyle="--", linewidth=2.5, alpha=0.8)

        # --- 坐标轴美化 ---
        # 1. 完全隐藏 Y 轴
        ax.get_yaxis().set_visible(False)
        
        # 2. 移除 X 轴标签，仅保留刻度
        ax.set_xlabel("") 
        
        # 3. 设置 X 轴刻度字体大小并加粗轴线
        ax.tick_params(axis='x', labelsize=24)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # 4. “透气”风格：移除上方、右方和左方（Y轴）的轴线
        # 因为 Y 轴已隐藏，despine 主要处理 offset
        sns.despine(left=True, offset=10, trim=True)

        # 5. 移除可能的图例
        if ax.get_legend():
            ax.get_legend().remove()

        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(manuscript_save_dir, f"{g_name}_denhist.svg")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return kl_div

    groups_to_plot = ['Auditory', 'Sensorimotor', 'Motor', 'DelayOnly']
    groups = {
        'Auditory': {'idx': LexDelay_Auditory_in_Delay_sig_idx, 'rgb': Auditory_col},
        'Sensorimotor': {'idx': LexDelay_Sensorimotor_in_Delay_sig_idx, 'rgb': Sensorimotor_col},
        'Motor': {'idx': LexDelay_Motor_in_Delay_sig_idx, 'rgb': Motor_col},
        'DelayOnly': {'idx': LexDelay_DelayOnly_sig_idx, 'rgb': Delay_col}
    }

    novWM_sig_idx = LexDelay_all_sig_idx - LexDelay_Delay_sig_idx
    groups_novWM = {
        'Auditory': {'idx': LexDelay_Aud_NoMotor_sig_idx & novWM_sig_idx, 'rgb': Auditory_col},
        'Sensorimotor': {'idx': LexDelay_Sensorimotor_sig_idx & novWM_sig_idx, 'rgb': Sensorimotor_col},
        'Motor': {'idx': LexDelay_Motor_sig_idx & novWM_sig_idx, 'rgb': Motor_col}
    }

    all_elec = set(range(len(data_LexDelay_Aud.labels[0])))

    def get_desaturated_color(rgb, factor=0.4):
        return [c + (1.0 - c) * factor for c in rgb]

    def make_comparison_cmap(base_rgb, desat_rgb):
        arr = np.array([
            [1, 1, 1],    
            base_rgb,     
            desat_rgb,    
            [1, 1, 0] 
        ], dtype=float)
        return ListedColormap(arr)

    def get_nan_mask_balanced(chs_coor, target_indices, fixed_bw=0.3, abs_thresh=1e-3):
        target_df = chs_coor.iloc[list(target_indices)]
        
        res = 2
        x_range, y_range, z_range = slice(-70, 71, res), slice(-100, 71, res), slice(-60, 81, res)
        x_grid, y_grid, z_grid = np.mgrid[x_range, y_range, z_range]
        grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
        
        final_density = np.zeros(x_grid.shape, dtype=np.float32)
        final_mask = np.full(x_grid.shape, np.nan, dtype=np.float32)

        hemi_logic = {
            'LH': target_df[target_df['x'] < 0],
            'RH': target_df[target_df['x'] > 0]
        }

        for hemi_name, hemi_df in hemi_logic.items():
            coords = hemi_df[['x', 'y', 'z']].values
            n = len(coords)
            if n < 3: 
                continue

            dist_matrix = cdist(coords, coords)
            avg_dists = np.sum(dist_matrix, axis=1) / (n - 1)
            point_weights = 1.0 / (avg_dists + 1e-6) 
            
            point_weights = point_weights / np.sum(point_weights) 

            kernel = stats.gaussian_kde(coords.T, weights=point_weights)
            kernel.set_bandwidth(bw_method=fixed_bw)
            
            hemi_mask_in_grid = (x_grid.ravel() < 0) if hemi_name == 'LH' else (x_grid.ravel() > 0)
            hemi_density_flat = np.zeros(grid_coords.shape[1])
            hemi_density_flat[hemi_mask_in_grid] = kernel(grid_coords[:, hemi_mask_in_grid])
            
            hemi_density = hemi_density_flat.reshape(x_grid.shape)
            final_density += hemi_density

            #non_zero_vals = hemi_density[hemi_density > 0]
            # if len(non_zero_vals) > 0:
            #     thresh = np.percentile(non_zero_vals, 95)
            #     final_mask[hemi_density >= thresh] = 1.0 
        non_zero_vals = hemi_density[final_density > 0]
        if len(non_zero_vals) > 0:
            thresh = np.percentile(non_zero_vals, 95)
            final_mask[final_density >= thresh] = 1.0         

        return final_mask, final_density

    def build_comparison_volume(m_vwm, m_novwm, voxel_size=2, origin=(-70, -100, -60),delay_only=False):
        if m_vwm is None or m_novwm is None: return None
        if not delay_only:
            combined = np.full(m_vwm.shape, np.nan, dtype=np.float32)
            is_vwm, is_novwm = m_vwm == 1.0, m_novwm == 1.0
            combined[is_vwm & ~is_novwm] = 1.0  
            combined[~is_vwm & is_novwm] = 2.0  
            combined[is_vwm & is_novwm]  = 3.0  
            affine = np.eye(4); affine[0,0]=affine[1,1]=affine[2,2]=float(voxel_size)
            affine[:3, 3] = np.array(origin, dtype=float)
            return nib.Nifti1Image(combined, affine)
        else:
            combined = np.full(m_vwm.shape, np.nan, dtype=np.float32)
            is_vwm, is_novwm = m_vwm == 1.0, m_novwm == 1.0
            combined[is_vwm & ~is_novwm] = 1.0  
            combined[~is_vwm & is_novwm] = 1.0  
            combined[is_vwm & is_novwm]  = 1.0  
            affine = np.eye(4); affine[0,0]=affine[1,1]=affine[2,2]=float(voxel_size)
            affine[:3, 3] = np.array(origin, dtype=float)
            return nib.Nifti1Image(combined, affine)

    def plot_all_electrodes_density(chs_coor, all_idx, fixed_bw=0.3):
        res_data = get_nan_mask_balanced(chs_coor, list(all_idx), fixed_bw=fixed_bw)
        if res_data is None: return
        _, density = res_data 

        voxel_size = 2
        origin = (-70, -100, -60)
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = float(voxel_size)
        affine[:3, 3] = np.array(origin, dtype=float)
        
        density_img = nib.Nifti1Image(density, affine)

        view = plotting.view_img_on_surf(
            density_img,
            surf_mesh='fsaverage',
            threshold='50%', 
            cmap=pink_cmap,  
            title="",
            colorbar=False,
            vol_to_surf_kwargs={'n_samples': 1, 'radius': 0.0}
        )

        ipy_display(view)

    plot_all_electrodes_density(chs_coor, all_elec)

    def plot_grouped_electrode_counts(groups, groups_novWM, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        labels = ['Auditory', 'Sensorimotor', 'Motor', 'DelayOnly']
        vwm_counts = []
        novwm_counts = []
        vwm_colors = []
        novwm_colors = []

        for label in labels:
            v_idx = groups[label]['idx']
            vwm_counts.append(len(v_idx))
            vwm_colors.append(groups[label]['rgb'])
            
            if label in groups_novWM:
                n_idx = groups_novWM[label]['idx']
                novwm_counts.append(len(n_idx))
                novwm_colors.append(get_desaturated_color(groups[label]['rgb'], factor=0.6))
            else:
                novwm_counts.append(0)
                novwm_colors.append([1, 1, 1, 0])

        x = np.arange(len(labels))
        width = 0.4 

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        offsets = np.array([-width/2, -width/2, -width/2, 0])
        ax.bar(x + offsets, vwm_counts, width, color=vwm_colors, edgecolor='none', linewidth=0)
        ax.bar(x[:3] + width/2, novwm_counts[:3], width, color=novwm_colors[:3], edgecolor='none', linewidth=0)

        # 坐标轴加粗
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        
        # 顶级期刊“透气”偏置风格
        sns.despine(offset=15, trim=True)
        
        # 移除所有文本标签
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        # 移除 X 轴刻度标签 (No x ticks)
        ax.set_xticklabels([]) 
        plt.yticks(fontsize=30)
        
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

        plt.tight_layout()

        # 保存为 SVG 格式，方便在 Adobe Illustrator 中排版
        save_path = os.path.join(save_dir, "ele_grp_counts.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.show()

    # 
    plot_grouped_electrode_counts(groups, groups_novWM, manuscript_save_dir)

    # 遍历组别进行绘图
    common_groups = ['Auditory', 'Sensorimotor', 'Motor', 'DelayOnly']
    abs_threshs = [1e-3,1e-3,1e-3,1e-4]

    def swap_dict_layers(d):
        swapped = {}
        for k1, v1 in d.items():
            for k2, v2 in v1.items():
                swapped.setdefault(k2, {})[k1] = v2
        return swapped

    for g_name,abs_thresh in zip(common_groups,abs_threshs):
        idx_vwm = groups[g_name]['idx']
        if g_name !='DelayOnly':
            idx_novwm = groups_novWM[g_name]['idx']
        else:
            idx_novwm = idx_vwm
        base_color = groups[g_name]['rgb']
        
        m_vwm, density_vwm = get_nan_mask_balanced(chs_coor, idx_vwm, abs_thresh=abs_thresh)
        m_novwm, density_novwm = get_nan_mask_balanced(chs_coor, idx_novwm)
        
        if m_vwm is not None and m_novwm is not None:
            if g_name !='DelayOnly':
                comp_img = build_comparison_volume(m_vwm, m_novwm)
            else:
                comp_img = build_comparison_volume(m_vwm, m_novwm,delay_only=True)
            desat_color = get_desaturated_color(base_color, factor=0.6)
            comp_cmap = make_comparison_cmap(base_color, desat_color)
            
            # 生成 3D 视图（无标题）
            view = plotting.view_img_on_surf(
                comp_img, threshold=0, vmin=0, vmax=3,
                surf_mesh='fsaverage',
                vol_to_surf_kwargs={'n_samples': 1, 'radius': 0.0},
                cmap=comp_cmap, symmetric_cmap=False,
                title="", # 移除标题
                colorbar=False
            )
            
            # 显示图例和视图
            ipy_display(view)
            df_vwm = analyze_bna_activation(m_vwm, 'BN_Atlas_246_1mm.nii.gz')
            df_novwm = analyze_bna_activation(m_novwm, 'BN_Atlas_246_1mm.nii.gz')
            if g_name !='DelayOnly':
                final_data = {
                    "vWM": df_vwm,
                    "no vWM": df_novwm
                }
            else:
                final_data = {
                    "vWM": df_vwm,
                    "no vWM": {'LH': {'SFG': 0, 'MFG': 0, 'IFG': 0, 'PreCG': 0},
                                'RH': {'SFG': 0, 'MFG': 0, 'IFG': 0, 'PreCG': 0}}
                }
            final_data = swap_dict_layers(final_data)
            # bar plots for frontal
            generate_neuro_publication_plot(final_data,save_path=os.path.join(manuscript_save_dir, f"frontal_distrib_{g_name}.svg"))
            # Desnity plots
            analyze_and_plot_density_comparison(density_vwm, density_novwm, m_vwm, m_novwm,g_name, base_color, desat_color)


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

    unit_scale = 3.0        # 每 1 秒数据在纸面上的物理长度 (inches/sec)
    left_padding_with_y = 1.6  # 有 Y 轴时的留白 (用于 Stim)
    left_padding_no_y = 0.2    # 无 Y 轴时的留白 (用于 Go/Resp, 仅保留极小呼吸空间)
    right_padding = 0.4     
    fig_height = 3.0        

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 2.0
    # --- 2. 循环处理 vWM (Delay) 和 novWM (Without_Delay) ---
    for roi_idx, roi_idx_tag in zip(
            (LexDelay_Delay_sig_idx, LexDelay_all_sig_idx - LexDelay_Delay_sig_idx),
            ('Delay', 'Without_Delay')):

        wm_tag = 'vWM' if roi_idx_tag == 'Delay' else 'novWM'
        linestyle = '-' if roi_idx_tag == 'Delay' else '--'
        
        # 定义功能组颜色
        plot_groups = [
            (LexDelay_Aud_NoMotor_sig_idx & roi_idx, 'Auditory', Auditory_col),
            (LexDelay_DelayOnly_sig_idx & roi_idx, 'Delay Only', Delay_col),
            (LexDelay_Sensorimotor_sig_idx & roi_idx, 'Sensorymotor', Sensorimotor_col),
            (LexDelay_Motor_sig_idx & roi_idx, 'Motor', Motor_col)
        ]

        alignments = [
            ('Stim', epoc_LexDelay_Aud, [-0.25, 1.5], True),
            ('Go', epoc_LexDelay_Go, [-0.25, 1.0], range(631, 650)),
            ('Resp', epoc_LexDelay_Resp, [-0.25, 1.0], range(631, 650))
        ]

        for align_tag, epoc_data, x_limits, bsl_val in alignments:
            # --- 核心：判断是否保留 Y 轴并计算宽度 ---
            has_y = (align_tag == 'Stim')
            current_left_pad = left_padding_with_y if has_y else left_padding_no_y
            
            x_duration = x_limits[1] - x_limits[0]
            # 物理宽度公式
            fig_width = (x_duration * unit_scale) + current_left_pad + right_padding
            
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
            fig.subplots_adjust(left=current_left_pad/fig_width, 
                                right=1.0 - (right_padding/fig_width), 
                                bottom=0.25, top=0.9)
            ax = plt.gca()

            # 调用波形绘制函数
            for sig_idx, label_text, group_col in plot_groups:
                if len(sig_idx) == 0: continue
                plot_wave(epoc_data, sig_idx, f'{label_text}',
                        group_col, linestyle, bsl_val, ylim=[-0.2, 1.5])

            # --- 3. 彻底移除 Y 轴逻辑 ---
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines['bottom'].set_linewidth(3)
            
            if not has_y:
                # 彻底隐藏左侧轴线、刻度和标签
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
                ax.yaxis.set_visible(False) # 禁用整个 Y 轴对象
            else:
                # 仅在 Stim 图中保留加粗 Y 轴
                ax.spines['left'].set_linewidth(3)
                ax.set_ylim([-0.2, 1.5])
                ax.tick_params(axis='y', labelsize=24, length=6, width=2.5)

            # X 轴刻度设置
            xticks = [0, 0.5, 1.0, 1.5]
            xticks = [t for t in xticks if x_limits[0] <= t <= x_limits[1]]
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
            
            plt.draw()
            labels = [l.get_text() for l in ax.get_xticklabels()]
            new_labels = ["0" if (l == "0.0" or l == ".0") else l for l in labels]
            ax.set_xticklabels(new_labels)
            
            # 呼吸透气风格 (despine 需要配合 has_y)
            sns.despine(ax=ax, offset=10, trim=True, left=not has_y)
            
            # X 轴标准化
            ax.set_xlim(x_limits)
            ax.tick_params(axis='x', labelsize=24, length=6, width=2.5)
            ax.spines['bottom'].set_bounds(x_limits[0], x_limits[1])
            
            ax.set_xlabel(''); ax.set_ylabel('')
            
            # 基准线
            ax.axvline(x=0, linestyle='--', color='#444444', linewidth=1.5, dashes=(5, 5), zorder=0)
            ax.axhline(y=0, linestyle='-', color='#DDDDDD', linewidth=1.0, zorder=0)

            if ax.get_legend(): ax.get_legend().remove()

            # --- 4. 保存至 Fig1 目录 ---
            save_filename = f"Wave_{wm_tag}_{align_tag}.svg"
            save_dir = os.path.join(manuscript_save_dir, '..', 'Fig1')
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir)
            
            save_path = os.path.join(save_dir, save_filename)
            plt.savefig(save_path, format='svg', bbox_inches=None)
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
            df = len(x_valid) - 2
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
                
                # Export 
                # %%
                
                for Hickok_ROI_data_type, Hickok_ROI_data_type_tag in zip(
                    (Hickok_ROI_data, Hickok_ROI_epoch),
                    ('mask','epoch')
                ):
                    # Create directory if it doesn't exist
                    output_dir = os.path.join("projects", "Greg_ROIs", f"{tag}")
                    os.makedirs(output_dir, exist_ok=True)

                    Hickok_ROI_data_type_dict = Hickok_ROI_data_type.to_dict()
                    Hickok_ROI_data_type_frame = pd.DataFrame(Hickok_ROI_data_type_dict)

                    Hickok_ROI_data_type_frame.columns = Hickok_ROI_data_type_frame.columns.astype(str).str.replace("#", "", regex=False)
                    Hickok_ROI_data_type_frame.index = Hickok_ROI_data_type_frame.index.astype(str).str.replace("#", "", regex=False)


                    Hickok_rawdata_save_dir = os.path.join("projects", "Greg_ROIs", f"{tag}")
                    os.makedirs(Hickok_rawdata_save_dir, exist_ok=True)

                    Hickok_ROI_data_type_frame.to_csv(
                        os.path.join(Hickok_rawdata_save_dir, f"{tag}_{epoch_tag}_{Hickok_ROI_data_type_tag}.csv"),
                        index=True
                    )

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

    import matplotlib.gridspec as gridspec

    # --- 1. 全局参数与环境设置 ---
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.size'] = 24  # 顶级期刊大字体
    plt.rcParams['axes.linewidth'] = 3 

    wav_bsl_corr_val = True
    go_resp_bsl = range(281, 300)

    if not os.path.exists(manuscript_save_dir):
        os.makedirs(manuscript_save_dir)

    fig = plt.figure(figsize=(11.5, 4), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.75, 1], wspace=0.1)

    # --- 2. ax1: Stimulus-locked ---
    ax1 = fig.add_subplot(gs[0])
    x_lim_stim = [-0.25, 1.5] 

    plot_wave(epoc_LexNoDelay_Aud, LexDelay_Aud_NoMotor_sig_idx & LexDelay_Delay_sig_idx, 
            '', Auditory_col, '-', wav_bsl_corr_val, ylim=[-0.2, 1.5])
    plot_wave(epoc_LexNoDelay_Aud, LexDelay_DelayOnly_sig_idx & LexDelay_Delay_sig_idx, 
            '', Delay_col, '-', wav_bsl_corr_val, ylim=[-0.2, 1.5])
    plot_wave(epoc_LexNoDelay_Aud, LexDelay_Sensorimotor_sig_idx & LexDelay_Delay_sig_idx, 
            '', Sensorimotor_col, '-', wav_bsl_corr_val, ylim=[-0.2, 1.5])
    plot_wave(epoc_LexNoDelay_Aud, LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx, 
            '', Motor_col, '-', wav_bsl_corr_val, ylim=[-0.2, 1.5])
    # --- 3. ax3: Response-locked ---
    ax3 = fig.add_subplot(gs[1])
    x_lim_resp = [-0.25, 1.0]

    plot_wave(epoc_LexNoDelay_Resp, LexDelay_Aud_NoMotor_sig_idx & LexDelay_Delay_sig_idx, '', Auditory_col, '-', False, ylim=[-0.2, 1.5])
    plot_wave(epoc_LexNoDelay_Resp, LexDelay_DelayOnly_sig_idx & LexDelay_Delay_sig_idx, '', Delay_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])
    plot_wave(epoc_LexNoDelay_Resp, LexDelay_Sensorimotor_sig_idx & LexDelay_Delay_sig_idx, '', Sensorimotor_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])
    plot_wave(epoc_LexNoDelay_Resp, LexDelay_Motor_sig_idx & LexDelay_Delay_sig_idx, '', Motor_col, '-', go_resp_bsl, ylim=[-0.2, 1.5])

    # --- 4. 核心：坐标轴精细化修饰 ---
    for ax in [ax1, ax3]:
        curr_xlim = x_lim_stim if ax == ax1 else x_lim_resp
        
        # A. 设定 X 轴刻度：明确从 0 开始，每 0.5s 一个
        # 这样即便线段始于 -0.25，刻度也只从 0 出现
        ax.set_xticks(np.arange(0, curr_xlim[1] + 0.01, 0.5))
        ax.set_xlim(curr_xlim)
        
        # B. 设定 Y 轴刻度：仅保留 0 和 1
        ax.set_yticks([0, 1])
        ax.set_ylim([-0.2, 1.2]) 

        # C. 移除边框并设置基础 Offset
        sns.despine(ax=ax, offset=15, trim=False) # 禁用自动 trim
        
        # D. 手动设置线段起止点 (Trim 替代方案)
        # X 轴线段：保持原定的 -0.25 到上限
        ax.spines['bottom'].set_bounds(curr_xlim[0], curr_xlim[1])
        # Y 轴线段：严格限制在 0 到 1 之间
        if ax == ax1:
            ax.spines['left'].set_bounds(0, 1)
        
        # E. 移除 Labels
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')
        
        # F. 辅助线与刻度样式 (24号字体)
        ax.axvline(x=0, linestyle='--', color='#444444', linewidth=1.5, dashes=(5, 5), zorder=0)
        ax.axhline(y=0, linestyle='-', color='#DDDDDD', linewidth=1, zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=24, direction='out', length=6, width=3)
        
        if ax == ax3:
            ax.set_yticklabels([])
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False)

    # 调整布局
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)

    # --- 5. 导出 SVG 矢量图 ---
    save_path = os.path.join(manuscript_save_dir, '..', 'Fig4', "Nodelay_trace.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')

    plt.show()
    print(f"Standardized and trimmed wave plot saved to: {save_path}")

    # Get electrodes HG power during the first 100ms after aud onset and after mtr onset
    _, _, _, _, _, paras_aud, *_ = sort_chs_by_actonset(data_LexNoDelay_Aud, epoc_LexNoDelay_Aud, cluster_twin, [0.05, 0.15], mask_data=True, select_electrodes=False)
    _, _, _, _, _, paras_mtr, *_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, [-0.05, 0.05], mask_data=True, select_electrodes=False)

    import seaborn as sns
    import matplotlib.ticker as mticker
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    # ==========================================
    # 1. 数据准备与标签提取
    # ==========================================
    group_map = {
        'Auditory': data_LexNoDelay_Aud.labels[0][list(LexDelay_Auditory_in_Delay_sig_idx)],
        'Sensory-motor': data_LexNoDelay_Aud.labels[0][list(LexDelay_Sensorimotor_in_Delay_sig_idx)],
        'Motor': data_LexNoDelay_Aud.labels[0][list(LexDelay_Motor_in_Delay_sig_idx)],
        'Delay-only': data_LexNoDelay_Aud.labels[0][list(LexDelay_DelayOnly_sig_idx)]
    }

    param = 'rms_value'
    x_order = ['Auditory', 'Sensory-motor', 'Motor', 'Delay-only']
    hue_colors_stage = {'Auditory Stage': Auditory_col, 'Motor Stage': Motor_col}

    fig4_dir = os.path.join(manuscript_save_dir, '..', 'Fig4')
    if not os.path.exists(fig4_dir):
        os.makedirs(fig4_dir)

    # ==========================================
    # 2. 数据处理与参与度统计
    # ==========================================
    plot_data = []
    participation_data = {}

    for g_name, labels in group_map.items():
        if len(labels) == 0: continue
        
        # 提取原始数值
        v_aud_raw = paras_aud.loc[paras_aud.index.intersection(labels), param]
        v_mtr_raw = paras_mtr.loc[paras_mtr.index.intersection(labels), param]
        total_n = len(labels)
        
        # 统计 0 值比例 (0 视为无效)
        a_valid = np.sum(v_aud_raw != 0)
        m_valid = np.sum(v_mtr_raw != 0)
        
        participation_data[g_name] = {
            'aud': [a_valid, total_n - a_valid],
            'mtr': [m_valid, total_n - m_valid]
        }

        # 构建长表
        for label in labels:
            if label in paras_aud.index and label in paras_mtr.index:
                va, vm = paras_aud.loc[label, param], paras_mtr.loc[label, param]
                # 过滤 0 值
                va = va if va != 0 else np.nan
                vm = vm if vm != 0 else np.nan
                plot_data.append({'Group': g_name, 'Stage': 'Auditory Stage', 'Value': va})
                plot_data.append({'Group': g_name, 'Stage': 'Motor Stage', 'Value': vm})

    df_long = pd.DataFrame(plot_data).dropna(subset=['Value'])

    # ==========================================
    # 3. 绘制分阶段参与度饼图 (数量-换行-百分比 + 无外部标签)
    # ==========================================
    # 统一灰色背景
    unified_grey = '#D3D3D3' 

    # 定义自定义格式化函数：数量\n(百分比)
    def make_participation_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            # 按照要求：数量 - 换行 - (百分比)
            return f'{val}\n({pct:.0f}%)'
        return my_autopct

    for g_name, counts in participation_data.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        
        # 配色：饱和色 vs 统一灰色
        colors_aud = [Auditory_col, unified_grey]
        colors_mtr = [Motor_col, unified_grey]
        
        # A. Auditory Stage Pie
        axes[0].pie(
            counts['aud'], 
            labels=None,      # 移除外部电极组标签
            colors=colors_aud, 
            autopct=None, # 先不显示百分比，后续通过自定义函数添加数量和百分比
            #autopct=make_participation_autopct(counts['aud']), # 数量\n(百分比)
            startangle=90,
            pctdistance=0.5, # 调整文字在圆内的位置
            textprops={'fontsize': 32, 'fontweight': 'bold', 'color': 'black'}, # 维持大字号
            wedgeprops={'linewidth': 2.5, 'edgecolor': 'white'}
        )
        
        # B. Motor Stage Pie
        axes[1].pie(
            counts['mtr'], 
            labels=None,      # 移除外部电极组标签
            colors=colors_mtr, 
            autopct=None, # 先不显示百分比，后续通过自定义函数添加数量和百分比
            #autopct=make_participation_autopct(counts['mtr']), # 数量\n(百分比)
            startangle=90,
            pctdistance=0.5,
            textprops={'fontsize': 32, 'fontweight': 'bold', 'color': 'black'}, # 维持大字号
            wedgeprops={'linewidth': 2.5, 'edgecolor': 'white'}
        )
        
        # 环形化与视觉纯净处理
        for ax in axes:
            # 保持 Donut 风格，中间留白
            ax.add_artist(plt.Circle((0,0), 0.5, fc='white'))
            ax.axis('equal')
            ax.set_title('') # 确保无标题

        plt.tight_layout()
        # 存入 Fig4 目录
        plt.savefig(os.path.join(fig4_dir, f'Pie_Split_{g_name}.svg'), format='svg', bbox_inches='tight')
        plt.show()

    # ==========================================
    # 4. 统计检验 (非参数)
    # ==========================================
    def run_nonparametric(df, stage):
        stage_df = df[df['Stage'] == stage]
        group_data = [stage_df[stage_df['Group'] == g]['Value'] for g in x_order]
        res = []
        for i in range(len(x_order)):
            for j in range(i + 1, len(x_order)):
                u, p = stats.mannwhitneyu(group_data[i], group_data[j])
                res.append({'Comparison': f'{x_order[i]} vs {x_order[j]}', 'p_raw': p})
        p_raws = [r['p_raw'] for r in res]
        _, p_fdr, _, _ = multipletests(p_raws, method='fdr_bh')
        for idx, r in enumerate(res): r['p_fdr'] = p_fdr[idx]
        return pd.DataFrame(res)

    print("--- Statistics ---")
    print("Auditory Stage:\n", run_nonparametric(df_long, 'Auditory Stage'))
    print("\nMotor Stage:\n", run_nonparametric(df_long, 'Motor Stage'))

    # ==========================================
    # 5. 绘制饱和色 Bar-Strip 图 (No Title)
    # ==========================================
    plt.figure(figsize=(10.5, 4))
    ax = plt.gca()

    # 柱状图
    sns.barplot(
        x='Group', y='Value', data=df_long, hue='Stage', hue_order=['Auditory Stage', 'Motor Stage'],
        order=x_order, palette=hue_colors_stage, edgecolor='none', saturation=1, 
        alpha=1.0, errorbar='se', capsize=0.05, err_kws={'linewidth': 3, 'color': 'black'}, zorder=1, ax=ax
    )

    # 散点图
    strip_plot = sns.stripplot(
        data=df_long, x='Group', y='Value', hue='Stage', hue_order=['Auditory Stage', 'Motor Stage'],
        order=x_order, size=7, alpha=1.0, jitter=0.25, dodge=True, marker='o', 
        linewidth=1.2, edgecolor='white', palette=hue_colors_stage, zorder=2, ax=ax
    )

    # 连线 (以后再搞吧累了真的)
    # for group_idx in range(len(x_order)):
    #     try: bsliang_add_connecting_lines(plt, group_idx, strip_plot)
    #     except: pass

    # --- 顶级期刊视觉规范 (移除标题) ---
    for spine in ax.spines.values(): spine.set_linewidth(3)
    sns.despine(offset=15, trim=True) # 呼吸感偏置

    plt.yticks(fontsize=30); plt.xticks(fontsize=0) 
    ax.set_title('') # 显式移除标题
    ax.set_xlabel(''); ax.set_ylabel(''); ax.tick_params(axis='x', length=0)

    if ax.get_legend(): ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(os.path.join(fig4_dir, f"RMS_BarStrip.svg"), format='svg', bbox_inches='tight')
    plt.show()
# %%
