# %% groups of patients
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

# Parameters from the lexical delay task
mean_word_len=0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0.4 # a short period of time that we may assume auditory decay takes
delay_len=0.5 # from task script
motor_prep_win=[-0.5,-0.1] # get windows for motor preparation (0.1s to avoid high gamma filter leakage)
motor_resp_win=[-0.1,0.75] # get windows for motor response (0.75s to avoid too much auditory feedback)
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

# %% Sort data and get significant electrode lists
import os
import pickle
import numpy as np
from utils.group import load_stats, sort_chs_by_actonset, plot_chs, plot_brain, find_com_sig_chs, plot_wave,set2arr,get_notmuscle_electrodes, chs2atlas, atlas2_hist, hickok_roi, plot_sig_roi_counts
import matplotlib.pyplot as plt

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

    data_LexDelay_Aud,subjs=load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_delay,stats_root_delay)
    data_LexDelay_Go, _ = load_stats(stat_type, 'Go'+Delayseleted, contrast, stats_root_delay, stats_root_delay)
    data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_delay, stats_root_delay)

    clean_chs_idx = get_notmuscle_electrodes(data_LexDelay_Aud)

    epoc_LexDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_delay,stats_root_delay)
    epoc_LexDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_delay,stats_root_delay)

elif groupsTag=="LexNoDelay":

    data_LexNoDelay_Aud,subjs=load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_nodelay)
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

    clean_chs_idx = get_notmuscle_electrodes(data_LexNoDelay_Aud)

    epoc_LexNoDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_nodelay)
    epoc_LexNoDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_nodelay)

elif groupsTag=="LexDelay&LexNoDelay":

    # first get the patient inform from no delay tasks and then extract the corresponding
    data_LexDelay_Aud,subjs=load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_nodelay,stats_root_delay)
    data_LexNoDelay_Aud,_=load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_nodelay)

    clean_chs_idx = get_notmuscle_electrodes(data_LexDelay_Aud)

    data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_nodelay, stats_root_delay)
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

    epoc_LexDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_delay)
    epoc_LexDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_nodelay)

    epoc_LexNoDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_delay)
    epoc_LexNoDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_nodelay)

# Get the ROI of labels
ch_labels_roi,ch_labels=chs2atlas(subjs)
hickok_roi_labels=hickok_roi(ch_labels_roi,ch_labels)
# Get sorted electrodes
if "LexDelay" in groupsTag:

    # sort the data according to the onset within a time range (Full)
    data_LexDelay_sorted,_,_,LexDelay_sig_idx = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[-10,10])
    LexDelay_sig_idx = LexDelay_sig_idx & clean_chs_idx
    # plot the data
    plot_chs(data_LexDelay_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_sig_idx)}")

    # (Auditory)
    data_LexDelay_Aud_sorted,_,_,LexDelay_Aud_sig_idx = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[-0.1,mean_word_len+auditory_decay])
    LexDelay_Aud_sig_idx = LexDelay_Aud_sig_idx & clean_chs_idx
    plot_chs(data_LexDelay_Aud_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-{'Auditory'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Aud_sig_idx)}")

    # (Delay)
    data_LexDelay_Delay_sorted,_,_,LexDelay_Delay_sig_idx=sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
    LexDelay_Delay_sig_idx = LexDelay_Delay_sig_idx & clean_chs_idx
    plot_chs(data_LexDelay_Delay_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-Delay_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Delay_sig_idx)}")

    # (Go)
    # data_LexDelay_Go_sorted, _, LexDelay_Go_sig_idx = sort_chs_by_actonset(data_LexDelay_Go, cluster_twin, [0.25, 0.75])
    # LexDelay_Go_sig_idx=LexDelay_Go_sig_idx & clean_chs_idx
    # plot_chs(data_LexDelay_Go_sorted, os.path.join(fig_save_dir, f'{'Go_inRep'}_{stat_type}-{contrast}.jpg'))

    # (Motor prepare)
    # !!!!!!!!!!In the future use **set** to replace the list indexing!!!!!!!!!!!!!!!!!

    data_LexDelay_Motor_Prep_sorted, _, _, LexDelay_Motor_Prep_sig_idx = sort_chs_by_actonset(data_LexDelay_Resp,epoc_LexDelay_Resp, cluster_twin, motor_prep_win)
    LexDelay_Motor_Prep_sig_idx = LexDelay_Motor_Prep_sig_idx & clean_chs_idx
    plot_chs(data_LexDelay_Motor_Prep_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Motor_Prep'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Motor_Prep_sig_idx)}")

    # (Motor response)
    data_LexDelay_Motor_Resp_sorted, _, _, LexDelay_Motor_Resp_sig_idx = sort_chs_by_actonset(data_LexDelay_Resp, epoc_LexDelay_Resp, cluster_twin, motor_resp_win)
    LexDelay_Motor_Resp_sig_idx = LexDelay_Motor_Resp_sig_idx & clean_chs_idx
    plot_chs(data_LexDelay_Motor_Resp_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Motor_Resp'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexDelay_Motor_Resp_sig_idx)}")

    # Channel selection: Auditory nomotor electrodes (auditory window:1, motor prep: 0)
    LexDelay_Aud_NoMotor_sig_idx = LexDelay_Aud_sig_idx - LexDelay_Motor_Prep_sig_idx

    # Channel selection: Sensorimotor electrodes (auditory window:1, motor prep: 1)
    LexDelay_Sensorimotor_sig_idx = LexDelay_Aud_sig_idx & LexDelay_Motor_Prep_sig_idx

    # Channel selection: Sensory OR motor electrodes (auditory window:1 or motor prep: 1 or motor resp: 1)
    LexDelay_Sensory_OR_Motor_sig_idx = LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx

    # Channel selection: Motor electrodes (auditory window:0, motor resp: 1)
    LexDelay_Motor_sig_idx = LexDelay_Motor_Resp_sig_idx - LexDelay_Aud_sig_idx

    # Channel selection: Delay only electrodes (delay electrodes ,with: auditory window:0, motor prep: 0, motor resp: 0)
    LexDelay_DelayOnly_sig_idx = LexDelay_Delay_sig_idx - (LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx)

    # Channel selection: Auditory electrodes in Delay electrodes
    LexDelay_Auditory_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Aud_NoMotor_sig_idx

    # Channel selection: Sensorimotor electrodes in Delay electrodes
    LexDelay_Sensorimotor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Sensorimotor_sig_idx

    # Channel selection: Motor electrodes in Delay electrodes
    LexDelay_Motor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Motor_sig_idx

    # Motor_prep only
    LexDelay_Motorprep_Only_sig_idx = (LexDelay_Motor_Prep_sig_idx - (LexDelay_Aud_NoMotor_sig_idx | LexDelay_Sensorimotor_sig_idx | LexDelay_Motor_sig_idx | LexDelay_DelayOnly_sig_idx))

    # Others
    # LexDelay_Other_sig_idx = ((LexDelay_Aud_sig_idx | LexDelay_Delay_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx)
    #                           - (LexDelay_Aud_NoMotor_sig_idx | LexDelay_Sensorimotor_sig_idx | LexDelay_Motor_sig_idx | LexDelay_DelayOnly_sig_idx))

    LexDelay_idxes=dict()
    LexDelay_idxes['LexDelay_Aud_NoMotor_sig_idx']=LexDelay_Aud_NoMotor_sig_idx
    LexDelay_idxes['LexDelay_Sensorimotor_sig_idx']=LexDelay_Sensorimotor_sig_idx
    LexDelay_idxes['LexDelay_Motor_sig_idx']=LexDelay_Motor_sig_idx
    LexDelay_idxes['LexDelay_DelayOnly_sig_idx']=LexDelay_DelayOnly_sig_idx
    LexDelay_idxes['LexDelay_Auditory_in_Delay_sig_idx']=LexDelay_Auditory_in_Delay_sig_idx
    LexDelay_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx']=LexDelay_Sensorimotor_in_Delay_sig_idx
    LexDelay_idxes['LexDelay_Motor_in_Delay_sig_idx']=LexDelay_Motor_in_Delay_sig_idx
    LexDelay_idxes['LexDelay_Motorprep_Only_sig_idx']=LexDelay_Motorprep_Only_sig_idx
    with open(os.path.join('projects','GLM','data', 'LexDelay_twin_idxes.npy'), "wb") as f:
        pickle.dump(LexDelay_idxes, f)

    del data_LexDelay_sorted, data_LexDelay_Aud_sorted, data_LexDelay_Delay_sorted, data_LexDelay_Motor_Prep_sorted, data_LexDelay_Motor_Resp_sorted

if "LexNoDelay" in groupsTag:

    # (Auditory)
    data_LexNoDelay_Aud_sorted,_,LexNoDelay_Aud_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Aud,cluster_twin,[-0.1,mean_word_len+auditory_decay])
    LexNoDelay_Aud_sig_idx = LexNoDelay_Aud_sig_idx & clean_chs_idx
    plot_chs(data_LexNoDelay_Aud_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexNoDelay-{'Auditory_inRep'}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexNoDelay_Aud_sig_idx)}")

    # (Go)
    # data_LexNoDelay_Go_sorted, _, LexNoDelay_Go_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Go, cluster_twin, [0.25, 0.75])
    # LexNoDelay_Go_sig_idx = LexNoDelay_Go_sig_idx & clean_chs_idx
    # plot_chs(data_LexNoDelay_Go_sorted, os.path.join(fig_save_dir, f'{'Go_inRep'}_{stat_type}-{contrast}.jpg'))

    # (Motor prepare)
    data_LexNoDelay_Motor_Prep_sorted, _, LexNoDelay_Motor_Prep_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Resp, cluster_twin, motor_prep_win)
    LexNoDelay_Motor_Prep_sig_idx = LexNoDelay_Motor_Prep_sig_idx & clean_chs_idx
    plot_chs(data_LexNoDelay_Motor_Prep_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexNoDelay-{'Motor_Prep'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexNoDelay_Motor_Prep_sig_idx)}")

    # (Motor response)
    data_LexNoDelay_Motor_Resp_sorted, _, LexNoDelay_Motor_Resp_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Resp, cluster_twin, motor_resp_win)
    LexNoDelay_Motor_Resp_sig_idx = LexNoDelay_Motor_Resp_sig_idx & clean_chs_idx
    plot_chs(data_LexNoDelay_Motor_Resp_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexNoDelay-{'Motor_Resp'+Delayseleted}_{stat_type}-{contrast}.jpg'),f"N chs = {len(LexNoDelay_Motor_Resp_sig_idx)}")

    # Channel selection: Auditory nomotor electrodes (auditory window:1, motor prep: 0)
    LexNoDelay_Aud_NoMotor_sig_idx = LexNoDelay_Aud_sig_idx - LexNoDelay_Motor_Prep_sig_idx

    # Channel selection: Sensorimotor electrodes (auditory window:1, motor prep: 1)
    LexNoDelay_Sensorimotor_sig_idx = LexNoDelay_Aud_sig_idx & LexNoDelay_Motor_Prep_sig_idx

    # Channel selection: Sensory OR motor electrodes (auditory window:1 or motor prep: 1 or motor resp: 1)
    LexNoDelay_Sensory_OR_Motor_sig_idx = LexNoDelay_Aud_sig_idx | LexNoDelay_Motor_Prep_sig_idx | LexNoDelay_Motor_Resp_sig_idx

    # Channel selection: Motor electrodes (auditory window:0, motor resp: 1)
    LexNoDelay_Motor_sig_idx = LexNoDelay_Motor_Resp_sig_idx - LexNoDelay_Aud_sig_idx

    del data_LexNoDelay_Aud_sorted, data_LexNoDelay_Motor_Prep_sorted, data_LexNoDelay_Motor_Resp_sorted

if "&" in groupsTag:
    # Select the Delay electrodes (all electrodes with significant delay activations)
    # in the lexical delay tasks,
    # then select the electrodes among them that had auditory responses in the no delay tasks.
    LexDelay_Delay_LexNoDelay_Aud_sig_idx = find_com_sig_chs(
        data_LexDelay_Aud.labels[0],LexDelay_Delay_sig_idx,
        data_LexNoDelay_Aud.labels[0],LexNoDelay_Aud_NoMotor_sig_idx)

    LexDelay_Delay_LexNoDelay_Sensorimotor_sig_idx = find_com_sig_chs(
        data_LexDelay_Aud.labels[0],LexDelay_Delay_sig_idx,
        data_LexNoDelay_Aud.labels[0],LexNoDelay_Sensorimotor_sig_idx)

    LexDelay_Delay_LexNoDelay_Motor_sig_idx = find_com_sig_chs(
        data_LexDelay_Aud.labels[0],LexDelay_Delay_sig_idx,
        data_LexNoDelay_Aud.labels[0],LexNoDelay_Motor_sig_idx)

    LexDelay_Delay_LexNoDelay_Silent_sig_idx = LexDelay_Delay_sig_idx - (
                LexNoDelay_Aud_NoMotor_sig_idx | LexNoDelay_Sensorimotor_sig_idx | LexNoDelay_Motor_sig_idx)

    # May do in_Silence electrodes later

# %% reassign electrode indices by conditions
Sensorimotor_col = [1, 0, 0]  # Sensorimotor (Red)
Auditory_col = [0, 1, 0]  # Auditory (Green)
Delay_col = [1, 0.65, 0]  # Delay (Orange)
Motor_col = [0, 0, 1]  # Motor (Blue)
Sensorimotor_Delay_col = [1, 0, 1]  # Sensorimotor-Delay (Purple)
Auditory_Delay_col = [1, 1, 0]  # Auditory-Delay (Yellow)
Delay_Motor_col = [0, 1, 1]  # Delay-Motor (Greenblue)
Waveplot_wth=10 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots

if groupsTag == "LexDelay":

    # %% Electrode selection
    # Location plot for different types of electrodes
    len_d=len(data_LexDelay_Aud.labels[0])
    for TypeLabel,chs_ov,pick_sig_idx in zip(
            ('Sensorimotor','Auditory','Delay','Delay_overlapped','Delay_only','Motor','Sensory_OR_Motor'),
            ([1000,0,0,0],[0,100,0,0],[0,0,10,0],[1000,100,10,1],[1000,100,10,1],[0,0,0,1],[1000,100,0,1]),
            (set2arr(LexDelay_Sensorimotor_sig_idx,len_d),
             set2arr(LexDelay_Aud_NoMotor_sig_idx,len_d),
             set2arr(LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_Delay_sig_idx,len_d),
             set2arr(LexDelay_DelayOnly_sig_idx,len_d),
             set2arr(LexDelay_Motor_sig_idx,len_d),
             set2arr(LexDelay_Sensory_OR_Motor_sig_idx,len_d))
    ):

        # Elecorde selection and color assigning

        color_map = {
            1000: Sensorimotor_col, # Sensorimotor (Orange)
             100: Auditory_col,  # Auditory (Red)
              10: Delay_col,  # Delay (Green)
               1: Motor_col,  # Motor (Blue)
            1010: Sensorimotor_Delay_col,  # Sensorimotor-Delay (Purple)
             110: Auditory_Delay_col, # Auditory-Delay (Yellow)
              11: Delay_Motor_col # Delay-Motor (Greenblue)
        }

        chs_col_idx=[chs_ov[0]*set2arr(LexDelay_Sensorimotor_sig_idx,len_d)[i]
                     +chs_ov[1]*set2arr(LexDelay_Aud_NoMotor_sig_idx,len_d)[i]
                     +chs_ov[2]*set2arr(LexDelay_Delay_sig_idx,len_d)[i]
                     +chs_ov[3]*set2arr(LexDelay_Motor_sig_idx,len_d)[i] for i in range(len_d)]
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
        atlas2_hist(ch_labels_roi,pick_labels,chs_cols_picked[0],os.path.join(fig_save_dir,f'Atlas histogram {TypeLabel.replace('/', ' ')}.tif'))
        plot_sig_roi_counts(hickok_roi_labels, chs_cols_picked[0], pick_sig_idx, os.path.join(fig_save_dir,f'Hickok ROI histogram {TypeLabel.replace('/', ' ')}.tif'))

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plot_wave(epoc_LexDelay_Aud, LexDelay_Sensorimotor_sig_idx, f'Sensorimotor n={len(LexDelay_Sensorimotor_sig_idx)}', Sensorimotor_col,'-',False)
    plot_wave(epoc_LexDelay_Aud, LexDelay_Aud_NoMotor_sig_idx, f'Auditory n={len(LexDelay_Aud_NoMotor_sig_idx)}',Auditory_col,'-',False)
    plot_wave(epoc_LexDelay_Aud, LexDelay_Motor_sig_idx, f'Motor n={len(LexDelay_Motor_sig_idx)}',Motor_col,'-',False)
    plot_wave(epoc_LexDelay_Aud, LexDelay_Delay_sig_idx, f'Delay n={len(LexDelay_Delay_sig_idx)}', Delay_col,'--',False)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.title('Z-scores in lexical delay repeat tasks (aligned to stim onset)')
    plt.legend(loc='upper right')
    plt.xlim([-0.25,1.6])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir,'LexDelay_sig_zscore_Aud.tif'),dpi=300)
    plt.close()

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to motor onset)
    plt.figure(figsize=(Waveplot_wth*(150/350), Waveplot_hgt))
    plot_wave(epoc_LexDelay_Resp, LexDelay_Sensorimotor_sig_idx,
              f'Sensorimotor n={len(LexDelay_Sensorimotor_sig_idx)}', Sensorimotor_col,'-',False)
    plot_wave(epoc_LexDelay_Resp, LexDelay_Aud_NoMotor_sig_idx, f'Auditory n={len(LexDelay_Aud_NoMotor_sig_idx)}',
              Auditory_col,'-',False)
    plot_wave(epoc_LexDelay_Resp, LexDelay_Motor_sig_idx, f'Motor n={len(LexDelay_Motor_sig_idx)}', Motor_col,'-',False)
    plot_wave(epoc_LexDelay_Resp, LexDelay_Delay_sig_idx, f'Delay n={len(LexDelay_Delay_sig_idx)}', Delay_col,'--',False)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.title('Z-scores (aligned to motor onset)')
    plt.legend().set_visible(False)
    plt.xlim([-0.25,1])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, 'LexDelay_sig_zscore_Resp.tif'), dpi=300)
    plt.close()

    # Plot Delay electrodes
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    num_delay_elec=len(LexDelay_Delay_sig_idx)
    plot_wave(epoc_LexDelay_Aud, LexDelay_Delay_sig_idx,
              f'Delay All n={num_delay_elec}', [0.5,0.5,0.5],'--',False)
    plot_wave(epoc_LexDelay_Aud, LexDelay_DelayOnly_sig_idx,
              f'Delay Only n={len(LexDelay_DelayOnly_sig_idx)} '
              f'({np.round(100*len(LexDelay_DelayOnly_sig_idx)/num_delay_elec,3)}%)', Delay_col,'-',False)
    plot_wave(epoc_LexDelay_Aud, LexDelay_Auditory_in_Delay_sig_idx,
              f'Auditory in Delay n={len(LexDelay_Auditory_in_Delay_sig_idx)} '
              f'({np.round(100*len(LexDelay_Auditory_in_Delay_sig_idx)/num_delay_elec,3)}%)', Auditory_Delay_col,'-',False)
    plot_wave(epoc_LexDelay_Aud, LexDelay_Sensorimotor_in_Delay_sig_idx,
              f'Sensorimotor in Delay n={len(LexDelay_Sensorimotor_in_Delay_sig_idx)} '
              f'({np.round(100*len(LexDelay_Sensorimotor_in_Delay_sig_idx)/num_delay_elec,3)}%)', Sensorimotor_Delay_col,'-',False)
    plot_wave(epoc_LexDelay_Aud, LexDelay_Motor_in_Delay_sig_idx,
              f'Motor in Delay n={len(LexDelay_Motor_in_Delay_sig_idx)} '
              f'({np.round(100*len(LexDelay_Motor_in_Delay_sig_idx)/num_delay_elec,3)}%)',Delay_Motor_col,'-',False)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.title('Z-scores in lexical delay repeat tasks for delay electrodes (aligned to stim onset)')
    plt.legend()
    # plt.xlim([-0.25, 1.6])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, 'LexDelay_Delay_sig_zscore_Resp.tif'), dpi=300)
    plt.close()

elif groupsTag == "LexNoDelay":
    for TypeLabel, chs_ov, pick_sig_idx in zip(
            ('Sensorimotor', 'Auditory', 'Motor', 'Sensory_OR_Motor'),
            ([1000, 0, 0, 0], [0, 100, 0, 0], [0, 0, 0, 1],[1000, 100, 0, 1]),
            (LexNoDelay_Sensorimotor_sig_idx, LexNoDelay_Aud_NoMotor_sig_idx, LexNoDelay_Motor_sig_idx,LexNoDelay_Sensory_OR_Motor_sig_idx)
    ):

        # Elecorde selection and color assigning

        color_map = {
            1000: Sensorimotor_col,  # Sensorimotor (Orange)
            100: Auditory_col,  # Auditory (Red)
            1: Motor_col  # Motor (Blue)
        }

        chs_col_idx = [
            chs_ov[0] * LexNoDelay_Sensorimotor_sig_idx[i] + chs_ov[1] * LexNoDelay_Aud_NoMotor_sig_idx[i] + chs_ov[3] * LexNoDelay_Motor_sig_idx[i] for i in
            range(len(data_LexNoDelay_Aud.labels[0]))]
        picks = [i for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[i] == 1]
        pick_labels = [data_LexNoDelay_Aud.labels[0][i] for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[
            i] == 1]  # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
        chs_cols = [color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(data_LexNoDelay_Aud.labels[0]))]
        chs_cols_picked = [chs_cols[i] for i in picks]

        # TRY also to plot valid (white?) vs. invalid electrodes (dark grey)
        plot_brain(subjs, pick_labels, chs_cols_picked, None,
                   os.path.join(fig_save_dir, f'{TypeLabel}_{stat_type}-{contrast}.jpg'))

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_Sensorimotor_sig_idx, f'Sensorimotor n={np.sum(LexNoDelay_Sensorimotor_sig_idx)}', Sensorimotor_col)
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_Aud_NoMotor_sig_idx, f'Auditory n={np.sum(LexNoDelay_Aud_NoMotor_sig_idx)}',Auditory_col)
    plot_wave(epoc_LexNoDelay_Aud, LexNoDelay_Motor_sig_idx, f'Motor n={np.sum(LexNoDelay_Motor_sig_idx)}',Motor_col)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.title('Lexical No Delay (Auditory onset aligned)')
    plt.show()

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to motor onset)
    plt.figure(figsize=(Waveplot_wth*(150/350), Waveplot_hgt))
    plot_wave(epoc_LexNoDelay_Resp, LexNoDelay_Sensorimotor_sig_idx, f'Sensorimotor n={np.sum(LexNoDelay_Sensorimotor_sig_idx)}', Sensorimotor_col)
    plot_wave(epoc_LexNoDelay_Resp, LexNoDelay_Aud_NoMotor_sig_idx, f'Auditory n={np.sum(LexNoDelay_Aud_NoMotor_sig_idx)}',Auditory_col)
    plot_wave(epoc_LexNoDelay_Resp, LexNoDelay_Motor_sig_idx, f'Motor n={np.sum(LexNoDelay_Motor_sig_idx)}',Motor_col)
    plt.axvline(x=0, linestyle='--', color='k')
    plt.title('Lexical No Delay (Auditory onset aligned)')
    plt.legend().set_visible(False)
    plt.show()

elif groupsTag=="LexDelay&LexNoDelay":

    TypeLabel='LexNoDelay_in_LexDelayDelay'
    chs_ov=[1000, 100, 0, 1]
    pick_sig_idx=LexDelay_Delay_sig_idx

    # Elecorde selection and color assigning

    color_map = {
        1000: Sensorimotor_col,  # Sensorimotor (Orange)
        100: Auditory_col,  # Auditory (Red)
        1: Motor_col,  # Motor (Blue)
        0: [1,1,1] # Silent (White)
    }

    chs_col_idx = [
        chs_ov[0] * LexDelay_Delay_LexNoDelay_Sensorimotor_sig_idx[i] + chs_ov[1] * LexDelay_Delay_LexNoDelay_Aud_sig_idx[i] + chs_ov[3] * LexDelay_Delay_LexNoDelay_Motor_sig_idx[i] for i in
        range(len(data_LexNoDelay_Aud.labels[0]))]
    picks = [i for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[i] == 1]
    pick_labels = [data_LexNoDelay_Aud.labels[0][i] for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[
        i] == 1]  # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
    chs_cols = [color_map.get(chs_col_idx[i], [0.5,0.5,0.5]) for i in range(len(data_LexNoDelay_Aud.labels[0]))]
    chs_cols_picked = [chs_cols[i] for i in picks]

    # TRY also to plot valid (white?) vs. invalid electrodes (dark grey)
    plot_brain(subjs, pick_labels, chs_cols_picked, None,
               os.path.join(fig_save_dir, f'{TypeLabel}_{stat_type}-{contrast}.jpg'))

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to auditory onset)
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    num_delay_elec=np.sum(pick_sig_idx)
    plot_wave(epoc_LexNoDelay_Aud, LexDelay_Delay_LexNoDelay_Sensorimotor_sig_idx, f'Sensorimotor in LexNoDelay n={np.sum(LexDelay_Delay_LexNoDelay_Sensorimotor_sig_idx)}'
              f'({np.round(100*np.sum(LexDelay_Delay_LexNoDelay_Sensorimotor_sig_idx)/num_delay_elec,3)}%)', Sensorimotor_col)
    plot_wave(epoc_LexNoDelay_Aud, LexDelay_Delay_LexNoDelay_Aud_sig_idx, f'Auditory in LexNoDelay n={np.sum(LexDelay_Delay_LexNoDelay_Aud_sig_idx)}'
              f'({np.round(100 * np.sum(LexDelay_Delay_LexNoDelay_Aud_sig_idx) / num_delay_elec, 3)}%)',Auditory_col)
    plot_wave(epoc_LexNoDelay_Aud, LexDelay_Delay_LexNoDelay_Motor_sig_idx, f'Motor in LexNoDelay n={np.sum(LexDelay_Delay_LexNoDelay_Motor_sig_idx)}'
              f'({np.round(100 * np.sum(LexDelay_Delay_LexNoDelay_Motor_sig_idx) / num_delay_elec, 3)}%)',Motor_col)
    plot_wave(epoc_LexNoDelay_Aud, LexDelay_Delay_LexNoDelay_Silent_sig_idx, f'Others in LexNoDelay n={np.sum(LexDelay_Delay_LexNoDelay_Silent_sig_idx)}'
              f'({np.round(100 * np.sum(LexDelay_Delay_LexNoDelay_Silent_sig_idx) / num_delay_elec, 3)}%)', [0.5,0.5,0.5])
    plt.axvline(x=0, linestyle='--', color='k')
    plt.title('LexDelay tasks delay electrodes in LexNoDelay (Auditory onset aligned)')
    plt.show()

    # Plot Sensorimotor, Auditory, and Motor electrodes (Aligned to motor onset)
    plt.figure(figsize=(Waveplot_wth*(150/350), Waveplot_hgt))
    plot_wave(epoc_LexNoDelay_Resp, LexDelay_Delay_LexNoDelay_Sensorimotor_sig_idx, f'x', Sensorimotor_col)
    plot_wave(epoc_LexNoDelay_Resp, LexDelay_Delay_LexNoDelay_Aud_sig_idx, f'x',Auditory_col)
    plot_wave(epoc_LexNoDelay_Resp, LexDelay_Delay_LexNoDelay_Motor_sig_idx, f'x',Motor_col)
    plot_wave(epoc_LexNoDelay_Resp, LexDelay_Delay_LexNoDelay_Silent_sig_idx, f'x', [0.5,0.5,0.5])
    plt.axvline(x=0, linestyle='--', color='k')
    plt.title('(Motor onset aligned)')
    plt.legend().set_visible(False)
    plt.show()