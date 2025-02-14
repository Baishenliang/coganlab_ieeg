# %% groups of patients
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
Delayseleted=''
#Delayseleted = '_inRep'

# Parameters from the lexical delay task
mean_word_len=0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0.4 # a short period of time that we may assume auditory decay takes
delay_len=0.5 # from task script
motor_prep_win=[-0.5,-0.1] # get windows for motor preparation
motor_resp_win=[0.25,0.75] # get windows for motor response
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

# %% Sort data and get significant electrode lists
import os
import numpy as np
from utils.group import load_stats, sort_chs_by_actonset, plot_chs, plot_brain, find_com_sig_chs

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

elif groupsTag=="LexNoDelay":

    data_LexNoDelay_Aud,subjs=load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_nodelay)
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

elif groupsTag=="LexDelay&LexNoDelay":

    # first get the patient inform from no delay tasks and then extract the corresponding
    data_LexDelay_Aud,subjs=load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_nodelay,stats_root_delay)
    data_LexNoDelay_Aud,_=load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_nodelay)

    data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_nodelay, stats_root_delay)
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

# Get sorted electrodes
if "LexDelay" in groupsTag:

    # sort the data according to the onset within a time range (Full)
    data_LexDelay_sorted,_,LexDelay_sig_idx = sort_chs_by_actonset(data_LexDelay_Aud,cluster_twin,[-10,10])
    # plot the data
    plot_chs(data_LexDelay_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-{stat_type}-{contrast}.jpg'))

    # (Auditory)
    data_LexDelay_Aud_sorted,_,LexDelay_Aud_sig_idx = sort_chs_by_actonset(data_LexDelay_Aud,cluster_twin,[-0.1,mean_word_len+auditory_decay])
    plot_chs(data_LexDelay_Aud_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-{'Auditory'+Delayseleted}_{stat_type}-{contrast}.jpg'))

    # (Delay)
    data_LexDelay_Delay_sorted,_,LexDelay_Delay_sig_idx=sort_chs_by_actonset(data_LexDelay_Aud,cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
    plot_chs(data_LexDelay_Delay_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexDelay-Delay_{stat_type}-{contrast}.jpg'))

    # (Go)
    # data_LexDelay_Go_sorted, _, LexDelay_Go_sig_idx = sort_chs_by_actonset(data_LexDelay_Go, cluster_twin, [0.25, 0.75])
    # plot_chs(data_LexDelay_Go_sorted, os.path.join(fig_save_dir, f'{'Go_inRep'}_{stat_type}-{contrast}.jpg'))

    # (Resp)
    data_LexDelay_Resp_sorted, _, LexDelay_Resp_sig_idx = sort_chs_by_actonset(data_LexDelay_Resp, cluster_twin, motor_prep_win)
    plot_chs(data_LexDelay_Resp_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexDelay-{'Resp'+Delayseleted}_{stat_type}-{contrast}.jpg'))

    # (Motor)
    # Motor electrodes are the Go electrodes subtracted by Auditory electrodes
    # i.e.,
    # Auditory electrodes:  **Activated** in auditory window; whether or not activated after Go does not matter.
    # Motor electrodes:  **Not** activated in auditory window; **Activated** after Go

    LexDelay_Motor_sig_idx = [0 if LexDelay_Aud_sig_idx[i] == 1 else LexDelay_Resp_sig_idx[i] for i in range(len(LexDelay_Resp_sig_idx))]
    # Delay only electrodes
    LexDelay_DelayOnly_sig_idx = [LexDelay_Delay_sig_idx[i] if (LexDelay_Aud_sig_idx[i] != 1 and LexDelay_Motor_sig_idx[i] != 1 and LexDelay_Delay_sig_idx[i]==1) else 0 for i in range(len(LexDelay_Delay_sig_idx))]

    del data_LexDelay_sorted, data_LexDelay_Aud_sorted, data_LexDelay_Delay_sorted, data_LexDelay_Resp_sorted

if "LexNoDelay" in groupsTag:

    # (Auditory)
    data_LexNoDelay_Aud_sorted,_,LexNoDelay_Aud_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Aud,cluster_twin,[-0.1,mean_word_len+auditory_decay])
    plot_chs(data_LexNoDelay_Aud_sorted,os.path.join(fig_save_dir,f'{groupsTag}-LexNoDelay-{'Auditory_inRep'}_{stat_type}-{contrast}.jpg'))

    # (Resp)
    data_LexNoDelay_Resp_sorted, _, LexNoDelay_Resp_sig_idx = sort_chs_by_actonset(data_LexNoDelay_Resp, cluster_twin, motor_prep_win)
    plot_chs(data_LexNoDelay_Resp_sorted, os.path.join(fig_save_dir, f'{groupsTag}-LexNoDelay-{'Resp_inRep'}_{stat_type}-{contrast}.jpg'))

    # (Motor)
    LexNoDelay_Motor_sig_idx = [0 if LexNoDelay_Aud_sig_idx[i] == 1 else LexNoDelay_Resp_sig_idx[i] for i in range(len(LexNoDelay_Resp_sig_idx))]

    del data_LexNoDelay_Aud_sorted, data_LexNoDelay_Resp_sorted

if "&" in groupsTag:
    # Select the Delay_only electrodes in the lexical delay tasks, \
    # then select the electrodes among them that had auditory responses in the no delay tasks.
    # The LexDelay_Delay_LexNoDelay_Aud_sig_idx will be aligned to the data_LexDelay_Aud
    LexDelay_Delay_LexNoDelay_Aud_sig_idx = find_com_sig_chs(
        data_LexDelay_Aud.labels[0],LexDelay_Delay_sig_idx,
        data_LexNoDelay_Aud.labels[0],LexNoDelay_Aud_sig_idx)

    LexDelay_Delay_LexNoDelay_Motor_sig_idx = find_com_sig_chs(
        data_LexDelay_Aud.labels[0],LexDelay_Delay_sig_idx,
        data_LexNoDelay_Aud.labels[0],LexNoDelay_Motor_sig_idx)

    # Get lexical no delay auditory channels and motor channels in the delay_only channels
    No_LexDelay_Delay_LexNoDelay_Aud_chs = np.sum(LexDelay_Delay_LexNoDelay_Aud_sig_idx)
    No_LexDelay_Delay_LexNoDelay_Motor_chs = np.sum(LexDelay_Delay_LexNoDelay_Motor_sig_idx)

    # May do mine electrodes later
    # Count number of 0s
    LexDelay_Delay_LexNoDelay_AudMotr_sig_idx = LexDelay_Delay_LexNoDelay_Aud_sig_idx + LexDelay_Delay_LexNoDelay_Motor_sig_idx
    No_LexDelay_Delay_LexNoDelay_Other_chs = np.sum(LexDelay_Delay_LexNoDelay_AudMotr_sig_idx == 0)

    print(f"In the Delay Only channels in the LexDelayAuditory channels: "
          f"{No_LexDelay_Delay_LexNoDelay_Aud_chs} are LexNoDelay Auditory channels, "
          f"{No_LexDelay_Delay_LexNoDelay_Motor_chs} are LexNoDelay Motor channels, "
          f"And {No_LexDelay_Delay_LexNoDelay_Other_chs} are other channels (including sensorimotor)")

# %% reassign electrode indices by conditions
if groupsTag == "LexDelay":
    for TypeLabel,chs_ov,pick_sig_idx in zip(
            ('Auditory','Delay','Delay_overlapped','Delay_only','Motor'),
            ([100,0,0],[0,10,0],[100,10,1],[100,10,1],[0,0,1]),
            (LexDelay_Aud_sig_idx,LexDelay_Delay_sig_idx,LexDelay_Delay_sig_idx,LexDelay_DelayOnly_sig_idx,LexDelay_Motor_sig_idx)
    ):

        # Elecorde selection and color assigning

        color_map = {
            100: [1, 0, 0],  # Auditory (Red)
             10: [0, 1, 0],  # Delay (Green)
              1: [0, 0, 1],  # Motor (Blue)
            110: [1, 1, 0], # Auditory-Delay (Yellow)
             11: [0, 1, 1], # Delay-Motor (Greenblue)
            111: [1, 1, 1]  # Auditory-Delay-Motor (White)
        }

        chs_col_idx=[chs_ov[0]*LexDelay_Aud_sig_idx[i]+chs_ov[1]*LexDelay_Delay_sig_idx[i]+chs_ov[2]*LexDelay_Motor_sig_idx[i] for i in range(len(data_LexDelay_Aud.labels[0]))]
        picks = [i for i in range(len(data_LexDelay_Aud.labels[0])) if pick_sig_idx[i] == 1]
        pick_labels = [data_LexDelay_Aud.labels[0][i] for i in range(len(data_LexDelay_Aud.labels[0])) if pick_sig_idx[i] == 1]        # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
        chs_cols =[color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(data_LexDelay_Aud.labels[0]))]
        chs_cols_picked=[chs_cols[i] for i in picks]

        # Plot (cannot plot D107,D042)
        if TypeLabel=='Delay_only':
            label_every=1
        else:
            label_every=None

        # TRY also to plot valid (white?) vs. invalid electrodes (dark grey)
        plot_brain(subjs, pick_labels,chs_cols_picked,label_every,os.path.join(fig_save_dir,f'{TypeLabel}_{stat_type}-{contrast}.jpg'))

elif groupsTag == "LexNoDelay":
    for TypeLabel, chs_ov, pick_sig_idx in zip(
            ('Auditory', 'Motor'),
            ([100, 0, 0], [0, 0, 1]),
            (LexNoDelay_Aud_sig_idx,LexNoDelay_Motor_sig_idx)
    ):

        # Elecorde selection and color assigning
        color_map = {
            100: [1, 0, 0],  # Auditory (Red)
            1: [0, 0, 1]  # Motor (Blue)
        }

        chs_col_idx = [chs_ov[0] * LexNoDelay_Aud_sig_idx[i] + chs_ov[2] * LexNoDelay_Motor_sig_idx[i] for i in range(len(data_LexNoDelay_Aud.labels[0]))]
        picks = [i for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[i] == 1]
        pick_labels = [data_LexNoDelay_Aud.labels[0][i] for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[i] == 1]        # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
        chs_cols = [color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(data_LexNoDelay_Aud.labels[0]))]
        chs_cols_picked = [chs_cols[i] for i in picks]

        # Plot (cannot plot D107,D042)
        plot_brain(subjs, pick_labels, chs_cols_picked,None,
                   os.path.join(fig_save_dir, f'{TypeLabel}_{stat_type}-{contrast}.jpg'))

elif groupsTag=="LexDelay&LexNoDelay":

    TypeLabel = 'LexDelay_Delay'
    chs_ov = [100, 10, 1]
    pick_sig_idx = LexDelay_Delay_sig_idx

    color_map = {
        100: [1, 0, 0],  # Auditory Electrodes only in LexNoDelay (Red)
        10: [0, 1, 0], # Delay electrodes only in LexDelay (Green)
        1: [0, 0, 1],  # Motor Electrodes only in LexNoDelay (Blue)
        110: [1, 1, 0],  # Delay electrodes in LexDelay & Auditory Electrodes only in LexNoDelay (Yellow)
        11: [0, 1, 1],  # Delay electrodes in LexDelay & Motor Electrodes only in LexNoDelay
    }

    chs_col_idx = [chs_ov[0] * LexNoDelay_Aud_sig_idx[i] + chs_ov[1] * LexDelay_Delay_sig_idx[i] + chs_ov[2] * LexNoDelay_Motor_sig_idx[i] for i in range(len(data_LexNoDelay_Aud.labels[0]))]
    picks = [i for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[i] == 1]
    pick_labels = [data_LexNoDelay_Aud.labels[0][i] for i in range(len(data_LexNoDelay_Aud.labels[0])) if pick_sig_idx[
        i] == 1]  # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
    chs_cols = [color_map.get(chs_col_idx[i], [1, 1, 1]) for i in range(len(data_LexNoDelay_Aud.labels[0]))]
    chs_cols_picked = [chs_cols[i] for i in picks]

    # Plot (cannot plot D107,D042)
    plot_brain(subjs, pick_labels, chs_cols_picked,None,
               os.path.join(fig_save_dir, f'{TypeLabel}_{stat_type}-{contrast}.jpg'))