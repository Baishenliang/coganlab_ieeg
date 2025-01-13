# %% define condition and load data
stat_type='mask'
#contrast='ave' # average, not contrasting different conditions
#contrast='ave_YN_Rep' # contrasting yesno to repetition
contrast='ave_Rep_YN' # contrasting repetition to yesno

# %% prerparation
import os
import numpy as np
from utils.group import load_stats, sort_chs_by_actonset, plot_chs, plot_brain

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
stats_root = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")

fig_save_dir = os.path.join(LAB_root, 'D_Data','LexicalDecRepDelay','Baishen_Figs','group')
if not os.path.exists(os.path.join(fig_save_dir)):
    os.mkdir(os.path.join(fig_save_dir))

stats_save_root = os.path.join(stats_root,'group')
if not os.path.exists(os.path.join(stats_save_root)):
    os.mkdir(os.path.join(stats_save_root))

# Parameters from the lexical delay task
mean_word_len=0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0.4 # a short period of time that we may assume auditory decay takes
delay_len=0.5 # from task script
motor_win=[-0.5,0] # get windows for motor responses (from Motor onset,not including motor onset (or else it should be [0.25, 0.75]))
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

# %% get auditory and delay electrodes
con='Auditory'
data,subjs=load_stats(stat_type,con,contrast,stats_root)

# sort the data according to the onset within a time range (Full)
data_sorted,_,all_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-10,10])
# plot the data
plot_chs(data_sorted,os.path.join(fig_save_dir,f'{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))

# (Auditory)
data_sorted_aud,_,aud_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-0.1,mean_word_len+auditory_decay])
plot_chs(data_sorted_aud,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))

# (Delay)
data_sorted_del,_,del_sig_idx=sort_chs_by_actonset(data,cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
plot_chs(data_sorted_del,os.path.join(fig_save_dir,f'Delay_{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))

del data_sorted, data_sorted_aud, data_sorted_del

# %% get Go and Response electrodes
go_sig_idx=[]
resp_sig_idx=[]

for con,trange in zip (('Go','Resp'),([0.25,0.75],motor_win)):

    data,_=load_stats(stat_type,con,contrast,stats_root)

    data_sorted,_,sig_idx=sort_chs_by_actonset(data,cluster_twin,trange)
    plot_chs(data_sorted,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))

    if con=='Go':
        go_sig_idx=sig_idx
    elif con=='Resp':
        resp_sig_idx=sig_idx

# Motor electrodes are the Go electrodes subtracted by Auditory electrodes
# i.e.,
# Auditory electrodes:  **Activated** in auditory window; whether or not activated after Go does not matter.
# Motor electrodes:  **Not** activated in auditory window; **Activated** after Go
motor_sig_idx = [0 if aud_sig_idx[i] == 1 else resp_sig_idx[i] for i in range(len(resp_sig_idx))]
# Currently not using resp_sig_idx

del data_sorted
# %% reassign electrode indices by conditions

for TypeLabel,chs_ov,pick_sig_idx in zip(
        ('Auditory','Delay','Delay_overlapped','Motor'),
        ([100,0,0],[0,10,0],[100,10,1],[0,0,1]),
        (aud_sig_idx,del_sig_idx,del_sig_idx,motor_sig_idx)
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

    chs_col_idx=[chs_ov[0]*aud_sig_idx[i]+chs_ov[1]*del_sig_idx[i]+chs_ov[2]*motor_sig_idx[i] for i in range(len(data.labels[0]))]
    picks=[i for i in range(len(data.labels[0])) if pick_sig_idx[i] == 1]
    # picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 100] # Use this to pick auditory only electrodes (i.e., no delay)
    chs_cols =[color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(data.labels[0]))]
    chs_cols_picked=[chs_cols[i] for i in picks]

    # Plot (cannot plot D107,D042)
    plot_brain(subjs, picks,chs_cols_picked,os.path.join(fig_save_dir,f'{TypeLabel}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}_3d.jpg'))
