# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions
#contrast='ave_YN_Rep' # contrasting yesno to repetition
#contrast='ave_Rep_YN' # contrasting repetition to yesno
#contrast='ave_W_NW' # contrasting word to nonword trials only in repetition
#contrast='ave_NW_W' # contrasting nonword to word trials only in repetition

# %% prerparation
import os
import numpy as np
from utils.group import load_stats, sort_chs_by_actonset, plot_chs, plot_brain

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
stats_root = os.path.join(LAB_root, 'BIDS-1.0_RetroCue', 'BIDS', "derivatives", "stats")

fig_save_dir = os.path.join(LAB_root, 'D_Data','Retro_Cue','Baishen_Figs','group')
if not os.path.exists(os.path.join(fig_save_dir)):
    os.mkdir(os.path.join(fig_save_dir))

stats_save_root = os.path.join(stats_root,'group')
if not os.path.exists(os.path.join(stats_save_root)):
    os.mkdir(os.path.join(stats_save_root))

# Parameters from the retro cue task
mean_word_len=0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0.4 # a short period of time that we may assume auditory decay takes
delay_len=1.5 # from task script
cue_len=0.7 #length of retro
motor_win=[-0.5,0] # get windows for motor responses (from Motor onset,not including motor onset (or else it should be [0.25, 0.75]))
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

# %% get auditory and delay electrodes
# selected='_in_REP_BTH' # select trials
# selected='_in_REP_1ST'
# selected='_in_REP_2ND'
# selected='_in_REV_BTH'
selected='_in_DRP_BTH'

# (Auditory1)
con='Auditory1'+selected
data,subjs=load_stats(stat_type,con,contrast,stats_root)
data_sorted_aud_1,_,aud_1_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-0.1,mean_word_len+auditory_decay])
plot_chs(data_sorted_aud_1,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))
del data_sorted_aud_1

# (Auditory2 + Delay1)
con='Auditory2'+selected
data,subjs=load_stats(stat_type,con,contrast,stats_root)
data_sorted_aud_2,_,aud_2_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-0.1,mean_word_len+auditory_decay+delay_len])
plot_chs(data_sorted_aud_2,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))
del data_sorted_aud_2

# (Cue + Delay1)
con='Cue'+selected
data,subjs=load_stats(stat_type,con,contrast,stats_root)
data_sorted_cue,_,cue_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-0.1,cue_len+delay_len])
plot_chs(data_sorted_cue,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))
del data_sorted_cue

if selected!='_in_DRP_BTH': # no "Go" or "Resp" for drop both conditions
    # (Go)
    con='Go'+selected
    data,subjs=load_stats(stat_type,con,contrast,stats_root)
    data_sorted_go,_,go_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-0.1,0.1])
    plot_chs(data_sorted_go,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))
    del data_sorted_go

    # (Resp)
    con='Resp'+selected
    data,subjs=load_stats(stat_type,con,contrast,stats_root)
    data_sorted_resp,_,resp_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-0.1,1])
    plot_chs(data_sorted_resp,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}.jpg'))
    del data_sorted_resp

else:
    go_sig_idx=[0 for i in range(len(aud_1_sig_idx))]

# Motor electrodes are the Go electrodes subtracted by Auditory electrodes
# i.e.,
# Auditory electrodes:  **Activated** in auditory window; whether or not activated after Go does not matter.
# Motor electrodes:  **Not** activated in auditory window; **Activated** after Go
if selected!='_in_DRP_BTH': # no "Go" or "Resp" for drop both conditions
    motor_sig_idx = [0 if aud_1_sig_idx[i] == 1 or aud_2_sig_idx[i] == 1 else resp_sig_idx[i] for i in range(len(resp_sig_idx))]
else:
    motor_sig_idx = go_sig_idx
# %% reassign electrode indices by conditions

for TypeLabel,chs_ov,pick_sig_idx in zip(
        ('Auditory1','Auditory2_Delay','Cue_Delay','Go','Motor'),
        ([10000,0,0,0,0],[0,1000,0,0,0],[0,0,100,0,0],[0,0,0,10,0],[0,0,0,0,1]),
        (aud_1_sig_idx,aud_2_sig_idx,cue_sig_idx,go_sig_idx,motor_sig_idx)
):
    if selected == '_in_DRP_BTH' and (chs_ov == 'Go' or chs_ov == 'Motor'):
        break

    # Elecorde selection and color assigning

    color_map = {
        10000: [1, 0, 0],  # Auditory1 (Red)
         1000: [1, 0, 0],  # Auditory2_Delay (Red)
          100: [0, 1, 0],  # Cue_Delay (Green)
           10: [0, 0, 1],  # Go (Blue)
            1: [1, 1, 0],  # Motor (Yellow)
    }

    chs_col_idx=[chs_ov[0]*aud_1_sig_idx[i]+chs_ov[1]*aud_2_sig_idx[i]+chs_ov[2]*cue_sig_idx[i]+chs_ov[3]*go_sig_idx[i]+chs_ov[4]*motor_sig_idx[i] for i in range(len(data.labels[0]))]
    picks = [i for i in range(len(data.labels[0])) if pick_sig_idx[i] == 1]
    if not picks:
        break
    chs_cols =[color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(data.labels[0]))]
    chs_cols_picked=[chs_cols[i] for i in picks]

    # Plot (cannot plot D107,D042)
    plot_brain(subjs, picks,chs_cols_picked,os.path.join(fig_save_dir,f'{TypeLabel}_{stat_type}-{contrast}_Tclusthres_{cluster_twin}_3d.jpg'))
