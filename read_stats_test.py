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
delay_len=1 # from task script
go_len=0.5 # from task script
cluster_twin=0.2 # length of sig cluster

# define condition and load data
stat_type='mask'
contrast='ave'

# %% get auditory and delay electrodes
con='Auditory'
data,subjs=load_stats(stat_type,con,contrast,stats_root)

# sort the data according to the onset within a time range (Full)
data_sorted,_,all_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-10,10])
# plot the data
plot_chs(data_sorted,os.path.join(fig_save_dir,f'{stat_type}-{contrast}.jpg'))

# (Auditory)
data_sorted_aud,_,aud_sig_idx=sort_chs_by_actonset(data,cluster_twin,[-0.1,mean_word_len+0.1])
plot_chs(data_sorted_aud,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}.jpg'))

# (Delay)
data_sorted_del,_,del_sig_idx=sort_chs_by_actonset(data,cluster_twin,[mean_word_len-0.1,mean_word_len+delay_len+0.1])
plot_chs(data_sorted_del,os.path.join(fig_save_dir,f'Delay_{stat_type}-{contrast}.jpg'))

del data_sorted, data_sorted_aud, data_sorted_del

# %% get Go and Response electrodes
go_sig_idx=[]
resp_sig_idx=[]

for con,trange in zip (('Go','Resp'),([-0.1,go_len+0.1],[-10, 10])):

    data,_=load_stats(stat_type,con,contrast,stats_root)

    data_sorted,_,sig_idx=sort_chs_by_actonset(data,cluster_twin,trange)
    plot_chs(data_sorted,os.path.join(fig_save_dir,f'{con}_{stat_type}-{contrast}.jpg'))

    if con=='Go':
        go_sig_idx=sig_idx
    elif con=='Resp':
        resp_sig_idx=sig_idx

del data_sorted
# %% reassign electrode indices by conditions

# In the future the follow will become a loop
# Auditory
# chs_ov=[1000,0,0,0]
# pick_sig_idx=aud_sig_idx

# # Delay
# chs_ov=[0,100,0,0]
# pick_sig_idx=del_sig_idx
#
# # Delay_overlapped
# chs_ov=[1000,100,10,1]
# pick_sig_idx=del_sig_idx
#
# # Go
# chs_ov=[0,0,10,0]
# pick_sig_idx=go_sig_idx
#
# # Resp
chs_ov=[0,0,0,1]
pick_sig_idx=resp_sig_idx

# Elecorde selection and color assigning

color_map = {
    1000: [1, 0, 0],  # Auditory (Red)
     100: [0, 1, 0],  # Delay (Green)
      10: [0, 0, 1],  # Go (Blue)
       1: [1, 0, 1],  # Resp (Purple)
    1100: [1, 0.65, 0], # Auditory-delay (Orange)
     110: [0, 1, 1], # Delay-Go (Greenblue)
     101: [1, 1, 0],  # Delay-Resp (Yellow)
     111: [1, 1, 0], # Delay-Go-Resp (Yellow)
    1101: [1, 1, 1], # Auditory-Delay-Resp (White)
    1111: [1, 1, 1] # Auditory-Delay-Go-Resp (White)
}

chs_col_idx=[chs_ov[0]*aud_sig_idx[i]+chs_ov[1]*del_sig_idx[i]+chs_ov[2]*go_sig_idx[i]+chs_ov[3]*resp_sig_idx[i] for i in range(len(data.labels[0]))]
picks=[i for i in range(len(data.labels[0])) if pick_sig_idx[i] == 1]
# picks=[i for i in range(len(data.labels[0])) if chs_col_idx[i] == 1000] # Use this to pick auditory only electrodes
chs_cols =[color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(data.labels[0]))]
chs_cols_picked=[chs_cols[i] for i in picks]

# Plot
plot_brain(subjs, picks,chs_cols_picked)
