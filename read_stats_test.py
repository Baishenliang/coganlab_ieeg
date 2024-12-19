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
contrast='ave_YN_Rep'

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
chs_cols=[[0.5,0.5,0.5]]*len(data.labels[0])
chs_sizes=[0]*len(data.labels[0])

# Define colors for each category in [0, 1] scale

color_map = {
    1: [0, 0.5, 1],  # Auditory only (Blue)
    10: [0.5, 0, 1],  # Delay only (Purple)
    100: [0, 1, 0],  # Go only or Delay Go (Green)
    110: [0, 1, 0],  # Go only or Delay Go (Green)
    1000: [1, 0.5, 0],  # Motor only (Orange)
    1001: [1, 0, 0],  # Auditory + Motor (Red)
    1101: [1, 0, 0],  # Auditory + Go + Motor (Red)
    11: [0, 1, 1],  # Auditory + Delay (Cyan)
    1100: [1, 1, 0],  # Go + Motor (Yellow)
    1110: [1, 1, 0],  # Delay Go + Motor (Yellow)
    1010: [0.5, 0.5, 0],  # Delay + Motor (Olive Green)
    111: [1, 1, 1],  # Mixed (White)
    1011: [1, 1, 1],  # Mixed (White)
    1110: [1, 1, 1],  # Mixed (White)
    1111: [1, 1, 1],  # Mixed (White)
}

chs_col_idx=[1*aud_sig_idx[i]+10*del_sig_idx[i]+100*go_sig_idx[i]+1000*resp_sig_idx[i] for i in range(len(data.labels[0]))]
chs_cols =[color_map.get(chs_col_idx[i], [0.5, 0.5, 0.5]) for i in range(len(data.labels[0]))]  # Default grey if not mapped

plot_brain(subjs, chs_cols,chs_sizes)