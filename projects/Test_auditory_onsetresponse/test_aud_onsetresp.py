#%% Import everything
import os
import matplotlib.pyplot as plt
# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\Test_auditory_onsetresponse\\test_aud_onsetresp.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions

# For lexical delay task, whether run the data only with repeat tasks
#Delayseleted=''
Delayseleted = '_inRep'

# Parameters from the lexical delay task
mean_word_len=0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0.4 # a short period of time that we may assume auditory decay takes
delay_len=0.5 # from task script
motor_prep_win=[-0.5,-0.1] # get windows for motor preparation (0.1s to avoid high gamma filter leakage)
motor_resp_win=[-0.1,0.75] # get windows for motor response (0.75s to avoid too much auditory feedback)
pre_stimonset_win=[-0.5,mean_word_len+auditory_decay]
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

First_col = [1, 0, 0]
Second_col = [0, 0, 1]
Waveplot_wth=10 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots

# %% Sort data and get significant electrode lists
import os
import pickle
import numpy as np
import utils.group as gp
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")

stats_save_root = os.path.join(stats_root_delay,'group')
if not os.path.exists(os.path.join(stats_save_root)):
    os.mkdir(os.path.join(stats_save_root))

data_LexDelay_Aud,subjs=gp.load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_delay,stats_root_delay)
epoc_LexDelay_Aud_first,_=gp.load_stats('power','Auditory_inRep','epo',stats_root_delay,stats_root_delay,split_half=1)
epoc_LexDelay_Aud_second,_=gp.load_stats('power','Auditory_inRep','epo',stats_root_delay,stats_root_delay,split_half=2)

#%% turns data to list of subject subdata
from collections import defaultdict
subject_sets = defaultdict(set)
for i,elec in enumerate(data_LexDelay_Aud.labels[0]):
    subj, elec_name = elec.split('-')
    subject_sets[subj].add(i)

for subj, sub_chs in subject_sets.items():
    data_LexDelay_Aud_s = gp.sel_subj_data(data_LexDelay_Aud,sub_chs)
    epoc_LexDelay_Aud_first_s = gp.sel_subj_data(epoc_LexDelay_Aud_first,sub_chs)
    epoc_LexDelay_Aud_second_s = gp.sel_subj_data(epoc_LexDelay_Aud_second,sub_chs)

    try:
        data_LexDelay_sorted_preonset_first, _, _, LexDelay_sig_idx_preonset = gp.sort_chs_by_actonset(data_LexDelay_Aud_s,epoc_LexDelay_Aud_first_s,
                                                                                                       cluster_twin,pre_stimonset_win)
        data_LexDelay_sorted_preonset_second, _, _, _ = gp.sort_chs_by_actonset(data_LexDelay_Aud_s, epoc_LexDelay_Aud_second_s,
                                                                                cluster_twin, pre_stimonset_win)

        gp.plot_chs(data_LexDelay_sorted_preonset_first, os.path.join('data', f'{subj} First half.jpg'),
                 f"{subj} First Half of trials")
        gp.plot_chs(data_LexDelay_sorted_preonset_second, os.path.join('data', f'{subj} Second half.jpg'),
                 f"{subj} Second Half of trials")

        plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
        gp.plot_wave(epoc_LexDelay_Aud_first_s, LexDelay_sig_idx_preonset,f'First Half of trials', First_col, '-', False)
        gp.plot_wave(epoc_LexDelay_Aud_second_s, LexDelay_sig_idx_preonset, f'Second Half of trials', Second_col, '-', False)
        plt.axvline(x=0, linestyle='--', color='k')
        plt.axhline(y=0, linestyle='--', color='k')
        plt.title(f"{subj} Auditory responses")
        plt.legend(loc='upper right')
        plt.xlim(pre_stimonset_win)
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join('data', f'{subj} wave.jpg'), dpi=300)
        plt.close()

    except Exception as e:
        continue
