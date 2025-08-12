#%% Introduction
# Prepare raw data for LME encoding
import os
import pickle
import numpy as np
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\lme\\prepare_raw.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sf_dir = 'data'
with open(os.path.join('..', 'GLM', 'data', f'Lex_twin_idxes_hg.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)

import sys
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
from ieeg.calc.fast import mixup

# %% function block
mean_word_len=0.65#0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0 # a short period of time that we may assume auditory decay takes
delay_len=1.125 # average length from sound offset to Go onset

def get_time_indexs(time_str,start_float:float=0,end_float:float=delay_len):
    time_str = [float(i) for i in time_str]
    start_idx = np.searchsorted(time_str, start_float, side='left')
    end_idx = np.searchsorted(time_str, end_float, side='right')
    indices = list(range(start_idx, end_idx))
    return indices

# %% groups of patients
datasource='hg' # 'glm_(Feature)' or 'hg'
groupsTag="LexDelay"
#groupsTag="LexDelay&LexNoDelay"

# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions
# For lexical delay task, whether run the data only with repeat tasks
trial_labels='CORRECT'

# %% Sort data and get significant electrode lists
import os
import numpy as np
import matplotlib.pyplot as plt
from ieeg.arrays.label import LabeledArray

if os.path.exists(os.path.join(sf_dir,'epoc_LexDelayRep_Aud.npy')):
    epoc_LexDelayRep_Aud = LabeledArray.fromfile(os.path.join(sf_dir,'epoc_LexDelayRep_Aud'))
else:
    stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
    stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

    fig_save_dir = os.path.join(LAB_root, 'D_Data','LexicalDecRepDelay','Baishen_Figs','LexicalDecRepDelay','group')
    if not os.path.exists(os.path.join(fig_save_dir)):
        os.mkdir(os.path.join(fig_save_dir))

    stats_save_root = os.path.join(stats_root_delay,'group')
    if not os.path.exists(os.path.join(stats_save_root)):
        os.mkdir(os.path.join(stats_save_root))

    if groupsTag=="LexDelay":
        epoc_LexDelayRep_Aud,_=gp.load_stats('zscore','Auditory_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=True)
        epoc_LexDelayRep_Aud.tofile(os.path.join(sf_dir,'epoc_LexDelayRep_Aud'))

# %% Select electrodes
loaded_data={}
for t_tag,t_range in zip(
        ('full','encode','delay'),
        ([-0.2,mean_word_len+auditory_decay+delay_len],[-0.2,mean_word_len+auditory_decay],[mean_word_len+auditory_decay,mean_word_len+auditory_decay+delay_len])
):
    for elec_grp,elec_idx in zip(
            ('Motor_delay','Auditory_delay','Sensorymotor_delay','Delay_only'),
            ('LexDelay_Motor_in_Delay_sig_idx','LexDelay_Auditory_in_Delay_sig_idx','LexDelay_Sensorimotor_in_Delay_sig_idx','LexDelay_DelayOnly_sig_idx')
    ):
        print(f'Now Doing {t_tag} {elec_grp}')
        m_chs = epoc_LexDelayRep_Aud.take(list(LexDelay_twin_idxes[elec_idx]), axis=1)
        m = m_chs.take(get_time_indexs(m_chs.labels[2], t_range[0], t_range[1]), axis=2)
        mixup(m, 0)
        gp.win_to_Rdataframe(m,os.path.join(sf_dir, f'epoc_LexDelayRep_Aud_{t_tag}_{elec_grp}'))