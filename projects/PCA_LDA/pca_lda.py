#%% Introduction
# This script is made for the comparison in HG traces among Repeat vs. YesNo, Delay vs. NoDelay, and Word vs. Nonword
import os

from practice_codes.Dec_2024_group_level.step5_compare_multitaper_4cons import font_size

script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\PCA_LDA\\pca_lda.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

import sys
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
from ieeg.decoding.decode import classes_from_labels,Decoder
from ieeg.calc.fast import mixup
from sklearn.metrics import ConfusionMatrixDisplay

# %% function block
delay_len=1.125 # average length from sound offset to Go onset
def get_time_indexs(time_str,start_float:float=0,end_float:float=delay_len):
    start_idx = np.searchsorted(time_str, start_float, side='left')
    end_idx = np.searchsorted(time_str, end_float, side='right')
    indices = list(range(start_idx, end_idx))
    return indices
# %% groups of patients
from pickle import FALSE

datasource='hg' # 'glm_(Feature)' or 'hg'
groupsTag="LexDelay"
sf_dir = '..\\results'
#groupsTag="LexDelay&LexNoDelay"

# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions
# For lexical delay task, whether run the data only with repeat tasks
trial_labels='CORRECT'

# %% Sort data and get significant electrode lists
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ieeg.arrays.label import LabeledArray

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

    epoc_LexDelayRep_Aud,_=gp.load_stats('zscore','Auditory_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=True)
    epoc_LexDelayRep_Aud.tofile(os.path.join(sf_dir,'epoc_LexDelayRep_Aud'))
    del epoc_LexDelayRep_Aud
    epoc_LexDelayRep_Aud=LabeledArray.fromfile(os.path.join(sf_dir,'epoc_LexDelayRep_Aud'))

# %% Select electrodes
with open(os.path.join('..','GLM','data', f'Lex_twin_idxes_hg.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)

for t_tag,t_range in zip(
        ('full','0_250ms','250_500ms','500_750ms','750_112ms'),
        ([0,delay_len],[0,0.25],[0.25,0.5],[0.5,0.75],[0.75,delay_len])
):
    for elec_grp,elec_idx in zip(
            ('Delay','Motor_delay','Auditory_delay','Sensorymotor_delay','Delay_only'),
            ('LexDelay_Delay_sig_idx','LexDelay_Motor_in_Delay_sig_idx','LexDelay_Auditory_in_Delay_sig_idx','LexDelay_Sensorimotor_in_Delay_sig_idx','LexDelay_DelayOnly_sig_idx')
    ):
        m_chs=epoc_LexDelayRep_Aud.take(list(LexDelay_twin_idxes[elec_idx]),axis=1)
        m=m_chs.take(get_time_indexs(m_chs.labels[2],t_range[0],t_range[1]),axis=2)

        cats, labels = classes_from_labels(m.labels[0],'/',2)
        mixup(m,0)
        # decoder = Decoder(cats, oversample=True, n_splits=5, n_repeats=100)
        decoder = Decoder(cats, n_splits=5, n_repeats=100)
        cm = decoder.cv_cm(m.__array__().swapaxes(0,1), labels, normalize='true')
        del m

        fig, ax = plt.subplots(figsize=(7,5))
        plt.rcParams['font.size'] = 20
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cats.keys())
        cm_plot=disp.plot(colorbar=False, ax=ax)
        im = cm_plot.im_
        im.set_clim(vmin=0.44, vmax=0.56)
        plt.title(f'{elec_grp}_{t_tag}')
        plt.tight_layout()
        plt.savefig(os.path.join(sf_dir,f'{elec_grp}_{t_tag}.tif'), dpi=300)
        plt.close()