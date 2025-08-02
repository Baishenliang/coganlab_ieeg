#%% Introduction
# This script is made for the comparison in HG traces among Repeat vs. YesNo, Delay vs. NoDelay, and Word vs. Nonword
import os
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
def check_non_nan_rows(arr):
    non_nan_rows_count = 0
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]).any():
            non_nan_rows_count += 1
            if non_nan_rows_count >= 3:
                return True
    return False

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
sf_dir = 'D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\PCA_LDA\\results'
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

m_chs=epoc_LexDelayRep_Aud.take(list(LexDelay_twin_idxes['LexDelay_DelayOnly_sig_idx']),axis=1)
m=m_chs.take(get_time_indexs(m_chs.labels[2]),axis=2)

cats, labels = classes_from_labels(m.labels[0],'/',2)
zero_indices = []
one_indices = []
for index, value in enumerate(labels):
    if value == 1:
        one_indices.append(index)
    else:
        zero_indices.append(index)
good_chs=[]
for i in range(0,len(m.labels[1])):
    ch=m.take([i], axis=1)
    a=ch.take(one_indices,axis=0).__array__()
    if not check_non_nan_rows(a):
        print(f'bad for word in chs {m.labels[1][i]}')
    b=ch.take(zero_indices,axis=0).__array__()
    if not check_non_nan_rows(b):
        print('bad for nonword in chs {m.labels[1][i]')
    if check_non_nan_rows(a) and check_non_nan_rows(b):
        good_chs.append(i)

m=m.take(good_chs,axis=1)
cats, labels = classes_from_labels(m.labels[0],'/',2)
mixup(m,0)
# decoder = Decoder(cats, oversample=True, n_splits=5, n_repeats=100)
decoder = Decoder(cats, n_splits=5, n_repeats=100)
cm = decoder.cv_cm(m.__array__().swapaxes(0,1), labels, normalize='true')
cm = np.mean(cm, axis=0)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cats.keys())
disp.plot(ax=ax)