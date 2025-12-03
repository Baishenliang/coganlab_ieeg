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
# Get the Aud/SM/Mtr Delay electrodes without vWM:
LexDelay_twin_idxes['LexDelay_Sensorimotor_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Sensorimotor_sig_idx']-LexDelay_twin_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx']
LexDelay_twin_idxes['LexDelay_Auditory_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Aud_NoMotor_sig_idx']-LexDelay_twin_idxes['LexDelay_Auditory_in_Delay_sig_idx']
LexDelay_twin_idxes['LexDelay_Motor_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Motor_sig_idx']-LexDelay_twin_idxes['LexDelay_Motor_in_Delay_sig_idx']

import sys
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
from ieeg.calc.fast import mixup

# %% function block
mean_word_len=0.65#0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0 # a short period of time that we may assume auditory decay takes
delay_len=1.125 # average length from sound offset to Go onset
encoding_mode='univariate'
if encoding_mode=='univariate':
    cbind_subjs=False
elif encoding_mode=='multivariate':
    cbind_subjs=True

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

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

if groupsTag=="LexDelay":
    data_LexDelay_Aud,_=gp.load_stats('mask','Auditory_inRep','ave',stats_root_delay,stats_root_delay)
    elec_labels=data_LexDelay_Aud.labels[0]
    NoDelay_append_startings=False
    epoc_LexDelayRep_Aud,_=gp.load_stats('zscore','Auditory_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)

    data_LexDelayRep_Resp, _ = gp.load_stats('mask', 'Resp_inRep', contrast, stats_root_delay, stats_root_delay)
    epoc_LexDelayRep_Resp, _ = gp.load_stats('zscore', 'Resp_inRep', 'epo', stats_root_delay, stats_root_delay,
                                       trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)

    data_LexDelayRep_Go, _ = gp.load_stats('mask', 'Go_inRep', contrast, stats_root_delay, stats_root_delay)
    epoc_LexDelayRep_Go, _ = gp.load_stats('zscore', 'Go_inRep', 'epo', stats_root_delay, stats_root_delay,
                                       trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)

if groupsTag=="LexDelay&LexNoDelay":
    # epoc_LexDelayRep_Aud, _ = gp.load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_delay,trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)
    # epoc_LexNoDelay_Aud, _ = gp.load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_nodelay,trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)
    data_LexNoDelay_Aud,_=gp.load_stats('mask','Auditory_inRep','ave',stats_root_nodelay,stats_root_nodelay)
    elec_labels=data_LexNoDelay_Aud.labels[0]

    # Get the LexNoDelay data aligned to Cue
    epoc_LexNoDelay_Cue, _ = gp.load_stats('zscore', 'Cue_inRep', 'epo', stats_root_nodelay, stats_root_nodelay,trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)
    # Generate trial-based Auditory and Response onsets
    auditory_stim_dicts, resp_dicts=gp.get_onset_times(epoc_LexNoDelay_Cue,encoding_mode)
    with open(os.path.join(sf_dir,'LexNoDelay_Cue_auditory_stim_dicts.pkl'), 'wb') as f:
        pickle.dump(auditory_stim_dicts, f)
    with open(os.path.join(sf_dir,'LexNoDelay_Cue_resp_dicts.pkl'), 'wb') as f:
        pickle.dump(resp_dicts, f)

    # Get the LexDelay data aligned to Go
    epoc_LexDelay_Go, _ = gp.load_stats('zscore', 'Go_inRep', 'epo', stats_root_nodelay, stats_root_delay,
                                        trial_labels=trial_labels, keeptrials=True, cbind_subjs=cbind_subjs)
    auditory_stim_dicts, resp_dicts = gp.get_onset_times(epoc_LexDelay_Go, encoding_mode, align_to='Go',fileTag='BIDS-1.0_LexicalDecRepDelay')
    with open(os.path.join(sf_dir, 'LexDelay_Go_auditory_stim_dicts.pkl'), 'wb') as f:
        pickle.dump(auditory_stim_dicts, f)
    with open(os.path.join(sf_dir, 'LexDelay_Go_resp_dicts.pkl'), 'wb') as f:
        pickle.dump(resp_dicts, f)

    # Get the LexDelay data aligned to Cue
    epoc_LexDelay_Cue, _ = gp.load_stats('zscore', 'Cue_inRep', 'epo', stats_root_nodelay, stats_root_delay,
                                        trial_labels=trial_labels, keeptrials=True, cbind_subjs=cbind_subjs)
    auditory_stim_dicts, resp_dicts = gp.get_onset_times(epoc_LexDelay_Go, encoding_mode, align_to='Cue',fileTag='BIDS-1.0_LexicalDecRepDelay')
    with open(os.path.join(sf_dir, 'LexDelay_Cue_auditory_stim_dicts.pkl'), 'wb') as f:
        pickle.dump(auditory_stim_dicts, f)
    with open(os.path.join(sf_dir, 'LexDelay_Cue_resp_dicts.pkl'), 'wb') as f:
        pickle.dump(resp_dicts, f)

    NoDelay_append_startings=True

# %% Select electrodes
def rearrange_elects(elec_grps, elec_idxs, epoc, epoc_tag, win_len:int=10):
    t_range = [-0.5, 6]
    for elec_grp, elec_idx in zip(elec_grps, elec_idxs):
        print(f'Now Doing {elec_grp}')
        if encoding_mode == 'multivariate':
            m_chs = epoc.take(list(LexDelay_twin_idxes[elec_idx]), axis=1)
            m = m_chs.take(get_time_indexs(m_chs.labels[2], t_range[0], t_range[1]), axis=2)
            # mixup(m, 0)
        elif encoding_mode == 'univariate':
            epoc_dict = {}
            for s, epoc_s in epoc.items():
                s_ele_sel = [
                    item.replace(f'{s}-', '')
                    for item in elec_labels[list(LexDelay_twin_idxes[elec_idx])]
                    if item.startswith(s)
                ]
                if not s_ele_sel:
                    continue
                m_chs = epoc_s.take(s_ele_sel, axis=1)
                m = m_chs.take(get_time_indexs(m_chs.labels[2], t_range[0], t_range[1]), axis=2)
                epoc_dict[s] = m
            del m
            m = epoc_dict

        gp.win_to_Rdataframe(m, os.path.join(sf_dir, f'{epoc_tag}_{elec_grp}'), win_len=win_len, append_pho=False,
                             NoDelay_append_startings=NoDelay_append_startings)  # 100s for phoneme responses

loaded_data={}
if groupsTag=="LexDelay":
    elec_grps=('Motor_vWM','Auditory_vWM','Sensorymotor_vWM','Delay_only','Motor_novWM','Auditory_novWM','Sensorymotor_novWM','Wgw_p55b','Wgw_a55b')#,'Delay_only')
             #'Hickok_Spt','Hickok_lPMC','Hickok_lIPL','Hickok_lIFG')
    elec_idxs=('LexDelay_Motor_in_Delay_sig_idx','LexDelay_Auditory_in_Delay_sig_idx','LexDelay_Sensorimotor_in_Delay_sig_idx','LexDelay_DelayOnly_sig_idx',
               'LexDelay_Motor_not_in_Delay_sig_idx','LexDelay_Auditory_not_in_Delay_sig_idx','LexDelay_Sensorimotor_not_in_Delay_sig_idx','Wgw_p55b','Wgw_a55b')
             #'Hikock_Spt','Hikock_lPMC','Hikock_lIPL','Hikock_lIFG')
    for epoc,epoc_tag in zip((epoc_LexDelayRep_Aud,epoc_LexDelayRep_Go,epoc_LexDelayRep_Resp),
                             ('epoc_LexDelayRep_Aud','epoc_LexDelayRep_Go','epoc_LexDelayRep_Resp')):
        rearrange_elects(elec_grps, elec_idxs, epoc, epoc_tag, win_len=20)
elif groupsTag=="LexDelay&LexNoDelay":

    elec_grps_vWM = ('Motor_vWM', 'Auditory_vWM', 'Sensorymotor_vWM', 'Delay_only_vWM')
    elec_idxs_vWM = (
    'LexDelay_Motor_in_Delay_sig_idx', 'LexDelay_Auditory_in_Delay_sig_idx', 'LexDelay_Sensorimotor_in_Delay_sig_idx',
    'LexDelay_DelayOnly_sig_idx')
    elec_grps_novWM = ('Motor_novWM', 'Auditory_novWM', 'Sensorymotor_novWM')
    elec_idxs_novWM = (
    'LexDelay_Motor_novWM_sig_idx', 'LexDelay_Auditory_novWM_sig_idx', 'LexDelay_Sensorimotor_novWM_sig_idx')

    for elec_cat,data_base in zip(
            ('vWM','vWM','vWM','novWM','novWM','novWM'),
            ('NoDelay', 'Delay_Go', 'Delay_Cue','NoDelay', 'Delay_Go', 'Delay_Cue')
    ):

        match(elec_cat,data_base):
            case('vWM','NoDelay'):
                elec_grps=elec_grps_vWM
                elec_idxs=elec_idxs_vWM
                epoc=epoc_LexNoDelay_Cue
                epoc_tag='epoc_LexNoDelay_Cue'
            case('novWM','NoDelay'):
                elec_grps=elec_grps_novWM
                elec_idxs=elec_idxs_novWM
                epoc=epoc_LexNoDelay_Cue
                epoc_tag='epoc_LexNoDelay_Cue'
            case('vWM','Delay_Go'):
                elec_grps=elec_grps_vWM
                elec_idxs=elec_idxs_vWM
                epoc=epoc_LexDelay_Go
                epoc_tag='epoc_LexDelay_Go'
            case('novWM','Delay_Go'):
                elec_grps=elec_grps_novWM
                elec_idxs=elec_idxs_novWM
                epoc=epoc_LexDelay_Go
                epoc_tag='epoc_LexDelay_Go'
            case('vWM','Delay_Cue'):
                elec_grps=elec_grps_vWM
                elec_idxs=elec_idxs_vWM
                epoc=epoc_LexDelay_Cue
                epoc_tag='epoc_LexDelay_Cue'
            case('novWM','Delay_Cue'):
                elec_grps=elec_grps_novWM
                elec_idxs=elec_idxs_novWM
                epoc=epoc_LexDelay_Cue
                epoc_tag='epoc_LexDelay_Cue'
        # print('==========================')
        # print(elec_grps)
        # print(elec_idxs)
        # print(epoc_tag)
        rearrange_elects(elec_grps, elec_idxs, epoc, epoc_tag, win_len=0)

