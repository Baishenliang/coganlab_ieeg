"""Group-analysis working copy for subsequent bipolar adaptation.

Copied from lexical_group_stats.py through the Lex_idxes save block.
Loads bipolar FIFs and aligns virtual channels before group classification.
"""
# %% start
from pathlib import Path
import sys

# Resolve shared utils and projects imports when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# %% groups of patients
from pickle import FALSE
from matplotlib_venn import venn3
import seaborn as sns
import matplotlib.ticker as mticker

datasource='hg' # 'glm_(Feature)' or 'hg'
groupsTag="LexDelay"
#groupsTag="LexDelay&LexNoDelay"

# %% define condition and load data
get_atlaslabels_from_ecogRecon = False # whether get atlas labels for each electrode, which is used for later analysis of the distribution of electrodes in different ROIs. If True, it will take a long time to run the code. So we set it to False after we get the labels and save them in the utils folder.
stat_type='mask'
contrast='ave' # average, not contrasting different conditions
#contrast='ave_YN_Rep' # contrasting yesno to repetition
#contrast='ave_Rep_YN' # contrasting repetition to yesno
#contrast='ave_W_NW' # contrasting word to nonword trials only in repetition
#contrast='ave_NW_W' # contrasting nonword to word trials only in repetition

# For lexical delay task, whether run the data only with repeat tasks
#Delayseleted=''
Delayseleted = '_inRep'
trial_labels='CORRECT'

# Parameters from the lexical delay task

# Strong Auditory response set
mean_word_len=0.65#0.65 # from utils/lexdelay_get_stim_length.m
auditory_decay=0 # a short period of time that we may assume auditory decay takes
delay_len=1.125 # average length from sound offset to Go onset

# mean_word_len=0.15#0.62 # from utils/lexdelay_get_stim_length.m
# auditory_decay=0 # a short period of time that we may assume auditory decay takes
# delay_len=1.525 # average length from sound offset to Go onset

motor_prep_win=[-0.75,0] #May try -200ms - 9ms later # get windows for motor preparation (0.1s to avoid high gamma filter leakage)
motor_resp_win=[0,1] # get windows for motor response (0.75s to avoid too much auditory feedback)
go_resp_win=[0, 0.75] # speech preparation signals from Go onset. 0.25s cut
pre_stimonset_win=[-0.5,0]
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

MotorPrep_col = [1.0, 0.0784, 0.5765] # Motor prepare
Sensorimotor_col = [1, 0, 0]  # Sensorimotor
Auditory_col = [0, 1, 0]  # Auditory
Delay_col = [1, 0.65, 0]  # Delay
Motor_col = [0, 0, 1]  # Motor
WGW_p55b_col=[0.74901961, 0.25098039, 0.74901961] # WGW 55b
WGW_a55b_col=[0, 0.5, 0.5] # WGW a55b
Sensorimotor_Delay_col = Sensorimotor_col#[1, 0, 1]  # Sensorimotor-Delay
Auditory_Delay_col = Auditory_col#[1, 1, 0]  # Auditory-Delay
Delay_Motor_col = Motor_col#[0, 1, 1]  # Delay-Motor

# %% Sort data and get significant electrode lists
import os
import pickle
import numpy as np
import pandas as pd
from utils.group import bsliang_add_connecting_lines,generate_neuro_publication_plot,get_roi_subj_matrix,read_sex_age_and_stats,get_subj_elec_idx, load_stats, sort_chs_by_actonset, plot_chs, plot_brain, plot_wave,set2arr, chs2atlas, atlas2_hist, plot_sig_roi_counts, get_sig_elecs_keyword, hickok_roi_sphere, get_sig_roi_counts, plot_roi_counts_comparison, sort_chs_by_actonset_combined, select_electrodes,onsets2col,elegroup_strip, create_gradient, process_and_plot_roi_labels,analyze_vwm_hemispheric_interaction
from utils.global_func import get_coor
import matplotlib.pyplot as plt
import projects.GLM.glm_utils as glm


def align_bipolar_arrays(arrays):
    """Use the same channel intersection/order for every condition and statistic."""
    from ieeg.arrays.label import LabeledArray
    first = next(iter(arrays.values()))
    common = list(first.labels[0])
    for name, arr in arrays.items():
        labels = list(arr.labels[0])
        if len(set(labels)) != len(labels):
            raise ValueError(f'Duplicate channels in {name}')
        common = [label for label in common if label in set(labels)]
    if not common:
        raise ValueError('No shared bipolar channels across loaded conditions')
    aligned = {}
    for name, arr in arrays.items():
        lookup = {label: i for i, label in enumerate(arr.labels[0])}
        idx = [lookup[label] for label in common]
        print(f'{name}: retaining {len(common)}/{len(lookup)} shared channels')
        aligned[name] = LabeledArray(np.asarray(arr)[idx], [common, arr.labels[1]])
    # Each mask must match its own waveform time axis (different events may differ).
    for name, arr in aligned.items():
        counterpart = name.replace('data_', 'epoc_', 1)
        if name.startswith('data_') and counterpart in aligned:
            times = np.asarray(arr.labels[1], dtype=float)
            other = np.asarray(aligned[counterpart].labels[1], dtype=float)
            if times.shape != other.shape or not np.allclose(times, other):
                raise ValueError(f'Mask/epoch time axes differ: {name}')
    return aligned


def get_bipolar_coords(labels, max_retries=3, retry_delay=2.0):
    """Get group-space midpoints from both reconstructed endpoints, without fitting."""
    import time

    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError('max_retries must be a nonnegative integer')
    if not np.isfinite(retry_delay) or retry_delay < 0:
        raise ValueError('retry_delay must be finite and nonnegative')
    pairs = []
    endpoints = {}
    for full in labels:
        parts = str(full).split('-')
        if len(parts) != 3:
            raise ValueError(f'Expected subject-anode-cathode label: {full}')
        subj, a, b = parts
        pairs.append((subj, a, b))
        endpoints.setdefault(subj, []).extend([f'{subj}-{a}', f'{subj}-{b}'])
    lookup = {}
    for subj, channels in endpoints.items():
        channels = list(dict.fromkeys(channels))
        for attempt in range(max_retries + 1):
            try:
                coords = get_coor(channels, 'group', interpolate=False)
                break
            except OSError as exc:
                if getattr(exc, 'winerror', None) != 1006:
                    raise
                if attempt == max_retries:
                    exc.add_note(f'{subj}: coordinate loading failed after '
                                 f'{max_retries + 1} attempts')
                    raise
                print(f'{subj}: WinError 1006; retry {attempt + 1}/{max_retries} '
                      f'in {retry_delay:g}s', flush=True)
                time.sleep(retry_delay)
        lookup.update({(row.subj, row.label): np.array([row.x, row.y, row.z], dtype=float)
                       for row in coords.itertuples()})
        print(f'{subj}: loaded reconstruction coordinates', flush=True)
    rows = []
    for subj, a, b in pairs:
        xyz = (lookup.get((subj, a), np.full(3, np.nan))
               + lookup.get((subj, b), np.full(3, np.nan))) / 2
        rows.append([subj, f'{a}-{b}', *xyz])
    result = pd.DataFrame(rows, columns=['subj', 'label', 'x', 'y', 'z'])
    if not np.isfinite(result[['x', 'y', 'z']].to_numpy()).all():
        raise ValueError('Missing endpoint coordinates for bipolar ROI assignment')
    return result


if datasource != 'hg':
    raise ValueError('This bipolar script currently supports datasource="hg" only')
if get_atlaslabels_from_ecogRecon:
    raise ValueError('chs2atlas requires separate bipolar endpoint adaptation; keep this option False')

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

fig_save_dir = os.path.join(LAB_root, 'D_Data','LexicalDecRepDelay','Baishen_Figs','LexicalDecRepDelay','group_bipolar')
if not os.path.exists(os.path.join(fig_save_dir)):
    os.mkdir(os.path.join(fig_save_dir))

stats_save_root = os.path.join(stats_root_delay,'group_bipolar')
if not os.path.exists(os.path.join(stats_save_root)):
    os.mkdir(os.path.join(stats_save_root))

if groupsTag=="LexDelay":

    if datasource=='hg':

        data_LexDelay_Aud,subjs=load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_delay,stats_root_delay, reference='bipolar')
        #data_LexDelay_Cue, _ = load_stats(stat_type, 'Cue'+Delayseleted, contrast, stats_root_delay, stats_root_delay)
        data_LexDelay_Delay, _ = load_stats(stat_type, 'Delay'+Delayseleted, contrast, stats_root_delay, stats_root_delay, reference='bipolar')
        data_LexDelay_Go, _ = load_stats(stat_type, 'Go'+Delayseleted, contrast, stats_root_delay, stats_root_delay, reference='bipolar')
        data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_delay, stats_root_delay, reference='bipolar')

        # Get the ROI of labels
        if get_atlaslabels_from_ecogRecon:
            ch_labels_roi,ch_labels=chs2atlas(subjs,data_LexDelay_Aud.labels[0])

        epoc_LexDelay_Aud,_=load_stats('zscore','Auditory'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels, reference='bipolar')
        #poc_LexDelay_Cue,_=load_stats('zscore','Cue'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels)
        epoc_LexDelay_Delay,_=load_stats('zscore','Delay'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels, reference='bipolar')
        epoc_LexDelay_Go,_=load_stats('zscore','Go'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels, reference='bipolar')
        epoc_LexDelay_Resp,_=load_stats('zscore','Resp'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels, reference='bipolar')

        if trial_labels == 'Word':
            epoc_LexDelay_Aud_nonword,_=load_stats('zscore','Auditory'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels='Nonword', reference='bipolar')
            epoc_LexDelay_Resp_nonword,_=load_stats('zscore','Resp'+Delayseleted,'epo',stats_root_delay,stats_root_delay,trial_labels='Nonword', reference='bipolar')

    elif datasource.split('_')[0]=='glm':
        subjs, _, _, chs, times = glm.fifread('Auditory_inRep', 'zscore', 'Repeat','ALL')
        data_LexDelay_Aud,epoc_LexDelay_Aud,_=glm.load_stats('Auditory'+Delayseleted,'zscore','Repeat','cluster_mask',datasource.split('_')[1],subjs,chs,times,'ALL')
        _, _, _, _, times = glm.fifread('Resp_inRep', 'zscore', 'Repeat','ALL')
        data_LexDelay_Resp,epoc_LexDelay_Resp,_=glm.load_stats('Resp'+Delayseleted,'zscore','Repeat','cluster_mask',datasource.split('_')[1],subjs,chs,times,'ALL')
        if get_atlaslabels_from_ecogRecon:
            ch_labels_roi, ch_labels = chs2atlas(subjs, data_LexDelay_Aud.labels[0])

elif groupsTag=="LexDelay&LexNoDelay":

    # first get the patient inform from no delay tasks and then extract the corresponding
    data_LexDelay_Aud,subjs=load_stats(stat_type,'Auditory'+Delayseleted,contrast,stats_root_nodelay,stats_root_delay, reference='bipolar')
    data_LexNoDelay_Aud,_=load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_nodelay, reference='bipolar')
    # data_LexNoDelay_Silence_Aud,_=load_stats(stat_type,'Auditory_inSilence',contrast,stats_root_nodelay,stats_root_nodelay)

    # Get the ROI of labels
    if get_atlaslabels_from_ecogRecon:
        ch_labels_roi, ch_labels = chs2atlas(subjs, data_LexDelay_Aud.labels[0])

    data_LexDelay_Resp, _ = load_stats(stat_type, 'Resp'+Delayseleted, contrast, stats_root_nodelay, stats_root_delay, reference='bipolar')
    data_LexNoDelay_Resp, _ = load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay, reference='bipolar')

    epoc_LexDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_delay,trial_labels=trial_labels, reference='bipolar')
    epoc_LexNoDelay_Aud,_=load_stats('zscore','Auditory_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels, reference='bipolar')
    # epoc_LexNoDelay_Silence_Aud,_=load_stats('zscore','Auditory_inSilence','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

    data_LexDelay_Go, _ = load_stats(stat_type, 'Go' + Delayseleted, contrast, stats_root_nodelay, stats_root_delay, reference='bipolar')
    epoc_LexDelay_Go, _ = load_stats('zscore', 'Go' + Delayseleted, 'epo', stats_root_nodelay, stats_root_delay,
                                     trial_labels=trial_labels, reference='bipolar')

    data_LexDelay_Cue, _ = load_stats(stat_type, 'Cue' + Delayseleted, contrast, stats_root_nodelay, stats_root_delay, reference='bipolar')
    epoc_LexDelay_Cue, _ = load_stats('zscore', 'Cue' + Delayseleted, 'epo', stats_root_nodelay, stats_root_delay,
                                      trial_labels =trial_labels, reference='bipolar')

    data_LexNoDelay_Cue, _ = load_stats(stat_type, 'Cue_inRep', contrast, stats_root_nodelay, stats_root_nodelay, reference='bipolar')
    epoc_LexNoDelay_Cue,_=load_stats('zscore','Cue_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels, reference='bipolar')

    epoc_LexDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_delay,trial_labels=trial_labels, reference='bipolar')
    epoc_LexNoDelay_Resp,_=load_stats('zscore','Resp_inRep','epo',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels, reference='bipolar')

array_names = [f'{kind}_LexDelay_{event}' for kind in ('data', 'epoc')
               for event in ('Aud', 'Delay', 'Go', 'Resp')]
if groupsTag == 'LexDelay&LexNoDelay':
    array_names = [f'{kind}_{task}_{event}' for kind in ('data', 'epoc')
                   for task, events in [('LexDelay', ('Aud', 'Go', 'Resp', 'Cue')),
                                        ('LexNoDelay', ('Aud', 'Resp', 'Cue'))]
                   for event in events]
elif trial_labels == 'Word':
    array_names += ['epoc_LexDelay_Aud_nonword', 'epoc_LexDelay_Resp_nonword']
loaded_arrays = {name: globals()[name] for name in array_names}
globals().update(align_bipolar_arrays(loaded_arrays))
subjs = list(dict.fromkeys('D' + str(label).split('-', 1)[0][1:].zfill(4)
                          for label in data_LexDelay_Aud.labels[0]))

if groupsTag=="LexNoDelay":
    chs_coor=get_bipolar_coords(data_LexNoDelay_Aud.labels[0])
else:
    chs_coor=get_bipolar_coords(data_LexDelay_Aud.labels[0])
hickok_roi_labels, hickok_roi_sig_idx, hickok_roi_atlas_labels =hickok_roi_sphere(chs_coor)
# Note the here the Spt, lPMC, and lIFG electrodes are not conjunction with Delay electrodes. They are just electrodes, all electrodes.
_,*_ = process_and_plot_roi_labels(hickok_roi_labels, hickok_roi_atlas_labels)

#%% Get sorted electrodes
Lex_idxes = dict()
if "LexDelay" in groupsTag:

    # sort the data according to the onset within a time range (Full)
    data_LexDelay_sorted,_,_,LexDelay_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[-10,10])

    # Get pre-onset activations:
    # data_LexDelay_sorted_preonset,_,_,LexDelay_sig_idx_preonset,*_ = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,pre_stimonset_win)

    # (Auditory)
    data_LexDelay_Aud_sorted,_,_,LexDelay_Aud_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[0,mean_word_len+auditory_decay])

    # (Delay)
    data_LexDelay_Delay_sorted,_,_,LexDelay_Delay_sig_idx,*_=sort_chs_by_actonset(data_LexDelay_Aud,epoc_LexDelay_Aud,cluster_twin,[mean_word_len+auditory_decay,mean_word_len+auditory_decay+delay_len])

    # (Go)
    data_LexDelay_Go_sorted, _,_, LexDelay_Go_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Go, epoc_LexDelay_Go, cluster_twin, go_resp_win)

    # (Motor response)
    data_LexDelay_Motor_Resp_sorted, _, _, LexDelay_Motor_Resp_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Resp, epoc_LexDelay_Resp, cluster_twin, motor_resp_win)

    # (Motor prepare)
    data_LexDelay_Motor_Prep_sorted, _, _, LexDelay_Motor_Prep_sig_idx,*_ = sort_chs_by_actonset(data_LexDelay_Resp,epoc_LexDelay_Resp, cluster_twin, motor_prep_win)
    LexDelay_Motor_Prep_sig_idx = LexDelay_Go_sig_idx & LexDelay_Motor_Prep_sig_idx

    # Channel selection: Auditory nomotor electrodes (auditory window:1, motor prep: 0)
    LexDelay_Aud_NoMotor_sig_idx = LexDelay_Aud_sig_idx - LexDelay_Motor_Prep_sig_idx

    # Channel selection: Sensorimotor electrodes (auditory window:1, motor prep: 1)
    LexDelay_Sensorimotor_sig_idx = LexDelay_Aud_sig_idx & LexDelay_Motor_Prep_sig_idx

    # Channel selection: Sensory OR motor electrodes (auditory window:1 or motor prep: 1 or motor resp: 1)
    LexDelay_Sensory_OR_Motor_sig_idx = LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx

    # Channel selection: All sig electrodes (auditory window:1 or delay: 1 or motor prep: 1 or motor resp: 1)
    LexDelay_all_sig_idx = LexDelay_Aud_sig_idx | LexDelay_Delay_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx

    # Channel selection: Motor electrodes (auditory window:0, motor resp: 1)
    #LexDelay_Motor_sig_idx = LexDelay_Motor_Resp_sig_idx - LexDelay_Aud_sig_idx
    LexDelay_Motor_sig_idx = LexDelay_Motor_Prep_sig_idx - LexDelay_Aud_sig_idx

    # Channel selection: Delay only electrodes (delay electrodes ,with: auditory window:0, motor prep: 0, motor resp: 0)
    #LexDelay_DelayOnly_sig_idx = LexDelay_Delay_sig_idx - (LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx)
    LexDelay_DelayOnly_sig_idx = LexDelay_Delay_sig_idx - (LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx)
    # Channel selection: Auditory electrodes in Delay electrodes
    LexDelay_Auditory_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Aud_NoMotor_sig_idx

    # Channel selection: Motor electrodes in Delay electrodes
    LexDelay_Motor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Motor_sig_idx

    # Motor_prep only
    LexDelay_Motorprep_Only_sig_idx = (LexDelay_Motor_Prep_sig_idx - (LexDelay_Aud_NoMotor_sig_idx | LexDelay_Sensorimotor_sig_idx | LexDelay_Motor_sig_idx | LexDelay_DelayOnly_sig_idx))

    # Channel selection: Sensorimotor electrodes in Delay electrodes
    LexDelay_Sensorimotor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Sensorimotor_sig_idx

    # Others
    # LexDelay_Other_sig_idx = ((LexDelay_Aud_sig_idx | LexDelay_Delay_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx)
    #                           - (LexDelay_Aud_NoMotor_sig_idx | LexDelay_Sensorimotor_sig_idx | LexDelay_Motor_sig_idx | LexDelay_DelayOnly_sig_idx))

    Lex_idxes['LexDelay_Aud_NoMotor_sig_idx']=LexDelay_Aud_NoMotor_sig_idx
    Lex_idxes['LexDelay_Sensorimotor_sig_idx']=LexDelay_Sensorimotor_sig_idx
    Lex_idxes['LexDelay_Motor_sig_idx']=LexDelay_Motor_sig_idx
    Lex_idxes['LexDelay_Delay_sig_idx']=LexDelay_Delay_sig_idx
    Lex_idxes['LexDelay_DelayOnly_sig_idx']=LexDelay_DelayOnly_sig_idx
    Lex_idxes['LexDelay_Auditory_in_Delay_sig_idx']=LexDelay_Auditory_in_Delay_sig_idx
    Lex_idxes['LexDelay_Auditory_not_in_Delay_sig_idx']=LexDelay_Aud_NoMotor_sig_idx-LexDelay_Auditory_in_Delay_sig_idx
    Lex_idxes['LexDelay_all_sig_idx']=LexDelay_all_sig_idx
    Lex_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx']=LexDelay_Sensorimotor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Sensorimotor_not_in_Delay_sig_idx']=LexDelay_Sensorimotor_sig_idx-LexDelay_Sensorimotor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Motor_in_Delay_sig_idx']=LexDelay_Motor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Motor_not_in_Delay_sig_idx']=LexDelay_Motor_sig_idx-LexDelay_Motor_in_Delay_sig_idx
    Lex_idxes['LexDelay_Motorprep_Only_sig_idx']=LexDelay_Motorprep_Only_sig_idx

    del data_LexDelay_sorted, data_LexDelay_Aud_sorted, data_LexDelay_Delay_sorted, data_LexDelay_Motor_Prep_sorted, data_LexDelay_Motor_Resp_sorted

    # Get the labels of all the motor electrodes in the temporal lobe
    Mot_elec_in_tmp=get_sig_elecs_keyword(data_LexDelay_Aud,LexDelay_Motor_sig_idx,'T')
if "LexNoDelay" in groupsTag:

    # (Auditory)
    data_LexNoDelay_Aud_sorted,_,_,LexNoDelay_Aud_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Aud,epoc_LexNoDelay_Aud, cluster_twin,[0,mean_word_len+auditory_decay])
    _,_,_,LexNoDelay_Aud_narraw_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Aud,epoc_LexNoDelay_Aud, cluster_twin,[0,50])

    # (Motor prepare)
    data_LexNoDelay_Motor_Prep_sorted,_,_,LexNoDelay_Motor_Prep_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, motor_prep_win)

    # (Motor response)
    data_LexNoDelay_Motor_Resp_sorted,_,_,LexNoDelay_Motor_Resp_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, motor_resp_win)
    _,_,_,LexNoDelay_Motor_narraw_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp, epoc_LexNoDelay_Resp, cluster_twin, [0,50])

    # (NoDelay Silence trials Whole win: Encoding)
    # data_LexNoDelay_Silence_Encode_sorted,_,_,LexNoDelay_Silence_Encode_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Silence_Aud,epoc_LexNoDelay_Silence_Aud, cluster_twin,[0,mean_word_len+auditory_decay])

    # (NoDelay Silence trials Whole win: Delay)
    # data_LexNoDelay_Silence_Del_sorted,_,_,LexNoDelay_Silence_Del_sig_idx,*_ = sort_chs_by_actonset(data_LexNoDelay_Silence_Aud,epoc_LexNoDelay_Silence_Aud, cluster_twin,[mean_word_len+auditory_decay,10])

    # (Encoding electrodes without Delay)
    # LexNoDelay_Silence_Encode_Only_sig_idx = LexNoDelay_Silence_Encode_sig_idx.difference(LexNoDelay_Silence_Del_sig_idx)

    # Channel selection: Auditory nomotor electrodes (auditory window:1, motor prep: 0)
    LexNoDelay_Aud_NoMotor_sig_idx = LexNoDelay_Aud_sig_idx - LexNoDelay_Motor_Prep_sig_idx

    # Channel selection: Sensorimotor electrodes (auditory window:1, motor prep: 1)
    LexNoDelay_Sensorimotor_sig_idx = LexNoDelay_Aud_sig_idx & LexNoDelay_Motor_Prep_sig_idx

    # Channel selection: Sensory OR motor electrodes (auditory window:1 or motor prep: 1 or motor resp: 1)
    LexNoDelay_Sensory_OR_Motor_sig_idx = LexNoDelay_Aud_sig_idx | LexNoDelay_Motor_Prep_sig_idx | LexNoDelay_Motor_Resp_sig_idx

    # Channel selection: All sig electrodes (auditory window:1 or delay: 1 or motor prep: 1 or motor resp: 1)
    LexNoDelay_all_sig_idx = LexNoDelay_Aud_sig_idx | LexNoDelay_Motor_Prep_sig_idx | LexNoDelay_Motor_Resp_sig_idx

    # Channel selection: Motor electrodes (auditory window:0, motor resp: 1)
    LexNoDelay_Motor_sig_idx = LexNoDelay_Motor_Resp_sig_idx - LexNoDelay_Aud_sig_idx

    # (Motor prep only)
    LexNoDelay_Motor_Prep_Only_sig_idx = LexNoDelay_Motor_Prep_sig_idx - (LexNoDelay_Aud_sig_idx | LexNoDelay_Motor_sig_idx)

    Lex_idxes['LexNoDelay_Aud_NoMotor_sig_idx'] = LexNoDelay_Aud_NoMotor_sig_idx
    Lex_idxes['LexNoDelay_Sensorimotor_sig_idx'] = LexNoDelay_Sensorimotor_sig_idx
    Lex_idxes['LexNoDelay_Sensory_OR_Motor_sig_idx'] = LexNoDelay_Sensory_OR_Motor_sig_idx
    Lex_idxes['LexNoDelay_all_sig_idx'] = LexNoDelay_all_sig_idx
    Lex_idxes['LexNoDelay_Motor_sig_idx'] = LexNoDelay_Motor_sig_idx
    Lex_idxes['LexNoDelay_Aud_sig_idx'] = LexNoDelay_Aud_sig_idx
    Lex_idxes['LexNoDelay_Motor_Resp_sig_idx'] = LexNoDelay_Motor_Resp_sig_idx
    Lex_idxes['LexNoDelay_Motor_narraw_sig_idx'] = LexNoDelay_Motor_narraw_sig_idx
    Lex_idxes['LexNoDelay_Aud_narraw_sig_idx'] = LexNoDelay_Aud_narraw_sig_idx
    # Lex_idxes['LexNoDelay_Silence_Encode_sig_idx'] = LexNoDelay_Silence_Encode_sig_idx
    # Lex_idxes['LexNoDelay_Silence_Encode_Only_sig_idx'] = LexNoDelay_Silence_Encode_Only_sig_idx
    # Lex_idxes['LexNoDelay_Silence_Del_sig_idx'] = LexNoDelay_Silence_Del_sig_idx

    if "LexDelay" in groupsTag:

        # Get Delay electrodes in No Delay Repeat tasks:
        data_LexNoDelay_Aud_DelDel=select_electrodes(data_LexNoDelay_Aud,LexDelay_Delay_sig_idx)
        data_LexNoDelay_Resp_DelDel=select_electrodes(data_LexNoDelay_Resp,LexDelay_Delay_sig_idx)
        epoc_LexNoDelay_Aud_DelDel=select_electrodes(epoc_LexNoDelay_Aud,LexDelay_Delay_sig_idx)
        epoc_LexNoDelay_Resp_DelDel=select_electrodes(epoc_LexNoDelay_Resp,LexDelay_Delay_sig_idx)
        _, _, _, LexNoDelay_Aud_DelDel_all,*_ = sort_chs_by_actonset(data_LexNoDelay_Aud_DelDel,epoc_LexNoDelay_Aud_DelDel,cluster_twin, [-0.1,10])
        _, _, _, LexNoDelay_Aud_DelDel_aud,*_ = sort_chs_by_actonset(data_LexNoDelay_Aud_DelDel,epoc_LexNoDelay_Aud_DelDel,cluster_twin, [0,mean_word_len])
        _, _, _, LexNoDelay_Aud_DelDel_mtrprep,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp_DelDel, epoc_LexNoDelay_Resp_DelDel, cluster_twin, motor_prep_win)
        _, _, _, LexNoDelay_Aud_DelDel_mtr,*_ = sort_chs_by_actonset(data_LexNoDelay_Resp_DelDel, epoc_LexNoDelay_Resp_DelDel, cluster_twin, motor_resp_win)
        delsm_full=set(range(1, len(LexDelay_Delay_sig_idx)+1))
        print(f'Auditory resp elec. in NoDelay Rep {len(LexNoDelay_Aud_DelDel_aud)}, {len(LexNoDelay_Aud_DelDel_aud) / len(delsm_full)}')
        print(f'Prob. Motorprep resp elec. NoDelay Rep {len(LexNoDelay_Aud_DelDel_mtrprep - (LexNoDelay_Aud_DelDel_aud | LexNoDelay_Aud_DelDel_mtr))}, {len(LexNoDelay_Aud_DelDel_mtrprep - (LexNoDelay_Aud_DelDel_aud | LexNoDelay_Aud_DelDel_mtr)) / len(delsm_full)}')
        print(f'Motor resp elec. NoDelay Rep {len(LexNoDelay_Aud_DelDel_mtr - LexNoDelay_Aud_DelDel_aud)}, {len(LexNoDelay_Aud_DelDel_mtr - LexNoDelay_Aud_DelDel_aud) / len(delsm_full)}')
        print(f'Silent elec. in NoDelay Rep {len(delsm_full-LexNoDelay_Aud_DelDel_all)}, {len(delsm_full-LexNoDelay_Aud_DelDel_all) / len(delsm_full)}')

    # May do in_Silence electrodes later
Lex_idxes['groupsTag']=groupsTag

Spt_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['Spt']
lPMC_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['lPMC']
lIPL_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['lIPL']
lIFG_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['lIFG']
Wgw_p55b_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['Wgw_p55b']
Wgw_a55b_sig_idx = LexDelay_all_sig_idx & hickok_roi_sig_idx['Wgw_a55b']

Lex_idxes['Hikock_Spt']=Spt_sig_idx
Lex_idxes['Hikock_lPMC']=lPMC_sig_idx
# Lex_idxes['Hikock_lIPL']=lIPL_sig_idx
Lex_idxes['Hikock_lIFG']=lIFG_sig_idx
# Lex_idxes['Wgw_p55b']=Wgw_p55b_sig_idx
# Lex_idxes['Wgw_a55b']=Wgw_a55b_sig_idx

with open(Path(__file__).resolve().parent / f'Lex_twin_idxes_{datasource}_bipolar.npy', "wb") as f:
    pickle.dump(Lex_idxes, f)

#%% store active electrodes and subjs
output_dir = REPO_ROOT / "projects" / "active_electrodes" / "bipolar"
os.makedirs(output_dir, exist_ok=True)

sig_indices_dict = {
    "Aud": LexDelay_Aud_sig_idx,
    "Delay": LexDelay_Delay_sig_idx,
    "Go": LexDelay_Go_sig_idx,
    "Motor_Resp": LexDelay_Motor_Resp_sig_idx,
}

all_labels = data_LexDelay_Aud.labels[0]

for name, sig_idx in sig_indices_dict.items():
    active_labels = [all_labels[idx] for idx in sig_idx]
    df = pd.DataFrame(active_labels, columns=["full_label"])

    if not df.empty:
        df[["subj", "electrode"]] = df["full_label"].str.split("-", n=1, expand=True)
        output_df = df[["subj", "electrode"]]
    else:
        output_df = pd.DataFrame(columns=["subj", "electrode"])

    file_path = os.path.join(output_dir, f"{name}.csv")
    output_df.to_csv(file_path, index=False)
