# Module for glm processing

#%% Import python modules
import sys
import pandas as pd
import numpy as np
import patsy
import mne
import os
import glob
import re
import pandas as pd
import glm_validate_plot as glm_plot
from joblib import Parallel, delayed
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp

# Locations
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
clean_root = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "clean")
stats_root = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")

#%% Functions

def fifread(event,stat,task_Tag,wordness):

    fif_name=f'{event}_{stat}-epo.fif'

    # Read single patient stats
    subjs = [name for name in os.listdir(stats_root) if
            os.path.isdir(os.path.join(stats_root, name)) and name.startswith('D')]
    import warnings
    subjs = [subj for subj in subjs if
            subj != 'D0024' and subj != 'D0107' and subj != 'D0042' and subj != 'D0115' and subj != 'D0117' and subj != 'D0079' and subj != 'D0100']  # always exclude D0115 now but not in ther future to keep the repeat
    if task_Tag=='Yes_No':
        subjs = [subj for subj in subjs if subj != 'D0115']
    warnings.warn(f"The following subjects are not included: D0107 D0042")
    # start looping to load patients
    # Read dictionaries for acoustic, phonemic, and the other stimulus-based feature matrix

    data_list = []
    filtered_events_list = []
    chs = []

    phoneme_codes = pd.read_pickle("phoneme_one_hot_dict.pickle")
    acoustic_codes = pd.read_pickle("envelope_feature_dict.pickle")

    for i, subject in enumerate(subjs):
        print(f"Now do patient eeee {subject}")

        # Load fif data
        subject_label_chs = 'D' + subject[1:].lstrip('0')
        subject = "sub-" + subject
        subject_No = subject.replace("sub-", "")
        subj_gamma_stats_dir = os.path.join(stats_root, subject_No)
        file_dir = os.path.join(subj_gamma_stats_dir, fif_name)
        epochs = mne.read_epochs(file_dir, False, preload=True)

        # Load events
        subj_gamma_clean_dir = os.path.join(clean_root, subject, 'ieeg')
        files = glob.glob(os.path.join(subj_gamma_clean_dir, '*acq-*_run-*_desc-clean_events.tsv'))
        files_sorted = sorted(files, key=lambda x: [int(i) for i in re.findall(r'acq-(\d+)_run-(\d+)', x)[0]])
        dfs = [pd.read_csv(f, sep='\t') for f in files_sorted]
        events_df = pd.concat(dfs, ignore_index=True)
        if wordness=='ALL':
            filtered_events_i = events_df[events_df['trial_type'].str.contains(event.split('_')[0])
                                          & events_df['trial_type'].str.contains('CORRECT')
                                          & events_df['trial_type'].str.contains(task_Tag)].reset_index(drop=True)
        else:
            filtered_events_i = events_df[events_df['trial_type'].str.contains(event.split('_')[0])
                                          & events_df['trial_type'].str.contains('CORRECT')
                                          & events_df['trial_type'].str.contains(task_Tag)
                                          & events_df['trial_type'].str.contains(wordness)].reset_index(drop=True)

        trial_split = filtered_events_i['trial_type'].str.split('/', expand=True)
        trial_split.columns = ['Stage', 'RepYesNo', 'Wordness', 'Stim', 'Correctness']
        filtered_events_i = pd.concat([filtered_events_i, trial_split], axis=1)
        for col in ['Stage', 'RepYesNo', 'Wordness', 'Stim', 'Correctness']:
            filtered_events_i[col] = filtered_events_i[col].astype('category')

        if subject == 'sub-D0102' and task_Tag == 'Repeat' and event.split('_')[0] == 'Auditory' and wordness !='Nonword':
            filtered_events_i = filtered_events_i[:-1]

        # Get data
        if wordness == 'ALL':
            if event.split('_')[0] == 'Auditory':
                data_i = epochs[f'Auditory_stim/{task_Tag}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Resp':
                data_i = epochs[f'Resp/{task_Tag}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Go':
                data_i = epochs[f'Go/{task_Tag}/CORRECT'].get_data()
            if i == 0:
                times = epochs.times
        else:
            if event.split('_')[0] == 'Auditory':
                data_i = epochs[f'Auditory_stim/{task_Tag}/{wordness}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Resp':
                data_i = epochs[f'Resp/{task_Tag}/{wordness}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Go':
                data_i = epochs[f'Go/{task_Tag}/{wordness}/CORRECT'].get_data()
            if i == 0:
                times = epochs.times
        chs_i = epochs.ch_names
        chs_i = [f"{subject_label_chs}-{ch}" for ch in chs_i]

        # Feature matrix
        wordness_dummy = (filtered_events_i.Wordness == "Word").astype(float)
        phoneme_vectors = []
        acoustic_vectors = []
        for stim in filtered_events_i.Stim:
            phoneme_vectors.append(phoneme_codes[stim])
            acoustic_vectors.append(acoustic_codes[stim])
        X_i = np.column_stack([np.ones(np.shape(data_i)[0]), wordness_dummy, phoneme_vectors, acoustic_vectors])

        # Test Multicollinearity
        # if i == 0:
        #     vif_data = glm_plot.check_multicollinearity(X_i)
        #     print(vif_data['VIF'])

        feature_mat_i = np.repeat(X_i[:, np.newaxis, :], np.shape(data_i)[1], axis=1)

        # store data
        data_list.append(data_i)
        filtered_events_list.append(feature_mat_i)
        chs.append(chs_i)

    return subjs, data_list, filtered_events_list, chs, times

def par_regress(filtered_events_list_i,feature_seleted,feature_controlled,data_i):
    """
    Partial regression to control the contributions of unseleted features:
    X1 ~ X2@beta -> X1res
    Y ~ X2@beta -> Yres
    return Yres and X1res to run further Yres ~ X1s@beta
    """
    # Get the selected features (X1)
    feature_mat_i = filtered_events_list_i[:, :, feature_seleted]
    # Get the control geatures (X2)
    feature_mat_i_ctr = filtered_events_list_i[:, :, feature_controlled]
    # X1 ~ X2@beta -> X1res
    _, feature_mat_i_res = compute_r2_loop(feature_mat_i_ctr, np.r_[0:np.shape(feature_mat_i_ctr)[2]], feature_mat_i)
    # Y ~ X2@beta -> Yres
    _, data_i_res = compute_r2_loop(feature_mat_i_ctr, np.r_[0:np.shape(feature_mat_i_ctr)[2]],data_i)
    return feature_mat_i_res, data_i_res

def compute_r2_ch(x, y,perm_feature_idx):
    # Run linear regression to get beta and R^2
    # x: observations * features
    # y: observations * times
    # Will return the average of the absolute beta values of the perm_feature_idx features：
    # beta_fea: times
    mask = ~np.isnan(y[:,0])
    y_clean = y[mask,:]
    x_clean = x[mask,:]
    coef,resi = np.linalg.lstsq(x_clean, y_clean, rcond=None)[:2]
    y_clean_res = y_clean - x_clean @ coef
    residual = resi if resi.size>0 else np.sum(y_clean_res ** 2,axis=0)
    r2 = 1-residual/(np.sum((y_clean - np.mean(y_clean, axis=0)) ** 2, axis=0))
    y_res = np.full_like(y, np.nan)
    y_res[mask,:]=y_clean_res
    # beta = np.sqrt(np.sum(np.square(np.take(coef, perm_feature_idx[1:], axis=0)), axis=0)) # removed the intercept
    return r2,y_res


def temporal_smoothing(data_i, window_size=5):
    from scipy.ndimage import uniform_filter1d
    #data_i: eeg data matrix, observations * channels * times
    smoothed_data = uniform_filter1d(data_i, size=window_size, axis=2, mode='nearest')
    return smoothed_data

def compute_r2_loop(feature_mat_i,perm_feature_idx,data_i):
    # loop through all the electrodes and run GLM
    # feature_mat_i: feature matrix, observations * channels * features
    # data_i: eeg data matrix, observations * channels * times
    # return
    #   beta_i: r2 matrix, channels * times
    n_trials, n_channels_i, n_times = data_i.shape
    beta_i = np.full((n_channels_i, n_times), np.nan)
    y_res_i = np.full((n_trials, n_channels_i, n_times), np.nan)
    for ch in range(n_channels_i):
        x = feature_mat_i[:, ch, :]
        y = data_i[:, ch, :]
        beta_i[ch,:], y_res_i[:,ch,:]= compute_r2_ch(x,y,perm_feature_idx)
    return beta_i, y_res_i

def permutation_baishen_parallel(feature_mat_i, data_i, n_perms,perm_feature_idx):
    n_obs = feature_mat_i.shape[0]
    # feature_mat_i: feature matrix, observations * channels * features
    # data_i: eeg data matrix, observations * channels * times
    def worker(_):
        perm_indices = np.random.permutation(n_obs)
        feature_mat_i_input=feature_mat_i.copy()
        feature_mat_i_perm = np.take(feature_mat_i, perm_indices, axis=0)
        feature_mat_i_input[:, :, perm_feature_idx] = np.take(feature_mat_i_perm, perm_feature_idx, axis=2)
        beta_i,_ = compute_r2_loop(feature_mat_i_input, perm_feature_idx,data_i)
        del feature_mat_i_input,feature_mat_i_perm
        return beta_i

    results = Parallel(n_jobs=-5)(delayed(worker)(k) for k in range(n_perms))
    null_r2 = np.stack(results, axis=0)  # shape: (n_perms, channels, times)
    return null_r2

def aaron_perm_gt_1d(diff, axis=0):
    # https://ieeg-pipelines.readthedocs.io/en/latest/_modules/ieeg/calc/stats.html#time_perm_cluster
    # May use the bottlenet.rankdata in the future:
    # https: // bottleneck.readthedocs.io / en / latest / reference.html  # bottleneck.rankdata
    m = diff.shape[axis] - 1
    sorted_indices = diff.argsort(axis=axis)  # Get sorted indices
    proportions = np.arange(diff.shape[axis]) / m  # Create proportions array
    # Rearrange to match original order
    return proportions[sorted_indices.argsort(axis=axis)]

def load_stats(event,stat,task_Tag,masktype,glm_fea,subjs,chs,times,wordness):

    print(subjs)

    from ieeg.arrays.label import LabeledArray
    mask_lst = []
    stat_lst = []
    chs_lst = []

    for i, subject in enumerate(subjs):

        subj_mask = np.load(os.path.join('data',f'{masktype} {subject} {event} {task_Tag} {wordness} {glm_fea}.npy'))
        subj_stat = np.load(os.path.join('data',f'org_r2 {subject} {event} {task_Tag} {wordness} {glm_fea}.npy'))

        # read original channel labels (before outlier and muscle channel removals)
        subj_chs_org_pattern = os.path.join(clean_root, f"*sub-{subject}", 'ieeg', f"*_acq-*_run-*_desc-clean_channels.tsv")
        subj_chs_org_file_list = glob.glob(subj_chs_org_pattern)

        org_labeled_chs = []

        if subj_chs_org_file_list:
            subj_chs_org_file_path = subj_chs_org_file_list[0]
            with open(subj_chs_org_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines[1:]:
                    columns = line.strip().split('\t')
                    org_labeled_chs.append(columns[0])

        if subject == 'D0026':
            # Some awkard channels in D0026 lexical no delay
            org_labeled_chs = [ch for ch in org_labeled_chs if 'RPF' not in ch]
        chs_i=chs[i]
        chs_i = [chs_i[i].replace("-", " ") for i in range(len(chs_i))]
        aligned_subj_mask, aligned_chs = gp.align_channel_data(subj_mask, chs_i, org_labeled_chs)
        aligned_subj_stat, _ = gp.align_channel_data(subj_stat, chs_i, org_labeled_chs)

        mask_lst.append(aligned_subj_mask)
        stat_lst.append(aligned_subj_stat)
        chs_lst.append(aligned_chs)

        # mask_lst.append(subj_mask)
        # stat_lst.append(subj_stat)
        # chs_lst.append(chs_i)

    mask_raw = np.concatenate(mask_lst, axis=0)
    stat_raw = np.concatenate(stat_lst, axis=0)
    chs_lst = np.concatenate(chs_lst, axis=0)
    labels = [chs_lst, times]
    masks=LabeledArray(mask_raw, labels)
    stats=LabeledArray(stat_raw, labels)

    return masks,stats,subjs