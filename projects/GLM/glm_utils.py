# Module for glm processing

#%% Import python modules
import pandas as pd
import numpy as np
import patsy
import mne
import os
import glob
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from statsmodels.stats.outliers_influence import variance_inflation_factor


#%% Functions
def check_multicollinearity(X_i):
    """
    Computes the Variance Inflation Factor (VIF) for each feature in the given matrix
    to detect multicollinearity. Also, plots the correlation matrix as a heatmap.

    Parameters:
    X_i : numpy.ndarray
        Feature matrix where each column represents a feature.

    Returns:
    pd.DataFrame
        A DataFrame showing the VIF values for each feature.
    """

    # Ensure the input is a NumPy array
    if not isinstance(X_i, np.ndarray):
        raise ValueError("Input X_i must be a NumPy array")

    # Compute VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = [f"X{i}" for i in range(X_i.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(X_i, i) for i in range(X_i.shape[1])]

    # Plot correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(X_i, rowvar=False)  # Compute correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap="coolwarm", xticklabels=vif_data["Feature"],
                yticklabels=vif_data["Feature"], annot_kws={"size": 5})
    plt.title("Feature Correlation Matrix")
    plt.show()

    return vif_data

def glm_fifread(subjs,stats_root,fif_name):
    # %% start looping to load patients
    # Read dictionaries for acoustic, phonemic, and the other stimulus-based feature matrix

    data_list = []
    filtered_events_list = []
    chs = []

    phoneme_codes = pd.read_pickle("phoneme_one_hot_dict.pickle")
    acoustic_codes = pd.read_pickle("envelope_feature_dict.pickle")

    for i, subject in enumerate(subjs):
        print(f"Now do patent {subject}")

        # %% Load fif data
        subject_label_chs = 'D' + subject[1:].lstrip('0')
        subject = "sub-" + subject
        subject_No = subject.replace("sub-", "")
        subj_gamma_stats_dir = os.path.join(stats_root, subject_No)
        file_dir = os.path.join(subj_gamma_stats_dir, fif_name)
        epochs = mne.read_epochs(file_dir, False, preload=True)

        # %% Load events
        subj_gamma_clean_dir = os.path.join(clean_root, subject, 'ieeg')
        files = glob.glob(os.path.join(subj_gamma_clean_dir, '*acq-*_run-*_desc-clean_events.tsv'))
        files_sorted = sorted(files, key=lambda x: [int(i) for i in re.findall(r'acq-(\d+)_run-(\d+)', x)[0]])
        dfs = [pd.read_csv(f, sep='\t') for f in files_sorted]
        events_df = pd.concat(dfs, ignore_index=True)
        filtered_events_i = events_df[events_df['trial_type'].str.contains(event)
                                      & events_df['trial_type'].str.contains('CORRECT')
                                      & events_df['trial_type'].str.contains(task_Tag)].reset_index(drop=True)
        trial_split = filtered_events_i['trial_type'].str.split('/', expand=True)
        trial_split.columns = ['Stage', 'RepYesNo', 'Wordness', 'Stim', 'Correctness']
        filtered_events_i = pd.concat([filtered_events_i, trial_split], axis=1)
        for col in ['Stage', 'RepYesNo', 'Wordness', 'Stim', 'Correctness']:
            filtered_events_i[col] = filtered_events_i[col].astype('category')

        if subject == 'sub-D0102' and task_Tag == 'Repeat' and event == 'Auditory':
            filtered_events_i = filtered_events_i[:-1]

        # Get data
        if event == 'Auditory':
            data_i = epochs[f'Auditory_stim/{task_Tag}/CORRECT'].get_data()
        elif event == 'Resp':
            data_i = epochs[f'Resp/{task_Tag}/CORRECT'].get_data()
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
        if i == 0:
            vif_data = check_multicollinearity(X_i)
            print(vif_data['VIF'])

        feature_mat_i = np.repeat(X_i[:, np.newaxis, :], np.shape(data_i)[1], axis=1)

        # store data
        data_list.append(data_i)
        filtered_events_list.append(feature_mat_i)
        chs.append(chs_i)

def compute_r2_ch(x, y):
    # Run linear regression to get beta and R^2
    # x: observations * features
    # y: observations * times
    # will get r2 for each time point of the y: Y(t)=betas(t)X(t)+e
    # return: r2: times
    mask = ~np.isnan(y[:,0])
    y_clean = y[mask,:]
    x_clean = x[mask,:]
    coef,resi = np.linalg.lstsq(x_clean, y_clean, rcond=None)[:2]
    residual = resi if resi.size>0 else np.sum((y_clean - x_clean @ coef) ** 2,axis=0)
    r2 = 1-residual/(np.sum((y_clean - np.mean(y_clean, axis=0)) ** 2, axis=0))
    return r2

def compute_r2_loop(feature_mat_i,data_i):
    # loop through all the electrodes and run GLM
    # feature_mat_i: feature matrix, observations * channels * features
    # data_i: eeg data matrix, observations * channels * times
    # return r2_i: r2 matrix, channels * times
    _, n_channels_i, n_times = data_i.shape
    r2_i = np.full((n_channels_i, n_times), np.nan)
    for ch in range(n_channels_i):
        x = feature_mat_i[:, ch, :]
        y = data_i[:, ch, :]
        r2_i[ch,:] = compute_r2_ch(x,y)
    return r2_i


# Plot GLM
def plot_patient(data_in, title, colbar_lab,times):
    plt.figure(figsize=(10, 5))
    plt.imshow(data_in, aspect='auto', cmap='gray_r', interpolation='nearest')  # , vmin=-0.5e-5, vmax=0.5e-5)
    plt.colorbar(label=colbar_lab)
    plt.xlabel("Time points")
    plt.ylabel("Channels")
    plt.title(title)

    xticks = np.arange(0, len(times), 20)
    plt.xticks(ticks=xticks, labels=np.round(times[xticks], 2))
    plt.xlabel("Time (s)")
    #
    # yticks = np.arange(0, len(chs), 5)
    # plt.yticks(ticks=yticks, labels=[chs[i] for i in yticks])
    plt.ylabel("Channels")

    zero_time_index = np.argmin(np.abs(times - 0))
    plt.axvline(x=zero_time_index, color='red', linestyle='--', linewidth=1.5, label="Time = 0")

    plt.show()

def permutation_baishen_parallel(feature_mat_i, data_i, n_perms):
    n_obs = feature_mat_i.shape[0]
    # feature_mat_i: feature matrix, observations * channels * features
    # data_i: eeg data matrix, observations * channels * times
    def worker(_):
        perm_indices = np.random.permutation(n_obs)
        perm_feature_mat = feature_mat_i[perm_indices, :, :]
        r2_i = compute_r2_loop(perm_feature_mat, data_i)
        return r2_i

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

