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
from joblib import Parallel, delayed
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time

# Locations
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
clean_root = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "clean")
stats_root = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

bin_wins=[(-0.5, -0.2), (-0.2, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),(1.0, 1.5)]

#%% Functions

def fifread(event,stat,task_Tag,wordness,bsl_contrast=False,Comp_task='',bin:bool=False,preonset_bsl_correct:bool=False):

    from ieeg.calc import scaling

    fif_name=f'{event}_{stat}-epo.fif'

    # Read single patient stats
    if Comp_task=='Rep_selective' or Comp_task=='Del_selective':
        subjs = [name for name in os.listdir(stats_root_nodelay) if
                os.path.isdir(os.path.join(stats_root_nodelay, name)) and name.startswith('D')]
    else:
        subjs = [name for name in os.listdir(stats_root) if
                os.path.isdir(os.path.join(stats_root, name)) and name.startswith('D')]

    import warnings
    subjs = [subj for subj in subjs if subj != 'D0107' and subj != 'D0042' and subj != 'D0115' and subj != 'D0117']  # always exclude D0115 now but not in ther future to keep the repeat
    if task_Tag=='Yes_No':
        subjs = [subj for subj in subjs if subj != 'D0115']
    warnings.warn(f"The following subjects are not included: D0107 D0042")
    # start looping to load patients
    # Read dictionaries for acoustic, phonemic, and the other stimulus-based feature matrix

    data_list = []
    filtered_events_list = []
    chs = []

    phoneme_codes = pd.read_pickle("D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\phoneme_one_hot_dict.pickle")
    acoustic_codes = pd.read_pickle("D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\envelope_feature_dict.pickle")

    for i, subject in enumerate(subjs):
        print(f"Now do patient {subject}")

        # Load fif data
        subject_label_chs = 'D' + subject[1:].lstrip('0')
        subject = "sub-" + subject
        subject_No = subject.replace("sub-", "")
        subj_gamma_stats_dir = os.path.join(stats_root, subject_No)
        file_dir = os.path.join(subj_gamma_stats_dir, fif_name)
        epochs = mne.read_epochs(file_dir, False, preload=True)

        if preonset_bsl_correct:
            base_fif_name = f'Cue_inRep_{stat}-epo.fif'
            base_file_dir = os.path.join(subj_gamma_stats_dir, base_fif_name)
            base_epochs = mne.read_epochs(base_file_dir, False, preload=True)
            base = base_epochs.crop(tmin=-0.5, tmax=0)
            # If I input the mne epochs to scaling.rescalre,
            # It will run this instead:
            # def _(line: BaseEpochs, baseline: BaseEpochs,
            #       mode: str = 'mean', copy: bool = False, picks: list = 'data',
            #       verbose=None) -> Epochs:
            # The thing is that the axes will become (0,2), that is get the baseline across trials and time points
            # (one baseline for each electrode)
            # So I choose to use the np.array version:
            base_data=base.get_data()
            if subject == 'sub-D0102' and (task_Tag == 'Repeat' or Comp_task == 'Rep_YN' or Comp_task == 'YN_Rep') and event.split('_')[0] == 'Auditory' and wordness != 'Nonword':
                base_data=base_data[:-1]
            bsl_corr_epoch_data = scaling.rescale(epochs.get_data(), base_data, 'zscore', copy=True)
            epochs_bsl_corr = mne.EpochsArray(
                data=bsl_corr_epoch_data,
                info=epochs.info,
                events=epochs.events,
                tmin=epochs.tmin,
                event_id=epochs.event_id,  # Keep original event_ids
                verbose=False)
            epochs=epochs_bsl_corr

        if bin:
            epochs=get_windowed_epoch_data(epochs,preonset_bsl_correct=True)

        if bsl_contrast:
            fif_name_cue = f'Cue_inRep_{stat}-epo.fif'
            file_dir = os.path.join(subj_gamma_stats_dir, fif_name_cue)
            epochs_bsl = mne.read_epochs(file_dir, False, preload=True)
            epochs_bsl_crop = epochs_bsl.copy().crop(tmin=-0.5, tmax=0)

        if Comp_task=='Rep_YN' or Comp_task=='YN_Rep':
            fif_name_YN = f'Auditory_inYN_{stat}-epo.fif'
            file_dir = os.path.join(subj_gamma_stats_dir, fif_name_YN)
            epochs_YN = mne.read_epochs(file_dir, False, preload=True)

        if Comp_task == 'Rep_selective' or Comp_task == 'Del_selective':
            fif_name_Ndel_inRep = f'{event.split('_')[0]}_inRep_{stat}-epo.fif'
            fif_name_Ndel_inSilence = f'{event.split('_')[0]}_inSilence_{stat}-epo.fif'
            subj_gamma_stats_dir_Ndel = os.path.join(stats_root_nodelay, subject_No)
            fif_dir_Ndel_inRep = os.path.join(subj_gamma_stats_dir_Ndel, fif_name_Ndel_inRep)
            epochs_Ndel_inRep = mne.read_epochs(fif_dir_Ndel_inRep, False, preload=True)
            fif_dir_Ndel_inSilence = os.path.join(subj_gamma_stats_dir_Ndel, fif_name_Ndel_inSilence)
            epochs_Ndel_inSilence = mne.read_epochs(fif_dir_Ndel_inSilence, False, preload=True)

        if not bsl_contrast:
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

        # Get data
        if wordness == 'ALL':
            if event.split('_')[0] == 'Auditory':
                data_i = epochs[f'Auditory_stim/{task_Tag}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Resp':
                data_i = epochs[f'Resp/{task_Tag}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Go':
                data_i = epochs[f'Go/{task_Tag}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Cue':
                data_i = epochs[f'Cue/{task_Tag}/CORRECT'].get_data()
            if i == 0:
                times = epochs.times
            if bsl_contrast:
                data_i_bsl = epochs_bsl_crop[f'Cue/{task_Tag}/CORRECT'].get_data()
            if Comp_task=='Rep_YN' or Comp_task=='YN_Rep':
                data_i_YN = epochs_YN[f'Auditory_stim/Yes_No/CORRECT'].get_data()
            elif Comp_task == 'Rep_selective' or Comp_task == 'Del_selective':
                channels_epochs_set = set(epochs.ch_names)
                channels_rep_set = set(epochs_Ndel_inRep.ch_names)
                channels_silence_set = set(epochs_Ndel_inSilence.ch_names)
                common_channels = channels_epochs_set.intersection(channels_rep_set, channels_silence_set)
                epochs_list = [epochs, epochs_Ndel_inRep, epochs_Ndel_inSilence]
                for k, current_epochs in enumerate(epochs_list):
                    channels_to_drop = [ch for ch in current_epochs.ch_names if ch not in common_channels]
                    if channels_to_drop:
                        current_epochs.drop_channels(channels_to_drop)
                iscrop=True
                if iscrop:
                    t_min_comp_task = -0.5
                    t_max_comp_task = 1.5
                    epochs.crop(t_min_comp_task,t_max_comp_task)
                    if i == 0:
                        times = epochs.times
                    epochs_Ndel_inRep.crop(t_min_comp_task,t_max_comp_task)
                    epochs_Ndel_inSilence.crop(t_min_comp_task,t_max_comp_task)
                if event.split('_')[0] == 'Auditory':
                    data_i = epochs[f'Auditory_stim/{task_Tag}/CORRECT'].get_data()
                    data_i_Ndel_inRep = epochs_Ndel_inRep[f'Auditory_stim/Repeat/CORRECT'].get_data()
                    data_i_Ndel_inSilence = epochs_Ndel_inSilence[f'Auditory_stim/:=:/CORRECT'].get_data()
                else:
                    data_i = epochs[f'{event.split('_')[0]}/{task_Tag}/CORRECT'].get_data()
                    data_i_Ndel_inRep = epochs_Ndel_inRep[f'{event.split('_')[0]}/Repeat/CORRECT'].get_data()
                    data_i_Ndel_inSilence = epochs_Ndel_inSilence[f'{event.split('_')[0]}/:=:/CORRECT'].get_data()

        else:
            if event.split('_')[0] == 'Auditory':
                data_i = epochs[f'Auditory_stim/{task_Tag}/{wordness}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Resp':
                data_i = epochs[f'Resp/{task_Tag}/{wordness}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Go':
                data_i = epochs[f'Go/{task_Tag}/{wordness}/CORRECT'].get_data()
            elif event.split('_')[0] == 'Cue':
                data_i = epochs[f'Cue/{task_Tag}/{wordness}/CORRECT'].get_data()
            if i == 0:
                times = epochs.times
            if bsl_contrast:
                data_i_bsl = epochs_bsl_crop[f'Cue/{task_Tag}/{wordness}/CORRECT'].get_data()
        chs_i = epochs.ch_names
        chs_i = [f"{subject_label_chs}-{ch}" for ch in chs_i]

        # Handle the case of D0102
        # D0102 has a different number of trials for the last trial
        if subject == 'sub-D0102' and (task_Tag == 'Repeat' or Comp_task == 'Rep_YN' or Comp_task == 'YN_Rep') and event.split('_')[0] == 'Auditory' and wordness != 'Nonword':
            if not bsl_contrast:
                filtered_events_i = filtered_events_i[:-1]
            else:
                data_i_bsl = data_i_bsl[:-1]
        if bsl_contrast and subject == 'sub-D0102' and task_Tag == 'Repeat' and event.split('_')[0] == 'Resp' and wordness != 'Nonword':
            data_i_bsl = data_i_bsl[:-1]

        if bsl_contrast:
            data_i_bsl_mean = np.nanmean(data_i_bsl, axis=2, keepdims=True)
            data_i_bsl_mean_repeated = np.repeat(data_i_bsl_mean, np.shape(data_i)[2], axis=2)
            data_i = np.concatenate([data_i, data_i_bsl_mean_repeated], axis=0)
            bsl_dummy = np.concatenate([np.ones(np.shape(data_i_bsl_mean_repeated)[0]), np.zeros(np.shape(data_i_bsl_mean_repeated)[0])]).astype(int)#.reshape(-1, 1)
            X_i = np.column_stack([np.ones(np.shape(data_i)[0]), bsl_dummy])
        elif Comp_task=='Rep_YN' or Comp_task=='YN_Rep':
            # Feature matrix
            len_rep=np.shape(data_i)[0]
            data_i = np.concatenate([data_i, data_i_YN], axis=0)
            if Comp_task=='Rep_YN':
                Rep_YN_dummy = np.concatenate([np.ones(len_rep), np.zeros(np.shape(data_i_YN)[0])]).astype(int)#.reshape(-1, 1)
            elif Comp_task=='YN_Rep':
                Rep_YN_dummy = np.concatenate([np.zeros(len_rep), np.ones(np.shape(data_i_YN)[0])]).astype(int)#.reshape(-1, 1)
            X_i = np.column_stack([np.ones(np.shape(data_i)[0]), Rep_YN_dummy])
        elif Comp_task=='Rep_selective' or Comp_task=='Del_selective':
            # Feature matrix
            len_del_rep=np.shape(data_i)[0]
            len_ndel_rep=np.shape(data_i_Ndel_inRep)[0]
            len_ndel_jl=np.shape(data_i_Ndel_inSilence)[0]
            data_i = np.concatenate([data_i, data_i_Ndel_inRep,data_i_Ndel_inSilence], axis=0)
            if Comp_task=='Rep_selective':
                Rep_NRep_dummy = np.concatenate([np.ones(len_del_rep), np.ones(len_ndel_rep), np.zeros(len_ndel_jl)]).astype(int)#.reshape(-1, 1)
                X_i = np.column_stack([np.ones(np.shape(data_i)[0]), Rep_NRep_dummy])
            elif Comp_task=='Del_selective':
                Del_NDel_dummy = np.concatenate([np.ones(len_del_rep), np.zeros(len_ndel_rep), np.zeros(len_ndel_jl)]).astype(int)#.reshape(-1, 1)
                X_i = np.column_stack([np.ones(np.shape(data_i)[0]), Del_NDel_dummy])

        else:
            # Feature matrix
            word_dummy = (filtered_events_i.Wordness == "Word").astype(float)
            nonword_dummy = (filtered_events_i.Wordness == "Nonword").astype(float)
            phoneme_vectors = []
            acoustic_vectors = []
            for stim in filtered_events_i.Stim:
                phoneme_vectors.append(phoneme_codes[stim])
                acoustic_vectors.append(acoustic_codes[stim])
            X_i = np.column_stack([np.ones(np.shape(data_i)[0]), word_dummy, nonword_dummy, phoneme_vectors, acoustic_vectors])

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


def get_windowed_epoch_data(epochs, window_definitions: list=bin_wins,preonset_bsl_correct:bool=False):
    """
    Segments MNE Epochs data into specified time windows, calculates the mean
    of the data within each window, and returns the midpoints of those windows.
    """

    # Get the actual time points from the MNE Epochs object
    epoch_times = epochs.times

    # Get the raw data from the MNE Epochs object
    # Shape: (n_epochs, n_channels, n_times)
    epoch_data_raw = epochs.get_data()

    # Pre-allocate array for efficiency
    n_epochs, n_channels, _ = epoch_data_raw.shape
    n_windows = len(window_definitions)

    # Initialize the output array for means
    # It will store (n_epochs, n_channels, n_windows)
    window_means_final = np.full((n_epochs, n_channels, n_windows), np.nan)
    window_stds_final = np.full((n_epochs, n_channels, n_windows), np.nan)
    window_midpoints = np.array([(start + end) / 2 for start, end in window_definitions])

    # Iterate through each epoch
    for i_epoch in range(n_epochs):
        # Iterate through each channel
        for i_channel in range(n_channels):
            # Get the 1D time series data for the current epoch and channel
            channel_data = epoch_data_raw[i_epoch, i_channel, :]

            # Iterate through each defined time window
            for i_window, (start_time, end_time) in enumerate(window_definitions):
                # Find indices corresponding to the start and end times in epoch_times
                start_idx = np.searchsorted(epoch_times, start_time, side='left')
                end_idx = np.searchsorted(epoch_times, end_time, side='right')

                # Ensure indices are within valid bounds
                start_idx = max(0, start_idx)
                end_idx = min(len(epoch_times), end_idx)

                # Extract data for the current window
                data_in_window = channel_data[start_idx:end_idx]

                # Calculate mean if there's data, otherwise store NaN
                if data_in_window.size > 0:
                    window_means_final[i_epoch, i_channel, i_window] = np.nanmean(data_in_window)
                    window_stds_final[i_epoch, i_channel, i_window] = np.nanstd(data_in_window)

                # If data_in_window is empty, it remains np.nan due to initialization

    if preonset_bsl_correct:
        window_means_final -= window_means_final[:,:,0,np.newaxis]
        window_means_final /= window_stds_final[:,:,0,np.newaxis]

    new_events = np.column_stack((np.arange(n_epochs),
                                  np.zeros(n_epochs, dtype=int),
                                  epochs.events[:, 2])).astype(int)

    windowed_epochs = mne.EpochsArray(
        data=window_means_final,
        info=epochs.info,
        events=new_events,
        tmin=window_midpoints[0],
        event_id=epochs.event_id, # Keep original event_ids
        verbose=False
    )

    return windowed_epochs

def bsl_t_fdr(epc,bsl):
    import numpy as np
    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import fdrcorrection

    t_vals = np.zeros(epc.shape[1])
    p_vals = np.zeros(epc.shape[1])

    for i in range(epc.shape[1]):
        t_stat, p_val_two_tailed = ttest_rel(epc[:, i], bsl, nan_policy='omit')
        t_vals[i] = t_stat
        if t_stat > 0:
            p_vals[i] = p_val_two_tailed / 2
        else:
            p_vals[i] = 1 - (p_val_two_tailed / 2)
    _, p_vals_fdr = fdrcorrection(p_vals, alpha=0.05)
    return t_vals, p_vals,p_vals_fdr

def par_regress(filtered_events_list_i,feature_seleted,feature_controlled,data_i,glm_out,alphas,isresidual: bool=True):
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
    _, feature_mat_i_res = compute_r2_loop(feature_mat_i_ctr, np.r_[0:np.shape(feature_mat_i_ctr)[2]], feature_mat_i,glm_out,alphas,isresidual)
    # Y ~ X2@beta -> Yres
    _, data_i_res = compute_r2_loop(feature_mat_i_ctr, np.r_[0:np.shape(feature_mat_i_ctr)[2]],data_i,glm_out,alphas,isresidual)
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

def ridge_cv_alpha2score(X_clean, y_clean, alphas_to_test, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2s = []
    for alpha in alphas_to_test:
        fold_r2_scores = []
        for train_index, test_index in kf.split(X_clean):
            X_train, X_test = X_clean[train_index], X_clean[test_index]
            y_train, y_test = y_clean[train_index], y_clean[test_index]

            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(X_train, y_train)

            y_pred = ridge_model.predict(X_test)

            r2 = mean_squared_error(y_test, y_pred)
            fold_r2_scores.append(r2)

        r2s.append(np.mean(fold_r2_scores))

    return r2s

def compute_r2_ch_ridge(x, y,perm_feature_idx,isresidual,glm_out: str='beta_abs',alpha: float=np.nan):
    """
    Computes the global R^2 score using Ridge regression with Leave-One-Out
    cross-validation across all time points simultaneously.
    """
    mask = ~np.isnan(y[:, 0])
    y_clean = y[mask, :]
    x_clean = x[mask, :]
    if glm_out == 'alpha':
        ridge_model = RidgeCV(alphas=np.logspace(-6, 6, 10), cv=KFold(n_splits=5))
        ridge_model.fit(x_clean, y_clean)
    elif glm_out == 'cv_r2':
        alphas = np.logspace(-6, 6, 10)
        r2_scores = ridge_cv_alpha2score(x_clean, y_clean, alphas, n_splits=5)
    else:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(x_clean, y_clean)

    # Calculate residuals using the best model
    if isresidual or glm_out=='r2_series':
        y_pred = ridge_model.predict(x_clean)
        y_clean_res = y_clean - y_pred
        y_res = np.full_like(y, np.nan)
        y_res[mask, :] = y_clean_res
    else:
        y_res = np.full_like(y, np.nan)

    if glm_out=='beta_abs':
        # output beta values: as a function of times
        # sum of abs betas
        coef = np.nanmean(np.abs(ridge_model.coef_[:,perm_feature_idx]), axis=1)
    elif glm_out=='beta':
        # output beta values: as a function of times
        # sum of betas
        coef = np.nanmean(np.maximum(0, ridge_model.coef_[:, perm_feature_idx]), axis=1)
    elif glm_out=='r2_series':
        # r2 time-resolved
        residual = np.nansum(y_clean_res ** 2, axis=0)
        coef = 1 - residual / (np.nansum((y_clean - np.nanmean(y_clean, axis=0)) ** 2, axis=0))
    elif glm_out == 'cv_r2':
        coef = r2_scores
    elif glm_out == 'r2':
        # output r^2 values: one single value for a whole function
        coef = ridge_model.score(x_clean, y_clean)
    elif glm_out == 'alpha':
        # coef = ridge_model.alpha_
        coef = 2.15443469e+03
    return coef, y_res


def remove_and_impute_outliers_3d(data_matrix,subjtag):
    """
    Independently performs outlier removal and mean imputation for each channel
    and time point in a 3D data matrix (observations * channels * times).

    Outliers are detected using the 3 * IQR (Interquartile Range) method.
    Imputation is done using the mean of the non-outlier observations for that
    specific channel and time point.

    Args:
        data_matrix (np.ndarray): The input 3D data matrix with shape
                                  (observations, channels, times).

    Returns:
        np.ndarray: The processed data matrix where outliers have been
                    imputed with the mean.
    """

    # Create a copy of the data to avoid modifying the original input matrix
    processed_data = data_matrix.copy()

    n_observations, n_channels, n_times = data_matrix.shape

    print(f"Removing outlier of data with shape: {data_matrix.shape}")

    # Iterate through each channel
    for ch_idx in range(n_channels):
        # Iterate through each time point
        for time_idx in range(n_times):

            observations_at_point = data_matrix[:, ch_idx, time_idx]
            valid_obs_for_iqr = observations_at_point[~np.isnan(observations_at_point)]
            if len(valid_obs_for_iqr) < 4:
                 continue

            Q1 = np.percentile(valid_obs_for_iqr, 25)
            Q3 = np.percentile(valid_obs_for_iqr, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - (3 * IQR)
            upper_bound = Q3 + (3 * IQR)

            is_not_nan = ~np.isnan(observations_at_point)
            is_within_bounds = (observations_at_point >= lower_bound) & \
                               (observations_at_point <= upper_bound)

            is_good_value = is_not_nan & is_within_bounds

            good_values = observations_at_point[is_good_value]

            if len(good_values) == 0:
                continue

            imputation_value = np.mean(good_values)

            positions_to_impute = is_not_nan & (~is_within_bounds)

            processed_data[positions_to_impute, ch_idx, time_idx] = imputation_value

            if np.sum(positions_to_impute) > 0:
                print(f"{subjtag} Channel {ch_idx}, Time {time_idx}: Replaced {np.sum(positions_to_impute)} statistical outliers {data_matrix[positions_to_impute, ch_idx, time_idx]} with mean {imputation_value:.3f}")

    print("Outlier removal and imputation complete. Original NaNs preserved.")
    del data_matrix
    return processed_data

def temporal_smoothing(data_i, window_size=5):
    from scipy.ndimage import uniform_filter1d
    #data_i: eeg data matrix, observations * channels * times
    smoothed_data = uniform_filter1d(data_i, size=window_size, axis=2, mode='nearest')
    print(f'Temporal smoothing with {window_size*10} ms completed.')
    return smoothed_data

def compute_r2_loop(feature_mat_i,perm_feature_idx,data_i,glm_out,alphas,isresidual=False):
    # loop through all the electrodes and run GLM
    # feature_mat_i: feature matrix, observations * channels * features
    # data_i: eeg data matrix, observations * channels * times
    # return
    #   beta_i: r2 matrix, channels * times
    n_trials, n_channels_i, n_times = data_i.shape
    if glm_out == 'beta_abs' or glm_out == 'beta' or glm_out == 'r2_series':
        coef_i = np.full((n_channels_i, n_times), np.nan)
    elif glm_out == 'cv_r2':
        coef_i = np.full((n_channels_i, 10), np.nan)
    elif glm_out == 'r2' or glm_out == 'alpha':
        coef_i = np.full((n_channels_i, 1), np.nan)
    y_res_i = np.full((n_trials, n_channels_i, n_times), np.nan)
    for ch in range(n_channels_i):
        x = feature_mat_i[:, ch, :]
        y = data_i[:, ch, :]
        if glm_out == 'beta_abs' or glm_out == 'beta' or glm_out == 'r2' or glm_out == 'r2_series':
            alpha = alphas[ch]
        elif glm_out == 'alpha' or glm_out == 'cv_r2':
            alpha = np.nan
        coef_i[ch,:], y_res_i[:,ch,:]= compute_r2_ch_ridge(x,y,perm_feature_idx,isresidual,glm_out,alpha)
    return coef_i, y_res_i

def gen_ind_perms(n_obs):
    main_seed = int(time.time())
    master_rng = np.random.RandomState(main_seed)
    seed1 = master_rng.randint(0, 2**31 - 1)
    seed2 = master_rng.randint(0, 2**31 - 1)
    rng1 = np.random.RandomState(seed1)
    rng2 = np.random.RandomState(seed2)
    perm_indices1 = rng1.permutation(n_obs)
    perm_indices2 = rng2.permutation(n_obs)
    return perm_indices1, perm_indices2

def permutation_baishen_parallel(feature_mat_i, data_i, n_perms,perm_feature_idx,glm_out,alphas):
    n_obs = feature_mat_i.shape[0]
    # feature_mat_i: feature matrix, observations * channels * features
    # data_i: eeg data matrix, observations * channels * times
    def worker(_):
        perm_indices1,perm_indices2 = gen_ind_perms(n_obs)
        feature_mat_i_input=feature_mat_i.copy()
        feature_mat_i_perm = np.take(feature_mat_i, perm_indices1, axis=0)
        data_i_perm = np.take(data_i, perm_indices2, axis=0)
        feature_mat_i_input[:, :, perm_feature_idx] = np.take(feature_mat_i_perm, perm_feature_idx, axis=2)
        beta_i,_ = compute_r2_loop(feature_mat_i_input, perm_feature_idx,data_i_perm,glm_out,alphas)
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

def load_stats(event,stat,task_Tag,masktype,glm_fea,subjs,chs,times,wordness,glm_out: str='beta_abs',bin: bool=False):

    print(subjs)

    if bin:
        times = np.array([(start + end) / 2 for start, end in bin_wins])

    from ieeg.arrays.label import LabeledArray
    mask_lst = []
    stat_lst = []
    chs_lst = []

    for i, subject in enumerate(subjs):

        subj_mask = np.load(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data',f'{masktype} {subject} {event} {task_Tag} {wordness} {glm_fea}.npy'))
        subj_stat = np.load(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM','data',f'org_r2 {subject} {event} {task_Tag} {wordness} {glm_fea}.npy'))

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
        aligned_subj_mask, aligned_chs = gp.align_channel_data(subj_mask, chs_i, org_labeled_chs,glm_out)
        aligned_subj_stat, _ = gp.align_channel_data(subj_stat, chs_i, org_labeled_chs,glm_out)

        mask_lst.append(aligned_subj_mask)
        stat_lst.append(aligned_subj_stat)
        chs_lst.append(aligned_chs)

        # mask_lst.append(subj_mask)
        # stat_lst.append(subj_stat)
        # chs_lst.append(chs_i)

    mask_raw = np.concatenate(mask_lst, axis=0)
    stat_raw = np.concatenate(stat_lst, axis=0)
    chs_lst = np.concatenate(chs_lst, axis=0)

    if glm_out == 'beta_abs' or glm_out == 'beta':
        labels = [chs_lst, times]
        masks=LabeledArray(mask_raw, labels)
        stats=LabeledArray(stat_raw, labels)
    elif glm_out == 'r2':
        labels = [chs_lst, [0]]
        masks=LabeledArray(mask_raw, labels)
        stats=LabeledArray(stat_raw, labels)

    return masks,stats,subjs

def plot_mean_se_for_alphas(data_matrix, alphas_x_coords,fig_names):
    import matplotlib.pyplot as plt
    if data_matrix.shape[1] != len(alphas_x_coords):
        raise ValueError(
            f"Got data_matrix.shape[1]={data_matrix.shape[1]} and len(alphas_x_coords)={len(alphas_x_coords)}"
        )
    mean_values = np.mean(data_matrix, axis=0)
    std_dev_values = np.std(data_matrix, axis=0)
    n_samples_per_alpha = data_matrix.shape[0]
    if n_samples_per_alpha <= 1:
        se_values = np.zeros_like(std_dev_values)
    else:
        se_values = std_dev_values / np.sqrt(n_samples_per_alpha)
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        alphas_x_coords,
        mean_values,
        yerr=se_values,
        fmt='-o',
        capsize=5,
        label='Mean with SE',
        color='blue',
        ecolor='red',
        elinewidth=1
    )
    plt.xscale('log')
    plt.xlabel('Alpha Value (log scale)')
    plt.ylabel('Mean sqaured error')
    plt.title('Mean Metric Value vs. Alpha with Standard Error')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_names)