import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
from mne.stats import fdr_correction
import mne
import os
import glob
import re
import matplotlib.pyplot as plt

#%% Parameters
event='Auditory'
stat='power' # or 'zscore'
fif_name=f'{event}_{stat}-epo.fif'
task_Tag='Yes_No'

#%% Locations
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
clean_root = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "clean")
stats_root = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")

#%% Read single patient stats
subjs = [name for name in os.listdir(stats_root) if
         os.path.isdir(os.path.join(stats_root, name)) and name.startswith('D')]
import warnings
subjs = [subj for subj in subjs if
         subj != 'D0107' and subj != 'D0042']  # and subj != 'D0028']
if task_Tag=='Yes_No':
    subjs = [subj for subj in subjs if subj != 'D0115']
warnings.warn(f"The following subjects are not included: D0107 D0042")

#D102 event number does not match trial number

# %% start looping
for _, subject in enumerate(subjs):
    print(f"Now do patent {subject}")

    #%% Load fif data
    subject = "sub-"+subject
    subject_No = subject.replace("sub-", "")
    subj_gamma_stats_dir = os.path.join(stats_root, subject_No)
    file_dir = os.path.join(subj_gamma_stats_dir, fif_name)
    epochs=mne.read_epochs(file_dir, False, preload=True)

    #%% Load events
    subj_gamma_clean_dir = os.path.join(clean_root, subject,'ieeg')
    files = glob.glob(os.path.join(subj_gamma_clean_dir, '*acq-*_run-*_desc-clean_events.tsv'))
    files_sorted = sorted(files, key=lambda x: [int(i) for i in re.findall(r'acq-(\d+)_run-(\d+)', x)[0]])
    dfs = [pd.read_csv(f, sep='\t') for f in files_sorted]
    events_df = pd.concat(dfs, ignore_index=True)
    filtered_events = events_df[events_df['trial_type'].str.contains(event)
                                & events_df['trial_type'].str.contains('CORRECT')
                                & events_df['trial_type'].str.contains(task_Tag)].reset_index(drop=True)
    trial_split = filtered_events['trial_type'].str.split('/', expand=True)
    trial_split.columns = ['Stage', 'RepYesNo', 'Wordness', 'Stim', 'Correctness']
    filtered_events = pd.concat([filtered_events, trial_split], axis=1)
    for col in ['Stage', 'RepYesNo', 'Wordness', 'Stim', 'Correctness']:
        filtered_events[col] = filtered_events[col].astype('category')

    if subject == 'sub-D0102' and task_Tag=='Repeat':
        filtered_events = filtered_events[:-1]
    #%% GLM

    # Get data
    if event=='Auditory':
        data = epochs[f'Auditory_stim/{task_Tag}/CORRECT'].get_data()
    elif event=='Resp':
        data = epochs[f'Resp/{task_Tag}/CORRECT'].get_data()
    times = epochs.times
    chs = epochs.ch_names
    # Loop for each patient

    # Build design matrix
    pvalues_list = np.full((len(chs), len(times)), np.nan)
    beta_list = np.full((len(chs), len(times)), np.nan)
    fdr_mask = np.full((len(chs), len(times)), np.nan)

    for ch_idx, ch in enumerate(chs):
        for t_idx, t in enumerate(times):
            HG_t = data[:, ch_idx, t_idx]
            df = pd.DataFrame({'Y': HG_t, 'Wordness': filtered_events.Wordness})
            formula = 'Y ~ C(Wordness)'
            Y, X = patsy.dmatrices(formula, data=df, return_type='dataframe')
            model = sm.OLS(Y, X, missing='drop')
            results = model.fit()
            pvalues_list[ch_idx, t_idx] = results.pvalues.iloc[1]
            beta_list[ch_idx, t_idx] = results.params.iloc[1]
        fdr_mask[ch_idx, :],_ = fdr_correction(pvalues_list[ch_idx, :])

    # Save GLM
    np.save(os.path.join(subj_gamma_stats_dir, f"GLM_{event}_{task_Tag}_{stat}_Wordness_Pvals.npy"), pvalues_list)
    np.save(os.path.join(subj_gamma_stats_dir, f"GLM_{event}_{task_Tag}_{stat}_Wordness_Betas.npy"), beta_list)
    np.save(os.path.join(subj_gamma_stats_dir, f"GLM_{event}_{task_Tag}_{stat}_Wordness_fdrmasks.npy"), fdr_mask)

    # Plot GLM
    # plt.figure(figsize=(10, 5))
    # plt.imshow(fdr_mask, aspect='auto', cmap='gray_r', interpolation='nearest')
    # plt.colorbar(label="Significance (0 = not significant, 1 = significant)")
    # plt.xlabel("Time points")
    # plt.ylabel("Channels")
    # plt.title("Significance Matrix")
    #
    # xticks = np.arange(0, fdr_mask.shape[1], 20)
    # plt.xticks(ticks=xticks, labels=np.round(times[xticks], 2))
    # plt.xlabel("Time (s)")
    #
    # yticks = np.arange(0, len(chs), 5)
    # plt.yticks(ticks=yticks, labels=[chs[i] for i in yticks])
    # plt.ylabel("Channels")
    #
    # zero_time_index = np.argmin(np.abs(times - 0))
    # plt.axvline(x=zero_time_index, color='red', linestyle='--', linewidth=1.5, label="Time = 0")
    #
    # plt.show()