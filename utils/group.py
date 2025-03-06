def load_stats(stat_type,con,contrast,stats_root_readID,stats_root_readdata):
    """
    Load patient level stats files (e.g., *.fif) for further group level analysis
    output is an ieeg LabeledArray

    e.g.:
    stat_type = 'mask'
    con = 'Auditory'
    contrast = 'ave'
    """
    import os
    import numpy as np
    import mne
    import glob
    from ieeg.arrays.label import LabeledArray

    match stat_type:
        case "zscore":
            fif_read = lambda f: mne.read_epochs(f, False, preload=True)
        case "power":
            fif_read = lambda f: mne.read_epochs(f, False, preload=True)
        case "significance":
            fif_read = mne.read_evokeds
        case "pval":
            fif_read = mne.read_evokeds
        case "mask":
            fif_read = mne.read_evokeds
        case "glm":
            fif_read = np.load

    subjs = [name for name in os.listdir(stats_root_readID) if os.path.isdir(os.path.join(stats_root_readID, name)) and name.startswith('D')]
    import warnings
    subjs = [subj for subj in subjs if subj != 'D0107' and subj != 'D0042']# and subj != 'D0028'] # problematic patients: 102 and 103: eeg electrodes, 107, plotting issues, 42: bad heading, each should be dealed with
    warnings.warn(f"The following subjects are not included: D0107 D0042")
    chs = []
    data_lst = []
    valid_subjs = []

    clean_root_readdata = stats_root_readdata.replace('stats', 'clean')

    for i, subject in enumerate(subjs):

        subj_gamma_stats_dir = os.path.join(stats_root_readdata, subject)


        subj_gamma_clean_dir = os.path.join(clean_root_readdata, f"sub-{subject}", "ieeg")

        if stat_type == 'glm':
            file_dir = os.path.join(subj_gamma_stats_dir, f'GLM_{con}_{contrast}.npy')
        else:
            file_dir = os.path.join(subj_gamma_stats_dir, f'{con}_{stat_type}-{contrast}.fif')
        # 对于GLM这里需要合并一下：电极还是要load fif的，但是mask/beta/p就是直接load的*.npy so

        if not os.path.exists(file_dir):
            continue
        else:
            valid_subjs.append(subject)

        # read patient data
        subj_dataset = fif_read(file_dir)

        match stat_type:
            case "zscore":
                subj_data_epo = subj_dataset._data
                subj_data = np.mean(subj_data_epo, axis=0)
                subj_chs = subj_dataset.ch_names
                if i == 0:
                    times = subj_dataset.times
            case "power":
                subj_data_epo = subj_dataset._data
                subj_data = np.mean(subj_data_epo, axis=0)
                subj_chs = subj_dataset.ch_names
                if i == 0:
                    times = subj_dataset.times
            case "significance":
                # Not yet tested, maybe wrong
                subj_data = subj_dataset[0].data
                subj_chs = subj_dataset[0].ch_names
                if i == 0:
                    times = subj_dataset[0].times
            case "pval":
                # Not yet tested, maybe wrong
                subj_data = subj_dataset[0].data
                subj_chs = subj_dataset[0].ch_names
                if i == 0:
                    times = subj_dataset[0].times
            case "mask":
                subj_data = subj_dataset[0].data
                subj_chs = subj_dataset[0].ch_names
                if i == 0:
                    times = subj_dataset[0].times
            case 'glm':
                subj_data = subj_dataset
                evoke_dir = os.path.join(subj_gamma_stats_dir, 'Auditory_inRep_mask-ave.fif')
                chs_time_dataset_forglm = mne.read_evokeds(evoke_dir)
                subj_chs = chs_time_dataset_forglm[0].ch_names
                if i == 0:
                    times = chs_time_dataset_forglm[0].times

        del subj_dataset # clear up memory

        # read original channel labels (before outlier and muscle channel removals)
        subj_chs_org_pattern = os.path.join(subj_gamma_clean_dir, f"*_acq-*_run-*_desc-clean_channels.tsv")
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

        good_labeled_chs = [f"{subject} {ch}" for ch in subj_chs]

        aligned_data,aligned_chs = align_channel_data(subj_data, good_labeled_chs, org_labeled_chs)

        data_lst.append(aligned_data)
        chs.extend(aligned_chs)

        # The following codes just do not take the outlier or muscle electrodes for considerations
        # This may (but most likely may not) affect comparing lexical delay and lexical no delay as they can have different bad electrodes.
        # subject = 'D' + subject[1:].lstrip('0')
        # good_labeled_chs_reformat = [f"{subject}-{ch}" for ch in subj_chs]
        # data_lst.append(subj_data)
        # chs.extend(good_labeled_chs_reformat)


    data_raw = np.concatenate(data_lst, axis=0)
    labels = [chs, times]
    data=LabeledArray(data_raw, labels)
    return data,valid_subjs

def sort_chs_by_actonset(data_in,win_len,time_range):
    """
    Selete channels with significant activation clusters (all mask==1 in any window with win_len) within a time range (time_range).
    Sort the significant channels according to the onset.

    e.g.:
    data: imported data in LabeledArray
    win_len = 0.2 (in seconds, the shortest sig cluster that consider effective)
    time_range = [-0.2, 1]
    """
    import numpy as np
    from ieeg.arrays.label import LabeledArray

    # %% get the onsets of the activation (an effective cluster is defined as 0.2s)
    times=data_in.labels[1]
    times = [float(i) for i in times]
    chs=data_in.labels[0]
    data=data_in.__array__()

    spf = 1 / (times[1] - times[0])  # Calculate the sampling frequency
    win = int(win_len * spf)  # Number of samples in 0.1 seconds

    onsets = {}

    for ch_idx, ch_name in enumerate(chs):
        ch_data = data[ch_idx]
        found = False

        search_start=np.argmin(np.abs(np.array(times) - time_range[0]))
        search_stop=np.argmin(np.abs(np.array(times) - time_range[1]))
        for start_idx in range(int(search_start),int(search_stop) - win + 1):
            win_data = ch_data[start_idx:start_idx + win]

            if np.all(win_data == 1):
                starting_time = times[start_idx]
                onsets[ch_name] = starting_time
                found = True
                break

        if not found:
            onsets[ch_name] = None  # No significant window found

    # %% select channels with significant activation clusters
    data_s = []
    chs_s = []
    chs_s_idx = [] # significant channels selected
    chs_s_all_idx = [0]*len(chs) # channel index with significant channels marked in 1
    onsets_s = []

    for ch_idx, ch_name in enumerate(chs):
        if onsets[ch_name] is not None:  # Check if the channel has a valid onset
            data_s.append(data[ch_idx])  # Add the channel data to the selected data list
            chs_s.append(ch_name)  # Add the channel name to the selected channel names list
            chs_s_idx.append(ch_idx)
            onsets_s.append(onsets[ch_name])
            chs_s_all_idx[ch_idx] = 1

    # Convert the selected data list to a numpy array
    data_s = np.array(data_s)

    # %% do the ranking
    sorted_indices = np.argsort(np.array(onsets_s))  # Get the indices that would sort the array
    data_s_sorted = data_s[sorted_indices]
    chs_s_sorted = [chs_s[i] for i in sorted_indices]
    onsets_s_sorted = [onsets_s[i] for i in sorted_indices]

    # %% return the data
    labels = [chs_s_sorted, times]
    data_out=LabeledArray(data_s_sorted, labels)
    # onset_out=LabeledArray(np.array(onsets_s_sorted), chs_s_sorted)
    return data_out,sorted_indices,chs_s_all_idx

def plot_chs(data_in,fig_save_dir_f):
    """
    plot the significant channels in a sorted order
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    times=data_in.labels[1]
    times = [float(i) for i in times]
    chs=data_in.labels[0]
    data=data_in.__array__()

    plt.figure(figsize=(2 ^ 15, 1 ^ 15))
    fig, ax = plt.subplots()
    ax.imshow(data, cmap='Reds')

    ch_gap = 20
    time_gap = 40

    channel_names = chs[::ch_gap]
    ax.set_yticks(range(0, len(channel_names) * ch_gap, ch_gap))
    ax.set_yticklabels(channel_names)
    time_stamps = [round(t, 3) for t in times[::time_gap]]
    ax.set_xticks(range(0, len(time_stamps) * time_gap, time_gap))
    ax.set_xticklabels(time_stamps)
    try:
        zero_time_index = np.where(np.array(times) == 0)[0][0]
    except Exception as e:
        times_array = np.array(times)
        zero_time_index = np.abs(times_array).argmin()
    ax.axvline(x=zero_time_index, color='black', linestyle='--', linewidth=1)
    fig.savefig(fig_save_dir_f, dpi=300)

def plot_brain(subjs,picks,chs_cols,label_every,fig_save_dir_f, **kwargs):
    subjs = ['D' + subj[1:].lstrip('0') for subj in subjs]
    from ieeg.viz.mri import plot_on_average
    fig3d = plot_on_average(subjs, picks=picks,color=chs_cols,hemi='split',
                            label_every=label_every, size=0.2, **kwargs)
    # fig3d.save_image(fig_save_dir_f)

def find_com_sig_chs(data1_labels, data1_sig_idx, data2_labels, data2_sig_idx):
    """
    Find common significant channels between two datasets.

    Args:
        data1_labels: Channel labels from first dataset
        data1_sig_idx: Significance indices from first dataset
        data2_labels: Channel labels from second dataset
        data2_sig_idx: Significance indices from second dataset

    Returns:
        common_sig_idx: Binary array marking common significant channels
        common_sig_labels: Labels of common significant channels
    """
    # Get significant channel labels from both datasets
    import numpy as np

    sig_channels1 = set([label for label, idx in zip(data1_labels, data1_sig_idx) if idx == 1])
    sig_channels2 = set([label for label, idx in zip(data2_labels, data2_sig_idx) if idx == 1])
    # Find common significant channels
    common_sig_channels = sig_channels1.intersection(sig_channels2)

    # Create binary index array for common significant channels
    common_sig_idx = np.zeros(len(data1_labels), dtype=int)
    for i, channel in enumerate(data1_labels):
        if channel in common_sig_channels:
            common_sig_idx[i] = 1

    return common_sig_idx

def align_channel_data(subj_data, good_labeled_chs, org_labeled_chs):
    """
    Aligns channel data by adding missing channels with NaN values.

    Parameters:
    -----------
    subj_data : numpy.ndarray
        Original subject data array with shape (n_channels, n_timepoints)
    good_labeled_chs : list
        List of current channel labels in the data
    org_labeled_chs : list
        List of all original channel labels

    Returns:
    --------
    numpy.ndarray
        Aligned data array with all original channels
    list
        Updated channel labels
    """
    import numpy as np

    if len(subj_data.shape) != 2:
        raise ValueError("Subject data must be 2-dimensional (channels x timepoints)")

    # Extract just the channel names from good_labeled_chs (removing subject prefix)
    current_chs = [ch.split()[-1] for ch in good_labeled_chs]

    # Get the number of timepoints from the original data
    n_timepoints = subj_data.shape[1]

    # Initialize the new data array with NaNs
    aligned_data = np.full((len(org_labeled_chs), n_timepoints), np.nan)

    # Create a mapping of current channels to their indices
    current_ch_indices = {ch: idx for idx, ch in enumerate(current_chs)}

    # Fill in the data for existing channels
    for new_idx, org_ch in enumerate(org_labeled_chs):
        if org_ch in current_ch_indices:
            aligned_data[new_idx] = subj_data[current_ch_indices[org_ch]]

    # Create new channel labels with subject prefix
    subject = good_labeled_chs[0].split()[0]  # Extract subject from first channel
    subject = 'D' + subject[1:].lstrip('0')
    aligned_chs = [f"{subject}-{ch}" for ch in org_labeled_chs]

    return aligned_data, aligned_chs

def plot_wave(data_in,sig_idx,con_label,col):

    import numpy as np
    import matplotlib.pyplot as plt

    times=data_in.labels[1]
    times = [float(i) for i in times]
    chs=data_in.labels[0]
    data=data_in.__array__()

    data_selected = np.full(np.shape(data), np.nan)
    for i in range(len(data_selected)):
        if sig_idx[i]==1:
            data_selected[i] = data[i]

    # Select a few key time points for labeling
    num_ticks = 6  # Adjust as needed
    tick_positions = np.linspace(0, len(times) - 1, num_ticks).astype(int)  # Select indices
    tick_labels = [times[a] for a in tick_positions]  # Get corresponding time labels

    # Compute the mean and SEM across trials while ignoring NaNs
    mean_waveform = np.nanmean(data_selected, axis=0)
    sem_waveform = np.nanstd(data_selected, axis=0) / np.sqrt(np.sum(~np.isnan(data_selected), axis=0))  # SEM ignoring NaNs

    # # Plot the mean waveform
    # plt.plot(times, mean_waveform, label=con_label, color=col)
    #
    # # Add shaded region for SEM
    # plt.fill_between(times, mean_waveform - sem_waveform, mean_waveform + sem_waveform, color=col, alpha=0.3)

    # Plot the mean waveform
    plt.plot(times, data_selected.transpose(), label=con_label, color=col)

    # Labels and title
    plt.axhline(0, color='k', linestyle='--', linewidth=1)  # Zero-line
    plt.xlabel('Time (s)')
    plt.xticks(tick_labels)  # Set fewer time labels
    plt.ylabel('Z-score')
    plt.legend()
    plt.show()
