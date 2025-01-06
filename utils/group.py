def load_stats(stat_type,con,contrast,stats_root):
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
    from ieeg.calc.mat import LabeledArray

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

    subjs = [name for name in os.listdir(stats_root) if os.path.isdir(os.path.join(stats_root, name)) and name.startswith('D')]
    #subjs = ["D0053", "D0054", "D0055", "D0057", "D0059", "D0063", "D0065", "D0066", "D0068", "D0069", "D0070", "D0071",
    #         "D0077", "D0079", "D0081", "D0094", "D0096", "D0101", "D0102", "D0103"]
    #subjs = [subj for subj in subjs if subj != 'D0107' and subj != 'D0023' ]
    chs = []
    data_lst = []

    for i, subject in enumerate(subjs):

        subj_gamma_stats_dir = os.path.join(stats_root, subject)

        file_dir = os.path.join(subj_gamma_stats_dir, f'{con}_{stat_type}-{contrast}.fif')
        subj_dataset = fif_read(file_dir)

        subj_data = subj_dataset[0].data
        subj_chs = subj_dataset[0].ch_names
        labeled_chs = [f"{subject} {ch}" for ch in subj_chs]

        data_lst.append(subj_data)
        chs.extend(labeled_chs)
        if i == 0:
            times = subj_dataset[0].times

    data_raw = np.concatenate(data_lst, axis=0)
    labels = [chs, times]
    data=LabeledArray(data_raw, labels)
    return data,subjs

def sort_chs_by_actonset(data_in,win_len,time_range):
    """
    Selete channels with significant activation clusters (all mask==1 in any window with win_len) within a time range (time_range).
    Sort the significant channels according to the onset.

    e.g.:
    data: imported data in LabeledArray
    win_len = 0.2 (in seconds, the shortest sig cluster that consider effective)
    time_range = [-0.2, 1]
    """
    from ieeg.calc.mat import LabeledArray
    import numpy as np
    from ieeg.calc.mat import LabeledArray

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

    ch_gap = 60
    time_gap = 100

    channel_names = chs[::ch_gap]
    ax.set_yticks(range(0, len(channel_names) * ch_gap, ch_gap))
    ax.set_yticklabels(channel_names)
    time_stamps = times[::time_gap]
    ax.set_xticks(range(0, len(time_stamps) * time_gap, time_gap))
    ax.set_xticklabels(time_stamps)
    try:
        zero_time_index = np.where(np.array(times) == 0)[0][0]
        ax.axvline(x=zero_time_index, color='black', linestyle='--', linewidth=1)
    except Exception as e:
        print('no zero time found')
    fig.savefig(fig_save_dir_f, dpi=300)

def plot_brain(subjs,picks,chs_cols):
    from ieeg.viz.mri import plot_on_average
    plot_on_average(subjs, picks=picks,color=chs_cols,hemi='lh',  size=0.35)