from pyqtgraph.util.cprint import color

def load_stats(stat_type,con,contrast,stats_root_readID,stats_root_readdata,split_half=0):
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
    subjs = [subj for subj in subjs if subj != 'D0107' and subj != 'D0042' and subj != 'D0115' and subj != 'D0117']# and subj != 'D0028'] # problematic patients: 102 and 103: eeg electrodes, 107, plotting issues, 42: bad heading, each should be dealed with
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

        if not os.path.exists(file_dir):
            continue
        else:
            valid_subjs.append(subject)

        # read patient data
        subj_dataset = fif_read(file_dir)

        match stat_type:
            case "zscore":
                subj_data_epo = subj_dataset._data
                match split_half:
                    case 0:
                        subj_data = np.mean(subj_data_epo, axis=0)
                    case 1:
                        half = subj_data_epo.shape[0] // 2
                        subj_data = np.std(subj_data_epo[:half], axis=0)
                    case 2:
                        half = subj_data_epo.shape[0] // 2
                        subj_data = np.std(subj_data_epo[half:], axis=0)
                subj_chs = subj_dataset.ch_names
                if i == 0:
                    times = subj_dataset.times
            case "power":
                subj_data_epo = subj_dataset._data
                match split_half:
                    case 0:
                        subj_data = np.mean(subj_data_epo, axis=0)
                    case 1:
                        half = subj_data_epo.shape[0] // 2
                        subj_data = np.std(subj_data_epo[:half], axis=0)
                    case 2:
                        half = subj_data_epo.shape[0] // 2
                        subj_data = np.std(subj_data_epo[half:], axis=0)
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

def sel_subj_data(data_in,chs_idx):
    from ieeg.arrays.label import LabeledArray
    times=data_in.labels[1]
    chs=data_in.labels[0]
    data=data_in.__array__() #data: chs*times
    data_sel=data[list(chs_idx)]
    chs_sel=chs[list(chs_idx)]
    labels = [chs_sel, times]
    data_out=LabeledArray(data_sel, labels)
    return data_out

def sort_chs_by_actonset(mask_in,data_in,win_len,time_range):
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
    mask=mask_in.__array__()

    spf = 1 / (times[1] - times[0])  # Calculate the sampling frequency
    win = int(win_len * spf)  # Number of samples in 0.1 seconds

    onsets = {}

    search_start = np.argmin(np.abs(np.array(times) - time_range[0]))
    search_stop = np.argmin(np.abs(np.array(times) - time_range[1]))

    for ch_idx, ch_name in enumerate(chs):
        ch_mask = mask[ch_idx]
        found = False

        for start_idx in range(int(search_start),int(search_stop) - win + 1):
            win_mask = ch_mask[start_idx:start_idx + win]

            if np.all(win_mask == 1):
                starting_time = times[start_idx]
                onsets[ch_name] = starting_time
                found = True
                break

        if not found:
            onsets[ch_name] = None  # No significant window found

    # %% select channels with significant activation clusters
    mask_s = []
    data_s = []
    data_us = np.full([np.shape(data)[0],np.shape(data)[1]],np.nan)
    chs_s = []
    chs_s_idx = []  # significant channels selected
    chs_s_all_idx = set()  # Use a set to store selected channel indices
    onsets_s = []

    for ch_idx, ch_name in enumerate(chs):
        onset = onsets.get(ch_name)  # Get the onset, avoiding repeated dictionary lookups
        if onset is not None:  # Check if the channel has a valid onset
            mask_s.append(mask[ch_idx])
            data_s.append(data[ch_idx])  # Add the channel data to the selected data list
            chs_s.append(ch_name)  # Add the channel name to the selected channel names list
            chs_s_idx.append(ch_idx)
            onsets_s.append(onset)
            chs_s_all_idx.add(ch_idx)  # Add index to the set
            data_us[ch_idx,:]=np.where(mask[ch_idx], data[ch_idx], np.nan) # the data_us contained all the channels and makred the inactive channels as nan

    # Convert the selected data list to a numpy array
    mask_s = np.array(mask_s)
    data_s = np.array(data_s)

    # %% do the ranking
    sorted_indices = np.argsort(np.array(onsets_s))  # Get the indices that would sort the array
    mask_s_sorted = mask_s[sorted_indices]
    data_s_sorted = data_s[sorted_indices]
    data_s_sorted = np.where(mask_s_sorted == 1, data_s_sorted, np.nan)
    chs_s_sorted = [chs_s[i] for i in sorted_indices]
    onsets_s_sorted = [onsets_s[i] for i in sorted_indices]

    # %% cut times:
    tw_idx=np.r_[search_start:search_stop+1]
    times = np.array(times)[tw_idx]
    data_s_sorted = data_s_sorted[:,tw_idx]
    data_us = data_us[:,tw_idx]
    # %% return the data
    labels = [chs_s_sorted, times.tolist()]
    data_out=LabeledArray(data_s_sorted, labels)
    labes_all = [chs, times.tolist()]
    data_us_out=LabeledArray(data_us, labes_all)
    # onset_out=LabeledArray(np.array(onsets_s_sorted), chs_s_sorted)
    return data_out,data_us_out,sorted_indices,chs_s_all_idx

def get_peak(data_in):
    import numpy as np
    from ieeg.arrays.label import LabeledArray

    # data: channels*times
    times=data_in.labels[1]
    times = [float(i) for i in times]
    data=data_in.__array__()

    max_values = []
    max_positions = []

    for channel in data:
        if np.all(np.isnan(channel)):
            max_values.append(np.nan)
            max_positions.append(np.nan)
        else:
            max_value = np.nanmax(channel)
            max_position = times[np.nanargmax(channel)]
            max_values.append(max_value)
            max_positions.append(max_position)

    return max_values, max_positions

def get_sig_elecs_keyword(data_in,sig_idx,keyword):
    chs = data_in.labels[0]
    out = []
    for i,val in enumerate(chs):
        if i in sig_idx and keyword in val:
            out.append(chs[i])
    return out

def plot_chs(data_in, fig_save_dir_fm,title):
    """
    plot the significant channels in a sorted order
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    times = data_in.labels[1]
    times = [float(i) for i in times]
    chs = data_in.labels[0]
    data = data_in.__array__()

    # Automatically calculate ch_gap and time_gap to avoid label overlap
    num_channels = len(chs)
    num_times = len(times)

    # Automatically adjust channel gap and time gap based on the number of channels and time points
    ch_gap = max(1, num_channels // 20)  # Adjust channel gap based on the total number of channels
    time_gap = max(1, num_times // 3)  # Adjust time gap based on the total number of time points

    # Create the plot
    plt.figure(figsize=(15, 15))  # Make the figure size large enough for labeling
    fig, ax = plt.subplots()
    im=ax.imshow(data, cmap='Blues')
    # fig.colorbar(im, ax=ax)
    ax.set_title(title)

    # Set channel names with adjusted gap
    channel_names = chs[::ch_gap]
    ax.set_yticks(range(0, len(channel_names) * ch_gap, ch_gap))
    ax.set_yticklabels(channel_names)

    # Set time stamps with adjusted gap
    time_stamps = [round(t, 3) for t in times[::time_gap]]
    ax.set_xticks(range(0, len(time_stamps) * time_gap, time_gap))
    ax.set_xticklabels(time_stamps)

    # Find the zero time index and add a vertical line
    try:
        zero_time_index = np.where(np.array(times) == 0)[0][0]
        jitter_time_index = np.where(np.array(times) == -.0234)[0][0]
    except Exception as e:
        times_array = np.array(times)
        zero_time_index = np.abs(times_array).argmin()
        jitter_time_index = np.abs(times_array+.0234).argmin()

    ax.axvline(x=zero_time_index, color='black', linestyle='--', linewidth=1)

    # Save the figure
    fig.savefig(fig_save_dir_fm, dpi=300)
    del fig
    plt.close()


def plot_brain(subjs,picks,chs_cols,label_every,fig_save_dir_f, dotsize=0.3,transparency=0.3,**kwargs):
    subjs = ['D' + subj[1:].lstrip('0') for subj in subjs]
    from ieeg.viz.mri import plot_on_average
    fig3d = plot_on_average(subjs, picks=picks,color=chs_cols,hemi='split',
                            label_every=label_every, size=dotsize,transparency=transparency, **kwargs)
    #fig3d.save_image(fig_save_dir_f)

def atlas2_hist(label2atlas_raw,chs_sel,col,fig_save_dir_fm):
    label2atlas={ch_sel: label2atlas_raw[ch_sel] for ch_sel in chs_sel}
    import matplotlib.pyplot as plt
    # Count the number of keys for each value
    value_counts = {}
    for key, value in label2atlas.items():
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    # Create the bar plot
    plt.figure(figsize=(15, 6))  # Make the figure size large enough for labeling
    # Sort the values by their count in descending order
    sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    # Create the bar plot
    plt.bar([item[0] for item in sorted_values], [item[1] for item in sorted_values],color=col)
    plt.xlabel('Atlas', fontsize=20)
    plt.ylabel('Number of Electrodes', fontsize=20)
    plt.xticks(fontsize=30, rotation=45, ha='right')
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig(fig_save_dir_fm, dpi=300)
    plt.close()

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

    # print missing channels
    for curr_chs in current_ch_indices.keys():
        if curr_chs not in org_labeled_chs:
            print(f"Lost channel: {subject}-{curr_chs}, found in stats but not found in channels.tsv")

    return aligned_data, aligned_chs

def plot_wave(data_in,sig_idx,con_label,col,Lstyle,bsl_crr,errtype='se',normalize=False):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    def rowwise_normalize(data: np.ndarray, axis: int, eps: float = 1e-8) -> np.ndarray:

        if axis == 1:
            min_vals = np.nanmin(data, axis=axis)[:, np.newaxis]
            max_vals = np.nanmax(data, axis=axis)[:, np.newaxis]
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = eps
        elif axis == 0:
            min_vals = np.nanmin(data, axis=axis)
            max_vals = np.nanmax(data, axis=axis)
            range_vals = max_vals - min_vals

        normalized = (data - min_vals) / range_vals
        return normalized

    times=data_in.labels[1]
    times = [float(i) for i in times]
    chs=data_in.labels[0]
    data=data_in.__array__()

    data_selected = np.full(np.shape(data), np.nan)
    for i in sig_idx:
        data_selected[i] = data[i]

    # Select a few key time points for labeling
    tick_labels = [t for t in times if t % 0.25 == 0]  # Keep only multiples of 0.5

    # Compute the mean and SEM across trials while ignoring NaNs
    mean_waveform = np.nanmean(data_selected, axis=0)
    # Normalize
    if normalize:
        mean_waveform = rowwise_normalize(mean_waveform,0)

    # Baseline correction (should remove this)
    if bsl_crr:
        mean_waveform = mean_waveform - np.nanmean(mean_waveform[:51])
    if errtype == 'std':
        sem_waveform = np.nanstd(data_selected, axis=0)
    elif errtype == 'se':
        sem_waveform = np.nanstd(data_selected, axis=0) / np.sqrt(np.sum(~np.isnan(data_selected), axis=0))  # SEM ignoring NaNs
    # Plot the mean waveform
    plt.plot(times, mean_waveform, label=con_label, color=col,linestyle=Lstyle)

    # Add shaded region for SEM
    plt.fill_between(times, mean_waveform - sem_waveform, mean_waveform + sem_waveform, color=col, alpha=0.3)

    plt.xlabel('Time (s)',fontsize=10)
    plt.xticks(tick_labels)

def time_avg_select(data_in,sig_idx_lst):

    import numpy as np
    import matplotlib.pyplot as plt
    chs=data_in.labels[0]
    data=data_in.__array__()
    data_avg=np.nanmean(data,axis=1)
    data_avg_select_final=[]
    chs_select_final=[]
    for sig_idx in sig_idx_lst:
        data_avg_select=data_avg[list(sig_idx)]
        chs_select=chs[list(sig_idx)]
        data_avg_select_final.append(data_avg_select)
        chs_select_final.append(chs_select)
    data_avg_select_final=np.concatenate(data_avg_select_final,axis=0)
    chs_select_final=np.concatenate(chs_select_final,axis=0)
    return data_avg, data_avg_select_final,chs_select_final

def set2arr(set,arr_len):
    """
    Change set to one-zero array
    Args:
        set: the set
        arr_len: length of array

    Returns:
        arr: the one-zero array

    """
    import numpy as np
    arr=np.zeros(arr_len,dtype=int)
    arr[list(set)]=1
    return arr

def chs2atlas(subjs,chs_all):
    import pandas as pd
    # %% Make Atlas histograms
    from ieeg.viz.mri import subject_to_info, gen_labels
    subjs_s = ['D' + subj[1:].lstrip('0') for subj in subjs]
    ch_labels = dict()
    for subj in subjs_s:
        info_i = subject_to_info(subj)
        ch_labels_k = gen_labels(info_i, subj, atlas='.BN_atlas')
        for key, value in ch_labels_k.items():
            ch_labels[f'{subj}-{key}'] = value

    # Extract relevant columns and create a mapping dictionary
    # Load the CSV file
    df = pd.read_csv('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\atlas.csv')
    # Create the dictionary
    mapping_dict = {}
    for index, row in df.iterrows():
        key = str(row['Anatomical and modified Cyto-architectonic descriptions']).split(',')[0]
        value = str(row['Left and Right Hemisphere']).split('_')[0]
        mapping_dict[key] = value
    mapping_dict['TE1.0/TE1.2'] = 'STG'
    ch_labels_roi = dict()
    ch_labels_clean = dict()
    for key, value in ch_labels.items():
        if key in chs_all:
            ch_labels_clean[key]=value
            try:
                ch_labels_roi[key] = mapping_dict[value.split("_")[0]]
            except KeyError as e:
                ch_labels_roi[key] = 'unknown'
    return ch_labels_roi,ch_labels_clean

def hickok_roi(ch_labels_roi,ch_labels):
    hickok_roi_labels = dict()
    for key, value in ch_labels_roi.items():
        value_raw=ch_labels[key]
        if value_raw!='Unknown':
            hemi=value_raw.split("_")[1]
        else:
            hemi='unknown'
        try:
            if (value=='IFG' and hemi=='L') or (value=='IPL' and hemi=='L'):
                hickok_roi_labels[key]='l'+value
            elif value_raw=='A441/42_L' or value_raw=='A22c_L' or value_raw=='G_L':
                hickok_roi_labels[key] ='Spt'
            elif value_raw=='A6cdl_L':
                hickok_roi_labels[key] ='lPMC'
            else:
                hickok_roi_labels[key]='N/A'
        except KeyError as e:
            hickok_roi_labels[key] = 'unknown'
    return hickok_roi_labels

def plot_sig_roi_counts(hickok_roi_labels, color, sig_idx,savedir):
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    keys = list(hickok_roi_labels.keys())
    if isinstance(sig_idx, set):
        selected_labels = [
            hickok_roi_labels[keys[i]]
            for i in sig_idx
            if hickok_roi_labels[keys[i]] != 'N/A'
        ]
    elif isinstance(sig_idx, np.ndarray):
        selected_labels=[]
        for i,val in enumerate(sig_idx):
            if val==1 and hickok_roi_labels[keys[i]] != 'N/A':
                selected_labels.append(hickok_roi_labels[keys[i]])
    label_counts = Counter(selected_labels)
    label_counts_s = dict(sorted(label_counts.items()))
    plt.figure(figsize=(6, 5))
    plt.bar(label_counts_s.keys(), label_counts_s.values(), color=color)
    # plt.xlabel('ROI Label',fontsize=30)
    # plt.ylabel('Number of Electrodes', fontsize=30)
    # plt.title('Significant Electrodes per ROI')
    plt.xticks(fontsize=40, rotation=45, ha='right')
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig(savedir,dpi=300)

def get_coor(chs):
    """
    Given a list of channels like ['D23-RASF1', 'D23-RASF2', ...],
    this function reads each subject's electrode file and returns
    a dataframe of 3D coordinates for the specified electrodes.
    """
    # Parse into (subject, label)
    import pandas as pd
    import os
    import numpy as np
    parsed = {}
    for ch in chs:
        subj, label = ch.split('-')
        if subj not in parsed:
            parsed[subj] = []
        parsed[subj].append(label)

    # Initialize result dataframe
    df_coords = pd.DataFrame(columns=['subj', 'label', 'x', 'y', 'z'])
    HOME = os.path.expanduser("~")

    for subj, labels in parsed.items():
        # File path
        path = os.path.join(HOME,f"Box\\ECoG_Recon\\{subj}\\elec_recon\\{subj}_elec_locations_RAS.txt")

        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Warning: File not found for subject {subj}")
            continue

        # Create dictionary from label to coordinates
        coord_map = {}
        for line in lines:
            parts = line.strip().split()
            label = parts[0]+parts[1]
            try:
                x, y, z = map(float, parts[2:5])
                coord_map[label] = (x, y, z)
            except ValueError:
                continue

        # Add matching labels
        for label in labels:
            if label in coord_map:
                x, y, z = coord_map[label]
                df_coords.loc[len(df_coords)] = [subj, label, x, y, z]
            else:
                df_coords.loc[len(df_coords)] = [subj, label, np.nan, np.nan, np.nan]

    return df_coords


def bsliang_add_connecting_lines(plt, k, strip):
    import pandas as pd
    GroupA_Xpos = strip.collections[k].get_offsets().data[:, 0].tolist()
    GroupA_Ypos = strip.collections[k].get_offsets().data[:, 1].tolist()
    GroupB_Xpos = strip.collections[k + 1].get_offsets().data[:, 0].tolist()
    GroupB_Ypos = strip.collections[k + 1].get_offsets().data[:, 1].tolist()

    Xs = []
    Ys = []
    for i in range(len(GroupA_Xpos)):
        Xs = Xs + [[GroupA_Xpos[i], GroupB_Xpos[i]]]
        Ys = Ys + [[GroupA_Ypos[i], GroupB_Ypos[i]]]

    data_line = {
        'Categories': Xs,
        'Values': Ys
    }
    data_line = pd.DataFrame(data_line)

    # Draw lines between the points
    for i in range(len(GroupA_Xpos)):
        plt.plot(data_line.Categories[i], data_line.Values[i], color='grey', linewidth=0.05, linestyle='--',
                 zorder=1)  # Draw

    return plt

def bsliang_align_yaxis(LIM, TICKS):
    # Set the y-axis label at the midpoint of the y-axis range
    midpoint_LIM = (LIM[0] + LIM[1]) / 2  # Calculate midpoint of y-axis range
    midpoint_TICS = (TICKS[0] + TICKS[-1]) / 2
    pos = 0.5 + (midpoint_TICS - midpoint_LIM) / (LIM[1] - LIM[0])
    return pos