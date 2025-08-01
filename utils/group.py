from pyqtgraph.util.cprint import color

def load_stats(stat_type,con,contrast,stats_root_readID,stats_root_readdata,split_half=0,trial_labels: str='CORRECT',keeptrials: bool=False):
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
    from ieeg.arrays.label import combine as cb_dict

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
    subjs = [subj for subj in subjs if subj != 'D0107' and subj != 'D0042' and subj != 'D0115' and subj != 'D0117'] # problematic patients: 102 and 103: eeg electrodes, 107, plotting issues, 42: bad heading, each should be dealed with
    # else:
    #     subjs = [subj for subj in subjs if subj != 'D0023' and subj != 'D0032' and subj != 'D0035'  and subj != 'D0038' and subj != 'D0042' and subj != 'D0044' and subj != 'D0107' and subj != 'D0042' and subj != 'D0115' and subj != 'D0117']# and subj != 'D0028'] # problematic patients: 102 and 103: eeg electrodes, 107, plotting issues, 42: bad heading, each should be dealed with

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
        try:
            subj_dataset = fif_read(file_dir)
        except Exception as e:
            continue

        match stat_type:
            case "zscore":
                subj_dataset = subj_dataset[trial_labels]
                subj_data_epo = subj_dataset._data
                if keeptrials:
                    subj_data = subj_data_epo
                else:
                    match split_half:
                        case 0:
                            subj_data = np.nanmean(subj_data_epo, axis=0)
                        case 1:
                            half = subj_data_epo.shape[0] // 2
                            subj_data = np.nanstd(subj_data_epo[:half], axis=0)
                        case 2:
                            half = subj_data_epo.shape[0] // 2
                            subj_data = np.nanstd(subj_data_epo[half:], axis=0)
                subj_chs = subj_dataset.ch_names
                times = subj_dataset.times
            case "power":
                subj_dataset = subj_dataset[trial_labels]
                subj_data_epo = subj_dataset._data
                if keeptrials:
                    subj_data = subj_data_epo
                else:
                    match split_half:
                        case 0:
                            subj_data = np.nanmean(subj_data_epo, axis=0)
                        case 1:
                            half = subj_data_epo.shape[0] // 2
                            subj_data = np.nanstd(subj_data_epo[:half], axis=0)
                        case 2:
                            half = subj_data_epo.shape[0] // 2
                            subj_data = np.nanstd(subj_data_epo[half:], axis=0)
                subj_chs = subj_dataset.ch_names
                times = subj_dataset.times
            case "significance":
                # Not yet tested, maybe wrong
                subj_data = subj_dataset[0].data
                subj_chs = subj_dataset[0].ch_names
                times = subj_dataset[0].times
            case "pval":
                # Not yet tested, maybe wrong
                subj_data = subj_dataset[0].data
                subj_chs = subj_dataset[0].ch_names
                times = subj_dataset[0].times
            case "mask":
                subj_data = subj_dataset[0].data
                subj_chs = subj_dataset[0].ch_names
                times = subj_dataset[0].times
            case 'glm':
                subj_data = subj_dataset
                evoke_dir = os.path.join(subj_gamma_stats_dir, 'Auditory_inRep_mask-ave.fif')
                chs_time_dataset_forglm = mne.read_evokeds(evoke_dir)
                subj_chs = chs_time_dataset_forglm[0].ch_names
                times = chs_time_dataset_forglm[0].times

        if keeptrials:
            arr = LabeledArray.from_signal(subj_dataset)
            trial_annoate = arr.labels[0]
            del arr
            # trial_annoate = make_array_unique(trial_annoate,'/')
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

        # May try this:
        # arr.labels[1]=Labels(['D' + subject[1:].lstrip('0')]) @ Labels(arr.labels[1])

        good_labeled_chs = [f"{subject} {ch}" for ch in subj_chs]

        if keeptrials:
            add_subj_labels=False
        else:
            add_subj_labels=True
        aligned_data,aligned_chs = align_channel_data(subj_data, good_labeled_chs, org_labeled_chs,add_subj_labels=add_subj_labels)

        data_lst.append(aligned_data)
        chs.extend(aligned_chs)

        # The following codes just do not take the outlier or muscle electrodes for considerations
        # This may (but most likely may not) affect comparing lexical delay and lexical no delay as they can have different bad electrodes.
        # subject = 'D' + subject[1:].lstrip('0')
        # good_labeled_chs_reformat = [f"{subject}-{ch}" for ch in subj_chs]
        # data_lst.append(subj_data)
        # chs.extend(good_labeled_chs_reformat)

        if keeptrials:
            labels=[trial_annoate,aligned_chs,times]
            subj_arr = LabeledArray(aligned_data, labels)
            subj_dict = subj_arr.to_dict()
            del subj_arr,aligned_data
            subject = 'D' + subject[1:].lstrip('0')
            if i==0:
                data=dict()
            data[subject]=subj_dict
            del subj_dict

    if keeptrials:
        data_dict=cb_dict(data,(0,2),'-')
        del data
        data=LabeledArray.from_dict(data_dict)
        del data_dict
    else:
        data_raw = np.concatenate(data_lst, axis=0)
        labels = [chs, times]
        data=LabeledArray(data_raw, labels)
    return data,valid_subjs

import numpy as np
def make_array_unique(arr: np.ndarray, delimiter: str) -> np.ndarray:
    """Make an array unique by appending a number to duplicate values.

    Parameters
    ----------
    arr : np.ndarray
        The array to make unique.
    delimiter : str
        The delimiter to use when appending a number to duplicate values.

    Returns
    -------
    np.ndarray
        The unique array.

    Examples
    --------
    # >>> arr = np.array(['a', 'b', 'c', 'a', 'b', 'c'])
    # >>> _make_array_unique(arr, '-')
    array(['a-0', 'b-0', 'c-0', 'a-1', 'b-1', 'c-1'], dtype='<U3')
    # >>> arr = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    # >>> _make_array_unique(arr, '-')
    array(['a', 'b', 'c', 'd', 'e', 'f'], dtype='<U1')
    """
    unique, inverse = np.unique(arr, return_inverse=True)
    if len(unique) == len(arr):
        return arr
    counts = np.bincount(inverse)
    max_dtype = np.max([len(u) for u in unique]) + 1 + len(str(max(counts)))
    out = np.empty_like(arr, dtype=f'<U{max_dtype}')
    for i, (u, c) in enumerate(zip(unique, counts)):
        if c == 1:
            out[inverse == i] = u
        else:
            indices = np.where(arr == u)[0]
            for j, index in enumerate(indices):
                out[index] = f"{u}{delimiter}{j}"
    return out

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

def sort_chs_by_actonset(mask_in,data_in,win_len,time_range,mask_data=False,bin:bool=False,chs_s_all_idx=None,sorted_indices=None,select_electrodes:bool=True):
    """
    Selete channels with significant activation clusters (all mask==1 in any window with win_len) within a time range (time_range).
    Sort the significant channels according to the onset.

    e.g.:
    data: imported data in LabeledArray
    win_len = 0.2 (in seconds, the shortest sig cluster that consider effective)
    time_range = [-0.2, 1]
    """
    import numpy as np
    import pandas as pd
    from ieeg.arrays.label import LabeledArray

    if (chs_s_all_idx is not None) and (sorted_indices is None):
        raise ValueError("sorted_indices cannot be None if chs_s_all_idx is provided")

    # %% get the onsets of the activation (an effective cluster is defined as 0.2s)
    times=data_in.labels[1]
    times = [float(i) for i in times]
    chs=data_in.labels[0]
    data=data_in.__array__()
    mask=mask_in.__array__()

    spf = 1 / (times[1] - times[0])  # Calculate the sampling frequency
    if bin:
        win = 1
    else:
        win = int(win_len * spf)  # Number of samples in 0.1 seconds

    onsets = {}

    if bin:
        search_start = np.searchsorted(times, time_range[0], side='left')
        search_stop = np.searchsorted(times, time_range[1], side='right') - 1
    else:
        search_start = np.argmin(np.abs(np.array(times) - time_range[0]))
        search_stop = np.argmin(np.abs(np.array(times) - time_range[1]))

    for ch_idx, ch_name in enumerate(chs):
        ch_mask = mask[ch_idx]
        found = False

        if bin:
            for start_idx in range(int(search_start),int(search_stop)+1):
                win_mask = ch_mask[start_idx]

                if np.all(win_mask == 1):
                    starting_time = times[start_idx]
                    onsets[ch_name] = starting_time
                    found = True
                    break
        else:
            for start_idx in range(int(search_start),int(search_stop) - win + 1):
                win_mask = ch_mask[start_idx:start_idx + win]

                if np.all(win_mask == 1):
                    starting_time = times[start_idx]
                    onsets[ch_name] = starting_time
                    found = True
                    break

        if not found:
            if select_electrodes:
                onsets[ch_name] = None  # No significant window found for '1'
            else:
                onsets[ch_name] = 1e3

    # %% select channels with significant activation clusters
    mask_s = []
    data_s = []
    data_us = np.full([np.shape(data)[0],np.shape(data)[1]],np.nan)
    chs_s = []
    onsets_s = []

    if chs_s_all_idx is not None:
        for ch_idx, ch_name in enumerate(chs):
            onset = onsets.get(ch_name)  # Get the onset, avoiding repeated dictionary lookups
            if ch_idx in chs_s_all_idx:  # Check if the channel has a valid onset
                mask_s.append(mask[ch_idx])
                data_s.append(data[ch_idx])  # Add the channel data to the selected data list
                chs_s.append(ch_name)  # Add the channel name to the selected channel names list
                onsets_s.append(onset)
                data_us[ch_idx, :] = np.where(mask[ch_idx], data[ch_idx],np.nan)
    else:
        chs_s_all_idx = set()  # Use a set to store selected channel indices
        for ch_idx, ch_name in enumerate(chs):
            onset = onsets.get(ch_name)  # Get the onset, avoiding repeated dictionary lookups
            if onset is not None:  # Check if the channel has a valid onset
                mask_s.append(mask[ch_idx])
                data_s.append(data[ch_idx])  # Add the channel data to the selected data list
                chs_s.append(ch_name)  # Add the channel name to the selected channel names list
                onsets_s.append(onset)
                chs_s_all_idx.add(ch_idx)  # Add index to the set
                data_us[ch_idx,:]=np.where(mask[ch_idx], data[ch_idx], np.nan) # the data_us contained all the channels and makred the inactive channels as nan

    # Convert the selected data list to a numpy array
    mask_s = np.array(mask_s)
    data_s = np.array(data_s)

    # %% do the ranking
    if sorted_indices is None:
        sorted_indices = np.argsort(np.array(onsets_s))  # Get the indices that would sort the array
    mask_s_sorted = mask_s[sorted_indices]
    data_s_sorted = data_s[sorted_indices]
    if mask_data:
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
    # %% get the parameters
    num_channels = data_s_sorted.shape[0]
    data_s_sorted_paras = {}

    for channel_idx in range(num_channels):

        channel_data = data_s_sorted[channel_idx]

        not_nan_values = channel_data[~np.isnan(channel_data)]
        activity_length = (len(not_nan_values)/np.shape(channel_data)[0])*(times[-1]-times[0])

        # Initialize values for peak, peak location, first non-NaN location and mean
        peak_value = 0
        peak_location = np.nan
        first_non_nan_location = np.nan
        mean_value = 0

        if activity_length > 0:

            peak_value = np.max(not_nan_values)
            peak_location = times[np.where(channel_data == peak_value)[0][0]]

            first_non_nan_location = times[np.where(~np.isnan(channel_data))[0][0]]

            mean_value = np.mean(not_nan_values)

        data_s_sorted_paras[chs_s_sorted[channel_idx]]={
            "activity_length": activity_length,
            "peak_value": peak_value,
            "peak_location": peak_location,
            "first_non_nan_location": first_non_nan_location,
            "mean_value": mean_value
        }

    data_s_sorted_paras_tab = pd.DataFrame.from_dict(data_s_sorted_paras, orient='index')

    return data_out,data_us_out,sorted_indices,chs_s_all_idx,onsets,data_s_sorted_paras_tab


def sort_chs_by_actonset_combined(mask_in1, mask_in2, win_len, time_range, bin: bool = False,sortonset_base=2,select_electrodes:bool=True):
    """
    Select channels with significant activation clusters (all mask==1 in any window with win_len) within a time range (time_range).
    The masks are combined as follows:
    - mask_in1 and mask_in2 both 0: Combined mask is 0
    - mask_in1 and mask_in2 both 1: Combined mask is 1 (considered for onset sorting)
    - mask_in1 = 1, mask_in2 = 0: Combined mask is 2
    - mask_in1 = 0, mask_in2 = 1: Combined mask is 3

    Sort the significant channels according to the onset of the '1' state in the combined mask.

    Args:
        mask_in1 (LabeledArray): First input mask data.
        mask_in2 (LabeledArray): Second input mask data.
        win_len (float): The shortest significant cluster duration to consider effective (in seconds).
        time_range (list): A list [start_time, end_time] defining the time window for searching.
        bin (bool): If True, treats the window as a single bin.

    Returns:
        tuple: A tuple containing:
            - combined_mask_out (LabeledArray): Sorted LabeledArray of combined masks for significant channels.
            - combined_mask_all_chs_out (LabeledArray): LabeledArray of combined masks for all channels (inactive marked as NaN).
            - sorted_indices (numpy.ndarray): Indices that would sort the original channels by onset.
            - chs_s_all_idx (set): Set of indices of channels found to have significant '1' clusters.
    """
    import numpy as np
    from ieeg.arrays.label import LabeledArray

    times1 = mask_in1.labels[1]
    times1 = [float(i) for i in times1]
    times2 = mask_in2.labels[1]
    times2 = [float(i) for i in times2]
    chs = mask_in1.labels[0]
    mask1_raw = mask_in1.__array__()
    mask2_raw = mask_in2.__array__()

    # --- Time Synchronization ---
    # Find the common start and end times
    start_time_common = max(times1[0], times2[0])
    end_time_common = min(times1[-1], times2[-1])

    # Get indices for mask1
    idx1_start = np.searchsorted(times1, start_time_common, side='left')
    idx1_end = np.searchsorted(times1, end_time_common, side='right') - 1

    # Get indices for mask2
    idx2_start = np.searchsorted(times2, start_time_common, side='left')
    idx2_end = np.searchsorted(times2, end_time_common, side='right') - 1

    # Slice masks and update times
    mask1 = mask1_raw[:, idx1_start:idx1_end + 1]
    mask2 = mask2_raw[:, idx2_start:idx2_end + 1]
    times = times1[idx1_start:idx1_end + 1] # Use times1 as the reference for the new common times

    # Combine masks
    # Resulting combined mask values:
    # 0: mask1=0, mask2=0
    # 1: mask1=1, mask2=1 (considered for onset)
    # 2: mask1=1, mask2=0
    # 3: mask1=0, mask2=1
    combined_mask = np.zeros_like(mask1, dtype=int)
    combined_mask[(mask1 == 0) & (mask2 == 0)] = 0
    combined_mask[(mask1 == 1) & (mask2 == 1)] = 1
    combined_mask[(mask1 == 1) & (mask2 == 0)] = 2
    combined_mask[(mask1 == 0) & (mask2 == 1)] = 3

    spf = 1 / (times[1] - times[0])  # Calculate the sampling frequency
    if bin:
        win = 1
    else:
        win = int(win_len * spf)  # Number of samples in win_len seconds

    onsets = {}

    if bin:
        search_start = np.searchsorted(times, time_range[0], side='left')
        search_stop = np.searchsorted(times, time_range[1], side='right') - 1
    else:
        search_start = np.argmin(np.abs(np.array(times) - time_range[0]))
        search_stop = np.argmin(np.abs(np.array(times) - time_range[1]))

    for ch_idx, ch_name in enumerate(chs):
        ch_combined_mask = combined_mask[ch_idx]
        found = False

        if bin:
            for start_idx in range(int(search_start), int(search_stop) + 1):
                win_mask = ch_combined_mask[start_idx]

                # We are looking for onset of '2' in the combined mask
                if np.all(win_mask == sortonset_base):
                    starting_time = times[start_idx]
                    onsets[ch_name] = starting_time
                    found = True
                    break
        else:
            for start_idx in range(int(search_start), int(search_stop) - win + 1):
                win_mask = ch_combined_mask[start_idx:start_idx + win]

                # We are looking for onset of '1' in the combined mask
                if np.all(win_mask == sortonset_base):
                    starting_time = times[start_idx]
                    onsets[ch_name] = starting_time
                    found = True
                    break

        if not found:
            if select_electrodes:
                onsets[ch_name] = None  # No significant window found for '1'
            else:
                onsets[ch_name] = 1e3

                # %% Select channels with significant activation clusters (where combined_mask == 1)
    combined_mask_s = []
    combined_mask_all_chs = np.full([np.shape(combined_mask)[0], np.shape(combined_mask)[1]], np.nan)
    chs_s = []
    chs_s_idx = []  # significant channels selected
    chs_s_all_idx = set()  # Use a set to store selected channel indices
    onsets_s = []

    for ch_idx, ch_name in enumerate(chs):
        onset = onsets.get(ch_name)  # Get the onset, avoiding repeated dictionary lookups
        if onset is not None:  # Check if the channel has a valid onset (i.e., a '1' cluster was found)
            combined_mask_s.append(combined_mask[ch_idx])
            chs_s.append(ch_name)
            chs_s_idx.append(ch_idx)
            onsets_s.append(onset)
            chs_s_all_idx.add(ch_idx)  # Add index to the set
            combined_mask_all_chs[ch_idx, :] = combined_mask[ch_idx]  # Fill for selected channels
        else:
            # For channels without a '1' cluster, fill with NaN or an appropriate placeholder
            combined_mask_all_chs[ch_idx,
            :] = np.nan  # Or you could decide to keep original combined_mask values for non-selected

    combined_mask_s = np.array(combined_mask_s)

    # %% Do the ranking
    sorted_indices = np.argsort(np.array(onsets_s))  # Get the indices that would sort the array
    combined_mask_s_sorted = combined_mask_s[sorted_indices]
    chs_s_sorted = [chs_s[i] for i in sorted_indices]
    onsets_s_sorted = [onsets_s[i] for i in sorted_indices]

    # %% Cut times
    tw_idx = np.r_[search_start:search_stop + 1]
    times = np.array(times)[tw_idx]
    combined_mask_s_sorted = combined_mask_s_sorted[:, tw_idx]
    combined_mask_all_chs = combined_mask_all_chs[:, tw_idx]

    # %% Return the data
    labels_significant = [chs_s_sorted, times.tolist()]
    combined_mask_out = LabeledArray(combined_mask_s_sorted, labels_significant)

    labels_all = [chs, times.tolist()]
    combined_mask_all_chs_out = LabeledArray(combined_mask_all_chs, labels_all)

    return combined_mask_out, combined_mask_all_chs_out, sorted_indices, chs_s_all_idx


def get_latency(data_in,mode:str='peak'):
    import numpy as np
    from ieeg.arrays.label import LabeledArray

    # data: channels*times
    times=data_in.labels[1]
    times = [float(i) for i in times]
    data=data_in.__array__()

    values = []
    positions = []

    for channel in data:
        if np.all(np.isnan(channel)):
            values.append(np.nan)
            positions.append(np.nan)
        else:
            if mode=='peak':
                value = np.nanmax(channel)
                position = times[np.nanargmax(channel)]
            elif mode=='onset':
                non_nan_indices = np.where(~np.isnan(channel))[0]
                first_non_nan_idx = non_nan_indices[0]
                value = channel[first_non_nan_idx]
                position = times[first_non_nan_idx]
            elif mode=='clustermid':
                non_nan_indices = np.where(~np.isnan(channel))[0]
                mid_non_nan_idx = non_nan_indices[non_nan_indices.size // 2]
                value = channel[mid_non_nan_idx]
                position = times[mid_non_nan_idx]
            values.append(value)
            positions.append(position)

    return values, positions

def get_sig_elecs_keyword(data_in,sig_idx,keyword):
    chs = data_in.labels[0]
    out = []
    for i,val in enumerate(chs):
        if i in sig_idx and keyword in val:
            out.append(chs[i])
    return out

def plot_chs(data_in, fig_save_dir_fm,title,is_ytick=False,bin:bool=False,discrete_y:bool=False,discrete_y_lables:list=['Both silent', 'Shared sig', 'Delay Rep only', 'NoDelay JL only'],percentage_vscale=True,vmin=0,vmax=100,is_colbar=True,fig_size:list=[4,10]):
    """
    plot the significant channels in a sorted order
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.colors as mcolors

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
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    if percentage_vscale:
        vmin = np.nanpercentile(data, vmin)
        vmax = np.nanpercentile(data, vmax)
    if bin:
        im = sns.heatmap(
            data,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar_kws={'label': 'Data Value'}
        )
    else:
        if not discrete_y:
            im=ax.imshow(data, cmap='Blues',vmin=vmin, vmax=vmax,interpolation='none')
            # sns.heatmap(data, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax,
            #             linewidths=0.5, linecolor='lightgrey',
            #             mask=np.isnan(data))  #
            if is_colbar:
                # Add the colorbar to the plot
                cbar = fig.colorbar(im, ax=ax, ticks=[vmin, vmax])
                # Label the ticks
                cbar.ax.set_yticklabels([f'Min: {vmin:.2f}', f'Max: {vmax:.2f}'])
                cbar.set_label('Data Range') # Add a label for the colorbar
        else:
            colors = ['lightgray', 'skyblue', 'mediumseagreen', 'salmon']  # Custom colors for 0, 1, 2, 3
            custom_cmap = mcolors.ListedColormap(colors)
            # Define boundaries for the discrete values
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
            norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
            im = ax.imshow(data, cmap=custom_cmap, norm=norm,
                           interpolation='nearest')  # 'nearest' is good for discrete data
            if is_colbar:
                # Create the colorbar
                cbar = fig.colorbar(im, cmap=custom_cmap, norm=norm, boundaries=bounds, ticks=[0, 1, 2, 3],
                                    orientation='vertical', shrink=0.8)
                cbar.set_ticklabels(discrete_y_lables)
                cbar.set_label('Shared significance')

    # fig.colorbar(im, ax=ax)
    ax.set_title(title)

    # Set channel names with adjusted gap
    channel_names = chs[::ch_gap]
    if is_ytick==True:
        ax.set_yticks(range(0, len(channel_names) * ch_gap, ch_gap))
        ax.set_yticklabels(channel_names)
    else:
        ax.yaxis.set_visible(False)

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
    ax.set_aspect('auto')
    plt.tight_layout()
    fig.savefig(fig_save_dir_fm, dpi=300)
    del fig
    plt.close()


def onsets2col(onsets, chs_sel, colormap_type: str = 'jet',treat_zero_as_nan:bool=False):

    import numpy as np
    import matplotlib.cm as cm

    valid_onsets = []
    for ch in chs_sel:
        if ch in onsets and np.isnan(onsets[ch]):
            valid_onsets.append(1000)
        elif ch in onsets and onsets[ch]==0 and treat_zero_as_nan:
            valid_onsets.append(1000)
        elif ch in onsets and onsets[ch] is not None:
            valid_onsets.append(onsets[ch])
        else:
            raise ValueError("ch not in valid onset index or have no onset")

    onset_values = np.array(valid_onsets)
    mask = onset_values <= 100
    onsets_for_normalization = onset_values[mask]
    normalized_onsets = np.full(onset_values.shape, np.nan, dtype=float)
    min_onset = np.min(onsets_for_normalization)
    max_onset = np.max(onsets_for_normalization)
    if max_onset == min_onset:
        normalized_onsets[mask] = 0.0
    else:
        normalized_onsets[mask] = (onsets_for_normalization - min_onset) / (max_onset - min_onset)
    colormap = cm.get_cmap(colormap_type)

    cols = []
    onset_idx = 0
    for ch in chs_sel:
        norm_val = normalized_onsets[onset_idx]
        if ~np.isnan(norm_val):
            rgb_color = [float(colormap(norm_val)[0]),float(colormap(norm_val)[1]),float(colormap(norm_val)[2])]
        else:
            rgb_color = [0.5,0.5,0.5]
        cols.append(rgb_color)
        onset_idx += 1
    return cols

def plot_brain(subjs,picks,chs_cols,label_every,fig_save_dir_f='x', dotsize=0.3,transparency=0.3,hemi: str='split',save_img: bool=False,**kwargs):
    subjs = ['D' + subj[1:].lstrip('0') for subj in subjs]
    from ieeg.viz.mri import plot_on_average
    if save_img:
        show=False
    else:
        show=True
    fig3d = plot_on_average(subjs, picks=picks,color=chs_cols,hemi=hemi,
                            label_every=label_every, size=dotsize,transparency=transparency, show=show,**kwargs)
    if save_img:
        fig3d.save_image(fig_save_dir_f)
        fig3d.close()


def plot_brain_window(mask, data, cluster_twin, wins_para, save_dir, col: list = [0, 0, 1],
                      mode:str="save_brain_plot",label_every=None,bin:bool=False):
    import os
    import numpy as np
    from ieeg.viz.mri import plot_on_average

    # Plot sig electrodes at a time window
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # make sliding windows
    #wins_para [starting_time, ending_time, gap, win_len], in seconds
    cut_wins=[]
    if bin:
        cut_wins = [[-0.2, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0], [1.0, 1.5]]
    else:
        for win_start in np.arange(wins_para[0],wins_para[1],wins_para[2]):
            win_start=np.round(win_start,3).item()
            cut_wins.append([win_start,np.round(win_start+wins_para[3],3).item()])

    avgs=np.empty((np.shape(mask.__array__())[0], len(cut_wins)))
    sigs=[]
    _, raw_all, _, _ = sort_chs_by_actonset(mask, data, cluster_twin, [cut_wins[0][0],cut_wins[-1][-1]],bin=bin)
    for i, cut_win in enumerate(cut_wins):
        if i==0:
            chs_all = raw_all.labels[0]
            subjs = [item.split('-')[0] for item in raw_all.labels[0]]
        # window raw
        try:
            _, raw, _, sig = sort_chs_by_actonset(mask, data, cluster_twin, cut_win,bin=bin)
            avg, _, _ = time_avg_select(raw, [sig], normalize=False)
        except Exception as e:
            # if no sig electrodes
            avg = [np.nan]*len(raw_all.labels[0])
            sig = set()
        avgs[:,i] = avg
        sigs.append(sig)
        del avg,sig

    #normalize:
    avgs=min_max_normalize_ignore_nan(avgs)

    if mode == "save_brain_plot":
        for i, cut_win in enumerate(cut_wins):

            # make data
            sig=sigs[i]
            avg=avgs[:,i]
            chs_sel = chs_all[list(sig)].tolist()
            avg_sel = avg[list(sig)]

            # plot brain
            for hemi in ['lh', 'rh']:
                cols = [adjust_saturation(np.array(col), val,map='jet') for val in avg_sel]
                if len(chs_sel)>0:
                    # if some electrodes sig
                    fig3d = plot_on_average(subjs, picks=chs_sel,color=cols,hemi=hemi,
                                            label_every=label_every, size=0.4,transparency=0.4)
                else:
                    fig3d = plot_on_average(subjs, picks=[1,2],color=[[0.5,0.5,0.5],[0.5,0.5,0.5]],hemi=hemi,
                                            label_every=None, size=0,transparency=0.4)
                fig3d.save_image(os.path.join(save_dir, f'{hemi}_{i:03d}_{cut_win[0]}_{cut_win[1]}.jpg'))
                fig3d.close()
                # del fig3d

    # save region hist
    if mode=="save_region_hist":
        for i, cut_win in enumerate(cut_wins):
            sig = sigs[i]
            chs_sel = chs_all[list(sig)].tolist()
            ch_labels_roi, _ = chs2atlas(subjs, chs_all)
            atlas2_hist(ch_labels_roi, chs_sel, col,
                        os.path.join(save_dir, f'Atlas_{cut_win[0]}_{cut_win[1]}.jpg'), ylim=[0, 100])

    # save hickok roi hist
    if mode=="save_hickok_roi":
        for i, cut_win in enumerate(cut_wins):
            sig = sigs[i]
            chs_coor = get_coor(chs_all, 'group')
            hickok_roi_labels, _ = hickok_roi_sphere(chs_coor)
            plot_sig_roi_counts(hickok_roi_labels, col, sig,
                                os.path.join(save_dir, f'Hickok_ROI_{cut_win[0]}_{cut_win[1]}.jpg'))

    return cut_wins,sigs,chs_all


def create_video_from_images(base_path, event, task_Tag, wordness, glm_fea, hemi, cut_win_list, fps: float=4):
    """
    Reads images from a specified path and compiles them into a video.

    Args:
        base_path (str): The base path where images are stored (e.g., 'plot').
        event (str): The event name.
        task_Tag (str): The task tag.
        wordness (str): Wordness information.
        glm_fea (str): GLM feature.
        hemi (str): Hemisphere ('lh' or 'rh').
        cut_win_list (list): A list containing all time windows [start, end].
                             Example: [[0.075, 0.175], [0.1, 0.2], ...]
        fps (int, optional): Frames per second for the output video. Defaults to 4.
    """
    import cv2
    import os

    image_folder = os.path.join(base_path, event, task_Tag, wordness, glm_fea)

    # Ensure the output directory exists
    output_dir = os.path.join(base_path, event, task_Tag, wordness, glm_fea)
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Get all matching image files and sort them naturally
    image_files = []
    for i in range(len(cut_win_list)):
        cut_win = cut_win_list[i]
        # Format cut_win values to 3 decimal places for consistent filename matching
        filename = f'{hemi}_{i:03d}_{cut_win[0]:.3f}_{cut_win[1]:.3f}.jpg'
        full_path = os.path.join(image_folder, filename)
        if os.path.exists(full_path):
            image_files.append(full_path)

    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])

    height, width, _ = first_image.shape  # Ignore channels
    video_name = os.path.join(output_dir, f'{hemi}.mp4')  # Output video filename

    # Define the video codec and create a VideoWriter object
    # 'mp4v' is a common and widely compatible video codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"Starting video generation: {video_name}")
    for idx, image_file in enumerate(image_files):
        img = cv2.imread(image_file)

        # Extract cut_win from the filename for robustness
        parts = os.path.basename(image_file).replace('.jpg', '').split('_')
        start_time = float(parts[2])
        end_time = float(parts[3])
        time_text = f"Time: [{start_time:.3f}, {end_time:.3f}]"

        # Add time window text to the image
        # Parameters: image, text, bottom-left corner coordinates, font, font scale, color (BGR), thickness, line type
        cv2.putText(img, time_text, (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(img)  # Write the processed image to the video

    out.release()  # Release the VideoWriter object
    print(f"Video generation complete: {video_name}")

def adjust_saturation(rgb_color, avg_value,map:str='none'):
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    if map == 'none':
        gray_color = np.array([0.5, 0.5, 0.5])
        red = np.array([1.0, 0, 0])
        if avg_value <= 0.5:
            t = avg_value / 0.5
            adjusted_col = (1 - t) * gray_color + t * rgb_color
        else:
            t = (avg_value - 0.5) / 0.5
            adjusted_col = (1 - t) * rgb_color + t * red
        return np.clip(adjusted_col, 0, 1).tolist()
    else:
        # Clip the value to ensure it's within [0, 1] for colormap mapping
        clipped_val = np.clip(avg_value, 0, 1)
        # Get the colormap object
        colormap = cm.get_cmap(map)
        # Get the RGBA color from the colormap for the given value
        # The output is (R, G, B, A), where A is alpha (transparency)
        rgba_color = colormap(clipped_val)
        # Return only the RGB components as a list
        return np.array(rgba_color[:3]).tolist()

def atlas2_hist(label2atlas_raw, chs_sel, col, fig_save_dir_fm, ylim: list=[0,25], is_percentage: bool = False,
                electrode_latency_df = None, electrode_colors: list = None,sort_ROI_by: str='count',reverse_sort:bool=True):

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns  # Import seaborn
    label2atlas = {ch_sel: label2atlas_raw[ch_sel] for ch_sel in chs_sel}

    # Count the number of keys for each value (atlas region)
    value_counts = {}

    total_electrodes = len(chs_sel)
    for key, value in label2atlas.items():
        if is_percentage:
            if value in value_counts:
                value_counts[value] += 100 / total_electrodes
            else:
                value_counts[value] = 100 / total_electrodes
        else:
            if value in value_counts:
                value_counts[value] += 1
            else:
                value_counts[value] = 1

    if ((electrode_latency_df is None) or (electrode_colors is None)) and sort_ROI_by=='latency':
        raise ValueError("cannot sort ROI by latency without a latency dataframe")

    if electrode_latency_df is not None and electrode_colors is not None:
        if len(chs_sel) != len(electrode_colors):
            raise ValueError("The number of selected channels (chs_sel) must match the number of electrode colors.")

        # Ensure latency_df has electrode labels as index and a 'latency' column
        if 'latency' not in electrode_latency_df.columns:
            # Try to infer latency column if not explicitly named 'latency'
            if electrode_latency_df.shape[1] == 1:
                electrode_latency_df.columns = ['latency']
            else:
                raise ValueError("electrode_latency_df must have a 'latency' column or be a single-column DataFrame.")

        value_sums = {}
        value_means = {}
        value_nonan_counts = {}
        for key, value in label2atlas.items():
            latency_value = electrode_latency_df.loc[key, 'latency']
            if pd.notna(latency_value):
                if value in value_sums:
                    value_sums[value] += latency_value
                    value_nonan_counts[value] +=1
                else:
                    value_sums[value] = latency_value
                    value_nonan_counts[value] = 1
        for key,value in value_sums.items():
            value_means[key] = value / value_nonan_counts[key]

    # Create the figure and primary axes for the bar plot
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # Sort the values by their count in descending order
    if sort_ROI_by == 'count':
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=reverse_sort)
    elif sort_ROI_by == 'latency':
        sorted_atlas_labels_by_latency = sorted(value_means.keys(), key=lambda k: value_means[k],reverse=reverse_sort)
        sorted_values = [(label, value_counts.get(label, 0)) for label in sorted_atlas_labels_by_latency]
    atlas_labels = [item[0] for item in sorted_values]
    counts = [item[1] for item in sorted_values]

    # Bar plot on the primary y-axis (ax1)
    ax1.bar(atlas_labels, counts, color=col)
    ax1.set_xlabel('Atlas', fontsize=20)
    if is_percentage:
        ax1.set_ylabel('Percentage of Electrodes', fontsize=20)
    else:
        ax1.set_ylabel('Number of Electrodes', fontsize=20)
    ax1.set_xticks(range(len(atlas_labels))) # Set tick locations
    ax1.set_xticklabels(atlas_labels, fontsize=30, rotation=45, ha='right') # Set labels with rotation and alignment
    ax1.tick_params(axis='y', labelsize=30)
    ax1.set_ylim(ylim)

    # --- Add the secondary Y-axis and Seaborn Stripplot ---
    if electrode_latency_df is not None and electrode_colors is not None:
        if len(chs_sel) != len(electrode_colors):
            raise ValueError("The number of selected channels (chs_sel) must match the number of electrode colors.")

        # Create a secondary y-axis
        ax2 = ax1.twinx()

        # Prepare data for stripplot in a DataFrame format suitable for Seaborn
        stripplot_data = []
        custom_palette = {} # To store colors for each electrode

        for i, ch_sel in enumerate(chs_sel):
            atlas_region = label2atlas.get(ch_sel)
            if atlas_region and atlas_region in atlas_labels: # Ensure atlas region is in our sorted labels
                if ch_sel in electrode_latency_df.index:
                    latency_value = electrode_latency_df.loc[ch_sel, 'latency']
                    if pd.notna(latency_value):
                        stripplot_data.append({'Atlas': atlas_region, 'Latency': latency_value, 'Electrode': ch_sel})
                        custom_palette[ch_sel] = electrode_colors[i] # Map electrode to its color
                else:
                    print(f"Warning: Electrode {ch_sel} not found in electrode_latency_df. Skipping latency plot for this electrode.")
            else:
                print(f"Warning: Electrode {ch_sel} atlas region '{atlas_region}' not in the main bar plot. Skipping latency plot for this electrode.")

        if stripplot_data: # Only plot if there's data
            df_stripplot = pd.DataFrame(stripplot_data)

            # Define the order of x-axis categories for Seaborn to match the bar plot
            order = atlas_labels

            # Use stripplot with hue set to 'Electrode' and a custom palette
            sns.stripplot(
                data=df_stripplot,
                x='Atlas',
                y='Latency',
                hue='Electrode', # Use 'Electrode' for unique color mapping
                palette=custom_palette, # Apply the custom palette
                jitter=0.2, # Adjust jitter to spread points horizontally
                dodge=False, # Do not dodge points based on hue
                ax=ax2, # Plot on the secondary axis
                s=6, # Marker size
                legend=False # Hide the legend for individual electrodes
            )

            ax2.set_ylabel('Latency', fontsize=20, color='grey')
            ax2.tick_params(axis='y', labelsize=30, colors='grey')
            # ax2.set_ylim(bottom=0)
            ax2.set_xlabel('') # Hide the x-axis label for the secondary plot as it's shared

    plt.tight_layout()
    plt.savefig(fig_save_dir_fm, dpi=300)
    plt.close()

def align_channel_data(subj_data, good_labeled_chs, org_labeled_chs, glm_out: str='beta',add_subj_labels: bool=True):
    """
    Aligns channel data by adding missing channels with NaN values.

    This function now supports subject data with either 2 dimensions (channels, timepoints)
    or 3 dimensions (trials, channels, timepoints). It aligns the channels to a
    predefined list of original channel labels, filling in NaN for any channels
    present in the `org_labeled_chs` but not in the `subj_data`.

    Parameters:
    -----------
    subj_data : numpy.ndarray
        Original subject data array.
        Expected shapes:
        - (n_channels, n_timepoints) for 2D data (e.g., 'beta' output without trials).
        - (n_trials, n_channels, n_timepoints) for 3D data (e.g., 'beta' output with trials).
        - If `glm_out` is 'r2', `n_timepoints` will typically be 1.
    good_labeled_chs : list
        List of current channel labels present in the `subj_data` array.
        Each label is expected to include a subject prefix (e.g., ['D001 ch1', 'D001 ch2']).
    org_labeled_chs : list
        List of all original/reference channel labels that the data should be aligned to.
        These should be just the channel names (e.g., ['ch1', 'ch2', 'ch3']).
    glm_out : str, optional
        Type of GLM output. This parameter primarily influences the interpretation
        of the last dimension of `subj_data`.
        - 'beta': Assumes the last dimension represents timepoints.
        - 'r2': Assumes the last dimension represents a single value (e.g., R-squared).
        Default is 'beta'.

    Returns:
    --------
    numpy.ndarray
        Aligned data array with all original channels, filled with NaNs where channels were missing.
        The shape will match the input `subj_data` dimensionality but with the number of
        channels expanded to `len(org_labeled_chs)`.
    list
        Updated channel labels for the aligned data, including the subject prefix.
    """
    import numpy as np
    # Get the number of dimensions of the input subject data
    ndim = subj_data.ndim

    # Validate input data dimensions
    if ndim not in [2, 3]:
        raise ValueError(
            "Subject data must be 2-dimensional (channels x timepoints) "
            "or 3-dimensional (trials x channels x timepoints)"
        )

    # Extract just the channel names from good_labeled_chs by removing the subject prefix.
    # Assumes channel labels are in the format "SubjectID ChannelName"
    current_chs = [ch.split()[-1] for ch in good_labeled_chs]

    # Determine the shape for the initialized aligned_data array based on input dimensions
    if ndim == 2:
        # For 2D data: (n_channels_in_input_data, n_timepoints_or_single_value)
        n_timepoints_or_single_value = subj_data.shape[1]
        output_shape = (len(org_labeled_chs), n_timepoints_or_single_value)
    else: # ndim == 3
        # For 3D data: (n_trials, n_channels_in_input_data, n_timepoints_or_single_value)
        n_trials = subj_data.shape[0]
        n_timepoints_or_single_value = subj_data.shape[2]
        output_shape = (n_trials, len(org_labeled_chs), n_timepoints_or_single_value)

    # Initialize the new data array with NaN values. This array will store the aligned data.
    aligned_data = np.full(output_shape, np.nan)

    # Create a dictionary mapping channel names from the `current_chs` list to their
    # corresponding row/slice index in the input `subj_data`.
    current_ch_indices = {ch: idx for idx, ch in enumerate(current_chs)}

    # Iterate through the `org_labeled_chs` (the desired full list of channels).
    # For each channel, if it exists in the input `subj_data`, copy its data into the
    # correct position in the `aligned_data` array.
    for new_idx, org_ch in enumerate(org_labeled_chs):
        if org_ch in current_ch_indices:
            # Get the index of the current channel in the original subj_data
            source_idx = current_ch_indices[org_ch]
            if ndim == 2:
                # If 2D, copy the entire row (channel data) from subj_data to aligned_data
                aligned_data[new_idx] = subj_data[source_idx]
            else: # If 3D, copy the entire slice for that channel across all trials and timepoints
                aligned_data[:, new_idx, :] = subj_data[:, source_idx, :]

    # Create new channel labels with subject prefix
    subject = good_labeled_chs[0].split()[0]  # Extract subject from first channel
    subject = 'D' + subject[1:].lstrip('0')
    # May try this:
    # arr.labels[1]=Labels(['D' + subject[1:].lstrip('0')]) @ Labels(arr.labels[1])
    if add_subj_labels:
        aligned_chs = [f"{subject}-{ch}" for ch in org_labeled_chs]
    else:
        aligned_chs = org_labeled_chs

    return aligned_data, aligned_chs

def plot_wave(data_in,sig_idx,con_label,col,Lstyle,bsl_crr,errtype='se',normalize=False,ylim: list=[-0.5,3.5]):

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
        mean_waveform = mean_waveform - np.nanmin(mean_waveform[:51])
    if errtype == 'std':
        sem_waveform = np.nanstd(data_selected, axis=0)
    elif errtype == 'se':
        sem_waveform = np.nanstd(data_selected, axis=0) / np.sqrt(np.sum(~np.isnan(data_selected), axis=0))  # SEM ignoring NaNs
    # Plot the mean waveform
    plt.plot(times, mean_waveform, label=con_label, color=col,linestyle=Lstyle)

    # Add shaded region for SEM
    plt.fill_between(times, mean_waveform - sem_waveform, mean_waveform + sem_waveform, color=col, alpha=0.3)

    plt.xlabel('Time (s)',fontsize=10)
    plt.ylim(ylim)
    plt.xticks(tick_labels)

def min_max_normalize_ignore_nan(arr):
    import numpy as np
    not_nan_mask = ~np.isnan(arr)
    valid_data = arr[not_nan_mask]
    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)
    IQR = Q3 - Q1
    # Define outlier bounds (3 * IQR)
    upper_bound = Q3 + 3 * IQR
    lower_bound = Q1 - 3 * IQR
    # Create a copy to avoid modifying the original array
    winsorized_arr = np.copy(arr)
    # Winsorize outliers
    winsorized_arr[not_nan_mask & (arr > upper_bound)] = upper_bound
    winsorized_arr[not_nan_mask & (arr < lower_bound)] = lower_bound
    # Perform min-max normalization on the winsorized data
    winsorized_valid_data = winsorized_arr[not_nan_mask]
    min_val = np.min(winsorized_valid_data)
    max_val = np.max(winsorized_valid_data)

    if (max_val - min_val) == 0:
        # Avoid division by zero if all non-NaN winsorized values are the same
        normalized_arr = np.full_like(arr, 0.0, dtype=float)
        normalized_arr[not_nan_mask] = 0.5 if min_val == max_val else 0.0 # Or 0.0 if you prefer
    else:
        normalized_arr = np.copy(arr).astype(float) # Ensure float type for division
        normalized_arr[not_nan_mask] = (winsorized_arr[not_nan_mask] - min_val) / (max_val - min_val)

    return normalized_arr

def time_avg_select(data_in,sig_idx_lst,normalize:bool=False):

    import numpy as np
    import matplotlib.pyplot as plt
    chs=data_in.labels[0]
    data=data_in.__array__()
    data_avg=np.nanmean(data,axis=1)
    if normalize:
        data_avg=min_max_normalize_ignore_nan(data_avg)
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

def select_electrodes(data_in, sig_idx_lst):
    """
    Selects specific electrodes from the input data (trials, channels, time_points)
    without performing a time average.
    """
    from ieeg.arrays.label import LabeledArray

    chs = data_in.labels[0]
    times = data_in.labels[1]

    data = data_in.__array__() # Original data, shape: (trials, channels, time_points)

    data_selected_final_groups = []
    chs_selected_final = []

    sig_idx_lst = list(sig_idx_lst)
    data_selected_final = data[sig_idx_lst, :]
    chs_selected_final = chs[sig_idx_lst]
    # Concatenate results along the channel axis (axis=1 in the new shape)
    # The shape will be (trials, total_selected_channels, time_points)
    labels_all = [chs_selected_final, times]
    data_out = LabeledArray(data_selected_final, labels_all)

    return data_out

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

def hickok_roi_sphere(df_coords,thres: float=15):

    # The fs_average mri space is actually MNI space:
    # "And because fsaverage is special in that it’s already in MNI space (its MRI-to-MNI transform is identity),
    # it should land in the equivalent anatomical location"
    # (source: https://mne.tools/stable/auto_tutorials/forward/
    # 50_background_freesurfer_mne.html)
    from scipy.spatial.distance import euclidean
    from collections import defaultdict
    hickok_roi_sig_idx = defaultdict(set)

    # Define the activation peaks
    activation_peaks = {
        'lIFG': (-56, 8, 20),
        'Spt': (-54, -40, 20),
        'lPMC': (-50, -4, 46),
        'lIPL': (-42, -50, 42)
    }

    hickok_roi_labels = {}

    # Iterate through each electrode in the DataFrame
    for index, row in df_coords.iterrows():
        subj = row['subj']
        label = row['label']
        electrode_coord = (row['x'], row['y'], row['z'])
        assigned_roi = 'N/A'
        min_distance = float('inf')

        # Check distance to each activation peak
        for roi_name, peak_coord in activation_peaks.items():
            distance = euclidean(electrode_coord, peak_coord)
            if distance <= thres:
                # If multiple peaks are within the threshold, assign to the closest one
                if distance < min_distance:
                    min_distance = distance
                    assigned_roi = roi_name

        # Create a unique identifier for the electrode
        electrode_id = f"{subj}-{label}"
        hickok_roi_labels[electrode_id] = assigned_roi
        hickok_roi_sig_idx[assigned_roi].add(index)

    return hickok_roi_labels, hickok_roi_sig_idx


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

def get_sig_roi_counts(hickok_roi_labels, sig_idx):
    """
    Calculates the number of significant electrodes per ROI when sig_idx is a set of indices.

    Args:
        hickok_roi_labels (dict): A dictionary where keys are electrode IDs
                                  and values are their assigned ROIs (e.g., 'lIFG' or 'N/A').
        sig_idx (set): A set containing the numerical indices of the significant electrodes.

    Returns:
        pd.Series: A Series with ROI labels as index and counts of significant electrodes as values.
                   Includes all known ROIs from hickok_roi_labels, with 0 if no significant electrodes.
    """
    import pandas as pd
    from collections import Counter

    keys = list(hickok_roi_labels.keys())

    # Collect ROI labels for significant, non-N/A electrodes
    selected_labels = [
        hickok_roi_labels[keys[i]]
        for i in sig_idx
        if hickok_roi_labels[keys[i]] != 'N/A'
    ]

    # Get all unique ROI labels (excluding 'N/A') from the full set of electrodes
    all_rois = set(roi for roi in hickok_roi_labels.values() if roi != 'N/A')

    # Count significant electrodes per ROI
    label_counts = Counter(selected_labels)

    # Create a Series, ensuring all possible ROIs are present, with 0 if no significant electrodes
    roi_series = pd.Series(label_counts, index=list(all_rois)).fillna(0).astype(int)

    # Sort the series by index (ROI label) for consistent order
    roi_series = roi_series.sort_index()

    return roi_series

def plot_roi_counts_comparison(df_roi_counts, savedir, y_label='Number of Electrodes', title_suffix='',ylim: list=[0,30]):
    """
    Plots the number of electrodes per ROI for different significant conditions.

    Args:
        df_roi_counts (pd.DataFrame): A DataFrame where index is ROI label,
                                      columns are sig_idx condition names,
                                      and values are electrode counts.
        savedir (str): The directory to save the individual plots.
        y_label (str, optional): The label for the y-axis. Defaults to 'Number of Electrodes'.
        title_suffix (str, optional): A suffix to add to the plot title. Defaults to ''.
    """
    # Create a figure for each ROI
    import matplotlib.pyplot as plt
    for roi_label in df_roi_counts.index:
        plt.figure(figsize=(6, 5)) # Use the same figure size as your original function

        # Get the counts for the current ROI across all sig_idx conditions
        counts = df_roi_counts.loc[roi_label]

        # Plotting the bar chart
        # x-axis will be the sig_idx names (DataFrame columns)
        # y-axis will be the counts
        plt.bar(counts.index, counts.values, color='skyblue') # You can define a color palette if needed

        # Set titles and labels
        plt.title(f'{roi_label} - Significant Electrodes {title_suffix}', fontsize=20)
        # plt.xlabel('Significance Condition', fontsize=15) # You can uncomment if needed
        plt.ylabel(y_label, fontsize=15)

        # Set tick parameters, adapting from your original function
        original_labels = counts.index
        # Process each label to keep only the last two parts
        processed_labels = []
        for label in original_labels:
            parts = label.split('/')
            if len(parts) >= 2:
                processed_labels.append('/'.join(parts[-2:]))
            else:
                processed_labels.append(label)  # Fallback if less than 2 parts
        # Set the x-ticks with the processed labels
        plt.xticks(ticks=range(len(original_labels)), labels=processed_labels,
                   fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=12) # Adjusted fontsize for better readability given the smaller figure size
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(f"{savedir}_{roi_label}_counts.png", dpi=300)
        plt.close() # Close the figure to free up memory

def get_coor(chs,method: str='individual'):
    """
    Given a list of channels like ['D23-RASF1', 'D23-RASF2', ...],
    this function reads each subject's electrode file and returns
    a dataframe of 3D coordinates for the specified electrodes.

    method:
        'individual': get individual electrode coordinates
        'group': use fs_average to transform to group space
        'ras': old method. read ras locations directly. ! May be wrong!
    """
    # Parse into (subject, label)
    import pandas as pd
    import os
    import mne
    import numpy as np
    from ieeg.viz.mri import subject_to_info, force2frame, get_sub_dir
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

        if method == 'individual':
            trans = mne.transforms.Transform(fro='head', to='mri')
        elif method == 'group':
            # get_subject_dir
            to_fsaverage = mne.read_talxfm(subj, get_sub_dir())
            trans = mne.transforms.Transform(fro='head', to='mri',trans=to_fsaverage['trans'])
        elif method == 'ras':
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

        if method == 'individual' or method == 'group':
            info = subject_to_info(subj)
            montage = info.get_montage()
            force2frame(montage, trans.from_str)
            montage.apply_trans(trans)
            coord_map = {k: v for k, v in montage.get_positions()['ch_pos'].items()}

        # Add matching labels
        for label in labels:
            if label in coord_map:
                x, y, z = coord_map[label]
                if method == 'individual' or method == 'group':
                    x = 1000*x
                    y = 1000*y
                    z = 1000*z
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