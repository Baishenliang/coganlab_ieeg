import pandas as pd
import re
import os
import glob
import numpy as np
from matplotlib import pyplot as plt


class NoSEEGChannelsError(ValueError):
    """No sEEG contacts remain for bipolar referencing."""


def drop_unlocalized_seeg(raw, coord_map, subject='', log_file=None):
    """Drop sEEG contacts without finite XYZ coordinates, in place.

    coord_map must describe the selected coordinate source; units do not affect
    validity checking. Non-sEEG channels are left unchanged. Return (raw, dropped).
    Remaining contacts can bridge removed numbers in subsequent bipolar pairing,
    subject to the usual distance and turning-angle gates. No interpolation.
    Raise NoSEEGChannelsError if no localized sEEG contacts remain.
    """
    names = [name for name, kind in zip(raw.ch_names, raw.get_channel_types())
             if kind == 'seeg']
    dropped = []
    for name in names:
        xyz = np.asarray(coord_map.get(name, []), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            dropped.append(name)
    message = (f"{subject}, Coordinate QC: excluding {len(dropped)} unlocalized "
               f"sEEG contacts: {dropped}; remaining: {len(names) - len(dropped)}")
    if log_file is not None:
        log_file.write(message + '\n')
    print(message)
    if len(dropped) == len(names):
        raise NoSEEGChannelsError("No sEEG contacts with valid localization remain")
    if dropped:
        raw.drop_channels(dropped)
    return raw, dropped


def bipolar_reference(raw, max_pair_dist_mm=20.0, max_turn_deg=60.0,
                      copy=True, return_pairs=False):
    """Apply Nanlin-style geometry-gated bipolar referencing to loaded sEEG.

    Call after dropping bad channels. Contacts are grouped by name prefix and
    sorted numerically; gaps such as A1--A3 are allowed. Following Nanlin's
    ``lib/shaft_bipolar.py``, a distance or turning-angle break starts a new
    shaft segment. Coordinates must be valid MNE montage coordinates in meters.

    The output contains only accepted bipolar channels, named ``A1-A3``, with
    signals A1 minus A3 and midpoint coordinates. Unpaired contacts are omitted.
    MNE preserves recording timing, annotations (including run boundaries), and
    source filenames. Input is unchanged by default; copy=False reduces memory
    use but modifies raw. Data must already be loaded. Non-sEEG channels are
    excluded before referencing; NoSEEGChannelsError is raised if none remain.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preloaded, not-yet-referenced data with bad channels already removed.
    max_pair_dist_mm : float
        Maximum distance between remaining neighboring contacts (default 20).
    max_turn_deg : float
        Maximum turn within a segment (default 60 degrees).
    copy : bool
        Copy the input before applying the reference (default True).
    return_pairs : bool
        Also return a DataFrame of every candidate pair, with acceptance and
        rejection reason, for inspection or saving as TSV.

    Returns
    -------
    referenced : mne.io.BaseRaw
        Bipolar data in prefix/numeric order.
    pairs : pandas.DataFrame
        Only returned when return_pairs=True. Angles are NaN at segment starts
        or when the distance gate already rejects the candidate.
    """
    import mne

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an MNE Raw object")
    if not raw.preload:
        raise ValueError("Load data with raw.load_data() before referencing")
    seeg_names = [name for name, kind in zip(raw.ch_names, raw.get_channel_types())
                  if kind == 'seeg']
    if not seeg_names:
        raise NoSEEGChannelsError("No sEEG channels remain after channel removal")
    if set(raw.info['bads']).intersection(seeg_names):
        raise ValueError("Drop bad channels before bipolar_reference()")
    if not np.isfinite(max_pair_dist_mm) or max_pair_dist_mm <= 0:
        raise ValueError("max_pair_dist_mm must be positive and finite")
    if not np.isfinite(max_turn_deg) or not 0 <= max_turn_deg <= 180:
        raise ValueError("max_turn_deg must be between 0 and 180")
    montage = raw.get_montage()
    if montage is None:
        raise ValueError("A montage with contact coordinates is required")
    positions = montage.get_positions()
    coords = positions['ch_pos']
    groups = {}
    for name in seeg_names:
        match = re.fullmatch(r'(.+?)(\d+)', name)
        if match is None:
            raise ValueError(f"Cannot parse contact name: {name}")
        if name not in coords or not np.isfinite(coords[name]).all():
            raise ValueError(f"Missing or invalid coordinate for {name}")
        prefix, number = match.groups()
        groups.setdefault(prefix, []).append((int(number), name))

    rows = []
    for prefix, contacts in sorted(groups.items()):
        contacts.sort()
        if len({number for number, _ in contacts}) != len(contacts):
            raise ValueError(f"Duplicate contact numbers in {prefix}")
        segment = [contacts[0][1]]
        segment_id = 0
        for _, ch2 in contacts[1:]:
            ch1 = segment[-1]
            vector = (coords[ch2] - coords[ch1]) * 1000.0
            distance = float(np.linalg.norm(vector))
            angle = np.nan
            reason = 'distance' if distance > max_pair_dist_mm else ''
            if not reason and len(segment) >= 2:
                previous = (coords[ch1] - coords[segment[-2]]) * 1000.0
                norm = np.linalg.norm(previous)
                angle = 0.0
                if norm >= 1e-12 and distance >= 1e-12:
                    cosine = np.dot(previous, vector) / (norm * distance)
                    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
                if angle > max_turn_deg:
                    reason = 'turn_angle'
            rows.append(dict(prefix=prefix, segment_id=segment_id,
                             ch1=ch1, ch2=ch2, bipolar_name=f'{ch1}-{ch2}',
                             dist_mm=distance, turn_deg=angle,
                             accepted=not bool(reason), reason=reason))
            if reason:
                segment = [ch2]
                segment_id += 1
            else:
                segment.append(ch2)

    pairs = pd.DataFrame(rows, columns=[
        'prefix', 'segment_id', 'ch1', 'ch2', 'bipolar_name',
        'dist_mm', 'turn_deg', 'accepted', 'reason'])
    accepted = [row for row in rows if row['accepted']]
    if not accepted:
        raise ValueError("No bipolar pairs passed the geometry checks")
    names = [row['bipolar_name'] for row in accepted]
    # Remove other channel types before MNE's matrix multiplication, so their
    # signals (including any NaNs) cannot affect the bipolar data.
    working = raw.copy() if copy else raw
    working.pick(seeg_names)
    referenced = mne.set_bipolar_reference(
        working, anode=[row['ch1'] for row in accepted],
        cathode=[row['ch2'] for row in accepted], ch_name=names,
        copy=False, drop_refs=True)
    # MNE drops paired sources; explicitly remove any unpaired originals too.
    referenced.pick(names)
    midpoints = {row['bipolar_name']: (coords[row['ch1']] + coords[row['ch2']]) / 2
                 for row in accepted}
    referenced.set_montage(mne.channels.make_dig_montage(
        ch_pos=midpoints, coord_frame=positions['coord_frame'],
        nasion=positions['nasion'], lpa=positions['lpa'], rpa=positions['rpa']))
    return (referenced, pairs) if return_pairs else referenced


def update_tsv(subj, search_dir,task_tag):
    """
    Searches for all TSV files matching the given `subj` identifier, processes each one by removing specific rows,
    and saves the updated file, overwriting the original.

    - The script searches for the files based on the `subj` identifier.
    - It processes all matching files and removes rows where `trial_type` is "BAD boundary" or "EDGE boundary".
    - Each modified TSV file replaces the original file.

    Parameters:
    - subj: str, the subject identifier (e.g., 'D53').
    - search_dir: str, directory to search for the files (default is the current directory).

    Raises:
    - ValueError: If no files or more than one matching file are found for a `subj` and those files have issues.
    """
    # Construct the pattern to match the filenames based on `subj`
    task_tag_clean = task_tag.replace('_', '')
    pattern = f"sub-{subj}_task-{task_tag_clean}_acq-.+?_run-.+?_desc-clean_events.tsv"

    # Search for all files in the specified directory that match the pattern
    files = [f for f in os.listdir(search_dir) if re.match(pattern, f)]

    if not files:
        raise ValueError(f"No files matching the pattern found for subj {subj}.")

    # If matching files are found, process each one
    for file in files:
        input_file = os.path.join(search_dir, file)

        # Read the TSV file
        df = pd.read_csv(input_file, sep="\t")

        # Remove rows with trial_type "BAD boundary" or "EDGE boundary"
        df_filtered = df[~df['trial_type'].isin(["BAD boundary", "EDGE boundary", "BAD_ACQ_SKIP"])]

        # Overwrite the original file with the filtered data
        df_filtered.to_csv(input_file, sep="\t", index=False)
        print(f"Processed and replaced the original file: {input_file}")

def detect_outlier(subj, search_dir, task_tag):
    """
    Detect outliers in files matching a specific pattern for a given subject.
    
    Args:
        task_tag: tag for task
        subj (str): Subject identifier.
        search_dir (str): Directory to search for the files. Defaults to the current directory.
    
    Returns:
        int: 1 if any file contains 'outlier' in the 'status_description' column, 0 otherwise.
    """
    # Construct the pattern to match the filenames based on `subj`
    task_tag_clean = task_tag.replace('_', '')
    pattern = f"sub-{subj}_task-{task_tag_clean}_acq-.+?_run-.+?_desc-clean_channels.tsv"
    
    # Search for all files in the specified directory that match the pattern
    files = [f for f in os.listdir(search_dir) if re.match(pattern, f)]
    
    if not files:
        raise ValueError(f"No files matching the pattern found for subj {subj}.")
    
    # Check each file for 'outlier' in the 'status_description' column
    for file in files:
        file_path = os.path.join(search_dir, file)
        # Read the file assuming it's a tab-separated values (TSV) file
        data = pd.read_csv(file_path, sep='\t')
        
        # If 'status_description' column contains 'outlier', return 1
        if 'status_description' in data.columns and 'outlier' in data['status_description'].values:
            return 1
    
    return 0

def load_eeg_chs(subject):
        
    """
    Load the eeg channels for a specific subject.

    Args:
        subject (str): Subject identifier.

    Returns:
        list: List of eeg channel names.
    """
    eeg_chs_loc=os.path.join('data','eeg_chans',f'{subject}_eeg_chans.csv')
    df = pd.read_csv(eeg_chs_loc, header=None)
    df.columns = ['eeg_chs']
    eeg_chs = df['eeg_chs'].tolist()
    return eeg_chs

def load_muscle_chs(subject):
        
    """
    Load the muscle channels for a specific subject.

    Args:
        subject (str): Subject identifier.

    Returns:
        list: List of muscle channel names.
    """
    muscle_chs_loc=os.path.join('data','muscle_chans',f'{subject}_muscle_chans.csv')
    df = pd.read_csv(muscle_chs_loc, header=None)
    df.columns = ['muscle_chs']
    muscle_chs = df['muscle_chs'].tolist()
    return muscle_chs

def update_muscle_chs(subj, search_dir,task_tag):
    """
    Update the status and status_description of specified electrodes in the subject's channel files
    based on the muscle channels loaded for the subject.

    Args:
        subj (str): Subject identifier.
        search_dir (str): Directory to search for the files. Defaults to the current directory.

    Returns:
        None
    """

    # Load muscle channels for the subject
    electrode_list = load_muscle_chs(subj)

    # Construct the pattern to match the filenames based on `subj`
    task_tag_clean = task_tag.replace('_', '')
    pattern = f"sub-{subj}_task-{task_tag_clean}_acq-.+?_run-.+?_desc-clean_channels.tsv"

    # Search for all files in the specified directory that match the pattern
    files = [f for f in os.listdir(search_dir) if re.match(pattern, f)]

    if not files:
        raise ValueError(f"No files matching the pattern found for subj {subj}.")

    for file in files:
        file_path = os.path.join(search_dir, file)
        # Read the file assuming it's a tab-separated values (TSV) file
        data = pd.read_csv(file_path, sep='\t')

        # Check and update the status and status_description for the electrodes
        for electrode in electrode_list:
            if electrode in data['name'].values:
                idx = data[data['name'] == electrode].index[0]
                if data.at[idx, 'status'] != 'bad' or data.at[idx, 'status_description'] != 'muscle':
                    data.at[idx, 'status'] = 'bad'
                    data.at[idx, 'status_description'] = 'muscle'

        # Save the updated file back to disk
        data.to_csv(file_path, sep='\t', index=False)

    print(f"Updated files for subject {subj}: {files}")

def plot_save_gammamask(mask,epoch_mask,subj_gamma_dir,fname):
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap='Reds')
    channel_names=epoch_mask.ch_names[::5]
    ax.set_yticks(range(0,len(channel_names)*5,5))
    ax.set_yticklabels(channel_names)
    time_stamps=epoch_mask.times[::20]
    ax.set_xticks(range(0,len(time_stamps)*20,20))
    ax.set_xticklabels(time_stamps)
    try:
        zero_time_index = np.where(epoch_mask.times == 0)[0][0]
        ax.axvline(x=zero_time_index, color='black', linestyle='--', linewidth=1)
    except Exception as e:
        print('no zero time found')
    fig.savefig(os.path.join(subj_gamma_dir,fname), dpi=300)
    plt.close(fig)
