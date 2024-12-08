import pandas as pd
import re
import os
import glob

def get_unused_chs(folder_path):
    """
    Compare electrode names in `*_channels.tsv` and `*electrodes.tsv` within a folder.
    Return the electrode names that are in `_channels.tsv` but not in `*electrodes.tsv`.

    Parameters:
    - folder_path (str): Path to the folder containing the TSV files.

    Returns:
    - list: Electrode names present in `_channels.tsv` but missing in `*electrodes.tsv`.
    """
    # Find the electrodes.tsv and channels.tsv files in the folder
    electrodes_files = glob.glob(os.path.join(folder_path, "*electrodes.tsv"))
    channels_files = glob.glob(os.path.join(folder_path, "*channels.tsv"))

    # Ensure only one of each file type exists
    if len(electrodes_files) != 1:
        raise FileNotFoundError(f"Expected exactly one *electrodes.tsv file, but found {len(electrodes_files)}.")
    if len(channels_files) != 1:
        raise FileNotFoundError(f"Expected exactly one *channels.tsv file, but found {len(channels_files)}.")

    electrodes_file = electrodes_files[0]
    channels_file = channels_files[0]

    # Load the TSV files into DataFrames
    electrodes_df = pd.read_csv(electrodes_file, sep="\t")
    channels_df = pd.read_csv(channels_file, sep="\t")

    # Ensure the necessary 'name' column exists in both files
    if 'name' not in electrodes_df.columns or 'name' not in channels_df.columns:
        raise ValueError("Both TSV files must contain a 'name' column.")

    # Remove the 'Trigger' electrode from the channels list
    channels_df = channels_df[channels_df['name'] != 'Trigger']
    channels_df = channels_df[channels_df['type'] != 'TRIG']

    # Create sets of electrode names from each file
    electrodes_set = set(electrodes_df['name'])
    channels_set = set(channels_df['name'])

    # Identify names present in channels.tsv but not in electrodes.tsv
    missing_electrodes = list(channels_set - electrodes_set)

    if pd.isna(missing_electrodes).any():
        return []
    else:
        return missing_electrodes



def update_tsv(subj, search_dir='.'):
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
    pattern = f"sub-{subj}_task-LexicalDecRepDelay_acq-.+?_run-.+?_desc-a_events.tsv"

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

import os
import re
import pandas as pd

def detect_outlier(subj, search_dir='.'):
    """
    Detect outliers in files matching a specific pattern for a given subject.
    
    Args:
        subj (str): Subject identifier.
        search_dir (str): Directory to search for the files. Defaults to the current directory.
    
    Returns:
        int: 1 if any file contains 'outlier' in the 'status_description' column, 0 otherwise.
    """
    # Construct the pattern to match the filenames based on `subj`
    pattern = f"sub-{subj}_task-LexicalDecRepDelay_acq-.+?_run-.+?_desc-a_channels.tsv"
    
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