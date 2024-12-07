import pandas as pd
import re
import os


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
        df_filtered = df[~df['trial_type'].isin(["BAD boundary", "EDGE boundary"])]

        # Overwrite the original file with the filtered data
        df_filtered.to_csv(input_file, sep="\t", index=False)
        print(f"Processed and replaced the original file: {input_file}")