import pandas as pd
import os

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")

def get_status(subj, base_dir, output_file):
    """
    Processes the "sub-XXX_task-LexicalDecRepDelay_acq-01_run-01_desc-a_channels.tsv" file
    for a given subject to extract muscle channel names and save them to a CSV.

    Parameters:
    - subj: str, the subject ID (e.g., "D0063").
    - base_dir: str, the base directory where subject folders are located.
    - output_file: str, the path to save the extracted muscle channel names.

    Workflow:
    1. Constructs the path to the TSV file for the given subject.
    2. Checks if the file exists; if not, logs an error and skips processing.
    3. Reads the TSV file and filters rows where 'status_description' equals "muscle".
    4. Extracts the 'name' column for these rows and saves it to a CSV file.

    Notes:
    - If no muscle channels are found, the function logs a message and exits.
    - Handles errors during file reading or writing gracefully.
    """
    # Construct the path to the subject's TSV file
    subj_dir = base_dir
    file_path = os.path.join(subj_dir, f"sub-{subj}_task-LexicalDecRepDelay_acq-01_run-01_desc-a_channels.tsv")

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep="\t")
        print(f"Processing file: {file_path}")

        # Filter rows where 'status_description' is 'muscle'
        muscle_channels = df[df['status_description'] == 'muscle']['name']

        # Check if any muscle channels were found
        if muscle_channels.empty:
            print("No muscle channels found.")
            return

        # Save the muscle channel names to a CSV file
        muscle_channels.to_csv(output_file, index=False, header=False)
        print(f"Saved muscle channel names to {output_file}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

subjs=["D0053", "D0054", "D0055", "D0057", "D0059","D0063",  "D0065", "D0066", "D0068", "D0069", "D0070", "D0071", "D0077", "D0079", "D0081", "D0094", "D0096", "D0101", "D0102", "D0103", "D0107"]

for subj in subjs:
    base_dir = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', 'derivatives', 'a', f'sub-{subj}',
                           'ieeg')
    output_path=os.path.join('data','muscle_chans',f'{subj}_muscle_chans.csv')
    get_status(subj, base_dir, output_path)