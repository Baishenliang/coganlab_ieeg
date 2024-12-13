# %%
# see: https://github.com/coganlab/SentenceRep_analysis/blob/main/analysis/fix/clean.py
import os
import re
import pandas as pd
from ieeg.navigate import channel_outlier_marker
from ieeg.io import get_data, raw_from_layout, save_derivative, update
from ieeg.mt_filter import line_filter

def update_tsv(subj, search_dir='.'):

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


# %% check if currently running a slurm job
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace")
    subj = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subj = 0

# %% Load Data
layout = get_data("LexicalDecRepDelay", LAB_root)
subject = f"D{subj:04}"
raw = raw_from_layout(layout, subject=subject, extension=".edf", desc=None,
                      preload=True)

# line noise filtering
line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
            filter_length='700ms', freqs=[60], notch_widths=20)
line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
            filter_length='20s', freqs=[60, 120, 180, 240],
            notch_widths=20)

# %% save data
bids_root = os.path.join(LAB_root,'BIDS-1.0_LexicalDecRepDelay','BIDS')
if not os.path.exists(os.path.join(bids_root, "derivatives")):
    os.mkdir(os.path.join(bids_root, "derivatives"))
    os.mkdir(os.path.join(bids_root, "derivatives", "clean"))
elif not os.path.exists(os.path.join(bids_root, "derivatives", "clean")):
    os.mkdir(os.path.join(bids_root, "derivatives", "clean"))
save_derivative(raw, layout, "clean", True)
del layout, raw

# %% remove "bad boundary" in events.tsv
tsv_loc = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', 'derivatives', 'clean', f'sub-{subject}',
                        'ieeg')
update_tsv(subject, tsv_loc)

## %% Mark outlier channels
layout = get_data("LexicalDecRepDelay", root=LAB_root)
raw = raw_from_layout(layout.derivatives['derivatives/clean'], subject=subject, desc='clean', extension='.edf',
                        preload=True)
derivative_loc = os.path.join(LAB_root, "BIDS-1.0_LexicalDecRepDelay","BIDS","derivatives","clean",f"sub-{subject}","ieeg")
raw.info['bads'] = channel_outlier_marker(raw, 3, 2)
update(raw, layout, "outlier")
del layout, raw