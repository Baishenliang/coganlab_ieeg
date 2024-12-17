# %% step up
import os
import mne

subject = 'D0053'
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
bids_root = os.path.join(LAB_root,'BIDS-1.0_LexicalDecRepDelay','BIDS')
subj_gamma_stats_dir = os.path.join(bids_root, "derivatives", "stats", subject)

# %% load data

stat_type='pval'
con='Auditory'
contrast='ave'

match stat_type:
    case "zscore":
        fif_read = lambda f: mne.read_epochs(f, False, preload=True)
    case "power":
        fif_read = lambda f: mne.read_epochs(f, False, preload=True)
    case "significance":
        fif_read = mne.read_evokeds
    case "pval":
        fif_read = mne.read_evokeds

file_dir = os.path.join(subj_gamma_stats_dir,f'{con}_{stat_type}-{contrast}.fif')
data = fif_read(file_dir)