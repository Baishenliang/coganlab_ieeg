def load_stats(stat_type,con,contrast):
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

    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    stats_root = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
    subjs = [name for name in os.listdir(stats_root) if os.path.isdir(os.path.join(stats_root, name))]

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
    return data