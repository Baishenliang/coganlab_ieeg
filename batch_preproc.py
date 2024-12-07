"""
    Batch preprocessing scripts for lexical delay task.
    Including: line noise filtering, outlier channels removal, and wavelet.
    Parameters are equal to Sentence Rep processing code unless a change is necessary (NEEDED TO BE RE-CHECKED!):
    https://github.com/coganlab/SentenceRep_analysis
    From Baishen Liang.
"""
# Preparation:

import os
import mne
import datetime
import numpy as np
from ieeg.navigate import crop_empty_data, channel_outlier_marker, trial_ieeg, outliers_to_nan
from ieeg.mt_filter import line_filter
from ieeg.io import get_data, raw_from_layout, save_derivative, update
from ieeg.calc.scaling import rescale
from ieeg.viz.ensemble import chan_grid
from ieeg.timefreq.utils import crop_pad, wavelet_scaleogram
from ieeg.viz.parula import parula_map
from bsliang_utils import get_unused_chs, update_tsv
from matplotlib import pyplot as plt

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
save_dir=os.path.join(HOME, "Box", "CoganLab", "D_Data","LexicalDecRepDelay","Baishen_Figs")
layout = get_data("LexicalDecRepDelay", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

# Log
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file_path = os.path.join('data', 'logs', f'batch_preproc_{current_time}.txt')
log_file = open(log_file_path, 'a')

# Subj list
subjects=["D0053", "D0054", "D0055", "D0057", "D0059", "D0063", "D0065", "D0066", "D0068", "D0069", "D0070", "D0071", "D0077", "D0079", "D0081", "D0094", "D0096", "D0101", "D0102", "D0103", "D0107"]

for subject in subjects:

    ## Line Noise Filtering
    try:
        log_file.write(f"{datetime.datetime.now()}, {subject}, Executing line noise filter\n")

        # load BIDS raw data
        raw = raw_from_layout(layout, subject=subject, preload=True, extension='.edf')

        # get and remove eeg or other unused channels (if it has)
        BIDS_loc = os.path.join(LAB_root, "BIDS-1.0_LexicalDecRepDelay", "BIDS", f"sub-{subject}", "ieeg")
        unused_chs = get_unused_chs(BIDS_loc)
        if unused_chs:
            raw.drop_channels(unused_chs)

        # line noise filtering
        line_filter(raw, mt_bandwidth=10., n_jobs=-3, copy=False, verbose=10,
                    filter_length='700ms', freqs=[60], notch_widths=20)
        line_filter(raw, mt_bandwidth=10., n_jobs=-3, copy=False, verbose=10,
                    filter_length='20s', freqs=[60, 120, 180, 240],
                    notch_widths=20)

        # crop and save data
        raw1 = crop_empty_data(raw)
        del raw
        raw = raw1
        bids_root = 'C:\\Users\\bl314\\Box\\CoganLab\\BIDS-1.0_LexicalDecRepDelay\\BIDS'
        if not os.path.exists(os.path.join(bids_root, "derivatives")):
            os.mkdir(os.path.join(bids_root, "derivatives"))
            os.mkdir(os.path.join(bids_root, "derivatives", "a"))
        elif not os.path.exists(os.path.join(bids_root, "derivatives", "a")):
            os.mkdir(os.path.join(bids_root, "derivatives", "a"))
        save_derivative(raw, layout, "a", True)
        del raw

        # remove "bad boundary" in events.tsv
        tsv_loc = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', 'derivatives', 'a', f'sub-{subject}',
                               'ieeg')
        update_tsv(subject, tsv_loc)
        log_file.write(f"{datetime.datetime.now()}, {subject}, Line noise filter completed\n")

    except Exception as e:
        log_file.write(f"{datetime.datetime.now()}, {subject}, Line noise filter failed with error: {str(e)}\n")

    try:
        log_file.write(f"{datetime.datetime.now()}, {subject}, Executing outlier chs removal\n")
        ## Mark outlier channels
        raw = raw_from_layout(layout.derivatives['derivatives/a'], subject=subject, desc='a', extension='.edf',
                              preload=True)
        raw.info['bads'] = channel_outlier_marker(raw, 3, 2)
        update(raw, layout, "outlier")
        del raw
        log_file.write(f"{datetime.datetime.now()}, {subject}, Outlier chs removal completed\n")

    except Exception as e:
        log_file.write(f"{datetime.datetime.now()}, {subject}, Outlier chs removal failed with error: {str(e)}\n")

    ## Wavelet

    try:
        log_file.write(f"{datetime.datetime.now()}, {subject}, Executing wavelet\n")

        # load data
        raw = raw_from_layout(layout.derivatives['derivatives/a'], subject=subject, desc='a', extension='.edf',
                              preload=True)

        # ref to average
        ch_type = raw.get_channel_types(only_data_chs=True)[0]
        raw.set_eeg_reference(ref_channels="average", ch_type=ch_type)

        # make direction
        if not os.path.exists(os.path.join(save_dir, subject)):
            os.mkdir(os.path.join(save_dir, subject))
        if not os.path.exists(os.path.join(save_dir, subject, 'wavelet')):
            os.mkdir(os.path.join(save_dir, subject, 'wavelet'))

        # wavelet
        for epoch, t, tag in zip(
                ('Cue/CORRECT', 'Auditory_stim/CORRECT','Delay/CORRECT', 'Go/CORRECT','Resp/CORRECT'),
                ((-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1)),
                ('Cue', 'Auditory','Delay','Go','Resp')):

            # Get the spectras
            t1 = t[0] - 0.5
            t2 = t[1] + 0.5
            times = (t1, t2)
            trials = trial_ieeg(raw, epoch, times, preload=True)
            outliers_to_nan(trials, outliers=10)

            spectra_wavelet = wavelet_scaleogram(trials, n_jobs=-3, decim=int(
                raw.info['sfreq'] / 200))  # 1/10 of the timepionts, don't take too long
            crop_pad(spectra_wavelet, "0.5s")  # cut the first and final 0.5s, change to zero

            # Get the baseline
            if epoch == 'Cue/CORRECT':
                base_wavelet = spectra_wavelet.copy().crop(-0.5, 0)
                base_wavelet = base_wavelet.average(lambda x: np.nanmean(x, axis=0), copy=True)

            # Baseline correction
            spectra_wavelet = spectra_wavelet.average(lambda x: np.nanmean(x, axis=0), copy=True)
            spectra_wavelet = rescale(spectra_wavelet, base_wavelet, copy=True, mode='ratio')
            spectra_wavelet._data = np.log10(spectra_wavelet._data) * 20

            # Save spectras
            filename = os.path.join(save_dir, subject, 'wavelet', f'{tag}-tfr.h5')
            mne.time_frequency.write_tfrs(filename, spectra_wavelet, overwrite=True)

            # Make spectrograms
            chan_grids = chan_grid(spectra_wavelet, size=(20, 10), vlim=(-2, 2), cmap=parula_map)

            # Save spectrograms
            fig_count = 0
            for fig in chan_grids:
                figdir = os.path.join(save_dir, subject, 'wavelet', f'{tag}_{fig_count + 1}.jpg')
                chan_grids[fig_count].savefig(figdir, dpi=300)
                plt.close(fig)
                fig_count += 1

            # Clean memory
            del spectra_wavelet, filename

        del base_wavelet
        log_file.write(f"{datetime.datetime.now()}, {subject}, Wavelet completed\n")

    except Exception as e:
        log_file.write(f"{datetime.datetime.now()}, {subject}, Wavelet failed with error: {str(e)}\n")

log_file.close()