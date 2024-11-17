import os
import numpy as np
import mne
from ieeg.io import get_data, raw_from_layout, save_derivative
from ieeg.navigate import trial_ieeg, outliers_to_nan, channel_outlier_marker, crop_empty_data
from ieeg.calc.scaling import rescale
from ieeg.viz.ensemble import chan_grid
from ieeg.timefreq.utils import crop_pad, wavelet_scaleogram
from ieeg.viz.parula import parula_map
from ieeg.navigate import channel_outlier_marker
save_dir='C:\\Users\\bl314\\Box\\CoganLab\\IndividualMeetings\\Baishen\\ieeg_results\\lexical_delay'

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("LexicalDecRepDelay", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

#process=1 # remove bad channels
process=2 # wavelet

for subject in ['D0102']:#,'D0103','D0107']:#[5,6,7,8,9]: #[1,2,5,6,7,8,9]

    if process==1:
        # subject = subjects[subj]

        if subject=='D0054':
            subject_Tag = 'D54'
        elif subject=='D0055':
            subject_Tag = 'D55'
        elif subject=='D0094':
            subject_Tag = 'D94'
        elif subject=='D0096':
            subject_Tag = 'D96'
        elif subject=='D0101':
            subject_Tag = 'D101'
        elif subject=='D0102':
            subject_Tag = 'D102'
        elif subject=='D0103':
            subject_Tag = 'D103'
        elif subject=='D0107':
            subject_Tag = 'D107B'
        else:
            print("Subject not found, please check.")

        raw = raw_from_layout(layout.derivatives['derivatives/clean'], subject=subject, desc='clean',extension='.edf',preload=False)

        # Remove EEG channels for D101
        found = 1
        if subject == 'D0053':
            eeg_channels_to_exclude = []
        elif subject == 'D0054':
            eeg_channels_to_exclude = []
        elif subject == 'D0055':
            eeg_channels_to_exclude = []
        elif subject == 'D0070':
            eeg_channels_to_exclude = []
        elif subject == 'D0094':
            eeg_channels_to_exclude = []
        elif subject == 'D0101':
            eeg_channels_to_exclude = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                                       'O1', 'O2', 'Fz', 'Cz', 'Pz']
        elif subject == 'D0102':
            eeg_channels_to_exclude = ['T5', 'T6', 'FZ', 'CZ', 'PZ', 'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                                       '02', 'F7', 'F8', 'T3', 'T4']
        elif subject == 'D0103':
            eeg_channels_to_exclude = ['FZ', 'CZ', 'PZ', 'F7', 'F8', 'T5', 'T6', 'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3',
                                       'P4', 'O1', 'O2', 'T3', 'T4']
        elif subject == 'D0107':
            eeg_channels_to_exclude = []
        else:
            print("Subject not found, please check.")
            found = 0

        if found and eeg_channels_to_exclude:
            raw.drop_channels(eeg_channels_to_exclude)

        # mark channel outliers as bad
        raw.info['bads'] = channel_outlier_marker(raw, 3, 2)
        # Exclude bad channels
        raw.drop_channels(raw.info['bads'])

        # Check if derivatives folder exists and create if not
        bids_root='C:\\Users\\bl314\\Box\\CoganLab\\BIDS-1.0_LexicalDecRepDelay\\BIDS'
        if not os.path.exists(os.path.join(bids_root, "derivatives")):
            os.mkdir(os.path.join(bids_root, "derivatives"))
            os.mkdir(os.path.join(bids_root, "derivatives", "a"))
        elif not os.path.exists(os.path.join(bids_root, "derivatives", "a")):
            os.mkdir(os.path.join(bids_root, "derivatives", "a"))
        save_derivative(raw, layout, "a", True)

        del raw

    elif process==2:

        raw = raw_from_layout(layout.derivatives['derivatives/a'], subject=subject, desc='a',extension='.edf',preload=True)

        ch_type = raw.get_channel_types(only_data_chs=True)[0]
        raw.set_eeg_reference(ref_channels="average", ch_type=ch_type)

        if not os.path.exists(os.path.join(save_dir, subject)):
            os.mkdir(os.path.join(save_dir, subject))
        if not os.path.exists(os.path.join(save_dir, subject,'wavelet')):
            os.mkdir(os.path.join(save_dir, subject,'wavelet'))

        # Wavelet is good to detect and remove muscle artifact channels
        # Also plot the subject brain
        for task, task_Tag in zip(('Repeat', 'Yes_No'), ('Rep', 'YN')):
            for epoch, t, tag in zip(
                    ('Auditory_stim/' + task + '/CORRECT', 'Delay/' + task + '/CORRECT', 'Resp/' + task + '/CORRECT'),
                    ((-0.5, 1.5), (-0.5, 1.5), (-0.5, 1)),
                    ('Auditory-' + task_Tag, 'Delay-' + task_Tag, 'Resp-' + task_Tag)
            ):

                # Get the spectras
                t1 = t[0] - 0.5
                t2 = t[1] + 0.5
                times = (t1, t2)
                trials = trial_ieeg(raw, epoch, times, preload=True)
                outliers_to_nan(trials, outliers=10)

                ##############################
                ####### Wavelet ##############
                ##############################

                spectra_wavelet = wavelet_scaleogram(trials, n_jobs=-3, decim=int(
                    raw.info['sfreq'] / 200))  # 1/10 of the timepionts, don't take too long
                crop_pad(spectra_wavelet, "0.5s")  # cut the first and final 0.5s, change to zero

                # Get the baseline
                if epoch == 'Auditory_stim/' + task + '/CORRECT':
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
                    fig_count += 1

                # Clean memory
                del spectra_wavelet, filename

            del base_wavelet

        del raw
