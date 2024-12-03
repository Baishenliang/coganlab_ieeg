# Initialize
import os
import numpy as np
import mne
from ieeg.io import get_data, raw_from_layout, update
from ieeg.navigate import trial_ieeg, outliers_to_nan, channel_outlier_marker, crop_empty_data
from ieeg.calc.scaling import rescale
from ieeg.viz.ensemble import chan_grid
from ieeg.timefreq.utils import crop_pad, wavelet_scaleogram
from ieeg.viz.parula import parula_map
import matplotlib.pyplot as plt
save_dir='C:\\Users\\bl314\\Box\\CoganLab\\IndividualMeetings\\Baishen\\ieeg_results\\lexical_delay'
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
layout = get_data("LexicalDecRepDelay", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

# Subjects to be processed
# Subjs_proc=['D0053','D0054','D0055','D0057','D0059','D0063','D0065','D0066','D0068','D0069','D0070','D0071','D0077','D0081','D0094','D0096','D0101','D0102','D0103','D0107']
Subjs_proc = ['D0057', 'D0059', 'D0063', 'D0065', 'D0066', 'D0068', 'D0069', 'D0070',
              'D0071', 'D0077', 'D0081', 'D0094', 'D0096', 'D0101', 'D0102', 'D0103', 'D0107']

for subject in Subjs_proc:
    try:

        # load subject
        print(subject)
        raw = raw_from_layout(layout.derivatives['derivatives/a'], subject=subject, desc='a', extension='.edf',
                              preload=True)

        # Re-reg to AVG
        ch_type = raw.get_channel_types(only_data_chs=True)[0]
        raw.set_eeg_reference(ref_channels="average", ch_type=ch_type)

        # check dir
        if not os.path.exists(os.path.join(save_dir, subject)):
            os.mkdir(os.path.join(save_dir, subject))
        if not os.path.exists(os.path.join(save_dir, subject, 'multitaper_4cons')):
            os.mkdir(os.path.join(save_dir, subject, 'multitaper_4cons'))

        # run multitaper
        for task, task_Tag in zip(('Repeat', 'Yes_No'), ('Rep', 'YN')):
            for word, word_Tag in zip(('Word', 'Nonword'), ('wrd', 'nwrd')):
                for epoch, t, tag in zip(
                        ('Auditory_stim/' + task + '/' + word + '/CORRECT', 'Delay/' + task + '/' + word + '/CORRECT', 'Go/' + task + '/' + word + '/CORRECT',
                         'Resp/' + task + '/' + word + '/CORRECT'),
                        ((-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1)),
                        ('Auditory-' + task_Tag + '-' + word_Tag, 'Delay-' + task_Tag + '-' + word_Tag, 'Go-' + task_Tag + '-' + word_Tag, 'Resp-' + task_Tag + '-' + word_Tag)
                ):

                    # Get the spectras
                    t1 = t[0] - 0.5
                    t2 = t[1] + 0.5
                    times = (t1, t2)
                    trials = trial_ieeg(raw, epoch, times, preload=True)
                    outliers_to_nan(trials, outliers=10)

                    freq = np.linspace(0.5, 200, num=80)
                    kwargs = dict(average=False, n_jobs=-3, freqs=freq, return_itc=False,
                                  n_cycles=freq / 2, time_bandwidth=4,
                                  # n_fft=int(trials.info['sfreq'] * 2.75),
                                  decim=20, )
                    # adaptive=True)
                    spectra_multitaper = trials.compute_tfr(method="multitaper", **kwargs)
                    crop_pad(spectra_multitaper, "0.5s")  # cut the first and final 0.5s, change to zero

                    # Get the baseline
                    if epoch == 'Auditory_stim/' + task + '/' + word + '/CORRECT':
                        base_multitaper = spectra_multitaper.copy().crop(-0.5, 0)
                        base_multitaper = base_multitaper.average(lambda x: np.nanmean(x, axis=0), copy=True)

                    # Baseline correction
                    spectra_multitaper = spectra_multitaper.average(lambda x: np.nanmean(x, axis=0), copy=True)
                    spectra_multitaper = rescale(spectra_multitaper, base_multitaper, copy=True, mode='ratio')

                    # Save spectras
                    filename = os.path.join(save_dir, subject, 'multitaper_4cons', f'{tag}-tfr.h5')
                    mne.time_frequency.write_tfrs(filename, spectra_multitaper, overwrite=True)

                    # Make spectrograms
                    chan_grids = chan_grid(spectra_multitaper, size=(20, 10), vlim=(0.7, 1.4), cmap=parula_map)

                    # Save spectrograms
                    fig_count = 0
                    for fig in chan_grids:
                        figdir = os.path.join(save_dir, subject, 'multitaper_4cons', f'{tag}_{fig_count + 1}.jpg')
                        chan_grids[fig_count].savefig(figdir, dpi=300)
                        plt.close(fig)
                        fig_count += 1

                    del trials, spectra_multitaper, filename
                del base_multitaper

    except Exception as e:
        print(e)