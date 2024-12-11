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
from bsliang_utils import get_unused_chs, update_tsv, detect_outlier, load_muscle_chs
from matplotlib import pyplot as plt

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
save_dir=os.path.join(HOME, "Box", "CoganLab", "D_Data","LexicalDecRepDelay","Baishen_Figs")

# Log
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file_path = os.path.join('data', 'logs', f'batch_preproc_{current_time}.txt')

# Subj list
subject_processing_dict = {
    "D0053": "multitaper",
    "D0054": "multitaper",
    "D0055": "multitaper",
    "D0057": "multitaper",
    "D0059": "multitaper",
    "D0063": "multitaper",
    "D0065": "multitaper",
    "D0066": "multitaper",
    "D0068": "multitaper",
    "D0069": "multitaper",
    "D0070": "multitaper",
    "D0071": "",
    "D0077": "whole",
    "D0079": "whole",
    "D0081": "whole",
    "D0094": "whole",
    "D0096": "whole",
    "D0101": "",
    "D0102": "whole",
    "D0103": "whole",
    "D0107": "whole",
}

for subject, processing_type in subject_processing_dict.items():

    if processing_type == "whole" or "linernoise" in processing_type:
        ## Line Noise Filtering

        print('=========================\n')
        print(f'Line Noise Filtering {subject}\n')
        print('=========================\n')

        log_file = open(log_file_path, 'a')
        try:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Executing line noise filter\n")

            # load BIDS raw data
            layout = get_data("LexicalDecRepDelay", root=LAB_root)
            raw = raw_from_layout(layout, subject=subject, preload=True, extension='.edf')

            # get and remove eeg or other unused channels (if it has)
            BIDS_loc = os.path.join(LAB_root, "BIDS-1.0_LexicalDecRepDelay", "BIDS", f"sub-{subject}", "ieeg")
            unused_chs = get_unused_chs(BIDS_loc)
            if unused_chs:
                raw.drop_channels(unused_chs)

            # line noise filtering
            line_filter(raw, mt_bandwidth=10., n_jobs=-10, copy=False, verbose=10,
                        filter_length='700ms', freqs=[60], notch_widths=20)
            line_filter(raw, mt_bandwidth=10., n_jobs=-10, copy=False, verbose=10,
                        filter_length='20s', freqs=[60, 120, 180, 240],
                        notch_widths=20)

            # crop and save data
            if subject=="D0079":
                raw1 = crop_empty_data(raw)
                del raw
                raw = raw1
            bids_root = os.path.join(LAB_root,'BIDS-1.0_LexicalDecRepDelay','BIDS')
            if not os.path.exists(os.path.join(bids_root, "derivatives")):
                os.mkdir(os.path.join(bids_root, "derivatives"))
                os.mkdir(os.path.join(bids_root, "derivatives", "a"))
            elif not os.path.exists(os.path.join(bids_root, "derivatives", "a")):
                os.mkdir(os.path.join(bids_root, "derivatives", "a"))
            save_derivative(raw, layout, "a", True)
            del raw
            del layout

            # remove "bad boundary" in events.tsv
            tsv_loc = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', 'derivatives', 'a', f'sub-{subject}',
                                   'ieeg')
            update_tsv(subject, tsv_loc)
            log_file.write(f"{datetime.datetime.now()}, {subject}, Line noise filter %%% completed %%% \n")

        except Exception as e:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Line noise filter !!! failed with error !!! : {str(e)}\n")
        log_file.close()

    if processing_type == "whole" or "outlierchs" in processing_type:

        print('=========================\n')
        print(f'Outlier chs removal {subject}\n')
        print('=========================\n')

        log_file = open(log_file_path, 'a')
        try:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Executing outlier chs removal\n")
            ## Mark outlier channels
            layout = get_data("LexicalDecRepDelay", root=LAB_root)
            raw = raw_from_layout(layout.derivatives['derivatives/a'], subject=subject, desc='a', extension='.edf',
                                  preload=True)
            derivative_loc = os.path.join(LAB_root, "BIDS-1.0_LexicalDecRepDelay","BIDS","derivatives","a",f"sub-{subject}","ieeg")
            is_outlier = detect_outlier(subject,derivative_loc)
            if is_outlier == 1:
                raise ValueError(
                    f"Outlier channels for the {subject} have been removed. Skip outlier channels removal now. If you want to re-do it, mark all the channels in the derivative as good and n/a for status first."
                )
            else:
                raw.info['bads'] = channel_outlier_marker(raw, 3, 2)
                update(raw, layout, "outlier")
                log_file.write(f"{datetime.datetime.now()}, {subject}, Outlier chs removal %%% completed %%% \n")
            del raw
            del layout

        except Exception as e:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Outlier chs removal !!! failed with error !!! : {str(e)}\n")
        log_file.close()

    if processing_type == "whole" or "wavelet" in processing_type:

        print('=========================\n')
        print(f'Wavelet {subject}\n')
        print('=========================\n')

        ## Wavelet
        log_file = open(log_file_path, 'a')
        try:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Executing wavelet\n")

            # load data
            layout = get_data("LexicalDecRepDelay", root=LAB_root)
            raw1 = raw_from_layout(layout.derivatives['derivatives/a'], subject=subject, desc='a', extension='.edf',
                                  preload=False)

            # drop bad channels
            raw = raw1.copy().drop_channels(raw1.info['bads'])
            del raw1
            raw.load_data()

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
                del spectra_wavelet, filename, chan_grids

            del base_wavelet, raw, layout
            log_file.write(f"{datetime.datetime.now()}, {subject}, Wavelet  %%% completed %%% \n")

        except Exception as e:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Wavelet  !!! failed with error !!! : {str(e)}\n")

        log_file.close()

    if processing_type == "whole" or "multitaper" in processing_type:

        print('=========================\n')
        print(f'Multitaper {subject}\n')
        print('=========================\n')

        log_file = open(log_file_path, 'a')
        try:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Executing multitaper\n")    
            ## Multitaper
            # load data
            layout = get_data("LexicalDecRepDelay", root=LAB_root)
            raw1 = raw_from_layout(layout.derivatives['derivatives/a'], subject=subject, desc='a', extension='.edf',
                                    preload=False)

            # read muscle artifact channels and update
            raw1_org_bads=raw1.info['bads']
            muscle_chs=load_muscle_chs(subject)
            muscle_chs_bads=[b for b in muscle_chs if b in raw1.ch_names]
            bads_new=list(set(raw1_org_bads+muscle_chs_bads))
            raw1.info.update(bads=bads_new)
            update(raw1, layout, "muscle")

            # drop bad channels
            raw = raw1.copy().drop_channels(raw1.info['bads'])
            del raw1
            raw.load_data()

            # ref to average
            ch_type = raw.get_channel_types(only_data_chs=True)[0]
            raw.set_eeg_reference(ref_channels="average", ch_type=ch_type)

            # make direction
            if not os.path.exists(os.path.join(save_dir, subject)):
                os.mkdir(os.path.join(save_dir, subject))
            if not os.path.exists(os.path.join(save_dir, subject, 'multitaper_4cons')):
                os.mkdir(os.path.join(save_dir, subject, 'multitaper_4cons'))

            # run multitaper
            for task, task_Tag in zip(('Repeat', 'Yes_No'), ('Rep', 'YN')):
                for word, word_Tag in zip(('Word', 'Nonword'), ('wrd', 'nwrd')):
                    for epoch, t, tag in zip(
                            ('Cue/' + task + '/' + word + '/CORRECT','Auditory_stim/' + task + '/' + word + '/CORRECT', 'Delay/' + task + '/' + word + '/CORRECT', 'Go/' + task + '/' + word + '/CORRECT',
                            'Resp/' + task + '/' + word + '/CORRECT'),
                            ((-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1)),
                            ('Cue-' + task_Tag + '-' + word_Tag, 'Auditory-' + task_Tag + '-' + word_Tag, 'Delay-' + task_Tag + '-' + word_Tag, 'Go-' + task_Tag + '-' + word_Tag, 'Resp-' + task_Tag + '-' + word_Tag)
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
                        if epoch == 'Cue/' + task + '/' + word + '/CORRECT':
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

                        del trials, spectra_multitaper, filename, chan_grids
                    del base_multitaper

            log_file.write(f"{datetime.datetime.now()}, {subject}, Multitaper  %%% completed %%% \n")

        except Exception as e:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Multitaper !!! failed with error !!! : {str(e)}\n")

        log_file.close()

    if processing_type == "whole" or "Gamma" in processing_type:

        print('=========================\n')
        print(f'Gamma band-pass filter {subject}\n')
        print('=========================\n')
    
        ## Wavelet
        log_file = open(log_file_path, 'a')
        try:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Executing Gamma band-pass filter\n")

            log_file.write(f"{datetime.datetime.now()}, {subject}, Gamma band-pass  %%% completed %%% \n")

        except Exception as e:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Gamma band-pass!!! failed with error !!! : {str(e)}\n")

        log_file.close()