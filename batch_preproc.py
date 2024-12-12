"""
    Batch preprocessing scripts for lexical delay task.
    Including: line noise filtering, outlier channels removal, and wavelet.
    Parameters are equal to Sentence Rep processing code unless a change is necessary (NEEDED TO BE RE-CHECKED!):
    https://github.com/coganlab/SentenceRep_analysis
    From Baishen Liang.
"""
# Preparation:

import os
import os.path as op
import mne
import datetime
import numpy as np
from itertools import product
from ieeg.navigate import crop_empty_data, channel_outlier_marker, trial_ieeg, outliers_to_nan
from ieeg.mt_filter import line_filter
from ieeg.io import get_data, raw_from_layout, save_derivative, update
from ieeg.calc import stats, scaling
from ieeg.calc.scaling import rescale
from ieeg.viz.ensemble import chan_grid
from ieeg.timefreq.utils import crop_pad, wavelet_scaleogram
from ieeg.timefreq import gamma, utils
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
    "D0081": "",
    "D0063": "gamma",
    "D0066": "linernoise/outlierchs/multitaper/gamma",
    "D0103": "linernoise/outlierchs/wavelet/multitaper/gamma",
    "D0094": "linernoise/outlierchs/wavelet/multitaper/gamma",
    "D0096": "linernoise/outlierchs/wavelet/multitaper/gamma",
    "D0053": "/gamma",
    "D0054": "/gamma",
    "D0055": "multitaper/gamma",
    "D0057": "/gamma",
    "D0059": "/gamma",
    "D0065": "multitaper/gamma",
    "D0068": "/gamma",
    "D0069": "/gamma",
    "D0070": "/gamma",
    "D0071": "linernoise/outlierchs/wavelet/multitaper/gamma",
    "D0077": "/gamma",
    "D0079": "linernoise/outlierchs/wavelet/multitaper/gamma",
    "D0101": "linernoise/outlierchs/wavelet/multitaper/gamma",
    "D0102": "linernoise/outlierchs/wavelet/multitaper/gamma",
    "D0107": "linernoise/outlierchs/wavelet/multitaper/gamma",
}

for subject, processing_type in subject_processing_dict.items():

    if "linernoise" in processing_type:
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

    if "outlierchs" in processing_type:

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

    if "wavelet" in processing_type:

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

    if "multitaper" in processing_type:

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

    if "gamma" in processing_type:

        print('=========================\n')
        print(f'Gamma band-pass filter and permutation {subject}\n')
        print('=========================\n')
    
        ## Wavelet
        log_file = open(log_file_path, 'a')
        try:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Executing Gamma band-pass filter and permutation \n")

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
            if not os.path.exists(os.path.join(save_dir, subject, 'stats')):
                os.mkdir(os.path.join(save_dir, subject, 'stats'))
            
            subj_gamma_dir=os.path.join(save_dir, subject, 'stats')

            # extract gamma
            out = []
            for epoch, t, tag in zip(
                    ('Cue/CORRECT', 'Cue/CORRECT', 'Auditory_stim/CORRECT','Delay/CORRECT', 'Go/CORRECT','Resp/CORRECT'),
                    ((-0.5, 0), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1)),
                    ('Baseline', 'Cue', 'Auditory','Delay','Go','Resp')):

                # Get the spectras
                t1 = t[0] - 0.5
                t2 = t[1] + 0.5
                times = (t1, t2)
                trials = trial_ieeg(raw, epoch, times, preload=True, reject_by_annotation=False)
                outliers_to_nan(trials, outliers=10)

                gamma.extract(trials, copy=False, n_jobs=-10)
                utils.crop_pad(trials, "0.5s")
                trials.resample(100)
                trials.filenames = raw.filenames
                out.append(trials)

            # Get and cut baseline
            base = out.pop(0)

            # Prepare for permutation
            mask = dict()
            data = []
            nperm = 100000
            sig2 = base.get_data(copy=True)

            # run permutation
            for task, task_Tag in zip(('Repeat', 'Yes_No'), ('Rep', 'YN')):
                for word, word_Tag in zip(('Word', 'Nonword'), ('wrd', 'nwrd')):
                    for epoch, t, tag in zip(
                            (out[0]['Cue/' + task + '/' + word + '/CORRECT'], out[1]['Auditory_stim/' + task + '/' + word + '/CORRECT'], 
                            out[2]['Delay/' + task + '/' + word + '/CORRECT'], out[3]['Go/' + task + '/' + word + '/CORRECT'],
                            out[4]['Resp/' + task + '/' + word + '/CORRECT']),
                            # (out[0][e] for e in ['Cue/' + task + '/' + word + '/CORRECT','Auditory_stim/' + task + '/' + word + '/CORRECT', 'Delay/' + task + '/' + word + '/CORRECT', 'Go/' + task + '/' + word + '/CORRECT',
                            # 'Resp/' + task + '/' + word + '/CORRECT']),
                            ((-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5), (-0.5, 1)),
                            ('Cue-' + task_Tag + '-' + word_Tag, 'Auditory-' + task_Tag + '-' + word_Tag, 'Delay-' + task_Tag + '-' + word_Tag, 'Go-' + task_Tag + '-' + word_Tag, 'Resp-' + task_Tag + '-' + word_Tag)
                    ):

                        sig1 = epoch.get_data(tmin=t[0], tmax=t[1], copy=True)

                        # time-perm
                        mask[tag], p_act = stats.time_perm_cluster(
                            sig1, sig2, p_thresh=0.05, axis=0, n_perm=nperm, n_jobs=-3,
                            ignore_adjacency=1)
                        epoch_mask = mne.EvokedArray(mask[tag], epoch.average().info,
                                                    tmin=t[0])

                        # plot mask
                        fig, ax = plt.subplots()
                        ax.imshow(mask[tag], cmap='Reds')
                        channel_names=epoch_mask.ch_names[::5]
                        ax.set_yticks(range(0,len(channel_names)*5,5))
                        ax.set_yticklabels(channel_names)
                        time_stamps=epoch_mask.times[::20]
                        ax.set_xticks(range(0,len(time_stamps)*20,20))
                        ax.set_xticklabels(time_stamps)
                        try:
                            zero_time_index = np.where(epoch_mask.times == 0)[0][0]
                            ax.axvline(x=zero_time_index, color='black', linestyle='--', linewidth=1)
                        except Exception as e:
                            print('no zero time found')
                        fig.savefig(os.path.join(subj_gamma_dir,f'{tag}.jpg'), dpi=300)
                        plt.close(fig)

                        # baseline correction
                        power = scaling.rescale(epoch, base, 'mean', copy=True)
                        z_score = scaling.rescale(epoch, base, 'zscore', copy=True)

                        # Calculate the p-value
                        p_vals = mne.EvokedArray(p_act, epoch_mask.info, tmin=t[0])

                        # p_vals = epoch_mask.copy()
                        data.append((tag, epoch_mask.copy(), power.copy(), z_score.copy(), p_vals.copy()))

            for tag, epoch_mask, power, z_score, p_vals in data:

                power.save(subj_gamma_dir + f"/{subject}_{tag}_power-epo.fif", overwrite=True,fmt='double')
                z_score.save(subj_gamma_dir + f"/{subject}_{tag}_zscore-epo.fif", overwrite=True,fmt='double')
                epoch_mask.save(subj_gamma_dir + f"/{subject}_{tag}_mask-ave.fif", overwrite=True)
                p_vals.save(subj_gamma_dir + f"/{subject}_{tag}_pval-ave.fif", overwrite=True)
            
            base.save(subj_gamma_dir + f"/{subject}_base-epo.fif", overwrite=True)
            del data

            log_file.write(f"{datetime.datetime.now()}, {subject}, Gamma band-pass and permutation  %%% completed %%% \n")

        except Exception as e:
            log_file.write(f"{datetime.datetime.now()}, {subject}, Gamma band-pass and permutation !!! failed with error !!! : {str(e)}\n")

        log_file.close()