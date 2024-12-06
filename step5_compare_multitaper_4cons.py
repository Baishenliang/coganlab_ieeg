# Import packages
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ieeg.viz.parula import parula_map
save_dir='C:\\Users\\bl314\\Box\\CoganLab\\IndividualMeetings\\Baishen\\ieeg_results\\lexical_delay'

Subjs_proc = ['D0053','D0054','D0055','D0057', 'D0059', 'D0063', 'D0065', 'D0066', 'D0068', 'D0069', 'D0070',
              'D0071', 'D0077', 'D0079','D0081', 'D0094', 'D0096', 'D0101', 'D0102', 'D0103', 'D0107']
plot_dir_in_Tag='multitaper_4cons'
plot_dir_out_Tag='multitaper_4cons_compare'

for subject in Subjs_proc:
    try:
        # Load files
        ## file names
        epoch_stim_rep_word = "Auditory-Rep-wrd"
        epoch_delay_rep_word = "Delay-Rep-wrd"
        epoch_resp_rep_word = "Resp-Rep-wrd"

        epoch_stim_rep_nonword = "Auditory-Rep-nwrd"
        epoch_delay_rep_nonword = "Delay-Rep-nwrd"
        epoch_resp_rep_nonword = "Resp-Rep-nwrd"

        epoch_stim_yn_word = "Auditory-YN-wrd"
        epoch_delay_yn_word = "Delay-YN-wrd"
        epoch_resp_yn_word = "Resp-YN-wrd"

        epoch_stim_yn_nonword = "Auditory-YN-nwrd"
        epoch_delay_yn_nonword = "Delay-YN-nwrd"
        epoch_resp_yn_nonword = "Resp-YN-nwrd"

        ## filepath
        fname_stim_rep_word = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_stim_rep_word}-tfr.h5')
        fname_delay_rep_word = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_delay_rep_word}-tfr.h5')
        fname_resp_rep_word = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_resp_rep_word}-tfr.h5')

        fname_stim_rep_nonword = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_stim_rep_nonword}-tfr.h5')
        fname_delay_rep_nonword = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_delay_rep_nonword}-tfr.h5')
        fname_resp_rep_nonword = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_resp_rep_nonword}-tfr.h5')

        fname_stim_yn_word = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_stim_yn_word}-tfr.h5')
        fname_delay_yn_word = os.path.join(save_dir, subject,plot_dir_in_Tag, f'{epoch_delay_yn_word}-tfr.h5')
        fname_resp_yn_word = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_resp_yn_word}-tfr.h5')

        fname_stim_yn_nonword = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_stim_yn_nonword}-tfr.h5')
        fname_delay_yn_nonword = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_delay_yn_nonword}-tfr.h5')
        fname_resp_yn_nonword = os.path.join(save_dir, subject, plot_dir_in_Tag, f'{epoch_resp_yn_nonword}-tfr.h5')

        ## load files
        spectra_stim_rep_word = mne.time_frequency.read_tfrs(fname_stim_rep_word)
        spectra_delay_rep_word = mne.time_frequency.read_tfrs(fname_delay_rep_word)
        spectra_resp_rep_word = mne.time_frequency.read_tfrs(fname_resp_rep_word)

        spectra_stim_rep_nonword = mne.time_frequency.read_tfrs(fname_stim_rep_nonword)
        spectra_delay_rep_nonword = mne.time_frequency.read_tfrs(fname_delay_rep_nonword)
        spectra_resp_rep_nonword = mne.time_frequency.read_tfrs(fname_resp_rep_nonword)

        spectra_stim_yn_word = mne.time_frequency.read_tfrs(fname_stim_yn_word)
        spectra_delay_yn_word = mne.time_frequency.read_tfrs(fname_delay_yn_word)
        spectra_resp_yn_word = mne.time_frequency.read_tfrs(fname_resp_yn_word)

        spectra_stim_yn_nonword = mne.time_frequency.read_tfrs(fname_stim_yn_nonword)
        spectra_delay_yn_nonword = mne.time_frequency.read_tfrs(fname_delay_yn_nonword)
        spectra_resp_yn_nonword = mne.time_frequency.read_tfrs(fname_resp_yn_nonword)

        spectra_stim_rep_word_m_nonword=spectra_stim_rep_word.__sub__(spectra_stim_rep_nonword)
        spectra_delay_rep_word_m_nonword=spectra_delay_rep_word.__sub__(spectra_delay_rep_nonword)
        spectra_resp_rep_word_m_nonword=spectra_resp_rep_word.__sub__(spectra_resp_rep_nonword)

        spectra_stim_yn_word_m_nonword=spectra_stim_yn_word.__sub__(spectra_stim_yn_nonword)
        spectra_delay_yn_word_m_nonword=spectra_delay_yn_word.__sub__(spectra_delay_yn_nonword)
        spectra_resp_yn_word_m_nonword=spectra_resp_yn_word.__sub__(spectra_resp_yn_nonword)

        spectra_stim_rep_m_yn_word=spectra_stim_rep_word.__sub__(spectra_stim_yn_word)
        spectra_delay_rep_m_yn_word=spectra_delay_rep_word.__sub__(spectra_delay_yn_word)
        spectra_resp_rep_m_yn_word=spectra_resp_rep_word.__sub__(spectra_resp_yn_word)

        spectra_stim_rep_m_yn_nonword=spectra_stim_rep_nonword.__sub__(spectra_stim_yn_nonword)
        spectra_delay_rep_m_yn_nonword=spectra_delay_rep_nonword.__sub__(spectra_delay_yn_nonword)
        spectra_resp_rep_m_yn_nonword=spectra_resp_rep_nonword.__sub__(spectra_resp_yn_nonword)

        # plot

        base_fontsize = 8
        fig_width, fig_height = (13.33, 7.5)  # 16:9 ppt scale
        dpi = 300
        scale_factor = min(fig_width / 10, fig_height / 5)
        font_size = base_fontsize * scale_factor
        vlim_spec = (0.7, 1.4)
        vlim_diff = (-1, 1)

        for channel_name in spectra_stim_rep_word.ch_names:

            print(channel_name)

            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

            plt.rcParams.update({
                'font.size': font_size,
                'axes.titlesize': font_size * 1.2,
                'axes.labelsize': font_size,
                'xtick.labelsize': font_size * 0.8,
                'ytick.labelsize': font_size * 0.8,
                'legend.fontsize': font_size * 0.8,
                'axes.linewidth': 0.3,
                'axes.edgecolor': 'black',
                'xtick.major.size': 1,
                'ytick.major.size': 1,
                'xtick.major.width': 0.3,
                'ytick.major.width': 0.3
            })

            gs0 = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)

            # Cluster1: Auditory stimuli
            gs00 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=4, subplot_spec=gs0[0],
                                                    width_ratios=[10 / 31, 10 / 31, 10 / 31, 1 / 32])

            # Cluster2: Delay
            gs01 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=4, subplot_spec=gs0[1],
                                                    width_ratios=[10 / 31, 10 / 31, 10 / 31, 1 / 32])

            # Cluster3: Resp
            gs02 = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=4, subplot_spec=gs0[2],
                                                    width_ratios=[10 / 31, 10 / 31, 10 / 31, 1 / 32])

            ############################################################

            ## Plot 1-1: Auditory-Rep-Word
            ax_aud_rep_word = fig.add_subplot(gs00[0, 0])
            spectra_stim_rep_word.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                       axes=ax_aud_rep_word, cmap=parula_map, title=channel_name)
            ax_aud_rep_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_aud_rep_word.set_title('\nStim')
            ax_aud_rep_word.set_xlabel("")
            ax_aud_rep_word.set_ylabel("Repeat\nFrequency(Hz)")
            ax_aud_rep_word.set_xticklabels("")

            ## Plot 1-2: Delay-Rep-Word
            ax_del_rep_word = fig.add_subplot(gs00[0, 1])
            spectra_delay_rep_word.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                       axes=ax_del_rep_word, cmap=parula_map, title=channel_name)
            ax_del_rep_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_del_rep_word.set_title('Word\nDelay')
            ax_del_rep_word.set_xlabel("")
            ax_del_rep_word.set_ylabel("")
            ax_del_rep_word.set_yticklabels("")
            ax_del_rep_word.set_xticklabels("")

            ## Plot 1-3: Resp-Rep-Word
            ax_res_rep_word = fig.add_subplot(gs00[0, 2])
            spectra_resp_rep_word.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                       axes=ax_res_rep_word, cmap=parula_map, title=channel_name)
            ax_aud_rep_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_res_rep_word.set_title('\nResp')
            ax_res_rep_word.set_xlabel("")
            ax_res_rep_word.set_ylabel("")
            ax_res_rep_word.set_yticklabels("")
            ax_res_rep_word.set_xticklabels("")

            # Colorbar
            ax_colorbar_rep = fig.add_subplot(gs00[0, 3])
            fig.colorbar(ax_res_rep_word.images[-1], cax=ax_colorbar_rep).ax.set_yscale("linear")

            ## Plot 1-4: Auditory-YN-Word
            ax_aud_yn_word = fig.add_subplot(gs00[1, 0])
            spectra_stim_yn_word.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                          axes=ax_aud_yn_word, cmap=parula_map, title=channel_name)
            ax_aud_yn_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_aud_yn_word.set_ylabel("YesNo\nFrequency(Hz)")
            ax_aud_yn_word.set_xlabel("")
            ax_aud_yn_word.set_xticklabels("")

            ## Plot 1-5: Delay-YN-Word
            ax_del_yn_word = fig.add_subplot(gs00[1, 1])
            spectra_delay_yn_word.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                           axes=ax_del_yn_word, cmap=parula_map, title=channel_name)
            ax_del_yn_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_del_yn_word.set_xlabel("")
            ax_del_yn_word.set_ylabel("")
            ax_del_yn_word.set_yticklabels("")
            ax_del_yn_word.set_xticklabels("")

            ## Plot 1-6: Resp-YN-Word
            ax_rsp_yn_word = fig.add_subplot(gs00[1, 2])
            spectra_resp_yn_word.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                           axes=ax_rsp_yn_word, cmap=parula_map, title=channel_name)
            ax_rsp_yn_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_rsp_yn_word.set_xlabel("")
            ax_rsp_yn_word.set_ylabel("")
            ax_rsp_yn_word.set_yticklabels("")
            ax_rsp_yn_word.set_xticklabels("")

            ## Plot 1-7: Auditory-(Rep-YN)-Word
            ax_aud_rep_m_yn_word = fig.add_subplot(gs00[2, 0])
            spectra_stim_rep_m_yn_word.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                          axes=ax_aud_rep_m_yn_word, cmap="RdBu_r", title=channel_name)
            ax_aud_rep_m_yn_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_aud_rep_m_yn_word.set_ylabel("Repeat - YesNo\nFrequency(Hz)")

            ## Plot 1-8: Delay-(Rep-YN)-Word
            ax_del_rep_m_yn_word = fig.add_subplot(gs00[2, 1])
            spectra_delay_rep_m_yn_word.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                           axes=ax_del_rep_m_yn_word, cmap="RdBu_r", title=channel_name)
            ax_del_rep_m_yn_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_del_rep_m_yn_word.set_xlabel("")
            ax_del_rep_m_yn_word.set_ylabel("")
            ax_del_rep_m_yn_word.set_yticklabels("")

            ## Plot 1-9: Resp-(Rep-YN)-Word
            ax_rsp_rep_m_yn_word = fig.add_subplot(gs00[2, 2])
            spectra_resp_rep_m_yn_word.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                           axes=ax_rsp_rep_m_yn_word, cmap="RdBu_r", title=channel_name)
            ax_rsp_rep_m_yn_word.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_rsp_rep_m_yn_word.set_xlabel("")
            ax_rsp_rep_m_yn_word.set_ylabel("")
            ax_rsp_rep_m_yn_word.set_yticklabels("")

            # Colorbar
            ax_colorbar_rep = fig.add_subplot(gs00[2, 3])
            fig.colorbar(ax_rsp_rep_m_yn_word.images[-1], cax=ax_colorbar_rep).ax.set_yscale("linear")
            ############################################################

            ## Plot 2-1: Auditory-Rep-Nonword
            ax_aud_rep_nonword = fig.add_subplot(gs01[0, 0])
            spectra_stim_rep_nonword.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                          axes=ax_aud_rep_nonword, cmap=parula_map, title=channel_name)
            ax_aud_rep_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_aud_rep_nonword.set_title('\nStim')
            ax_aud_rep_nonword.set_xlabel("")
            ax_aud_rep_nonword.set_ylabel("")
            ax_aud_rep_nonword.set_yticklabels("")
            ax_aud_rep_nonword.set_xticklabels("")

            ## Plot 2-2: Delay-Rep-Nonword
            ax_del_rep_nonword = fig.add_subplot(gs01[0, 1])
            spectra_delay_rep_nonword.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                           axes=ax_del_rep_nonword, cmap=parula_map, title=channel_name)
            ax_del_rep_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_del_rep_nonword.set_title('Nonword\nDelay')
            ax_del_rep_nonword.set_xlabel("")
            ax_del_rep_nonword.set_ylabel("")
            ax_del_rep_nonword.set_yticklabels("")
            ax_del_rep_nonword.set_xticklabels("")

            ## Plot 2-3: Resp-Rep-Nonword
            ax_rsp_rep_nonword = fig.add_subplot(gs01[0, 2])
            spectra_resp_rep_nonword.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                           axes=ax_rsp_rep_nonword, cmap=parula_map, title=channel_name)
            ax_rsp_rep_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_rsp_rep_nonword.set_title('\nResp')
            ax_rsp_rep_nonword.set_xlabel("")
            ax_rsp_rep_nonword.set_ylabel("")
            ax_rsp_rep_nonword.set_yticklabels("")
            ax_rsp_rep_nonword.set_xticklabels("")

            ## Plot 2-4: Auditory-YN-Nonword
            ax_aud_yn_nonword = fig.add_subplot(gs01[1, 0])
            spectra_stim_yn_nonword.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                          axes=ax_aud_yn_nonword, cmap=parula_map, title=channel_name)
            ax_aud_yn_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_aud_yn_nonword.set_xlabel("")
            ax_aud_yn_nonword.set_ylabel("")
            ax_aud_yn_nonword.set_yticklabels("")
            ax_aud_yn_nonword.set_xticklabels("")

            ## Plot 2-5: Delay-YN-Nonword
            ax_del_yn_nonword = fig.add_subplot(gs01[1, 1])
            spectra_delay_yn_nonword.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                           axes=ax_del_yn_nonword, cmap=parula_map, title=channel_name)
            ax_del_yn_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_del_yn_nonword.set_xlabel("")
            ax_del_yn_nonword.set_ylabel("")
            ax_del_yn_nonword.set_yticklabels("")
            ax_del_yn_nonword.set_xticklabels("")

            ## Plot 2-6: Resp-YN-Nonword
            ax_rsp_yn_nonword = fig.add_subplot(gs01[1, 2])
            spectra_resp_yn_nonword.plot(channel_name, vlim=vlim_spec, show=False, colorbar=False, fmax=200,
                                           axes=ax_rsp_yn_nonword, cmap=parula_map, title=channel_name)
            ax_rsp_yn_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_rsp_yn_nonword.set_xlabel("")
            ax_rsp_yn_nonword.set_ylabel("")
            ax_rsp_yn_nonword.set_yticklabels("")
            ax_rsp_yn_nonword.set_xticklabels("")

            ## Plot 2-7: Auditory-(Rep-YN)-Nonword
            ax_aud_rep_m_yn_nonword = fig.add_subplot(gs01[2, 0])
            spectra_stim_rep_m_yn_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                          axes=ax_aud_rep_m_yn_nonword, cmap="RdBu_r", title=channel_name)
            ax_aud_rep_m_yn_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_aud_rep_m_yn_nonword.set_xlabel("")
            ax_aud_rep_m_yn_nonword.set_ylabel("")
            ax_aud_rep_m_yn_nonword.set_yticklabels("")

            ## Plot 2-8: Delay-(Rep-YN)-Nonword
            ax_del_rep_m_yn_nonword = fig.add_subplot(gs01[2, 1])
            spectra_delay_rep_m_yn_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                           axes=ax_del_rep_m_yn_nonword, cmap="RdBu_r", title=channel_name)
            ax_del_rep_m_yn_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_del_rep_m_yn_nonword.set_xlabel("")
            ax_del_rep_m_yn_nonword.set_ylabel("")
            ax_del_rep_m_yn_nonword.set_yticklabels("")

            ## Plot 2-9: Resp-(Rep-YN)-Nonword
            ax_rsp_rep_m_yn_nonword = fig.add_subplot(gs01[2, 2])
            spectra_resp_rep_m_yn_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                           axes=ax_rsp_rep_m_yn_nonword, cmap="RdBu_r", title=channel_name)
            ax_rsp_rep_m_yn_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_rsp_rep_m_yn_nonword.set_xlabel("")
            ax_rsp_rep_m_yn_nonword.set_ylabel("")
            ax_rsp_rep_m_yn_nonword.set_yticklabels("")

            ############################################################

            ## Plot 3-1: Auditory-Rep-(Word - Nonword)
            ax_aud_rep_word_m_nonword = fig.add_subplot(gs02[0, 0])
            spectra_stim_rep_word_m_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                       axes=ax_aud_rep_word_m_nonword, cmap="RdBu_r", title=channel_name)
            ax_aud_rep_word_m_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_aud_rep_word_m_nonword.set_title('\nStim')
            ax_aud_rep_word_m_nonword.set_xlabel("")
            ax_aud_rep_word_m_nonword.set_ylabel("")
            ax_aud_rep_word_m_nonword.set_xticklabels("")

            ## Plot 3-2: Delay-Rep-(Word - Nonword)
            ax_delay_rep_word_m_nonword = fig.add_subplot(gs02[0, 1])
            spectra_delay_rep_word_m_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                       axes=ax_delay_rep_word_m_nonword, cmap="RdBu_r", title=channel_name)
            ax_delay_rep_word_m_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_delay_rep_word_m_nonword.set_title('Difference\nDelay')
            ax_delay_rep_word_m_nonword.set_xlabel("")
            ax_delay_rep_word_m_nonword.set_ylabel("")
            ax_delay_rep_word_m_nonword.set_yticklabels("")
            ax_delay_rep_word_m_nonword.set_xticklabels("")


            ## Plot 3-3: Resp-Rep-(Word - Nonword)
            ax_resp_rep_word_m_nonword = fig.add_subplot(gs02[0, 2])
            spectra_resp_rep_word_m_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                       axes=ax_resp_rep_word_m_nonword, cmap="RdBu_r", title=channel_name)
            ax_resp_rep_word_m_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_resp_rep_word_m_nonword.set_title('\nResp')
            ax_resp_rep_word_m_nonword.set_xlabel("")
            ax_resp_rep_word_m_nonword.set_ylabel("")
            ax_resp_rep_word_m_nonword.set_yticklabels("")
            ax_resp_rep_word_m_nonword.set_xticklabels("")

            ## Plot 3-4: Auditory-YN-(Word - Nonword)
            ax_aud_yn_word_m_nonword = fig.add_subplot(gs02[1, 0])
            spectra_stim_yn_word_m_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                       axes=ax_aud_yn_word_m_nonword, cmap="RdBu_r", title=channel_name)
            ax_aud_yn_word_m_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_aud_yn_word_m_nonword.set_xlabel("")
            ax_aud_yn_word_m_nonword.set_ylabel("")
            ax_aud_yn_word_m_nonword.set_xticklabels("")

            ## Plot 3-5: Delay-YN-(Word - Nonword)
            ax_delay_yn_word_m_nonword = fig.add_subplot(gs02[1, 1])
            spectra_delay_yn_word_m_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                       axes=ax_delay_yn_word_m_nonword, cmap="RdBu_r", title=channel_name)
            ax_delay_yn_word_m_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_delay_yn_word_m_nonword.set_xlabel("")
            ax_delay_yn_word_m_nonword.set_ylabel("")
            ax_delay_yn_word_m_nonword.set_yticklabels("")
            ax_delay_yn_word_m_nonword.set_xticklabels("")

            ## Plot 3-6: Resp-YN-(Word - Nonword)
            ax_resp_yn_word_m_nonword = fig.add_subplot(gs02[1, 2])
            spectra_resp_yn_word_m_nonword.plot(channel_name, vlim=vlim_diff, show=False, colorbar=False, fmax=200,
                                       axes=ax_resp_yn_word_m_nonword, cmap="RdBu_r", title=channel_name)
            ax_resp_yn_word_m_nonword.axvline(0, linewidth=0.5, color="black", linestyle=":")
            ax_resp_yn_word_m_nonword.set_xlabel("")
            ax_resp_yn_word_m_nonword.set_ylabel("")
            ax_resp_yn_word_m_nonword.set_yticklabels("")
            ax_resp_yn_word_m_nonword.set_xticklabels("")

            # Colorbar
            ax_colorbar_rep = fig.add_subplot(gs02[0, 3])
            fig.colorbar(ax_resp_rep_word_m_nonword.images[-1], cax=ax_colorbar_rep).ax.set_yscale("linear")

            if not os.path.exists(os.path.join(save_dir, subject,plot_dir_out_Tag)):
                os.mkdir(os.path.join(save_dir, subject,plot_dir_out_Tag))

            figdir = os.path.join(save_dir, subject,plot_dir_out_Tag,f'{channel_name}.jpg')
            plt.savefig(figdir,dpi=300)
            plt.close(fig)
            #plt.show()
        # chan_grid(spectra, size = (20,10),vlim=vlim, fmax=200, show=True, cmap=parula_map,)# yscale="log")



    except Exception as e:
        print(e)