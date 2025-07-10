#%% Import everything
import os
# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

import sys
import pandas as pd
import glm_utils as glm
import numpy as np
import itertools
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle


#%% Set parameters
mask_type='glm' #hg: used high-gamma permutation time-cluster masks; glm: use glm permutation time-cluster masks
plot_wave_type='stat' #stat: plot the HG stat in wave plots; mask: plot the HG significant mask in wave plots.
mask_corr_type = 'cluster_mask'  # cluster_mask: mask from glm time perm cluster; # org_mask: mask from permutation (original R2 ranked in null distribution) # fdr_mask: after fdr correction.
bin=False
if bin:
    mask_corr_type = 'org_mask'  # cluster_mask: mask from glm time perm cluster; # org_mask: mask from permutation (original R2 ranked in null distribution) # fdr_mask: after fdr correction.
else:
    mask_corr_type='cluster_mask' #cluster_mask: mask from glm time perm cluster; # org_mask: mask from permutation (original R2 ranked in null distribution) # fdr_mask: after fdr correction.

with open('glm_config.json', 'r') as f:
    config = json.load(f)

# Extract parameters from config
Acoustic_col = config['Acoustic_col']
Phonemic_col = config['Phonemic_col']
Lexical_col = config['Lexical_col']

event_suffix=('inYN')
events = [f"Auditory_{event_suffix}",f"Resp_{event_suffix}"]
stat = "zscore"
if event_suffix=='inYN':
    task_Tags = ["Yes_No"]
elif event_suffix=='inRep':
    task_Tags = ["Repeat"]
wordnesses = ["ALL"]
glm_feas = ["Word","Nonword"]
cluster_twin=0.011
mean_word_len=0.5
auditory_decay=0
delay_len=1
# motor_prep_win=[-0.5,-0.1]
# motor_resp_win=[0.25,0.75]
Waveplot_wth=18 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots

# %% get hg masks
if mask_type == 'hg':

    from ieeg.arrays.label import LabeledArray
    # % Set parameters: HG
    groupsTag = "LexDelay"
    stat_type = 'mask'
    contrast = 'ave'  # average, not contrasting different conditions

    # For lexical delay task, whether run the data only with repeat tasks
    # Delayseleted=''
    Delayseleted = '_inRep'
    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "Box", "CoganLab")

    stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
    stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

    if groupsTag == "LexDelay":
        hgmask_aud, _ = gp.load_stats(stat_type, 'Auditory' + Delayseleted, contrast, stats_root_delay, stats_root_delay)
        hgmask_resp, _ = gp.load_stats(stat_type, 'Resp' + Delayseleted, contrast, stats_root_delay, stats_root_delay)

    # Select a electrode category to analyze e.g., Auditory, Sensory-motor, or Motor.
    # Set the other active electrodes as nan
    with open(os.path.join('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM', 'data',
                           'Lex_twin_idxes_hg.npy'), "rb") as f:
        Lex_twin_idxes_hg = pickle.load(f)
        # !!!!!!!!!!!!!!!!! Know that now we just select the Motor electrodes !!!!!!!!!!!!!!!!
        sel_idx=Lex_twin_idxes_hg['LexDelay_Motor_sig_idx']

    hgmask_aud_data=hgmask_aud.__array__()
    mask = np.ones(hgmask_aud_data.shape[0], dtype=bool)
    mask[list(sel_idx)] = False
    hgmask_aud_data[mask, :] = np.nan
    labels = [hgmask_aud.labels[0], hgmask_aud.labels[1]]
    hgmask_aud = LabeledArray(hgmask_aud_data, labels)

    hgmask_resp_data=hgmask_resp.__array__()
    mask = np.ones(hgmask_resp_data.shape[0], dtype=bool)
    mask[list(sel_idx)] = False
    hgmask_aud_data[mask, :] = np.nan
    labels = [hgmask_resp.labels[0], hgmask_resp.labels[1]]
    hgmask_resp = LabeledArray(hgmask_resp_data, labels)

#%% Load masks, sort, getting sort index, and plot the ranks
# sig_idx_arr=np.empty((len(events), len(task_Tags),len(glm_feas),4), dtype=object)
# sig_idx_lab=tuple(tuple(item) for item in (events,task_Tags,glm_feas,('all','aud','del','resp')))
# sig_idx=LabeledArray(sig_idx_arr,sig_idx_lab)
sig_idx=dict()
avg=dict()
stass=dict()
peaks_all=dict()
peaks_aud=dict()
peaks_del=dict()
peaks_aud_del=dict()
for event, task_Tag, wordness in itertools.product(events,task_Tags,wordnesses):
    subjs, _, _, chs, times = glm.fifread(event, 'zscore', task_Tag,wordness,preonset_bsl_correct=False)
    for glm_fea in glm_feas:
        # if task_Tag == "Yes_No" and ((event=='Auditory' and glm_fea=='Acoustic') or glm_fea=='Acoustic'):# (event != "Resp" or glm_fea != "Lexical"):
        #     continue
        if wordness != "ALL" and glm_fea == "Lexical":
            continue
        else:
            masks,stats,_=glm.load_stats(event,stat,task_Tag,mask_corr_type,glm_fea,subjs,chs,times,wordness,bin=bin)
            if mask_type == 'glm':
                if event.split('_')[0]=='Auditory' or event.split('_')[0]=='Cue':
                    hgmask_aud=masks
                elif event.split('_')[0]=='Resp':
                    hgmask_resp=masks

            if plot_wave_type=='stat':
                if glm_feas[0]=='Rep_YN' or glm_feas[0]=='YN_Rep':
                    stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}']=stats*5e8
                else:
                    stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}']=stats*5e3
            elif plot_wave_type=='mask':
                stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}']=masks*100
            del masks,stats
            if event.split('_')[0]=='Auditory':
                # whole trial
                all_masks_sorted,all_masks_raw,_,all_masks_sig = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'],cluster_twin,[-0.5,5],mask_data=True,bin=bin)
                # plot time slices
                # gp.plot_brain_window(hgmask_aud, stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,
                #                      [-0.5,1.6,0.025,0.1],
                #                      os.path.join('plot', f'{event}/{task_Tag}/{wordness}/{glm_fea}'),mode="save_region_hist",label_every=None,bin=bin)

                cut_wins,plot_brain_sigs,plot_brain_chs_all=gp.plot_brain_window(hgmask_aud, stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,
                                     [-0.25,1.6,0.025,0.1],
                                     os.path.join('plot', f'{event}/{task_Tag}/{wordness}/{glm_fea}'),mode="save_brain_plot",label_every=None,bin=bin)
                for i, cut_win in enumerate(cut_wins):
                    sig = plot_brain_sigs[i]
                    chs_sel = plot_brain_chs_all[list(sig)].tolist()
                    ch_labels_roi, _ = gp.chs2atlas(subjs, plot_brain_chs_all)
                    gp.atlas2_hist(ch_labels_roi, chs_sel, [0,0,1],
                                os.path.join(os.path.join('plot', f'{event}/{task_Tag}/{wordness}/{glm_fea}'), f'Atlas_{cut_win[0]}_{cut_win[1]}.jpg'), ylim=[0, 20])

                # for hemi in ['lh','rh']:
                #     gp.create_video_from_images('plot', event, task_Tag, wordness, glm_fea, hemi, cut_wins)
                _, all_masks_peak = gp.get_latency(all_masks_raw,'clustermid')
                gp.plot_chs(all_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{wordness}_{glm_fea}_all.jpg'),f"N chs = {len(all_masks_sig)}",bin=bin)
                sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/all"] = all_masks_sig
                peaks_all[f"{event}/{task_Tag}/{wordness}/{glm_fea}/all"] = all_masks_peak
                if mask_type=='glm' and bin==False:
                    # preonset window
                    _,_,_,preonset_masks_sig = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,[-0.5,0])
                    sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/preonset"] = preonset_masks_sig
                    # auditory window
                    aud_masks_sorted,aud_masks_raw,_,aud_masks_sig = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,[-0.1,mean_word_len+auditory_decay])
                    _,aud_masks_peak=gp.get_latency(aud_masks_raw,'clustermid')
                    aud_masks_avg,_,_=gp.time_avg_select(aud_masks_raw, [aud_masks_sig],normalize=True)
                    gp.plot_chs(aud_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{wordness}_{glm_fea}_aud.jpg'),f"N chs = {len(aud_masks_sig)}")
                    sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/aud"] = aud_masks_sig
                    peaks_aud[f"{event}/{task_Tag}/{wordness}/{glm_fea}/aud"] = aud_masks_peak
                    avg[f"{event}/{task_Tag}/{wordness}/{glm_fea}/aud"] = aud_masks_avg
                    # delay window
                    del_masks_sorted,del_masks_raw,_,del_masks_sig = gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,[mean_word_len+auditory_decay-0.1,mean_word_len+auditory_decay+delay_len+0.1])
                    _,del_masks_peak=gp.get_latency(del_masks_raw,'clustermid')
                    del_masks_avg,_,_=gp.time_avg_select(del_masks_raw, [del_masks_sig],normalize=True)
                    gp.plot_chs(del_masks_sorted,os.path.join('plot',f'{event}_{task_Tag}_{wordness}_{glm_fea}_del.jpg'),f"N chs = {len(del_masks_sig)}")
                    sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/del"] = del_masks_sig
                    peaks_del[f"{event}/{task_Tag}/{wordness}/{glm_fea}/del"] = del_masks_peak
                    avg[f"{event}/{task_Tag}/{wordness}/{glm_fea}/del"] = del_masks_avg
                    # aud_delay window (before motor)
                    _,aud_del_masks_raw,_,_=gp.sort_chs_by_actonset(hgmask_aud,stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,[0.1,mean_word_len+auditory_decay+delay_len+0.1])
                    _,aud_del_masks_peak=gp.get_latency(aud_del_masks_raw,'clustermid')
                    peaks_aud_del[f"{event}/{task_Tag}/{wordness}/{glm_fea}/aud_del"] = aud_del_masks_peak

            elif event.split('_')[0]=="Resp":
                # response window
                resp_masks_sorted, resp_masks_raw, _, resp_masks_sig = gp.sort_chs_by_actonset(hgmask_resp, stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin, [-0.1, 5])
                gp.plot_brain_window(hgmask_resp, stass[f'{event}/{task_Tag}/{wordness}/{glm_fea}'], cluster_twin,
                                     [-0.5,1,0.025,0.1],
                                     os.path.join('plot', f'{event}/{task_Tag}/{wordness}/{glm_fea}'))
                resp_masks_avg, _, _ = gp.time_avg_select(resp_masks_raw, [resp_masks_sig],normalize=True)
                gp.plot_chs(resp_masks_sorted, os.path.join('plot', f'{event}_{task_Tag}_{wordness}_{glm_fea}_resp.jpg'),
                            f"N chs = {len(resp_masks_sig)}")
                sig_idx[f"{event}/{task_Tag}/{wordness}/{glm_fea}/resp"] = resp_masks_sig
                avg[f"{event}/{task_Tag}/{wordness}/{glm_fea}/resp"] = resp_masks_avg

#%% Plot peaks and do stats
if mask_type=='glm':
    for df,peak_Tag in zip((pd.DataFrame(peaks_all),pd.DataFrame(peaks_aud),pd.DataFrame(peaks_del),pd.DataFrame(peaks_aud_del)),('All','Aud','Del','Aud_Del')):
        # reshape data
        df = df.dropna(how='all')
        df.columns = ['/'.join(col.split('/')[-2:-1]) for col in df.columns]
        df_long = df.reset_index().melt(id_vars='index', var_name='feature', value_name='value')
        df_long.rename(columns={'index': 'trial'}, inplace=True)

        #plot
        plt.figure(figsize=(8, 11))

        boxplot_colors= [Acoustic_col, Phonemic_col, Lexical_col]
        stripplot_colors = boxplot_colors

        ytitles = ['Peak latency from stim onset (ms)']
        subtitles = [f'GLM R-squared peak latency in {peak_Tag}']
        x_order = ['Acoustic', 'Phonemic', 'Lexical']

        y_limits = [(0,1.5)]
        # y_ticks = [range(0, 1.5, 0.1)]

        for i, var in enumerate(['value'], start=1):

            plt.subplot(1, 1, i)

            barbar = sns.barplot(x='feature', y=var, errorbar=None, data=df_long, order=x_order, saturation=1,
                                 fill=True, alpha=1, linewidth=0.8, capsize=0.1, zorder=1)
            j = 0
            for patch in barbar.patches:
                patch.set_facecolor(boxplot_colors[j])
                j = j + 1
                if j == 3:
                    break

            # ax=sns.boxplot(x='Group', y=var, data=data,showfliers=False, hue='Group',order=x_order,saturation=1)
            sns.despine()

            stripstrip = sns.stripplot(df_long, x="feature", y=var, size=4, alpha=1, jitter=0.1, linewidth=0.5,
                                       edgecolor='white', order=x_order, zorder=2, dodge=True)

            for k in range(3):
                path_collection = stripstrip.collections[k]
                path_collection.set_facecolor(stripplot_colors[k])

            # gp.bsliang_add_connecting_lines(plt, 0, stripstrip)
            # gp.bsliang_add_connecting_lines(plt, 3, stripstrip)

            # Choice 3: bar plot with fill - errbar
            ax2 = sns.barplot(x='feature', y=var, errorbar='se', data=df_long, order=x_order, saturation=1,
                              fill=False, alpha=0.5, linewidth=0, capsize=0.1, err_kws={'linewidth': 0.8, 'color': 'black'},
                              zorder=3)

            plt.xlabel('')
            plt.ylabel(ytitles[i - 1])#, y=gp.bsliang_align_yaxis(y_limits[i - 1], y_ticks[i - 1]))
            plt.ylim(y_limits[i - 1])
            # plt.yticks(y_ticks[i - 1])
            plt.title(subtitles[i - 1], y=1.4)
            plt.gca().tick_params(axis='x', direction='in', length=0, labelrotation=45)
            plt.gca().tick_params(axis='y', direction='in', length=2)
            # plt.gca().get_legend().remove()

        plt.tight_layout(w_pad=1.5)
        plt.savefig(os.path.join('plot',f'Peak latency {peak_Tag}.tif'), dpi=300,bbox_inches='tight', transparent=False)

    # Stats
    from scipy.stats import ttest_ind

    ttest_ind(peaks_aud_del[f'Auditory_inRep/{task_Tag}/ALL/Acoustic/aud_del'],peaks_aud_del[f'Auditory_inRep/{task_Tag}/ALL/Phonemic/aud_del'],nan_policy='omit')
    ttest_ind(peaks_aud_del[f'Auditory_inRep/{task_Tag}/ALL/Phonemic/aud_del'],peaks_aud_del[f'Auditory_inRep/{task_Tag}/ALL/Lexical/aud_del'],nan_policy='omit')
    ttest_ind(peaks_aud_del[f'Auditory_inRep/{task_Tag}/ALL/Acoustic/aud_del'],peaks_aud_del[f'Auditory_inRep/{task_Tag}/ALL/Lexical/aud_del'],nan_policy='omit')

    ttest_ind(peaks_all[f'Auditory_inRep/{task_Tag}/ALL/Acoustic/all'],peaks_all[f'Auditory_inRep/{task_Tag}/ALL/Phonemic/all'],nan_policy='omit')
    ttest_ind(peaks_all[f'Auditory_inRep/{task_Tag}/ALL/Phonemic/all'],peaks_all[f'Auditory_inRep/{task_Tag}/ALL/Lexical/all'],nan_policy='omit')
    ttest_ind(peaks_all[f'Auditory_inRep/{task_Tag}/ALL/Acoustic/all'],peaks_all[f'Auditory_inRep/{task_Tag}/ALL/Lexical/all'],nan_policy='omit')

    ttest_ind(peaks_aud[f'Auditory_inRep/{task_Tag}/ALL/Acoustic/aud'],peaks_aud[f'Auditory_inRep/{task_Tag}/ALL/Phonemic/aud'],nan_policy='omit')
    ttest_ind(peaks_aud[f'Auditory_inRep/{task_Tag}/ALL/Phonemic/aud'],peaks_aud[f'Auditory_inRep/{task_Tag}/ALL/Lexical/aud'],nan_policy='omit')
    ttest_ind(peaks_aud[f'Auditory_inRep/{task_Tag}/ALL/Acoustic/aud'],peaks_aud[f'Auditory_inRep/{task_Tag}/ALL/Lexical/aud'],nan_policy='omit')

    ttest_ind(peaks_del[f'Auditory_inRep/{task_Tag}/ALL/Acoustic/del'],peaks_del[f'Auditory_inRep/{task_Tag}/ALL/Phonemic/del'],nan_policy='omit')
    ttest_ind(peaks_del[f'Auditory_inRep/{task_Tag}/ALL/Acoustic/del'],peaks_del[f'Auditory_inRep/{task_Tag}/ALL/Lexical/del'],nan_policy='omit')
    ttest_ind(peaks_del[f'Auditory_inRep/{task_Tag}/ALL/Phonemic/del'],peaks_del[f'Auditory_inRep/{task_Tag}/ALL/Lexical/del'],nan_policy='omit')

#%% plot significant electrodes
for wordness in wordnesses[:2]:
    for md,md_Tag in zip(['all','aud','del'],['whole trial','auditory window','delay window']):
        if mask_type == 'hg' and (md=='aud' or md=='del'):
            continue
        if md=='all':
            wid_scale=1
        elif md=='aud':
            xlim_l=-0.1
            xlim_r=1.5#mean_word_len + auditory_decay
            wid_scale=(xlim_r-xlim_l)*100/350
        elif md=='del':
            xlim_l=0.5
            xlim_r=1.5
            wid_scale = (xlim_r - xlim_l)*100/350
        plt.figure(figsize=(Waveplot_wth*wid_scale, Waveplot_hgt))
        if wordness == 'ALL':
            if glm_feas[0]=='Rep_YN' or glm_feas[0]=='YN_Rep':
                gp.plot_wave(stass[f'{events[0]}/{task_Tag}/{wordness}/Rep_YN'], sig_idx[f"{events[0]}/{task_Tag}/{wordness}/Rep_YN/{md}"],
                             'Repeat - Yes_No', [117/255, 139/255, 95/255], '-',True)
                gp.plot_wave(stass[f'{events[0]}/{task_Tag}/{wordness}/YN_Rep'], sig_idx[f"{events[0]}/{task_Tag}/{wordness}/YN_Rep/{md}"],
                             'Yes_No - Repeat', [64/255, 100/255, 98/255], '-',True)
            if glm_feas[0]=='Word' or glm_feas[0]=='Nonword':
                gp.plot_wave(stass[f'{events[0]}/{task_Tag}/{wordness}/Word']*10e4, sig_idx[f"{events[0]}/{task_Tag}/{wordness}/Word/{md}"],
                             'Word-Nonword', [117/255, 139/255, 95/255], '-',True)
                gp.plot_wave(stass[f'{events[0]}/{task_Tag}/{wordness}/Nonword']*10e4, sig_idx[f"{events[0]}/{task_Tag}/{wordness}/Nonword/{md}"],
                             'Nonword-Word', [64/255, 100/255, 98/255], '-',True)
            else:
                gp.plot_wave(stass[f'{events[0]}/{task_Tag}/{wordness}/Acoustic'], sig_idx[f"{events[0]}/{task_Tag}/{wordness}/Acoustic/{md}"],
                             'Acoustic', Acoustic_col, '-',True)
                gp.plot_wave(stass[f'{events[0]}/{task_Tag}/{wordness}/Phonemic'], sig_idx[f"{events[0]}/{task_Tag}/{wordness}/Phonemic/{md}"],
                             'Phonemic', Phonemic_col, '-',True)
                gp.plot_wave(stass[f'{events[0]}/{task_Tag}/{wordness}/Lexical'], sig_idx[f"{events[0]}/{task_Tag}/{wordness}/Lexical/{md}"],
                             'Lexical', Lexical_col, '-',True)
        elif wordness == 'Word':
            gp.plot_wave(stass[f'{events[0]}/Yes_No/Word/Acoustic'], sig_idx[f"{events[0]}/Yes_No/Word/Acoustic/{md}"],
                         'Acoustic_Word', Acoustic_col, '-',True)
            gp.plot_wave(stass[f'{events[0]}/Yes_No/Word/Phonemic'], sig_idx[f"{events[0]}/Yes_No/Word/Phonemic/{md}"],
                         'Phonemic_Word', Phonemic_col, '-',True)
            gp.plot_wave(stass[f'{events[0]}/Yes_No/Nonword/Acoustic'], sig_idx[f"{events[0]}/Yes_No/Nonword/Acoustic/{md}"],
                         'Acoustic_Nonword', Acoustic_col, '--',True)
            gp.plot_wave(stass[f'{events[0]}/Yes_No/Nonword/Phonemic'], sig_idx[f"{events[0]}/Yes_No/Nonword/Phonemic/{md}"],
                         'Phonemic_Nonword', Phonemic_col, '--',True)
        plt.axvline(x=0, linestyle='--', color='k')
        plt.axhline(y=0, linestyle='--', color='k')
        if wordness == 'ALL':
            wordness_Tag = 'All'
        else:
            wordness_Tag = 'Word or Nonword'
        # plt.title(f'GLM:  {wordness_Tag} in {md_Tag}',fontsize=20)
        plt.title(f'GLM in {md_Tag} Phase',fontsize=20)
        if plot_wave_type == 'stat':
            plt.ylabel(r'GLM Sum|β| bsl corrected',fontsize=20)
        elif plot_wave_type == 'mask':
            plt.ylabel(r'Perc. of sig. elec. (%)',fontsize=20)
        plt.xlabel('Time from auditory onset (s)')
        plt.gca().spines[['top', 'right']].set_visible(False)
        if md == 'aud' or md=='del':
            plt.xlim(xlim_l, xlim_r)
        plt.tight_layout()
        if md=='all' or md == 'aud':
            plt.legend(fontsize=20)
        plt.savefig(os.path.join('plot',f'wave auditory onset {event_suffix} {wordness} {md} {plot_wave_type}.tif'),dpi=300)
        plt.close()

    xlim_l = -0.2
    xlim_r = mean_word_len
    wid_scale = (xlim_r - xlim_l)*100/350
    plt.figure(figsize=(Waveplot_wth*wid_scale, Waveplot_hgt))
    if wordness == 'ALL':
        gp.plot_wave(stass[f'{events[1]}/{task_Tag}/{wordness}/Acoustic'], sig_idx[f"{events[1]}/{task_Tag}/{wordness}/Acoustic/resp"],
                     f'Acoustic', Acoustic_col, '-',True)
        gp.plot_wave(stass[f'{events[1]}/{task_Tag}/{wordness}/Phonemic'], sig_idx[f"{events[1]}/{task_Tag}/{wordness}/Phonemic/resp"],
                     'Phonemic', Phonemic_col, '-',True)
        gp.plot_wave(stass[f'{events[1]}/{task_Tag}/{wordness}/Lexical'], sig_idx[f"{events[1]}/{task_Tag}/{wordness}/Lexical/resp"], 'Lexical', Lexical_col,'-',True)
        # gp.plot_wave(stass[f'Resp/Yes_No/{wordness}/Phonemic'], sig_idx[f"Resp/Yes_No/{wordness}/Phonemic/resp"], 'Phonemic in Decision', Phonemic_col,'--',False)
        # gp.plot_wave(stass[f'Resp/Yes_No/{wordness}/Lexical'], sig_idx[f"Resp/Yes_No/{wordness}/Lexical/resp"], 'Lexical status in Decision', Lexical_col,'--',False)
    elif wordness == 'Word':
        gp.plot_wave(stass[f'{events[1]}/{task_Tag}/Word/Acoustic'], sig_idx[f"{events[1]}/{task_Tag}/Word/Acoustic/resp"],
                     'Acoustic_Word', Acoustic_col, '-',True)
        gp.plot_wave(stass[f'{events[1]}/{task_Tag}/Word/Phonemic'], sig_idx[f"{events[1]}/{task_Tag}/Word/Phonemic/resp"],
                     'Phonemic_Word', Phonemic_col, '-',True)
        gp.plot_wave(stass[f'{events[1]}/{task_Tag}/Nonword/Acoustic'], sig_idx[f"{events[1]}/{task_Tag}/Nonword/Acoustic/resp"],
                     'Acoustic_Nonword', Acoustic_col, '--',True)
        gp.plot_wave(stass[f'{events[1]}/{task_Tag}/Nonword/Phonemic'], sig_idx[f"{events[1]}/{task_Tag}/Nonword/Phonemic/resp"],
                     'Phonemic_Nonword', Phonemic_col, '--',True)

    if wordness == 'ALL':
        wordness_Tag = 'All'
    else:
        wordness_Tag = wordness

    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.title(f'GLM:  {wordness_Tag} in resp window',fontsize=20)
    if plot_wave_type == 'stat':
        plt.ylabel(r'GLM R^2 bsl corrected', fontsize=20)
    elif plot_wave_type == 'mask':
        plt.ylabel(r'Perc. of sig. elec. (%)', fontsize=20)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.legend(fontsize=15)
    # plt.xlim(xlim_l, xlim_r)
    plt.xlabel('Time from motor onset (s)')
    plt.tight_layout()
    plt.savefig(os.path.join('plot', f'wave motor onset in {wordness} {plot_wave_type}.tif'),dpi=300)
    plt.close()

    with open(os.path.join('data', f'sig_idx_{mask_corr_type}.npy'), "wb") as f:
        pickle.dump(sig_idx, f)
    with open(os.path.join('data', f'avg_{mask_corr_type}.npy'), "wb") as f:
        pickle.dump(avg, f)

#%% Get confusion matrix of glm sig electrode sets
if 1==0:
    for wordness in wordnesses:

        if wordness == 'ALL':
            keys_of_interest = [
                "Auditory_inRep/Repeat/ALL/Acoustic/aud",
                "Auditory_inRep/Repeat/ALL/Phonemic/aud",
                "Auditory_inRep/Repeat/ALL/Lexical/aud",
                "Auditory_inRep/Repeat/ALL/Acoustic/del",
                "Auditory_inRep/Repeat/ALL/Phonemic/del",
                "Auditory_inRep/Repeat/ALL/Lexical/del",
                "Resp_inRep/Repeat/ALL/Acoustic/resp",
                "Resp_inRep/Repeat/ALL/Phonemic/resp",
                "Resp_inRep/Repeat/ALL/Lexical/resp"
            ]
        else:
            keys_of_interest = [
                f"Auditory_inRep/Repeat/{wordness}/Acoustic/aud",
                f"Auditory_inRep/Repeat/{wordness}/Phonemic/aud",
                f"Auditory_inRep/Repeat/{wordness}/Acoustic/del",
                f"Auditory_inRep/Repeat/{wordness}/Phonemic/del",
                f"Resp_inRep/Repeat/{wordness}/Acoustic/resp",
                f"Resp_inRep/Repeat/{wordness}/Phonemic/resp"
             ]

        filtered_sets = {key: sig_idx[key] for key in keys_of_interest}
        short_names = {key: "\n".join(key.split("/")[-2:]).replace('del', 'Delay').replace('aud', 'Auditory').replace('resp', 'Response') for key in keys_of_interest}
        conf_matrix = pd.DataFrame(index=short_names.values(), columns=short_names.values())

        for key1 in keys_of_interest:
            for key2 in keys_of_interest:
                conf_matrix.loc[short_names[key1], short_names[key2]] = (len(filtered_sets[key1] & filtered_sets[key2]) / len(filtered_sets[key1])) * 100

        conf_matrix = conf_matrix.astype(float)

        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=True, fmt=".2f",cmap="rocket_r", xticklabels=short_names.values(),
                        yticklabels=short_names.values(), annot_kws={"size": 14},vmin=0, vmax=100,cbar=False)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.title("Shared encoding electrodes across features and phase (%)",fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join('plot', f'GLM electrode sharing in {wordness}.tif'),dpi=300)
        plt.close()

