# Set dir
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,ttest_ind,linregress
import seaborn as sns
from ieeg.calc.stats import time_cluster
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from ieeg.arrays.label import LabeledArray
import pingouin as pg
from contextlib import redirect_stdout

script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\lme\\prepare_raw.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join("..", "GLM")))
import glm_utils as glm
import sys
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
from scipy.ndimage import gaussian_filter1d,uniform_filter1d

#%% load HG files
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")
trial_labels='CORRECT'
# with open(os.path.join('..', 'GLM', 'data', f'Lex_twin_idxes_hg.npy'), "rb") as f:
#     LexDelay_twin_idxes = pickle.load(f)
data_LexNoDelay_Aud,_=gp.load_stats('mask','Auditory_inRep','ave',stats_root_nodelay,stats_root_nodelay)
epoc_LexNoDelay_Aud, _ = gp.load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_nodelay,trial_labels=trial_labels,keeptrials=False)
data_LexDelay_Aud,_=gp.load_stats('mask','Auditory_inRep','ave',stats_root_nodelay,stats_root_delay)
epoc_LexDelay_Aud, _ = gp.load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_delay,trial_labels=trial_labels,keeptrials=False)

#%% Run time cluster

font_scale=1.5
plt.rcParams['font.size'] = 14*font_scale
plt.rcParams['axes.titlesize'] = 16*font_scale
plt.rcParams['axes.labelsize'] = 12*font_scale
plt.rcParams['xtick.labelsize'] = 12*font_scale
plt.rcParams['ytick.labelsize'] = 12*font_scale
plt.rcParams['legend.fontsize'] = 12*font_scale

def get_traces_clus(raw, alpha:float=0.05, alpha_clus:float=0.05,mode:str='time_cluster'):
    # Load data
    # aud_delay_org = pd.read_csv('Aud_delay_org.csv')
    # aud_delay_perm = pd.read_csv('Aud_delay_perm.csv')
    # raw=bsl_correct(raw)
    time_point = np.unique(raw['time_point'].to_numpy())
    r2s_i_df = raw.pivot_table(
        index='perm',
        columns='time_point',
        values='chi_squared_obs'
    )
    r2s_i = r2s_i_df.to_numpy()

    # r2_i, 1-d time series of original glm values
    # null_r2_i, 2-d time series of original glm values: n_perm*time
    r2_i=r2s_i[0,:]
    r2_i = np.expand_dims(r2_i, axis=0)
    null_r2_i=r2s_i[1:,:]

    # Get original mask
    org_p_i = glm.aaron_perm_gt_1d(r2s_i, axis=0)[0] # 1-d time series
    mask_i_org = (org_p_i > (1 - alpha)).astype(int) # 1-d time series (binary)

    # Get null mask
    null_p_i = glm.aaron_perm_gt_1d(null_r2_i, axis=0) # 2-d time series: n_perm*time
    mask_null_i=(null_p_i>(1-alpha)).astype(int) # # 2-d time series: n_perm*time (binary)

    if mode == 'time_cluster':
        # Time perm cluster
        stat_out = time_cluster(mask_i_org, mask_null_i,1 - alpha_clus)
    elif mode == 'fdr':
        #fdr
        stat_out, p_fdr, _, _ = multipletests(1-org_p_i, alpha=alpha_clus, method='fdr_bh')

    return time_point,r2_i[0],stat_out

def get_subj_elec(filename,del_nodel_tag,elec_typ,pred_onset):
    prefix = f'results\\{del_nodel_tag}_{elec_typ}_'
    suffix = f'_{pred_onset}.csv'
    remaining_string = filename.split(prefix)[-1]
    extracted_part = remaining_string.split(suffix)[0]
    return extracted_part

#%% Get data
pred_onsets=('resp_onset','aud_onset')
del_nodel_tags = ('epoc_LexNoDelay_Cue', 'epoc_LexDelay_Cue', 'epoc_LexDelay_Go')
elec_typs = ('Sensorymotor_vWM', 'Motor_vWM', 'Auditory_vWM', 'Delay_only_vWM', 'Sensorymotor_novWM','Motor_novWM', 'Auditory_novWM')

#%% Plotting
onset_pred_lags_all = dict()
onset_pred_pows_all = dict()
subj_elecs_list_all = dict()
plot_elec_pic=False
r2_lsts=dict()
for pred_onset in pred_onsets:
    onset_pred_lags_onset = dict()
    onset_pred_pows_onset = dict()
    subj_elec_lists_onset = dict()
    for del_nodel_tag in del_nodel_tags:
        if (del_nodel_tag == 'epoc_LexDelay_Cue' and pred_onset == 'resp_onset') or (del_nodel_tag == 'epoc_LexDelay_Go' and pred_onset == 'aud_onset'):
            continue
        onset_pred_lags = dict()
        onset_pred_pows = dict()
        cross_corr_lsts = dict()
        subj_elec_lists = dict()
        for elec_typ in elec_typs:
            #%% read raw hg data
            r2_lst=[]
            clus_lst=[]
            cross_corr_lst=[]

            subj_elec_lst=[]
            subj_elec_onsets=[]
            subj_elec_crosscorr_peak=[]
            subj_elec_crosscorr_prepow=[]
            hg_filename=f'data\\{del_nodel_tag}_{elec_typ}_long.csv'
            hg_raw = pd.read_csv(hg_filename)
            # read data
            file_pattern = f"results/{del_nodel_tag}_{elec_typ}_*_{pred_onset}.csv"
            matching_files = glob.glob(file_pattern, recursive=False)
            ymax=2e-1

            #%% loop for each electrodes
            for filename in matching_files:
                para_sig_bar=[1e-1,0]
                if plot_elec_pic:
                    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(19, 15))
                    ax = axs[0]
                    ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7,linewidth=3)
                raw = pd.read_csv(filename)
                subj_elec=get_subj_elec(filename,del_nodel_tag,elec_typ,pred_onset)
                print(f'{del_nodel_tag}_{elec_typ}:{subj_elec}')

                # plot predicted onsets
                try:
                    subj=subj_elec.split('_')[0]
                    elec=subj_elec.split('_')[1]
                except Exception as e:
                    continue
                hg_raw_subj_elec = hg_raw[
                    (hg_raw['subject'] == subj) &
                    (hg_raw['electrode'] == elec) &
                    (hg_raw['time_point']==0)
                ]

                # if hg_raw_subj_elec.empty:
                #     continue

                resp_onset_data_subj_elec = hg_raw_subj_elec[pred_onset].dropna().tolist()
                n_dots = len(resp_onset_data_subj_elec)
                random_y_values = np.random.uniform(0, ymax-(1e-2), n_dots)
                if plot_elec_pic:
                    ax.scatter(resp_onset_data_subj_elec, random_y_values, color='red', s=50, alpha=0.7)

                # Plot predicting time series
                time_point, time_series, mask_time_clus = get_traces_clus(raw, 5e-2, 5e-2,mode='time_cluster')
                time_series=gaussian_filter1d(time_series, sigma=1, mode='nearest')

                if plot_elec_pic:
                    ax.plot(time_point, time_series, color=[0,0,1], linewidth=5,label='Aud prediction')

                # Plot HG traces
                hg_raw_subj_elec_rep = hg_raw[
                    (hg_raw['subject'] == subj) &
                    (hg_raw['electrode'] == elec)
                ]
                subj_elec_hg_series = hg_raw_subj_elec_rep.groupby('time_point')['value'].mean().tolist()
                subj_elec_hg_series = (np.max(time_series)*(subj_elec_hg_series-np.min(subj_elec_hg_series))/
                                       (np.max(subj_elec_hg_series)-np.min(subj_elec_hg_series)))
                # if len(time_point)!=len(subj_elec_hg_series):
                #     continue
                if plot_elec_pic:
                    ax.plot(time_point,subj_elec_hg_series,color=[0,1,0], linewidth=5,label='HG trace (normalized)')

                # Do HG-onsethistogram cross-correlation and plot
                    # histogram
                onset_times = np.array(resp_onset_data_subj_elec)
                counts, _ = np.histogram(onset_times, bins=time_point)
                counts=np.append(counts, 0)
                total_onsets = len(onset_times)
                onset_prob_series = counts / total_onsets
                    # cross-corr (move beta, keep histogram)
                cross_corr = np.correlate(time_series,onset_prob_series, mode='full')
                time_resolution = np.diff(time_point)[0]
                lags_index = np.arange(-(len(time_point) - 1), len(time_point))
                # Get peak time loc
                if pred_onset=='aud_onset':
                    LOWER_BOUND_SEC = -0.5
                    UPPER_BOUND_SEC = 0.5
                    LOWER_BOUND_SEC_mtrprep = LOWER_BOUND_SEC
                    UPPER_BOUND_SEC_mtrprep = UPPER_BOUND_SEC
                elif pred_onset=='resp_onset':
                    LOWER_BOUND_SEC = -1.5
                    UPPER_BOUND_SEC = 1.5
                    LOWER_BOUND_SEC_mtrprep = LOWER_BOUND_SEC
                    UPPER_BOUND_SEC_mtrprep = 0
                lags_sec = lags_index * time_resolution
                start_idx_limit = np.argmax(lags_sec >= LOWER_BOUND_SEC)
                end_idx_limit = np.argmax(lags_sec > UPPER_BOUND_SEC)-1
                limited_cross_corr = cross_corr[start_idx_limit: end_idx_limit + 1]
                # total_sum_limit = np.sum(limited_cross_corr)
                # half_sum_threshold_limit = total_sum_limit / 2.0
                # cumulative_sum_limit = np.cumsum(limited_cross_corr)
                # median_sum_index_limited = np.argmax(cumulative_sum_limit >= half_sum_threshold_limit)
                max_sum_index_limited = np.argmax(limited_cross_corr)
                # median_corr_index = start_idx_limit + median_sum_index_limited
                # optimal_lag_k = lags_index[median_corr_index]
                max_corr_index = start_idx_limit + max_sum_index_limited
                optimal_lag_k = lags_index[max_corr_index]
                optimal_lag_time = optimal_lag_k * time_resolution
                time_lags = lags_index * time_resolution
                i_ts_peak = np.argmax(onset_prob_series)
                time_ts_peak = time_point[i_ts_peak]
                cross_corr_peak = time_ts_peak+optimal_lag_time
                # Get power of motor prep
                start_idx_limit_mtrprep = np.argmax(lags_sec >= LOWER_BOUND_SEC_mtrprep)
                end_idx_limit_mtrprep = np.argmax(lags_sec > UPPER_BOUND_SEC_mtrprep)-1
                limited_cross_corr_mtrprep = cross_corr[start_idx_limit_mtrprep: end_idx_limit_mtrprep + 1]
                mtr_prep_rms = np.sqrt(np.nanmean(limited_cross_corr_mtrprep**2))

                # plot CC indexes
                if plot_elec_pic:
                    ax.axvline(x=time_ts_peak, color='red', linestyle='--', alpha=0.7,linewidth=3,label='onset peak')
                    ax.axvline(x=cross_corr_peak, color='blue', linestyle='--', alpha=0.7,linewidth=3,label='onset peak + cc peak lag')

                    ax1=axs[1]
                    ax1.plot(time_lags, cross_corr, label='Cross-Correlation')
                    ax1.axvline(
                        optimal_lag_time,
                        color='r',
                        linestyle='--',
                        linewidth=1.5,
                        label=f'Optimal Lag: {optimal_lag_time:.4f} s ({optimal_lag_k} steps)'
                    )
                    ax1.axvline(0, color='k', linestyle=':', linewidth=0.8)
                    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
                    ax1.set_ylim([-0.02,0.1])
                    ax1.set_title('Cross-Correlation vs. Time Lag')
                    ax1.set_xlabel('Time Lag (seconds)')
                    ax1.set_ylabel('Cross-Correlation Value (Unnormalized)')
                    ax1.legend()
                    ax1.grid(True, linestyle='--')

                r2_lst.append(time_series)
                clus_lst.append(mask_time_clus.astype(int))
                cross_corr_lst.append(cross_corr)
                subj_elec_lst.append(subj_elec)
                subj_elec_onsets.append(time_ts_peak)
                subj_elec_crosscorr_peak.append(cross_corr_peak)
                subj_elec_crosscorr_prepow.append(mtr_prep_rms)

                if plot_elec_pic:
                    ax.legend(loc='upper left')
                    true_indices = np.where(mask_time_clus)[0]
                    if true_indices.size > 0:
                        split_points = np.where(np.diff(true_indices) != 1)[0] + 1
                        clusters_indices = np.split(true_indices, split_points)

                        for k, cluster in enumerate(clusters_indices):
                            start_index = cluster[0]
                            end_index = cluster[-1]

                            time_step = time_point[1] - time_point[0]
                            start_time = time_point[start_index] - time_step / 2
                            end_time = time_point[end_index] + time_step / 2

                            label = f'clust{k} of pho'
                            ax.plot([start_time, end_time], [para_sig_bar[0],para_sig_bar[0]],
                                    color=[0,0,1],alpha=0.4,
                                    linewidth=10,  # Make the line thick like a bar
                                    solid_capstyle='butt')  # Makes the line ends flat

                    ax.set_ylabel("$X^2$")#, fontsize=20)
                    ax.tick_params(axis='both', which='major')#, labelsize=16)
                    ax.set_xlim(-0.2, 5)
                    ax.set_ylim(-2e-2,ymax)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    plt.savefig(os.path.join('figs', f'{del_nodel_tag}_{elec_typ}_{subj_elec}_full_{pred_onset}.tif'), dpi=100)
                    plt.close()

            #%% Plot HG signals in Delay/No Delay and R-aquared predicting onsets
            subj_elec_lists[elec_typ]=subj_elec_lst
            cross_corr_lsts[elec_typ]=np.stack(cross_corr_lst, axis=0)
            r2_lsts[elec_typ]=np.stack(r2_lst, axis=0)
            onset_pred_pows[elec_typ]=np.array(subj_elec_crosscorr_prepow)

            # Generate Label array
            labels=(subj_elec_lst,time_point)
            r2_arr=LabeledArray(np.stack(r2_lst,axis=0), labels)
            clus_arr=LabeledArray(np.stack(clus_lst,axis=0), labels)
            # Get HG activity for Delay and Nodelay (aligned to Delay)
            subj_elec_lst_search=[f'{s.split('_')[0]}-{s.split('_')[1]}' for s in subj_elec_lst]
            Mask_NoDel_Aud = data_LexNoDelay_Aud.take(subj_elec_lst_search, axis=0)
            Mask_Delay_Aud = data_LexDelay_Aud.take(subj_elec_lst_search,axis=0)
            Epoc_NoDel_Aud = epoc_LexNoDelay_Aud.take(subj_elec_lst_search, axis=0)
            Epoc_Delay_Aud = epoc_LexDelay_Aud.take(subj_elec_lst_search, axis=0)

            # Plot HG activity for Delay
            epoc_del, _, sorted_indices,*_ = gp.sort_chs_by_actonset(Mask_Delay_Aud,
                                                          Epoc_Delay_Aud,
                                                          0.011, [-0.2, 5],
                                                          mask_data=False,
                                                          select_electrodes=False)

            gp.plot_chs(epoc_del, os.path.join('figs', f'Del_{elec_typ}_{pred_onset}.tif'),
                     f'{pred_onset}', percentage_vscale=False, vmin=0, vmax=3, is_colbar=False,
                     fig_size=[4, 20 * (np.shape(epoc_del)[0] / 250)])

            # Plot HG activity for NoDelay
            epoc_nodel, *_ = gp.sort_chs_by_actonset(Mask_NoDel_Aud,
                                                          Epoc_NoDel_Aud,
                                                          0.011, [-0.2, 5],
                                                          sorted_indices=sorted_indices,
                                                          mask_data=False,
                                                          select_electrodes=False)

            gp.plot_chs(epoc_nodel, os.path.join('figs', f'NoDel_{elec_typ}_{pred_onset}.tif'),
                     f'{pred_onset}', percentage_vscale=False, vmin=0, vmax=3, is_colbar=False,
                     fig_size=[3, 20 * (np.shape(epoc_del)[0] / 250)])

            # Plot R-2: aligned to onsets in Delay
            epoch_sort_mask, *_ = gp.sort_chs_by_actonset(clus_arr,
                                                          r2_arr,
                                                          0.011, [-0.2, 5],
                                                          sorted_indices=sorted_indices,
                                                          mask_data=False,
                                                          select_electrodes=False)
            subj_elec_onsets_array = np.array(subj_elec_onsets)
            subj_elec_onsets_array = subj_elec_onsets_array[sorted_indices.tolist()]
            subj_elec_crosscorr_peak_array = np.array(subj_elec_crosscorr_peak)
            subj_elec_crosscorr_peak_array = subj_elec_crosscorr_peak_array[sorted_indices.tolist()]
            gp.plot_chs(epoch_sort_mask, os.path.join('figs', f'NoDel_pred_{elec_typ}_{pred_onset}.tif'),
                        f'{pred_onset}', percentage_vscale=False, vmin=0, vmax=5e-1, is_colbar=False,
                        fig_size=[6, 20 * (np.shape(r2_lst)[0] / 250)],
                        scatter_onsets=subj_elec_onsets_array.tolist())
                        #scatter_onsets_2=subj_elec_crosscorr_peak_array.tolist())

            # Get the cc lag:
            cc_lag = subj_elec_crosscorr_peak_array - subj_elec_onsets_array
            onset_pred_lags[elec_typ] = cc_lag

            # R-2: aligned to onsets in nodelay r^2 cluster onsets
            if 1==0:
                epoch_sort_mask, _, sorted_indices,*_= gp.sort_chs_by_actonset(clus_arr,
                                                              r2_arr,
                                                              0.011, [-0.2, 5],
                                                              mask_data=True,
                                                              select_electrodes=True)
                subj_elec_onsets_array = np.array(subj_elec_onsets)
                subj_elec_onsets_array = subj_elec_onsets_array[sorted_indices.tolist()]
                subj_elec_crosscorr_peak_array = np.array(subj_elec_crosscorr_peak)
                subj_elec_crosscorr_peak_array = subj_elec_crosscorr_peak_array[sorted_indices.tolist()]
                gp.plot_chs(epoch_sort_mask, os.path.join('figs', f'NoDel_pred_{elec_typ}_{pred_onset}_sigr2.tif'),
                            f'{pred_onset}', percentage_vscale=False, vmin=0, vmax=5e-1, is_colbar=False,
                            fig_size=[6, 20 * (np.shape(r2_lst)[0] / 250)],
                            scatter_onsets=subj_elec_onsets_array.tolist(),
                            scatter_onsets_2=subj_elec_crosscorr_peak_array.tolist())


            if elec_typ=='Auditory_NoDelay':
                #%% glm HG correlations

                # Get windowed paras
                # predicting onsets
                _, _, _, _, _, paras_glm, *_ = gp.sort_chs_by_actonset(clus_arr,
                                                              r2_arr,
                                                              0.011, [0, 0.2],
                                                              sorted_indices=sorted_indices,
                                                              mask_data=True,
                                                              select_electrodes=False,scatter_onsets=subj_elec_onsets_array.tolist())
                paras_glm.index = paras_glm.index.str.replace('_', '-')

                # high gamma
                _, _, _, _, _, paras_hg, *_ = gp.sort_chs_by_actonset(Mask_NoDel_Aud,
                                                              Epoc_NoDel_Aud,
                                                              0.011, [0, 0.2],
                                                              sorted_indices=sorted_indices,
                                                              mask_data=True,
                                                              select_electrodes=False)

                # extract key parameters of the responses
                df_merged = pd.DataFrame({
                    'sum_value_glm': paras_glm['rms_value'],
                    'sum_value_hg': paras_hg['rms_value']
                }).dropna()
                df_merged = df_merged[df_merged['sum_value_hg'] != 0]
                df_merged2 = df_merged[df_merged['sum_value_glm'] != 0]
                data_glm_aligned = df_merged2['sum_value_glm']
                data_hg_aligned = df_merged2['sum_value_hg']

                # do correlation
                correlation, p_value = pearsonr(data_glm_aligned, data_hg_aligned)

                # Fig1: correlation
                sns.set_theme(style="whitegrid")
                plt.figure(figsize=(8, 6))

                sns.scatterplot(
                    x=data_glm_aligned,
                    y=data_hg_aligned,
                    s=100,
                    color='#3498db'
                )

                sns.regplot(
                    x=data_glm_aligned,
                    y=data_hg_aligned,
                    scatter=False,
                    color='#e74c3c',
                    line_kws={'linestyle': '--', 'linewidth': 2}
                )

                plt.title(
                    f'Correlations between Auditory HG and stim onset GLM',
                    fontsize=14,
                    fontweight='bold'
                )
                plt.xlabel('RMS GLM R-squared (200ms win)', fontsize=12)
                plt.ylabel('RMS HG Z-score (200ms win)', fontsize=12)

                plt.text(
                    x=np.min(data_glm_aligned),
                    y=np.max(data_hg_aligned),
                    s=f'$r={correlation:.3f},p={p_value:.3f}$',
                    fontsize=12,
                    color='#e74c3c',
                    ha='left',
                    va='top'
                )

                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join('figs', f'NoDel_{elec_typ}_{pred_onset}_glm_rms_corrplot.tif'), dpi=100)

                # Fig2: predicting glm vs. not predicting glm
                df_merged['GLM_Zero'] = np.where(
                    df_merged['sum_value_glm'] == 0,
                    'Not Predicting (n={})',
                    'Predicting (n={})'
                )

                counts = df_merged['GLM_Zero'].value_counts()

                def update_label(label):
                    if 'Not Predicting' in label:
                        count = counts.get(label, 0)
                        return label.format(count)
                    elif 'Predicting' in label:
                        count = counts.get(label, 0)
                        return label.format(count)
                    return label


                df_merged['GLM_Zero'] = df_merged['GLM_Zero'].apply(update_label)

                category_order = [label for label in df_merged['GLM_Zero'].unique() if 'Not Predicting' in label]
                category_order += [label for label in df_merged['GLM_Zero'].unique() if 'Predicting' in label]

                plt.figure(figsize=(9, 7))
                sns.set_theme(style="whitegrid")

                sns.violinplot(
                    data=df_merged,
                    x='GLM_Zero',
                    y='sum_value_hg',
                    order=category_order,
                    inner=None,
                    palette=["#ADD8E6", "#F08080"],
                    linewidth=1.5
                )

                sns.stripplot(
                    data=df_merged,
                    x='GLM_Zero',
                    y='sum_value_hg',
                    order=category_order,
                    color='gray',
                    size=6,
                    jitter=True,
                    alpha=0.7
                )

                plt.title('GLM sig clusters', fontsize=14)
                plt.xlabel('Predicting stim onsets by GLM', fontsize=12)
                plt.ylabel('Auditory responses HG rms of z-score', fontsize=12)
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.savefig(os.path.join('figs', f'{del_nodel_tag}_{elec_typ}_{pred_onset}_glm_rms_violin.tif'), dpi=100)

                group_zero = df_merged[df_merged['sum_value_glm'] == 0]['sum_value_hg']
                group_non_zero = df_merged[df_merged['sum_value_glm'] != 0]['sum_value_hg']
                t_statistic, p_value = ttest_ind(group_zero, group_non_zero, equal_var=True)
                print("--- Independent Samples T-Test Results ---")
                print(f"T-Statistic: {t_statistic:.4f}")
                print(f"df: {len(group_non_zero)+len(group_zero)-2}")
                print(f"P-Value: {p_value:.4f}")

        onset_pred_lags_onset[del_nodel_tag] = onset_pred_lags
        onset_pred_pows_onset[del_nodel_tag] = onset_pred_pows
        subj_elec_lists_onset[del_nodel_tag] = subj_elec_lists
        #%% compare predicting onset r^2 (ttests between each other)

        for onset_pred_data,onset_pred_tag,vert_boxplot in zip(
                (onset_pred_lags,onset_pred_pows),
                ('peak_lag','pred_pow'),
                (False,True)

        ):
            with open(os.path.join('figs', f'stats_{del_nodel_tag}_{pred_onset}_{onset_pred_tag}.txt'), 'w', encoding='utf-8') as f:
                with redirect_stdout(f):
                    # --- 2. Convert Dictionary to Long Format DataFrame ---
                    onset_pred_lags_stats=onset_pred_data.copy()
                    onset_pred_lags_stats.pop('Delay_only_vWM', None)
                    df_long_independent = pd.DataFrame({
                        'Lag_Value': [],
                        'Factor_Combined': []
                    })

                    for factor, values in onset_pred_lags_stats.items():
                        temp_df = pd.DataFrame({
                            'Lag_Value': values,
                            'Factor_Combined': factor
                        })
                        df_long_independent = pd.concat([df_long_independent, temp_df], ignore_index=True)

                    # --- 3. Separate Independent Factors (Electrode and Activity) ---
                    df_long_independent['Activity'] = df_long_independent['Factor_Combined'].apply(lambda x: x.split('_')[-1])
                    df_long_independent['Electrode'] = df_long_independent['Factor_Combined'].apply(
                        lambda x: '_'.join(x.split('_')[:-1]))

                    print("--- Long Format Data Head ---")
                    print(df_long_independent.head())
                    print("-" * 50)

                    # --- 4. Perform Two-Way Independent ANOVA ---
                    aov_results_independent = pg.anova(
                        data=df_long_independent,
                        dv='Lag_Value',
                        between=['Electrode', 'Activity']
                    )

                    print("--- Independent ANOVA Results (Main Effects & Interaction) ---")
                    pg.print_table(aov_results_independent)
                    print("-" * 50)

                    # --- 5. Post-Hoc Contrasts ---
                    # --- Simple Effects Analysis with FDR Correction ---

                    print("--- Simple Effect 1: Electrode within Activity (FDR corrected) ---")
                    posthoc_results = pd.DataFrame()
                    for activity_level, group_df in df_long_independent.groupby('Activity'):
                        # Run independent T-tests for all pairs of 'Electrode' within this 'Activity' level
                        ttests_result = pg.pairwise_tests(
                            data=group_df,
                            dv='Lag_Value',
                            between='Electrode',  # Compares Electrode levels
                            padjust='fdr'
                        )

                        ttests_result['Grouping_Factor'] = 'Activity'
                        ttests_result['Grouping_Level'] = activity_level
                        posthoc_results = pd.concat([posthoc_results, ttests_result], ignore_index=True)

                    pg.print_table(posthoc_results[posthoc_results['Grouping_Factor'] == 'Activity'])
                    print("-" * 50)

                    print("--- Simple Effect 2: Activity within Electrode (FDR corrected) ---")

                    for electrode_level, group_df in df_long_independent.groupby('Electrode'):
                        # Run independent T-tests for all pairs of 'Activity' within this 'Electrode' level
                        ttests_result = pg.pairwise_tests(
                            data=group_df,
                            dv='Lag_Value',
                            between='Activity',  # Compares Activity levels (only 2 levels, so it's a single T-test)
                            padjust='fdr'
                        )

                        ttests_result['Grouping_Factor'] = 'Electrode'
                        ttests_result['Grouping_Level'] = electrode_level
                        posthoc_results = pd.concat([posthoc_results, ttests_result], ignore_index=True)

                    pg.print_table(posthoc_results[posthoc_results['Grouping_Factor'] == 'Electrode'])
                    print("-" * 50)

            #%% boxplot: compare predicting onset r^2 (ttests between each other)
            MotorPrep_col = [1.0, 0.0784, 0.5765]  # Motor prepare
            Sensorimotor_col = [1, 0, 0]  # Sensorimotor
            Auditory_col = [0, 1, 0]  # Auditory
            Delay_col = [1, 0.65, 0]  # Delay
            Motor_col = [0, 0, 1]  # Motor

            ordered_keys=['Sensorymotor_vWM','Sensorymotor_novWM','Auditory_vWM','Auditory_novWM','Motor_vWM','Motor_novWM','Delay_only_vWM']
            new_tick_labels = ['SM_vWM', 'SM_novWM', 'Aud_vWM','Aud_novWM', 'M_vWM', 'M_novWM', 'DelOnly_vWM']
            onset_pred_lags_data = [onset_pred_data[key] for key in ordered_keys]
            colors = [Sensorimotor_col,Sensorimotor_col, Auditory_col, Auditory_col, Motor_col,Motor_col, Delay_col]

            fig, ax = plt.subplots(figsize=(10, 6))
            boxplot_parts = ax.boxplot(
                onset_pred_lags_data,
                tick_labels=new_tick_labels,
                vert=vert_boxplot,
                patch_artist=True,
                showmeans=True
            )

            for patch, color in zip(boxplot_parts['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor('black')
                patch.set_alpha(0.6)
            if onset_pred_tag=='peak_lag':
                ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
                ax.set_title('CrossCorr peak distribution')
                ax.set_xlabel('Lag aligned to speech onset (s)')
                ax.set_ylabel('Electrode group')
            elif onset_pred_tag=='pred_pow':
                ax.set_title('CrossCorr pre-onset power distribution')
                ax.set_ylabel('RMS power')
                ax.set_xlabel('Electrode group')
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join('figs', f'boxplot_{del_nodel_tag}_{pred_onset}_{onset_pred_tag}.tif'), dpi=100)

        #%% Traceplots: crosscorr
        results = {}

        for key, data_array in cross_corr_lsts.items():
            mean_curve = np.mean(data_array, axis=0)
            std_dev_curve = np.std(data_array, axis=0)
            N = data_array.shape[0]
            sem_curve = std_dev_curve / np.sqrt(N)
            results[key] = {
                'mean': mean_curve,
                'sem': sem_curve
            }

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = [Sensorimotor_col, Sensorimotor_col, Auditory_col,Auditory_col, Motor_col,Motor_col, Delay_col]
        ln_stls = ['-','--','-','--','-','--','-']
        color_idx = 0

        for key in ordered_keys:
            res = results[key]
            mean = res['mean']
            sem = res['sem']

            color = colors[color_idx]
            linestyle = ln_stls[color_idx]

            ax.plot(time_lags, mean, label=key, color=color, linewidth=2, linestyle=linestyle)

            ax.fill_between(
                time_lags,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.2
            )

            color_idx += 1

        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7, linewidth=1)

        ax.set_title('CrossCorr distribution')
        ax.set_xlabel('Lag aligned to speech onset (s)')
        ax.set_ylabel('CrossCorr Value (mean ± SEM)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc='best')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join('figs', f'cctimetraces_{del_nodel_tag}_{pred_onset}.tif'), dpi=100)

    onset_pred_lags_all[pred_onset] = onset_pred_lags_onset
    onset_pred_pows_all[pred_onset] = onset_pred_pows_onset
    subj_elecs_list_all[pred_onset] = subj_elec_lists_onset

#%% Correlation plots of delay vs. nodelay

def plot_crr_scat(x_data, y_data, pred_onset, elec_typ,data_tag):

    r_pearson, p_pearson = pearsonr(x_data, y_data)
    slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    plt.figure(figsize=(7, 7))
    plt.plot(x_fit, y_fit, color='red', linewidth=2)
    plt.scatter(x_data, y_data, alpha=0.6, edgecolors='w', linewidths=0.5)
    # min_val = min(x_data.min(), y_data.min())
    # max_val = max(x_data.max(), y_data.max())
    # plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    plt.title(
        f'{elec_typ} Correlation ({pred_onset})\n'
    )
    plt.xlabel(f'Nodel')
    plt.ylabel(f'Del')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join('figs', f'corr_{elec_typ}_{pred_onset}_{data_tag}.tif'), dpi=100)
    plt.close()

    return r_pearson, p_pearson

for pred_onset,del_nodel_tag_noDel,del_nodel_tag_Del in zip(pred_onsets,
                                                                   ('epoc_LexNoDelay_Cue','epoc_LexNoDelay_Cue'),
                                                                   ('epoc_LexDelay_Go', 'epoc_LexDelay_Cue')):
    for elec_typ in elec_typs:

        #!!!!!!!!!!!!!!!!! Just temporal adjustment, SHOULD CHECK why there are non-overlaps!
        nodel_subj_elecs_list = subj_elecs_list_all[pred_onset][del_nodel_tag_noDel][elec_typ]
        del_subj_elecs_list = subj_elecs_list_all[pred_onset][del_nodel_tag_Del][elec_typ]
        overlap_elecs = set(nodel_subj_elecs_list) & set(del_subj_elecs_list)

        nodel_pred_lags = onset_pred_lags_all[pred_onset][del_nodel_tag_noDel][elec_typ]
        del_pred_lags = onset_pred_lags_all[pred_onset][del_nodel_tag_Del][elec_typ]
        nodel_pred_lags_ovlp = []
        for elec, lag in zip(nodel_subj_elecs_list, nodel_pred_lags):
            if elec in overlap_elecs:
                nodel_pred_lags_ovlp.append(lag)
        del_pred_lags_ovlp = []
        for elec, lag in zip(del_subj_elecs_list, del_pred_lags):
            if elec in overlap_elecs:
                del_pred_lags_ovlp.append(lag)
        r,p=plot_crr_scat(np.array(del_pred_lags_ovlp), np.array(nodel_pred_lags_ovlp), pred_onset, elec_typ,'peak_lag')
        print(f'Peak lags: corr {elec_typ} {pred_onset}, r = {round(r,3)}, p = {round(p,3)}')

        nodel_pred_pows = onset_pred_pows_all[pred_onset][del_nodel_tag_noDel][elec_typ]
        del_pred_pows = onset_pred_pows_all[pred_onset][del_nodel_tag_Del][elec_typ]
        nodel_pred_pows_ovlp = []
        for elec, lag in zip(nodel_subj_elecs_list, nodel_pred_pows):
            if elec in overlap_elecs:
                nodel_pred_pows_ovlp.append(lag)
        del_pred_pows_ovlp = []
        for elec, lag in zip(del_subj_elecs_list, del_pred_pows):
            if elec in overlap_elecs:
                del_pred_pows_ovlp.append(lag)
        r,p=plot_crr_scat(np.array(del_pred_pows_ovlp), np.array(nodel_pred_pows_ovlp), pred_onset, elec_typ,'pred_pow')
        print(f'Peak pows: corr {elec_typ} {pred_onset}, r = {round(r,3)}, p = {round(p,3)}')
