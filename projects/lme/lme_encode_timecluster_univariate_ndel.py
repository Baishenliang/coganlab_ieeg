# Set dir
import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle
from ieeg.calc.stats import time_cluster
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from ieeg.arrays.label import LabeledArray

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

def get_subj_elec(filename,pred_onset,aud_onset):
    prefix = f'results\\NoDel_{elec_typ}_'
    suffix = f'_full_{aud_onset}.csv'
    remaining_string = filename.split(prefix)[-1]
    extracted_part = remaining_string.split(suffix)[0]
    return extracted_part

#%% Get data
#elec_typs=('Auditory_delay','Sensorymotor_delay','Motor_delay','Delay_only')
elec_typs=('Auditory_NoDelay',)

#%% Plotting
pred_onset='aud_onset'
plot_elec_pic=False
for elec_typ in elec_typs:
    # read raw hg data
    r2_lst=[]
    clus_lst=[]
    subj_elec_lst=[]
    subj_elec_onsets=[]
    hg_filename=f'data\\epoc_LexNoDelay_Cue_full_{elec_typ}_long.csv'
    hg_raw = pd.read_csv(hg_filename)
    # read data
    file_pattern = f"results/NoDel_{elec_typ}_*_full_{pred_onset}.csv"
    matching_files = glob.glob(file_pattern, recursive=False)
    ymax=2e-1
    for filename in matching_files[0:50]:
        para_sig_bar=[1e-1,0]
        if plot_elec_pic:
            fig, ax = plt.subplots(figsize=(19, 6))
            ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7,linewidth=3)
        raw = pd.read_csv(filename)
        subj_elec=get_subj_elec(filename,elec_typ,pred_onset)
        print(subj_elec)

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

        r2_lst.append(time_series)
        clus_lst.append(mask_time_clus.astype(int))
        subj_elec_lst.append(subj_elec)
        subj_elec_onsets.append(float(np.min(resp_onset_data_subj_elec)))

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
                    ax.set_xlim(-0.2, time_point.max())
                    ax.set_ylim(-2e-2,ymax)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    plt.savefig(os.path.join('figs', f'NoDel_{elec_typ}_{subj_elec}_full_{pred_onset}.tif'), dpi=100)
                    plt.close()

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
                                                  mask_data=True,
                                                  select_electrodes=False)

    gp.plot_chs(epoc_del, os.path.join('figs', f'Del_{elec_typ}_{pred_onset}.tif'),
             f'{pred_onset}', percentage_vscale=False, vmin=0, vmax=3, is_colbar=False,
             fig_size=[4, 20 * (np.shape(epoc_del)[0] / 250)])

    # Plot HG activity for Delay
    epoc_nodel, *_ = gp.sort_chs_by_actonset(Mask_NoDel_Aud,
                                                  Epoc_NoDel_Aud,
                                                  0.011, [-0.2, 5],
                                                  sorted_indices=sorted_indices,
                                                  mask_data=True,
                                                  select_electrodes=False)

    gp.plot_chs(epoc_nodel, os.path.join('figs', f'NoDel_{elec_typ}_{pred_onset}.tif'),
             f'{pred_onset}', percentage_vscale=False, vmin=0, vmax=3, is_colbar=False,
             fig_size=[3, 20 * (np.shape(epoc_del)[0] / 250)])

    # Plot r2
    epoch_sort_mask, *_ = gp.sort_chs_by_actonset(clus_arr,
                                                  r2_arr,
                                                  0.011, [-0.2, 5],
                                                  sorted_indices=sorted_indices,
                                                  mask_data=True,
                                                  select_electrodes=False)
    subj_elec_onsets_array = np.array(subj_elec_onsets)
    subj_elec_onsets_array = subj_elec_onsets_array[sorted_indices.tolist()]
    gp.plot_chs(epoch_sort_mask, os.path.join('figs', f'NoDel_pred_{elec_typ}_{pred_onset}.tif'),
             f'{pred_onset}', percentage_vscale=False, vmin=0, vmax=1e-1, is_colbar=False,
             fig_size=[6, 20 * (np.shape(r2_lst)[0] / 250)],scatter_onsets=subj_elec_onsets_array.tolist())

    # Get windowed paras
    _, _, _, _, _, paras, *_ = gp.sort_chs_by_actonset(clus_arr,
                                                  r2_arr,
                                                  0.011, [0, 0.2],
                                                  sorted_indices=sorted_indices,
                                                  mask_data=True,
                                                  select_electrodes=False,scatter_onsets=subj_elec_onsets_array.tolist())
