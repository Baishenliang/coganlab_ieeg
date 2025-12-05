#%% Set dir
import os
import re
import sys
import numpy as np
import pandas as pd
from ieeg.calc.stats import time_cluster
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from partd.utils import suffix
from requests.packages import target
from statsmodels.stats.multitest import multipletests
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\lme\\prepare_raw.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join("..", "GLM")))
import glm_utils as glm
from scipy.ndimage import gaussian_filter1d,uniform_filter1d
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp

#%% Run time cluster

font_scale=2
plt.rcParams['font.size'] = 14*font_scale
plt.rcParams['axes.titlesize'] = 16*font_scale
plt.rcParams['axes.labelsize'] = 12*font_scale
plt.rcParams['xtick.labelsize'] = 12*font_scale
plt.rcParams['ytick.labelsize'] = 12*font_scale
plt.rcParams['legend.fontsize'] = 12*font_scale

MotorPrep_col = [1.0, 0.0784, 0.5765] # Motor prepare
Sensorimotor_col = [1, 0, 0]  # Sensorimotor
Auditory_col = [0, 1, 0]  # Auditory
Delay_col = [1, 0.65, 0]  # Delay
Motor_col = [0, 0, 1]  # Motor

# Feature colors
aco_col = [0, 0.502, 0.502]      # Teal (青色)
pho_col = [0.502, 0, 0.502]      # Purple (紫色)
wordness_col = [1, 0, 1] # Magenta (洋红色)

def expand_sequence(prefix: str, suffix: str, max_number: int) -> tuple:
    expanded_list = [f'{prefix}{i}{suffix}' for i in range(1, max_number + 1)]
    return tuple(expanded_list)

def get_traces_clus(raw, alpha:float=0.05, alpha_clus:float=0.05,mode:str='time_cluster',target_fea:str=None,input:str='R2'):
    # Load data
    # aud_delay_org = pd.read_csv('Aud_delay_org.csv')
    # aud_delay_perm = pd.read_csv('Aud_delay_perm.csv')
    # raw=bsl_correct(raw)
    time_point = np.unique(raw['time_point'].to_numpy())
    if target_fea is None:
        r2s_i_df = raw.pivot_table(
            index='perm',
            columns='time_point',
            values='chi_squared_obs'
        )
    elif isinstance(target_fea, str):
        raw_temp = raw.copy()
        raw_temp['abs_target_fea'] = raw_temp[target_fea]#.abs()

        r2s_i_df = raw_temp.pivot_table(
            index='perm',
            columns='time_point',
            values='abs_target_fea'
        )
    elif isinstance(target_fea, list):
        abs_features = raw[target_fea]#.abs()
        mean_abs_feature = abs_features.mean(axis=1)

        raw_temp = raw.copy()
        raw_temp['mean_abs_features'] = mean_abs_feature

        r2s_i_df = raw_temp.pivot_table(
            index='perm',
            columns='time_point',
            values='mean_abs_features'
        )
    r2s_i = r2s_i_df.to_numpy()

    # r2_i, 1-d time series of original glm values
    # null_r2_i, 2-d time series of original glm values: n_perm*time
    r2_i=r2s_i[0,:]
    r2_i = np.expand_dims(r2_i, axis=0)
    if input=='R2':
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
    elif input=='p':
        stat_out, p_fdr, _, _ = multipletests(r2_i[0], alpha=alpha_clus, method='fdr_bh')

        return time_point,r2_i[0],stat_out

def add_alignment_vlines(ax, alignment):
    """
    Adds vertical lines for specific event markers to a matplotlib Axes object
    based on the alignment type.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw the lines on.
        alignment (str): The alignment type ('Aud', 'Go', or 'Resp').
    """
    if alignment == 'Aud':
        # Stim offsets
        ax.axvline(x=0.59, color='red', linestyle='--', alpha=0.7, linewidth=3)
        # ax.axvline(x=0.59 + 0.1480 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=0.59 - 0.1480 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # Delay offsets
        ax.axvline(x=1.7201, color='red', linestyle='--', alpha=0.7, linewidth=3)
        # ax.axvline(x=1.7201 + 0.1459 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=1.7201 - 0.1459 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
    elif alignment == 'Go':
        # Motor onsets
        ax.axvline(x=0.7961, color='red', linestyle='--', alpha=0.7, linewidth=3)
        # ax.axvline(x=0.7961 + 1.0532 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=0.7961 - 1.0532 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
    elif alignment == 'Resp':
        # Motor offsets
        ax.axvline(x=0.6096, color='red', linestyle='--', alpha=0.7, linewidth=3)
        ax.axvline(x=0.6096 + 0.2190, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0.6096 - 0.2190, color='red', linestyle='--', alpha=0.7, linewidth=1)
#%% Plotting
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
is_normalize=False
is_bsl_correct=False
mode='time_cluster'
#for elec_grp in ['Hickok_Spt','Hickok_lPMC','Hickok_lIFG']:
# for elec_grp in ['Auditory_delay','Sensorymotor_delay','Motor_delay','Delay_only']:
baseline=dict()
baseline_beta_rms=dict()
baseline_std=dict()
baseline_beta_rms_std=dict()
test_lambdas=[0.1, 1, 10, 20, 40, 60, 80,100, 1000]
for vWM, vwm_linestyle in zip(('vWM', 'novWM'), ('-', '--')):
    for alignment,xlim_align in zip(
            ('Aud','Resp','Go'),
            ([-0.2, 1.75],[-0.2, 1.25],[-0.2, 1.25])):
        for elec_grp,elec_col,vWM_lambda,novWM_lambda,fea_plot_yscale in zip(('Auditory','Sensorymotor','Motor','Delay_only'),
                                                            (Auditory_col,Sensorimotor_col,Motor_col,Delay_col),
                                                            (10, 20, 10, 5),
                                                            (10, 10, 50, 10),
                                                            (3.5,1.6,1.3,1.3)):
            fea = 'ACC'
            fea_tag = 'ACC'
            para_sig_barbar = [0.2, 0.01]

            fig, ax = plt.subplots(figsize=(5.6*(xlim_align[1]-xlim_align[0]), 5))
            j=0
            ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7,linewidth=3)
            add_alignment_vlines(ax, alignment)
            fea_cols=gp.create_gradient(elec_col, len(test_lambdas)+1)[:-1]
            for i,test_lambda in enumerate(test_lambdas):

                filename = f"results/LexDelayRep_{elec_grp}_{alignment}_All_testλ_{test_lambda}.csv"
                raw = pd.read_csv(filename)

                input_r2 = 'p'
                pthres_r2 = [1e-2, 1e-2]
                vwm_text_r2 = 'ACC'
                target_fea_r2 = f'{fea}_{vWM}'
                time_point, time_series_r2, *_ = get_traces_clus(raw, pthres_r2[0], pthres_r2[1], mode=mode, target_fea=target_fea_r2, input=input_r2)
                target_fea_p = f'{fea}_{vWM}_p'
                _, _, mask_time_clus = get_traces_clus(raw, pthres_r2[0], pthres_r2[1], mode=mode, target_fea=target_fea_p, input=input_r2)

                time_series_r2 = gaussian_filter1d(time_series_r2, sigma=2, mode='nearest')

                if alignment == 'Aud':
                    baseline[elec_grp] = np.min(time_series_r2[(time_point > -0.2) & (time_point <= 0)])
                    baseline_std[elec_grp] = np.std(time_series_r2[(time_point > -0.2) & (time_point <= 0)])
                
                if is_normalize:
                    time_series_r2 = (time_series_r2 - baseline[elec_grp]) / baseline_std[elec_grp]
                    time_series_r2 = time_series_r2 / (np.max(time_series_r2[(time_point > xlim_align[0]) & (time_point <= xlim_align[1])]) - np.min((time_point > xlim_align[0]) & (time_point <= xlim_align[1])))
                    para_sig_bar = [1, 1e-1]
                else:
                    if is_bsl_correct:
                        time_series_r2 = (time_series_r2 - baseline[elec_grp])
                    para_sig_bar = para_sig_barbar
                
                ax.plot(time_point, time_series_r2, label=f"{elec_grp}{vwm_text_r2}", color=fea_cols[i], linewidth=5, linestyle=vwm_linestyle)

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
                        ax.plot([start_time, end_time], [para_sig_bar[0]-para_sig_bar[1]*(j-1),para_sig_bar[0]-para_sig_bar[1]*(j-1)],
                                color=fea_cols[i],alpha=0.4,
                                linewidth=10,  # Make the line thick like a bar
                                solid_capstyle='butt')  # Makes the line ends flat
                    j=j+1
                    
            ax.tick_params(axis='both', which='major')#, labelsize=16)
            ax.ticklabel_format(
                axis='y',
                style='sci',
                scilimits=(-3, -3),  # 将指数固定为 10^-3
                useMathText=True  # 使用 LaTeX 格式显示指数，如 10⁻³
            )
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.legend(fontsize=18)
            ax.set_xlim(xlim_align)#time_point.max())
            # if elec_grp=='Motor' and alignment=='Aud':
            #     ax.set_xlim([0.5,xlim_align[1]])
            ax.legend().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(-0.002,para_sig_bar[0]+2*para_sig_bar[1])
            plt.tight_layout()
            plt.savefig(os.path.join('figs','multencode', f'{elec_grp}_{fea_tag}_{alignment}_{vWM}_testλ.tif'), dpi=300)
            plt.close()