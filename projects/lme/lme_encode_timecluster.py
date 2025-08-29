# Set dir
import os
import sys
import numpy as np
import pandas as pd
from ieeg.calc.stats import time_cluster
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from pyparsing import alphas

script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\lme\\prepare_raw.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join("..", "GLM")))
import glm_utils as glm
from scipy.ndimage import gaussian_filter1d,uniform_filter1d

#%% Run time cluster


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



#%% Plotting
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
is_normalize=False
mode='time_cluster'
for elec_grp in ['Auditory_delay','Sensorymotor_delay','Motor_delay','Delay_only']:

    for fea,fea_tag in zip(('Wordvec','wordness','aco','pho'),
                                 ('Embedding','Lexical status','Acoustic','Phonemic')):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7)
        i = 0
        if fea=='aco' or fea=='pho':
            wordnesses=('All','Word','Nonword','Nonword-Word')
        elif fea=='Frq' or fea=='Uni_Pos_SC':
            wordnesses=('All','Word','Nonword','Word-Nonword')
        else:
            wordnesses=('All','Word','Nonword','Word-Nonword')
        for wordness in wordnesses:
            if fea=='Frq' and wordness!='Word':
                continue
            if fea=='wordness' and wordness!='All':
                continue
            if wordness =='All' or wordness =='Word' or wordness =='Nonword':
                filename = f"results/{elec_grp}_full_{fea}_{wordness}.csv"
                raw = pd.read_csv(filename)
            else:
                filename_Word = f"results/{elec_grp}_full_{fea}_Word.csv"
                filename_Nonword = f"results/{elec_grp}_full_{fea}_Nonword.csv"
                raw_word = pd.read_csv(filename_Word)
                raw_nonword = pd.read_csv(filename_Nonword)
                if wordness=='Nonword-Word':
                    chi_squared_diff = raw_nonword['chi_squared_obs'] - raw_word['chi_squared_obs']
                elif wordness=='Word-Nonword':
                    chi_squared_diff = raw_word['chi_squared_obs'] - raw_nonword['chi_squared_obs']
                # Create the new 'raw' DataFrame
                raw = raw_word[['perm', 'time_point']].copy()
                raw['chi_squared_obs'] = chi_squared_diff
            time_point, time_series, mask_time_clus = get_traces_clus(raw, 1/1e2, 1/1e2,mode=mode)
            time_series=gaussian_filter1d(time_series, sigma=1, mode='nearest')
            # win_len=10
            # time_series=uniform_filter1d(time_series, size=win_len, axis=0, mode='nearest',origin=(win_len - 1) // 2)
            if is_normalize:
                time_series = (time_series - np.min(time_series[(time_point > -0.2) & (time_point <= 0)]))# / (np.max(time_series) - np.min(time_series))
                # time_series = (time_series - np.mean(time_series[time_point<=0])) / (np.max(time_series) - np.min(time_series[time_point<=0]))
                para_sig_bar = [1,1e-1]
            else:
                time_series = time_series - np.mean(time_series[(time_point > -0.2) & (time_point <= 0)])
                para_sig_bar = [12,1]

            if wordness == 'All' or wordness == 'Word' or wordness == 'Nonword':
                ax.plot(time_point, time_series, label=f"{fea_tag} {wordness}", color=colors[i-1], linewidth=2)
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
                    ax.plot([start_time, end_time], [para_sig_bar[0]-para_sig_bar[1]*(i-1),para_sig_bar[0]-para_sig_bar[1]*(i-1)],
                            color=colors[i - 1],alpha=0.4,
                            linewidth=4,  # Make the line thick like a bar
                            solid_capstyle='butt')  # Makes the line ends flat
            i+=1
        ax.set_title(f"{elec_grp} electrodes aligned to stim onset (Unique)", fontsize=16)
        ax.set_xlabel("Time (seconds) aligned to stim onset", fontsize=12)
        ax.set_ylabel("$X^2$ (Baseline corrected)", fontsize=12)
        ax.legend()
        ax.set_xlim(-0.2, time_point.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join('figs', f'{elec_grp}_{fea_tag}.tif'), dpi=300)
        plt.close()
