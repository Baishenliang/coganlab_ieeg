# Set dir
import os
import sys
import numpy as np
import pandas as pd
from ieeg.calc.stats import time_cluster
import matplotlib.pyplot as plt

script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\lme\\prepare_raw.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join("..", "GLM")))
import glm_utils as glm

#%% Run time cluster
def get_traces_clus(raw_filename, alpha:float=0.05, alpha_clus:float=0.05):
    # Load data
    # aud_delay_org = pd.read_csv('Aud_delay_org.csv')
    # aud_delay_perm = pd.read_csv('Aud_delay_perm.csv')
    raw = pd.read_csv(raw_filename)
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

    # Time perm cluster
    mask_time_clus = time_cluster(mask_i_org, mask_null_i,1 - alpha_clus)

    return time_point,r2_i[0],mask_time_clus

#%% Plotting
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for pho_tag,pho_pos in zip(('Consonant','Vowel'),([1,3,5],[2,4])):
    for j in range(0,2):
        fig, ax = plt.subplots(figsize=(12, 4))
        for i in pho_pos:
            filename = f"results/Auditory_delay_full_pho{i}_pho{j}aln.csv"
            time_point, time_series, mask_time_clus = get_traces_clus(filename, 0.05, 0.05)
            ax.plot(time_point, time_series, label=f'pho{i}', color=colors[i-1], linewidth=2)
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

                    label = f'clust{k} of pho{i}'
                    ax.axvspan(start_time, end_time, color=colors[i-1], alpha=0.4, label=label)
        if j>0:
            ax.set_title(f"{pho_tag} encoding in Auditory Delay electrodes aligned to pho{j} onset", fontsize=16)
        else:
            ax.set_title(f"{pho_tag} encoding in Auditory Delay electrodes aligned to stim onset", fontsize=16)
        ax.set_xlabel("Time (seconds) aligned to stim onset", fontsize=12)
        ax.set_ylabel("χ² to baseline model", fontsize=12)
        ax.legend()
        ax.set_xlim(time_point.min(), time_point.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join('figs', f'{pho_tag} Auditory_delay_full_pho{j}aln.tif'), dpi=300)
        plt.close()
