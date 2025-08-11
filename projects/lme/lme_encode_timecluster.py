# Set dir
import os
import sys
import numpy as np
import pandas as pd
from ieeg.calc.stats import time_cluster
import matplotlib.pyplot as plt

script_dir = os.path.dirname("/projects/lme/lme_encode_timecluster.py")
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join("..", "GLM")))
import glm_utils as glm

# Setting parameters
alpha=0.05
alpha_clus=0.05

#%% Run time cluster
# Load data
# aud_delay_org = pd.read_csv('Aud_delay_org.csv')
# aud_delay_perm = pd.read_csv('Aud_delay_perm.csv')
aud_delay_org = pd.read_csv('../PCA_LDA/Aud_delay_org_wordness.csv')
aud_delay_perm = pd.read_csv('../PCA_LDA/Aud_delay_perm_wordness.csv')

time_point = aud_delay_org['time_point'].to_numpy()
r2_i = aud_delay_org['chi_squared_comp'].to_numpy()
null_r2_i_df = aud_delay_perm.pivot_table(
    index='perm',
    columns='time_point',
    values='chi_squared_obs'
)
null_r2_i = null_r2_i_df.to_numpy()

# r2_i, 1-d time series of original glm values
# null_r2_i, 2-d time series of original glm values: n_perm*time

# Get original mask
r2_i = np.expand_dims(r2_i, axis=0)
r2s_i = np.concatenate([r2_i, null_r2_i], axis=0)
org_p_i = glm.aaron_perm_gt_1d(r2s_i, axis=0)[0] # 1-d time series
mask_i_org = (org_p_i > (1 - alpha)).astype(int) # 1-d time series (binary)

# Get null mask
null_p_i = glm.aaron_perm_gt_1d(null_r2_i, axis=0) # 2-d time series: n_perm*time
mask_null_i=(null_p_i>(1-alpha)).astype(int) # # 2-d time series: n_perm*time (binary)

# Time perm cluster
mask_time_clus = time_cluster(mask_i_org, mask_null_i,1 - alpha_clus)

#%% Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_point, r2_i[0], label='χ² value', color='royalblue', linewidth=2)
true_indices = np.where(mask_time_clus)[0]

if true_indices.size > 0:
    split_points = np.where(np.diff(true_indices) != 1)[0] + 1
    clusters_indices = np.split(true_indices, split_points)

    for i, cluster in enumerate(clusters_indices):
        start_index = cluster[0]
        end_index = cluster[-1]

        time_step = time_point[1] - time_point[0]
        start_time = time_point[start_index] - time_step / 2
        end_time = time_point[end_index] + time_step / 2

        label = 'Sig. Cluster' if i == 0 else ""
        ax.axvspan(start_time, end_time, color='gold', alpha=0.4, label=label)

ax.set_title("Encoding of lexical status in Auditory Delay electrodes", fontsize=16)
ax.set_xlabel("Time (seconds) aligned to stim onset", fontsize=12)
ax.set_ylabel("χ² to baseline model", fontsize=12)
ax.legend()
ax.set_xlim(time_point.min(), time_point.max())
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join('../PCA_LDA/results', f'Auditory_Delay_multiencode_wordness.tif'), dpi=300)
plt.close()
