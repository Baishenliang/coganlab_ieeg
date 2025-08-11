# Set dir
import os
import sys
import numpy as np
import pandas as pd
from ieeg.calc.stats import time_cluster

script_dir = os.path.dirname("D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\PCA_LDA\\lme_encode_timecluster.py")
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join("..", "GLM")))
import glm_utils as glm

# Setting parameters
alpha=0.05
alpha_clus=0.05

# Load data
aud_delay_org = pd.read_csv('Aud_delay_org.csv')
aud_delay_perm = pd.read_csv('Aud_delay_perm.csv')
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

