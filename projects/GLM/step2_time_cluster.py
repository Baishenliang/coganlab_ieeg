import os
# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
# script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
# current_dir = os.getcwd()
# if current_dir != script_dir:
#     os.chdir(script_dir)

import numpy as np
import json
import glm_utils as glm
from ieeg.calc.stats import time_cluster

# Set parameters


#%% Read data
with open('glm_config.json', 'r') as f:
    config = json.load(f)

# Extract parameters from config
alpha = config['alpha']
alpha_clus = config['alpha_clus']
n_perms = config['n_perms']
event = config['event']
stat = config['stat']
task_Tag = config['task_Tag']
wordness = config['wordness']
glm_fea = config['glm_fea']
f_ranges = config['feature_ranges'][glm_fea]

if len(f_ranges) == 1:
    feature_seleted = np.r_[0,f_ranges[0]]
else:
    feature_seleted = np.r_[0, f_ranges[0]:f_ranges[1]]

subjs, _, _, chs, _ = glm.fifread(event,stat,task_Tag)

for i, _ in enumerate(subjs):
    print(f"Time cluster: Patient {subjs[i]}")

    # %% Generate null significant matrix (nperms*channels*times) (GLM_one_perm_from_null vs. GLM_null)
    # shape: (n_perms, channels, times)
    null_r2_i = np.load(f"data\\null_r2 {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy")
    # Get the significancy for each permutation that is with **Larger** r2s than null distribution
    mask_null_i=(glm.aaron_perm_gt_1d(null_r2_i, axis=0)>(1-alpha)).astype(int)
    del null_r2_i

    #%% Run time-cluster correction
    # feature_mat_i: feature matrix, observations * channels * features
    # data_i: eeg data matrix, observations * channels * times
    # r2_i: r2 matrix, channels * features * times
    # load original masks and null masks
    mask_i_org = np.load(f'data\\org_mask {subjs[i]} {event} {task_Tag} {glm_fea}.npy')
    # run time cluster correction
    # The output p_time_clus tells how **LARGER** the clusters in the real mask are as compared with the clusters in the permuted masks
    # The larger the values, the higher the probabilities are
    mask_time_clus = np.full([mask_i_org.shape[0], mask_i_org.shape[1]], np.nan)
    for chs in range(mask_i_org.shape[0]):
        mask_time_clus[chs, :] = time_cluster(mask_i_org[chs, :], mask_null_i[:, chs, :], 1 - alpha_clus)
    del mask_i_org, mask_null_i
    np.save(f"data\\cluster_mask {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy", mask_time_clus)
    del mask_time_clus
    os.remove(f"data\\null_r2 {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy")

