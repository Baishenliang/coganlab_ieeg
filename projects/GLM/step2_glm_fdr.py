import os
# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
# script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
# current_dir = os.getcwd()
# if current_dir != script_dir:
#     os.chdir(script_dir)

import argparse
import numpy as np
import json
import glm_utils as glm
from statsmodels.stats.multitest import fdrcorrection

# Set parameters

def main(event, task_Tag, glm_fea, wordness):

    #%% Read data
    with open('glm_config.json', 'r') as f:
        config = json.load(f)

    # Extract parameters from config
    alpha = config['alpha']
    alpha_clus = config['alpha_clus']
    stat = config['stat']

    subjs, _, _, chs, _ = glm.fifread(event,stat,task_Tag,wordness)

    for i, _ in enumerate(subjs):
        print(f"Time cluster: Patient {subjs[i]} in {event} {task_Tag} {wordness} {glm_fea}")

        # %% Generate null significant matrix (nperms*channels*times) (GLM_one_perm_from_null vs. GLM_null)
        # shape: (n_perms, channels, times)
        p_i_org = np.load(f'data\\org_p {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy')
        p_time_clus = np.full([p_i_org.shape[0], p_i_org.shape[1]], np.nan)
        for chs in range(p_i_org.shape[0]):
            ts=p_i_org[chs, :]
            rs, _ = fdrcorrection(ts, alpha=0.05)
            p_time_clus[chs, :] = rs
        np.save(f"data\\fdr_mask {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy", p_time_clus)
        del p_time_clus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument("--event", type=str, required=True, help="Event type")
    parser.add_argument("--task_Tag", type=str, required=True, help="Task tag")
    parser.add_argument("--glm_fea", type=str, required=True, help="GLM feature")
    parser.add_argument("--wordness", type=str, required=True, help="Wordness category")

    args = parser.parse_args()
    main(args.event, args.task_Tag, args.glm_fea, args.wordness)