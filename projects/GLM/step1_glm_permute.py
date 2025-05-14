import argparse

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

def main(event, task_Tag, glm_fea, wordness):

    #%% Read data
    with open('glm_config.json', 'r') as f:
        config = json.load(f)

    # Extract parameters from config
    if glm_fea=='BSL_correct':
        model = glm_fea
        stat = 'power'
    elif glm_fea=='Full':
        model = 'full'
        stat = 'power'
    else:
        model = config['model']
        stat = config['stat'] # zscore, power
    alpha = config['alpha']
    n_perms = config['n_perms']
    f_ranges = config['feature_ranges'][glm_fea]
    f_ranges_ctr = config['control_feature_ranges'][glm_fea]

    if model=='partial' and glm_fea!='Acoustic':
        feature_controlled = np.r_[f_ranges_ctr[0]:f_ranges_ctr[1]]
    else:
        feature_controlled = []

    if len(f_ranges) == 1:
        feature_seleted = np.r_[0,f_ranges[0]]
    else:
        feature_seleted = np.r_[0,f_ranges[0]:f_ranges[1]]

    if glm_fea=='BSL_correct':
        subjs, data_list_raw, filtered_events_list, chs, times = glm.fifread(event,stat,task_Tag,wordness, bsl_contrast=True)
    else:
        subjs, data_list_raw, filtered_events_list, chs, times = glm.fifread(event,stat,task_Tag,wordness)

    #%% Smooth data
    data_list=[]
    for i, data_i_raw in enumerate(data_list_raw):
        data_i=glm.temporal_smoothing(data_i_raw, window_size=5) #smoothing window: 1=10ms
        data_list.append(data_i)
    #%% Generate null distributions for each patient
    for i, data_i in enumerate(data_list):
        # feature_mat_i: feature matrix, observations * channels * features
        # data_i: eeg data matrix, observations * channels * times
        print(f"Generate null distribution Patient {subjs[i]} in {event} {task_Tag} {wordness} {glm_fea}")
        if model=='simple' or (model=='partial' and glm_fea=='Acoustic') or model=='BSL_correct':
            feature_mat_i=filtered_events_list[i][:,:,feature_seleted]
            null_r2 = glm.permutation_baishen_parallel(feature_mat_i, data_i, n_perms,np.r_[0:np.shape(feature_mat_i)[2]])
        elif model=='partial':
            # Get the residuals of data and seleted features controlling out unseleted features
            feature_mat_i_res, data_i_res = glm.par_regress(filtered_events_list[i], feature_seleted,feature_controlled, data_i)
            # Get null distribution
            null_r2 = glm.permutation_baishen_parallel(feature_mat_i_res, data_i_res, n_perms,
                                                       np.r_[0:np.shape(feature_mat_i_res)[2]])
        elif model=='full':
            null_r2 = glm.permutation_baishen_parallel(filtered_events_list[i], data_i, n_perms,feature_seleted)
    # save the null distribution
        np.save(os.path.join('data',f'null_r2 {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy'), null_r2)
        del null_r2

    #%% Generate an original significant matrix (channels*times) (GLM_original vs. GLM_null)
    for i, data_i in enumerate(data_list):
        # feature_mat_i: feature matrix, observations * channels * features
        # data_i: eeg data matrix, observations * channels * times
        # r2_i: r2 matrix, channels * features * times
        print(f"Getting uncorrected significance: Patient {subjs[i]} in {event} {task_Tag} {wordness} {glm_fea}")
        if model=='simple' or (model=='partial' and glm_fea=='Acoustic') or model=='BSL_correct':
            feature_mat_i=filtered_events_list[i][:,:,feature_seleted]
            r2_i,_ = glm.compute_r2_loop(feature_mat_i, np.r_[0:np.shape(feature_mat_i)[2]],data_i)
        elif model == 'partial':
            # Get the residuals of data and seleted features controlling out unseleted features
            feature_mat_i_res, data_i_res = glm.par_regress(filtered_events_list[i], feature_seleted,
                                                            feature_controlled, data_i)
            r2_i, _ = glm.compute_r2_loop(feature_mat_i_res, np.r_[0:np.shape(feature_mat_i_res)[2]], data_i_res)
        elif model=='full':
            r2_i, _ = glm.compute_r2_loop(filtered_events_list[i], feature_seleted,data_i)
        np.save(f'data\\org_r2 {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy', r2_i)
        r2_i = np.expand_dims(r2_i, axis=0)
        null_r2_i = np.load(f"data\\null_r2 {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy")
        r2s_i = np.concatenate([r2_i, null_r2_i], axis=0)
        del r2_i, null_r2_i
        # get significance of the original r2 against the permutation distribution
        # return: mask_left_i_org channels*times
        org_p_i = glm.aaron_perm_gt_1d(r2s_i, axis=0)[0]
        np.save(f'data\\org_p {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy', org_p_i)
        mask_i_org = (org_p_i > (1 - alpha)).astype(int)
        np.save(f'data\\org_mask {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy', mask_i_org)
        del org_p_i, mask_i_org

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process experiment parameters.")
    parser.add_argument("--event", type=str, required=True, help="Event type")
    parser.add_argument("--task_Tag", type=str, required=True, help="Task tag")
    parser.add_argument("--glm_fea", type=str, required=True, help="GLM feature")
    parser.add_argument("--wordness", type=str, required=True, help="Wordness category")

    args = parser.parse_args()
    main(args.event, args.task_Tag, args.glm_fea, args.wordness)