import argparse

import os
# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

import numpy as np
import json
import glm_utils as glm

def main(event, task_Tag, glm_fea, wordness,glm_out):

    # # %% For testing:
    # event="Auditory_inRep"
    # task_Tag="Repeat"
    # glm_fea="Phonemic"
    # wordness="ALL"
    # glm_out="beta"

    #%% Read data
    with open('glm_config.json', 'r') as f:
        config = json.load(f)

    # Extract parameters from config
    if glm_fea=='BSL_correct':
        model = glm_fea
        stat = 'power'
    else:
        model = config['model']
        stat = config['stat'] # zscore, power
    alpha = config['alpha']
    n_perms = config['n_perms']
    f_ranges = config['feature_ranges'][glm_fea]
    f_ranges_ctr = config['control_feature_ranges'][glm_fea]
    query_ridge_aloha = False

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

    #%% Smooth data and remove outliers (using 3IQR)
    data_list=[]
    alpha_list=[]
    cv_r_list=[]
    for i, data_i_raw in enumerate(data_list_raw):
        # data_i_raw: eeg data matrix, observations * channels * times
        print(f'Processing {subjs[i]}')
        data_i_outrm=glm.remove_and_impute_outliers_3d(data_i_raw)
        data_i=glm.temporal_smoothing(data_i_raw, window_size=5) #smoothing window: 1=10ms
        data_list.append(data_i)

    #%% Get the electrode wise alpha-r^2 plots
    if query_ridge_aloha:
        for i, data_i in enumerate(data_list):
            # feature_mat_i: feature matrix, observations * channels * features
            # data_i: eeg data matrix, observations * channels * times
            print(f"Getting CV r^2 for Ridge alpha Patient {subjs[i]} in {event} {task_Tag} {wordness} {glm_fea}")
            if model=='simple' or (model=='partial' and glm_fea=='Acoustic') or model=='BSL_correct':
                feature_mat_i=filtered_events_list[i][:,:,feature_seleted]
                cv_r_i, _ = glm.compute_r2_loop(feature_mat_i, np.r_[0:np.shape(feature_mat_i)[2]], data_i, 'cv_r2',np.nan)
                cv_r_list.append(cv_r_i)
            # partial暂时未开启CV功能
            # elif model=='partial':
            #     # Get the residuals of data and seleted features controlling out unseleted features
            #     feature_mat_i_res, data_i_res = glm.par_regress(filtered_events_list[i], feature_seleted,feature_controlled, data_i,glm_out,alpha_i)
            #     # Get null distribution
            #     cv_r_i, _ = glm.compute_r2_loop(feature_mat_i_res, np.r_[0:np.shape(feature_mat_i_res)[2]], data_i_res,'cv_r2',np.nan)
            #     cv_r_list.append(cv_r_i)
            elif model=='full':
                cv_r_i, _ = glm.compute_r2_loop(filtered_events_list[i], feature_seleted, data_i, 'cv_r2',np.nan)
                cv_r_list.append(cv_r_i)
        cv_r_list=np.concatenate(cv_r_list)
        # np.save(os.path.join('data',f'CV_mse {event} {task_Tag} {wordness} {glm_fea}.npy'), cv_r_list)
        glm.plot_mean_se_for_alphas(cv_r_list, np.logspace(-6, 6, 10),f'CV_mse {event} {task_Tag} {wordness} {glm_fea}.jpg')
    #%% Generate null distributions for each patient
    for i, data_i in enumerate(data_list):
        # feature_mat_i: feature matrix, observations * channels * features
        # data_i: eeg data matrix, observations * channels * times
        print(f"Generate null distribution Patient {subjs[i]} in {event} {task_Tag} {wordness} {glm_fea}")
        if model=='simple' or (model=='partial' and glm_fea=='Acoustic') or model=='BSL_correct':
            feature_mat_i=filtered_events_list[i][:,:,feature_seleted]
            alpha_i, _ = glm.compute_r2_loop(feature_mat_i, np.r_[0:np.shape(feature_mat_i)[2]], data_i, 'alpha',np.nan)
            alpha_list.append(alpha_i)
            null_r2 = glm.permutation_baishen_parallel(feature_mat_i, data_i, n_perms,np.r_[0:np.shape(feature_mat_i)[2]],glm_out,alpha_i)
        elif model=='partial':
            # Get the residuals of data and seleted features controlling out unseleted features
            # 这一部分写法不是很规范，但是因为最后我定住了alpha所以下面获得alpha_i的语句只是如写，之后要改
            feature_mat_i = filtered_events_list[i][:, :, feature_seleted]
            alpha_i, _ = glm.compute_r2_loop(feature_mat_i, np.r_[0:np.shape(feature_mat_i)[2]], data_i, 'alpha',
                                             np.nan)
            alpha_list.append(alpha_i)
            feature_mat_i_res, data_i_res = glm.par_regress(filtered_events_list[i], feature_seleted,feature_controlled, data_i,glm_out,alpha_i)
            # Get null distribution
            null_r2 = glm.permutation_baishen_parallel(feature_mat_i_res, data_i_res, n_perms,
                                                       np.r_[0:np.shape(feature_mat_i_res)[2]],glm_out,alpha_i)
        elif model=='full':
            alpha_i, _ = glm.compute_r2_loop(filtered_events_list[i], feature_seleted, data_i, 'alpha',np.nan)
            alpha_list.append(alpha_i)
            null_r2 = glm.permutation_baishen_parallel(filtered_events_list[i], data_i, n_perms,feature_seleted,glm_out,alpha_i)

    # save the null distribution
        np.save(os.path.join('data',f'null_r2 {subjs[i]} {event} {task_Tag} {wordness} {glm_fea}.npy'), null_r2)
        del null_r2

    #%% Generate an original significant matrix (channels*times) (GLM_original vs. GLM_null)
    for i, data_i in enumerate(data_list):
        # feature_mat_i: feature matrix, observations * channels * features
        # data_i: eeg data matrix, observations * channels * times
        # r2_i: r2 matrix, channels * features * times
        alpha_i=alpha_list[i]
        print(f"Getting uncorrected significance: Patient {subjs[i]} in {event} {task_Tag} {wordness} {glm_fea}")
        if model=='simple' or (model=='partial' and glm_fea=='Acoustic') or model=='BSL_correct':
            feature_mat_i=filtered_events_list[i][:,:,feature_seleted]
            r2_i,_ = glm.compute_r2_loop(feature_mat_i, np.r_[0:np.shape(feature_mat_i)[2]],data_i,glm_out,alpha_i)
        elif model == 'partial':
            # Get the residuals of data and seleted features controlling out unseleted features
            feature_mat_i_res, data_i_res = glm.par_regress(filtered_events_list[i], feature_seleted,
                                                            feature_controlled, data_i,glm_out,alpha_i,True)
            r2_i, _ = glm.compute_r2_loop(feature_mat_i_res, np.r_[0:np.shape(feature_mat_i_res)[2]], data_i_res,glm_out,alpha_i)
        elif model=='full':
            r2_i, _ = glm.compute_r2_loop(filtered_events_list[i], feature_seleted,data_i,glm_out,alpha_i)
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
    parser.add_argument("--glm_out", type=str, required=True, help="Event type")

    args = parser.parse_args()
    main(args.event, args.task_Tag, args.glm_fea, args.wordness, args.glm_out)