#%% Introduction
# This script is made for the comparison in HG traces among Repeat vs. YesNo, Delay vs. NoDelay, and Word vs. Nonword
import os
import sys
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp

# %% groups of patients
from pickle import FALSE
# Relocate the working directory if needed
# Only need it if run it in an editor. If run in terminal, use cd.
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\GLM\\step1_glm_permute.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
datasource='hg' # 'glm_(Feature)' or 'hg'
groupsTag="LexDelay"
sf_dir = 'D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\YN_WNW_HGtraces_compare_06082025\\plots'
#groupsTag="LexDelay&LexNoDelay"

# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions

# For lexical delay task, whether run the data only with repeat tasks
trial_labels='CORRECT'

# %% Sort data and get significant electrode lists
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

fig_save_dir = os.path.join(LAB_root, 'D_Data','LexicalDecRepDelay','Baishen_Figs','LexicalDecRepDelay','group')
if not os.path.exists(os.path.join(fig_save_dir)):
    os.mkdir(os.path.join(fig_save_dir))

stats_save_root = os.path.join(stats_root_delay,'group')
if not os.path.exists(os.path.join(stats_save_root)):
    os.mkdir(os.path.join(stats_save_root))

if groupsTag=="LexDelay":

    data_LexDelayRep_Aud,subjs=gp.load_stats('mask','Auditory_inRep','ave',stats_root_delay,stats_root_delay)
    epoc_LexDelayRep_Aud,_=gp.load_stats('pval','Auditory_inRep','ave',stats_root_delay,stats_root_delay,trial_labels=trial_labels)

    # Get the ROI of labels
    ch_labels_roi,ch_labels=gp.chs2atlas(subjs,data_LexDelayRep_Aud.labels[0])

    data_LexDelayRep_Resp, _ = gp.load_stats('mask', 'Resp_inRep', 'ave', stats_root_delay, stats_root_delay)
    epoc_LexDelayRep_Resp, _ = gp.load_stats('pval', 'Resp_inRep', 'ave', stats_root_delay, stats_root_delay)

    data_LexDelayRep_NWW_Aud,_=gp.load_stats('mask','Auditory_inRep_NWW','ave',stats_root_delay,stats_root_delay)
    epoc_LexDelayRep_NWW_Aud,_=gp.load_stats('pval','Auditory_inRep_NWW','ave',stats_root_delay,stats_root_delay)

    data_LexDelayRep_NWW_Resp, _ = gp.load_stats('mask', 'Resp_inRep_NWW', 'ave', stats_root_delay, stats_root_delay)
    epoc_LexDelayRep_NWW_Resp, _ = gp.load_stats('pval', 'Resp_inRep_NWW', 'ave', stats_root_delay, stats_root_delay)

    data_LexDelayRep_WNW_Aud,_=gp.load_stats('mask','Auditory_inRep_WNW','ave',stats_root_delay,stats_root_delay)
    epoc_LexDelayRep_WNW_Aud,_=gp.load_stats('pval','Auditory_inRep_WNW','ave',stats_root_delay,stats_root_delay)

    data_LexDelayRep_WNW_Resp, _ = gp.load_stats('mask', 'Resp_inRep_WNW', 'ave', stats_root_delay, stats_root_delay)
    epoc_LexDelayRep_WNW_Resp, _ = gp.load_stats('pval', 'Resp_inRep_WNW', 'ave', stats_root_delay, stats_root_delay)

    data_LexDelayYN_Aud,_=gp.load_stats('mask','Auditory_inYN','ave',stats_root_delay,stats_root_delay)
    epoc_LexDelayYN_Aud,_=gp.load_stats('pval','Auditory_inYN','ave',stats_root_delay,stats_root_delay)

    data_LexDelayYN_Resp, _ = gp.load_stats('mask', 'Resp_inYN', 'ave', stats_root_delay, stats_root_delay)
    epoc_LexDelayYN_Resp, _ = gp.load_stats('pval', 'Resp_inYN', 'ave', stats_root_delay, stats_root_delay)

    data_LexDelayYN_NWW_Aud,_=gp.load_stats('mask','Auditory_inYN_NWW','ave',stats_root_delay,stats_root_delay)
    epoc_LexDelayYN_NWW_Aud,_=gp.load_stats('pval','Auditory_inYN_NWW','ave',stats_root_delay,stats_root_delay)

    data_LexDelayYN_NWW_Resp, _ = gp.load_stats('mask', 'Resp_inYN_NWW', 'ave', stats_root_delay, stats_root_delay)
    epoc_LexDelayYN_NWW_Resp, _ = gp.load_stats('pval', 'Resp_inYN_NWW', 'ave', stats_root_delay, stats_root_delay)

    data_LexDelayYN_WNW_Aud,_=gp.load_stats('mask','Auditory_inYN_WNW','ave',stats_root_delay,stats_root_delay)
    epoc_LexDelayYN_WNW_Aud,_=gp.load_stats('pval','Auditory_inYN_WNW','ave',stats_root_delay,stats_root_delay)

    data_LexDelayYN_WNW_Resp, _ = gp.load_stats('mask', 'Resp_inYN_WNW', 'ave', stats_root_delay, stats_root_delay)
    epoc_LexDelayYN_WNW_Resp, _ = gp.load_stats('pval', 'Resp_inYN_WNW', 'ave', stats_root_delay, stats_root_delay)

    data_LexDelayYN_Rep_Aud,_=gp.load_stats('mask','Auditory_inYN_inRep','ave',stats_root_delay,stats_root_delay)
    epoc_LexDelayYN_Rep_Aud,_=gp.load_stats('pval','Auditory_inYN_inRep','ave',stats_root_delay,stats_root_delay)

    data_LexDelayYN_Rep_Resp, _ = gp.load_stats('mask', 'Resp_inYN_inRep', 'ave', stats_root_delay, stats_root_delay)
    epoc_LexDelayYN_Rep_Resp, _ = gp.load_stats('pval', 'Resp_inYN_inRep', 'ave', stats_root_delay, stats_root_delay)


elif groupsTag=="LexDelay&LexNoDelay":

    # first get the patient inform from no delay tasks and then extract the corresponding
    data_LexDelayRep_Aud,subjs=gp.load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_delay)
    data_LexNoDelayRep_Aud,_=gp.load_stats(stat_type,'Auditory_inRep',contrast,stats_root_nodelay,stats_root_nodelay)
    data_LexNoDelayJL_Aud,_=gp.load_stats(stat_type,'Auditory_inSilence',contrast,stats_root_nodelay,stats_root_nodelay)

    # Get the ROI of labels
    ch_labels_roi,ch_labels=gp.chs2atlas(subjs,data_LexDelayRep_Aud.labels[0])

    data_LexDelayRep_Resp, _ = gp.load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_delay)
    data_LexNoDelayRep_Resp, _ = gp.load_stats(stat_type, 'Resp_inRep', contrast, stats_root_nodelay, stats_root_nodelay)

    epoc_LexDelayRep_Aud,_=gp.load_stats('pval','Auditory_inRep','ave',stats_root_nodelay,stats_root_delay,trial_labels=trial_labels)
    epoc_LexNoDelayRep_Aud,_=gp.load_stats('pval','Auditory_inRep','ave',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)
    epoc_LexNoDelayJL_Aud,_=gp.load_stats('pval','Auditory_inSilence','ave',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

    epoc_LexDelayRep_Resp,_=gp.load_stats('pval','Resp_inRep','ave',stats_root_nodelay,stats_root_delay,trial_labels=trial_labels)
    epoc_LexNoDelayRep_Resp,_=gp.load_stats('pval','Resp_inRep','ave',stats_root_nodelay,stats_root_nodelay,trial_labels=trial_labels)

chs_coor=gp.get_coor(data_LexDelayRep_Aud.labels[0],'group')
hickok_roi_labels, hickok_roi_sig_idx=gp.hickok_roi_sphere(chs_coor)

# %% Get Auditory, Sensory-motor, and Motor electrodes for Repeat and YesNo compared to baseline：

# Parameters from the lexical delay task
mean_word_len=0.5#0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0.0 # a short period of time that we may assume auditory decay takes
delay_len=1 # from task script
motor_prep_win=[-0.25,-0.1] # get windows for motor preparation (0.1s to avoid high gamma filter leakage)
motor_resp_win=[-0.1,0.75] # get windows for motor response (0.75s to avoid too much auditory feedback)
pre_stimonset_win=[-0.5,0]
cluster_twin=0.011 # length of sig cluster (if it is 0.011, one sample only)

Lex_idxes = dict()
for tag,masks,epocs in zip(
        ('Rep','YN'),
        ((data_LexDelayRep_Aud,data_LexDelayRep_Resp),(data_LexDelayYN_Aud,data_LexDelayYN_Resp)),
        ((epoc_LexDelayRep_Aud,epoc_LexDelayRep_Resp),(epoc_LexDelayYN_Aud,epoc_LexDelayYN_Resp))
):

    Lex_idx = dict()

    # (Auditory)
    _, _, _, LexDelay_Aud_sig_idx = gp.sort_chs_by_actonset(masks[0], epocs[0],cluster_twin,[-0.1, mean_word_len + auditory_decay])

    # (Delay)
    _, _, _, LexDelay_Delay_sig_idx = gp.sort_chs_by_actonset(masks[0], epocs[0],cluster_twin,[mean_word_len + auditory_decay - 0.1,
                                                                                     mean_word_len + auditory_decay + delay_len + 0.1])
    # (Motor prepare)
    _, _, _, LexDelay_Motor_Prep_sig_idx = gp.sort_chs_by_actonset(masks[1],epocs[1],cluster_twin, motor_prep_win)

    # (Motor response)
    _, _, _, LexDelay_Motor_Resp_sig_idx = gp.sort_chs_by_actonset(masks[1],epocs[1],cluster_twin, motor_resp_win)

    # Channel selection: Auditory nomotor electrodes (auditory window:1, motor prep: 0)
    LexDelay_Aud_NoMotor_sig_idx = LexDelay_Aud_sig_idx - LexDelay_Motor_Prep_sig_idx

    # Channel selection: Sensorimotor electrodes (auditory window:1, motor prep: 1)
    LexDelay_Sensorimotor_sig_idx = LexDelay_Aud_sig_idx & LexDelay_Motor_Prep_sig_idx

    # Channel selection: Sensory OR motor electrodes (auditory window:1 or motor prep: 1 or motor resp: 1)
    LexDelay_Sensory_OR_Motor_sig_idx = LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx

    # Channel selection: All sig electrodes in Aud epoch
    LexDelay_Aud_all_sig_idx = LexDelay_Aud_sig_idx | LexDelay_Delay_sig_idx

    # Channel selection: All sig electrodes in Resp epoch
    LexDelay_Resp_all_sig_idx = LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx

    # Channel selection: Motor electrodes (auditory window:0, motor resp: 1)
    LexDelay_Motor_sig_idx = LexDelay_Motor_Resp_sig_idx - LexDelay_Aud_sig_idx

    # Channel selection: Delay only electrodes (delay electrodes ,with: auditory window:0, motor prep: 0, motor resp: 0)
    LexDelay_DelayOnly_sig_idx = LexDelay_Delay_sig_idx - (
                LexDelay_Aud_sig_idx | LexDelay_Motor_Prep_sig_idx | LexDelay_Motor_Resp_sig_idx)

    # Channel selection: Auditory electrodes in Delay electrodes
    LexDelay_Auditory_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Aud_NoMotor_sig_idx

    # Channel selection: Sensorimotor electrodes in Delay electrodes
    LexDelay_Sensorimotor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Sensorimotor_sig_idx

    # Channel selection: Motor electrodes in Delay electrodes
    LexDelay_Motor_in_Delay_sig_idx = LexDelay_Delay_sig_idx & LexDelay_Motor_sig_idx

    # Motor_prep only
    LexDelay_Motorprep_Only_sig_idx = (LexDelay_Motor_Prep_sig_idx - (
                LexDelay_Aud_NoMotor_sig_idx | LexDelay_Sensorimotor_sig_idx | LexDelay_Motor_sig_idx | LexDelay_DelayOnly_sig_idx))

    Lex_idx['LexDelay_Aud_NoMotor_sig_idx']=LexDelay_Aud_NoMotor_sig_idx
    Lex_idx['LexDelay_Sensorimotor_sig_idx']=LexDelay_Sensorimotor_sig_idx
    Lex_idx['LexDelay_Motor_sig_idx']=LexDelay_Motor_sig_idx
    Lex_idx['LexDelay_Delay_sig_idx']=LexDelay_Delay_sig_idx
    Lex_idx['LexDelay_DelayOnly_sig_idx']=LexDelay_DelayOnly_sig_idx
    Lex_idx['LexDelay_Auditory_in_Delay_sig_idx']=LexDelay_Auditory_in_Delay_sig_idx
    Lex_idx['LexDelay_Aud_all_sig_idx']=LexDelay_Aud_all_sig_idx
    Lex_idx['LexDelay_Resp_all_sig_idx']=LexDelay_Resp_all_sig_idx
    Lex_idx['LexDelay_Sensorimotor_in_Delay_sig_idx']=LexDelay_Sensorimotor_in_Delay_sig_idx
    Lex_idx['LexDelay_Motor_in_Delay_sig_idx']=LexDelay_Motor_in_Delay_sig_idx
    Lex_idx['LexDelay_Motorprep_Only_sig_idx']=LexDelay_Motorprep_Only_sig_idx

    Lex_idxes[tag]=Lex_idx

for tag,masks,epocs in zip(
        ('Rep_WNW','YN_WNW','Rep_NWW','YN_NWW'),
        ((data_LexDelayRep_WNW_Aud,data_LexDelayRep_WNW_Resp),
         (data_LexDelayYN_WNW_Aud,data_LexDelayYN_WNW_Resp),
         (data_LexDelayRep_NWW_Aud,data_LexDelayRep_NWW_Resp),
         (data_LexDelayYN_NWW_Aud,data_LexDelayYN_NWW_Resp)),
        ((epoc_LexDelayRep_WNW_Aud,epoc_LexDelayRep_WNW_Resp),
         (epoc_LexDelayYN_WNW_Aud,epoc_LexDelayYN_WNW_Resp),
         (epoc_LexDelayRep_NWW_Aud,epoc_LexDelayRep_NWW_Resp),
         (epoc_LexDelayYN_NWW_Aud,epoc_LexDelayYN_NWW_Resp))
):

    Lex_idx = dict()

    # (Auditory)
    _, _, _, LexDelay_Aud_sig_idx = gp.sort_chs_by_actonset(masks[0], epocs[0],cluster_twin,[-0.1, 10])

    # (Motor response)
    _, _, _, LexDelay_Motor_Resp_sig_idx = gp.sort_chs_by_actonset(masks[1],epocs[1],cluster_twin, [-0.1, 10])

    Lex_idx['Aud_epoc']=LexDelay_Aud_sig_idx
    Lex_idx['Resp_epoc']=LexDelay_Motor_Resp_sig_idx

    Lex_idxes[tag]=Lex_idx

# YesNo - Repeat
_, _, _, LexDelay_Aud_YN_Rep_sig_idx = gp.sort_chs_by_actonset(data_LexDelayYN_Rep_Aud, epoc_LexDelayYN_Rep_Aud,cluster_twin,[-0.1, 10])
_, _, _, LexDelay_Resp_YN_Rep_sig_idx = gp.sort_chs_by_actonset(data_LexDelayYN_Rep_Resp, epoc_LexDelayYN_Rep_Resp,cluster_twin,[-0.1, 10])
Lex_idx = dict()
Lex_idx['Aud_epoc'] = LexDelay_Aud_YN_Rep_sig_idx
Lex_idx['Resp_epoc'] = LexDelay_Resp_YN_Rep_sig_idx
Lex_idxes['YN_Rep'] = Lex_idx

# %% Plot 3d brain surfaces for each group of sig electrodes：
len_d = len(data_LexDelayRep_Aud.labels[0])
for Lex_idx_tag,Lex_idx in Lex_idxes.items():
    # if Lex_idx_tag=='Rep' or Lex_idx_tag=='YN':
    #     continue
    hickok_roi_all = pd.DataFrame()
    for ele_subgrp_tag,ele_subgrp in Lex_idx.items():
        chs_cols_picked = [[1,0,0] for i in range(len(ele_subgrp))]
        pick_labels = [data_LexDelayRep_Aud.labels[0][i] for i in list(ele_subgrp)]
        gp.plot_brain(subjs, pick_labels, chs_cols_picked,None,os.path.join(sf_dir,f'{Lex_idx_tag}_{ele_subgrp_tag}_lh.jpg'),hemi='lh',save_img=True)
        gp.plot_brain(subjs, pick_labels, chs_cols_picked,None,os.path.join(sf_dir,f'{Lex_idx_tag}_{ele_subgrp_tag}_rh.jpg'),hemi='rh',save_img=True)
        gp.atlas2_hist(ch_labels_roi, pick_labels, chs_cols_picked[0],
                    os.path.join(sf_dir, f'Atlas histogram {Lex_idx_tag} {ele_subgrp_tag}.tif'),ylim=[0,50])
        gp.plot_sig_roi_counts(hickok_roi_labels, chs_cols_picked[0], gp.set2arr(ele_subgrp,len_d),
                            os.path.join(sf_dir, f'Hickok ROI histogram {Lex_idx_tag} {ele_subgrp_tag}.tif'))
        hickok_roi_all[f'{Lex_idx_tag}_{ele_subgrp_tag}'] = gp.get_sig_roi_counts(hickok_roi_labels,ele_subgrp)

    gp.plot_roi_counts_comparison(hickok_roi_all,os.path.join(sf_dir, f'Hickok ROI his {Lex_idx_tag} {ele_subgrp_tag}'))

# %% Plot venn plots for overlapping between Repeat and YN：
from matplotlib_venn import venn3,venn2
import matplotlib.pyplot as plt

Waveplot_wth=10 # Width of wave plots
Waveplot_hgt=4 # Height of wave plots

# get electrodes with auditory and with motor responses from the LexDelay YN dataset
rep_aud_idx=Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx'] | Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']
rep_mtr_idx=Lex_idxes['Rep']['LexDelay_Motor_sig_idx'] | Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']
yn_aud_idx=Lex_idxes['YN']['LexDelay_Aud_NoMotor_sig_idx'] | Lex_idxes['YN']['LexDelay_Sensorimotor_sig_idx']
yn_mtr_idx=Lex_idxes['YN']['LexDelay_Motor_sig_idx'] | Lex_idxes['YN']['LexDelay_Sensorimotor_sig_idx']

for tag, rep_ele_idxs, yn_ele_idx1s, yn_ele_idx2s, yn_ele_idx3s in zip(
        ('Aud','SM','Del','Mtr'),
        (Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx'],Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx'],Lex_idxes['Rep']['LexDelay_Delay_sig_idx'],Lex_idxes['Rep']['LexDelay_Motor_sig_idx']),
        (Lex_idxes['YN']['LexDelay_Aud_NoMotor_sig_idx'],Lex_idxes['YN']['LexDelay_Sensorimotor_sig_idx'],Lex_idxes['YN']['LexDelay_Delay_sig_idx'],Lex_idxes['YN']['LexDelay_Motor_sig_idx']),
        (yn_aud_idx, yn_aud_idx,yn_aud_idx, yn_mtr_idx),
        (Lex_idxes['YN']['LexDelay_Aud_all_sig_idx'],Lex_idxes['YN']['LexDelay_Aud_all_sig_idx'],Lex_idxes['YN']['LexDelay_Aud_all_sig_idx'],Lex_idxes['YN']['LexDelay_Resp_all_sig_idx'])
):

    # electrodes overlaped with the same set of electrodes (e.g., Auditory in Repeat with Auditory in Yes_No)
    plt.figure(figsize=(6, 6))
    venn2([rep_ele_idxs, yn_ele_idx1s], (f'{tag}_inRep', f'{tag}_inYN'))
    plt.tight_layout()
    plt.savefig(os.path.join(sf_dir,f'{tag}_inRep_inYN.tif'), dpi=300)
    plt.close()

    # electrodes overlaped with the same set of electrodes (e.g., Auditory in Repeat with Auditory&SM  in Yes_No)
    venn2([rep_ele_idxs, yn_ele_idx2s], (f'{tag}_inRep', f'{tag}_SM_inYN'))
    plt.tight_layout()
    plt.savefig(os.path.join(sf_dir,f'{tag}_inRep_{tag}_SM_inYN.tif'), dpi=300)
    plt.close()

    # electrodes overlaped with the same set of electrodes (e.g., Auditory in Repeat with all activated electrodes in Yes_No)
    venn2([rep_ele_idxs, yn_ele_idx3s], (f'{tag}_inRep', f'All_inYN'))
    plt.tight_layout()
    plt.savefig(os.path.join(sf_dir,f'{tag}_inRep_All_inYN.tif'), dpi=300)
    plt.close()

for tag,epoc in zip(
        ('Rep','YN'),
        (epoc_LexDelayRep_Aud,epoc_LexDelayYN_Aud)
):
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plt.title('Z-scores in lexical delay repeat tasks (aligned to stim onset)', fontsize=20)
    plt.xlim([-0.25, 2.5])
    gp.plot_wave(epoc, rep_aud_idx & yn_aud_idx, f'AudSM_inRep & AudSM_inYN', [1,0,0], '-', False,ylim=[0,1.5])
    gp.plot_wave(epoc, yn_mtr_idx - rep_mtr_idx, f'AudSM_inYN - AudSM_inRep', [0,1,0], '-', False,ylim=[0,1.5])
    gp.plot_wave(epoc, rep_mtr_idx - yn_mtr_idx, f'AudSM_inRep - AudSM_inYN', [0,0,1], '-', False,ylim=[0,1.5])
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.legend(loc='upper right', fontsize=15)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(sf_dir,f'AudSM_inRep_inYN_{tag}_epoc.tif'), dpi=300)
    plt.close()

for tag, epoc in zip(
        ('Rep', 'YN'),
        (epoc_LexDelayRep_Aud, epoc_LexDelayYN_Aud)
):
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plt.title('Z-scores in lexical delay repeat tasks (aligned to stim onset)', fontsize=20)
    plt.xlim([-0.25, 2.5])
    gp.plot_wave(epoc, Lex_idxes['Rep']['LexDelay_Delay_sig_idx'] & Lex_idxes['YN']['LexDelay_Delay_sig_idx'], f'Del_inRep & Del_inYN', [1,0,0], '-', False,ylim=[0,1.5])
    gp.plot_wave(epoc, Lex_idxes['Rep']['LexDelay_Delay_sig_idx'] - Lex_idxes['YN']['LexDelay_Delay_sig_idx'], f'Del_inYN - Del_inRep', [0,1,0], '-', False,ylim=[0,1.5])
    gp.plot_wave(epoc, Lex_idxes['YN']['LexDelay_Delay_sig_idx'] - Lex_idxes['Rep']['LexDelay_Delay_sig_idx'], f'Del_inRep - Del_inYN', [0,0,1], '-', False,ylim=[0,1.5])
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.legend(loc='upper right', fontsize=15)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(sf_dir,f'Del_inRep_inYN_{tag}_epoc.tif'), dpi=300)
    plt.close()

# %% Plot venn plots for overlapping between Repeat and Word：

# Plot traces for NWW and WNW (aligned to Auditory onset):
plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
plt.title('Word>Nonword and Nonword>Word electrodes (aligned to stim onset)', fontsize=20)
plt.xlim([-0.25, 2.5])
gp.plot_wave(epoc_LexDelayRep_WNW_Aud, Lex_idxes['Rep_WNW']['Aud_epoc'], f'Repeat W-NW epoc N={len(Lex_idxes['Rep_WNW']['Aud_epoc'])}', [233/255,181/255,193/255], '-', False,ylim=[-0.25,0.5])
gp.plot_wave(epoc_LexDelayRep_NWW_Aud, Lex_idxes['Rep_NWW']['Aud_epoc'], f'Repeat NW-W epoc N={len(Lex_idxes['Rep_NWW']['Aud_epoc'])}', [180/255,197/255,217/255], '-', False,ylim=[-0.25,0.5])
gp.plot_wave(epoc_LexDelayYN_WNW_Aud, Lex_idxes['YN_WNW']['Aud_epoc'], f'YesNo W-NW epoc N={len(Lex_idxes['YN_WNW']['Aud_epoc'])}', [233/255,181/255,193/255], '--', False,ylim=[-0.25,0.5])
gp.plot_wave(epoc_LexDelayYN_NWW_Aud, Lex_idxes['YN_NWW']['Aud_epoc'], f'YesNo NW-W epoc N={len(Lex_idxes['YN_NWW']['Aud_epoc'])}', [180/255,197/255,217/255], '--', False,ylim=[-0.25,0.5])
plt.axvline(x=0, linestyle='--', color='k')
plt.axhline(y=0, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=15)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(sf_dir,f'NWW and WNW.tif'), dpi=300)
plt.close()

# Plot traces for NWW and WNW (aligned to Motor onset):
plt.figure(figsize=(Waveplot_wth * (150 / 350), Waveplot_hgt))
plt.title('Word>Nonword and Nonword>Word electrodes (aligned to motor onset)', fontsize=20)
plt.xlim([-0.25, 1])
gp.plot_wave(epoc_LexDelayRep_WNW_Resp, Lex_idxes['Rep_WNW']['Resp_epoc'], f'Repeat W-NW epoc N={len(Lex_idxes['Rep_WNW']['Resp_epoc'])}', [233/255,181/255,193/255], '-', False,ylim=[-0.25,0.5])
gp.plot_wave(epoc_LexDelayRep_NWW_Resp, Lex_idxes['Rep_NWW']['Resp_epoc'], f'Repeat NW-W epoc N={len(Lex_idxes['Rep_NWW']['Resp_epoc'])}', [180/255,197/255,217/255], '-', False,ylim=[-0.25,0.5])
gp.plot_wave(epoc_LexDelayYN_WNW_Resp, Lex_idxes['YN_WNW']['Resp_epoc'], f'YesNo W-NW epoc N={len(Lex_idxes['YN_WNW']['Resp_epoc'])}', [233/255,181/255,193/255], '--', False,ylim=[-0.25,0.5])
gp.plot_wave(epoc_LexDelayYN_NWW_Resp, Lex_idxes['YN_NWW']['Resp_epoc'], f'YesNo NW-W epoc N={len(Lex_idxes['YN_NWW']['Resp_epoc'])}', [180/255,197/255,217/255], '--', False,ylim=[-0.25,0.5])
plt.axvline(x=0, linestyle='--', color='k')
plt.axhline(y=0, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=15).set_visible(False)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(sf_dir,f'NWW and WNW Resp.tif'), dpi=300)
plt.close()

for tag,bslcomp_ele,taskcomp_ele1,taskcomp_ele2 in zip(
        ('Aud_Aud','SM_Aud','Delay_Aud','DelayOnly_Aud','SM_Resp','M_Resp'),
        (Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx'],
         Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx'],
         Lex_idxes['Rep']['LexDelay_Delay_sig_idx'],
         Lex_idxes['Rep']['LexDelay_DelayOnly_sig_idx'],
         Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx'],
         Lex_idxes['Rep']['LexDelay_Motor_sig_idx']),
        (Lex_idxes['Rep_WNW']['Aud_epoc'],
         Lex_idxes['Rep_WNW']['Aud_epoc'],
         Lex_idxes['Rep_WNW']['Aud_epoc'],
         Lex_idxes['Rep_WNW']['Aud_epoc'],
         Lex_idxes['Rep_WNW']['Resp_epoc'],
         Lex_idxes['Rep_WNW']['Resp_epoc']),
        (Lex_idxes['Rep_NWW']['Aud_epoc'],
         Lex_idxes['Rep_NWW']['Aud_epoc'],
         Lex_idxes['Rep_NWW']['Aud_epoc'],
         Lex_idxes['Rep_NWW']['Aud_epoc'],
         Lex_idxes['Rep_NWW']['Resp_epoc'],
         Lex_idxes['Rep_NWW']['Resp_epoc'])
):
    # electrodes overlaped with the same set of electrodes (e.g., Auditory in Repeat with Auditory in Yes_No)
    plt.figure(figsize=(6, 6))
    venn3([bslcomp_ele,taskcomp_ele1,taskcomp_ele2], (f'BSLcontrast_inRep', f'Word-Nonword_inRep', f'Nonword-Word_inRep'))
    plt.tight_layout()
    plt.savefig(os.path.join(sf_dir, f'{tag}_inRep_WNW_NWW.tif'), dpi=300)
    plt.close()

#%% YN vs Repeat
# Sensory-motor & Nonword-Word:
plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
plt.title('YN vs. Repeat', fontsize=20)
plt.xlim([-0.25, 2.5])
gp.plot_wave(epoc_LexDelayRep_Aud, Lex_idxes['Rep']['LexDelay_Aud_all_sig_idx'], f'Repeat epoc N={len(Lex_idxes['Rep']['LexDelay_Aud_all_sig_idx'])}', [233/255,181/255,193/255], '-', False,ylim=[-0.25,1.5])
gp.plot_wave(epoc_LexDelayYN_Aud, Lex_idxes['YN']['LexDelay_Aud_all_sig_idx'], f'YesNo epoc N={len(Lex_idxes['YN']['LexDelay_Aud_all_sig_idx'])}', [180/255,197/255,217/255], '-', False,ylim=[-0.25,1.5])
gp.plot_wave(epoc_LexDelayYN_Rep_Aud, Lex_idxes['YN_Rep']['Aud_epoc'], f'YesNo-Repeat epoc N={len(Lex_idxes['YN_Rep']['Aud_epoc'])}', [180/255,197/255,217/255], '--', False,ylim=[-0.25,1.5])
plt.axvline(x=0, linestyle='--', color='k')
plt.axhline(y=0, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=15)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(sf_dir,f'YN_Rep.tif'), dpi=300)
plt.close()


#%% Specific for Auditory & Nonword-Word, and Sensory-motor & Nonword-Word, and Word-Nonword & Sensorymotor

# Auditory & Nonword-Word:
plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
plt.title('Auditory electrodes and Nonword-word', fontsize=20)
plt.xlim([-0.25, 2.5])
gp.plot_wave(epoc_LexDelayRep_Aud, Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc'], f'Aud epoc N={len(Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc'])}', [233/255,181/255,193/255], '-', False,ylim=[-0.25,1.5])
gp.plot_wave(epoc_LexDelayRep_NWW_Aud, Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc'], f'NW-W epoc N={len(Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc'])}', [180/255,197/255,217/255], '-', False,ylim=[-0.25,1.5])
plt.axvline(x=0, linestyle='--', color='k')
plt.axhline(y=0, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=15)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(sf_dir,f'Auditory electrodes and NWW.tif'), dpi=300)
plt.close()

gp.plot_brain(subjs, list(Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc']), chs_cols_picked, None,
              os.path.join(sf_dir, f'Auditory electrodes and NWW _lh.jpg'), hemi='lh', save_img=True)
gp.plot_brain(subjs, list(Lex_idxes['Rep']['LexDelay_Aud_NoMotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc']), chs_cols_picked, None,
              os.path.join(sf_dir, f'Auditory electrodes and NWW _rh.jpg'), hemi='rh', save_img=True)

# Sensory-motor & Nonword-Word:
plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
plt.title('Sensorymotor electrodes and Nonword-word', fontsize=20)
plt.xlim([-0.25, 2.5])
gp.plot_wave(epoc_LexDelayRep_Aud, Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc'], f'SM epoc N={len(Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc'])}', [233/255,181/255,193/255], '-', False,ylim=[-0.25,1.5])
gp.plot_wave(epoc_LexDelayRep_NWW_Aud, Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc'], f'NW-W epoc N={len(Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc'])}', [180/255,197/255,217/255], '-', False,ylim=[-0.25,1.5])
plt.axvline(x=0, linestyle='--', color='k')
plt.axhline(y=0, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=15)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(sf_dir,f'SM electrodes and NWW.tif'), dpi=300)
plt.close()

gp.plot_brain(subjs, list(Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc']), chs_cols_picked, None,
              os.path.join(sf_dir, f'Sensorimotor electrodes and NWW _lh.jpg'), hemi='lh', save_img=True)
gp.plot_brain(subjs, list(Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_NWW']['Aud_epoc']), chs_cols_picked, None,
              os.path.join(sf_dir, f'Sensorimotor electrodes and NWW _rh.jpg'), hemi='rh', save_img=True)

# Motor & Nonword-Word:
plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
plt.title('Motor and Word-Nonword', fontsize=20)
plt.xlim([-0.25, 2.5])
gp.plot_wave(epoc_LexDelayRep_Aud, Lex_idxes['Rep']['LexDelay_Motor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc'], f'Motor epoc N={len(Lex_idxes['Rep']['LexDelay_Motor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc'])}', [233/255,181/255,193/255], '-', False,ylim=[-0.25,1.5])
gp.plot_wave(epoc_LexDelayRep_WNW_Aud, Lex_idxes['Rep']['LexDelay_Motor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc'], f'W-NW epoc N={len(Lex_idxes['Rep']['LexDelay_Motor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc'])}', [180/255,197/255,217/255], '-', False,ylim=[-0.25,1.5])
plt.axvline(x=0, linestyle='--', color='k')
plt.axhline(y=0, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=15)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(sf_dir,f'Motor electrodes and WNW.tif'), dpi=300)
plt.close()

gp.plot_brain(subjs, list(Lex_idxes['Rep']['LexDelay_Motor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc']), chs_cols_picked, None,
              os.path.join(sf_dir, f'Motor electrodes and WNW _lh.jpg'), hemi='lh', save_img=True)
gp.plot_brain(subjs, list(Lex_idxes['Rep']['LexDelay_Motor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc']), chs_cols_picked, None,
              os.path.join(sf_dir, f'Motor electrodes and WNW _rh.jpg'), hemi='rh', save_img=True)

# Sensory-motor & Nonword-Word:
plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
plt.title('Sensorymotor electrodes and Word-Nonword', fontsize=20)
plt.xlim([-0.25, 2.5])
gp.plot_wave(epoc_LexDelayRep_Aud, Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc'], f'SM epoc N={len(Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc'])}', [233/255,181/255,193/255], '-', False,ylim=[-0.25,1.5])
gp.plot_wave(epoc_LexDelayRep_WNW_Aud, Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc'], f'W-NW epoc N={len(Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc'])}', [180/255,197/255,217/255], '-', False,ylim=[-0.25,1.5])
plt.axvline(x=0, linestyle='--', color='k')
plt.axhline(y=0, linestyle='--', color='k')
plt.legend(loc='upper right', fontsize=15)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(sf_dir,f'SM electrodes and WNW.tif'), dpi=300)
plt.close()

gp.plot_brain(subjs, list(Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc']), chs_cols_picked, None,
              os.path.join(sf_dir, f'Sensorimotor electrodes and WNW _lh.jpg'), hemi='lh', save_img=True)
gp.plot_brain(subjs, list(Lex_idxes['Rep']['LexDelay_Sensorimotor_sig_idx']&Lex_idxes['Rep_WNW']['Aud_epoc']), chs_cols_picked, None,
              os.path.join(sf_dir, f'Sensorimotor electrodes and WNW _rh.jpg'), hemi='rh', save_img=True)

#%% Hickok ROIs
# Plot the Hickok ROI traces
Sensorimotor_col = [1, 0, 0]  # Sensorimotor
Auditory_col = [0, 1, 0]  # Auditory
Delay_col = [1, 0.65, 0]  # Delay
Motor_col = [0, 0, 1]  # Motor
for tag, epoc1,sig1,epoc2,sig2 in zip(
        ('YN_Rep','NWW','WNW'),
        (epoc_LexDelayYN_Rep_Aud,epoc_LexDelayRep_NWW_Aud,epoc_LexDelayRep_WNW_Aud),
        (Lex_idxes['YN_Rep']['Aud_epoc'],Lex_idxes['Rep_NWW']['Aud_epoc'],Lex_idxes['Rep_WNW']['Aud_epoc']),
        (epoc_LexDelayYN_Rep_Aud,epoc_LexDelayYN_NWW_Aud,epoc_LexDelayYN_WNW_Aud),
        (Lex_idxes['YN_Rep']['Aud_epoc'],Lex_idxes['YN_NWW']['Aud_epoc'],Lex_idxes['YN_WNW']['Aud_epoc'])
):
    # Aligned to Auditory
    plt.figure(figsize=(Waveplot_wth, Waveplot_hgt))
    plt.title(f'Z-scores in {tag}', fontsize=20)
    wav_bsl_corr = False
    plt.xlim([-0.25, 2.5])
    gp.plot_wave(epoc1, sig1 & hickok_roi_sig_idx['Spt'],
              f'Spt n={len(sig1 & hickok_roi_sig_idx['Spt'])}',
              Auditory_col, '-', wav_bsl_corr)
    gp.plot_wave(epoc1, sig1 & hickok_roi_sig_idx['lPMC'],
              f'lPMC n={len(sig1 & hickok_roi_sig_idx['lPMC'])}', Sensorimotor_col, '-', wav_bsl_corr)
    gp.plot_wave(epoc1, sig1 & hickok_roi_sig_idx['lIFG'],
              f'lIFG n={len(sig1 & hickok_roi_sig_idx['lIFG'])}', Motor_col, '-', wav_bsl_corr)
    gp.plot_wave(epoc1, sig1 & hickok_roi_sig_idx['lIPL'],
              f'lIPL n={len(sig1 & hickok_roi_sig_idx['lIPL'])}', Delay_col, '-', wav_bsl_corr)
    if tag != 'YN_Rep':
        gp.plot_wave(epoc2, sig2 & hickok_roi_sig_idx['Spt'],
                  f'Spt (Yes_No) n={len(sig2 & hickok_roi_sig_idx['Spt'])}',
                  Auditory_col, '--', wav_bsl_corr)
        gp.plot_wave(epoc2, sig2 & hickok_roi_sig_idx['lPMC'],
                  f'lPMC (Yes_No) n={len(sig2 & hickok_roi_sig_idx['lPMC'])}', Sensorimotor_col, '--',
                  wav_bsl_corr)
        gp.plot_wave(epoc2, sig2 & hickok_roi_sig_idx['lIFG'],
                  f'lIFG (Yes_No) n={len(sig2 & hickok_roi_sig_idx['lIFG'])}', Motor_col, '--',
                  wav_bsl_corr)
        gp.plot_wave(epoc2, sig2 & hickok_roi_sig_idx['lIPL'],
                  f'lIPL (Yes_No) n={len(sig2 & hickok_roi_sig_idx['lIPL'])}', Delay_col, '--',
                  wav_bsl_corr)
    plt.axvline(x=0, linestyle='--', color='k')
    # plt.axhline(y=0, linestyle='--', color='k')
    plt.legend(loc='upper right', fontsize=15)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(sf_dir, f'{tag}_Aud_Hikcok_ROI.tif'), dpi=300)
    plt.close()