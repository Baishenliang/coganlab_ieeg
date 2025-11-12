# Set dir
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


#%% Plotting
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
is_normalize=False
is_bsl_correct=False
mode='time_cluster'
#for elec_grp in ['Hickok_Spt','Hickok_lPMC','Hickok_lIFG']:
# for elec_grp in ['Auditory_delay','Sensorymotor_delay','Motor_delay','Delay_only']:
baseline=dict()
baseline_std=dict()
test_lamndas=[1e-2,1e-1,'1','10','100','1000']
for vWM_lambda in test_lamndas:
    #for novWM_lambda in test_lamndas:
    novWM_lambda='1'
    for alignment,xlim_align in zip(
            ('Aud','Resp','Go'),
            ([-0.2, 1.75],[-0.2, 1],[-0.2, 1])):
        for elec_grp,elec_col in zip(('Auditory','Sensorymotor','Delay_only','Motor'),
                                     (Auditory_col,Sensorimotor_col,Delay_col,Motor_col)):
            # for elec_grp in ['Auditory_delay','Sensorymotor_delay']:
            # for elec_grp in ['Sensorymotor_delay']:
            # for fea,fea_tag,para_sig_barbar in zip(('Wordvec','wordness','aco','pho'),
            #                              ('Embedding','Lexical status','Acoustic','Phonemic'),
            #                             ([6,1.2],[8,1.2],[20,1.2],[20,1.2])):
            # for fea, fea_tag in zip(expand_sequence('aco','vWM',9)+expand_sequence('pho','vWM',11)+('wordnessWord',),
            #                                          expand_sequence('Acoustic','vWM',9)+expand_sequence('Phonemic','vWM',11)+('LexStatus',)):
            # # # for fea, fea_tag in zip(('wordnessWord',),
            # # #                         ('LexStatus',)):
            #     para_sig_barbar=[0.8, 0.02]
            # for fea, fea_tag, para_sig_barbar in zip(('aco','pho','wordnessWord','pho_wordnessWord','R2'),
            #                                          ('Acoustic','Phonemic','LexStatus','pho_wordnessWord','R2'),
            #                                          ([0.13, 0.001],[0.13, 0.001],[0.13, 0.001],[0.13, 0.001],[0.01, 0.001])):
            for fea, fea_tag, para_sig_barbar in zip(('ACC',),
                                                     ('ACC',),
                                                     ([0.1, 0.01],)):
                # for fea, fea_tag, para_sig_barbar in zip(('aco','pho'),
                #                                          ('Acoustic','Phonemic'),
                #                                          ([2, 0.1],[0.1, 0.03])):
                # for fea, fea_tag, para_sig_barbar in zip(('vow','con'),
                #                                          ('Vowel','Consonant'),
                #                                          ([0.8, 0.03],[0.8, 0.03])):
                # for fea, fea_tag, para_sig_barbar in zip(('aco1',),
                #                                          ('Phonemic1',),
                #                                          ([0.8, 0.03],)):
                # for fea, fea_tag, para_sig_barbar in zip(('vow', 'con'),
                #                                          ('Vowel', 'Consonant'),
                #                                          ([0.8, 0.03], [0.8, 0.03])):

                fig, ax = plt.subplots(figsize=(5.6*(xlim_align[1]-xlim_align[0]), 5))

                ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7,linewidth=3)
                if alignment == 'Aud':
                    ax.axvline(x=0.65, color='red', linestyle='--', alpha=0.7,linewidth=3)
                    ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.7,linewidth=3)

                filename = f"results/{elec_grp}_{alignment}_All_vWMλ_{vWM_lambda}_novWMλ_{novWM_lambda}.csv"
                raw = pd.read_csv(filename)

                j = 0
                for vWM,vwm_linestyle,input,pthres,vwm_text,sig_bar_col in zip(
                        ('vWM','vWM_p','novWM','novWM_p'),#,'diff'),
                        ('-','-','--','--'),#,'--'),
                        ('p','p','p','p'),#,'R2'),
                        ([2.5e-2,1e-4],[2.5e-2,1e-4],[2.5e-2,1e-4],[2.5e-2,1e-4]),#,[5e-2,5e-1]),
                        ('ACC','p','ACC','p'),#,'diff'),
                        (elec_col,elec_col,elec_col,elec_col)):#,[0.5,0.5,0.5])):
                    if fea == 'aco':
                        target_fea = list(expand_sequence('aco',vWM,9))
                    elif fea == 'pho':
                        target_fea = list(expand_sequence('pho',vWM,11))
                    elif fea == 'pho_wordnessWord':
                        target_fea = list(expand_sequence('wordnessWord.pho',vWM,11))
                    elif fea == 'wordnessWord':
                        target_fea = fea+vWM
                    elif fea == 'ACC':
                        target_fea = fea+f'_{vWM}'
                    time_point, time_series, mask_time_clus = get_traces_clus(raw, pthres[0], pthres[1],mode=mode,target_fea=target_fea,input=input)

                    time_series=gaussian_filter1d(time_series, sigma=2, mode='nearest')
                    # win_len=10
                    # time_series=uniform_filter1d(time_series, size=win_len, axis=0, mode='nearest',origin=(win_len - 1) // 2)
                    if alignment == 'Aud':
                        baseline[elec_grp] = np.min(time_series[(time_point > -0.2) & (time_point <= 0)])
                        baseline_std[elec_grp]=np.std(time_series[(time_point > -0.2) & (time_point <= 0)])
                    if is_normalize:
                        time_series = (time_series - baseline[elec_grp]) / baseline_std[elec_grp]
                        time_series = time_series/ (np.max(time_series[(time_point > xlim_align[0]) & (time_point <= xlim_align[1])]) - np.min((time_point > xlim_align[0]) & (time_point <= xlim_align[1])))
                        para_sig_bar = [1,1e-1]
                    else:
                        if is_bsl_correct:
                            time_series = (time_series - baseline[elec_grp])
                        para_sig_bar = para_sig_barbar

                    if vWM=='vWM' or vWM=='novWM':
                        ax.plot(time_point, time_series, label=f"{elec_grp}{vwm_text}", color=elec_col, linewidth=5,linestyle=vwm_linestyle)
                    true_indices = np.where(mask_time_clus)[0]
                    if true_indices.size > 0 and (vWM == 'vWM_p' or vWM == 'novWM_p' or vWM == 'diff'):
                        split_points = np.where(np.diff(true_indices) != 1)[0] + 1
                        clusters_indices = np.split(true_indices, split_points)

                        for k, cluster in enumerate(clusters_indices):
                            start_index = cluster[0]
                            end_index = cluster[-1]

                            time_step = time_point[1] - time_point[0]
                            start_time = time_point[start_index] - time_step / 2
                            end_time = time_point[end_index] + time_step / 2

                            label = f'clust{k} of pho'
                            if vWM != 'diff':
                                ax.plot([start_time, end_time], [para_sig_bar[0]-para_sig_bar[1]*(j-1),para_sig_bar[0]-para_sig_bar[1]*(j-1)],
                                        color=sig_bar_col,alpha=0.4,
                                        linewidth=10,  # Make the line thick like a bar
                                        solid_capstyle='butt')  # Makes the line ends flat
                            elif vWM == 'diff' and elec_grp !='Delay_only':
                                ax.axvspan(
                                    xmin=start_time,
                                    xmax=end_time,
                                    facecolor=sig_bar_col,
                                    alpha=0.2,
                                    edgecolor='none'
                                )
                        j=j+1

                # ax.set_title(f"{fea_tag}", fontsize=24)
                # ax.set_xlabel("Time (seconds) aligned to stim onset", fontsize=20)
                # if 'R2' in fea:
                #     ax.set_ylabel("Model $R^2$ (normalized)")
                # else:
                #     ax.set_ylabel("feature $β$")#, fontsize=20)
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
                ax.legend().set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim(-0.002,para_sig_bar[0]+2*para_sig_bar[1])
                plt.tight_layout()
                plt.savefig(os.path.join('figs','multencode', f'{elec_grp}_{fea_tag}_{alignment}_All_vWMλ_{vWM_lambda}_novWMλ_{novWM_lambda}.tif'), dpi=300)
                plt.close()
