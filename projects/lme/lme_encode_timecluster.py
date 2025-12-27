#%% Set dir
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
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp

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
WGW_p55b_col=[0.74901961, 0.25098039, 0.74901961] # WGW 55b
WGW_a55b_col=[0, 0.5, 0.5] # WGW a55b

# Feature colors
aco_col = [0, 0.502, 0.502]      # Teal (青色)
pho_col = [0.502, 0, 0.502]      # Purple (紫色)
wordness_col = [1, 0, 1] # Magenta (洋红色)

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

def add_alignment_vlines(ax, alignment):
    """
    Adds vertical lines for specific event markers to a matplotlib Axes object
    based on the alignment type.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw the lines on.
        alignment (str): The alignment type ('Aud', 'Go', or 'Resp').
    """
    if alignment == 'Aud':
        # Stim offsets
        ax.axvline(x=0.59, color='red', linestyle='--', alpha=0.7, linewidth=3)
        # ax.axvline(x=0.59 + 0.1480 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=0.59 - 0.1480 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # Delay offsets
        ax.axvline(x=1.7201, color='red', linestyle='--', alpha=0.7, linewidth=3)
        # ax.axvline(x=1.7201 + 0.1459 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=1.7201 - 0.1459 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
    elif alignment == 'Go':
        # Motor onsets
        ax.axvline(x=0.7961, color='red', linestyle='--', alpha=0.7, linewidth=3)
        # ax.axvline(x=0.7961 + 1.0532 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=0.7961 - 1.0532 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
    elif alignment == 'Resp':
        # Motor offsets
        ax.axvline(x=0.6096, color='red', linestyle='--', alpha=0.7, linewidth=3)
        ax.axvline(x=0.6096 + 0.2190, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0.6096 - 0.2190, color='red', linestyle='--', alpha=0.7, linewidth=1)

#%% Plotting
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
is_normalize=False
is_bsl_correct=False
mode='time_cluster'
#for elec_grp in ['Hickok_Spt','Hickok_lPMC','Hickok_lIFG']:
# for elec_grp in ['Auditory_delay','Sensorymotor_delay','Motor_delay','Delay_only']:
baseline=dict()
baseline_beta_rms=dict()
baseline_std=dict()
baseline_beta_rms_std=dict()
# test_lamndas=[1e-2,1e-1,'1','10','100','1000']
vWM_lambda='100'
for alignment,xlim_align in zip(
        ('Aud','Resp','Go'),
        ([-0.2, 1.75],[-0.2, 1.25],[-0.2, 1.25])):
    for elec_grp,elec_col,fea_plot_yscale in zip(('Sensorymotor','Auditory','Delay_only','Wgw_p55b','Wgw_a55b','Motor'),
                                                         (Sensorimotor_col,Auditory_col,Delay_col,WGW_p55b_col,WGW_a55b_col,Motor_col),
                                                         (0.8,1,1.3,1.2,1,0.7)):
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
                                                 ([0.07, 0.007],)):
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

            # For testing only:
            # alignment='Aud'
            # xlim_align=[-0.2, 1.75]
            # elec_grp='Auditory'
            # elec_col=Auditory_col
            # vWM_lambda=20
            # fea='ACC'
            # fea_tag='ACC'
            # para_sig_barbar=[0.1,0.01]
            # vWM='vWM'
            
            fig, ax = plt.subplots(figsize=(5.6*(xlim_align[1]-xlim_align[0]), 5))

            ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7,linewidth=3)
            add_alignment_vlines(ax, alignment)

            filename = f"results/LexDelayRep_{elec_grp}_{alignment}_All_huge_testλ_{vWM_lambda}.csv"
            raw = pd.read_csv(filename)

            j = 0
            true_indices_by_vWM = {}
            for vWM,vwm_linestyle,input,pthres,vwm_text,sig_bar_col in zip(
                    ('vWM','vWM_p'),#,'diff'),
                    ('-','-'),#,'--'),
                    ('R2','p'),#,'R2'),
                    ([2.5e-2,2.5e-2],[1e-3,1e-3]),#,[2.5e-2,2.5e-2]),#,[5e-2,1e-1]),
                    ('ACC','p'),#,'diff'),
                    (elec_col,elec_col)):#,[0.5,0.5,0.5])):
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

                if vWM=='vWM':
                    ax.plot(time_point, time_series, label=f"{elec_grp}{vwm_text}", color=elec_col, linewidth=5,linestyle=vwm_linestyle)
                true_indices = np.where(mask_time_clus)[0]
                true_indices_by_vWM[vWM] = true_indices
                if true_indices.size > 0 and (vWM == 'vWM'):
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
                scilimits=(-2, -2),  # 将指数固定为 10^-3
                useMathText=True  # 使用 LaTeX 格式显示指数，如 10⁻³
            )
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.legend(fontsize=18)
            ax.set_xlim(xlim_align)#time_point.max())
            # if elec_grp=='Motor' and alignment=='Aud':
            #     ax.set_xlim([0.5,xlim_align[1]])
            ax.legend().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(-0.002,para_sig_bar[0]+2*para_sig_bar[1])
            plt.tight_layout()
            plt.savefig(os.path.join('figs','z_score_unnormalized_huge', f'λ{vWM_lambda}',f'{elec_grp}_{fea_tag}_{alignment}_All_rawpow_vWMλ_{vWM_lambda}.tif'), dpi=300)
            plt.close()

            # %%  Now plot the beta traces for each feature
            group_beta_type='max' # 'rms' or 'max'
            raw_org = raw[raw['perm'] == 0]
            
            for is_vWM in ('_vWM',):

                all_rms_data = {}
                all_rms_data_sig = {}
                time_points_plot = None
                            # Define a color map for features for better visualization
                feature_colors = {
                    'aco': aco_col,
                    'pho': pho_col,
                    'wordnessWord': wordness_col,
                    'wordnessNonword:aco': aco_col,
                    'wordnessNonword:pho': pho_col,
                    'sem': [0.2, 0.8, 0.2]
                }

                for beta_fea in ('aco','pho','wordnessNonword:pho','sem'):#,'wordnessWord:pho','wordnessWord:aco'):
                    print(f'Feature beta plots for {beta_fea}')

                    # 1. 预先定义好所有的主效应列 (Main Effects)
                    # 确保它们已经按顺序排好 (1, 2, ... 11)
                    fea_columns_aco = [col for col in raw.columns if col.startswith('aco') and ':' not in col and is_vWM in col]
                    fea_columns_pho = [col for col in raw.columns if col.startswith('pho') and ':' not in col and is_vWM in col]
                    fea_columns_sem = [col for col in raw.columns if col.startswith('sem') and ':' not in col and is_vWM in col]


                    # 2. 根据 beta_fea 确定当前要分析的特征列 (fea_columns)
                    if beta_fea == "aco":
                        fea_columns = ['time_point'] + fea_columns_aco
                    elif beta_fea == "pho":
                        fea_columns = ['time_point'] + fea_columns_pho
                    elif beta_fea == "sem":
                        fea_columns = ['time_point'] + fea_columns_sem
                    elif (":" in beta_fea) and ("aco" in beta_fea):
                        # 如果是交互项，提取对应的交互列
                        fea_columns = ['time_point'] +['aco1:wordnessNonword_vWM']+ [col for col in raw.columns if col.startswith(beta_fea) and is_vWM in col]
                    elif (":" in beta_fea) and ("pho" in beta_fea):
                        fea_columns = ['time_point'] + [col for col in raw.columns if col.startswith(beta_fea) and is_vWM in col]

                    # 提取当前分析所需的子集 (这里只包含时间 + 当前关注的特征)
                    raw_org_fea = raw_org[fea_columns].copy()

                    # 3. 确定要计算 RMS/Max 的数值列
                    rms_cols = [col for col in fea_columns if col != 'time_point']

                    # 4. 计算逻辑
                    if group_beta_type == 'rms':
                        raw_org_fea['rms'] = np.sqrt(raw_org_fea[rms_cols].pow(2).mean(axis=1))

                    elif group_beta_type == 'max':
                        if ":" not in beta_fea:
                            # 情况 A: 只有主效应，直接取绝对值最大
                            max_abs_beta = raw_org_fea[rms_cols].abs().max(axis=1)
                        
                        else:
                            # 情况 B: 交互项，计算 |Main + Diff| - |Main|
                            # 必须先确定对应的主效应列是哪一组
                            if "aco" in beta_fea:
                                current_main_cols = fea_columns_aco
                            elif "pho" in beta_fea:
                                current_main_cols = fea_columns_pho
                                
                            # [CRITICAL FIX 1]: 维度检查
                            # 确保交互项列数 == 主效应列数 (例如都是 9 个 aco)
                            if len(rms_cols) != len(current_main_cols):
                                raise ValueError(f"Shape mismatch: Interaction cols ({len(rms_cols)}) vs Main cols ({len(current_main_cols)})")
                                
                            # [CRITICAL FIX 2]: 使用 .values 进行计算
                            # raw_org_fea[rms_cols] 是交互项数据 (DataFrame)
                            # raw_org[current_main_cols] 是主效应数据 (DataFrame, 注意要从 raw_org 取!)
                            # 使用 .values 强制转换成 numpy array，忽略列名差异，直接按位置相加
                            
                            diff_vals = raw_org_fea[rms_cols].values
                            main_vals = raw_org[current_main_cols].values 
                            
                            # 公式: |Main + Diff| - |Main|
                            # Nonword > Word
                            sensitivity_gain = np.abs(main_vals + diff_vals) - np.abs(main_vals)
                            # Word > Nonword
                            #sensitivity_gain = np.abs(main_vals) - np.abs(main_vals + diff_vals)
                            
                            # 取每一行(每个时间点)里增益最大的那个特征
                            max_abs_beta = np.max(sensitivity_gain, axis=1)

                        # 归一化 (可选，视你的需求而定，原代码保留)
                        n_cols = len(rms_cols)
                        raw_org_fea['rms'] = max_abs_beta / np.sqrt(n_cols)
                        all_rms_data[beta_fea] = raw_org_fea.groupby('time_point')['rms'].mean()
                        
                        # Also for data with permutations
                    fea_columns_perm = ['perm'] + fea_columns
                    raw_fea = raw[fea_columns_perm].copy()
                    
                    if group_beta_type == 'rms':
                        raw_fea['rms'] = np.sqrt(raw_fea[rms_cols].pow(2).mean(axis=1))
                        
                    elif group_beta_type == 'max':
                        # Case A: 主效应 (没有冒号)，直接取绝对值最大
                        if ":" not in beta_fea:
                            max_abs_beta = raw_fea[rms_cols].abs().max(axis=1)
                        
                        # Case B: 交互项，计算 |Main + Diff| - |Main|
                        else:
                            # 1. 确定对应的主效应列名 (aco 或 pho)
                            # 这些变量应该在之前的代码块里定义过
                            if "aco" in beta_fea:
                                current_main_cols = fea_columns_aco
                            elif "pho" in beta_fea:
                                current_main_cols = fea_columns_pho
                            
                            # 2. 提取数据矩阵 (转化为 numpy array 以忽略列名)
                            # Diff 值: 直接从当前的 raw_fea 取
                            diff_vals = raw_fea[rms_cols].values
                            
                            # Main 值: *关键步骤*
                            # 必须从原始大表 'raw' 中，根据 raw_fea 的索引 (index)，
                            # 取出对应的 Permutation 轮次下的主效应值。
                            main_vals = raw.loc[raw_fea.index, current_main_cols].values
                            
                            # 3. 维度安全检查 (防止列数对不上)
                            if diff_vals.shape[1] != main_vals.shape[1]:
                                raise ValueError(f"Shape mismatch: Diff cols {diff_vals.shape[1]} vs Main cols {main_vals.shape[1]}")
                            
                            # 4. 核心公式: |Main + Diff| - |Main|
                            # Nonword > Word
                            sensitivity_gain = np.abs(main_vals + diff_vals) - np.abs(main_vals)
                            # Word > Nonword
                            #sensitivity_gain = np.abs(main_vals) - np.abs(main_vals + diff_vals)
                            
                            # 5. 取每行的最大值 (axis=1)
                            max_abs_beta = np.max(sensitivity_gain, axis=1)

                        # 归一化
                        n_cols = len(rms_cols)
                        raw_fea['rms'] = max_abs_beta / np.sqrt(n_cols)
                        raw_fea = raw_fea[['perm', 'time_point', 'rms']]
                        pthres=[1e-2,5e-2]
                        time_point, time_series, mask_time_clus = get_traces_clus(raw_fea, pthres[0], pthres[1],mode=mode,target_fea='rms',input='R2')
                        true_indices = np.where(mask_time_clus)[0]
                        all_rms_data_sig[beta_fea] = true_indices

                    # Plotting the acoustic features from raw_org_aco
                    fig, ax = plt.subplots(figsize=(5.6*(xlim_align[1]-xlim_align[0]), 5))
                    ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7,linewidth=3)
                    add_alignment_vlines(ax, alignment)
                    if time_points_plot is None:
                        time_points_plot = sorted(raw_org_fea['time_point'].unique())

                    # Pivot the table to have time_point as index and features as columns
                    plot_cols = [col for col in fea_columns if col != 'time_point'] + ['rms']
                    raw_org_fea_plot = raw_org_fea.pivot_table(index='time_point', columns=None, values=plot_cols)

                    if ":" in beta_fea:
                        fea_cols = gp.create_gradient(feature_colors[beta_fea.split(":")[1]], raw_org_fea_plot.shape[1])[:-1]
                    else:
                        fea_cols = gp.create_gradient(feature_colors[beta_fea], raw_org_fea_plot.shape[1])[:-1]

                    for i, col in enumerate(raw_org_fea_plot.columns):
                        if col == 'rms':
                            ax.plot(time_points_plot, raw_org_fea_plot[col], label='RMS', color='grey', linewidth=4, alpha=0.8)
                        else:
                            ax.plot(time_points_plot, raw_org_fea_plot[col], label=col,color=fea_cols[i-1])

                    # ax.set_xlabel("Time (secs)")
                    # ax.set_ylabel("β")
                    ax.tick_params(axis='both', which='major')#, labelsize=16)
                    ax.ticklabel_format(
                        axis='y',
                        style='sci',
                        scilimits=(-2, -2),  # 将指数固定为 10^-3
                        useMathText=True  # 使用 LaTeX 格式显示指数，如 10⁻³
                    )
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                    # ax.legend(fontsize=18)
                    ax.set_xlim(xlim_align)#time_point.max())
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_ylim(-1e-2*fea_plot_yscale,1e-2*fea_plot_yscale)
                    plt.tight_layout()
                    plt.savefig(os.path.join('figs', 'z_score_unnormalized_huge', f'λ{vWM_lambda}',f'{elec_grp}_{alignment}_{is_vWM}_{beta_fea.replace(":", "_")}_betas.tif'), dpi=100)
                    plt.close(fig)

                # %% Plot all collected RMS traces
                # if elec_grp!='Motor' or alignment!='Aud':
                fig_rms, ax_rms = plt.subplots(figsize=(5.6*(xlim_align[1]-xlim_align[0]), 5))
                # elif elec_grp=='Motor' and alignment=='Aud':
                #     fig_rms, ax_rms = plt.subplots(figsize=(5.6*(1.4/2.15)*(xlim_align[1]-xlim_align[0]), 5))
                ax_rms.axvline(x=0, color='grey', linestyle='--', alpha=0.7,linewidth=3)
                add_alignment_vlines(ax_rms, alignment)

                j=0
                for beta_fea, rms_series in all_rms_data.items():
                    # Normalize or baseline correction
                    if alignment == 'Aud':
                        if elec_grp not in baseline_beta_rms: # Initialize inner dicts if they don't exist
                            baseline_beta_rms[elec_grp] = {}
                            baseline_beta_rms_std[elec_grp] = {}
                        baseline_beta_rms[elec_grp][beta_fea] = np.min(rms_series[(np.array(time_points_plot) > -0.2) & (np.array(time_points_plot) <= 0)])
                        baseline_beta_rms_std[elec_grp][beta_fea]=np.std(rms_series[(np.array(time_points_plot) > -0.2) & (np.array(time_points_plot) <= 0)])
                    if is_normalize:
                        rms_series = (rms_series - baseline_beta_rms[elec_grp][beta_fea]) / baseline_beta_rms_std[elec_grp][beta_fea]
                        rms_series = rms_series/ (np.max(rms_series[(np.array(time_points_plot) > xlim_align[0]) & (np.array(time_points_plot) <= xlim_align[1])]) - np.min((np.array(time_points_plot) > xlim_align[0]) & (np.array(time_points_plot) <= xlim_align[1])))
                    else:
                        if is_bsl_correct:
                            rms_series = (rms_series - baseline_beta_rms[elec_grp][beta_fea])
                    rms_series=gaussian_filter1d(rms_series, sigma=2, mode='nearest')
                    # Use the defined color for the feature, or a default color if not specified
                    color = feature_colors.get(beta_fea, '#333333')  # Default to a dark grey
                    # Use dashed lines for interaction terms
                    linestyle = '--' if ':' in beta_fea else '-'
                    ax_rms.plot(time_points_plot, rms_series, label=beta_fea, linewidth=3, color=color, linestyle=linestyle)

                    true_indices = all_rms_data_sig[beta_fea]
                    if is_vWM == '_vWM':
                        is_vWM_label = 'vWM_p'
                    true_indices_mask=true_indices_by_vWM[is_vWM_label]
                    #true_indices = np.intersect1d(true_indices, true_indices_mask)
                    if true_indices.size > 0:
                        split_points = np.where(np.diff(true_indices) != 1)[0] + 1
                        clusters_indices = np.split(true_indices, split_points)

                        for k, cluster in enumerate(clusters_indices):
                            start_index = cluster[0]
                            end_index = cluster[-1]

                            time_step = time_points_plot[1] - time_points_plot[0]
                            start_time = time_points_plot[start_index] - time_step / 2
                            end_time = time_points_plot[end_index] + time_step / 2

                            label = f'clust{k} of pho'
                            ax_rms.plot([start_time, end_time], [1e-1*fea_plot_yscale-(5e-3)*(j-1),1e-1*fea_plot_yscale-(5e-3)*(j-1)],
                                    color=color,alpha=0.4,
                                    linewidth=5,  # Make the line thick like a bar
                                    solid_capstyle='butt')  # Makes the line ends flat
                        j=j+1

                # ax_rms.set_xlabel("Time (secs)")
                # ax_rms.set_ylabel("RMS of βs")
                # ax_rms.set_title(f"Feature Set RMS ({elec_grp} - {alignment})")
                ax_rms.tick_params(axis='both', which='major')
                ax_rms.ticklabel_format(
                    axis='y',
                    style='sci',
                    scilimits=(-2, -2),
                    useMathText=True
                )
                ax_rms.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
                ax_rms.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                # ax_rms.legend(fontsize=18)
                ax_rms.set_xlim(xlim_align)
                # if elec_grp=='Motor' and alignment=='Aud':
                #     ax_rms.set_xlim([0.5,xlim_align[1]])
                ax_rms.spines['top'].set_visible(False)
                ax_rms.spines['right'].set_visible(False)
                plt.tight_layout()
                plt.savefig(os.path.join('figs', 'z_score_unnormalized_huge', f'λ{vWM_lambda}',f'{elec_grp}_{alignment}_{is_vWM}_all_rms_betas.tif'), dpi=100)
                plt.close(fig_rms)

    # %%
