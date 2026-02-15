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
import itertools

#%% Run time cluster
import tkinter as tk
from tkinter import messagebox

def show_warning_popup():
    # 初始化一个隐藏的根窗口（否则会多出一个空白方框）
    root = tk.Tk()
    root.withdraw() 
    
    # 弹出警告框
    messagebox.showwarning(title="分析警告", message="请使用T值画acc trace！")
    
    # 销毁窗口资源
    root.destroy()

# --- 在你需要的地方调用它 ---
#show_warning_popup()

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

def get_traces_clus(raw, alpha:float=0.05, alpha_clus:float=0.05, mode:str='time_cluster', target_fea:str=None, input:str='R2', calc_start_time:float=-0.25):
    """
    Args:
        calc_start_time (float): Only calculate stats for time points >= this value. 
                                 Time points before this will be set to non-significant (0).
    """
    # Load data
    time_point = np.unique(raw['time_point'].to_numpy())
    
    if target_fea is None:
        r2s_i_df = raw.pivot_table(
            index='perm',
            columns='time_point',
            values='chi_squared_obs'
        )
    elif isinstance(target_fea, str):
        raw_temp = raw.copy()
        raw_temp['abs_target_fea'] = raw_temp[target_fea]

        r2s_i_df = raw_temp.pivot_table(
            index='perm',
            columns='time_point',
            values='abs_target_fea'
        )
    elif isinstance(target_fea, list):
        abs_features = raw[target_fea]
        mean_abs_feature = abs_features.mean(axis=1)

        raw_temp = raw.copy()
        raw_temp['mean_abs_features'] = mean_abs_feature

        r2s_i_df = raw_temp.pivot_table(
            index='perm',
            columns='time_point',
            values='mean_abs_features'
        )
    
    r2s_i = r2s_i_df.to_numpy()
    
    # Extract the full time series for output/plotting
    r2_i_full = r2s_i[0, :]

    # --- Slicing Logic for Calculation ---
    if calc_start_time is not None:
        # Find indices where time is within the window
        valid_indices = np.where(time_point >= calc_start_time)[0]
    else:
        valid_indices = np.arange(len(time_point))
    
    # Initialize the full statistical output mask as all False (0)
    stat_out_full = np.zeros(len(time_point), dtype=int)
    
    # If no time points match the condition, return early
    if len(valid_indices) == 0:
        return time_point, r2_i_full, stat_out_full

    # Slice the data to include only the valid time window
    r2s_i_calc = r2s_i[:, valid_indices]
    r2_i_calc = r2s_i_calc[0, :]
    r2_i_calc = np.expand_dims(r2_i_calc, axis=0) # Shape: (1, n_time_calc)

    if input == 'R2':
        null_r2_i_calc = r2s_i_calc[1:, :]

        # Get original mask (on sliced data)
        org_p_i_calc = glm.aaron_perm_gt_1d(r2s_i_calc, axis=0)[0] # 1-d time series
        mask_i_org_calc = (org_p_i_calc > (1 - alpha)).astype(int) # 1-d time series (binary)

        # Get null mask (on sliced data)
        null_p_i_calc = glm.aaron_perm_gt_1d(null_r2_i_calc, axis=0) # 2-d time series: n_perm*time
        mask_null_i_calc = (null_p_i_calc > (1 - alpha)).astype(int) # 2-d time series (binary)

        if mode == 'time_cluster':
            # Time perm cluster on sliced data
            stat_out_calc = time_cluster(mask_i_org_calc, mask_null_i_calc, 1 - alpha_clus)
        elif mode == 'fdr':
            # FDR on sliced data
            stat_out_calc, p_fdr, _, _ = multipletests(1 - org_p_i_calc, alpha=alpha_clus, method='holm')

    elif input == 'p':
        # Assuming r2_i contains p-values directly
        stat_out_calc, p_fdr, _, _ = multipletests(r2_i_calc[0], alpha=alpha_clus, method='holm')
    
    # Fill the calculated stats back into the full array at the correct positions
    stat_out_full[valid_indices] = stat_out_calc.astype(int)

    return time_point, r2_i_full, stat_out_full

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
        ax.axvline(x=0.59, color=[138/255,12/255,0/255], linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=0.59 + 0.1480 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=0.59 - 0.1480 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # Delay offsets
        ax.axvline(x=1.5, color=[138/255,12/255,0/255], linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=1.7201 + 0.1459 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=1.7201 - 0.1459 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
    elif alignment == 'Go':
        # Motor onsets
        ax.axvline(x=0.7961, color=[138/255,12/255,0/255], linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=0.7961 + 1.0532 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
        # ax.axvline(x=0.7961 - 1.0532 / 2, color='red', linestyle='--', alpha=0.7, linewidth=1)
    elif alignment == 'Resp':
        # Motor offsets
        ax.axvline(x=0.6096, color=[138/255,12/255,0/255], linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0.6096 + 0.2190, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0.6096 - 0.2190, color='red', linestyle='--', alpha=0.7, linewidth=1)

#%% Plotting & Big Figure Generation
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d
import utils.group as gp

# --- 基础配置 ---
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# 常用颜色定义
MotorPrep_col = [1.0, 0.0784, 0.5765] 
Sensorimotor_col = [1, 0, 0]  
Auditory_col = [0, 1, 0]  
Delay_col = [1, 0.65, 0]  
Motor_col = [0, 0, 1]  
WGW_p55b_col=[0.74901961, 0.25098039, 0.74901961] 
WGW_a55b_col=[0, 0.5, 0.5] 
aco_col = [0, 0.502, 0.502]      
pho_col = [0.502, 0, 0.502]      
wordness_col = [1, 0, 1] 

is_normalize = True
is_bsl_correct = True
mode = 'time_cluster'

# 定义所有要画的电极组及其对应参数
opts_yn = ['', '_yn']#,'', '_yn','_forSilence'
opts_huge = ['','onlysem']#'onlysemproxy', '_huge', 'onlysem']
delay_nodelays = ["LexDelayRep",] #"LexDelay","LexNoDelay","LexDelayRep"

for is_yn, is_huge,delay_nodelay in itertools.product(opts_yn, opts_huge,delay_nodelays):
    # --- 根据 is_huge 设定统一的 Y-Scale (使用 match case) ---
    match is_huge:
        case '_onlysem':
            unified_y_scale = 0.3  #
        case 'onlysem':
            unified_y_scale = 10  #
        case 'onlysemproxy':
            unified_y_scale = 10 
        case '_huge':
            unified_y_scale = 2   # [请调节] _huge
        case '':
            unified_y_scale = 0.3   # [请调节] 默认值 (类似 else)

    # --- 定义所有要画的电极组 (使用统一的 Scale) ---
    all_elec_configs = [
        # (Name, Color, Y-Scale)
        ('Auditory', Auditory_col, unified_y_scale),
        ('Sensorymotor', Sensorimotor_col, unified_y_scale),
        ('Motor', Motor_col, unified_y_scale),
        ('Delay_only', Delay_col, unified_y_scale),
        ('Wgw_p55b', WGW_p55b_col, unified_y_scale),
        ('Wgw_a55b', WGW_a55b_col, unified_y_scale)
    ]

    # [修改] 定义列（Alignment）及其参数 - 现在 Go 在第二列，Resp 在第三列
    all_alignments = [
        # (Name, X-Lim)
        ('Aud', [-0.2, 1.75])
    ]

    # 计算每一列的时间长度，作为宽度的比例
    width_ratios = [xlim[1] - xlim[0] for _, xlim in all_alignments]

    # 全局参数
    baseline = dict()
    baseline_beta_rms = dict()
    baseline_std = dict()
    baseline_beta_rms_std = dict()
    vWM_lambda = '0.001'

    # 辅助函数
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # 将电极组按 3 个一组进行切分
    elec_chunks = list(chunks(all_elec_configs, 4))

    # =============================================================================
    # 1. 绘制 ACC (R2) 大图
    # =============================================================================
    print("Starting ACC Big Plots...")
    for chunk_idx, current_chunk in enumerate(elec_chunks):
        n_rows = 1
        n_cols = len(current_chunk)
        
        # 设置图幅
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5 * n_cols, 5 * n_rows), 
                                sharey=True)
        
        if n_cols == 1: axes = [axes]
        
        for col_idx, (elec_grp, elec_col, fea_plot_yscale) in enumerate(current_chunk):
            ax = axes[col_idx]
            para_sig_barbar = [0.1, 0.007] # Reset for each subplot
            
            for _, (alignment, xlim_align) in enumerate(all_alignments):
                    
                # --- 绘图基础设置 ---
                ax.axvline(x=0, color='grey', linestyle='--', alpha=0.7, linewidth=3)
                add_alignment_vlines(ax, alignment)
                
                # --- 加载数据 ---
                filename = f"results/{delay_nodelay}_{elec_grp}_{alignment}_All{is_yn}{is_huge}_testλ_{vWM_lambda}.csv"
                if not os.path.exists(filename):
                    print(f"Warning: File not found {filename}")
                    continue
                raw = pd.read_csv(filename)
                
                # --- 统计与绘图逻辑 (ACC) ---
                fea = 'ACC'
                j = 0
                true_indices_by_vWM = {}
                for vWM, vwm_linestyle, input_type, pthres, vwm_text, sig_bar_col in zip(
                        ('vWM', 'vWM_p'), ('-', '-'), ('R2', 'p'),
                        ([5e-2, 1e-4], [1e-3, 1e-20]), ('ACC', 'p'),
                        (elec_col, elec_col)):
                    
                    target_fea = fea + f'_{vWM}'
                    time_point, time_series, mask_time_clus = get_traces_clus(
                        raw, pthres[0], pthres[1], mode=mode, target_fea=target_fea, input=input_type)
                    
                    time_series = gaussian_filter1d(time_series, sigma=2, mode='nearest')
                    
                    if alignment == 'Aud':
                        baseline[elec_grp] = np.min(time_series[(time_point > -0.2) & (time_point <= 0)])
                        baseline_std[elec_grp] = np.std(time_series[(time_point > -0.2) & (time_point <= 0)])
                    
                    if is_normalize:
                        b_val = baseline.get(elec_grp, 0)
                        b_std = baseline_std.get(elec_grp, 1)
                        time_series = (time_series - b_val) / b_std
                        denom = (np.max(time_series[(time_point > xlim_align[0]) & (time_point <= xlim_align[1])]) - 
                                np.min((time_point > xlim_align[0]) & (time_point <= xlim_align[1])))
                        if denom != 0: time_series = time_series / denom
                        para_sig_bar = [1, 1e-1]
                    else:
                        if is_bsl_correct:
                            b_val = baseline.get(elec_grp, 0)
                            time_series = (time_series - b_val)
                        para_sig_bar = para_sig_barbar

                    if vWM == 'vWM':
                        ax.plot(time_point, time_series, label=f"{elec_grp}{vwm_text}", 
                                color=elec_col, linewidth=5, linestyle=vwm_linestyle)
                    
                    true_indices = np.where(mask_time_clus)[0]
                    true_indices_by_vWM[vWM] = true_indices
                    if true_indices.size > 0 and (vWM == 'vWM_p'):
                        split_points = np.where(np.diff(true_indices) != 1)[0] + 1
                        clusters_indices = np.split(true_indices, split_points)
                        for k, cluster in enumerate(clusters_indices):
                            start_time = time_point[cluster[0]] - (time_point[1]-time_point[0])/2
                            end_time = time_point[cluster[-1]] + (time_point[1]-time_point[0])/2
                            ax.plot([start_time, end_time], 
                                    [para_sig_bar[0]-para_sig_bar[1]*(j-1), para_sig_bar[0]-para_sig_bar[1]*(j-1)],
                                    color=sig_bar_col, alpha=0.4, linewidth=10, solid_capstyle='butt')
                        j += 1
                    
                    # --- 坐标轴修饰 ---
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, -2), useMathText=True)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax.set_xlim(xlim_align)
                    ax.set_ylim(-0.002, para_sig_bar[0] + 2 * para_sig_bar[1])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Title for each subplot is the electrode group
                    ax.set_title(elec_grp, fontsize=16, fontweight='bold', pad=10)
                    
                    # YLabel only for the first plot
                    if col_idx == 0:
                        ax.set_ylabel("ACC", fontsize=16, fontweight='bold')

                    # X 轴标签
                    ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
                        
        fig.suptitle(f"ACC Aligned to {all_alignments[0][0]}", fontsize=20, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        save_name = os.path.join('figs', f'z_score_unnormalized{is_yn}{is_huge}', 
                                f'BigPlot_ACC_Part{chunk_idx+1}_vWMλ_{vWM_lambda}.tif')
        plt.savefig(save_name, dpi=100)
        plt.close()


    # =============================================================================
    # 2. 绘制 Beta (Feature) 和 RMS 大图
    # =============================================================================
    print("Starting Beta and RMS Big Plots...")

    group_beta_type = 'max' # 'rms' or 'max' or 'avg'
    match is_huge:
        case '_onlysem':
            target_beta_features = ['sem'] 
        case 'onlysem':
            target_beta_features = ['sem'] 
        case 'onlysemproxy':
            target_beta_features = ['sem'] 
        case '_huge':
            #target_beta_features = ['aco', 'pho', 'wordnessNonword:pho', 'sem']
            target_beta_features = ['aco_main', 'pho_main', 'sem']  
        case _:
            target_beta_features = ['pho_word','pho_nonword','pho_gain']
            #target_beta_features = ['aco_main','pho_main','pho_interact']
            #target_beta_features = ['aco_main','pho_main','sem']

    feature_colors = {
        'aco_main': [243/255,65/255,162/255], 'pho_main': [138/255,12/255,0/255], 'wordnessNonword_vWM': wordness_col,
        'wordnessNonword:aco': aco_col, 'wordnessNonword:pho': [0.5,0.5,0.5], 'sem': [101/255, 69/255, 1],
        'pho_word': [138/255,12/255,0/255],
        'pho_nonword': [1, 108/255, 93/255],  # Lighter purple for non-words
        'pho_gain': Delay_col,          # Orange for gain
        'aco_interact': [0.7, 0.7, 0.7], # Light grey
        'pho_interact': [0.4, 0.4, 0.4]  # Dark grey
    }

    feature_tags = {
    'pho_word': 'Phonemic words',
    'pho_nonword': 'Phonemic nonwords',
    'pho_main': 'Phonemic main effects',
    'aco_main': 'Acoustic main effects',
    'pho_gain': 'Phonemic nonword gain',
    'aco_interact': 'Aco Lex interaction',
    'pho_interact': 'Pho Lex interaction',
    'aco': 'Acoustic',
    'pho': 'Phonemic words',
    'sem': 'Semantic',
    'wordnessNonword:pho': 'Phonemic nonwords',
    'wordnessNonword:aco': 'Acoustic nonwords',
    'wordnessNonword_vWM': 'Lexical status'
    }

    for chunk_idx, current_chunk in enumerate(elec_chunks):
        n_rows = 1
        n_cols = len(current_chunk)
        
        # 1. RMS 画布
        fig_rms, axes_rms = plt.subplots(n_rows, n_cols, 
                                        figsize=(5 * n_cols, 5 * n_rows), 
                                        sharey=True)
        if n_cols == 1: axes_rms = [axes_rms]
        
        # 2. Beta 画布字典
        figs_beta = {}
        axes_beta = {}
        for fea in target_beta_features:
            f, a = plt.subplots(n_rows, n_cols, 
                                figsize=(5 * n_cols, 5 * n_rows), 
                                sharey=True)
            if n_cols == 1: a = [a]
            figs_beta[fea] = f
            axes_beta[fea] = a

        # if is_huge!='':
        #     continue
        for col_idx, (elec_grp, elec_col, fea_plot_yscale) in enumerate(current_chunk):
            for _, (alignment, xlim_align) in enumerate(all_alignments):

                    raw = None
                    for is_huge_opt in opts_huge:
                        filename = f"results/{delay_nodelay}_{elec_grp}_{alignment}_All{is_yn}{is_huge_opt}_testλ_{vWM_lambda}.csv"
                        if not os.path.exists(filename):
                            print(f"Warning: File not found {filename}")
                            continue

                        temp_raw = pd.read_csv(filename)
                        if raw is None:
                            raw = temp_raw
                        else:
                            # Merge new columns, avoiding duplicates of keys
                            raw = pd.merge(raw, temp_raw.drop(columns=[c for c in temp_raw.columns if c in raw.columns and c not in ['perm', 'time_point']]), on=['perm', 'time_point'], how='outer')

                    if raw is None:
                        continue
                    
                    all_rms_data = {} 
                    all_rms_data_sig = {}

                    raw_org=raw.copy()
                    raw_org = raw_org[raw_org['perm'] == 0]
                    time_points_plot = sorted(raw_org['time_point'].unique())
                    
                    process_features = target_beta_features 
                    
                    for beta_fea in process_features:
                        is_vWM = '_vWM'
                        
                        fea_columns_aco_word = [col for col in raw.columns if col.startswith('aco') and ':' not in col and is_vWM in col]
                        fea_columns_pho_word = [col for col in raw.columns if col.startswith('pho') and ':' not in col and is_vWM in col]
                        fea_columns_aco_diff = [col for col in raw.columns if 'aco' in col and ':' in col and is_vWM in col]
                        fea_columns_pho_diff = [col for col in raw.columns if 'pho' in col and ':' in col and is_vWM in col]
                        fea_columns_sem = [col for col in raw.columns if col.startswith('sem') and ':' not in col and is_vWM in col]


                        match beta_fea:
                            case "aco_word" | "pho_word" | "sem" | "wordnessNonword_vWM":
                                # simple effect of word
                                if beta_fea == "aco_word":
                                    fea_columns = fea_columns_aco_word
                                elif beta_fea == "pho_word":
                                    fea_columns = fea_columns_pho_word
                                elif beta_fea == "sem":
                                    fea_columns = fea_columns_sem
                                elif beta_fea == "wordnessNonword_vWM":
                                    fea_columns = ["wordnessNonword_vWM"]
                                raw_fea = raw[['perm']+['time_point'] + fea_columns].copy()
                                if group_beta_type == 'avg':
                                    val = raw_fea[fea_columns].abs().mean(axis=1)
                                elif group_beta_type == 'max':
                                    val = raw_fea[fea_columns].abs().max(axis=1)
                                elif group_beta_type == 'rms':
                                    val = np.sqrt(raw_fea[fea_columns].pow(2).mean(axis=1))
                            case "aco_nonword" | "pho_nonword" | "aco_main" | "pho_main" | "aco_gain" | "pho_gain" | "aco_interact" | "pho_interact":
                                if "aco" in beta_fea:
                                    fea_columns = fea_columns_aco_word
                                    fea_columns_diff = fea_columns_aco_diff
                                elif "pho" in beta_fea:
                                    fea_columns = fea_columns_pho_word
                                    fea_columns_diff = fea_columns_pho_diff
                                raw_fea_word = raw[['perm']+['time_point'] + fea_columns].copy()
                                raw_fea = raw_fea_word.copy()
                                raw_fea_diff = raw[['perm']+['time_point'] + fea_columns_diff].copy()
                                word_vals = raw_fea_word[fea_columns].values
                                diff_vals = raw_fea_diff[fea_columns_diff].values
                                if "nonword" in beta_fea:
                                    # simple effect of nonword
                                    sensitivity_gain = np.abs(word_vals + diff_vals)
                                elif "main" in beta_fea:
                                    # main effect
                                    sensitivity_gain = np.abs(word_vals + diff_vals) + np.abs(word_vals)
                                elif "gain" in beta_fea:
                                    # gain of nonword - word
                                    sensitivity_gain = np.abs(word_vals + diff_vals) - np.abs(word_vals) 
                                elif "interact" in beta_fea:
                                    # interaction effect
                                    sensitivity_gain = diff_vals       
                                if group_beta_type == 'avg':
                                    val = np.mean(sensitivity_gain, axis=1)
                                elif group_beta_type == 'max':  
                                    val = np.max(sensitivity_gain, axis=1)
                                elif group_beta_type == 'rms':
                                    val = sensitivity_gain


                        raw_fea['rms'] = val / np.sqrt(len(fea_columns)-1)
                        all_rms_data[beta_fea] = raw_fea.loc[raw_fea['perm'] == 0].groupby('time_point')['rms'].mean()

                        # Stats
                        raw_fea = raw_fea[['perm', 'time_point', 'rms']]
                        # For max:
                        pthres = [1e-2, 2.5e-2]
                        # For average:
                        # pthres = [2e-3, 5e-2]
                        _, _, mask_time_clus = get_traces_clus(raw_fea, pthres[0], pthres[1], mode=mode, target_fea='rms', input='R2')
                        all_rms_data_sig[beta_fea] = np.where(mask_time_clus)[0]

                        # --- 绘制单特征 Beta 图 ---
                        if 1==0 and beta_fea in target_beta_features:
                            ax_b = axes_beta[beta_fea][col_idx]
                            ax_b.axvline(x=0, color='grey', linestyle='--', alpha=0.7, linewidth=3)
                            add_alignment_vlines(ax_b, alignment)
                            
                            plot_cols = [col for col in fea_columns if col != 'time_point'] + ['rms']
                            raw_org_fea_plot = raw_fea.loc[raw_fea['perm'] == 0].pivot_table(index='time_point', values=plot_cols)
                            
                            if ":" in beta_fea:
                                fea_cols_grad = gp.create_gradient(feature_colors[beta_fea.split(":")[1]], raw_org_fea_plot.shape[1])[:-1]
                            else:
                                fea_cols_grad = gp.create_gradient(feature_colors[beta_fea], raw_org_fea_plot.shape[1])[:-1]

                            for i_col, col_name in enumerate(raw_org_fea_plot.columns):
                                if col_name == 'rms':
                                    ax_b.plot(time_points_plot, raw_org_fea_plot[col_name], label='RMS', color='grey', linewidth=4, alpha=0.8)
                                else:
                                    c_idx = i_col - 1 if i_col > 0 else 0 
                                    if c_idx >= len(fea_cols_grad): c_idx = -1
                                    ax_b.plot(time_points_plot, raw_org_fea_plot[col_name], color=fea_cols_grad[c_idx])
                            
                            ax_b.ticklabel_format(axis='y', style='sci', scilimits=(-2, -2), useMathText=True)
                            ax_b.set_xlim(xlim_align)
                            ax_b.set_ylim(-1e-2 * fea_plot_yscale, 1e-2 * fea_plot_yscale)
                            ax_b.spines['top'].set_visible(False)
                            ax_b.spines['right'].set_visible(False)

                            ax_b.set_title(elec_grp, fontsize=16, fontweight='bold', pad=10)
                            if col_idx == 0:
                                ax_b.set_ylabel("beta", fontsize=16, fontweight='bold')
                                        
                            ax_b.set_xlabel("Time (s)", fontsize=14, fontweight='bold')

                    # --- 绘制 RMS 图 ---
                    ax_r = axes_rms[col_idx]
                    
                    # 1. 基础参考线优化
                    ax_r.axvline(x=0, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8, zorder=0)
                    ax_r.axhline(y=0, color='#DDDDDD', linewidth=1, zorder=0)
                    
                    add_alignment_vlines(ax_r, alignment)
                    
                    j_sig = 0
                    for beta_fea, rms_series in all_rms_data.items():
                        # ... (数值计算与平滑部分保持不变) ...
                        rms_series = gaussian_filter1d(rms_series, sigma=4, mode='nearest')
                        color = feature_colors.get(beta_fea, '#333333')
                        linestyle = '--' if ':' in beta_fea else '-'
                        plot_label = feature_tags.get(beta_fea, beta_fea)
                        
                        # 2. 主线条绘制
                        if beta_fea != 'pho_gain':
                            ax_r.plot(time_points_plot, rms_series, linewidth=2.5, 
                                    color=color, linestyle=linestyle, alpha=0.9, 
                                    solid_capstyle='round', zorder=2)
                        
                        # 3. 显著性条绘制
                        true_indices = all_rms_data_sig[beta_fea]
                        if true_indices.size > 0:
                            split_points = np.where(np.diff(true_indices) != 1)[0] + 1
                            clusters = np.split(true_indices, split_points)
                            for i, clus in enumerate(clusters):
                                s_time = time_points_plot[clus[0]]
                                e_time = time_points_plot[clus[-1]]
                                sig_y = fea_plot_yscale - (0.05 * fea_plot_yscale * j_sig)
                                ax_r.plot([s_time, e_time], [sig_y, sig_y],
                                        color=color, alpha=0.6, 
                                        label=plot_label if i == 0 else None,
                                        linewidth=4, solid_capstyle='butt', zorder=1)
                            j_sig += 1

                    # --- 子图坐标轴细节控制 ---
                    
                    # 4. 基础解构
                    ax_r.spines[['top', 'right']].set_visible(False)
                    ax_r.spines['bottom'].set_position(('outward', 10))
                    
                    # 5. 根据位置决定是否保留 Y 轴
                    if col_idx == 0:
                        # 第一幅图保留 Y 轴及标签
                        ax_r.spines['left'].set_visible(True)
                        ax_r.spines['left'].set_position(('outward', 10))
                        ax_r.set_ylabel(f"{group_beta_type.upper()} Beta", fontsize=14, fontweight='medium', labelpad=10)
                        ax_r.tick_params(axis='y', which='major', direction='out', length=5, width=1.2, labelsize=12, pad=5,labelleft=True)
                    else:
                        # 第二、三幅图彻底隐藏 Y 轴
                        ax_r.spines['left'].set_visible(False)
                        ax_r.set_ylabel('')
                        ax_r.set_yticklabels([])
                        ax_r.tick_params(axis='y', left=False) # 隐藏刻度线
                    
                    # 6. X 轴通用设置
                    ax_r.tick_params(axis='x', which='major', direction='out', length=5, width=1.2, labelsize=12, pad=5)
                    ax_r.set_xlabel("Time (s)", fontsize=14, fontweight='medium', labelpad=10)
                    ax_r.set_xlim(xlim_align)
                    ax_r.set_ylim(-0.05 * fea_plot_yscale, fea_plot_yscale * 1.1)
                    ax_r.set_title(elec_grp, fontsize=16, fontweight='bold', pad=20)

        # --- 全局图例与保存设置 ---
        
        # 7. 获取图例句柄 (从包含 label 的子图中获取)
        handles, labels = axes_rms[1].get_legend_handles_labels()
        
        # 8. 图例放置在右侧外部，完全不重叠
        # loc='center left' 配合 bbox_to_anchor=(1.02, 0.5) 
        # fig_rms.legend(handles, labels, loc='center left', frameon=False, 
        #             fontsize=11, ncol=1, bbox_to_anchor=(0.85, 0.5))

        # 9. 调整布局，rect 参数为图例留出右侧空间 (0.8 意味着绘图区只占左侧 80%)
        plt.tight_layout(rect=[0.05, 0, 0.85, 0.95])

        # 保存设置
        save_dir_rms = os.path.join('figs', 'publication_quality_plots')
        os.makedirs(save_dir_rms, exist_ok=True)
        save_name_rms = os.path.join(save_dir_rms, f'RMS_Dynamics_vWM_{vWM_lambda}.tif')
        plt.savefig(save_name_rms, dpi=300) 
        plt.close(fig_rms)
        
        for fea, f in figs_beta.items():
            f.suptitle(f"Beta for {fea} Aligned to {all_alignments[0][0]}", fontsize=20, fontweight='bold')
            plt.figure(f.number)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_name_beta = os.path.join('figs', f'z_score_unnormalized{is_yn}{is_huge}', 
                                        f'BigPlot_Beta_Part{chunk_idx+1}_{fea.replace(":", "_")}.tif')
            plt.savefig(save_name_beta, dpi=100)
            plt.close(f)

    print("All plots finished.")
# %%
