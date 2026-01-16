#%% Imports and Setup
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from statsmodels.stats.multitest import multipletests
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\lme\\prepare_raw.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)

# --- Configuration ---
# 设置字体和绘图风格
font_scale = 2
plt.rcParams['font.size'] = 14 * font_scale
plt.rcParams['axes.titlesize'] = 16 * font_scale
plt.rcParams['axes.labelsize'] = 14 * font_scale
plt.rcParams['xtick.labelsize'] = 12 * font_scale
plt.rcParams['ytick.labelsize'] = 12 * font_scale
plt.rcParams['legend.fontsize'] = 12 * font_scale

# --- Colors ---
MotorPrep_col = [1.0, 0.0784, 0.5765]
Sensorimotor_col = [1, 0, 0]
Auditory_col = [0, 1, 0]
Delay_col = [1, 0.65, 0]
Motor_col = [0, 0, 1]
WGW_p55b_col = [0.74901961, 0.25098039, 0.74901961]
WGW_a55b_col = [0, 0.5, 0.5]

# Feature colors
aco_col = [0, 0.502, 0.502]      # Teal
pho_col = [0.502, 0, 0.502]      # Purple
wordness_col = [1, 0, 1]         # Magenta
sem_col = [0.2, 0.8, 0.2]        # Green

feature_colors = {
    'aco': aco_col, 'pho': pho_col, 'wordnessNonword_vWM': wordness_col,
    'wordnessNonword:aco': aco_col, 'wordnessNonword:pho': [0.5,0.5,0.5], 'sem': [0.2, 0.8, 0.2],
    'pho_word': pho_col,
    'pho_nonword': [0.7, 0.3, 0.7], # Lighter purple for non-words
    'pho_main': [0.3, 0, 0.3],     # Darker purple for main effect
    'pho_gain': Delay_col          # Orange for gain
}

# --- Feature Tags for Plotting ---
feature_tags = {
    'pho_word': 'Phonemic words',
    'pho_nonword': 'Phonemic nonwords',
    'pho_main': 'Phonemic main effects',
    'pho_gain': 'Phonemic nonword gain',
    'aco': 'Acoustic',
    'pho': 'Phonemic words',
    'sem': 'Semantic',
    'wordnessNonword:pho': 'Phonemic nonwords',
    'wordnessNonword:aco': 'Acoustic nonwords',
    'wordnessNonword_vWM': 'Lexical status'
}

# --- Helper Functions ---

def get_significance_stars(p_val):
    """Return significance stars based on p-value."""
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    else:
        return ''

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#%% Main Processing Loop

# --- Parameters ---
is_normalize = False
is_bsl_correct = False
vWM_lambda = '0.001'
mean_word_len=0.65#0.65 # from utils/lexdelay_get_stim_length.m

# Experiment Iterators
opts_yn = ['_forSilence','_yn']#,'', '_yn','_forSilence'
opts_yn_base = ['_forSilence','']#,'', '_yn','_forSilence'
opts_huge = ['onlysem',""] # 可以修改为 ['_huge', 'onlysem', 'onlysemproxy'] 等

# Time Windows definition
time_windows = [
    (0.55, 0.75), (0.75, 0.95),
]
time_labels = ["0.55-0.75", "0.75-0.95"]
n_windows = len(time_windows)

# Loop through all configuration combinations
for is_yn, is_yn_base in zip(opts_yn, opts_yn_base):
    
    unified_y_scale = 0.5 # Unified scale for combined plot
    target_beta_features = ['pho_word', 'pho_nonword', 'sem']

    # Define Electrode Groups
    all_elec_configs = [
        ('Auditory', Auditory_col, unified_y_scale),
        ('Sensorymotor', Sensorimotor_col, unified_y_scale),
        ('Motor', Motor_col, unified_y_scale),
        ('Delay_only', Delay_col, unified_y_scale),
        ('Wgw_p55b', WGW_p55b_col, unified_y_scale),
        ('Wgw_a55b', WGW_a55b_col, unified_y_scale)
    ]

    # Define Alignments (Columns)
    all_alignments = [ # Only plot Aud alignment
        ('Aud', [-0.2, 1.75])
    ] 

    # Split electrodes into chunks (3 rows per figure)
    elec_chunks = list(chunks(all_elec_configs, 3))
    
    if is_yn == "_yn":
        delay_nodelay_target = 'LexDelayRep'
        delay_nodelay_base = 'LexDelayRep'
    elif is_yn == "_forSilence":
        delay_nodelay_target = 'LexDelay'
        delay_nodelay_base = 'LexNoDelay'
    
    print(f"Processing combined plot for: {delay_nodelay_target} vs {delay_nodelay_base} | {is_yn}...")

    # --- Iterate through chunks (Figures) ---
    for chunk_idx, current_chunk in enumerate(elec_chunks):
        n_rows = len(current_chunk)
        n_cols = 3

        # Initialize Figure - electrode groups as columns
        fig_rms, axes_rms = plt.subplots(1, n_rows, figsize=(16, 7), sharey=True)
        if n_rows == 1: 
            axes_rms = [axes_rms] # Ensure axes_rms is always a list for consistent indexing
        
        # --- Iterate through Rows (Electrode Groups) ---
        for row_idx, (elec_grp, elec_col, fea_plot_yscale) in enumerate(current_chunk):
            
            # --- Iterate through Columns (Alignments) ---
            for col_idx, (alignment, xlim_align) in enumerate(all_alignments):
                ax = axes_rms[row_idx] # Index by column

                # --- Load and Merge Data from different 'is_huge' settings ---
                raw = None
                raw_base = None

                for is_huge_opt in opts_huge:
                    current_is_huge = is_huge_opt
                    if is_yn == '_forSilence' and is_huge_opt == 'onlysem':
                        current_is_huge = '_onlysem'

                    filename = f"results/{delay_nodelay_target}_{elec_grp}_{alignment}_All{is_yn}{current_is_huge}_testλ_{vWM_lambda}.csv"
                    filename_base = f"results/{delay_nodelay_base}_{elec_grp}_{alignment}_All{is_yn_base}{current_is_huge}_testλ_{vWM_lambda}.csv"

                    if not os.path.exists(filename) or not os.path.exists(filename_base):
                        continue

                    temp_raw = pd.read_csv(filename)
                    temp_raw_base = pd.read_csv(filename_base)

                    if raw is None:
                        raw = temp_raw
                        raw_base = temp_raw_base
                    else:
                        # Merge new columns, avoiding duplicates of keys
                        raw = pd.merge(raw, temp_raw.drop(columns=[c for c in temp_raw.columns if c in raw.columns and c not in ['perm', 'time_point']]), on=['perm', 'time_point'], how='outer')
                        raw_base = pd.merge(raw_base, temp_raw_base.drop(columns=[c for c in temp_raw_base.columns if c in raw_base.columns and c not in ['perm', 'time_point']]), on=['perm', 'time_point'], how='outer')

                if raw is None:
                    ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
                    continue
                
                # Split Perm 0 (Observed) and Null Perms
                raw_org = raw[raw['perm'] == 0]
                raw_org_base = raw_base[raw_base['perm'] == 0]
                
                # Dictionary to store plotting data for this subplot
                # Structure: {feature: {'means': [val1, val2...], 'pvals': [p1, p2...]}}
                bar_plot_data = {} 
                all_raw_pvals = []
                
                # --- PROCESS EACH FEATURE ---
                for beta_fea in target_beta_features:
                    is_vWM = '_vWM'
                    group_beta_type = 'max' # 'rms' or 'max' or 'avg'

                    # --- 1. Identify Columns (基于 Feature Name) ---
                    fea_columns_aco_word = [col for col in raw.columns if col.startswith('aco') and ':' not in col and is_vWM in col]
                    fea_columns_pho_word = [col for col in raw.columns if col.startswith('pho') and ':' not in col and is_vWM in col]
                    fea_columns_aco_diff = [col for col in raw.columns if 'aco' in col and ':' in col and is_vWM in col]
                    fea_columns_pho_diff = [col for col in raw.columns if 'pho' in col and ':' in col and is_vWM in col]
                    fea_columns_sem = [col for col in raw.columns if col.startswith('sem') and ':' not in col and is_vWM in col]

                    # --- 2. Define Mode and Columns ---
                    calculation_mode = None # 'simple', 'nonword', 'main', 'gain'
                    target_main_cols = []
                    target_diff_cols = []
                    
                    match beta_fea:
                        case "aco_word" | "pho_word" | "sem" | "wordnessNonword_vWM":
                            calculation_mode = 'simple'
                            if beta_fea == "aco_word": target_main_cols = fea_columns_aco_word
                            elif beta_fea == "pho_word": target_main_cols = fea_columns_pho_word
                            elif beta_fea == "sem": target_main_cols = fea_columns_sem
                            elif beta_fea == "wordnessNonword_vWM": target_main_cols = ['wordnessNonword_vWM']
                            
                        case "aco_nonword" | "pho_nonword" | "aco_main" | "pho_main" | "aco_gain" | "pho_gain":
                            if "aco" in beta_fea:
                                target_main_cols = fea_columns_aco_word
                                target_diff_cols = fea_columns_aco_diff
                            elif "pho" in beta_fea:
                                target_main_cols = fea_columns_pho_word
                                target_diff_cols = fea_columns_pho_diff
                                
                            if "nonword" in beta_fea: calculation_mode = 'nonword'
                            elif "main" in beta_fea: calculation_mode = 'main'
                            elif "gain" in beta_fea: calculation_mode = 'gain'

                    # --- 3. Calculation Functions ---

                    def get_val_matrix(df_source, main_cols, diff_cols, mode):
                        """计算原始数值矩阵 (n_samples x n_features)"""
                        main_vals = df_source[main_cols].values
                        
                        if mode == 'simple':
                            return np.abs(main_vals)
                        
                        # 对于 Interaction/Diff 相关的模式
                        diff_vals = df_source[diff_cols].values
                        
                        if mode == 'nonword':
                            return np.abs(main_vals + diff_vals)
                        elif mode == 'main':
                            return np.abs(main_vals + diff_vals) + np.abs(main_vals)
                        elif mode == 'gain':
                            return np.abs(main_vals + diff_vals) - np.abs(main_vals)
                        return np.abs(main_vals) # fallback

                    def aggregate_features(val_matrix, group_type):
                        """根据 group_beta_type 对特征维度 (axis=1) 进行聚合"""
                        if group_type == 'avg':
                            return np.mean(val_matrix, axis=1)
                        elif group_type == 'max':
                            return np.max(val_matrix, axis=1)
                        elif group_type == 'rms':
                            # Root Mean Square across features
                            return np.sqrt(np.mean(val_matrix**2, axis=1))
                        return np.mean(val_matrix, axis=1) # fallback

                    # --- 4. Process Data (Raw & Base) ---
                    
                    # 准备数据副本
                    cols_needed = ['perm', 'time_point'] + target_main_cols + target_diff_cols
                    raw_fea = raw[cols_needed].copy()
                    raw_fea_base = raw_base[cols_needed].copy()
                    
                    # A. 计算 Task (Raw)
                    mat_task = get_val_matrix(raw_fea, target_main_cols, target_diff_cols, calculation_mode)
                    val_task = aggregate_features(mat_task, group_beta_type)
                    
                    # B. 计算 Base (Baseline)
                    mat_base = get_val_matrix(raw_fea_base, target_main_cols, target_diff_cols, calculation_mode)
                    val_base = aggregate_features(mat_base, group_beta_type)

                    # --- 5. Normalize and Subtract Baseline ---
                    # 这里的 n_feats 用于最后的归一化 (根据你的参考代码逻辑)
                    n_feats = len(target_main_cols) 
                    
                    # 归一化
                    raw_fea['rms'] = val_task / np.sqrt(n_feats)
                    raw_fea_base['rms'] = val_base / np.sqrt(n_feats)
                    
                    # 减去基线 (Subtract Baseline)
                    raw_fea['rms'] = raw_fea['rms'] - raw_fea_base['rms']

                    # --- 6. Time Binning & Stats ---
                    feature_means = []
                    feature_raw_pvals = []
                    
                    for (t_start, t_end) in time_windows:
                        # Select data in window
                        mask_window = (raw_fea['time_point'] > t_start) & (raw_fea['time_point'] <= t_end)
                        data_window = raw_fea[mask_window]
                        
                        # Average RMS over time window for each permutation
                        # 注意：这里是对时间窗内的样本取平均
                        perm_means = data_window.groupby('perm')['rms'].mean()
                        
                        obs_mean = perm_means.loc[0]  # Observed mean (Perm 0)
                        null_means = perm_means.loc[1:] # Null distribution
                        
                        feature_means.append(obs_mean)
                        
                        # Calculate Two-tailed P-value
                        n_perms = len(null_means)
                        if n_perms > 0:
                            # 1. 计算右尾 (Observed >= Null) 的数量
                            n_ge = np.sum(null_means >= obs_mean)
                            
                            # 2. 计算左尾 (Observed <= Null) 的数量
                            n_le = np.sum(null_means <= obs_mean)
                            
                            # 3. 取较小的那个尾巴 (the more extreme tail)
                            min_tail_count = min(n_ge, n_le)
                            
                            # 4. 双尾公式: (2 * min_count + 1) / (N + 1)
                            # 乘以2是因为我们要看两个方向
                            p_val = (2 * min_tail_count + 1) / (n_perms + 1)
                            
                            # P值不能超过1.0
                            if p_val > 1.0: p_val = 1.0
                            
                        else:
                            p_val = 1.0
                        
                        feature_raw_pvals.append(p_val)

                    # Store results
                    all_raw_pvals.extend(feature_raw_pvals)
                    bar_plot_data[beta_fea] = {
                        'means': feature_means,
                        'pvals': [] # Placeholder, will be filled after correction
                    }

                # --- Perform Correction on ALL p-values ---
                if len(all_raw_pvals) > 0:
                    _, all_corrected_pvals, _, _ = multipletests(all_raw_pvals, alpha=0.05, method='holm')
                else:
                    all_corrected_pvals = []
                
                # --- Distribute corrected p-values back ---
                p_val_idx = 0
                for beta_fea in target_beta_features:
                    bar_plot_data[beta_fea]['pvals'] = all_corrected_pvals[p_val_idx : p_val_idx + n_windows]
                    p_val_idx += n_windows

                # --- 6. Plotting ---
                n_features = len(target_beta_features)
                bar_width = 0.8 / n_features
                indices = np.arange(n_windows)
                
                for i, beta_fea in enumerate(target_beta_features):
                    data = bar_plot_data[beta_fea]
                    means = data['means']
                    pvals = data['pvals'] # These are now FDR corrected
                    
                    # Calculate x positions for clustered bars
                    x_pos = indices + (i - n_features/2 + 0.5) * bar_width
                    color = feature_colors.get(beta_fea, '#333333')

                    # Get display label for the legend
                    plot_label = feature_tags.get(beta_fea, beta_fea)
                    
                    # Draw Bars
                    bars = ax.bar(x_pos, means, width=bar_width, label=plot_label, color=color, alpha=0.8)
                    
                    # Add Stars
                    for j, rect in enumerate(bars):
                        height = rect.get_height()
                        p_val = pvals[j]
                        star_text = get_significance_stars(p_val)
                        
                        if star_text:
                            # Adjust text position based on bar height
                            offset = 0.05 * fea_plot_yscale
                            y_text = height + 0.01 * fea_plot_yscale if height >= 0 else height - 0.05 * fea_plot_yscale
                            va = 'bottom' if height >= 0 else 'top'
                            
                            ax.text(rect.get_x() + rect.get_width()/2.0, y_text, star_text,
                                    ha='center', va=va, fontsize=16, fontweight='bold', color='black')

                # --- 7. Formatting ---
                ax.set_xticks(indices)
                ax.set_xticklabels(time_labels)
                # ax.set_xticklabels(time_labels, rotation=45, ha='right')
                
                # Adjust Y-Limits dynamically or fixed
                ax.set_ylim(-0.4*fea_plot_yscale, fea_plot_yscale)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Headers
                ax.set_title(elec_grp, fontsize=18, fontweight='bold', pad=20)
                
                # Row Labels
                if row_idx == 0:
                    ax.set_ylabel("Avg Beta Gain", fontsize=16, fontweight='bold')
                
                # X Labels
                ax.set_xlabel("Time Window (s)", fontsize=14, fontweight='bold')
                
                # Legend (Top Left of Third Column)
                if row_idx == 2:
                    ax.legend(loc='upper right', frameon=False, fontsize=14)

        # --- Save Figure ---
        fig_rms.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent overlap
        save_dir = os.path.join('figs', f'diff_combined{is_yn}_{delay_nodelay_base}')
        os.makedirs(save_dir, exist_ok=True)
        
        save_name = os.path.join(save_dir, f'BigPlot_Part{chunk_idx+1}_RMS_Bar_FDR_vWMλ_{vWM_lambda}.tif')
        
        plt.savefig(save_name, dpi=100)
        plt.close(fig_rms)
        print(f"Saved: {save_name}")

print("Done.")
# %%
