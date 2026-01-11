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
    'aco': aco_col, 
    'pho': pho_col, 
    'wordnessNonword_vWM': wordness_col,
    'wordnessNonword:aco': aco_col, 
    'wordnessNonword:pho': [0.5, 0.5, 0.5], 
    'sem': sem_col
}

# --- Feature Tags for Plotting ---
feature_tags = {
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
for (is_yn, is_yn_base), is_huge in itertools.product(zip(opts_yn, opts_yn_base), opts_huge):
    
    if is_yn == '_forSilence' and is_huge == 'onlysem':
        is_huge = '_onlysem'

    # 1. Determine Y-Axis Scale based on analysis mode
    match is_huge:
        case '_onlysem': unified_y_scale = 0.3 
        case 'onlysem': unified_y_scale = 0.5 
        case 'onlysemproxy': unified_y_scale = 10 
        case '_huge': unified_y_scale = 10 
        case _: unified_y_scale = 0.5

    # 2. Determine Features to plot
    match is_huge:
        case '_onlysem' | 'onlysem' | 'onlysemproxy': 
            target_beta_features = ['sem']
        case '_huge': 
            target_beta_features = ['aco', 'pho', 'wordnessNonword:pho', 'sem']
        case _: 
            target_beta_features = ['pho', 'wordnessNonword:pho']

    # 3. Define Electrode Groups
    all_elec_configs = [
        ('Auditory', Auditory_col, unified_y_scale),
        ('Sensorymotor', Sensorimotor_col, unified_y_scale),
        ('Motor', Motor_col, unified_y_scale),
        ('Delay_only', Delay_col, unified_y_scale),
        ('Wgw_p55b', WGW_p55b_col, unified_y_scale),
        ('Wgw_a55b', WGW_a55b_col, unified_y_scale)
    ]

    # 4. Define Alignments (Columns)
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
    
    print(f"Processing: {delay_nodelay_target} vs {delay_nodelay_base} | {is_huge} | {is_yn}...")

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
                
                # Construct Filenames
                filename = f"results/{delay_nodelay_target}_{elec_grp}_{alignment}_All{is_yn}{is_huge}_testλ_{vWM_lambda}.csv"
                filename_base = f"results/{delay_nodelay_base}_{elec_grp}_{alignment}_All{is_yn_base}{is_huge}_testλ_{vWM_lambda}.csv"
                
                # Skip if files don't exist
                if not os.path.exists(filename) or not os.path.exists(filename_base):
                    ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
                    continue

                # Load Data
                raw = pd.read_csv(filename)
                raw_base = pd.read_csv(filename_base)
                
                # Split Perm 0 (Observed) and Null Perms
                raw_org = raw[raw['perm'] == 0]
                raw_org_base = raw_base[raw_base['perm'] == 0]
                
                # Dictionary to store plotting data for this subplot
                # Structure: {feature: {'means': [val1, val2...], 'pvals': [p1, p2...]}}
                bar_plot_data = {} 
                
                # --- PROCESS EACH FEATURE ---
                for beta_fea in target_beta_features:
                    is_vWM = '_vWM'
                    
                    # 1. Identify Columns based on Feature Name
                    fea_columns_aco = [col for col in raw.columns if col.startswith('aco') and ':' not in col and is_vWM in col]
                    fea_columns_pho = [col for col in raw.columns if col.startswith('pho') and ':' not in col and is_vWM in col]
                    fea_columns_sem = [col for col in raw.columns if col.startswith('sem') and ':' not in col and is_vWM in col]

                    if beta_fea == "aco": fea_columns = ['time_point'] + fea_columns_aco
                    elif beta_fea == "pho": fea_columns = ['time_point'] + fea_columns_pho
                    elif beta_fea == "wordnessNonword_vWM": fea_columns = ['time_point'] + ['wordnessNonword_vWM']
                    elif beta_fea == "sem": fea_columns = ['time_point'] + fea_columns_sem
                    elif (":" in beta_fea) and ("aco" in beta_fea):
                        fea_columns = ['time_point'] +['aco1:wordnessNonword_vWM']+ [col for col in raw.columns if col.startswith(beta_fea) and is_vWM in col]
                    elif (":" in beta_fea) and ("pho" in beta_fea):
                        fea_columns = ['time_point'] + [col for col in raw.columns if col.startswith(beta_fea) and is_vWM in col]
                    
                    rms_cols = [col for col in fea_columns if col != 'time_point']

                    # --- 2. Calculate RMS for Perm 0 (Observed) ---
                    # We compute the "Sensitivity Gain" logic (Avg Beta approach)
                    raw_org_fea = raw_org[fea_columns].copy()
                    raw_org_fea_base = raw_org_base[fea_columns].copy()

                    # Logic for difference/interaction terms vs main terms
                    if (":" in beta_fea) or (beta_fea == 'pho') or (beta_fea == 'aco'):
                        if "aco" in beta_fea: current_main_cols = fea_columns_aco
                        elif "pho" in beta_fea: current_main_cols = fea_columns_pho
                        
                        diff_vals = raw_org_fea[rms_cols].values
                        diff_vals_base = raw_org_fea_base[rms_cols].values
                        main_vals = raw_org[current_main_cols].values
                        main_vals_base = raw_org_base[current_main_cols].values

                        # "avg" logic for gain calculation
                        if (beta_fea == 'pho') or (beta_fea == 'aco'):
                            sensitivity_gain = np.abs(main_vals + diff_vals) + np.abs(main_vals)
                            sensitivity_gain_base = np.abs(main_vals_base + diff_vals_base) + np.abs(main_vals_base)
                        else:
                            sensitivity_gain = np.abs(main_vals + diff_vals) - np.abs(main_vals)
                            sensitivity_gain_base = np.abs(main_vals_base + diff_vals_base) - np.abs(main_vals_base)
                        
                        val = np.mean(sensitivity_gain, axis=1)
                        val_base = np.mean(sensitivity_gain_base, axis=1)
                    else:
                        val = raw_org_fea[rms_cols].abs().mean(axis=1)
                        val_base = raw_org_fea_base[rms_cols].abs().mean(axis=1)

                    # Normalize by sqrt(n)
                    raw_org_fea['rms'] = val / np.sqrt(len(rms_cols))
                    raw_org_fea_base['rms'] = val_base / np.sqrt(len(rms_cols))
                    # Subtract Baseline
                    raw_org_fea['rms'] = raw_org_fea['rms'] - raw_org_fea_base['rms']

                    # --- 3. Calculate RMS for ALL Perms (Null Distribution) ---
                    # Reuse columns but include 'perm'
                    fea_columns_perm = ['perm'] + fea_columns
                    raw_fea = raw[fea_columns_perm].copy()
                    raw_fea_base = raw_base[fea_columns_perm].copy()
                    
                    if (":" in beta_fea) or (beta_fea == 'pho') or (beta_fea == 'aco'):
                         if "aco" in beta_fea: current_main_cols = fea_columns_aco
                         elif "pho" in beta_fea: current_main_cols = fea_columns_pho
                         
                         diff_vals_p = raw_fea[rms_cols].values
                         diff_vals_p_base = raw_fea_base[rms_cols].values
                         main_vals_p = raw.loc[raw_fea.index, current_main_cols].values
                         main_vals_p_base = raw_base.loc[raw_fea_base.index, current_main_cols].values
                         
                         if (beta_fea == 'pho') or (beta_fea == 'aco'):
                             # simple effect of words
                            sg_p = np.abs(main_vals_p)
                            sg_p_base = np.abs(main_vals_p_base)
                             # main effect of words + nonwords
                            #  sg_p = np.abs(main_vals_p + diff_vals_p) + np.abs(main_vals_p)
                            #  sg_p_base = np.abs(main_vals_p_base + diff_vals_p_base) + np.abs(main_vals_p_base)
                         else:
                             # main effect of nonwords
                             sg_p = np.abs(main_vals_p + diff_vals_p)
                             sg_p_base = np.abs(main_vals_p_base + diff_vals_p_base)
                             # gain from nonwords - words
                            # sg_p = np.abs(main_vals_p + diff_vals_p) - np.abs(main_vals_p)
                            # sg_p_base = np.abs(main_vals_p_base + diff_vals_p_base) - np.abs(main_vals_p_base)
                         
                         val_p = np.mean(sg_p, axis=1)
                         val_p_base = np.mean(sg_p_base, axis=1)
                    else:
                        val_p = raw_fea[rms_cols].abs().mean(axis=1)
                        val_p_base = raw_fea_base[rms_cols].abs().mean(axis=1)

                    raw_fea['rms'] = val_p / np.sqrt(len(rms_cols))
                    raw_fea_base['rms'] = val_p_base / np.sqrt(len(rms_cols))
                    raw_fea['rms'] = raw_fea['rms'] - raw_fea_base['rms']

                    # --- 4. Time Binning & Raw P-value Calculation ---
                    feature_means = []
                    feature_raw_pvals = []
                    
                    for (t_start, t_end) in time_windows:
                        # Select data in window
                        mask_window = (raw_fea['time_point'] > t_start) & (raw_fea['time_point'] <= t_end)
                        data_window = raw_fea[mask_window]
                        
                        # Average RMS over time window for each permutation
                        perm_means = data_window.groupby('perm')['rms'].mean()
                        
                        obs_mean = perm_means.loc[0]  # Observed mean (Perm 0)
                        null_means = perm_means.loc[1:] # Null distribution
                        
                        feature_means.append(obs_mean)
                        
                        # Calculate One-tailed P-value
                        n_perms = len(null_means)
                        if n_perms > 0:
                            # Test: Observed > Null
                            p_val = (np.sum(null_means >= obs_mean) + 1) / (n_perms + 1)
                        else:
                            p_val = 1.0
                        
                        feature_raw_pvals.append(p_val)

                    #Apply FDR correction within this feature across time windows
                    if len(feature_raw_pvals) > 0:
                         _, feature_corrected_pvals, _, _ = multipletests(feature_raw_pvals, alpha=0.025, method='fdr_bh')
                    else:
                         feature_corrected_pvals = feature_raw_pvals
                    # feature_corrected_pvals = feature_raw_pvals

                    # Store results for this feature
                    bar_plot_data[beta_fea] = {
                        'means': feature_means,
                        'pvals': feature_corrected_pvals
                    }

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
                ax.set_ylim(-0.2 * fea_plot_yscale, 0.2 * fea_plot_yscale)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Headers
                ax.set_title(elec_grp, fontsize=18, fontweight='bold', pad=20)
                
                # Row Labels
                if row_idx == 0:
                    ax.set_ylabel("Avg Beta Gain", fontsize=16, fontweight='bold')
                
                # X Labels
                ax.set_xlabel("Time Window (s)", fontsize=14, fontweight='bold')
                
                # Legend (Top Left Only)
                if row_idx == 0:
                    ax.legend(loc='upper right', frameon=False, fontsize=14)

        # --- Save Figure ---
        fig_rms.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent overlap
        save_dir = os.path.join('figs', f'diff{is_yn}{is_huge}_{delay_nodelay_base}')
        os.makedirs(save_dir, exist_ok=True)
        
        save_name = os.path.join(save_dir, f'BigPlot_Part{chunk_idx+1}_RMS_Bar_FDR_vWMλ_{vWM_lambda}.tif')
        
        plt.savefig(save_name, dpi=100)
        plt.close(fig_rms)
        print(f"Saved: {save_name}")

print("Done.")
# %%
