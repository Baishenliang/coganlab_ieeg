#%% Introduction
# Prepare raw data for LME encoding
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp

HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\lme\\prepare_raw.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sf_dir = 'data'
with open(os.path.join('..', 'GLM', 'data', f'Lex_twin_idxes_hg.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)
update_dict = False
# Get the Aud/SM/Mtr Delay electrodes without vWM:
LexDelay_twin_idxes['LexDelay_Sensorimotor_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Sensorimotor_sig_idx']-LexDelay_twin_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx']
LexDelay_twin_idxes['LexDelay_Auditory_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Aud_NoMotor_sig_idx']-LexDelay_twin_idxes['LexDelay_Auditory_in_Delay_sig_idx']
LexDelay_twin_idxes['LexDelay_Motor_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Motor_sig_idx']-LexDelay_twin_idxes['LexDelay_Motor_in_Delay_sig_idx']

import sys
sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
from ieeg.calc.fast import mixup

# %% function block
mean_word_len=0.65#0.62 # from utils/lexdelay_get_stim_length.m
auditory_decay=0 # a short period of time that we may assume auditory decay takes
delay_len=1.125 # average length from sound offset to Go onset

def get_time_indexs(time_str,start_float:float=0,end_float:float=delay_len):
    time_str = [float(i) for i in time_str]
    start_idx = np.searchsorted(time_str, start_float, side='left')
    end_idx = np.searchsorted(time_str, end_float, side='right')
    indices = list(range(start_idx, end_idx))
    return indices

def rearrange_elects(elec_grps, elec_idxs, epoc,t_range:list=[-0.5, 2]):
    data_list = []
    chs_list = []
    times_list = None
    grp_list = []

    for elec_grp, elec_idx in zip(elec_grps, elec_idxs):
        print(f'Now Doing {elec_grp}')
        m_chs = epoc.take(list(LexDelay_twin_idxes[elec_idx]), axis=0)
        m = m_chs.take(get_time_indexs(m_chs.labels[1], t_range[0], t_range[1]), axis=1)
        m_chs_labels = m.labels[0]
        m_times = m.labels[1].tolist()
        m_data = m.__array__()
        print(f'max time point {[m_data.shape[1]-1]}')

        data_list.append(m_data)
        chs_list.extend(m_chs_labels)
        if times_list is None:
            times_list = m_times 
        grp_list.extend([elec_grp] * len(m_chs_labels))

    if data_list:
        return np.vstack(data_list), chs_list, times_list, grp_list
    else:
        return np.empty((0, 450)), [], []


# %% groups of patients
datasource='hg' # 'glm_(Feature)' or 'hg'
groupsTag="LexDelay"
#groupsTag="LexDelay&LexNoDelay"

# %% define condition and load data
stat_type='mask'
contrast='ave' # average, not contrasting different conditions
# For lexical delay task, whether run the data only with repeat tasks
trial_labels='CORRECT'

# %% Sort data and get significant electrode lists
import os
import numpy as np

stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

if groupsTag=="LexDelay":
    data_LexDelay_Aud,subjs=gp.load_stats('mask','Auditory_inRep','ave',stats_root_delay,stats_root_delay,cbind_subjs=False)
    elec_labels=data_LexDelay_Aud.labels[0]
    data_LexDelay_Cue, _ = gp.load_stats('mask', 'Cue_inRep', 'ave', stats_root_delay, stats_root_delay,cbind_subjs=False)
    data_LexDelay_Go, _ = gp.load_stats('mask', 'Go_inRep', 'ave', stats_root_delay, stats_root_delay,cbind_subjs=False)
    data_LexDelay_Resp, _ = gp.load_stats('mask', 'Resp_inRep', 'ave', stats_root_delay, stats_root_delay,cbind_subjs=False)

    epoc_LexDelayRep_Aud,_=gp.load_stats('zscore','Auditory_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=False,cbind_subjs=False)
    epoc_LexDelayRep_Go,_=gp.load_stats('zscore','Go_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=False,cbind_subjs=False)
    epoc_LexDelayRep_Resp,_=gp.load_stats('zscore','Resp_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=False,cbind_subjs=False)


if groupsTag=="LexDelay&LexNoDelay":
    # epoc_LexDelayRep_Aud, _ = gp.load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_delay,trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)
    # epoc_LexNoDelay_Aud, _ = gp.load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_nodelay,trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)
    data_LexNoDelay_Aud,_=gp.load_stats('mask','Auditory_inRep','ave',stats_root_nodelay,stats_root_nodelay)
    elec_labels=data_LexNoDelay_Aud.labels[0]

arrays_to_hstack = []
final_chs = None
final_times = []
final_grps = None

if groupsTag=="LexDelay":
    # elec_grps=('Spt','lPMC','lIFG')
    # elec_idxs=('Hikock_Spt','Hikock_lPMC','Hikock_lIFG')
    # elec_grps=('Sensorymotor_in_Delay',)
    # elec_idxs=('LexDelay_Sensorimotor_in_Delay_sig_idx',)
    elec_grps=('Spt',)
    elec_idxs=('Hikock_Spt',)
    # elec_grps=('lPMC',)
    # elec_idxs=('Hikock_lPMC',)
    # elec_grps=('lIFG',)
    # elec_idxs=('Hikock_lIFG',)
    for epoc,t_range,epoc_tag in zip((epoc_LexDelayRep_Aud,epoc_LexDelayRep_Go,epoc_LexDelayRep_Resp),
                             ([-0.5, 1.5], [-0.5, 1], [-0.5, 1.25]),
                             ('Stim','Go','Resp')):
        curr_arr, curr_chs, curr_times, curr_grps = rearrange_elects(elec_grps, elec_idxs, epoc,t_range=t_range)
        
        # Save final_array as csv
        df_final = pd.DataFrame(curr_arr.T, index=curr_times, columns=curr_chs)
        if not os.path.exists(os.path.join(script_dir, sf_dir)):
            os.makedirs(os.path.join(script_dir, sf_dir))
        df_final.to_csv(os.path.join(script_dir, sf_dir, f'{'_'.join(elec_grps)}_{epoc_tag}_zscore_epo.csv'))

        # Read final_array from csv
        # df_final_read = pd.read_csv(os.path.join(script_dir, sf_dir, f'{'_'.join(elec_grps)}_{epoc_tag}_zscore_epo.csv'), index_col=0)
        # final_times_read = df_final_read.index.to_numpy(dtype=float)
        # final_chs_read = df_final_read.columns.tolist()
        # final_array_read = df_final_read.values.T
        
        arrays_to_hstack.append(curr_arr)
        final_times.extend(curr_times)
    
        if final_chs is None:
            final_chs = curr_chs
            final_grps = curr_grps

    final_array = np.hstack(arrays_to_hstack)
    final_times = np.array(final_times, dtype=float)

    # Setup for non-linear time axis (concatenated epochs)
    x_linear = np.arange(len(final_times))
    tick_indices = [i for i, t in enumerate(final_times) if abs(t - round(t / 0.25) * 0.25) < 1e-4]
    zero_indices = [i for i, t in enumerate(final_times) if abs(t) < 1e-4]
    minus_point_five_indices = [i for i, t in enumerate(final_times) if abs(t + 0.5) < 1e-4]
    def time_formatter(x, pos):
        idx = int(x)
        if 0 <= idx < len(final_times):
            if abs(final_times[idx] + 0.5) < 1e-4: return ""
            return f"{final_times[idx]:.2f}".rstrip('0').rstrip('.')
        return ""



if final_array.min() < 0:
    print(f"Negative values detected (min={final_array.min():.2f}). Shifting data to satisfy NMF non-negativity constraint...")
    final_array = final_array - final_array.min()

#%% NMF verification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

def evaluate_nmf_stability(X, k_range, n_repeats=100):
    """
    Evaluate NMF model stability (Cophenetic Correlation) and fit quality (Explained Variance).
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data matrix (n_channels, n_timepoints). Must be non-negative.
    k_range : list or range
        Range of ranks to test (e.g., range(2, 10)).
    n_repeats : int
        Number of NMF runs per k to estimate stability.
        
    Returns:
    --------
    results : dict
        Contains 'k_vals', 'rss', 'explained_variance', and 'cophenetic'.
    """
    
    # 1. Calculate Total Sum of Squares (Total Variance) of the original data
    # This is required to calculate Explained Variance (EV)
    # Total SS = ||X||_F^2
    total_ss = np.linalg.norm(X, ord='fro')**2
    
    rss_scores = [] 
    ev_scores = []          # New: Store Explained Variance
    cophenetic_scores = []
    
    print(f"Starting NMF evaluation for k = {k_range[0]} to {k_range[-1]}...")
    
    for k in k_range:
        consensus_matrix = np.zeros((X.shape[0], X.shape[0]))
        current_rss_list = []
        
        for i in range(n_repeats):
            # Run NMF
            model = NMF(n_components=k, init='random', random_state=None, 
                        max_iter=500, solver='cd', tol=1e-4)
            W = model.fit_transform(X)
            
            # Record RSS (Squared reconstruction error)
            # model.reconstruction_err_ is the Frobenius norm ||X - WH||
            rss = model.reconstruction_err_**2
            current_rss_list.append(rss)
            
            # Hard Clustering for Consensus Matrix
            labels = np.argmax(W, axis=1)
            connectivity = (labels[:, None] == labels[None, :]).astype(float)
            consensus_matrix += connectivity
            
        # Normalize Consensus Matrix
        consensus_matrix /= n_repeats
        
        # --- Calculate Cophenetic Correlation (Stability) ---
        dist_matrix = 1 - consensus_matrix
        dist_vec = squareform(dist_matrix)
        Z = linkage(dist_vec, method='average')
        c, _ = cophenet(Z, dist_vec)
        
        # --- Calculate Average Metrics ---
        avg_rss = np.mean(current_rss_list)
        
        # Calculate Explained Variance (EV)
        # EV = 1 - (Residual / Total)
        avg_ev = 1 - (avg_rss / total_ss)
        
        rss_scores.append(avg_rss)
        ev_scores.append(avg_ev)
        cophenetic_scores.append(c)
        
        print(f"Rank k={k}: EV={avg_ev:.2%}, Cophenetic={c:.4f}")

    return {
        'k_vals': list(k_range),
        'rss': rss_scores,
        'explained_variance': ev_scores,
        'cophenetic': cophenetic_scores
    }

def plot_stability_metrics(results):
    """
    Plots Explained Variance (Fit) and Cophenetic Correlation (Stability).
    """
    k_vals = results['k_vals']
    ev = results['explained_variance']
    ccc = results['cophenetic']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Plot 1: Explained Variance (Left Axis) ---
    color = 'tab:red'
    ax1.set_xlabel('Number of Components (k)', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', color=color, fontsize=12)
    ax1.plot(k_vals, ev, 'o--', color=color, lw=2, label='Explained Variance (Fit)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Format Y-axis as percentage
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Cophenetic Correlation (Right Axis) ---
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Cophenetic Correlation (Stability)', color=color, fontsize=12)
    ax2.plot(k_vals, ccc, 's-', color=color, lw=2, label='Stability (CCC)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('NMF Model Selection: Stability vs. Fit', fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Usage Example ---
metrics = evaluate_nmf_stability(final_array, k_range=range(2, 10))
plot_stability_metrics(metrics)

# %% Auto define optimal k using BIC

def auto_find_optimal_k_nmf(X, k_range=range(2, 10), plot=True):
    """
    使用 BIC (Bayesian Information Criterion) 自动寻找 NMF 的最佳秩 (k)。
    
    原理：假设残差服从高斯分布，根据重建误差计算 BIC。寻找使 BIC 最小的 k。
    """
    
    n_samples, n_features = X.shape
    n_datapoints = n_samples * n_features  # 数据点总数 (N)
    
    bic_scores = []
    
    print("正在计算各 k 值的 BIC 准则...")
    for k in k_range:
        # 运行 NMF
        model = NMF(n_components=k, init='nndsvd', max_iter=500, random_state=42)
        model.fit(X)
        
        # 1. 计算 RSS (残差平方和)
        rss = model.reconstruction_err_**2
        
        # 2. 计算自由度 (自由参数的数量 P)
        # W 矩阵有 n_samples * k 个参数，H 矩阵有 k * n_features 个参数
        n_parameters = k * (n_samples + n_features)
        
        # 3. 计算 BIC
        # 公式: N * ln(RSS / N) + P * ln(N)
        # 第一项代表误差 (越小越好)，第二项代表复杂度惩罚 (越小越好)
        bic = n_datapoints * np.log(rss / n_datapoints) + n_parameters * np.log(n_datapoints)
        
        bic_scores.append(bic)
        print(f"  k={k} | RSS={rss:.2f} | Parameters={n_parameters} | BIC={bic:.2f}")

    # 自动找出 BIC 最小的 k
    optimal_k = k_range[np.argmin(bic_scores)]
    print(f"\n✅ 根据 BIC 准则，最佳的 Component 数量 (k) 是: {optimal_k}")
    
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(k_range, bic_scores, 'o-', color='tab:purple', linewidth=2, markersize=8)
        
        # 标出最低点
        min_bic = min(bic_scores)
        plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k = {optimal_k}')
        plt.plot(optimal_k, min_bic, 'r*', markersize=15)
        
        plt.title('NMF Optimal k Selection using BIC', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Components (k)', fontsize=12)
        plt.ylabel('BIC Score (Lower is better)', fontsize=12)
        plt.xticks(k_range)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        save_dir = '../Greg_ROIs/fig'
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
            
        save_filename = "auto_find_optimal_k_nmf.svg"
        save_path = os.path.join(save_dir, save_filename)

        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')

        plt.show()
        plt.show()
        
    return optimal_k

# --- 运行示例 ---
# 假设 final_array 是你的数据
optimal_k = auto_find_optimal_k_nmf(final_array, k_range=range(2, 10))
print(f"你可以直接将 n_components 设为: {optimal_k}")

#%% 0. Global Setup & Imports

Sensorimotor_col = [1, 0, 0]  
Auditory_col = [0, 1, 0]  
Motor_col = [0, 0, 1]  
Delay_col = [1, 0.65, 0]

macro_color_dict = {
    'SM_Auditory': Auditory_col,
    'SM_Motor': Motor_col,
    'Sustained': Delay_col
}

#%% 1. NMF Data Preprocessing & Execution
X = final_array.copy()
if X.min() < 0:
    X = X - X.min()

n_components = 5
model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

W = model.fit_transform(X)
H = model.components_

comp_names = [f'Comp_{i+1}' for i in range(n_components)]

#%% 2. Plot NMF Component Traces (H Matrix)
plt.figure(figsize=(10, 4))
colors = plt.cm.tab10.colors

for i in range(n_components):
    plt.plot(x_linear, H[i, :], label=comp_names[i], color=colors[i], linewidth=2.5, alpha=0.85)

for idx in zero_indices:
    plt.axvline(x=idx, color='k', linestyle='--', linewidth=1, alpha=0.5)
for idx in minus_point_five_indices:
    plt.axvline(x=idx, color='k', linestyle='-', linewidth=1, alpha=0.5)

plt.title(f'Temporal Traces of the {n_components} NMF Components (H Matrix)', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Component Amplitude (a.u.)', fontsize=12)
plt.gca().xaxis.set_major_locator(mticker.FixedLocator(tick_indices))
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
plt.legend(loc='upper right', bbox_to_anchor=(1.18, 1), frameon=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

save_dir = '../Greg_ROIs/fig'
if not os.path.exists(save_dir): 
    os.makedirs(save_dir)
    
save_filename = "Comp_traces.svg"
save_path = os.path.join(save_dir, save_filename)

plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')

plt.show()

#%% 3. Hierarchical Clustering (Forcing 3 Macro Clusters)
dist_matrix = pdist(H, metric='correlation')
Z = linkage(dist_matrix, method='average')

target_macro_clusters = 3
macro_labels = fcluster(Z, target_macro_clusters, criterion='maxclust')

unique_labels = sorted(np.unique(macro_labels))
custom_macro_names = ['SM_Auditory', 'SM_Motor', 'Sustained']
label_to_name = {label: custom_macro_names[i] for i, label in enumerate(unique_labels)}
macro_mapping = {comp_names[i]: label_to_name[macro_labels[i]] for i in range(n_components)}

plt.figure(figsize=(6, 4))
dendro = dendrogram(Z, labels=comp_names, orientation='top', 
                    link_color_func=lambda x: 'tab:blue')
plt.title(f'Hierarchical Clustering (Forced to {target_macro_clusters} Macro Clusters)', fontsize=12, fontweight='bold')
plt.ylabel('Distance (1 - Pearson r)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

save_dir = '../Greg_ROIs/fig'
if not os.path.exists(save_dir): 
    os.makedirs(save_dir)
    
save_filename = "Dendrogram_Macro_Clusters.svg"
save_path = os.path.join(save_dir, save_filename)

plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
plt.show()

#%% 4. Hard Clustering & Saving Indices
# 1. 构建包含空间权重矩阵 W 的 DataFrame
df_weights = pd.DataFrame(W, columns=comp_names)
df_weights.insert(0, 'Channel', final_chs)
df_weights.insert(1, 'Group', final_grps)

# 2. 为每个电极分配主导成分和宏观大类 (Macro Cluster)
df_weights['Base_Comp'] = df_weights[comp_names].idxmax(axis=1)
df_weights['Dominant_Comp'] = df_weights['Base_Comp'].map(macro_mapping)

# 3. 将硬分类结果对应的电极 index 更新到字典中并保存
for category in df_weights['Dominant_Comp'].dropna().unique():
    category_chs = set(df_weights[df_weights['Dominant_Comp'] == category]['Channel'])
    try:
        category_idx = set([i for i, x in enumerate(data_LexDelay_Aud.labels[0]) if x in category_chs])
        LexDelay_twin_idxes[f'LexDelay_Sensorimotor_in_Delay_sig_idx_{category}'] = category_idx
    except NameError:
        pass 

try:
    if update_dict:
        with open(os.path.join('..', 'GLM', 'data', f'Lex_twin_idxes_hg.npy'), "wb") as f:
            pickle.dump(LexDelay_twin_idxes, f)
except NameError:
    pass

#%% 5. Anatomical Distribution Pie Charts
df_mean_weights = df_weights.groupby('Group')[comp_names].mean()
groups = df_mean_weights.index

if 'color_dict' not in locals():
    import matplotlib.cm as cm
    color_dict = dict(zip(groups, cm.tab20.colors[:len(groups)]))

fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 5))
if n_components == 1: axes = [axes]

for i, col_name in enumerate(comp_names):
    ax = axes[i]
    values = df_mean_weights[col_name]
    valid_idx = values > 0.01
    valid_values = values[valid_idx]
    valid_groups = groups[valid_idx]
    
    if len(valid_values) == 0: continue
        
    ax.pie(valid_values, labels=valid_groups, autopct='%1.1f%%', 
           colors=[color_dict.get(g, 'gray') for g in valid_groups], 
           startangle=140, textprops={'fontsize': 12})
    
    macro_belonging = macro_mapping[col_name]
    ax.set_title(f'{col_name}\n({macro_belonging})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

#%% 6. Plot Aligned Traces (Stim / Go / Resp)
plot_groups = []
target_macros = ['SM_Auditory', 'Sustained', 'SM_Motor']

for macro in target_macros:
    category_chs = set(df_weights[df_weights['Dominant_Comp'] == macro]['Channel'])
    sig_idx = [i for i, x in enumerate(data_LexDelay_Aud.labels[0]) if x in category_chs]
    group_col = macro_color_dict.get(macro, [0, 0, 0])
    plot_groups.append((sig_idx, macro, group_col))

unit_scale = 2.0
left_padding_with_y = 1.6
left_padding_no_y = 0.2
right_padding = 0.4
fig_height = 3.0

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 2.0

alignments = [
    ('Stim', epoc_LexDelayRep_Aud, [-0.25, 1.5], True),
    ('Go', epoc_LexDelayRep_Go, [-0.25, 1.0], range(631, 650)),
    ('Resp', epoc_LexDelayRep_Resp, [-0.25, 1.0], range(631, 650))
]

for align_tag, epoc_data, x_limits, bsl_val in alignments:
    
    has_y = (align_tag == 'Stim')
    current_left_pad = left_padding_with_y if has_y else left_padding_no_y
    
    x_duration = x_limits[1] - x_limits[0]
    fig_width = (x_duration * unit_scale) + current_left_pad + right_padding
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    fig.subplots_adjust(left=current_left_pad/fig_width, right=1.0 - (right_padding/fig_width), bottom=0.25, top=0.9)
    ax = plt.gca()

    for sig_idx, label_text, group_col in plot_groups:
        if len(sig_idx) == 0: continue
        gp.plot_wave(epoc_data, sig_idx, f'{label_text}', group_col, '-', bsl_val, ylim=[-0.2, 4.2])

    ax.set_ylim([-0.3, 4.2])
    
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    
    if not has_y:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.yaxis.set_visible(False) 
    else:
        ax.spines['left'].set_linewidth(3)
        ax.set_yticks([0, 2, 4])
        ax.tick_params(axis='y', labelsize=24, length=6, width=2.5)

    xticks = [0, 0.5, 1.0, 1.5]
    xticks = [t for t in xticks if x_limits[0] <= t <= x_limits[1]]
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    plt.draw()
    labels = [l.get_text() for l in ax.get_xticklabels()]
    new_labels = ["0" if (l == "0.0" or l == ".0") else l for l in labels]
    ax.set_xticklabels(new_labels)
    
    sns.despine(ax=ax, offset=10, trim=True, left=not has_y)
    
    ax.set_xlim(x_limits)
    ax.tick_params(axis='x', labelsize=24, length=6, width=2.5)
    ax.spines['bottom'].set_bounds(x_limits[0], x_limits[1])
    
    ax.set_xlabel(''); ax.set_ylabel('')
    
    ax.axvline(x=0, linestyle='--', color='#444444', linewidth=1.5, dashes=(5, 5), zorder=0)
    ax.axhline(y=0, linestyle='-', color='#DDDDDD', linewidth=1.0, zorder=0)

    if align_tag == 'Resp':
        ax.legend(loc='upper right', frameon=False, fontsize=12, handlelength=1.5)
    else:
        if ax.get_legend(): ax.get_legend().remove()

    save_dir = '../Greg_ROIs/fig'
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        
    save_filename = f"Spt_trace_{align_tag}.svg"
    save_path = os.path.join(save_dir, save_filename)
    
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches=None)
    plt.show()

#%% 7. Plot All Electrodes on Brain Map
plot_df = df_weights.dropna(subset=['Dominant_Comp'])
channels_all = plot_df['Channel'].tolist()
cols_lst_all = [macro_color_dict.get(comp, [0.5, 0.5, 0.5]) for comp in plot_df['Dominant_Comp']]

try:
    gp.plot_brain(subjs, channels_all, cols_lst_all, None, 'All Macro Clusters', 0.3, 0.2, hemi='lh')
except Exception as e:
    pass

# %% 8. Extract Power for Stats & Plot Brain Maps + Bar+Strip
_, _, _, _, _, paras_aud, *_ = gp.sort_chs_by_actonset(data_LexDelay_Aud, epoc_LexDelayRep_Aud, 0.011, [0, 0.25], mask_data=True, select_electrodes=False)
_, _, _, _, _, paras_mtr, *_  = gp.sort_chs_by_actonset(data_LexDelay_Resp, epoc_LexDelayRep_Resp, 0.011, [-0.25, 0], mask_data=True, select_electrodes=False)
#% 1. Extract Power from DataFrames and Prepare Stats
plot_df = df_weights.dropna(subset=['Dominant_Comp']).copy()

aud_powers = []
mot_powers = []
valid_chs = []
clusters = []

for idx, row in plot_df.iterrows():
    ch = row['Channel']
    if ch in paras_aud.index and ch in paras_mtr.index:
        p_aud = paras_aud.loc[ch, 'mean_value']
        p_mot = paras_mtr.loc[ch, 'mean_value']
        
        aud_powers.append(p_aud)
        mot_powers.append(p_mot)
        valid_chs.append(ch)
        clusters.append(row['Dominant_Comp'])

aud_powers = np.array(aud_powers)
mot_powers = np.array(mot_powers)

aud_z = (aud_powers - np.mean(aud_powers)) / np.std(aud_powers)
mot_z = (mot_powers - np.mean(mot_powers)) / np.std(mot_powers)

df_stats = pd.DataFrame({
    'Channel': valid_chs,
    'Macro_Cluster': clusters,
    'Auditory_Power_Raw': aud_powers,
    'Motor_Power_Raw': mot_powers,
    'Auditory_Power_Z': aud_z,
    'Motor_Power_Z': mot_z
})

#% 2. Plot Brain Maps (Using Z-scores for better visual contrast)
norm_aud = mcolors.Normalize(vmin=np.percentile(aud_z, 5), vmax=np.percentile(aud_z, 95))
cmap_aud = plt.cm.Greens
cols_aud = [cmap_aud(norm_aud(val))[:3] for val in df_stats['Auditory_Power_Z']]

try:
    gp.plot_brain(subjs, df_stats['Channel'].tolist(), cols_aud, None, 'Auditory Power Z-score (0-250ms)', 0.3, 0.2,hemi='lh')
except Exception as e:
    pass

norm_mot = mcolors.Normalize(vmin=np.percentile(mot_z, 5), vmax=np.percentile(mot_z, 95))
cmap_mot = plt.cm.Blues
cols_mot = [cmap_mot(norm_mot(val))[:3] for val in df_stats['Motor_Power_Z']]

try:
    gp.plot_brain(subjs, df_stats['Channel'].tolist(), cols_mot, None, 'Motor Power Z-score (-250-0ms)', 0.3, 0.2,hemi='lh')
except Exception as e:
    pass

#% 3. Plot Bar+Strip (Using Raw Power for intuitive interpretation)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 2.0

order = ['SM_Auditory', 'Sustained', 'SM_Motor']
palette = {'SM_Auditory': Auditory_col, 'Sustained': Delay_col, 'SM_Motor': Motor_col}

metrics = [
    ('Auditory_Power_Raw', 'Auditory z-score (avg.)', 'BarStrip_Auditory_Power_Raw.svg'),
    ('Motor_Power_Raw', 'Motor z-score (avg.)', 'BarStrip_Motor_Power_Raw.svg')
]

for col, ylabel, save_name in metrics:
    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    
    sns.barplot(data=df_stats, x='Macro_Cluster', y=col, order=order, palette=palette, 
                alpha=0.5, capsize=0.1, errwidth=2.5, ax=ax, edgecolor='none')
    
    sns.stripplot(data=df_stats, x='Macro_Cluster', y=col, order=order, palette=palette, 
                  jitter=True, alpha=0.8, size=6, ax=ax)
    
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    
    ax.tick_params(axis='y', labelsize=16, length=6, width=2.5)
    ax.tick_params(axis='x', labelsize=14, length=6, width=2.5)
    
    plt.ylabel(ylabel, fontsize=16, fontweight='bold')
    plt.xlabel('')
    
    sns.despine(ax=ax, offset=10, trim=True)
    plt.tight_layout()
    
    save_dir = '../Greg_ROIs/fig'
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
# %%
