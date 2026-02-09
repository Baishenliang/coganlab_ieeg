#%% Introduction
# Prepare raw data for LME encoding
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    # elec_grps=('Spt',)
    # elec_idxs=('Hikock_Spt',)
    # elec_grps=('lPMC',)
    # elec_idxs=('Hikock_lPMC',)
    elec_grps=('lIFG',)
    elec_idxs=('Hikock_lIFG',)
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

#%% NMF
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# --- 1. Data Preprocessing (NMF requires non-negative input) ---
# Check if negative values exist in the data
X = final_array.copy()
if X.min() < 0:
    print(f"Negative values detected (min={X.min():.2f}). Shifting data to satisfy NMF non-negativity constraint...")
    X = X - X.min()

# --- 2. Run NMF ---
n_components = 3
# init='nndsvd' typically yields more consistent and sparse results
model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

# W: (n_samples, n_components) -> e.g., (91, 3) 
# Contribution weights of each electrode (sample) to each component
W = model.fit_transform(X)

# H: (n_components, n_features) -> e.g., (3, 900)
# Time series (Trace) for each component
H = model.components_

# --- 3. Plotting: Weighted Average Traces of Top 50% Electrodes ---
import matplotlib.ticker as ticker

# 确保使用原始信号进行绘图 (如果 X 被 shift 过，这里最好用原始的 final_array)
# 假设 final_array 是 (n_channels, n_timepoints)
signal_data = final_array 

plt.figure(figsize=(12, 4))

# 颜色定义 (保持不变)
MotorPrep_col = [1.0, 0.0784, 0.5765] 
Sensorimotor_col = [1, 0, 0]  
Auditory_col = [0, 1, 0]  
Delay_col = [1, 0.65, 0]  
Motor_col = [0, 0, 1]  
colors = [
    Sensorimotor_col,
    Motor_col, 
    Delay_col,
    Auditory_col, 
    Delay_col, 
    Sensorimotor_col, 
    '#7f7f7f', 
    '#9467bd', 
    '#8c564b'  
]
#colors_trace = [[1,0,0],[0,1,0],[1,165/255,0]] #Spt
#colors_trace = [[1,0,0],[1,165/255,0]] #lPMC
colors_trace = [[191/255,191/255,191/255],[0,0,0],[127/255,127/255,127/255]] #lIFG

# comp_names 需要在外面定义好，或者根据 n_components 生成
#comp_names = ['Auditory_continuous', 'Auditory_motor', 'Auditory_onset'] #Spt
#comp_names = ['Auditory_continuous', 'Auditory_onset'] #PMC
comp_names = ['delay pre-articulation','articulation','both delay and articulation'] #lIFG
# ==============================================================================
# 1. 配置区域：在这里手动指定“名字-颜色”和“NMF索引-名字”的对应关系
# ==============================================================================

# --- A. 颜色配置 (请根据当前是 Spt, lPMC 还是 lIFG 选择解注) ---

# [配置 1: lPMC] (Continuous=红, Onset=橙)
# color_map = {
#     'Auditory_continuous': [1, 0, 0],       # Red
#     'Auditory_onset':      [1, 165/255, 0]  # Orange
# }

# [配置 2: Spt] (Motor=红, Onset=绿, Continuous=橙)
# color_map = {
#     'Auditory_motor':      [1, 0, 0],       # Red
#     'Auditory_onset':      [0, 1, 0],       # Green
#     'Auditory_continuous': [1, 165/255, 0]  # Orange
# }

# [配置 3: lIFG] (Delay=浅灰, Articulation=黑, Both=深灰) <--- 当前生效
color_map = {
    'delay pre-articulation':      [191/255, 191/255, 191/255], # Light Gray
    'articulation':                [0, 0, 0],                   # Black
    'both delay and articulation': [127/255, 127/255, 127/255]  # Dark Gray
}


# --- B. 身份绑定 (CRITICAL STEP!) ---
# 跑完 NMF 后，看一眼 W 或 H，确认 Component 0, 1, 2 分别是谁，然后修改这里。
# Key 是 NMF 的 index, Value 必须是上面 color_map 里定义的 key

# lIFG 配置示例 (请根据您的 NMF 实际结果修改 0, 1, 2 的对应关系!)
nmf_identity = {
    0: 'both delay and articulation', 
    1: 'delay pre-articulation',       
    2: 'articulation'            
}

# (备用：lPMC 配置)
# nmf_identity = {
#     0: 'Auditory_continuous',
#     1: 'Auditory_onset'
# }


# --- C. 图例顺序 ---
# 列表里的名字顺序决定了画图和图例的顺序 (谁在前面谁就在图例上面)

# [lIFG 顺序]
legend_order = ['delay pre-articulation', 'articulation', 'both delay and articulation']

# [lPMC / Spt 顺序 (备用)]
# legend_order = ['Auditory_onset', 'Auditory_continuous', 'Auditory_motor']


# ==============================================================================
# 2. 绘图循环 (已修改为按“名字”循环，而非按 Index 循环)
# ==============================================================================

# 筛选出当前任务中实际存在的名字进行循环
active_names = [name for name in legend_order if name in nmf_identity.values()]

for name in active_names:
    
    # 1. 反向查找：找到该名字对应的 NMF Index
    # (通过 Value 找 Key)
    target_indices = [k for k, v in nmf_identity.items() if v == name]
    if not target_indices: continue #以防万一
    i = target_indices[0] 
    
    # --- 以下逻辑保持您原本的处理流程 ---

    # 2. 获取当前 Component 的权重
    component_weights = W[:, i]
    
    # 3. 确定 Top 25% 的阈值 (75th percentile)
    threshold = np.percentile(component_weights, 75)
    
    # 4. 找出符合条件的电极索引
    top_indices = np.where(component_weights >= threshold)[0]
    
    if len(top_indices) == 0:
        continue

    # 5. 提取原始信号和权重
    selected_signals = signal_data[top_indices, :]  
    raw_weights = component_weights[top_indices]    
    
    # 6. 权重归一化 (Normalization)
    weight_sum = np.sum(raw_weights)
    if weight_sum > 0:
        norm_weights = raw_weights / weight_sum
    else:
        norm_weights = np.ones_like(raw_weights) / len(raw_weights)
    
    # 7. 计算加权平均 (Weighted Mean)
    weighted_mean = np.sum(selected_signals * norm_weights[:, np.newaxis], axis=0)
    
    # 8. 计算加权标准误 (Weighted SEM)
    weighted_variance = np.sum(norm_weights[:, np.newaxis] * (selected_signals - weighted_mean)**2, axis=0)
    weighted_std = np.sqrt(weighted_variance)
    sem = weighted_std / np.sqrt(len(top_indices))
    
    # 9. 绘图 (使用 color_map 中锁定的颜色)
    color = color_map[name]
    label_text = f"{name} (n={len(top_indices)})"
    
    plt.plot(x_linear, weighted_mean, label=label_text, color=color, linewidth=2)
    plt.fill_between(x_linear, weighted_mean - sem, weighted_mean + sem, color=color, alpha=0.2)

for idx in zero_indices:
    plt.axvline(x=idx, color='k', linestyle='--', linewidth=1, alpha=0.5)
for idx in minus_point_five_indices:
    plt.axvline(x=idx, color='k', linestyle='-', linewidth=1, alpha=0.5)

plt.title('Weighted Average Traces (Top 25% Electrodes per Component)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (Z-score)') # 或者是 Arbitrary Unit，取决于 final_array 是什么

# 设置坐标轴格式
plt.gca().xaxis.set_major_locator(ticker.FixedLocator(tick_indices))
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))

plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# --- 4. Generate DataFrame: Electrode Contributions [FIXED] ---

# 1. 根据 nmf_identity 的索引顺序 (0, 1, 2...) 生成正确的列名列表
# 确保列名与 W 矩阵的列 (0, 1, 2...) 一一对应
sorted_comp_names = [nmf_identity[i] for i in range(n_components)]

# 2. 使用排好序的名字创建 DataFrame
df_weights = pd.DataFrame(W, columns=sorted_comp_names)

# 3. 插入电极信息
df_weights.insert(0, 'Channel', final_chs)
df_weights.insert(1, 'Group', final_grps)

# 4. 确定 Dominant Component
# 现在列名是对的，idxmax 也就对了
df_weights['Dominant_Comp'] = df_weights[sorted_comp_names].idxmax(axis=1)

# --- 以下保存逻辑保持不变 ---
# Save the new clusters
# 1. Get the electrode categories from the Dominant_Comp column...
for category in df_weights['Dominant_Comp'].unique():
    # 2. transform each category of electrodes...
    category_chs = set(df_weights[df_weights['Dominant_Comp'] == category]['Channel'])
    # 3. make a index set...
    category_idx = set([i for i, x in enumerate(data_LexDelay_Aud.labels[0]) if x in category_chs])
    # 4. append the sets...
    LexDelay_twin_idxes[f'LexDelay_Sensorimotor_in_Delay_sig_idx_{category}'] = category_idx

if update_dict:
    # 5. save...
    with open(os.path.join('..', 'GLM', 'data', f'Lex_twin_idxes_hg.npy'), "wb") as f:
        pickle.dump(LexDelay_twin_idxes, f)

# Display first few rows
print("Electrode Weights DataFrame (First 5 rows):")
print(df_weights.head())

# --- Plot Pie Charts (Logic Updated) ---
# 注意：这里我们也要用 sorted_comp_names 来画图，确保和 DataFrame 列名一致
import matplotlib.pyplot as plt
import numpy as np

# Calculate Mean Weights
df_mean_weights = df_weights.groupby('Group')[sorted_comp_names].mean()

# Setup Colors
groups = df_mean_weights.index
# 确保颜色字典存在 (如果前面没定义)
if 'color_dict' not in locals():
    # 简单的 fallback 颜色
    import matplotlib.cm as cm
    color_dict = dict(zip(groups, cm.tab20.colors[:len(groups)]))

# Draw Pie Charts
fig, axes = plt.subplots(1, n_components, figsize=(5 * n_components, 6))
if n_components == 1: axes = [axes]

# 遍历排好序的名字
for i, col_name in enumerate(sorted_comp_names):
    ax = axes[i]
    
    values = df_mean_weights[col_name]
    
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=groups, 
        autopct='%1.1f%%', 
        colors=[color_dict.get(g, 'gray') for g in groups], 
        startangle=140,
        textprops={'fontsize': 14} # 字体改小一点防止重叠
    )
    
    # 标题加上对应的功能名
    ax.set_title(f'{col_name}\nMean Weight Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# # Display the numerical table as well for reference
# print("Mean Weights per Group:")
# display(df_mean_weights)
# %%
import pandas as pd
import numpy as np

def export_electrode_weights(W, electrode_names=None):
    """
    Export NMF spatial weights (W matrix) into a detailed DataFrame.
    
    Parameters:
    -----------
    W : numpy.ndarray
        The NMF spatial weight matrix with shape (n_electrodes, n_components).
    electrode_names : list of str, optional
        List of electrode names (e.g., ['LA1', 'LA2']). 
        If None, indices are used.
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing absolute weights, relative weights (%), 
        and dominant cluster assignments.
    """
    n_electrodes, n_components = W.shape
    
    # 1. Create the base DataFrame with absolute weights
    comp_cols = [f'Comp_{i+1}' for i in range(n_components)]
    df = pd.DataFrame(W, columns=comp_cols)
    
    # 2. Add electrode identifiers
    if electrode_names is not None:
        if len(electrode_names) != n_electrodes:
            print(f"Warning: Number of names ({len(electrode_names)}) "
                  f"does not match W rows ({n_electrodes}).")
        else:
            df.insert(0, 'Electrode', electrode_names)
    else:
        df.insert(0, 'Electrode_Idx', range(n_electrodes))

# 3. Determine 'Hard Clustering' (Dominant Component) WITH NORMALIZATION
    
    # --- 修改开始 ---
    # 创建一个临时的 W DataFrame 用于计算
    W_temp = df[comp_cols].copy()
    
    # 【关键步骤】：列归一化 (Column Normalization)
    # 将每一列除以该列的最大值。这样每一列的数值范围都是 [0, 1]。
    # 这代表了："在这个 Component 内部，这个电极排老几？"
    # 而不是："绝对数值是多少？"
    W_normalized = W_temp / W_temp.max(axis=0)
    
    # 使用归一化后的矩阵来决定谁是老大
    df['Dominant_Cluster'] = W_normalized.idxmax(axis=1)
    
    # (可选) 你也可以保留归一化后的值用于检查
    # df[[f'{c}_Norm' for c in comp_cols]] = W_normalized
    # --- 修改结束 ---

    # 4. Calculate Relative Weights (这里可以用原始值，也可以用归一化值，看你定义)
    # 通常计算百分比贡献时，使用原始值 W 是物理意义上正确的（能量贡献占比）。
    # 但由于 Onset 的 W 虚高，这里的百分比可能也会偏向 Onset。
    # 如果你想看“功能上的倾向性”，建议也使用 W_normalized 来算百分比。
    
    # 方案 A (物理能量占比 - 维持现状):
    row_sums = df[comp_cols].sum(axis=1) + 1e-9 
    
    # 方案 B (功能倾向占比 - 推荐尝试):
    # row_sums = W_normalized.sum(axis=1) + 1e-9
    # 这里的 df[col] 也要改成 W_normalized[col]
    
    for col in comp_cols:
        # Create new columns like 'Comp_1_Percent'
        df[f'{col}_Percent'] = df[col] / row_sums
    
    # 5. Calculate Selectivity (Confidence)
    # Metric: Ratio of the dominant weight to the total weight sum.
    # Range: [1/k, 1.0]. Higher values indicate the electrode is exclusive to one cluster.
    df['Selectivity'] = df[comp_cols].max(axis=1) / row_sums

    return df

# --- Usage Example ---

# Assuming 'W' is your spatial matrix from model.fit_transform(X)
# and 'elec_names' is your list of channel labels

# df_weights = export_electrode_weights(W, electrode_names=elec_names)

# View the first few rows
print(df_weights.head().round(3))
manual_coding=pd.read_csv(os.path.join('..','Greg_ROIs', 'Hickok_ROI_electrode_manual_coding.csv'))
manual_coding = manual_coding.rename(columns={'chs': 'Channel'})
manual_coding = manual_coding.merge(df_weights[['Channel', 'Dominant_Comp']].drop_duplicates(), on='Channel', how='left')

comp_color_map = dict(zip(comp_names, colors[:len(comp_names)]))

unique_groups = manual_coding['Group'].unique()
for group in unique_groups:
    group_data = manual_coding[manual_coding['Group'] == group]
    unique_tags = group_data['manual_tag'].unique()
    fig, axes = plt.subplots(1, len(unique_tags), figsize=(5 * len(unique_tags), 6))
    if len(unique_tags) == 1:
        axes = [axes]
    for i, tag in enumerate(unique_tags):
        ax = axes[i]
        tag_data = group_data[group_data['manual_tag'] == tag]
        if not tag_data.empty:
            type_counts = tag_data['Dominant_Comp'].value_counts()
            pie_colors = [comp_color_map[comp] for comp in type_counts.index]
            ax.pie(type_counts, labels=type_counts.index, autopct=lambda p: '{:.0f}'.format(p * sum(type_counts) / 100), startangle=140, textprops={'fontsize': 16}, colors=pie_colors)
        ax.set_title(f'{group}\n{tag}', fontsize=18)
    plt.tight_layout()
    plt.show()

ch_to_idx = {ch: i for i, ch in enumerate(final_chs)}

for group in unique_groups:
    group_data = manual_coding[manual_coding['Group'] == group]
    unique_tags = group_data['manual_tag'].unique()
    for tag in unique_tags:
        tag_data = group_data[group_data['manual_tag'] == tag]
        unique_comps = tag_data['Dominant_Comp'].unique()
        for comp in unique_comps:
            channels = tag_data[tag_data['Dominant_Comp'] == comp]['Channel'].tolist()
            indices = [ch_to_idx[ch] for ch in channels if ch in ch_to_idx]
            
            if not indices:
                continue
                
            traces = final_array[indices, :]
            
            plt.figure(figsize=(12, 4))
            plt.plot(x_linear, traces.T, color=comp_color_map[comp], alpha=0.5, linewidth=2)
            for idx in zero_indices:
                plt.axvline(x=idx, color='k', linestyle='--', linewidth=1, alpha=0.5)
            for idx in minus_point_five_indices:
                plt.axvline(x=idx, color='k', linestyle='-', linewidth=1, alpha=0.5)
            plt.title(f'Traces for {group} | {tag} | {comp} (n={len(indices)})')
            plt.xlabel('Time (s)')
            plt.gca().xaxis.set_major_locator(ticker.FixedLocator(tick_indices))
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
            plt.ylabel('Amplitude')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            plt.show()

# Brain plot for projected weights:
for i, comp in enumerate(comp_names):
    w = df_weights[comp].values
    w_norm = (w - w.min()) / (w.max() - w.min())
    cols_lst = [list(np.array(colors_trace[i]) * val + np.array([1, 1, 1]) * (1 - val)) for val in w_norm]
    gp.plot_brain(subjs, df_weights.Channel.to_list(), cols_lst, None, comp, 0.3, 0.2)

# Brain plot for dominant component
cols_lst = [comp_color_map[comp] for comp in df_weights['Dominant_Comp']]
gp.plot_brain(subjs, df_weights.Channel.to_list(), cols_lst, None, 'Dominant_Comp', 0.3, 0.2)

# --- 5. Identify Hub Electrodes ---
# 1. Provincial Hubs: Top contributors for each component
print("\n--- Provincial Hubs (Top 5 per Component) ---")
for comp in comp_names:
    print(f"\nComponent: {comp}")
    print(df_weights.nlargest(5, comp)[['Channel', 'Group', comp]])

# 2. Connector Hubs: Electrodes that are "hub" to the four components (high participation)
# Participation Coefficient: P_i = 1 - sum((w_ij / w_i_total)^2)
w_matrix = df_weights[comp_names].values
w_total = w_matrix.sum(axis=1, keepdims=True)
w_total[w_total == 0] = 1e-10 # Avoid division by zero
df_weights['Total_Weight'] = w_total.flatten()
df_weights['Participation_Coef'] = 1 - np.sum((w_matrix / w_total)**2, axis=1)

print("\n--- Connector Hubs (High Participation across 4 components) ---")
# Filter for electrodes with significant weight (e.g., top 50% of total weight) to avoid noise
significant_elecs = df_weights[df_weights['Total_Weight'] > df_weights['Total_Weight'].quantile(0.75)]
print(significant_elecs.nlargest(10, 'Participation_Coef')[['Channel', 'Group', 'Participation_Coef', 'Total_Weight']])



# Brain plot for participation coefficient of the hub electrodes
w = significant_elecs['Participation_Coef'].values
w_norm = (w - w.min()) / (w.max() - w.min())
cols_lst = [list(np.array([1.0, 0.0784, 0.5765]) * val + np.array([1, 1, 1]) * (1 - val)) for val in w_norm]
gp.plot_brain(subjs, significant_elecs.Channel.to_list(), cols_lst, None, 'Dominant_Comp', 0.3, 0.2)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_individual_traces_by_dominant_comp(epochs_list, epoch_names, df_weights, 
                                            time_windows=[(-0.5, 2.0), (-0.5, 2.0), (-0.5, 2.0)],
                                            colors_dict=None):
    """
    Plots INDIVIDUAL traces for each electrode, grouped by Dominant Component.
    Rows: Components
    Columns: Alignments (Stim, Go, Resp)
    """
    
    # 1. 确定有哪些 Component (按字母顺序或自定义顺序)
    # 如果想固定顺序，可以手动指定 comps 列表，例如 ['Auditory_onset', 'Auditory_continuous']
    comps = sorted(df_weights['Dominant_Comp'].unique())
    n_comps = len(comps)
    n_aligns = len(epochs_list)
    
    # 2. 准备画布
    # sharey='row': 同一行的 Y 轴刻度共享，方便横向对比
    fig, axes = plt.subplots(nrows=n_comps, ncols=n_aligns, 
                             figsize=(5 * n_aligns, 3.5 * n_comps), 
                             sharey='row') 
    
    # 确保 axes 总是二维数组，防止 n=1 时报错
    if n_comps == 1: axes = np.array([axes])
    if n_aligns == 1: axes = axes.reshape(-1, 1)
    # 如果只有一行多列，确保形状正确
    if axes.ndim == 1: axes = axes.reshape(n_comps, n_aligns)

    # 3. 循环绘制
    for i, comp_name in enumerate(comps):
        
        # 获取属于该 Component 的所有电极 ID
        comp_elecs = df_weights[df_weights['Dominant_Comp'] == comp_name]['Channel'].tolist()
        color = colors_dict.get(comp_name, 'k') if colors_dict else 'k'
        
        for j, (epoch, align_name, t_range) in enumerate(zip(epochs_list, epoch_names, time_windows)):
            ax = axes[i, j]
            
            # --- 数据提取 (适配 MNE/Epochs 对象结构) ---
            try:
                # 1. 找到时间索引
                # 假设 epoch.labels[1] 是时间轴
                # 如果 get_time_indexs 是外部函数，请确保它在上下文中可用
                # 这里用简单的 np.where 模拟通用逻辑
                full_times = np.array(epoch.labels[1], dtype=float)
                t_idx_start = np.abs(full_times - t_range[0]).argmin()
                t_idx_end = np.abs(full_times - t_range[1]).argmin()
                
                # 2. 切片时间 (使用 MNE 风格的 take 或 numpy 切片)
                # 假设 epoch 是自定义对象，支持 take(indices, axis=1)
                # 如果是 numpy array (chs, times)，直接切片即可
                
                # 这里沿用您之前的逻辑：先切片
                indices = np.arange(t_idx_start, t_idx_end + 1)
                sliced_epoch = epoch.take(indices, axis=1)
                
                # 3. 获取数据矩阵 (n_channels, n_times_sliced)
                full_sig = sliced_epoch.__array__()
                if full_sig.ndim == 3: # 如果是 (n_trials, n_chs, n_times)，做平均
                    full_sig = np.mean(full_sig, axis=0)
                
                # 更新时间轴
                times = full_times[indices]
                all_chs = sliced_epoch.labels[0]
                ch_to_idx = {ch: k for k, ch in enumerate(all_chs)}
                
                # 4. 提取目标电极的 Trace
                valid_traces = []
                for ch in comp_elecs:
                    if ch in ch_to_idx:
                        valid_traces.append(full_sig[ch_to_idx[ch], :])
                
                # --- 绘图 (关键修改) ---
                if valid_traces:
                    traces_arr = np.array(valid_traces) # (n_elecs, n_times)
                    
                    # 绘制所有单根曲线
                    # traces_arr.T 形状变为 (n_times, n_elecs)，matplotlib 会自动按列画多条线
                    ax.plot(times, traces_arr.T, color=color, linewidth=0.8, alpha=0.5)
                    
                    # (可选) 如果想叠一条加粗的平均线，解注下面两行
                    # mean_trace = np.mean(traces_arr, axis=0)
                    # ax.plot(times, mean_trace, color='k', linewidth=1.5, linestyle='--', alpha=0.8)
                    
                else:
                    ax.text(0.5, 0.5, 'No Electrodes', ha='center', transform=ax.transAxes)

            except Exception as e:
                print(f"Error plotting {comp_name} - {align_name}: {e}")
                ax.text(0.5, 0.5, 'Data Error', ha='center', transform=ax.transAxes)

            # --- 格式化美化 ---
            # 标题 (仅第一行)
            if i == 0:
                ax.set_title(f"{align_name}", fontsize=14, fontweight='bold', pad=15)
            
            # Y轴标签 (仅第一列，显示类别名和电极数)
            if j == 0:
                ax.set_ylabel(f"{comp_name}\n(n={len(valid_traces)})", fontsize=12, fontweight='bold', rotation=90)
            
            # 辅助线
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)   # 0点
            ax.axvline(x=-0.5, color='gray', linestyle='-', linewidth=1, alpha=0.5) # 基线/起始点
            
            # 去框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # X轴刻度
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5)) 
            if i == n_comps - 1:
                ax.set_xlabel('Time (s)', fontsize=12)
            
    plt.tight_layout()
    plt.show()


# ==========================================
# 使用示例 (Configuration)
# ==========================================

# 1. 准备数据列表
epochs_list = [epoc_LexDelayRep_Aud, epoc_LexDelayRep_Go, epoc_LexDelayRep_Resp]
epoch_names = ['Stimulus Aligned', 'Go Aligned', 'Response Aligned']
# 定义时间窗：画 -0.5s 到 2.0s
time_windows = [(-0.5, 2.0), (-0.5, 2.0), (-0.5, 2.0)] 

# 2. 定义颜色 (根据当前分析的脑区解注对应部分)

# --- A. lPMC Configuration ---
# colors_dict = {
#     'Auditory_continuous': [1, 0, 0],       # Red
#     'Auditory_onset':      [1, 165/255, 0]  # Orange
# }

# --- B. Spt Configuration ---
colors_dict = {
    'Auditory_motor':      [1, 0, 0],       # Red
    'Auditory_onset':      [0, 1, 0],       # Green
    'Auditory_continuous': [1, 165/255, 0]  # Orange
}

# 3. 运行
plot_individual_traces_by_dominant_comp(
    epochs_list, 
    epoch_names, 
    df_weights, 
    time_windows=time_windows, 
    colors_dict=colors_dict
)
# %%
