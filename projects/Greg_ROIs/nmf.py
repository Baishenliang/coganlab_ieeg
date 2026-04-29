#%% Introduction
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns
import itertools
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist, squareform
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.append(os.path.abspath(os.path.join("..", "..")))
import utils.group as gp
from ieeg.calc.fast import mixup

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

LexDelay_twin_idxes['LexDelay_Sensorimotor_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Sensorimotor_sig_idx']-LexDelay_twin_idxes['LexDelay_Sensorimotor_in_Delay_sig_idx']
LexDelay_twin_idxes['LexDelay_Auditory_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Aud_NoMotor_sig_idx']-LexDelay_twin_idxes['LexDelay_Auditory_in_Delay_sig_idx']
LexDelay_twin_idxes['LexDelay_Motor_novWM_sig_idx']=LexDelay_twin_idxes['LexDelay_Motor_sig_idx']-LexDelay_twin_idxes['LexDelay_Motor_in_Delay_sig_idx']

#%% function block
mean_word_len = 0.65 
auditory_decay = 0 
delay_len = 1.125 

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

#%% groups of patients
datasource = 'hg' 
groupsTag = "LexDelay"

#%% define condition and load data
stat_type = 'mask'
contrast = 'ave' 
trial_labels = 'CORRECT'

#%% Sort data and get significant electrode lists
stats_root_delay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepDelay', 'BIDS', "derivatives", "stats")
stats_root_nodelay = os.path.join(LAB_root, 'BIDS-1.0_LexicalDecRepNoDelay', 'BIDS', "derivatives", "stats")

if groupsTag=="LexDelay":
    data_LexDelay_Aud,subjs=gp.load_stats('mask','Auditory_inRep','ave',stats_root_delay,stats_root_delay,cbind_subjs=False)
    elec_labels=data_LexDelay_Aud.labels[0]
    #data_LexDelay_Cue, _ = gp.load_stats('mask', 'Cue_inRep', 'ave', stats_root_delay, stats_root_delay,cbind_subjs=False)
    data_LexDelay_Go, _ = gp.load_stats('mask', 'Go_inRep', 'ave', stats_root_delay, stats_root_delay,cbind_subjs=False)
    data_LexDelay_Resp, _ = gp.load_stats('mask', 'Resp_inRep', 'ave', stats_root_delay, stats_root_delay,cbind_subjs=False)

    epoc_LexDelayRep_Aud,_=gp.load_stats('zscore','Auditory_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=False,cbind_subjs=False)
    epoc_LexDelayRep_Go,_=gp.load_stats('zscore','Go_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=False,cbind_subjs=False)
    epoc_LexDelayRep_Resp,_=gp.load_stats('zscore','Resp_inRep','epo',stats_root_delay,stats_root_delay,trial_labels=trial_labels,keeptrials=False,cbind_subjs=False)

    chs_coor=gp.get_coor(data_LexDelay_Aud.labels[0],'group')

arrays_to_hstack = []
final_chs = None
final_times = []
final_grps = None

if groupsTag=="LexDelay":
    elec_grps=('Spt','lPMC','lIFG')
    elec_idxs=('Hikock_Spt','Hikock_lPMC','Hikock_lIFG')
    
    for epoc,t_range,epoc_tag in zip((epoc_LexDelayRep_Aud,epoc_LexDelayRep_Go,epoc_LexDelayRep_Resp),
                                     ([-0.5, 1.5], [-0.5, 1], [-0.5, 1.25]),
                                     ('Stim','Go','Resp')):
        curr_arr, curr_chs, curr_times, curr_grps = rearrange_elects(elec_grps, elec_idxs, epoc,t_range=t_range)
        
        df_final = pd.DataFrame(curr_arr.T, index=curr_times, columns=curr_chs)
        if not os.path.exists(os.path.join(script_dir, sf_dir)):
            os.makedirs(os.path.join(script_dir, sf_dir))
        df_final.to_csv(os.path.join(script_dir, sf_dir, f"{'_'.join(elec_grps)}_{epoc_tag}_zscore_epo.csv"))
        
        arrays_to_hstack.append(curr_arr)
        final_times.extend(curr_times)
    
        if final_chs is None:
            final_chs = curr_chs
            final_grps = curr_grps

    final_array = np.hstack(arrays_to_hstack)
    final_times = np.array(final_times, dtype=float)

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
    print(f"Negative values detected (min={final_array.min():.2f}). Shifting data to satisfy NMF constraint...")
    final_array = final_array - final_array.min()

#%% NMF verification
def evaluate_nmf_stability(X, k_range, n_repeats=100):
    total_ss = np.linalg.norm(X, ord='fro')**2
    rss_scores = [] 
    ev_scores = []
    cophenetic_scores = []
    
    print(f"Starting NMF evaluation for k = {k_range[0]} to {k_range[-1]}...")
    for k in k_range:
        consensus_matrix = np.zeros((X.shape[0], X.shape[0]))
        current_rss_list = []
        
        for i in range(n_repeats):
            model = NMF(n_components=k, init='random', random_state=None, max_iter=500, solver='cd', tol=1e-4)
            W = model.fit_transform(X)
            rss = model.reconstruction_err_**2
            current_rss_list.append(rss)
            labels = np.argmax(W, axis=1)
            connectivity = (labels[:, None] == labels[None, :]).astype(float)
            consensus_matrix += connectivity
            
        consensus_matrix /= n_repeats
        dist_matrix = 1 - consensus_matrix
        dist_vec = squareform(dist_matrix)
        Z = linkage(dist_vec, method='average')
        c, _ = cophenet(Z, dist_vec)
        
        avg_rss = np.mean(current_rss_list)
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
    k_vals = results['k_vals']
    ev = results['explained_variance']
    ccc = results['cophenetic']
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Number of Components (k)', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', color=color, fontsize=12)
    ax1.plot(k_vals, ev, 'o--', color=color, lw=2, label='Explained Variance (Fit)')
    ax1.tick_params(axis='y', labelcolor=color)
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Cophenetic Correlation (Stability)', color=color, fontsize=12)
    ax2.plot(k_vals, ccc, 's-', color=color, lw=2, label='Stability (CCC)')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('NMF Model Selection: Stability vs. Fit', fontsize=14)
    plt.tight_layout()
    plt.show()

metrics = evaluate_nmf_stability(final_array, k_range=range(2, 10))
plot_stability_metrics(metrics)

#%% Auto define optimal k using BIC
def auto_find_optimal_k_nmf(X, k_range=range(2, 10), plot=True):
    n_samples, n_features = X.shape
    n_datapoints = n_samples * n_features
    bic_scores = []
    
    print("正在计算各 k 值的 BIC 准则...")
    for k in k_range:
        model = NMF(n_components=k, init='nndsvd', max_iter=500, random_state=42)
        model.fit(X)
        rss = model.reconstruction_err_**2
        n_parameters = k * (n_samples + n_features)
        bic = n_datapoints * np.log(rss / n_datapoints) + n_parameters * np.log(n_datapoints)
        bic_scores.append(bic)
        print(f"  k={k} | RSS={rss:.2f} | Parameters={n_parameters} | BIC={bic:.2f}")

    optimal_k = k_range[np.argmin(bic_scores)]
    print(f"\n✅ 根据 BIC 准则，最佳的 Component 数量 (k) 是: {optimal_k}")
    
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(k_range, bic_scores, 'o-', color='tab:purple', linewidth=2, markersize=8)
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
        plt.savefig(os.path.join(save_dir, save_filename), format='svg', dpi=300, bbox_inches='tight')
        plt.show()
        
    return optimal_k

optimal_k = auto_find_optimal_k_nmf(final_array, k_range=range(2, 10))

#%% 0. Global Setup & Custom Colors
Sensorimotor_col = [1, 0, 0]  
Auditory_col = [0, 1, 0]  
Motor_col = [0, 0, 1]  
Delay_col = [1, 0.65, 0]
Yellow_col = [1, 1, 0]        # 黄色 (RGB: 255, 255, 0)
Purple_col = [0.5, 0, 0.5]    # 紫色 (RGB: 128, 0, 128)

macro_color_dict = {
    'Auditory_Motorprep': Yellow_col, 
    'Auditory_sustained': Delay_col, 
    'Delay': Auditory_col, 
    'Auditory_transient': Motor_col, 
    'Auditory_Motor': Purple_col 
}


#%% 1. NMF Data Preprocessing & Execution
X = final_array.copy()
if X.min() < 0:
    X = X - X.min()

n_components = 5
model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

W = model.fit_transform(X)
H = model.components_
comp_names = [list(macro_color_dict.keys())[i] for i in range(n_components)]

#%% 2. Plot NMF Component Traces (H Matrix)
target_macros = ['Auditory_transient', 'Auditory_sustained', 'Delay', 'Auditory_Motorprep', 'Auditory_Motor']
name_to_idx = {name: i for i, name in enumerate(comp_names)}

plt.figure(figsize=(10, 4))

for name in target_macros:
    if name in name_to_idx:
        idx = name_to_idx[name]
        plt.plot(x_linear, H[idx, :], label=name, color=macro_color_dict.get(name), linewidth=2.5, alpha=0.85)

for idx in zero_indices:
    plt.axvline(x=idx, color='k', linestyle='--', linewidth=1, alpha=0.5)
for idx in minus_point_five_indices:
    plt.axvline(x=idx, color='k', linestyle='-', linewidth=1, alpha=0.5)

plt.title(f'Temporal Traces of the {n_components} NMF Components (H Matrix)', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Component Amplitude (a.u.)', fontsize=12)
plt.gca().xaxis.set_major_locator(mticker.FixedLocator(tick_indices))
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), frameon=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

save_dir = '../Greg_ROIs/fig'
if not os.path.exists(save_dir): os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, "Comp_traces.svg"), format='svg', dpi=300, bbox_inches='tight')
plt.show()

#%% 3. Hierarchical Clustering (Forcing 3 Macro Clusters)
# dist_matrix = pdist(H, metric='correlation')
# Z = linkage(dist_matrix, method='average')

# target_macro_clusters = 3
# macro_labels = fcluster(Z, target_macro_clusters, criterion='maxclust')

# unique_labels = sorted(np.unique(macro_labels))
# custom_macro_names = ['SM_Auditory', 'SM_Motor', 'Sustained']
# label_to_name = {label: custom_macro_names[i] for i, label in enumerate(unique_labels)}
# macro_mapping = {comp_names[i]: label_to_name[macro_labels[i]] for i in range(n_components)}

# plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['axes.linewidth'] = 2.0
# plt.rcParams['lines.linewidth'] = 2.5

# fig = plt.figure(figsize=(6, 4), dpi=300)
# ax = plt.gca()
# dendro = dendrogram(Z, labels=comp_names, orientation='top', link_color_func=lambda x: 'tab:blue', ax=ax)

# ax.spines[['top', 'right']].set_visible(False)
# ax.spines['left'].set_linewidth(3)
# ax.spines['bottom'].set_linewidth(3)
# ax.tick_params(axis='y', labelsize=16, length=6, width=2.5)
# ax.tick_params(axis='x', labelsize=16, length=0, width=2.5)
# plt.ylabel('Distance (1 - Pearson r)', fontsize=18, labelpad=10)

# sns.despine(ax=ax, offset=10, trim=True)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "Dendrogram_Macro_Clusters.svg"), format='svg', dpi=300, bbox_inches='tight')
# plt.show()
# plt.rcParams['lines.linewidth'] = 1.5

#%% 4. Hard Clustering & Saving Indices
df_weights = pd.DataFrame(W, columns=comp_names)
df_weights.insert(0, 'Channel', final_chs)
df_weights.insert(1, 'Group', final_grps)

df_weights['Base_Comp'] = df_weights[comp_names].idxmax(axis=1)

for category in df_weights['Base_Comp'].dropna().unique():
    category_chs = set(df_weights[df_weights['Base_Comp'] == category]['Channel'])
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

for i, col_name in enumerate(target_macros):
    ax = axes[i]
    values = df_mean_weights[col_name]
    valid_idx = values > 0.01
    valid_values = values[valid_idx]
    valid_groups = groups[valid_idx]
    
    if len(valid_values) == 0: continue
        
    ax.pie(valid_values, labels=valid_groups, autopct='%1.1f%%', 
           colors=[color_dict.get(g, 'gray') for g in valid_groups], startangle=140, textprops={'fontsize': 12})
    ax.set_title(f'{col_name}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

#%% 6. Plot Aligned Traces (Stim / Go / Resp)
plot_groups = []

for macro in target_macros:
    category_chs = set(df_weights[df_weights['Base_Comp'] == macro]['Channel'])
    sig_idx = [i for i, x in enumerate(data_LexDelay_Aud.labels[0]) if x in category_chs]
    plot_groups.append((sig_idx, macro, macro_color_dict.get(macro, [0.5, 0.5, 0.5])))

unit_scale = 2.0
left_padding_with_y = 1.6
left_padding_no_y = 0.2
right_padding = 0.4
fig_height = 3.0

alignments = [
    ('Stim', epoc_LexDelayRep_Aud, [-0.25, 1.5], True),
    ('Go', epoc_LexDelayRep_Go, [-0.25, 1.0], range(631, 650)),
    ('Resp', epoc_LexDelayRep_Resp, [-0.25, 1.0], range(631, 650))
]

target_macros_as = target_macros + ['All']
for align_tag, epoc_data, x_limits, bsl_val in alignments:
    has_y = (align_tag == 'Stim')
    current_left_pad = left_padding_with_y if has_y else left_padding_no_y
    x_duration = x_limits[1] - x_limits[0]
    fig_width = (x_duration * unit_scale) + current_left_pad + right_padding
    
    for macro in target_macros_as:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        fig.subplots_adjust(left=current_left_pad/fig_width, right=1.0 - (right_padding/fig_width), bottom=0.25, top=0.9)
        ax = plt.gca()

        if macro == 'All':
            for sig_idx, label_text, group_col in plot_groups:
                if len(sig_idx) == 0: continue
                gp.plot_wave(epoc_data, sig_idx, f'{label_text}', group_col, '-', bsl_val, ylim=[-0.3, 5],average_trace=True)
        else:
            sig_idx, label_text, group_col = plot_groups[target_macros.index(macro)]
            if len(sig_idx) == 0: continue
            for roi_tag, roi_elec_idx,roi_col in zip(elec_grps,
                elec_idxs,
                (Auditory_col,Sensorimotor_col,Motor_col)):
                sig_idx_roi = LexDelay_twin_idxes[roi_elec_idx] & set(sig_idx)
                if len(sig_idx_roi) == 0: continue
                gp.plot_wave(epoc_data, sig_idx_roi, f'{roi_tag}', roi_col, '-', bsl_val, ylim=[-0.3, 6],average_trace=False)
             

        #ax.set_ylim([-0.3, 5])
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
        ax.set_xticks([t for t in xticks if x_limits[0] <= t <= x_limits[1]])
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        
        plt.draw()
        ax.set_xticklabels(["0" if (l == "0.0" or l == ".0") else l for l in [l.get_text() for l in ax.get_xticklabels()]])
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
        if ax.get_legend(): ax.get_legend().remove()

        plt.savefig(os.path.join(save_dir, f"Spt_trace_{macro}_{align_tag}.svg"), format='svg', dpi=300, bbox_inches=None)
        plt.show()

    #

#%% 7. NMF Brain Maps & Hub Electrodes
ch_to_idx = {ch: i for i, ch in enumerate(final_chs)}

for comp in comp_names:
    channels = df_weights[df_weights['Base_Comp'] == comp]['Channel'].tolist()
    indices = [ch_to_idx[ch] for ch in channels if ch in ch_to_idx]
    
    if not indices: continue
    traces = final_array[indices, :]
    comp_col = macro_color_dict[comp]
    
    plt.figure(figsize=(10, 3), dpi=300)
    plt.plot(x_linear, traces.T, color=comp_col, alpha=0.3, linewidth=1.5)
    for idx in zero_indices:
        plt.axvline(x=idx, color='k', linestyle='--', linewidth=1, alpha=0.5)
    for idx in minus_point_five_indices:
        plt.axvline(x=idx, color='k', linestyle='-', linewidth=1, alpha=0.5)
        
    plt.title(f'Raw Traces | {comp} | n={len(indices)}', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.gca().xaxis.set_major_locator(mticker.FixedLocator(tick_indices))
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    plt.show()

chs_coor=gp.get_coor(df_weights.Channel.to_list(),'group')

for i, comp in enumerate(comp_names):
    w = df_weights[comp].values
    w_norm = w / w.max() if w.max() > 0 else w
    
    base_col = np.array(macro_color_dict.get(comp, [0.5, 0.5, 0.5]))
    
    cols_lst = [list(base_col * val + np.array([1, 1, 1]) * (1 - val)) for val in w_norm]
    try:
        #gp.plot_brain(subjs, df_weights.Channel.to_list(), cols_lst, None, f'Weight: {comp}', 0.3, 0.2, hemi='lh')
        gp.plot_brain(picks=df_weights.Channel.to_list(), chs_coor=chs_coor, chs_cols= cols_lst, dotsize=0.2, transparency=0.2)

    except Exception:
        pass

# cols_lst = [macro_color_dict.get(comp, [0.5, 0.5, 0.5]) for comp in df_weights['Dominant_Comp']]
# try:
#     #gp.plot_brain(subjs, df_weights.Channel.to_list(), cols_lst, None, 'Dominant_Comp', 0.3, 0.2, hemi='lh')
#     gp.plot_brain(picks=df_weights.Channel.to_list(), chs_coor=chs_coor, chs_cols= cols_lst, dotsize=0.2, transparency=0.2)
# except Exception:
#     pass

print("\n--- Provincial Hubs (Top 5 per Component) ---")
for comp in comp_names:
    print(f"\nComponent: {comp}")
    print(df_weights.nlargest(5, comp)[['Channel', 'Group', comp]])

w_matrix = df_weights[comp_names].values
w_total = w_matrix.sum(axis=1, keepdims=True)
w_total[w_total == 0] = 1e-10 
df_weights['Total_Weight'] = w_total.flatten()
df_weights['Participation_Coef'] = 1 - np.sum((w_matrix / w_total)**2, axis=1)

print("\n--- Connector Hubs (High Participation across 4 components) ---")
significant_elecs = df_weights[df_weights['Total_Weight'] > df_weights['Total_Weight'].quantile(0.75)].copy()
print(significant_elecs.nlargest(10, 'Participation_Coef')[['Channel', 'Group', 'Participation_Coef', 'Total_Weight']])

w = significant_elecs['Participation_Coef'].values
if len(w) > 0:
    w_norm = (w - w.min()) / (w.max() - w.min()) if w.max() > w.min() else w
    cols_lst = [list(np.array([1.0, 0.0784, 0.5765]) * val + np.array([1, 1, 1]) * (1 - val)) for val in w_norm]
    try:
        gp.plot_brain(subjs, significant_elecs.Channel.to_list(), cols_lst, None, 'Participation Coef (Hubs)', 0.3, 0.2, hemi='lh')
    except Exception:
        pass

#%% 8. Extract Power for Stats & Plot Brain Maps + Bar+Strip
_, _, _, _, _, paras_aud, *_ = gp.sort_chs_by_actonset(data_LexDelay_Aud, epoc_LexDelayRep_Aud, 0.011, [0.05, 0.25], mask_data=True, select_electrodes=False)
_, _, _, _, _, paras_mtr, *_  = gp.sort_chs_by_actonset(data_LexDelay_Resp, epoc_LexDelayRep_Resp, 0.011, [-0.25, -0.05], mask_data=True, select_electrodes=False)

plot_df = df_weights.dropna(subset=['Base_Comp']).copy()

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
        clusters.append(row['Base_Comp'])

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

norm_aud = mcolors.Normalize(vmin=np.percentile(aud_z, 5), vmax=np.percentile(aud_z, 95))
cmap_aud = plt.cm.Greens
cols_aud = [cmap_aud(norm_aud(val))[:3] for val in df_stats['Auditory_Power_Z']]

try:
    #gp.plot_brain(subjs, df_stats['Channel'].tolist(), cols_aud, None, 'Auditory Power Z-score (0-250ms)', 0.3, 0.2,hemi='lh')
    gp.plot_brain(picks=df_stats['Channel'].tolist(), chs_coor=chs_coor, chs_cols= cols_aud, dotsize=0.2, transparency=0.2)
except Exception:
    pass

norm_mot = mcolors.Normalize(vmin=np.percentile(mot_z, 5), vmax=np.percentile(mot_z, 95))
cmap_mot = plt.cm.Blues
cols_mot = [cmap_mot(norm_mot(val))[:3] for val in df_stats['Motor_Power_Z']]

try:
    #gp.plot_brain(subjs, df_stats['Channel'].tolist(), cols_mot, None, 'Motor Power Z-score (-250-0ms)', 0.3, 0.2,hemi='lh')
    gp.plot_brain(picks=df_stats['Channel'].tolist(), chs_coor=chs_coor, chs_cols= cols_mot, dotsize=0.2, transparency=0.2)

except Exception:
    pass

metrics = [
    ('Auditory_Power_Raw', 'Auditory z-score (avg.)', 'BarStrip_Auditory_Power_Raw.svg'),
    ('Motor_Power_Raw', 'Motor z-score (avg.)', 'BarStrip_Motor_Power_Raw.svg')
]

pairs = list(itertools.combinations(target_macros, 2))

for col, ylabel, save_name in metrics:
    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    
    plotting_params = {
        'data': df_stats, 'x': 'Macro_Cluster', 'y': col, 'order': target_macros
    }

    sns.barplot(**plotting_params, palette=macro_color_dict, 
                alpha=0.5, capsize=0.1, errwidth=2.5, ax=ax, edgecolor='none')
    
    sns.stripplot(**plotting_params, palette=macro_color_dict, 
                  jitter=True, alpha=0.8, size=6, ax=ax, rasterized=True)
    
    # 手动执行统计检验，以应用FDR校正和自定义星号
    pvalues = [mannwhitneyu(
        df_stats[df_stats['Macro_Cluster'] == p[0]][col],
        df_stats[df_stats['Macro_Cluster'] == p[1]][col],
        nan_policy='omit'
    ).pvalue for p in pairs]

    if pvalues:
        # 应用FDR (Benjamini-Hochberg) 校正
        _, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')

        # 自定义显著性阈值 (最多三个星)
        pvalue_thresholds = [(0.001, '***'), (0.01, '**'), (0.05, '*')]
        
        def p_to_stars(p):
            for thr, star in pvalue_thresholds:
                if p <= thr: return star
            return 'ns' # Not Significant
        
        annotations = [p_to_stars(p) for p in pvals_corrected]
        
        # 过滤掉不显著的配对，使图形更清晰
        annot_pairs = [pair for i, pair in enumerate(pairs) if annotations[i] != 'ns']
        annot_texts = [text for text in annotations if text != 'ns']

        if annot_pairs:
            annotator = Annotator(ax, annot_pairs, **plotting_params)
            annotator.set_custom_annotations(annot_texts)
            annotator.configure(loc='inside', verbose=0)
            annotator.annotate()

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    
    ax.tick_params(axis='y', labelsize=16, length=6, width=2.5)
    #ax.tick_params(axis='x', labelsize=14, length=6, width=2.5)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    
    plt.ylabel(ylabel, fontsize=16, fontweight='bold')
    plt.xlabel('')
    
    sns.despine(ax=ax, offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name), format='svg', dpi=300, bbox_inches='tight')
    plt.show()
# %%
# %%
import seaborn as sns
from sklearn.linear_model import LinearRegression

chs_coor_stats = gp.get_coor(df_stats['Channel'].tolist(), 'group')

analysis_targets = [
    ('Auditory_Power_Z', aud_z, np.array(plt.cm.tab10.colors[2][:3])),
    ('Motor_Power_Z', mot_z, np.array(plt.cm.tab10.colors[0][:3]))
]

for name, z_vals, base_col in analysis_targets:
    x_coords = chs_coor_stats['x'].values
    y_coords = chs_coor_stats['y'].values
    z_coords = chs_coor_stats['z'].values
    
    valid_mask = ~np.isnan(x_coords) & ~np.isnan(y_coords) & ~np.isnan(z_coords) & ~np.isnan(z_vals)
    
    xyz_valid = np.column_stack((x_coords[valid_mask], y_coords[valid_mask], z_coords[valid_mask]))
    vals_valid = z_vals[valid_mask]
    y_valid = y_coords[valid_mask] 
    
    reg = LinearRegression().fit(xyz_valid, vals_valid)
    
    grad_vector = reg.coef_
    grad_norm = np.linalg.norm(grad_vector)
    
    if grad_norm > 0:
        grad_unit = grad_vector / grad_norm
        projection_1d = np.dot(xyz_valid, grad_unit)
    else:
        projection_1d = np.zeros_like(vals_valid)

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.linewidth'] = 2.0
    
    fig1, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    
    sns.regplot(x=y_valid, y=vals_valid, ax=axes[0], color=base_col,
                scatter_kws={'alpha': 0.7, 's': 50, 'edgecolors': 'white', 'linewidths': 0.5},
                line_kws={'linewidth': 3, 'color': '#333333', 'alpha': 0.8})
    axes[0].set_title(f'{name} | Y-Axis (A-P) Gradient', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Y Coordinate (mm)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Z-scored Power', fontsize=14, fontweight='bold')
    axes[0].invert_xaxis()
    
    sns.regplot(x=projection_1d, y=vals_valid, ax=axes[1], color=base_col,
                scatter_kws={'alpha': 0.7, 's': 50, 'edgecolors': 'white', 'linewidths': 0.5},
                line_kws={'linewidth': 3, 'color': '#333333', 'alpha': 0.8})
    axes[1].set_title(f'{name} | Optimal 3D Gradient', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Optimal Gradient Projection', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Z-scored Power', fontsize=14, fontweight='bold')
    
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.tick_params(axis='both', labelsize=12, length=6, width=2.5)
        sns.despine(ax=ax, offset=10, trim=True)
        
    plt.tight_layout()
    
    save_dir = '../Greg_ROIs/fig'
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f"Gradient_Scatter_2D_{name}.svg"), format='svg', dpi=300, bbox_inches='tight')
    plt.show()

    fig2 = plt.figure(figsize=(6, 6), dpi=300)
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(xyz_valid[:, 1], xyz_valid[:, 0], xyz_valid[:, 2], 
                c=base_col.reshape(1,-1), s=40, alpha=0.6, edgecolors='white', linewidths=0.5)
    
    mean_xyz = np.mean(xyz_valid, axis=0)
    
    if grad_norm > 0:
        scale_factor = (np.max(xyz_valid[:, 1]) - np.min(xyz_valid[:, 1])) * 0.7
        arrow_vector = grad_unit * scale_factor
        
        ax2.quiver(mean_xyz[1], mean_xyz[0], mean_xyz[2], 
                   arrow_vector[1], arrow_vector[0], arrow_vector[2], 
                   color='#333333', linewidth=4, arrow_length_ratio=0.2, pivot='tail', zorder=10)

    ax2.set_xlabel('Y (P-A)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('X (L-R)', fontsize=12, fontweight='bold')
    ax2.set_zlabel('Z (I-S)', fontsize=12, fontweight='bold')
    
    ax2.invert_xaxis()
    
    x_lim = ax2.get_xlim(); y_lim = ax2.get_ylim(); z_lim = ax2.get_zlim()
    max_range = np.array([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]]).max() / 2.0
    mid_x = (x_lim[1]+x_lim[0])/2.0; mid_y = (y_lim[1]+y_lim[0])/2.0; mid_z = (z_lim[1]+z_lim[0])/2.0
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax2.view_init(elev=20, azim=-60) 
    ax2.set_title(f'{name} | Vector View', fontsize=14, fontweight='bold', pad=10)
    
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_edgecolor('white')
    ax2.yaxis.pane.set_edgecolor('white')
    ax2.zaxis.pane.set_edgecolor('white')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Gradient_3D_Vector_{name}.svg"), format='svg', dpi=300, bbox_inches='tight')
    plt.show()
# %%
