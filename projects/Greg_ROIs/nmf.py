#%% Introduction
# Prepare raw data for LME encoding
import os
import pickle
import numpy as np
HOME = os.path.expanduser("~")
LAB_root = os.path.join(HOME, "Box", "CoganLab")
script_dir = os.path.dirname('D:\\bsliang_Coganlabcode\\coganlab_ieeg\\projects\\lme\\prepare_raw.py')
current_dir = os.getcwd()
if current_dir != script_dir:
    os.chdir(script_dir)
sf_dir = 'data'
with open(os.path.join('..', 'GLM', 'data', f'Lex_twin_idxes_hg.npy'), "rb") as f:
    LexDelay_twin_idxes = pickle.load(f)
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
    grp_list = []

    for elec_grp, elec_idx in zip(elec_grps, elec_idxs):
        print(f'Now Doing {elec_grp}')
        m_chs = epoc.take(list(LexDelay_twin_idxes[elec_idx]), axis=0)
        m = m_chs.take(get_time_indexs(m_chs.labels[1], t_range[0], t_range[1]), axis=1)
        m_chs_labels = m.labels[0]
        m_data = m.__array__()
        print(f'max time point {[m_data.shape[1]-1]}')

        data_list.append(m_data)
        chs_list.extend(m_chs_labels)
        grp_list.extend([elec_grp] * len(m_chs_labels))

    if data_list:
        return np.vstack(data_list), chs_list, grp_list
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
    data_LexDelay_Aud,_=gp.load_stats('mask','Auditory_inRep','ave',stats_root_delay,stats_root_delay,cbind_subjs=False)
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
final_grps = None

if groupsTag=="LexDelay":
    elec_grps=('Spt','lPMC','lIFG')
    elec_idxs=('Hikock_Spt','Hikock_lPMC','Hikock_lIFG')
    # elec_grps=('Sensorymotor_in_Delay',)
    # elec_idxs=('LexDelay_Sensorimotor_in_Delay_sig_idx',)
    for epoc,t_range in zip((epoc_LexDelayRep_Aud,epoc_LexDelayRep_Go,epoc_LexDelayRep_Resp),
                             ([-0.5, 2], [-0.5, 1.5], [-0.5, 2])):
        curr_arr, curr_chs, curr_grps = rearrange_elects(elec_grps, elec_idxs, epoc,t_range=t_range)
        arrays_to_hstack.append(curr_arr)
    
        if final_chs is None:
            final_chs = curr_chs
            final_grps = curr_grps

    final_array = np.hstack(arrays_to_hstack)

if final_array.min() < 0:
    print(f"Negative values detected (min={final_array.min():.2f}). Shifting data to satisfy NMF non-negativity constraint...")
    final_array = final_array - final_array.min()

#%% NMF verification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

def evaluate_nmf_stability(X, k_range, n_repeats=50):
    """
    Evaluate NMF model stability and reconstruction error across a range of ranks (k).
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data matrix with shape (n_channels, n_timepoints). 
        Must be non-negative (e.g., High-Gamma envelope).
    k_range : list or range
        Range of components (ranks) to test (e.g., range(2, 10)).
    n_repeats : int
        Number of NMF runs with random initialization for each k to estimate stability.
        
    Returns:
    --------
    results : dict
        Dictionary containing 'k_vals', 'rss' (reconstruction errors), 
        and 'cophenetic' (stability scores).
    """
    
    # Initialize lists to store metrics
    rss_scores = []       # Reconstruction errors (Frobenius norm)
    cophenetic_scores = [] # Stability scores (0 to 1, higher is better)
    
    print(f"Starting NMF stability evaluation for k = {k_range[0]} to {k_range[-1]}...")
    
    for k in k_range:
        # Initialize consensus matrix: (n_channels x n_channels)
        # Stores how often two channels are clustered together
        consensus_matrix = np.zeros((X.shape[0], X.shape[0]))
        
        current_rss_list = []
        
        for i in range(n_repeats):
            # Run NMF with random initialization to test robustness
            # solver='cd' (Coordinate Descent) is standard for NMF
            model = NMF(n_components=k, init='random', random_state=None, 
                        max_iter=500, solver='cd', tol=1e-4)
            
            W = model.fit_transform(X)
            
            # Record reconstruction error (Residual Sum of Squares)
            current_rss_list.append(model.reconstruction_err_)
            
            # Determine cluster assignments (Hard Clustering)
            # Assign each channel to the component with the highest weight
            labels = np.argmax(W, axis=1)
            
            # Update consensus matrix
            # If channel i and j have the same label, add 1 to entry (i, j)
            # Using broadcasting for efficiency
            connectivity = (labels[:, None] == labels[None, :]).astype(float)
            consensus_matrix += connectivity
            
        # Normalize consensus matrix by the number of repeats (values between 0 and 1)
        consensus_matrix /= n_repeats
        
        # --- Calculate Cophenetic Correlation Coefficient (CCC) ---
        # Convert consensus matrix to distance matrix (Distance = 1 - Similarity)
        dist_matrix = 1 - consensus_matrix
        
        # Extract upper triangle for scipy linkage
        dist_vec = squareform(dist_matrix)
        
        # Hierarchical clustering (Average Linkage)
        Z = linkage(dist_vec, method='average')
        
        # Calculate CCC: Correlation between original distances and dendrogram distances
        c, _ = cophenet(Z, dist_vec)
        
        # Store average metrics for this k
        rss_scores.append(np.mean(current_rss_list))
        cophenetic_scores.append(c)
        
        print(f"Rank k={k}: RSS={np.mean(current_rss_list):.4f}, Cophenetic Coeff={c:.4f}")

    return {
        'k_vals': list(k_range),
        'rss': rss_scores,
        'cophenetic': cophenetic_scores
    }

def plot_stability_metrics(results):
    """
    Plot Reconstruction Error (Elbow Curve) and Stability (Cophenetic Coefficient).
    """
    k_vals = results['k_vals']
    rss = results['rss']
    ccc = results['cophenetic']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot RSS (Elbow Method)
    color = 'tab:red'
    ax1.set_xlabel('Number of Components (k)', fontsize=12)
    ax1.set_ylabel('Reconstruction Error (RSS)', color=color, fontsize=12)
    ax1.plot(k_vals, rss, 'o--', color=color, lw=2, label='RSS (Fit)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Cophenetic Correlation (Stability) on secondary y-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Cophenetic Correlation (Stability)', color=color, fontsize=12)
    ax2.plot(k_vals, ccc, 's-', color=color, lw=2, label='CCC (Stability)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('NMF Model Selection: Stability vs. Fit', fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Usage Example ---
# Assuming 'data_matrix' is your prepared (n_channels, n_timepoints) array
metrics = evaluate_nmf_stability(final_array, k_range=range(2, 10))
plot_stability_metrics(metrics)

#%% NMF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# --- 1. Data Preprocessing (NMF requires non-negative input) ---
# Check if negative values exist in the data
X = final_array.copy()
if X.min() < 0:
    print(f"Negative values detected (min={X.min():.2f}). Shifting data to satisfy NMF non-negativity constraint...")
    X = X - X.min()

# --- 2. Run NMF ---
n_components = 4
# init='nndsvd' typically yields more consistent and sparse results
model = NMF(n_components=n_components, init='nndsvd', random_state=42, max_iter=500)

# W: (n_samples, n_components) -> e.g., (91, 3) 
# Contribution weights of each electrode (sample) to each component
W = model.fit_transform(X)

# H: (n_components, n_features) -> e.g., (3, 900)
# Time series (Trace) for each component
H = model.components_

# --- 3. Plotting: Show Traces of Components ---
plt.figure(figsize=(12, 4))
colors = [
    '#1f77b4', # Blue
    '#ff7f0e', # Orange
    '#2ca02c', # Green
    '#d62728', # Red
    '#7f7f7f', # Grey
    '#9467bd', # Purple
    '#8c564b'  # Brown
]
for i in range(n_components):
    plt.plot(H[i], label=f'Component {i+1}', color=colors[i], linewidth=2)

plt.title('NMF Temporal Components (Traces)')
plt.xlabel('Time Points')
plt.ylabel('Amplitude (Arbitrary Unit)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 4. Generate DataFrame: Electrode Contributions ---
# Create base DataFrame
df_weights = pd.DataFrame(W, columns=[f'Comp_{i+1}_Weight' for i in range(n_components)])

# Insert electrode information (Channel and Group)
df_weights.insert(0, 'Channel', final_chs)
df_weights.insert(1, 'Group', final_grps)

# (Optional) Identify the dominant component for each electrode
df_weights['Dominant_Comp'] = df_weights[[f'Comp_{i+1}_Weight' for i in range(n_components)]].idxmax(axis=1)

# Display first few rows
print("Electrode Weights DataFrame (First 5 rows):")

# If you want to see the average contribution of each Group to different Components, run this:
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Calculate Mean Weights per Group ---
# (Exactly as you requested)
df_mean_weights = df_weights.groupby('Group')[[f'Comp_{i+1}_Weight' for i in range(n_components)]].mean()

# --- 2. Setup Colors ---
# Create a consistent color map for the groups so "Hickok_Spt" is the same color in all charts
groups = df_mean_weights.index
# Use a colormap (e.g., 'tab20' is good for categorical data)
colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(groups)))
color_dict = dict(zip(groups, colors))

# --- 3. Plot Pie Charts ---
fig, axes = plt.subplots(1, 4, figsize=(18, 6))

components_to_plot = [0,1,2,3]  # Corresponds to Components 4, 5, and 6

for i, comp_idx in enumerate(components_to_plot):
    col_name = f'Comp_{comp_idx+1}_Weight'
    ax = axes[i]  # Use the enumerated index for the subplot axis
    
    # Get data for this component
    values = df_mean_weights[col_name]
    
    # Draw Pie Chart
    # autopct='%1.1f%%' displays the percentage
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=groups, 
        autopct='%1.1f%%', 
        colors=[color_dict[g] for g in groups], # Ensure consistent coloring
        startangle=140,
        textprops={'fontsize': 30}
    )
    
    ax.set_title(f'Component {comp_idx+1}\nMean Weight Distribution', fontsize=14, fontweight='bold')

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

    # 3. Determine 'Hard Clustering' (Dominant Component)
    # Identifies which component has the highest weight for each electrode
    df['Dominant_Cluster'] = df[comp_cols].idxmax(axis=1)
    
    # 4. Calculate Relative Weights (Percentage contribution)
    # Useful for analyzing network composition regardless of signal amplitude
    # Add epsilon to avoid division by zero
    row_sums = df[comp_cols].sum(axis=1) + 1e-9 
    
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

# Export to CSV for further analysis
# df_weights.to_csv('NMF_Electrode_Weights.csv', index=False)
#%% Plot NMF Components by ROI Group
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_nmf_components_by_roi_group(sig, chs, df_weights, times=None, group_col='Group', fs=100):
    """
    Plots weighted NMF component traces separated by anatomical/functional ROI groups.
    
    Fix:
    - Converts 'times' to float to ensure numerical x-axis ticks work correctly.
    """
    
    # --- 1. Data Preparation ---
    
    # Handle 3D Epochs data
    if sig.ndim == 3:
        print("Detected 3D data, averaging over trials...")
        sig = np.mean(sig, axis=0)
        
    ch_to_idx = {name: i for i, name in enumerate(chs)}
    
    # --- CRITICAL FIX: Ensure Time Axis is Float ---
    if times is not None:
        # Convert strings to floats to allow numerical ticking (0.25s)
        try:
            time_axis = np.array(times, dtype=float)
        except ValueError:
            print("Warning: Could not convert 'times' to float. Using index-based time.")
            time_axis = np.arange(sig.shape[1]) / fs
    else:
        time_axis = np.arange(sig.shape[1]) / fs
    
    comp_cols = [c for c in df_weights.columns if c.startswith('Comp_') and not c.endswith('Percent')]
    
    if group_col not in df_weights.columns:
        raise KeyError(f"Column '{group_col}' not found in df_weights.")
        
    unique_groups = df_weights[group_col].dropna().unique()
    n_groups = len(unique_groups)
    
    # Setup Figure
    fig, axes = plt.subplots(nrows=n_groups, ncols=1, figsize=(10, 4 * n_groups), sharex=True)
    if n_groups == 1: axes = [axes]
    
    colors = plt.cm.tab10.colors 
    
    # --- 2. Iterate Groups ---
    for i, group_name in enumerate(unique_groups):
        ax = axes[i]
        roi_df = df_weights[df_weights[group_col] == group_name]
        has_data = False 
        
        # --- 3. Iterate Components ---
        for j, comp_name in enumerate(comp_cols):
            color = colors[j % len(colors)]
            valid_traces = []
            valid_weights = []
            
            for _, row in roi_df.iterrows():
                ch_name = row['Channel']
                weight = row[comp_name]
                
                if weight < 0.01 or ch_name not in ch_to_idx:
                    continue
                    
                idx = ch_to_idx[ch_name]
                raw_trace = sig[idx, :]
                
                if not np.isnan(raw_trace).any():
                    valid_traces.append(raw_trace)
                    valid_weights.append(weight)
            
            # --- 4. Weighted Average ---
            if len(valid_traces) > 0:
                has_data = True
                traces_arr = np.array(valid_traces)
                weights_arr = np.array(valid_weights)
                
                total_weight = np.sum(weights_arr)
                weighted_mean = np.sum(traces_arr * weights_arr[:, np.newaxis], axis=0) / total_weight
                
                ax.plot(time_axis, weighted_mean, color=color, linewidth=2, label=f"{comp_name}")
                
        # --- 5. Formatting ---
        ax.set_title(f"ROI Group: {group_name} (n={len(roi_df)} electrodes)", fontweight='bold')
        ax.set_ylabel('Weighted Amplitude')
        
        # Style updates
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set Ticks: Every 0.25s
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        
        if has_data:
            ax.legend(loc='upper right', fontsize='small', ncol=1, frameon=False)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', transform=ax.transAxes)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

# 2. Prepare Data
# Extract time range -0.5 to 2.0
# Assuming 'get_time_indexs' returns indices for slicing
for n,t_range in zip((epoc_LexDelayRep_Aud, epoc_LexDelayRep_Go, epoc_LexDelayRep_Resp),
                      ([-0.5, 2], [-0.5, 2], [-0.5, 2])):
    m = n.take(get_time_indexs(n.labels[1], t_range[0], t_range[1]), axis=1)

    sig = m.__array__()  # Signal data
    chs = m.labels[0]    # Channel names
    times = m.labels[1]  # Time points array

    # 3. Run plotting with the 'times' argument
    plot_nmf_components_by_roi_group(sig, chs, df_weights, times=times, group_col='Group')# %%

# %%
