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


if groupsTag=="LexDelay&LexNoDelay":
    # epoc_LexDelayRep_Aud, _ = gp.load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_delay,trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)
    # epoc_LexNoDelay_Aud, _ = gp.load_stats('zscore', 'Auditory_inRep', 'epo', stats_root_nodelay, stats_root_nodelay,trial_labels=trial_labels,keeptrials=True,cbind_subjs=cbind_subjs)
    data_LexNoDelay_Aud,_=gp.load_stats('mask','Auditory_inRep','ave',stats_root_nodelay,stats_root_nodelay)
    elec_labels=data_LexNoDelay_Aud.labels[0]

arrays_to_hstack = []
final_chs = None
final_grps = None

if groupsTag=="LexDelay":
    elec_grps=('Spt','lPMC','lIPL','lIFG')
    elec_idxs=('Hikock_Spt','Hikock_lPMC','Hikock_lIPL','Hikock_lIFG')
    for epoc,t_range in zip((data_LexDelay_Aud,data_LexDelay_Go,data_LexDelay_Resp),
                             ([-0.5, 2], [-0.5, 1.5], [-0.5, 2])):
        curr_arr, curr_chs, curr_grps = rearrange_elects(elec_grps, elec_idxs, epoc,t_range=t_range)
        arrays_to_hstack.append(curr_arr)
    
        if final_chs is None:
            final_chs = curr_chs
            final_grps = curr_grps

    final_array = np.hstack(arrays_to_hstack)

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
n_components = 7
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
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

components_to_plot = [4, 5, 6]  # Corresponds to Components 4, 5, and 6

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
