import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# Parameters
# ==========================================
BASE_DIR = r"D:\bsliang_Coganlabcode\coganlab_ieeg\projects\Greg_ROIs"
ROIS = ['lIFG', 'lPMC', 'Spt']
ALIGNMENTS = ['Stim', 'Go', 'Resp']
NMF_TYPES = ['Auditory_Motor', 'Auditory_Motorprep', 'Auditory_sustained', 'Auditory_transient', 'Delay']

# Sampling rate parameters
FS = 100  # 100 Hz (dt = 0.01s)
TIME_WINDOW = (0.0, 1.0) # 0 to +1 s
MAX_LAG_SAMPLES = 30     # +/- 300 ms maximum lag search window (to capture lag=25)

# ==========================================
# Data Loading and Preprocessing (Steps 1 - 3)
# ==========================================
def load_and_preprocess_data(alignment):
    """Extract data for a specific alignment and execute Steps 1-3."""
    
    all_data = []
    
    # Load NMF and Epoch data for all ROIs
    for roi in ROIS:
        # Load NMF labels
        nmf_file = os.path.join(BASE_DIR, f"{roi}_NMF_cluster.csv")
        nmf_df = pd.read_csv(nmf_file)
        
        # Extract Subject ID (e.g., 'D80-LTPS4' -> 'D80')
        nmf_df['Subject'] = nmf_df['Electrode_ID'].apply(lambda x: x.split('-')[0])
        
        # Load Epoch data
        if roi == 'lIFG':
            roi_tag = 'lIFG (vPCSA)'
        elif roi == 'lPMC':
            roi_tag = 'lPMC (dPCSA)'
        elif roi == 'Spt':
            roi_tag = 'Spt'
        epoch_file = os.path.join(BASE_DIR, roi_tag, f"{roi_tag}_{alignment}_epoch.csv")
        epoch_df = pd.read_csv(epoch_file, index_col=0) # Index is Time
        
        # Melt wide format to long format for easier merging
        epoch_long = epoch_df.reset_index().melt(id_vars='index', var_name='Electrode_ID', value_name='Signal')
        epoch_long.rename(columns={'index': 'Time'}, inplace=True)
        
        # Merge epoch data with NMF labels
        merged = pd.merge(epoch_long, nmf_df, on='Electrode_ID', how='inner')
        all_data.append(merged)
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Step 1: Use only data from patients with electrodes in more than one ROI.
    # Calculate the number of unique ROIs per subject
    roi_counts = full_df.groupby('Subject')['Group'].nunique()
    valid_subjects = roi_counts[roi_counts > 1].index
    full_df = full_df[full_df['Subject'].isin(valid_subjects)]
    
    # Step 2: For each subject separately, average electrode timeseries within each ROI separated by electrode type.
    # Average across electrodes: group by (Subject, ROI, NMF_Component, Time)
    subj_avg = full_df.groupby(['Subject', 'Group', 'NMF_Component', 'Time'])['Signal'].mean().reset_index()
    
    # Step 3: Average across subjects so that the data is separated by electrode type and ROI but collapsed across electrodes and subjects.
    # Average across subjects: group by (ROI, NMF_Component, Time)
    grand_avg = subj_avg.groupby(['Group', 'NMF_Component', 'Time'])['Signal'].mean().reset_index()
    
    return grand_avg

# ==========================================
# Correlation and Lag Computation (Steps 4 - 5)
# ==========================================
def compute_lagged_correlation(x, y, max_lag):
    """Compute Pearson r for the best lag. x and y are time-series arrays."""
    best_r = 0
    best_lag = 0
    p_val_at_best = 1.0
    
    for shift in range(-max_lag, max_lag + 1):
        # shift < 0: x leads y (x shifts left / occurs earlier)
        # shift > 0: y leads x (x shifts right / occurs later)
        if shift < 0:
            r, p = pearsonr(x[:shift], y[-shift:])
        elif shift > 0:
            r, p = pearsonr(x[shift:], y[:-shift])
        else:
            r, p = pearsonr(x, y)
            
        if abs(r) > abs(best_r):
            best_r = r
            best_lag = shift
            p_val_at_best = p
            
    return best_r, p_val_at_best, best_lag

# ==========================================
# Plotting Logic (Matching PI's format)
# ==========================================
def plot_connectivity_matrix(data, alignment, use_lag=False):
    """Plot the network graphs for the 5 Subtypes under a specific alignment."""
    # 为保存图片创建文件夹
    output_dir = os.path.join(BASE_DIR, 'connectivity')
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"{'Lagged ' if use_lag else ''}Subtype-specific cross-ROI connectivity\n({alignment}-aligned, 0 to +1 s)", 
                 fontsize=16, fontweight='bold', y=1.05)
    
    # Fixed node positions: Isosceles triangle (lPMC top, lIFG bottom-left, Spt bottom-right)
    pos = {'lPMC': (0, 1), 'lIFG': (0, 0), 'Spt': (1, 0)}
    
    for i, nmf_type in enumerate(NMF_TYPES):
        ax = axes[i]
        ax.set_title(nmf_type, fontweight='bold')
        
        # Extract 0-1s data for the current subtype
        sub_data = data[(data['NMF_Component'] == nmf_type) & 
                        (data['Time'] >= TIME_WINDOW[0]) & 
                        (data['Time'] <= TIME_WINDOW[1])]
        
        if sub_data.empty:
            ax.axis('off')
            continue
            
        # Pivot to wide format (Time as index, ROIs as columns)
        ts_df = sub_data.pivot(index='Time', columns='Group', values='Signal')
        
        # Ensure all 3 ROIs are available
        available_rois = ts_df.columns.tolist()
        
        G = nx.DiGraph() if use_lag else nx.Graph()
        G.add_nodes_from(ROIS)
        
        # Calculate pairwise connections
        pairs = [('lPMC', 'lIFG'), ('lPMC', 'Spt'), ('lIFG', 'Spt')]
        for roi1, roi2 in pairs:
            if roi1 in available_rois and roi2 in available_rois:
                x = ts_df[roi1].values
                y = ts_df[roi2].values
                
                # Skip if sample size is too small (e.g., < 10 timepoints)
                if len(x) < 10: continue 
                
                # Step 5: Lagged correlation
                if use_lag:
                    r, p, lag = compute_lagged_correlation(x, y, MAX_LAG_SAMPLES)
                    # Significance threshold (strictly p < 0.05 per PI's instruction)
                    if p < 0.05 and abs(r) > 0.4:
                        # Determine direction: lag < 0 means x (roi1) leads y (roi2)
                        if lag < 0:
                            G.add_edge(roi1, roi2, weight=r, lag=abs(lag))
                        else:
                            G.add_edge(roi2, roi1, weight=r, lag=abs(lag))
                
                # Step 4: Simple correlation (No lag)
                else:
                    r, p = pearsonr(x, y)
                    if p < 0.05 and abs(r) > 0.4:
                        G.add_edge(roi1, roi2, weight=r)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color='#1f77b4')
        nx.draw_networkx_labels(G, pos, ax=ax, font_weight='bold', font_size=12)
        
        # Draw edges
        for u, v, d in G.edges(data=True):
            r_val = d['weight']
            # Line style based on correlation sign: positive = solid, negative = dashed
            style = 'solid' if r_val > 0 else 'dashed'
            width = abs(r_val) * 8 # Line width proportional to absolute r value
            
            if use_lag:
                # Draw directed arrows
                nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], ax=ax, 
                                       width=width, style=style, arrows=True, 
                                       arrowsize=25, arrowstyle='-|>', node_size=3500)
                edge_label = f"r={r_val:.2f}\nlag={d['lag']}"
            else:
                # Draw undirected lines
                nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], ax=ax, 
                                       width=width, style=style)
                edge_label = f"{r_val:.2f}"
                
            # Draw edge labels
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v): edge_label}, 
                                         ax=ax, font_size=9, label_pos=0.4)
            
        ax.axis('off')
        
    plt.tight_layout()

    # 构建文件名并保存图片，然后关闭图形以释放内存
    lag_suffix = 'lagged' if use_lag else 'simple'
    filename = f"{alignment}_{lag_suffix}_connectivity.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    for alignment in ALIGNMENTS:
        print(f"Processing {alignment}-aligned data...")
        # Get aggregated data from Steps 1-3
        agg_data = load_and_preprocess_data(alignment)
        
        # Step 4: Simple correlation (No lag)
        plot_connectivity_matrix(agg_data, alignment, use_lag=False)
        
        # Step 5: Lagged correlation
        plot_connectivity_matrix(agg_data, alignment, use_lag=True)