import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings

# Suppress LMM convergence warnings for a clean console output
warnings.filterwarnings("ignore") 

# ==========================================
# Parameters
# ==========================================
BASE_DIR = r"D:\bsliang_Coganlabcode\coganlab_ieeg\projects\Greg_ROIs"
OUTPUT_DIR = os.path.join(BASE_DIR, "connectivity")
ROIS = ['lIFG', 'lPMC', 'Spt']
ALIGNMENTS = ['Stim', 'Go', 'Resp']
NMF_TYPES = ['Auditory_Motor', 'Auditory_Motorprep', 'Auditory_sustained', 'Auditory_transient', 'Delay']

# Sampling parameters
FS = 100  # 100 Hz (dt = 0.01s)
TIME_WINDOW = (0.0, 1.0) # Core analysis window: 0 to +1 s

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Data Loading and Preprocessing
# ==========================================
def load_and_preprocess_data(alignment):
    """
    Load data, merge NMF labels, and average electrodes of the same type 
    within each ROI, separately for EACH SUBJECT.
    """
    all_data = []
    
    for roi in ROIS:
        # Load NMF labels
        nmf_file = os.path.join(BASE_DIR, f"{roi}_NMF_cluster.csv")
        nmf_df = pd.read_csv(nmf_file)
        
        # Extract Subject ID
        nmf_df['Subject'] = nmf_df['Electrode_ID'].apply(lambda x: x.split('-')[0])
        
        # Match specific folder tags
        if roi == 'lIFG':
            roi_tag = 'lIFG (vPCSA)'
        elif roi == 'lPMC':
            roi_tag = 'lPMC (dPCSA)'
        elif roi == 'Spt':
            roi_tag = 'Spt'
            
        epoch_file = os.path.join(BASE_DIR, roi_tag, f"{roi_tag}_{alignment}_epoch.csv")
        epoch_df = pd.read_csv(epoch_file, index_col=0)
        
        # Wide to long format
        epoch_long = epoch_df.reset_index().melt(id_vars='index', var_name='Electrode_ID', value_name='Signal')
        epoch_long.rename(columns={'index': 'Time'}, inplace=True)
        
        merged = pd.merge(epoch_long, nmf_df, on='Electrode_ID', how='inner')
        all_data.append(merged)
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Step 1: Retain subjects with electrodes in more than one ROI
    roi_counts = full_df.groupby('Subject')['Group'].nunique()
    valid_subjects = roi_counts[roi_counts > 1].index
    full_df = full_df[full_df['Subject'].isin(valid_subjects)]
    
    # Step 2: Average electrode timeseries within each ROI separated by NMF type 
    # for EACH subject separately. (Crucial for LMM input)
    subj_avg = full_df.groupby(['Subject', 'Group', 'NMF_Component', 'Time'])['Signal'].mean().reset_index()
    
    return subj_avg

# ==========================================
# LMM Connectivity & Plotting (Triangle Layout)
# ==========================================
def plot_lmm_connectivity_matrix(data, alignment):
    """
    Fit LMM on subject-averaged ROI signals and plot right-angled triangle graphs.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"LMM Fixed-Effect cross-ROI connectivity\n({alignment}-aligned, 0 to +1 s)", 
                 fontsize=16, fontweight='bold', y=1.05)
    
    # Right-angled triangle positions
    pos = {'lPMC': (0, 1), 'lIFG': (0, 0), 'Spt': (1, 0)}
    
    for i, nmf_type in enumerate(NMF_TYPES):
        ax = axes[i]
        ax.set_title(nmf_type, fontweight='bold')
        
        # Filter for the 0-1s window
        sub_data = data[(data['NMF_Component'] == nmf_type) & 
                        (data['Time'] >= TIME_WINDOW[0]) & 
                        (data['Time'] <= TIME_WINDOW[1])]
        
        if sub_data.empty:
            ax.axis('off')
            continue
            
        # Pivot: rows = (Subject, Time), columns = ROI
        ts_df = sub_data.pivot(index=['Subject', 'Time'], columns='Group', values='Signal').reset_index()
        available_rois = ts_df.columns.tolist()
        
        G = nx.Graph()
        G.add_nodes_from(ROIS)
        
        # Fit LMM for each pair
        pairs = [('lPMC', 'lIFG'), ('lPMC', 'Spt'), ('lIFG', 'Spt')]
        for roi1, roi2 in pairs:
            if roi1 in available_rois and roi2 in available_rois:
                # Drop instances where a subject doesn't have the NMF type in one of the ROIs
                df_pair = ts_df[['Subject', roi1, roi2]].dropna()
                
                # Minimum sample check for LMM
                if len(df_pair) >= 10 and df_pair['Subject'].nunique() >= 2: 
                    try:
                        # LMM Formula: ROI1 ~ ROI2 + (1 | Subject)
                        md = smf.mixedlm(f"{roi1} ~ {roi2}", df_pair, groups=df_pair["Subject"])
                        mdf = md.fit(disp=False)
                        
                        p_val = mdf.pvalues[roi2]
                        beta = mdf.params[roi2]
                        
                        # Only plot significant edges
                        if p_val < 0.05:
                            G.add_edge(roi1, roi2, weight=beta)
                    except Exception:
                        continue

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=4000, node_color='#1f77b4')
        nx.draw_networkx_labels(G, pos, ax=ax, font_weight='bold', font_size=14)
        
        # Draw edges
        for u, v, d in G.edges(data=True):
            beta_val = d['weight']
            style = 'solid' if beta_val > 0 else 'dashed'
            width = min(abs(beta_val) * 8, 10) # Cap line width
            
            nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], ax=ax, width=width, style=style)
            
            edge_label = f"β={beta_val:.2f}"
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(u,v): edge_label}, 
                ax=ax, font_size=9, label_pos=0.5,
                bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.9, boxstyle='round,pad=0.2')
            )
            
        ax.axis('off')
        
        # Prevent edge cropping
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.4, 1.4)
        
    plt.tight_layout()

    # Save figure
    filename = f"{alignment}_LMM_Triangle_connectivity.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  -> Saved: {filename}")
    plt.close(fig)

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    print(f"Output directory initialized at: {OUTPUT_DIR}\n")
    
    for alignment in ALIGNMENTS:
        print(f"Processing {alignment}-aligned data with LMM...")
        agg_data = load_and_preprocess_data(alignment)
        plot_lmm_connectivity_matrix(agg_data, alignment)
        
    print("\nAll done! Check the connectivity folder for the triangle plots.")