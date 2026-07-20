import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory path
base_dir = r"C:\Users\bl314\Box\CoganLab\BIDS-1.0_LexicalDecRepDelay\BIDS\derivatives\clean"

# Find all sub-D* subject directories
subj_dirs = sorted(glob.glob(os.path.join(base_dir, "sub-D*")))

subject_rts = {}
all_rts = []

for subj_dir in subj_dirs:
    subj_id = os.path.basename(subj_dir)  # e.g., sub-D0055
    ieeg_dir = os.path.join(subj_dir, "ieeg")
    
    if not os.path.exists(ieeg_dir):
        continue
    
    # Find all *_desc-clean_events.tsv files for the current subject
    tsv_files = sorted(glob.glob(os.path.join(ieeg_dir, "*_desc-clean_events.tsv")))
    subj_rts = []
    
    for tsv_file in tsv_files:
        try:
            df = pd.read_csv(tsv_file, sep='\t')
        except Exception as e:
            print(f"Failed to read file {tsv_file}: {e}")
            continue
            
        if 'trial_type' not in df.columns or 'onset' not in df.columns:
            continue
            
        trial_types = df['trial_type'].astype(str).values
        onsets = df['onset'].values
        
        # Iterate through events within the current run to pair 'Go' and the immediately following 'Resp'
        for i in range(len(df) - 1):
            curr_type = trial_types[i]
            next_type = trial_types[i + 1]
            
            if curr_type.startswith('Go') and next_type.startswith('Resp'):
                rt = onsets[i + 1] - onsets[i]
                # Filter out negative or unrealistic RT values (0 to 3.0s window)
                if 0 < rt < 3.0:
                    subj_rts.append(rt)
                    
    if subj_rts:
        subject_rts[subj_id] = subj_rts
        all_rts.extend(subj_rts)
        print(f"{subj_id}: Extracted {len(subj_rts)} valid RTs, mean = {np.mean(subj_rts):.3f}s")


# =============================================================================
# Figure 1: Individual Subjects' RT Distributions
# =============================================================================
plt.figure(figsize=(10, 6), dpi=300)
sns.set_theme(style="ticks")

for subj_id, rts in subject_rts.items():
    # Kernel Density Estimation for individual subjects
    sns.kdeplot(rts, label=f"{subj_id} (n={len(rts)})", linewidth=1.8, alpha=0.7)

plt.title("Individual Reaction Time Distributions Across Subjects", fontsize=14, pad=12)
plt.xlabel("Reaction Time (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim([0, 2.0])  # Adjust x-axis range according to expected RTs

sns.despine()
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
plt.tight_layout()

fig1_path = os.path.join(base_dir, "RT_distribution_individual_subjects.png")
#plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.show()


# =============================================================================
# Figure 2: Pooled RT Distribution Across All Trials
# =============================================================================
mean_rt = np.mean(all_rts)
median_rt = np.median(all_rts)
std_rt = np.std(all_rts)

print("\n--- Pooled Statistics ---")
print(f"Total Subjects: {len(subject_rts)}")
print(f"Total Pooled Trials: {len(all_rts)}")
print(f"Mean RT: {mean_rt:.3f} s")
print(f"Median RT: {median_rt:.3f} s")
print(f"SD: {std_rt:.3f} s")

plt.figure(figsize=(8, 5), dpi=300)
sns.set_theme(style="ticks")

# Plot pooled histogram with KDE overlay
sns.histplot(
    all_rts, 
    kde=True, 
    stat="density", 
    bins=40, 
    color="#2b5c8f", 
    edgecolor="white", 
    linewidth=0.5,
    alpha=0.6
)

# Reference lines for mean and median
plt.axvline(mean_rt, color='#d95f02', linestyle='--', linewidth=2, label=f'Mean: {mean_rt:.3f} s')
plt.axvline(median_rt, color='#7570b3', linestyle=':', linewidth=2, label=f'Median: {median_rt:.3f} s')

plt.title(f"Pooled Reaction Time Distribution (N = {len(all_rts)} trials, {len(subject_rts)} subjects)", fontsize=12, pad=12)
plt.xlabel("Reaction Time (s)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.xlim([0, 2.0])

sns.despine()
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(frameon=False, fontsize=10)
plt.tight_layout()

fig2_path = os.path.join(base_dir, "RT_distribution_pooled.png")
#plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.show()