#!/usr/bin/env python3
"""
02_make_figures.py

Creates four standalone grant/share figures from outputs of 01_run_analysis.py.
No subplots are used.

Figures:
1. ESM+ rate by exact bipolar physiology class (Wilson 95% CI).
2. ESM+ rates for core binary physiological features, with within-subject
   permutation p-values.
3. Conditional odds-ratio forest plot for the feature-level analyses.
4. Participant-level distribution of motor-preparation sites and ESM+ sites.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---------------- Figure 1: exact classes ----------------
class_rates = pd.read_csv(OUT / "class_rates.csv")
class_rates = class_rates[class_rates["n_pairs"] > 0].copy()

# Ordered for mechanistic readability.
order = [
    "Sensory-motor vWM",
    "Auditory vWM",
    "Motor vWM",
    "Delay-only",
    "Auditory no-vWM",
    "Motor no-vWM",
    "Other / unclassified",
]
class_rates["liang_class"] = pd.Categorical(class_rates["liang_class"], categories=order, ordered=True)
class_rates = class_rates.sort_values("liang_class")

x = np.arange(len(class_rates))
y = class_rates["ESM_pos_rate"].to_numpy()
lower = y - class_rates["Wilson95_low"].to_numpy()
upper = class_rates["Wilson95_high"].to_numpy() - y

fig = plt.figure(figsize=(11.2, 6.5))
plt.bar(x, y)
plt.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=4)
plt.xticks(
    x,
    [
        "Sensory-motor\nvWM",
        "Auditory\nvWM",
        "Motor\nvWM",
        "Delay-only",
        "Auditory\nno-vWM",
        "Motor\nno-vWM",
        "Other /\nunclassified",
    ],
)
plt.ylabel("ESM language-positive rate")
plt.ylim(0, 1.08)
plt.title("Exact bipolar LexicalDelay physiology classes at ESM-tested sites")

for i, row in class_rates.reset_index(drop=True).iterrows():
    plt.text(
        i,
        min(row["Wilson95_high"] + 0.035, 1.02),
        f"{int(row['n_ESM_pos'])}/{int(row['n_pairs'])}",
        ha="center", va="bottom", fontsize=10
    )

plt.tight_layout()
plt.savefig(FIG / "Figure1_ESM_rate_by_exact_physiology_class.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: feature-level rates ----------------
features = pd.read_csv(OUT / "feature_associations.csv")

feature_names = ["Encoding", "Motor preparation", "Delay", "Sensory-motor vWM"]
features = features.set_index("feature").loc[feature_names].reset_index()

# Rates from the 2x2 counts.
features["present_rate"] = features["exposed_pos"] / (features["exposed_pos"] + features["exposed_neg"])
features["absent_rate"] = features["unexposed_pos"] / (features["unexposed_pos"] + features["unexposed_neg"])

# SEM for a Bernoulli proportion.
features["present_sem"] = np.sqrt(
    features["present_rate"] * (1 - features["present_rate"])
    / (features["exposed_pos"] + features["exposed_neg"])
)
features["absent_sem"] = np.sqrt(
    features["absent_rate"] * (1 - features["absent_rate"])
    / (features["unexposed_pos"] + features["unexposed_neg"])
)

x = np.arange(len(features))
width = 0.36
fig = plt.figure(figsize=(10.5, 6.6))
plt.bar(x - width/2, features["present_rate"], width=width, label="Feature present")
plt.bar(x + width/2, features["absent_rate"], width=width, label="Feature absent")
plt.errorbar(
    x - width/2, features["present_rate"], yerr=features["present_sem"],
    fmt="none", ecolor="black", capsize=4
)
plt.errorbar(
    x + width/2, features["absent_rate"], yerr=features["absent_sem"],
    fmt="none", ecolor="black", capsize=4
)

plt.xticks(x, ["Encoding", "Motor\npreparation", "Delay", "Sensory-motor\nvWM"])
plt.ylabel("ESM language-positive rate")
plt.ylim(0, 1.08)
plt.title("Bipolar physiological features associated with ESM language positivity")
plt.legend(frameon=False)

for i, row in features.iterrows():
    p = row["within_subject_perm_p_one_sided"]
    top = max(
        row["present_rate"] + row["present_sem"],
        row["absent_rate"] + row["absent_sem"]
    )
    plt.text(i, min(top + 0.10, 1.01), f"within-subj. p = {p:.3f}",
             ha="center", va="bottom", fontsize=10)
    plt.text(
        i - width/2,
        row["present_rate"] + row["present_sem"] + 0.02,
        f"{int(row['exposed_pos'])}/{int(row['exposed_pos'] + row['exposed_neg'])}",
        ha="center", va="bottom", fontsize=9
    )
    plt.text(
        i + width/2,
        row["absent_rate"] + row["absent_sem"] + 0.02,
        f"{int(row['unexposed_pos'])}/{int(row['unexposed_pos'] + row['unexposed_neg'])}",
        ha="center", va="bottom", fontsize=9
    )

plt.tight_layout()
plt.savefig(FIG / "Figure2_feature_level_ESM_associations.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 3: conditional OR forest ----------------
features = pd.read_csv(OUT / "feature_associations.csv")
features = features.set_index("feature").loc[feature_names].reset_index()

# We show two estimates for each feature:
# subject-fixed and subject×anatomy-stratified conditional logistic ORs.
y_base = np.arange(len(features))[::-1]
offset = 0.12

fig = plt.figure(figsize=(9.5, 6.2))

for i, row in features.iterrows():
    y1 = y_base[i] + offset
    y2 = y_base[i] - offset

    # Subject fixed
    if np.isfinite(row["subject_conditional_OR"]):
        xval = row["subject_conditional_OR"]
        lo = row["subject_conditional_CI_low"]
        hi = row["subject_conditional_CI_high"]
        plt.errorbar(
            xval, y1,
            xerr=np.array([[xval - lo], [hi - xval]]),
            fmt="o", capsize=3,
            label="Subject-fixed" if i == 0 else None
        )

    # Subject × anatomy
    if np.isfinite(row["subject_anatomy_conditional_OR"]):
        xval = row["subject_anatomy_conditional_OR"]
        lo = row["subject_anatomy_conditional_CI_low"]
        hi = row["subject_anatomy_conditional_CI_high"]
        plt.errorbar(
            xval, y2,
            xerr=np.array([[xval - lo], [hi - xval]]),
            fmt="s", capsize=3,
            label="Subject × anatomy" if i == 0 else None
        )

plt.axvline(1, linestyle="--", linewidth=1)
plt.xscale("log")
plt.yticks(y_base, feature_names)
plt.xlabel("Conditional odds ratio for ESM positivity (log scale)")
plt.title("Subject- and anatomy-adjusted physiological associations with ESM")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIG / "Figure3_conditional_OR_forest.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 4: participant-level motor preparation ----------------
participant = pd.read_csv(OUT / "participant_summary.csv")
participant = participant.sort_values("subject")

x = np.arange(len(participant))
fig = plt.figure(figsize=(10.5, 5.8))
plt.bar(x, participant["n_motor_prep"], label="Motor-preparation pairs")
plt.bar(x, participant["n_motor_prep_ESM_pos"], label="ESM+ among motor-preparation pairs")
plt.xticks(x, participant["subject"], rotation=45, ha="right")
plt.ylabel("Number of matched bipolar pairs")
plt.title("Motor-preparation sites by participant")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIG / "Figure4_motor_preparation_by_participant.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figures written to:", FIG)
