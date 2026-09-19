#!/usr/bin/env python3
"""
01_run_analysis.py

Maps exact bipolar LexicalDelay physiology classes onto clean ESM-tested
bipolar pairs, then runs:

1. ESM+ rate by exact Liang physiology class.
2. Primary class contrasts:
   - Sensory-motor vWM vs all other classes
   - Any vWM (delay-active) vs no-vWM/unclassified
   - Sensory-motor vWM vs Auditory vWM
   - Sensory-motor vWM vs all other vWM classes
3. Feature-level analyses:
   - encoding
   - motor preparation
   - delay
   - sensory-motor vWM
4. Subject-preserving within-subject permutation tests (macro AUC).
5. Subject-fixed and subject×anatomy conditional logistic models.
6. Joint conditional-logit models with encoding + motor preparation + delay.

Outputs are written to ../outputs as CSV files.
"""

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

N_PERM = 10000
SEED = 20260914

def canonical_pair(s):
    """Canonicalize pair labels such as LFOA14-LFOA13 or LFOA 13-14 -> LFOA13-14."""
    if pd.isna(s):
        return None
    s = str(s).strip().replace(" ", "")

    # repeated lead on both contacts, e.g. LFOA13-LFOA14
    m = re.match(r"^([A-Za-z0-9]+?)(\d+)-([A-Za-z0-9]+?)(\d+)$", s)
    if m:
        lead1, n1, lead2, n2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if lead1 == lead2:
            lo, hi = sorted((n1, n2))
            return f"{lead1}{lo}-{hi}"

    # single lead, e.g. LFOA13-14
    m = re.match(r"^([A-Za-z0-9]+?)(\d+)-(\d+)$", s)
    if m:
        lead, n1, n2 = m.group(1), int(m.group(2)), int(m.group(3))
        lo, hi = sorted((n1, n2))
        return f"{lead}{lo}-{hi}"

    return s

def auc_fast(y, score):
    """ROC AUC by pairwise ranking, with ties = 0.5."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float((np.sum(d > 0) + 0.5 * np.sum(d == 0)) / d.size)

def macro_auc_permutation(df, score_col, n_perm=N_PERM, seed=SEED):
    """
    Macro-average subject-specific AUC across subjects with both ESM+ and ESM-.
    Shuffle ESM labels within subject. One-sided p tests higher score at ESM+.
    """
    groups = []
    subject_aucs = {}

    for subject, g in df[["subject", "strict_esm_pos", score_col]].dropna().groupby("subject"):
        y = g["strict_esm_pos"].to_numpy(dtype=int)
        score = g[score_col].to_numpy(dtype=float)
        if len(np.unique(y)) == 2:
            a = auc_fast(y, score)
            subject_aucs[subject] = a
            groups.append((subject, y, score))

    if not groups:
        return np.nan, np.nan, {}, []

    observed = float(np.mean(list(subject_aucs.values())))
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm, dtype=float)

    for k in range(n_perm):
        aucs = []
        for subject, y, score in groups:
            yp = y.copy()
            rng.shuffle(yp)
            aucs.append(auc_fast(yp, score))
        perm_stats[k] = np.mean(aucs)

    p = (np.sum(perm_stats >= observed) + 1) / (n_perm + 1)
    return observed, float(p), subject_aucs, [x[0] for x in groups]

def pooled_2x2(df, pred):
    """Exposure=True vs False, outcome ESM+=1 vs 0."""
    predv = df[pred].astype(bool)
    y = df["strict_esm_pos"].astype(int)

    a = int((predv & (y == 1)).sum())
    b = int((predv & (y == 0)).sum())
    c = int((~predv & (y == 1)).sum())
    d = int((~predv & (y == 0)).sum())

    table = np.array([[a, b], [c, d]], dtype=float)
    odds, p = fisher_exact(table, alternative="two-sided")

    # Table2x2 gives asymptotic log-OR confidence interval.
    try:
        t = Table2x2(table, shift_zeros=True)
        lo, hi = t.oddsratio_confint()
    except Exception:
        lo, hi = np.nan, np.nan

    return {
        "exposed_pos": a, "exposed_neg": b,
        "unexposed_pos": c, "unexposed_neg": d,
        "pooled_OR": float(odds),
        "pooled_OR_CI_low": float(lo),
        "pooled_OR_CI_high": float(hi),
        "fisher_p_two_sided": float(p),
    }

def conditional_logit(df, pred, group_cols):
    d = df[[pred, "strict_esm_pos"] + group_cols].dropna().copy()
    d["_group"] = d[group_cols].astype(str).agg("|".join, axis=1)

    # Outcome-varying strata contribute to conditional likelihood.
    varying = d.groupby("_group")["strict_esm_pos"].nunique()
    keep = varying[varying > 1].index
    d = d[d["_group"].isin(keep)].copy()

    # Count truly informative strata for this predictor.
    informative = 0
    for _, g in d.groupby("_group"):
        if g[pred].nunique() == 2 and g["strict_esm_pos"].nunique() == 2:
            informative += 1

    result = {
        "n_outcome_varying_strata": int(len(keep)),
        "n_predictor_and_outcome_informative_strata": int(informative),
        "n_rows_in_model": int(len(d)),
        "conditional_OR": np.nan,
        "conditional_OR_CI_low": np.nan,
        "conditional_OR_CI_high": np.nan,
        "conditional_p": np.nan,
    }

    if len(d) == 0 or d[pred].nunique() < 2:
        return result

    try:
        model = ConditionalLogit(
            d["strict_esm_pos"].astype(int),
            d[[pred]].astype(float),
            groups=d["_group"]
        )
        fit = model.fit(disp=False, maxiter=1000)
        beta = float(fit.params[pred])
        se = float(fit.bse[pred])
        result.update({
            "conditional_OR": float(np.exp(beta)),
            "conditional_OR_CI_low": float(np.exp(beta - 1.96 * se)),
            "conditional_OR_CI_high": float(np.exp(beta + 1.96 * se)),
            "conditional_p": float(fit.pvalues[pred]),
        })
    except Exception:
        pass

    return result

def conditional_logit_multi(df, preds, group_cols):
    d = df[preds + ["strict_esm_pos"] + group_cols].dropna().copy()
    d["_group"] = d[group_cols].astype(str).agg("|".join, axis=1)
    varying = d.groupby("_group")["strict_esm_pos"].nunique()
    keep = varying[varying > 1].index
    d = d[d["_group"].isin(keep)].copy()

    rows = []
    try:
        model = ConditionalLogit(
            d["strict_esm_pos"].astype(int),
            d[preds].astype(float),
            groups=d["_group"]
        )
        fit = model.fit(disp=False, maxiter=1000)
        for pred in preds:
            beta = float(fit.params[pred])
            se = float(fit.bse[pred])
            rows.append({
                "predictor": pred,
                "groups": "+".join(group_cols),
                "n_outcome_varying_strata": int(len(keep)),
                "n_rows_in_model": int(len(d)),
                "OR": float(np.exp(beta)),
                "CI_low": float(np.exp(beta - 1.96 * se)),
                "CI_high": float(np.exp(beta + 1.96 * se)),
                "p": float(fit.pvalues[pred]),
            })
    except Exception as e:
        for pred in preds:
            rows.append({
                "predictor": pred,
                "groups": "+".join(group_cols),
                "n_outcome_varying_strata": int(len(keep)),
                "n_rows_in_model": int(len(d)),
                "OR": np.nan, "CI_low": np.nan, "CI_high": np.nan, "p": np.nan,
            })
    return rows

# ------------------------------------------------------------------
# Build pair-level analysis dataset
# ------------------------------------------------------------------
phys = pd.read_csv(DATA / "bipolar_physiology.csv")
esm = pd.read_csv(DATA / "lexdelay_esm_sitelevel.csv")

phys["pair_canon"] = phys["match_pair"].map(canonical_pair)
esm["pair_canon"] = (
    esm["electrode"].astype(str)
       .str.replace(r"^D\d+_", "", regex=True)
       .map(canonical_pair)
)

if phys.duplicated(["match_subject", "pair_canon"]).any():
    raise RuntimeError("Duplicate physiology subject/pair keys detected.")
if esm.duplicated(["subject", "pair_canon"]).any():
    raise RuntimeError("Duplicate ESM subject/pair keys detected.")

merged = esm.merge(
    phys,
    left_on=["subject", "pair_canon"],
    right_on=["match_subject", "pair_canon"],
    how="inner",
    suffixes=("_esm", "_phys")
)

# Ensure booleans.
for col in ["liang_encoding", "liang_motor_preparation", "liang_delay", "liang_SM_vWM"]:
    merged[col] = merged[col].astype(bool)

merged["is_SM_vWM"] = merged["liang_class"].eq("Sensory-motor vWM")
merged["is_any_vWM"] = merged["liang_delay"]
merged["is_Auditory_vWM"] = merged["liang_class"].eq("Auditory vWM")

merged.to_csv(OUT / "analysis_dataset.csv", index=False)

# Coverage/QC.
esm_keys = set(zip(esm["subject"], esm["pair_canon"]))
matched_keys = set(zip(merged["subject"], merged["pair_canon"]))
esm["matched_to_bipolar_physiology"] = [
    (s, p) in matched_keys for s, p in zip(esm["subject"], esm["pair_canon"])
]
coverage = (
    esm.groupby("matched_to_bipolar_physiology")
       .agg(n_pairs=("electrode", "size"), n_ESM_pos=("strict_esm_pos", "sum"))
       .reset_index()
)
coverage["ESM_pos_rate"] = coverage["n_ESM_pos"] / coverage["n_pairs"]
coverage.to_csv(OUT / "coverage_summary.csv", index=False)

# Participant summary.
participant_rows = []
for subject, g in merged.groupby("subject"):
    participant_rows.append({
        "subject": subject,
        "n_pairs": len(g),
        "n_ESM_pos": int(g["strict_esm_pos"].sum()),
        "n_SM_vWM": int(g["is_SM_vWM"].sum()),
        "n_SM_vWM_ESM_pos": int(g.loc[g["is_SM_vWM"], "strict_esm_pos"].sum()),
        "n_any_vWM": int(g["is_any_vWM"].sum()),
        "n_any_vWM_ESM_pos": int(g.loc[g["is_any_vWM"], "strict_esm_pos"].sum()),
        "n_motor_prep": int(g["liang_motor_preparation"].sum()),
        "n_motor_prep_ESM_pos": int(g.loc[g["liang_motor_preparation"], "strict_esm_pos"].sum()),
    })
pd.DataFrame(participant_rows).to_csv(OUT / "participant_summary.csv", index=False)

# ------------------------------------------------------------------
# Exact class rates
# ------------------------------------------------------------------
class_order = [
    "Sensory-motor vWM",
    "Auditory vWM",
    "Motor vWM",
    "Delay-only",
    "Sensory-motor no-vWM",
    "Auditory no-vWM",
    "Motor no-vWM",
    "Other / unclassified",
]

class_rows = []
for cls in class_order:
    g = merged[merged["liang_class"].eq(cls)]
    n = len(g)
    k = int(g["strict_esm_pos"].sum())
    if n > 0:
        lo, hi = proportion_confint(k, n, method="wilson")
        rate = k / n
    else:
        lo = hi = rate = np.nan
    class_rows.append({
        "liang_class": cls,
        "n_pairs": n,
        "n_ESM_pos": k,
        "ESM_pos_rate": rate,
        "Wilson95_low": lo,
        "Wilson95_high": hi,
        "n_subjects": int(g["subject"].nunique()),
    })
pd.DataFrame(class_rows).to_csv(OUT / "class_rates.csv", index=False)

# ------------------------------------------------------------------
# Feature-level analyses
# ------------------------------------------------------------------
feature_specs = [
    ("Encoding", "liang_encoding"),
    ("Motor preparation", "liang_motor_preparation"),
    ("Delay", "liang_delay"),
    ("Sensory-motor vWM", "liang_SM_vWM"),
]

feature_rows = []
subject_auc_rows = []

for label, pred in feature_specs:
    pooled = pooled_2x2(merged, pred)
    auc, perm_p, subj_aucs, informative_subjects = macro_auc_permutation(
        merged, pred, n_perm=N_PERM, seed=SEED + len(feature_rows) + 1
    )
    subj_model = conditional_logit(merged, pred, ["subject"])
    anat_model = conditional_logit(merged, pred, ["subject", "broad_anatomy"])

    feature_rows.append({
        "feature": label,
        "predictor_column": pred,
        **pooled,
        "macro_within_subject_AUC": auc,
        "within_subject_perm_p_one_sided": perm_p,
        "n_informative_subjects_for_AUC": len(informative_subjects),
        "subject_conditional_OR": subj_model["conditional_OR"],
        "subject_conditional_CI_low": subj_model["conditional_OR_CI_low"],
        "subject_conditional_CI_high": subj_model["conditional_OR_CI_high"],
        "subject_conditional_p": subj_model["conditional_p"],
        "subject_anatomy_conditional_OR": anat_model["conditional_OR"],
        "subject_anatomy_conditional_CI_low": anat_model["conditional_OR_CI_low"],
        "subject_anatomy_conditional_CI_high": anat_model["conditional_OR_CI_high"],
        "subject_anatomy_conditional_p": anat_model["conditional_p"],
        "subject_anatomy_informative_strata": anat_model["n_predictor_and_outcome_informative_strata"],
    })

    for subject, a in subj_aucs.items():
        subject_auc_rows.append({
            "analysis": label,
            "subject": subject,
            "AUC": a
        })

pd.DataFrame(feature_rows).to_csv(OUT / "feature_associations.csv", index=False)
pd.DataFrame(subject_auc_rows).to_csv(OUT / "subject_AUCs.csv", index=False)

# ------------------------------------------------------------------
# Primary physiology-class contrasts
# ------------------------------------------------------------------
contrast_specs = [
    (
        "SM-vWM vs all other classes",
        merged.copy(),
        "is_SM_vWM",
    ),
    (
        "Any vWM vs no-vWM/unclassified",
        merged.copy(),
        "is_any_vWM",
    ),
    (
        "SM-vWM vs Auditory-vWM",
        merged[merged["liang_class"].isin(["Sensory-motor vWM", "Auditory vWM"])].copy(),
        "is_SM_vWM",
    ),
    (
        "SM-vWM vs other vWM",
        merged[merged["liang_delay"]].copy(),
        "is_SM_vWM",
    ),
]

contrast_rows = []
for idx, (name, dfc, pred) in enumerate(contrast_specs):
    pooled = pooled_2x2(dfc, pred)
    auc, perm_p, subj_aucs, informative_subjects = macro_auc_permutation(
        dfc, pred, n_perm=N_PERM, seed=SEED + 100 + idx
    )
    subj_model = conditional_logit(dfc, pred, ["subject"])
    anat_model = conditional_logit(dfc, pred, ["subject", "broad_anatomy"])

    exposed = dfc[dfc[pred].astype(bool)]
    unexposed = dfc[~dfc[pred].astype(bool)]

    contrast_rows.append({
        "contrast": name,
        "n_exposed": len(exposed),
        "ESM_pos_exposed": int(exposed["strict_esm_pos"].sum()),
        "ESM_rate_exposed": exposed["strict_esm_pos"].mean() if len(exposed) else np.nan,
        "n_unexposed": len(unexposed),
        "ESM_pos_unexposed": int(unexposed["strict_esm_pos"].sum()),
        "ESM_rate_unexposed": unexposed["strict_esm_pos"].mean() if len(unexposed) else np.nan,
        **pooled,
        "macro_within_subject_AUC": auc,
        "within_subject_perm_p_one_sided": perm_p,
        "n_informative_subjects_for_AUC": len(informative_subjects),
        "subject_conditional_OR": subj_model["conditional_OR"],
        "subject_conditional_CI_low": subj_model["conditional_OR_CI_low"],
        "subject_conditional_CI_high": subj_model["conditional_OR_CI_high"],
        "subject_conditional_p": subj_model["conditional_p"],
        "subject_anatomy_conditional_OR": anat_model["conditional_OR"],
        "subject_anatomy_conditional_CI_low": anat_model["conditional_OR_CI_low"],
        "subject_anatomy_conditional_CI_high": anat_model["conditional_OR_CI_high"],
        "subject_anatomy_conditional_p": anat_model["conditional_p"],
        "subject_anatomy_informative_strata": anat_model["n_predictor_and_outcome_informative_strata"],
    })

pd.DataFrame(contrast_rows).to_csv(OUT / "primary_contrasts.csv", index=False)

# ------------------------------------------------------------------
# Joint encoding + motor preparation + delay conditional models
# ------------------------------------------------------------------
joint_rows = []
preds = ["liang_encoding", "liang_motor_preparation", "liang_delay"]
joint_rows.extend(conditional_logit_multi(merged, preds, ["subject"]))
joint_rows.extend(conditional_logit_multi(merged, preds, ["subject", "broad_anatomy"]))
pd.DataFrame(joint_rows).to_csv(OUT / "joint_conditional_models.csv", index=False)

print("Analysis complete.")
print(f"Matched pairs: {len(merged)}; ESM+: {int(merged.strict_esm_pos.sum())}; subjects: {merged.subject.nunique()}")
