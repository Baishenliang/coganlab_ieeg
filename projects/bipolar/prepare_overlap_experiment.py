"""Filter existing bipolar physiology to identical average first-contact classes."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from utils.group import load_stats
from projects.bipolar.prepare_esm_comparison import physiology_table

root = str(Path.home() / 'Box/CoganLab/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/stats')
data, _ = load_stats('mask', 'Auditory_inRep', 'ave', root, root, reference='average')
average = physiology_table(list(data.labels[0]), HERE.parents[1] / 'projects/GLM/data/Lex_twin_idxes_hg.npy', bipolar=False)
# Padded channels without measured data must not become an agreement on inactivity.
average['average_available'] = np.isfinite(np.asarray(data)).any(axis=1)
average = average[['match_subject', 'anode', 'liang_class', 'average_available']].rename(columns={'liang_class': 'average_class'})
bp = pd.read_csv(HERE / 'esm_comparison/bipolar_physiology.csv')
audit = bp.merge(average, on=['match_subject', 'anode'], how='left', validate='many_to_one')
audit['same_class'] = audit.liang_class.eq(audit.average_class) & audit.average_available.eq(True)
out = HERE / 'esm_comparison_overlap'
out.mkdir(exist_ok=True)
audit.to_csv(out / 'class_agreement_audit.csv', index=False)
audit.loc[audit.same_class, bp.columns].to_csv(out / 'bipolar_physiology.csv', index=False)
print(audit.groupby(['liang_class', 'same_class']).size().to_string())
print(f'Retained {audit.same_class.sum()}/{len(audit)} pairs')
