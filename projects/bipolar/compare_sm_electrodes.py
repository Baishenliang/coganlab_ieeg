"""Compare saved SM_vWM indices using the exact label order from each analysis.

Interactive use (keep the two original label arrays separately):
    from projects.bipolar.compare_sm_electrodes import compare_sm
    tables = compare_sm(bipolar_labels, average_labels)

CLI: python projects/bipolar/compare_sm_electrodes.py \
    --bipolar-labels bipolar_labels.csv --average-labels average_labels.csv
Label CSVs must contain a full_label column, in original index order.
Indices are resolved BEFORE excluding D24/D26. Do not supply only SM_vWM labels.
"""
from pathlib import Path
import argparse
import pickle
import re
import sys
from numbers import Integral

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SM_KEY = 'LexDelay_Sensorimotor_in_Delay_sig_idx'
EXCLUDED = {'D0024', 'D0026'}


def load_analysis_labels(stats_root):
    """Reproduce LexDelay hg/CORRECT label order using the shared loader."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from utils.group import load_stats

    root = str(stats_root)
    print('Loading average Auditory_inRep labels...', flush=True)
    data, _ = load_stats('mask', 'Auditory_inRep', 'ave', root, root,
                         reference='average')
    average = list(data.labels[0])
    del data
    common = None
    for stat, contrast in [('mask', 'ave'), ('zscore', 'epo')]:
        for event in ('Auditory', 'Delay', 'Go', 'Resp'):
            print(f'Loading bipolar {event}_inRep {stat} labels...', flush=True)
            data, _ = load_stats(stat, event + '_inRep', contrast, root, root,
                                 trial_labels='CORRECT', reference='bipolar')
            labels = list(data.labels[0])
            del data
            if len(set(labels)) != len(labels):
                raise ValueError(f'Duplicate bipolar labels: {event} {stat}')
            available = set(labels)
            common = labels if common is None else [ch for ch in common if ch in available]
    if not common:
        raise ValueError('No shared bipolar channels across conditions')
    print(f'Loaded {len(average)} average and {len(common)} aligned bipolar labels')
    return common, average


def parse_label(label, bipolar):
    parts = str(label).split('-')
    if len(parts) != (3 if bipolar else 2):
        raise ValueError(f'Invalid channel label: {label}')
    match = re.fullmatch(r'D(\d+)', parts[0])
    if not match or any(not p for p in parts[1:]):
        raise ValueError(f'Invalid channel label: {label}')
    return (f'D{int(match[1]):04d}', *parts[1:])


def load_selection(path, labels, bipolar):
    # These are project-owned pickle files, despite their .npy extension.
    with open(path, 'rb') as file:
        saved = pickle.load(file)
    if saved.get('groupsTag') != 'LexDelay':
        raise ValueError(f'{path}: expected LexDelay indices')
    parsed = [parse_label(label, bipolar) for label in labels]
    if len(set(parsed)) != len(parsed):
        raise ValueError('Duplicate channel labels')
    # Validate every index set, not just SM_vWM, to catch truncated label arrays.
    for key, values in saved.items():
        if isinstance(values, set):
            if any(not isinstance(i, Integral) or i < 0 or i >= len(parsed)
                   for i in values):
                raise ValueError(f'{path}: {key} is incompatible with supplied labels')
    selected = {parsed[i] for i in saved[SM_KEY]}
    return ({p for p in parsed if p[0] not in EXCLUDED},
            {p for p in selected if p[0] not in EXCLUDED})


def plot_sm_overlap(contacts, output_dir):
    """Plot pooled SM_vWM sets, identifying contacts by subject plus first contact."""
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    contacts = contacts[~contacts.subject.isin(EXCLUDED)]
    def selected(column):
        mask = contacts[column].astype(str).str.lower().eq('true')
        return set(zip(contacts.loc[mask, 'subject'], contacts.loc[mask, 'contact']))
    average = selected('average_SM_vWM')
    bipolar = selected('anode_of_bipolar_SM_vWM')
    shared = len(average & bipolar)
    union = len(average | bipolar)
    score = f'{shared / union:.1%}' if union else 'N/A'
    fig, ax = plt.subplots(figsize=(7, 6))
    venn2(subsets=(len(average - bipolar), len(bipolar - average), shared),
          set_labels=(f'Average SM_vWM (n={len(average)})',
                      f'Bipolar SM_vWM first contacts (n={len(bipolar)})'), ax=ax)
    ax.set_title(f'Overall SM_vWM overlap\nIntersection / union (Jaccard): {score}')
    fig.text(0.5, 0.03, 'D24/D26 excluded; matched by subject + first contact.\n'
             'Includes contacts absent from the other reference dataset.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'SM_vWM_overlap_venn.png', dpi=300)
    plt.close(fig)
    print(f'SM_vWM overlap: average={len(average)}, bipolar={len(bipolar)}, '
          f'shared={shared}, union={union}, Jaccard={score}')
    return len(average), len(bipolar), shared, union


def compare_sm(bipolar_labels, average_labels, output_dir=None,
               bipolar_indices=HERE / 'Lex_twin_idxes_hg_bipolar.npy',
               average_indices=REPO / 'projects/GLM/data/Lex_twin_idxes_hg.npy'):
    """Return/write tables; labels must match the original saved index order.

    Match only the first contact (anode): A1-A2 maps to A1.
    The reference contact A2 does not contribute to overlap counts.
    Uses SM intersect Delay from each reference independently.
    """
    bp_all, bp_sm = load_selection(bipolar_indices, bipolar_labels, True)
    car_all, car_sm = load_selection(average_indices, average_labels, False)
    bp_contacts = {(s, a) for s, a, b in bp_all}
    bp_sm_contacts = {(s, a) for s, a, b in bp_sm}
    pair_rows = []
    for s, a, b in sorted(bp_all):
        if (s, a, b) not in bp_sm and (s, a) not in car_sm:
            continue
        states = [('SM_vWM' if (s, ch) in car_sm else
                   'non_SM_vWM' if (s, ch) in car_all else 'not_in_average_data')
                  for ch in (a,)]
        pair_rows.append([s, f'{a}-{b}', (s, a, b) in bp_sm, *states])
    pairs = pd.DataFrame(pair_rows, columns=[
        'subject', 'bipolar_channel', 'bipolar_SM_vWM', 'anode_average_status'])
    contact_rows = []
    for s, ch in sorted(car_sm | bp_sm_contacts):
        car = (s, ch) in car_sm
        bp = (s, ch) in bp_sm_contacts
        if car and bp:
            category = 'both'
        elif bp:
            category = 'bipolar_anode_only' if (s, ch) in car_all else 'not_in_average_data'
        else:
            category = 'average_only' if (s, ch) in bp_contacts else 'not_in_bipolar_anodes'
        linked = ';'.join(f'{a}-{b}' for sub, a, b in sorted(bp_sm)
                          if sub == s and ch == a)
        contact_rows.append([s, ch, car, bp, category, linked])
    contacts = pd.DataFrame(contact_rows, columns=[
        'subject', 'contact', 'average_SM_vWM', 'anode_of_bipolar_SM_vWM',
        'comparison', 'bipolar_SM_vWM_pairs'])
    summary_rows = []
    for s in sorted({p[0] for p in bp_all | car_all}):
        rows = contacts[contacts.subject == s]
        summary_rows.append({
            'subject': s,
            'average_SM_vWM_contacts': sum(p[0] == s for p in car_sm),
            'bipolar_SM_vWM_pairs': sum(p[0] == s for p in bp_sm),
            'bipolar_SM_vWM_unique_anodes': sum(p[0] == s for p in bp_sm_contacts),
            **{key: int((rows.comparison == key).sum()) for key in (
                'both', 'average_only', 'bipolar_anode_only',
                'not_in_average_data', 'not_in_bipolar_anodes')},
        })
    summary = pd.DataFrame(summary_rows)
    output = Path(output_dir) if output_dir else HERE / 'sm_vwm_comparison'
    output.mkdir(parents=True, exist_ok=True)
    tables = dict(summary=summary, pairs=pairs, contacts=contacts)
    for name, table in tables.items():
        table.to_csv(output / f'SM_vWM_{name}.csv', index=False, encoding='utf-8-sig')
    plot_sm_overlap(contacts, output)
    (output / 'README.txt').write_text(
        'Excluded: D0024, D0026 (after resolving indices).\n'
        f'Index key: {SM_KEY}; SM intersect Delay within each reference.\n'
        f'Bipolar indices: {bipolar_indices}\nAverage indices: {average_indices}\n'
        'Labels supplied by caller must be in the original saved index order.\n'
        'Integer bounds checks cannot verify historical label order.\n'
        'both = average SM_vWM contact also the first contact (anode) of a bipolar SM_vWM pair.\n'
        'Only the first contact is compared; the second/reference contact is ignored.\n'
        'Missing from a reference dataset is distinct from classified non-SM_vWM.\n'
        'pairs includes every bipolar SM_vWM pair and non-SM_vWM pairs whose first contact is average SM_vWM.\n',
        encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'Comparison tables saved to {output}')
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bipolar-labels', type=Path)
    parser.add_argument('--average-labels', type=Path)
    parser.add_argument('--stats-root', type=Path, default=Path.home() / 'Box' / 'CoganLab'
                        / 'BIDS-1.0_LexicalDecRepDelay' / 'BIDS' / 'derivatives' / 'stats')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if bool(args.bipolar_labels) != bool(args.average_labels):
        parser.error('Supply both label CSVs, or neither for automatic loading')
    if args.bipolar_labels:
        bipolar_labels = pd.read_csv(args.bipolar_labels).full_label.tolist()
        average_labels = pd.read_csv(args.average_labels).full_label.tolist()
    else:
        bipolar_labels, average_labels = load_analysis_labels(args.stats_root)
    tables = compare_sm(bipolar_labels, average_labels, args.output_dir)
