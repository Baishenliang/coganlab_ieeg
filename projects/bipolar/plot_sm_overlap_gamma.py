"""Plot SM overlap groups across four epochs; run this file without arguments.

Uses sm_comparison/SM_contacts.csv from compare_sm_electrodes.py.
Each row is one SM group; solid/dashed curves show average/bipolar reference.
Stored gamma z-scores are trial-averaged by load_stats, then contact-averaged.
No extra baseline subtraction or smoothing is applied.
"""
from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from projects.bipolar.compare_sm_electrodes import parse_label, EXCLUDED

EPOCHS = [('Auditory', (-0.25, 1.5)), ('Delay', (-0.25, 1.5)),
          ('Go', (-0.25, 1.0)), ('Resp', (-0.25, 1.0))]


def load_groups(path):
    table = pd.read_csv(path)
    groups = {'Intersection': set(), 'Average only': set(), 'Bipolar only': set()}
    for row in table.itertuples():
        key = parse_label(f'{row.subject}-{row.contact}', False)
        if key[0] in EXCLUDED:
            continue
        average = str(row.average_SM).lower() == 'true'
        bipolar = str(row.anode_of_bipolar_SM).lower() == 'true'
        if average or bipolar:
            tag = 'Intersection' if average and bipolar else 'Average only' if average else 'Bipolar only'
            groups[tag].add(key)
    return groups


def contact_traces(data, contacts, bipolar):
    """Match by subject + first contact; average multiple pairs per anode."""
    indices = {}
    for i, label in enumerate(data.labels[0]):
        key = parse_label(label, bipolar)[:2]
        if key in contacts:
            indices.setdefault(key, []).append(i)
    values = np.asarray(data)
    return {key: np.nanmean(values[idx], axis=0) for key, idx in indices.items()}


def plot_groups(stats_root, contacts_path, output_dir):
    from utils.group import load_stats

    groups = load_groups(contacts_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True,
                             gridspec_kw={'width_ratios': [1.75, 1.75, 1.25, 1.25]})
    audit, waves = [], []
    root = str(stats_root)
    for col, (event, limits) in enumerate(EPOCHS):
        for reference, color, style in [('average', '#2878B5', '-'), ('bipolar', '#D55E00', '--')]:
            print(f'Loading {event}: {reference}', flush=True)
            data, _ = load_stats('zscore', event + '_inRep', 'epo', root, root,
                                 trial_labels='CORRECT', reference=reference)
            times = np.asarray(data.labels[1], dtype=float)
            for row, (group, contacts) in enumerate(groups.items()):
                ax = axes[row, col]
                traces = contact_traces(data, contacts, reference == 'bipolar')
                for subject, contact in sorted(contacts):
                    audit.append([event, reference, group, subject, contact,
                                  (subject, contact) in traces])
                if not traces:
                    ax.plot([], [], color=color, linestyle=style, label=f'{reference}: n=0')
                    continue
                array = np.stack(list(traces.values()))
                valid = np.isfinite(array).sum(axis=0)
                mean = np.nanmean(array, axis=0)
                sem = np.full(mean.shape, np.nan)
                enough = valid > 1
                sem[enough] = np.nanstd(array[:, enough], axis=0, ddof=1) / np.sqrt(valid[enough])
                ax.plot(times, mean, color=color, linestyle=style,
                        label=f'{reference}: n={len(traces)}')
                ax.fill_between(times, mean-sem, mean+sem, color=color, alpha=0.15)
                waves.extend([event, reference, group, t, m, e, n]
                             for t, m, e, n in zip(times, mean, sem, valid))
            del data
        for row, (group, contacts) in enumerate(groups.items()):
            ax = axes[row, col]
            ax.set_xlim(limits)
            ax.axvline(0, color='0.4', linestyle='--', linewidth=1)
            ax.axhline(0, color='0.8', linewidth=0.8)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(event)
            ax.set_xticks([t for t in [0, 0.5, 1, 1.5] if t <= limits[1]])
            ax.legend(frameon=False, fontsize=8)
            if col == 0:
                ax.set_ylabel(f'{group} (N={len(contacts)})\nHigh-gamma z-score')
            if row == 2:
                ax.set_xlabel('Time from event (s)')
    fig.suptitle('SM overlap groups: average vs bipolar reference', fontsize=15)
    fig.text(0.5, 0.01, 'Matched by subject + first contact; D24/D26 excluded. '
             'Mean ± SEM across available contacts (not subjects).', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'SM_overlap_gamma.{extension}', dpi=300)
    plt.close(fig)
    pd.DataFrame(audit, columns=['epoch', 'reference', 'group', 'subject', 'contact', 'available']).to_csv(
        output / 'SM_overlap_gamma_channels.csv', index=False)
    pd.DataFrame(waves, columns=['epoch', 'reference', 'group', 'time', 'mean', 'sem', 'n_valid']).to_csv(
        output / 'SM_overlap_gamma_traces.csv', index=False)
    (output / 'SM_overlap_gamma_notes.txt').write_text(
        f'Groups from: {contacts_path}\nStats from: {stats_root}\n'
        'Four epochs, CORRECT trials; stored z-scores, no additional baseline or smoothing.\n'
        'Group membership is fixed by SM_contacts.csv; only first contacts are matched.\n'
        'Only groups include contacts absent from the other reference dataset.\n'
        'Each reference uses its available contacts; see availability CSV and panel n.\n'
        'Trials are averaged independently within each reference; trials are not paired.\n'
        'Multiple bipolar pairs with the same first contact are averaged before the group mean.\n'
        'Bands are descriptive contact SEM, not subject-level inference.\n', encoding='utf-8')
    print(f'Saved plots and audit tables to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats-root', type=Path, default=Path.home() / 'Box/CoganLab'
                        / 'BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/stats')
    parser.add_argument('--contacts', type=Path, default=HERE / 'sm_comparison/SM_contacts.csv')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'figs/sm_overlap_gamma')
    args = parser.parse_args()
    plot_groups(args.stats_root, args.contacts, args.output_dir)
