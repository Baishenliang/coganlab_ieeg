"""Build exact bipolar physiology classes and audit full-pair ESM matches.

Run without arguments in the ieeg environment. This prepares tables only;
it does not recode ESM outcomes or treat rows as independent stimulation sites.
"""
from pathlib import Path
import argparse
import pickle
import re
import sys
from numbers import Integral

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from projects.bipolar.compare_sm_electrodes import load_analysis_labels, parse_label

DEFAULT_BOOK = Path.home() / 'Box/CoganLab/Papers/2026/SM_VerbalWorkingMemory/Stim/lexdelay_esm_paper_analysis.xlsx'


def physiology_table(labels, indices_path, bipolar=True):
    with open(indices_path, 'rb') as stream:
        saved = pickle.load(stream)
    if saved.get('groupsTag') != 'LexDelay':
        raise ValueError('Expected LexDelay indices')
    keys = ['Aud_NoMotor', 'Sensorimotor', 'Motor', 'Delay', 'Sensorimotor_in_Delay']
    selections = {key: set(saved[f'LexDelay_{key}_sig_idx']) for key in keys}
    for name, values in saved.items():
        if isinstance(values, set) and any(not isinstance(i, Integral) or i < 0 or i >= len(labels) for i in values):
            raise ValueError(f'Indices incompatible with label inventory: {name}')
    aud, sm, motor, delay, sm_delay = (selections[k] for k in keys)
    if (aud & sm) or (aud & motor) or (sm & motor) or sm_delay != sm & delay:
        raise ValueError('Inconsistent physiological class indices')
    rows = []
    for i, label in enumerate(labels):
        parsed = parse_label(label, bipolar)
        subject, a = parsed[:2]
        b = parsed[2] if bipolar else ''
        encoding = i in aud or i in sm
        preparation = i in sm or i in motor
        wm = i in delay
        base = 'Sensory-motor' if i in sm else 'Auditory' if i in aud else 'Motor' if i in motor else None
        category = (base + (' vWM' if wm else ' no-vWM')) if base else 'Delay-only' if wm else 'Other / unclassified'
        rows.append([i, str(label), subject, a, b, f'{a}-{b}', encoding, preparation, wm, i in sm_delay, category])
    table = pd.DataFrame(rows, columns=['phys_index','phys_label','match_subject','anode','cathode',
        'match_pair','liang_encoding','liang_motor_preparation','liang_delay','liang_SM_vWM','liang_class'])
    if table.duplicated(['match_subject','match_pair']).any():
        raise ValueError('Duplicate full bipolar labels')
    return table


def normalize_stim(subject, electrode):
    subject = str(subject).strip()
    if not re.fullmatch(r'D\d+', subject):
        raise ValueError('invalid subject')
    subject = f'D{int(subject[1:]):04d}'
    label = str(electrode).strip()
    prefix = re.match(r'^(D\d+)[_-](.+)$', label)
    if prefix:
        if f'D{int(prefix[1][1:]):04d}' != subject:
            raise ValueError('subject mismatch')
        label = prefix[2]
    parts = label.split('-')
    if len(parts) != 2:
        raise ValueError('expected two contacts')
    a, b = parts
    stem = re.fullmatch(r'(.+?)(\d+)', a)
    if not stem:
        raise ValueError('invalid first contact')
    if b.isdigit():
        b = stem[1] + b
    if not re.fullmatch(r'.+?\d+', b) or a == b:
        raise ValueError('invalid second contact')
    return subject, f'{a}-{b}'


def match_stimulation(stim, physiology):
    required = {'subject','electrode','stim_result','stim_behavior','strict_esm_pos'}
    if not required.issubset(stim.columns):
        raise ValueError(f'Missing workbook columns: {required - set(stim.columns)}')
    stim = stim.copy()
    stim.insert(0, 'excel_row', range(2, len(stim) + 2))
    keys, errors = [], []
    for row in stim.itertuples():
        try:
            keys.append(normalize_stim(row.subject, row.electrode))
            errors.append('')
        except ValueError as exc:
            keys.append((None, None))
            errors.append(str(exc))
    stim['match_subject'] = [k[0] for k in keys]
    stim['match_pair'] = [k[1] for k in keys]
    stim['parse_error'] = errors
    stim['duplicate_pair_row'] = stim.duplicated(['match_subject','match_pair'], keep=False) & stim.parse_error.eq('')
    merged = stim.merge(physiology, on=['match_subject','match_pair'], how='left', validate='many_to_one', indicator=True)
    known_subjects = set(physiology.match_subject)
    known_pairs = set(zip(physiology.match_subject, physiology.match_pair))
    statuses = []
    for row in merged.itertuples():
        if row.parse_error:
            status = 'invalid_label'
        elif pd.notna(row.phys_index):
            status = 'matched'
        elif row.match_subject not in known_subjects:
            status = 'subject_not_in_physiology'
        else:
            a, b = row.match_pair.split('-')
            status = 'reversed_pair_requires_review' if (row.match_subject, f'{b}-{a}') in known_pairs else 'pair_not_in_physiology'
        statuses.append(status)
    merged['match_status'] = statuses
    return merged.drop(columns='_merge')


def prepare(workbook, labels, indices_path, output_dir):
    physiology = physiology_table(labels, indices_path)
    stim = pd.read_excel(workbook, sheet_name='Site_level')
    merged = match_stimulation(stim, physiology)
    matched = merged.match_status.eq('matched')
    used = set(zip(merged.loc[matched,'match_subject'], merged.loc[matched,'match_pair']))
    unused = physiology[[key not in used for key in zip(physiology.match_subject, physiology.match_pair)]]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {'bipolar_physiology': physiology, 'esm_all_rows': merged,
              'esm_matched': merged[matched], 'esm_unmatched': merged[~matched],
              'physiology_without_esm_match': unused,
              'match_summary': merged.groupby(['subject','match_status'], dropna=False).size().reset_index(name='rows')}
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False, encoding='utf-8-sig')
    (output / 'README.txt').write_text(
        f'Workbook: {workbook}\nIndices: {indices_path}\n'
        'Full subject + ordered pair matching; no first-contact-only or gap bridging matches.\n'
        'Original workbook outcome, anatomy and Nanlin class fields are unchanged; Liang fields use liang_ prefix.\n'
        'SM vWM = saved Sensorimotor_in_Delay indices. Encoding = Aud_NoMotor union Sensorimotor; '
        'motor preparation = Motor union Sensorimotor. These are not speech-response flags.\n'
        'Other / unclassified does not imply physiologically inactive.\n'
        'Unmatched physiology is missing, not negative. Unmatched ESM rows remain in audit.\n'
        'physiology_without_esm_match does not prove no stimulation occurred.\n'
        'Excel rows are not independent stimulation sites. Duplicate-pair flags do not identify shared clinical sites.\n'
        'Original stim_site_id/source mapping is required before final inference.\n'
        'Current inventory must match the historical saved index order; bounds alone cannot verify this.\n', encoding='utf-8')
    print(tables['match_summary'].to_string(index=False))
    print(f'Matched {matched.sum()}/{len(merged)} workbook rows; output: {output}')
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, default=DEFAULT_BOOK)
    parser.add_argument('--indices', type=Path, default=HERE / 'Lex_twin_idxes_hg_bipolar.npy')
    parser.add_argument('--labels', type=Path, help='Optional original full_label CSV in saved index order')
    parser.add_argument('--stats-root', type=Path, default=Path.home() / 'Box/CoganLab/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/stats')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'esm_comparison')
    args = parser.parse_args()
    labels = pd.read_csv(args.labels).full_label.tolist() if args.labels else load_analysis_labels(args.stats_root)[0]
    prepare(args.workbook, labels, args.indices, args.output_dir)
