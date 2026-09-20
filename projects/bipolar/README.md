# Bipolar physiology and stimulation analysis

This folder adapts the LexicalDelay analysis to bipolar recordings using Liang's time windows, compares reference methods, and prepares electrical stimulation mapping (ESM) comparisons. This guide replaces the three earlier Markdown documents.

## 1. Bipolar preprocessing

The shared preprocessing pipeline supports common-average and bipolar reference for multitaper and gamma analyses. Bipolar signals subtract the second contact from the first, for example `LFO11-LFO12`. Bad and unlocalized contacts are excluded before pairing; numbering gaps are allowed only when the geometry criteria permit them.

Coordinates normally come from the raw montage. Reconstruction coordinates provide a fallback without interpolation; D107, D128, and D139 use forced reconstruction. Bipolar outputs have separate names from common-average outputs.

## 2. Group physiology classification

`bipolar_group_stats.py` loads bipolar masks and gamma z-score epochs, aligns their common channels, and applies Liang's encoding, motor-preparation, and delay criteria. The four epochs are Auditory, Delay, Go, and Resp. Reconstruction coordinates are read per participant, with bounded retries for Windows error 1006.

The saved index file is `Lex_twin_idxes_hg_bipolar.npy` (a pickle file despite its extension). SM vWM means sensory-motor activity **and** delay activity, using `LexDelay_Sensorimotor_in_Delay_sig_idx`. It does not require agreement with average-reference classification.

Index files contain integer positions, so their original channel order must be preserved. Automatic label loading reproduces the current group-analysis channel intersection, but cannot verify that an older index file used the same inventory. Changes to the inventory require matching indices and labels to be regenerated.

## 3. Compare reference methods

Activate the `ieeg` environment and run commands from the repository root:

```powershell
python projects/bipolar/compare_sm_electrodes.py
python projects/bipolar/plot_sm_overlap_gamma.py
```

The comparison originally used all SM contacts; it now uses **SM vWM** independently from each reference. D24 and D26 are excluded. For this comparison only, bipolar `A1-A2` matches average `A1`; the reference contact A2 is ignored.

The first script loads reference-specific FIFs, resolves indices, and writes tables and a pooled Venn plot to `sm_vwm_comparison/`:

- `SM_vWM_summary.csv`: participant counts and overlap.
- `SM_vWM_contacts.csv`: shared or reference-specific first contacts.
- `SM_vWM_pairs.csv`: related bipolar pairs and average classification.
- `SM_vWM_overlap_venn.png`: pooled overlap and intersection/union (Jaccard).

Contacts absent from the other reference inventory are distinguished from contacts classified as non-SM-vWM. The Venn includes both types of reference-specific membership.

The second script uses the new contact table to plot three groups (intersection, average only, bipolar only) across four epochs. Outputs are in `figs/sm_vwm_overlap_gamma/`. Each panel shows both references where available, with mean and contact-level SEM. Stored z-scores are used without extra baseline subtraction or smoothing. Trials are averaged independently per reference; these are not paired-trial or participant-level inference plots. Availability and waveform CSVs accompany the figures. Old all-SM outputs are retained separately.

Default FIF root: `~/Box/CoganLab/BIDS-1.0_LexicalDecRepDelay/BIDS/derivatives/stats`. Use `--stats-root` to override it. Optional original-order label CSVs, each with a `full_label` column, avoid automatic loading:

```powershell
python projects/bipolar/compare_sm_electrodes.py --bipolar-labels bipolar_labels.csv --average-labels average_labels.csv
```

## 4. Inspect individual gamma waveforms

```powershell
python projects/bipolar/compare_gamma_references.py
```

This separate script defaults to D0023 Auditory_inRep z-score FIFs. It matches trials by event sample and description, then plots each bipolar channel alongside **both** average-reference endpoints in 4-by-4 pages. It does not subtract endpoint z-scores. PNGs and a metadata TXT are saved to `figs/D0023/`. Use `--common`, `--bipolar`, and `--output-dir` to override the inputs and destination.

## 5. Build the physiology-to-ESM table

```powershell
python projects/bipolar/prepare_esm_comparison.py
```

This implements ESM steps 1–2: build exact bipolar classes and match stimulation records. It reads the bipolar index file and the `Site_level` sheet of:

`~/Box/CoganLab/Papers/2026/SM_VerbalWorkingMemory/Stim/lexdelay_esm_paper_analysis.xlsx`

The framing and earlier exploratory analysis are documented in `LexicalDelay_ESM_analysis_detailed_methods_results.docx` in the same Box folder. The goal is to replace its approximate physiological classes with Liang's exact classes.

Matching uses **participant plus the full ordered pair**, not just the first contact. For example, `D0063_ROF6-7` matches `D63-ROF6-ROF7`. Numbering gaps do not substitute for adjacent pairs; reversed pairs are flagged for review. Unmatched physiology stays missing, not negative. All bipolar SM vWM pairs are eligible, regardless of average-reference results.

Outputs in `esm_comparison/`:

| File | Contents |
| --- | --- |
| `bipolar_physiology.csv` | Full classified inventory, original indices, pair labels, and Liang classes |
| `esm_all_rows.csv` | All stimulation rows, original fields, and match status |
| `esm_matched.csv`, `esm_unmatched.csv` | Matched records and records needing review |
| `physiology_without_esm_match.csv` | Physiology pairs without an exact match in this workbook |
| `match_summary.csv` | Counts by participant and match status |

In the physiology table, `phys_label` identifies the pair, `liang_class` gives its type, and `liang_SM_vWM` identifies the target group. Categories include sensory-motor, auditory, and motor vWM/no-vWM, delay-only, and other/unclassified. Motor preparation is not a generic speech-response flag; other/unclassified does not establish inactivity.

Original `stim_result`, `stim_behavior`, `strict_esm_pos`, anatomy, and `repeat_class` are preserved. New physiology columns have a `liang_` prefix. The script does not recode ESM outcomes or run inferential statistics. Use `--workbook`, `--indices`, `--stats-root`, or `--output-dir` to override paths; `--labels labels.csv` supplies the original full inventory directly. The shared automatic loader also reads average labels, but they do not influence ESM selection.

## 6. Current results and open matching issue

The inspected output snapshot contains 5,683 bipolar pairs from 47 participants, including 455 SM vWM pairs. Of 193 stimulation rows, 96 match, 75 have no matching pair, and 22 have no matching participant. The matched sample has 21 existing ESM-positive and 75 negative labels. Nine matched pairs are SM vWM (7 positive, 2 negative); seven of these pairs are from D0081, with one each from D0137 and D0140. These are descriptive counts, not independent stimulation-site estimates.

The largest coverage gaps are:

| Participant | Excel rows | Matched | Unmatched |
| --- | ---: | ---: | ---: |
| D0081 | 35 | 11 | 24 |
| D0103 | 34 | 12 | 22 |
| D0129 | 22 | 10 | 12 |
| D0115 | 22 | 0 | 22 |

For the first three participants, pairs exist in the stimulation workbook but not in the final physiology inventory. Different contact selections/pairings, preprocessing exclusions, coordinate checks, or cross-epoch intersection may explain this; the cause is not yet confirmed. D0115 is explicitly excluded by `utils.group.load_stats`. Do not insert additional labels into existing saved indices without regenerating the corresponding classification.

Of the earlier analysis's 37 SM vWM rows, 25 match and 12 do not. The difference between its 37 rows and the new 9 therefore cannot be attributed entirely to time-window definitions. Subsequent analyses should report this coverage limitation rather than classify unmatched rows as negative.

## 7. Remaining ESM steps

1. **Outcome review (step 3):** inspect `stim_result` and `stim_behavior` against the earlier language-positive and exclusion rules. Initial counts are 144 `no_effect` and 49 `behavioral_effect`, consistent in totals with the existing binary labels; detailed review is not complete.
2. **Independent stimulation sites (step 4):** obtain the original `stim_site_id` mapping. No duplicate participant/pair rows were found, but several pairs may share one clinical stimulation site. Row numbers and repeated descriptions are not reliable site identifiers.
3. **Association analysis (step 5):** summarize class-specific ESM-positive rates, prioritize SM vWM versus other classes, and account for stimulation-site dependence, participant, and anatomy before final inference.
4. **Additional figure (step 6):** plot class rates and adjusted effects. Repeat-versus-Decision analysis is a later extension requiring consistent physiological definitions for both conditions.

No new inferential ESM tests or association figures have been completed. Generated tables and plots remain local; rerunning scripts overwrites same-named outputs.


## Reference-agreement experiment

Branch: `experiment/esm-reference-overlap`.
Run `python projects/bipolar/prepare_overlap_experiment.py`, then Greg's
`01_run_analysis.py` and `02_make_figures.py` in the reproducibility bundle.
The experiment retains bipolar pairs only when their first contact has the same
full class in average reference, including vWM/no-vWM status. Other/unclassified
is also compared as its own category; missing average data is not agreement.
It uses the existing bipolar physiology CSV (not newly computed allchs results).
As with the original workflow, average indices must correspond to the currently
loaded channel order; historical order cannot be verified from indices alone.
Inputs and rejected rows are audited in `esm_comparison_overlap/class_agreement_audit.csv`.
The filtered inventory goes to `esm_comparison_overlap/bipolar_physiology.csv`.
Greg's scripts on this branch write `outputs_overlap/` and `figures_overlap/`;
original outputs are preserved. This is a selected agreement subset, so its
association estimates apply to that subset rather than all bipolar recordings.

Archived trial result (2026-09-20): 3,589 of 5,762 bipolar pairs retained,
including 242 SM vWM pairs. ESM matching yielded 57 pairs and 11 positives;
SM vWM was 5/6 positive, Auditory vWM 0/3, Motor no-vWM 3/3,
Auditory no-vWM 0/2, and Other/unclassified 3/43. No Motor vWM,
Delay-only, or SM no-vWM pairs matched. The pooled SM-vWM-versus-rest
Fisher test gave OR 37.5 and p approximately 0.000598, but conditional
models produced extreme estimates and overflowing confidence intervals.
This sparse exploratory subset does not support reliable adjusted inference.
Keep this experiment on its archive branch; do not merge it into main.
