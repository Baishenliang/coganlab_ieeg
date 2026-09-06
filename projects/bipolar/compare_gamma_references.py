"""Plot trial-mean gamma z-scores for bipolar channels and both CAR endpoints.

Run with the ieeg environment, for example:
    python projects/bipolar/compare_gamma_references.py

Each panel shows three already-baseline-normalized waveforms. CAR endpoint
z-scores are not subtracted: that would not reproduce bipolar gamma z-scores.
Only matching trials are used, aligned by event sample and event description.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


DEFAULT_STATS = Path(
    "C:/Users/bl314/Box/CoganLab/BIDS-1.0_LexicalDecRepDelay/"
    "BIDS/derivatives/stats/D0023"
)


def trial_keys(epochs):
    descriptions = {value: name for name, value in epochs.event_id.items()}
    keys = [(int(event[0]), descriptions[int(event[2])]) for event in epochs.events]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate event sample/description keys; cannot align trials safely")
    return keys


def compare(common_path, bipolar_path, output_dir, rows=4, cols=4):
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    common = mne.read_epochs(common_path, preload=False)
    bipolar = mne.read_epochs(bipolar_path, preload=False)
    if common.times.shape != bipolar.times.shape or not np.allclose(
        common.times, bipolar.times, rtol=0, atol=1e-9
    ):
        raise ValueError("The files have different time axes; no automatic resampling performed")
    common_keys, bipolar_keys = trial_keys(common), trial_keys(bipolar)
    common_lookup = {key: i for i, key in enumerate(common_keys)}
    matched = [(common_lookup[key], i) for i, key in enumerate(bipolar_keys)
               if key in common_lookup]
    if not matched:
        raise ValueError("No matching trials between the two files")
    common_idx, bipolar_idx = map(list, zip(*matched))
    print(f"Matched trials: {len(matched)}; common total: {len(common_keys)}; "
          f"bipolar total: {len(bipolar_keys)}")
    # Match against full original labels instead of assuming they lack hyphens.
    endpoints = []
    common_names = set(common.ch_names)
    for label in bipolar.ch_names:
        candidates = [(a, label[len(a) + 1:]) for a in common.ch_names
                      if label.startswith(a + "-")
                      and label[len(a) + 1:] in common_names]
        if len(candidates) != 1:
            raise ValueError(f"Cannot uniquely match both CAR endpoints for {label!r}")
        endpoints.append(candidates[0])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = common.times
    colors = ("#222222", "#2878B5", "#D55E00")
    per_page = rows * cols
    stem = Path(bipolar_path).name.removesuffix("-epo.fif")
    for start in range(0, len(bipolar.ch_names), per_page):
        stop = min(start + per_page, len(bipolar.ch_names))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0),
                                 sharex=True, sharey=True, squeeze=False)
        page_common_names = list(dict.fromkeys(
            ch for pair in endpoints[start:stop] for ch in pair))
        # Load one page of channels at a time, keeping memory bounded.
        common_data = common[common_idx].get_data(picks=page_common_names)
        bipolar_data = bipolar[bipolar_idx].get_data(picks=bipolar.ch_names[start:stop])
        common_means = dict(zip(page_common_names, np.nanmean(common_data, axis=0)))
        bipolar_means = np.nanmean(bipolar_data, axis=0)
        for local, index in enumerate(range(start, stop)):
            ax = axes.flat[local]
            label = bipolar.ch_names[index]
            a, b = endpoints[index]
            for wave, color, legend in zip(
                (bipolar_means[local], common_means[a], common_means[b]),
                colors, ("Bipolar", f"CAR {a}", f"CAR {b}")
            ):
                ax.plot(times, wave, color=color, linewidth=1.1, label=legend)
            ax.axvline(0, color="0.5", linestyle="--", linewidth=0.7)
            ax.axhline(0, color="0.7", linewidth=0.6)
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=7, frameon=False, loc="upper right")
            ax.grid(alpha=0.15)
        for ax in list(axes.flat)[stop-start:]:
            ax.set_visible(False)
        fig.supxlabel("Time from auditory onset (s)")
        fig.supylabel("Gamma z-score (trial mean)")
        page = start // per_page + 1
        fig.suptitle(f"{stem} | matched trials: {len(matched)} | page {page}\n"
                     "Bipolar vs both common-average endpoints; shared y-axis within page",
                     fontsize=12)
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.94))
        target = output_dir / f"{stem}_comparison_page-{page:02d}.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        print(f"Saved {target}")
    (output_dir / f"{stem}_comparison_info.txt").write_text(
        f"Common: {Path(common_path).resolve()}\nBipolar: {Path(bipolar_path).resolve()}\n"
        f"Matched trials: {len(matched)}\nCommon trials: {len(common_keys)}\n"
        f"Bipolar trials: {len(bipolar_keys)}\n"
        "Each line is the NaN-aware trial mean of stored gamma z-scores.\n"
        "No extra baseline correction, smoothing, or endpoint subtraction.\n"
        "NaNs are omitted per channel/time; valid trial counts may differ.\n",
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path,
                        default=DEFAULT_STATS / "Auditory_inRep_zscore-epo.fif")
    parser.add_argument("--bipolar", type=Path,
                        default=DEFAULT_STATS / "Auditory_inRep_zscore_bipolar-epo.fif")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "figs" / "D0023")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()
    compare(args.common, args.bipolar, args.output_dir, args.rows, args.cols)
