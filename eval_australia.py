#!/usr/bin/env python3
"""Evaluate the detector zero-shot on the Australian cane toad set.

The most distant transfer condition in the paper: a different
continent, a different recording programme, and a set of non-target
species absent from the training data (dingo, pheasant coucal,
kookaburra, boobook and several Cyclorana frogs). Only the cane toad,
*Rhinella marina*, is shared with the training data, so only the RHIMAR
output is scored.

Labels are positional: a clip is a positive if and only if its immediate
parent folder is the cane toad folder. Everything else -- including
Background and Unidentified Sounds -- is a negative.

Two operating points, as for Iriomote: `original` applies the Ishigaki
threshold unchanged, while `sweep-best` maximises F1 on this set, which
requires labelled data from it and is selected on the same clips it is
scored on.

Edit the paths below, then:

    python eval_australia.py

Runtime: this is ~39,000 clips, so expect tens of minutes on the first
run. The spectrogram cache makes every later run take seconds.
"""

from pathlib import Path

import numpy as np

from common import (
    SPECIES,
    binary_metrics,
    counts_at_threshold,
    load_thresholds,
    plot_metric_bars,
    run_tflite,
    specs_from_folder_tree,
    sweep_best_f1,
    write_csv,
)

# ---------------------------------------------------------------------------
# Paths — edit these once for your machine
# ---------------------------------------------------------------------------

# Root of the cane toad dataset: the directory holding the per-class
# folders (`Background`, `Rhinella marina_Cane Toad_Modified`, ...).
DATA_DIR = "data/australia_cane_toad"

# Where to save/reuse the extracted spectrograms (~1.3 GB). Set to None
# to re-extract from audio every run.
CACHE_PATH = "data/australia_specs.npz"

MODEL_PATH = "model/model.tflite"
THRESHOLDS_PATH = "model/thresholds_int8.json"
OUT_DIR = "results"

# Set to an integer to score only an evenly-spaced subset of that many
# clips -- a quick smoke test. None scores everything.
LIMIT = None

# ---------------------------------------------------------------------------

TARGET_SPECIES = "RHIMAR"
POSITIVE_FOLDER = "Rhinella marina_Cane Toad_Modified"
STRATEGY_DISPLAY = {"original": "Original threshold",
                    "sweep-best": "Sweep best"}

thresholds = load_thresholds(THRESHOLDS_PATH)
cache = Path(CACHE_PATH) if CACHE_PATH else None

if cache is not None and cache.is_file():
    print(f"Loading cached spectrograms from {cache} ...")
    cached = np.load(cache, allow_pickle=False)
    specs, labels = cached["specs"], cached["labels"].astype(bool)
else:
    if not Path(DATA_DIR).is_dir():
        raise SystemExit(
            f"No dataset at {DATA_DIR}. Edit DATA_DIR at the top of this "
            f"file.")
    print(f"Extracting spectrograms from {DATA_DIR} ...")
    print("  (39k clips takes a while; it is cached for later runs)")
    specs, labels, _ = specs_from_folder_tree(
        DATA_DIR, lambda parent: parent == POSITIVE_FOLDER,
        limit=LIMIT, lowpass=False)   # already band-limited
    labels = np.asarray(labels, dtype=bool)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, specs=specs, labels=labels.astype(np.uint8))
        print(f"  Cached to {cache}")

print(f"  {len(labels)} clips: {int(labels.sum())} positives, "
      f"{int((~labels).sum())} negatives\n")

print("Running INT8 inference ...")
probs = run_tflite(MODEL_PATH, specs, progress_every=5000)
print()

col = SPECIES.index(TARGET_SPECIES)
p, y = probs[:, col], labels.astype(np.float32)
original_t = float(thresholds["f1_optimised"][col])
sweep_t, sweep_counts = sweep_best_f1(p, y)

rows = []
for strategy, t, counts in (
        ("original", original_t, counts_at_threshold(p, y, original_t)),
        ("sweep-best", sweep_t, sweep_counts)):
    m = binary_metrics(*counts)
    rows.append(dict(
        dataset="australia_zeroshot", species=TARGET_SPECIES,
        threshold_strategy=strategy, threshold=t, **m))
    print(f"  {TARGET_SPECIES} [{strategy:10s}] t={t:.2f}  "
          f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
          f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})")

print()
write_csv(rows, Path(OUT_DIR) / "australia_metrics.csv")
n_pos = rows[0]["tp"] + rows[0]["fn"]
plot_metric_bars(
    [(f"{TARGET_SPECIES} — Australian cane toad set  (n={n_pos})", rows)],
    Path(OUT_DIR) / "fig_australia.pdf",
    group_label_fn=lambda r: (f"{STRATEGY_DISPLAY[r['threshold_strategy']]}"
                              f"\n(t={r['threshold']:.2f})"),
)
