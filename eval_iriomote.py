#!/usr/bin/env python3
"""Evaluate the detector zero-shot on the Iriomote playback set.

Zero-shot: the model saw no Iriomote audio during training. The
recordings come from a different island, different equipment and a
different ambient soundscape, so this measures transfer rather than
fit.

Two operating points are reported per species:

  original     the threshold derived from the Ishigaki validation
               split, applied unchanged -- the deployment-realistic
               condition
  sweep-best   the threshold maximising F1 on Iriomote itself. Reaching
               it requires labelled data from the site, which the
               `original` strategy does not, and it is selected on the
               same clips it is scored on -- so the value reported here
               is an optimistic estimate of what recalibration would
               achieve on further recordings. It is reported in order to
               separate the model's ranking performance from
               threshold-calibration error

Run `prepare_iriomote.py` first to build the bundle this reads, then:

    python eval_iriomote.py

Runtime: a few seconds (550 clips).
"""

import json
from pathlib import Path

import numpy as np

from common import (
    SPECIES,
    binary_metrics,
    counts_at_threshold,
    load_thresholds,
    plot_metric_bars,
    run_tflite,
    sweep_best_f1,
    write_csv,
)

# ---------------------------------------------------------------------------
# Paths — edit these once for your machine
# ---------------------------------------------------------------------------

# Directory written by `prepare_iriomote.py` (specs.npy, labels.npy,
# dates.npy, metadata.json). Must match that script's OUT_DIR.
BUNDLE_DIR = "data/iriomote_bundle"

MODEL_PATH = "model/model.tflite"
THRESHOLDS_PATH = "model/thresholds_int8.json"
OUT_DIR = "results"

# ---------------------------------------------------------------------------

INVASIVE_SPECIES = ["POLLEU", "RHIMAR"]
STRATEGY_DISPLAY = {"original": "Original threshold",
                    "sweep-best": "Sweep best"}

bundle = Path(BUNDLE_DIR)
if not (bundle / "specs.npy").is_file():
    raise SystemExit(
        f"No bundle at {bundle}. Run prepare_iriomote.py first, then edit "
        f"BUNDLE_DIR at the top of this file so the two agree.")

specs = np.load(bundle / "specs.npy")
labels = np.load(bundle / "labels.npy")
with open(bundle / "metadata.json") as f:
    meta = json.load(f)
if meta["species"] != SPECIES:
    raise SystemExit(
        f"Bundle species order {meta['species']} does not match the model's "
        f"output order {SPECIES}.")
print(f"Loaded {len(specs)} clips from {bundle}")
print("  (specs are already z-scored by prepare_iriomote.py)\n")

thresholds = load_thresholds(THRESHOLDS_PATH)

print("Running INT8 inference ...")
probs = run_tflite(MODEL_PATH, specs, progress_every=200)
print()

rows = []
for sp in INVASIVE_SPECIES:
    col = SPECIES.index(sp)
    p, y = probs[:, col], labels[:, col]
    original_t = float(thresholds["f1_optimised"][col])
    sweep_t, sweep_counts = sweep_best_f1(p, y)
    for strategy, t, counts in (
            ("original", original_t, counts_at_threshold(p, y, original_t)),
            ("sweep-best", sweep_t, sweep_counts)):
        m = binary_metrics(*counts)
        rows.append(dict(
            dataset="iriomote_playback", species=sp,
            threshold_strategy=strategy, threshold=t, **m))
        print(f"  {sp} [{strategy:10s}] t={t:.2f}  F1={m['f1']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})")

print()
write_csv(rows, Path(OUT_DIR) / "iriomote_metrics.csv")

panels = []
for sp in INVASIVE_SPECIES:
    sp_rows = [r for r in rows if r["species"] == sp]
    n_pos = sp_rows[0]["tp"] + sp_rows[0]["fn"]
    panels.append((f"{sp}  (n={n_pos})", sp_rows))
plot_metric_bars(
    panels, Path(OUT_DIR) / "fig_iriomote.pdf",
    group_label_fn=lambda r: (f"{STRATEGY_DISPLAY[r['threshold_strategy']]}"
                              f"\n(t={r['threshold']:.2f})"),
)
