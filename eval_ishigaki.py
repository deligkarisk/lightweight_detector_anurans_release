#!/usr/bin/env python3
"""Evaluate the detector on the Ishigaki (Kimura et al.) held-out test set.

This is the in-domain condition: the model was trained on the training
split of this same dataset, and each species is scored at the decision
threshold derived from the validation split -- the threshold a deployed
device would carry.

Labels come from the directory names. Each test subdirectory is a
comma-separated list of the species audible in the clips it holds
("BUECHO,FEJSAK"), or "Background" for clips with no target species.

Edit the paths below, then:

    python eval_ishigaki.py

Runtime: about a minute (760 clips).
"""

from pathlib import Path

import numpy as np

from common import (
    SPECIES,
    binary_metrics,
    counts_at_threshold,
    folder_to_label,
    load_thresholds,
    plot_metric_bars,
    run_tflite,
    specs_from_folder_tree,
    write_csv,
)

# ---------------------------------------------------------------------------
# Paths — edit these once for your machine
# ---------------------------------------------------------------------------

# Directory holding the `test/` split unpacked from
# `train-and-test-datasets.zip`. Either the directory that *contains*
# `test/`, or `test/` itself.
DATA_DIR = "data/kimura/train-and-test-datasets"

MODEL_PATH = "model/model.tflite"
THRESHOLDS_PATH = "model/thresholds_int8.json"
OUT_DIR = "results"

# Set to an integer to score only an evenly-spaced subset of that many
# clips -- a quick smoke test. None scores everything.
LIMIT = None

# ---------------------------------------------------------------------------

INVASIVE_SPECIES = ["POLLEU", "RHIMAR"]

test_dir = Path(DATA_DIR) / "test"
if not test_dir.is_dir():
    test_dir = Path(DATA_DIR)          # allow pointing straight at test/
if not test_dir.is_dir():
    raise SystemExit(
        f"No test directory found at {DATA_DIR}. Edit DATA_DIR at the top "
        f"of this file.")

thresholds = load_thresholds(THRESHOLDS_PATH)

print(f"Extracting spectrograms from {test_dir} ...")
specs, labels, _ = specs_from_folder_tree(
    test_dir, folder_to_label, limit=LIMIT, progress_every=200)
labels = np.asarray(labels, dtype=np.float32)
print(f"  {len(specs)} clips, {int(labels.sum())} species-positives\n")

print("Running INT8 inference ...")
probs = run_tflite(MODEL_PATH, specs, progress_every=200)
print()

rows = []
for sp in INVASIVE_SPECIES:
    col = SPECIES.index(sp)
    p, y = probs[:, col], labels[:, col]
    t = float(thresholds["f1_optimised"][col])
    m = binary_metrics(*counts_at_threshold(p, y, t))
    rows.append(dict(
        dataset="ishigaki_test", species=sp, threshold_strategy="f1",
        threshold=t, **m,
    ))
    print(f"  {sp}: t={t:.2f}  F1={m['f1']:.3f}  P={m['precision']:.3f}  "
          f"R={m['recall']:.3f}  "
          f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']})")

print()
write_csv(rows, Path(OUT_DIR) / "ishigaki_metrics.csv")
plot_metric_bars(
    [("Ishigaki test set (in-domain, validation-derived thresholds)", rows)],
    Path(OUT_DIR) / "fig_ishigaki.pdf",
    group_label_fn=lambda r: f"{r['species']}\n(t={r['threshold']:.2f})",
)
