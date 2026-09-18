#!/usr/bin/env python3
"""Build the Iriomote playback evaluation bundle from the published audio.

Run this once before `eval_iriomote.py`. It turns the 39 annotated
playback recordings into the same 550 three-second windows the paper
evaluates, writing `specs.npy`, `labels.npy`, `dates.npy` and a
`metadata.json` provenance record.

Two window sources:

* **Annotation windows.** Each row of a `*_annotations.csv` becomes one
  or more 3 s windows. Annotations shorter than 3 s (every POLLEU call
  box is well under 1 s) yield a single window centred on the call
  midpoint. Annotations of 3 s or longer -- RHIMAR playback sessions run
  6-30 s and bundle many calls into one row -- are tiled at a 3 s stride
  so each window carries a distinct chunk.

* **Background windows.** Up to 50 per file, sampled from regions that
  fall outside every annotation (each padded by 0.5 s). These are the
  only strict-ambient clips in the bundle, so they are what the
  background false-positive rate is measured on.

Determinism: the background sampler draws from a single RNG seeded once
at 42 and shared across files, which are processed in sorted order. Both
the seed and the ordering are part of the reproduction -- changing either
gives a different (still valid, but not the paper's) background pool.

Usage:
    python prepare_iriomote.py --playback-dir path/to/playback \\
                               --out data/iriomote_bundle
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np

from common import (
    MAX_FILTER_HZ,
    NUM_CLASSES,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SEGMENT_SECONDS,
    SPECIES,
    apply_lowpass,
    extract_melspec,
    folder_to_label,
    iter_audio_files,
    normalize_specs,
)

# ---------------------------------------------------------------------------
# Paths — edit these once for your machine
# ---------------------------------------------------------------------------

# Directory holding the playback WAVs and their *_annotations.csv
# companions, unpacked from `playback.zip`.
PLAYBACK_DIR = "data/kimura/playback"

# Where to write specs.npy / labels.npy / dates.npy / metadata.json.
# `eval_iriomote.py`'s BUNDLE_DIR must point at this same directory.
OUT_DIR = "data/iriomote_bundle"

# ---------------------------------------------------------------------------
# Selection parameters
# ---------------------------------------------------------------------------

# Only the playback recordings are evaluated. The published archive also
# contains 28 `Site4_Okinawa_*_absent` control recordings, which this
# prefix excludes -- they were not part of the paper's set.
#
# NOTE: these filenames are those of the *public* release. The internal
# copies of the same 39 recordings were named `YMBK_playback_*`; the
# published ones are `Site4_playback_*`. The file contents and their
# sort order are identical, so the bundle reproduces either way, but the
# prefix below must match whichever naming you have on disk -- with the
# wrong one the script selects zero files.
INCLUDE_FILENAME_PREFIX = "Site4_playback_"
FILENAME_DATE_REGEX = re.compile(r"^Site4_playback_(\d{8})_")

ANNOTATION_CSV_SUFFIX = "_annotations.csv"

# The published annotation CSVs have the header
# `name,start_seconds,stop_seconds,channel`.
COL_LABEL = "name"
COL_START = "start_seconds"
COL_STOP = "stop_seconds"

# Windowing.
LONG_ANNOTATION_STRIDE_SECONDS = 3.0   # non-overlapping tiles
N_BACKGROUND_PER_FILE = 50
BACKGROUND_MARGIN_S = 0.5
BACKGROUND_RNG_SEED = 42


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def read_annotations(csv_path: str | os.PathLike
                     ) -> list[tuple[float, float, str]]:
    """Parse one annotation CSV into (start_s, end_s, label) triples.

    Rows with unparseable or non-increasing times are dropped.
    """
    rows: list[tuple[float, float, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {COL_LABEL, COL_START, COL_STOP} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path}: missing column(s) {sorted(missing)}; "
                f"header was {reader.fieldnames}")
        for r in reader:
            try:
                t0 = float(r[COL_START])
                t1 = float(r[COL_STOP])
            except (TypeError, ValueError):
                continue
            if not (t1 > t0):
                continue
            rows.append((t0, t1, str(r[COL_LABEL]).strip()))
    return rows


# ---------------------------------------------------------------------------
# Window placement
# ---------------------------------------------------------------------------

def annotation_window_bounds(t0: float, t1: float,
                             audio_duration: float) -> tuple[float, float]:
    """A single 3 s window centred on [t0, t1], clamped into the file.

    If the file itself is shorter than 3 s, returns the whole file and
    the caller zero-pads.
    """
    mid = 0.5 * (t0 + t1)
    start = mid - 0.5 * SEGMENT_SECONDS
    end = start + SEGMENT_SECONDS
    if audio_duration <= SEGMENT_SECONDS:
        return 0.0, audio_duration
    if start < 0:
        return 0.0, SEGMENT_SECONDS
    if end > audio_duration:
        return audio_duration - SEGMENT_SECONDS, audio_duration
    return start, end


def annotation_windows(t0: float, t1: float,
                       audio_duration: float) -> list[tuple[float, float]]:
    """One centred window for a short annotation; a tiled series for a
    long one. Tiles start at t0 and step by the stride until the next
    window would run past t1."""
    if (t1 - t0) < SEGMENT_SECONDS:
        return [annotation_window_bounds(t0, t1, audio_duration)]

    starts: list[float] = []
    t = t0
    while t + SEGMENT_SECONDS <= t1 + 1e-6:
        starts.append(t)
        t += LONG_ANNOTATION_STRIDE_SECONDS
    if not starts:                      # unreachable given the guard above
        starts.append(t0)

    windows: list[tuple[float, float]] = []
    for s in starts:
        s = max(0.0, s)
        if s + SEGMENT_SECONDS > audio_duration:
            s = max(0.0, audio_duration - SEGMENT_SECONDS)
        windows.append((s, min(audio_duration, s + SEGMENT_SECONDS)))
    return windows


def slice_to_segment(audio: np.ndarray, start_s: float,
                     end_s: float, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Exactly SEGMENT_SAMPLES samples from [start_s, end_s], zero-padded
    at the end if the slice falls short."""
    i0 = max(0, int(round(start_s * fs)))
    i1 = min(len(audio), int(round(end_s * fs)))
    seg = audio[i0:i1].astype(np.float32, copy=False)
    if len(seg) < SEGMENT_SAMPLES:
        out = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        out[:len(seg)] = seg
        return out
    return seg[:SEGMENT_SAMPLES]


def sample_background_intervals(audio_duration: float,
                                covered: list[tuple[float, float]],
                                n_wanted: int,
                                rng: np.random.Generator
                                ) -> list[tuple[float, float]]:
    """Up to `n_wanted` non-overlapping 3 s windows from the regions of
    the file no annotation touches (each annotation padded by
    BACKGROUND_MARGIN_S on both sides).

    Free regions are sampled in proportion to how much slack they have,
    and the sampler gives up after 5x n_wanted attempts -- a file whose
    annotations cover most of its duration simply yields fewer windows.
    """
    if n_wanted <= 0 or audio_duration < SEGMENT_SECONDS:
        return []

    padded = sorted(
        (max(0.0, t0 - BACKGROUND_MARGIN_S),
         min(audio_duration, t1 + BACKGROUND_MARGIN_S))
        for t0, t1 in covered
    )
    merged: list[tuple[float, float]] = []
    for a, b in padded:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    free: list[tuple[float, float]] = []
    prev_end = 0.0
    for a, b in merged:
        if a - prev_end >= SEGMENT_SECONDS:
            free.append((prev_end, a))
        prev_end = max(prev_end, b)
    if audio_duration - prev_end >= SEGMENT_SECONDS:
        free.append((prev_end, audio_duration))
    if not free:
        return []

    weights = np.clip(
        np.asarray([b - a - SEGMENT_SECONDS + 1e-9 for a, b in free]), 0, None)
    if weights.sum() == 0:
        return []
    probs = weights / weights.sum()

    out: list[tuple[float, float]] = []
    attempts = 0
    while len(out) < n_wanted and attempts < 5 * n_wanted:
        idx = int(rng.choice(len(free), p=probs))
        a, b = free[idx]
        start = float(rng.uniform(a, b - SEGMENT_SECONDS))
        end = start + SEGMENT_SECONDS
        if not any(not (end <= s0 or start >= s1) for s0, s1 in out):
            out.append((start, end))
        attempts += 1
    return sorted(out)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def date_from_filename(path: str | os.PathLike) -> str:
    """ISO date parsed out of the filename, or "" if it doesn't match."""
    m = FILENAME_DATE_REGEX.match(os.path.basename(str(path)))
    if not m:
        return ""
    s = m.group(1)
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def annotation_path_for(audio_path: Path) -> Path:
    return audio_path.with_name(audio_path.stem + ANNOTATION_CSV_SUFFIX)


def discover(playback_root: Path) -> list[Path]:
    all_files = [p for p, _ in iter_audio_files(playback_root)]
    kept = sorted(p for p in all_files
                  if p.name.startswith(INCLUDE_FILENAME_PREFIX))
    print(f"\nDiscovery: {playback_root}")
    print(f"  Audio files found            : {len(all_files)}")
    print(f"  Matching '{INCLUDE_FILENAME_PREFIX}*'   : {len(kept)}")
    with_csv = sum(1 for p in kept if annotation_path_for(p).is_file())
    print(f"  ...with a companion CSV      : {with_csv}")
    if not kept:
        raise SystemExit(
            f"\nERROR: no files matched the prefix "
            f"'{INCLUDE_FILENAME_PREFIX}'. If your copy of the playback "
            f"recordings uses different filenames, adjust "
            f"INCLUDE_FILENAME_PREFIX and FILENAME_DATE_REGEX at the top "
            f"of this script.")
    return kept


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

playback_root = Path(PLAYBACK_DIR)
out_root = Path(OUT_DIR)

if not playback_root.is_dir():
    raise SystemExit(
        f"Playback directory not found: {playback_root}\n"
        f"Edit PLAYBACK_DIR at the top of this file.")

files = discover(playback_root)
rng = np.random.default_rng(BACKGROUND_RNG_SEED)

specs_out: list[np.ndarray] = []
labels_out: list[np.ndarray] = []
dates_out: list[str] = []
per_species_counts = {sp: 0 for sp in SPECIES}
per_date_counts: dict[str, int] = {}
unknown_labels: dict[str, int] = {}
n_background = n_failed = 0
n_files_with_csv = n_files_no_csv = 0
n_annotations_processed = n_unparseable_dates = 0

for file_idx, fpath in enumerate(files, start=1):
    file_date = date_from_filename(fpath)
    if not file_date:
        print(f"  Warning: no parseable date, skipping {fpath.name}",
              file=sys.stderr)
        n_unparseable_dates += 1
        n_failed += 1
        continue
    try:
        audio, _ = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        print(f"  Warning: could not load {fpath}: {exc}", file=sys.stderr)
        n_failed += 1
        continue

    # Filter the whole recording once, before slicing -- filtering each
    # 3 s window separately would give different edge behaviour.
    audio = apply_lowpass(audio, SAMPLE_RATE, MAX_FILTER_HZ)
    audio_duration = len(audio) / SAMPLE_RATE

    csv_path = annotation_path_for(fpath)
    annotations: list[tuple[float, float, str]] = []
    if csv_path.is_file():
        try:
            annotations = read_annotations(csv_path)
            n_files_with_csv += 1
        except Exception as exc:
            print(f"  Warning: could not parse {csv_path}: {exc}",
                  file=sys.stderr)
            n_failed += 1
            continue
    else:
        n_files_no_csv += 1

    # -- Annotation-derived windows ------------------------------------------
    covered: list[tuple[float, float]] = []
    for t0, t1, raw_label in annotations:
        covered.append((t0, t1))
        label_vec = folder_to_label(raw_label)
        if (label_vec.sum() == 0
                and raw_label.strip().lower() != "background"):
            unknown_labels[raw_label] = unknown_labels.get(raw_label, 0) + 1
        for ws, we in annotation_windows(t0, t1, audio_duration):
            seg = slice_to_segment(audio, ws, we)
            specs_out.append(extract_melspec(seg))
            labels_out.append(label_vec.astype(np.float32))
            dates_out.append(file_date)
            per_date_counts[file_date] = per_date_counts.get(file_date, 0) + 1
            if label_vec.sum() == 0:
                n_background += 1
            else:
                for j, sp in enumerate(SPECIES):
                    if label_vec[j] >= 0.5:
                        per_species_counts[sp] += 1
        n_annotations_processed += 1

    # -- Background windows --------------------------------------------------
    n_bg_wanted = (N_BACKGROUND_PER_FILE if annotations
                   else max(N_BACKGROUND_PER_FILE,
                            int(audio_duration // SEGMENT_SECONDS)))
    for ws, we in sample_background_intervals(
            audio_duration, covered, n_bg_wanted, rng):
        seg = slice_to_segment(audio, ws, we)
        specs_out.append(extract_melspec(seg))
        labels_out.append(np.zeros(NUM_CLASSES, dtype=np.float32))
        dates_out.append(file_date)
        per_date_counts[file_date] = per_date_counts.get(file_date, 0) + 1
        n_background += 1

    if file_idx % 10 == 0 or file_idx == len(files):
        print(f"  [{file_idx}/{len(files)}] files; "
              f"{len(specs_out)} windows so far", flush=True)

if not specs_out:
    raise SystemExit("ERROR: no windows produced.")

X = normalize_specs(np.asarray(specs_out, dtype=np.float32))
y = np.asarray(labels_out, dtype=np.float32)
dates = np.asarray(dates_out, dtype="<U10")

if int(y[:, SPECIES.index("POLLEU")].sum()) == 0 and \
   int(y[:, SPECIES.index("RHIMAR")].sum()) == 0:
    raise SystemExit(
        "ERROR: zero POLLEU and zero RHIMAR positives. Check that the "
        "annotation labels use the species codes in common.SPECIES.")

out_root.mkdir(parents=True, exist_ok=True)
np.save(out_root / "specs.npy", X)
np.save(out_root / "labels.npy", y)
np.save(out_root / "dates.npy", dates)

sorted_dates = sorted(per_date_counts)
meta = {
    "created":                   datetime.now().isoformat(timespec="seconds"),
    "source_dir":                str(playback_root),
    "include_filename_prefix":   INCLUDE_FILENAME_PREFIX,
    "window_centering":          "centered",
    "long_annotation_stride_s":  LONG_ANNOTATION_STRIDE_SECONDS,
    "n_background_per_file":     N_BACKGROUND_PER_FILE,
    "background_margin_s":       BACKGROUND_MARGIN_S,
    "background_rng_seed":       BACKGROUND_RNG_SEED,
    "sample_rate":               SAMPLE_RATE,
    "max_filter_hz":             MAX_FILTER_HZ,
    "segment_samples":           SEGMENT_SAMPLES,
    "spec_shape":                [128, 64],
    "species":                   SPECIES,
    "n_clips":                   int(len(X)),
    "n_files_with_csv":          n_files_with_csv,
    "n_files_no_csv":            n_files_no_csv,
    "n_failed":                  n_failed,
    "n_unparseable_dates":       n_unparseable_dates,
    "n_annotations_processed":   n_annotations_processed,
    "n_background":              n_background,
    "per_species_counts":        per_species_counts,
    "n_dates":                   len(sorted_dates),
    "min_date":                  sorted_dates[0] if sorted_dates else "",
    "max_date":                  sorted_dates[-1] if sorted_dates else "",
    "per_date_counts":           {k: per_date_counts[k] for k in sorted_dates},
    "unknown_annotation_labels": unknown_labels,
}
with open(out_root / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone. {len(X)} windows from {len(files)} files "
      f"({n_annotations_processed} annotations, {n_background} background).")
for sp in SPECIES:
    if per_species_counts[sp]:
        print(f"    {sp:<8} {per_species_counts[sp]:>5}")
if unknown_labels:
    print("  Unrecognised annotation labels (ignored):")
    for k, v in sorted(unknown_labels.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<40s} {v:>5}")
print(f"  Saved {out_root}/")
print("\nExpected for the published playback set: 550 windows, "
      "358 annotations, 37 background, POLLEU 317, RHIMAR 196.")
