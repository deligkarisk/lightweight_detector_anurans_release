"""Shared primitives for the three evaluation scripts.

Self-contained: nothing here imports from the paper's research pipeline.
Each function is a direct port of the corresponding routine in that
pipeline, so these scripts reproduce the published values exactly rather
than approximately.

The audio -> prediction chain, end to end:

    librosa.load(path, sr=16000)
      -> truncate / zero-pad to 48000 samples (3 s)
      -> Butterworth order-5 low-pass at 6 kHz
      -> mel spectrogram: n_mels=128, n_fft=1024, hop=750
      -> power_to_db(ref=np.max), cropped / padded to 64 frames
      -> per-sample z-score over both axes
      -> quantize to int8, tf.lite.Interpreter, dequantize
      -> per-species probabilities in [0, 1]

The final sigmoid is baked into the TFLite graph (the last op is
LOGISTIC), so the interpreter's dequantized output is already a
probability -- no activation is applied here.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

# Alphabetical; each index is the corresponding output-tensor column.
# The model is multi-label: six native species are emitted alongside the
# two invasives, but only the invasives are scored by these scripts.
SPECIES: list[str] = [
    "BUECHO",   # 0
    "FEJSAK",   # 1
    "KUREIF",   # 2
    "MICKUR",   # 3  Microhyla kuramotoi
    "NIDOKI",   # 4
    "POLLEU",   # 5  Polypedates leucomystax  -- invasive, scored
    "RHIMAR",   # 6  Rhinella marina          -- invasive, scored
    "ZHAOWS",   # 7
]
NUM_CLASSES = len(SPECIES)

SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 3 * SAMPLE_RATE      # 48 000 = 3 s at 16 kHz
SEGMENT_SECONDS = SEGMENT_SAMPLES / SAMPLE_RATE

# Low-pass cutoff applied before mel extraction. Must match the value the
# model was trained with; an altered cutoff degrades every score without
# raising an error.
MAX_FILTER_HZ = 6000

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}

# Threshold sweep grid: 0.05 -> 0.95 in steps of 0.01 (91 points).
THRESHOLD_GRID: np.ndarray = np.linspace(0.05, 0.95, 91)


def folder_to_label(name: str) -> np.ndarray:
    """Parse a label string into an 8-dim multi-hot vector.

    Used for both the Ishigaki folder names ("RHIMAR,FEJSAK") and the
    Iriomote annotation labels ("POLLEU"). "Background" -- and any code
    not in SPECIES -- contributes nothing, so an unrecognised label
    yields the all-zero vector rather than an error.
    """
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    if name.strip().lower() == "background":
        return vec
    for code in name.split(","):
        code = code.strip()
        if code in SPECIES:
            vec[SPECIES.index(code)] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Audio -> spectrogram
# ---------------------------------------------------------------------------

def apply_lowpass(y: np.ndarray, sr: int, cutoff_hz: float | None,
                  order: int = 5) -> np.ndarray:
    """Butterworth low-pass. Returns y unchanged if the cutoff is absent
    or at/above Nyquist."""
    if cutoff_hz is None or cutoff_hz >= sr / 2:
        return np.asarray(y, dtype=np.float32)
    sos = butter(order, cutoff_hz, btype="low", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)


def extract_melspec(segment: np.ndarray, sr: int = SAMPLE_RATE,
                    segment_samples: int = SEGMENT_SAMPLES) -> np.ndarray:
    """Fixed-shape (128, 64) log-mel spectrogram for one 3 s segment."""
    S = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_mels=128, n_fft=1024,
        hop_length=segment_samples // 64,
    )
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    if S_db.shape[1] > 64:
        S_db = S_db[:, :64]
    elif S_db.shape[1] < 64:
        S_db = np.pad(S_db, ((0, 0), (0, 64 - S_db.shape[1])), mode="constant")
    return S_db


def normalize_specs(specs: np.ndarray) -> np.ndarray:
    """Per-sample z-score over the (freq, time) axes. Input (N, 128, 64)."""
    mean = specs.mean(axis=(1, 2), keepdims=True)
    std = specs.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((specs - mean) / std).astype(np.float32)


def wav_to_spec(path: str | os.PathLike, lowpass: bool = True) -> np.ndarray:
    """Load one clip and return its un-normalized (128, 64) spectrogram.

    Clips longer than 3 s are truncated, shorter ones zero-padded at the
    end. Call `normalize_specs` on the stacked result, not per clip.

    `lowpass=False` skips the 6 kHz filter, which is required for the
    Australian recordings as those have already been filtered. Leave it
    True otherwise: the model was trained on filtered audio.
    """
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if len(y) > SEGMENT_SAMPLES:
        y = y[:SEGMENT_SAMPLES]
    elif len(y) < SEGMENT_SAMPLES:
        y = np.pad(y, (0, SEGMENT_SAMPLES - len(y)), mode="constant")
    if lowpass:
        y = apply_lowpass(y, SAMPLE_RATE, MAX_FILTER_HZ)
    return extract_melspec(y, sr, SEGMENT_SAMPLES)


def iter_audio_files(root: str | os.PathLike):
    """Yield (path, parent_folder_name) for every audio file under root,
    in a deterministic order.

    Ignores the macOS metadata entries present in the published
    archives: `__MACOSX/` directories, `.DS_Store` files, and
    AppleDouble `._*` companions.
    """
    for dirpath, _dirnames, filenames in os.walk(root):
        parent = Path(dirpath).name
        if parent.startswith(".") or parent == "__MACOSX":
            continue
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTS:
                yield Path(dirpath) / fname, parent


def specs_from_folder_tree(root, label_fn, limit=None, progress_every=1000,
                           lowpass=True):
    """Walk `root`, extract a spectrogram per clip, and build labels with
    `label_fn(parent_folder_name)`.

    Returns (specs, labels, sources) with specs already z-scored.
    Unreadable files are skipped with a warning rather than aborting a
    multi-hour run.
    """
    items = list(iter_audio_files(root))
    if not items:
        raise RuntimeError(f"No audio files found under {root}")
    if limit is not None and limit < len(items):
        # Even stride rather than the first N -- the walk is folder by
        # folder, so a head slice would draw entirely from whichever
        # class sorts first and could contain no positives at all.
        items = items[::max(1, len(items) // limit)][:limit]
        print(f"  --limit {limit}: evenly-spaced subset across all folders")
    print(f"  Found {len(items)} clips under {root}")

    specs, labels, sources, failed = [], [], [], 0
    for i, (path, parent) in enumerate(items, start=1):
        try:
            spec = wav_to_spec(path, lowpass=lowpass)
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"  WARNING: could not load {path}: {exc}")
            continue
        specs.append(spec)
        labels.append(label_fn(parent))
        sources.append(parent)
        if i % progress_every == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] extracted  (failures: {failed})",
                  flush=True)

    if not specs:
        raise RuntimeError(f"No clips could be loaded from {root}")
    return (normalize_specs(np.asarray(specs, dtype=np.float32)),
            np.asarray(labels),
            sources)


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def run_tflite(tflite_path: str | os.PathLike, specs: np.ndarray,
               progress_every: int = 5000) -> np.ndarray:
    """Run the INT8 model over (N, 128, 64) specs -> (N, 8) probabilities.

    The default XNNPACK delegate fails to prepare some of this graph's
    INT8 op patterns, so it is disabled; the reference kernels are slower
    but always succeed.
    """
    import tensorflow as tf
    from tensorflow.lite.python.interpreter import OpResolverType

    with open(tflite_path, "rb") as f:
        model_content = f.read()

    interpreter = tf.lite.Interpreter(
        model_content=model_content,
        experimental_op_resolver_type=(
            OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    probs = np.empty((len(specs), NUM_CLASSES), dtype=np.float32)
    for i, spec in enumerate(specs):
        x = spec[np.newaxis, ..., np.newaxis].astype(np.float32)
        x_int8 = np.clip(np.round(x / in_scale + in_zero),
                         -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], x_int8)
        interpreter.invoke()
        raw = interpreter.get_tensor(out["index"])
        probs[i] = (raw.astype(np.float32) - out_zero) * out_scale
        if progress_every and ((i + 1) % progress_every == 0
                               or i + 1 == len(specs)):
            print(f"  [{i + 1}/{len(specs)}] scored", flush=True)
    return probs


def load_thresholds(path: str | os.PathLike) -> dict:
    """Read thresholds_int8.json and sanity-check it against SPECIES."""
    with open(path) as f:
        payload = json.load(f)
    if payload.get("precision") != "int8":
        raise ValueError(
            f"{path}: expected precision 'int8', got "
            f"{payload.get('precision')!r}. The FP32 thresholds are a "
            f"different operating point and would shift every count.")
    if payload["species"] != SPECIES:
        raise ValueError(
            f"{path}: species order {payload['species']} does not match "
            f"the model's output order {SPECIES}.")
    return payload


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Per-species metrics from scalar counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    n_neg = tn + fp
    fpr = fp / n_neg if n_neg > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    num = float(tp) * tn - float(fp) * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = num / den if den > 0 else 0.0
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=precision, recall=recall, fpr=fpr, f1=f1, mcc=mcc)


def counts_at_threshold(probs: np.ndarray, y: np.ndarray,
                        t: float) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) for a 1-D probability vector at threshold t."""
    pred = probs > t
    pos = y.astype(bool)
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & ~pos).sum())
    return tp, fp, fn, tn


def sweep_best_f1(probs: np.ndarray, y: np.ndarray,
                  grid: np.ndarray = THRESHOLD_GRID
                  ) -> tuple[float, tuple[int, int, int, int]]:
    """Threshold on `grid` maximising F1, and its counts."""
    best_t, best_f1, best_counts = float(grid[0]), -1.0, (0, 0, 0, 0)
    for t in grid:
        counts = counts_at_threshold(probs, y, float(t))
        f1 = binary_metrics(*counts)["f1"]
        if f1 > best_f1:
            best_t, best_f1, best_counts = float(t), f1, counts
    return best_t, best_counts


def write_csv(rows: list[dict], path: str | os.PathLike,
              float_decimals: int = 6) -> None:
    """Write metric rows, rounding floats so diffs stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, float_decimals)
                            if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# F1 / Precision / Recall, in legend order.
BAR_COLORS = ("steelblue", "seagreen", "indianred")
FIG_WIDTH_IN = 6.85      # 174 mm, Springer double-column
PANEL_HEIGHT_IN = 2.75
DPI = 300


def plot_metric_bars(panels: list[tuple[str, list[dict]]],
                     out_path: str | os.PathLike,
                     group_label_fn=None) -> None:
    """Grouped F1 / Precision / Recall bars, one panel per species.

    `panels` is [(panel_title, rows)], where each row is a metrics dict
    carrying `f1`, `precision`, `recall`, `threshold` and a
    `threshold_strategy` used for the x tick label.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if group_label_fn is None:
        def group_label_fn(row):
            return f"{row['threshold_strategy']}\n(t={row['threshold']:.2f})"

    metric_labels = ("F1", "Precision", "Recall")
    metric_keys = ("f1", "precision", "recall")
    bar_w = 0.25

    fig, axes = plt.subplots(
        len(panels), 1,
        figsize=(FIG_WIDTH_IN, PANEL_HEIGHT_IN * len(panels)),
        squeeze=False,
    )
    for ax, (title, rows) in zip(axes[:, 0], panels):
        x = np.arange(len(rows))
        values = np.array([[r[k] for k in metric_keys] for r in rows])
        for i, (lbl, color) in enumerate(zip(metric_labels, BAR_COLORS)):
            ax.bar(x + (i - 1) * bar_w, values[:, i], bar_w,
                   color=color, edgecolor="black", label=lbl)
        for xi, row_vals in zip(x, values):
            for i, v in enumerate(row_vals):
                ax.text(xi + (i - 1) * bar_w, v + 0.015, f"{v:.2f}",
                        ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([group_label_fn(r) for r in rows])
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("Score")
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out_path}")
