# A 66 KB invasive-frog detector — model and reproduction code

This repository deposits the trained detector from *A lightweight
deep-learning detector for the real-time monitoring of the invasive
frogs* Rhinella marina *and* Polypedates leucomystax, together with the
minimum code needed to reproduce its reported results.

<!-- Add the paper DOI here once it is assigned. -->

The model is a 33,208-parameter CNN quantized to INT8, **66.6 KB** on
disk, that detects two invasive amphibians in 3-second field recordings:

| Code | Species | Common name |
|---|---|---|
| `POLLEU` | *Polypedates leucomystax* | Asian common tree frog |
| `RHIMAR` | *Rhinella marina* | Cane toad |

It also classifies six co-occurring native species (`BUECHO`, `FEJSAK`,
`KUREIF`, `MICKUR`, `NIDOKI`, `ZHAOWS`) as additional multi-label
outputs. However, the paper scores only the two
invasives, so these scripts report only those.

This is **inference only** — no training code. Three scripts reproduce
the per-site results on the three evaluation sets.

## Contents

```
model/model.tflite            66.6 KB INT8 model, 8 sigmoid outputs
model/thresholds_int8.json    per-species decision thresholds
common.py                     audio → spectrogram → inference → metrics
prepare_iriomote.py           builds the Iriomote evaluation windows
eval_ishigaki.py              in-domain test set
eval_iriomote.py              zero-shot, different island
eval_australia.py             zero-shot, different continent
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

None of the audio is bundled here. All three datasets are public. 
Please download them and place them in the `data/` directory.


### Ishigaki and Iriomote

Both come from the Kimura et al. release [1]:
<https://doi.org/10.17605/OSF.IO/JU594>.

Download and unpack it; you need two of the archives inside:

- `train-and-test-datasets.zip` → gives `test/`, 760 WAVs in 31
  label-named folders. Used by `eval_ishigaki.py`. (The `train/` split
  is not needed — this repo does not train.)
- `playback.zip` → gives `playback/`, 134 files. Used by
  `prepare_iriomote.py`.

### Australia

The Cane Toad Acoustic Classifier training data [2]:
<https://doi.org/10.5281/zenodo.13826911>.

About 39,000 clips in per-class folders. `eval_australia.py` treats a
clip as a cane toad positive if and only if its immediate parent folder
is `Rhinella marina_Cane Toad_Modified`; every other folder, including
`Background` and `Unidentified Sounds`, is a negative.

## Running

Each script is a flat top-level script: no arguments, no functions to
call. Settings live in a **Paths** block at the top of each file — edit
those once for your machine, then run the script.

| Script | Constant | Points at |
|---|---|---|
| `prepare_iriomote.py` | `PLAYBACK_DIR` | the unpacked `playback/` directory |
| `prepare_iriomote.py` | `OUT_DIR` | where to write the bundle |
| `eval_ishigaki.py` | `DATA_DIR` | the directory containing `test/` |
| `eval_iriomote.py` | `BUNDLE_DIR` | the bundle (match `prepare_iriomote.py`'s `OUT_DIR`) |
| `eval_australia.py` | `DATA_DIR` | the cane toad dataset root |
| `eval_australia.py` | `CACHE_PATH` | where to cache the extracted spectrograms |

All four also declare `MODEL_PATH`, `THRESHOLDS_PATH` and `OUT_DIR` for
the results, and, optionally, `LIMIT` — set it to an
integer to score an evenly-spaced subset as a quick test.

The defaults assume a `data/` directory next to the scripts:

```
data/kimura/train-and-test-datasets/test/
data/kimura/playback/
data/australia_cane_toad/
```

so if you unpack the archives there, nothing needs editing. Absolute
paths work equally well.

```bash
# 1. Ishigaki — in-domain held-out test set (~1 minute)
python eval_ishigaki.py

# 2. Iriomote — zero-shot playback set (~2 minutes total)
python prepare_iriomote.py
python eval_iriomote.py

# 3. Australia — zero-shot cane toad set (tens of minutes)
python eval_australia.py
```

Each writes a metrics CSV and a bar chart to `results/`. The Australia
set is ~39,000 clips and is by far the slowest; the spectrogram cache
(~1.3 GB, at `CACHE_PATH`) means every later run takes seconds.

## Expected output

Every value below was reproduced from the published data using the
pinned requirements. They should match exactly rather than
approximately, as the pipeline is deterministic end to end.

**Ishigaki** (`results/ishigaki_metrics.csv`) — in-domain, at the
threshold derived from the validation split, which is what a deployed
device would ship with:

| Species | t | F1 | Precision | Recall | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| POLLEU | 0.62 | 0.884 | 0.930 | 0.843 | 107 | 8 | 20 | 625 |
| RHIMAR | 0.79 | 0.865 | 0.882 | 0.849 | 45 | 6 | 8 | 701 |

**Iriomote** (`results/iriomote_metrics.csv`) — zero-shot, 550 windows:

| Species | Strategy | t | F1 | Precision | Recall | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| POLLEU | original | 0.62 | 0.536 | 0.937 | 0.375 | 119 | 8 | 198 | 225 |
| POLLEU | sweep-best | 0.42 | 0.765 | 0.812 | 0.722 | 229 | 53 | 88 | 180 |
| RHIMAR | original | 0.79 | 0.923 | 1.000 | 0.857 | 168 | 0 | 28 | 354 |
| RHIMAR | sweep-best | 0.54 | 0.990 | 0.995 | 0.985 | 193 | 1 | 3 | 353 |

**Australia** (`results/australia_metrics.csv`) — zero-shot, RHIMAR
only. The directory traversal yields 39,123 clips: 22,006 cane toad
positives and 17,117 negatives.

| Species | Strategy | t | F1 | Precision | Recall | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| RHIMAR | original | 0.79 | 0.894 | 0.856 | 0.936 | 20605 | 3476 | 1401 | 13641 |
| RHIMAR | sweep-best | 0.77 | 0.895 | 0.845 | 0.950 | 20915 | 3837 | 1091 | 13280 |

`prepare_iriomote.py` should report 550 windows from 39 files: 358
annotations, 37 background, POLLEU 317, RHIMAR 196.

### The two threshold strategies

On the zero-shot sets each species is scored at two operating points:

- **`original`** — the threshold derived from the Ishigaki validation
  split, applied unchanged. This is the deployment-realistic condition:
  it requires no labelled data at the new site.
- **`sweep-best`** — the threshold maximising F1 on the evaluation set
  itself, selected from a 91-point grid (0.05 to 0.95, step 0.01).
  Reaching this operating point requires labelled data from the target
  site, which the `original` strategy does not; Note that the threshold is
  selected on the same clips it is scored on, so the value reported here
  is an optimistic estimate of what recalibration would achieve on
  further recordings from that site. It is reported in order to separate
  the model's ranking performance from threshold-calibration error.

The difference between the two threshold strategies is also of interest. 
For RHIMAR it is small at both sites, indicating that the detector 
transfers and remains approximately calibrated. For POLLEU at Iriomote 
it is large (F1 0.54 to 0.77): ranking performance transfers but calibration 
does not, and the transferred threshold is substantially too high for that
site.

## How inference works

To apply the model to other recordings, the full chain is:

```
librosa.load(path, sr=16000)
  → truncate / zero-pad to 48000 samples (3 s)
  → Butterworth order-5 low-pass at 6 kHz
  → mel spectrogram: n_mels=128, n_fft=1024, hop_length=750
  → power_to_db(ref=np.max) → crop/pad to 64 frames → (128, 64)
  → per-sample z-score over both axes (mean/std of that clip, eps 1e-6)
  → quantize to int8 with the input tensor's scale/zero-point
  → tf.lite.Interpreter → dequantize the output
  → 8 per-species probabilities in [0, 1]
```

## Reproducing the Iriomote windows

`prepare_iriomote.py` turns the 39 annotated playback recordings into
550 three-second windows. Each annotation row becomes one window centred
on the call (annotations shorter than 3 s — every POLLEU call box) or a
series of 3 s tiles (annotations of 3 s or more — the RHIMAR playback
sessions, which bundle many calls into one row). Up to 50 background
windows per file are then sampled from the regions no annotation
touches.

## Citation

The paper's DOI is not yet assigned. Until then, cite it as:

> A lightweight deep-learning detector for the real-time monitoring of
> the invasive frogs *Rhinella marina* and *Polypedates leucomystax*.

## License

MIT — see `LICENSE`. The datasets are distributed under their own terms;
see their respective records.


## References

[1] Kimura, K., Fukuyama, I. & Fukuyama, K. Deep learning-based detector of invasive alien frogs, 
Polypedates leucomystax and Rhinella marina, on an island at invasion front. 
Biol Invasions 27, 95 (2025). https://doi.org/10.1007/s10530-025-03553-0


[2] Leung, K. W., Allen-Ankins, S., & Schwarzkopf, L. (2024). 
Cane Toad Acoustic Classifier Audio Training Data [Dataset]. 
Zenodo. https://doi.org/10.5281/zenodo.13826911