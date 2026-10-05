# A lightweight invasive-frog detector; model and reproduction code

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
outputs from the recordings in Ishigaki island, Japan. However, the paper scores only the two
invasives, so these scripts report only those.


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
  is not needed).
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

Each script is a flat top-level script with no arguments. Settings live in a **Paths** block at the top of each file — edit
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

### The two threshold strategies

On the zero-shot sets each species is scored at two operating points:

- **`original`** — the threshold derived from the Ishigaki validation
  split, applied unchanged. This is the deployment-realistic condition as
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