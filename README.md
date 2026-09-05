# audio2chart

[![arXiv](https://img.shields.io/badge/arXiv-2511.03337-b31b1b.svg)](https://arxiv.org/abs/2511.03337)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/audio2chart/blob/main/notebooks/audio2chart_charting.ipynb)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Collection-orange)](https://huggingface.co/collections/3podi/audio2chart-v10)

**audio2chart** is an open-source deep learning framework for **audio to chart generation**, converting raw audio into structured `.chart` files used in Guitar Hero style rhythm games.

Input: a `.wav` or `.mp3` audio file (minimum 30 seconds)
Output: a playable `.chart` file compatible with Clone Hero

A complete description of the methodology, architecture, and experiments is in the [arXiv publication](https://arxiv.org/abs/2511.03337).

---

## Quick start on Google Colab

Try it in your browser with no local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/audio2chart/blob/main/notebooks/audio2chart_charting.ipynb)

The notebook installs dependencies, downloads a pretrained model from Hugging Face, and transcribes your own `.mp3` or `.wav` into a `.chart`.

---

## Requirements

**Python 3.13 is required** for the pinned `requirements.txt`. This is narrower than it looks, and installing on anything else will fail:

| pin | constraint |
|---|---|
| `audioop-lts==0.2.2` | Python **>= 3.13** (backports the `audioop` module removed in 3.13) |
| `networkx==3.5` | Python **>= 3.11** |
| `numba==0.62.1` | ships wheels for cp310-cp313; its sdist hard-refuses **>= 3.14** |

To run on Python 3.11 or 3.12, drop the `audioop-lts` line first. There is no supported path on 3.10 or 3.14.

## Installation

```bash
git clone https://github.com/3podi/audio2chart.git
cd audio2chart
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then:

```bash
pip install -r requirements.txt
```

### GPU support

The `torch` wheels on PyPI are **CPU-only on Windows**. `pip install -r requirements.txt` therefore gives you a CPU build, and `torch.cuda.is_available()` returns `False`. For CUDA, reinstall the torch trio from the PyTorch index:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 --force-reinstall torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

Inference works on CPU; a 3-minute song takes roughly two minutes. Training on CPU is not practical.

---

## Inference

```bash
python generate.py path/to/audio.wav
```

This downloads the default model (`3podi/charter-v1.0-40-M-best-acc`) plus the `facebook/encodec_24khz` audio encoder (~89 MB total, cached after the first run) and writes `./<song_name>/notes.chart`.

Note that `--output` takes a **directory**, not a filename. The file inside is always named `notes.chart`.

### The beat grid

By default the chart is written with a **fixed 200 BPM** `[SyncTrack]`. Notes still line up with the audio, because they are placed by absolute time, but the bar lines and beat grid are musically meaningless: the chart will look wrong in an editor and will not snap to anything.

Three ways to get a real grid, best first:

**1. Reuse an existing chart's tempo map** (exact, no estimation):

```bash
python generate.py song.wav --sync-from path/to/existing/notes.chart --snap 32
```

Copies `Resolution` and the full `[SyncTrack]` (including tempo changes and time signatures) from a chart you already have. Useful for re-charting a song, or generating an extra difficulty.

**2. Detect the tempo from the audio** (automatic, accurate for steady tempo):

```bash
python generate.py song.wav --detect-tempo --snap 32
```

Runs a phase-locked search over the onset envelope. This is deliberately *not* `librosa.beat.beat_track`, whose tempo is octave-unstable and too imprecise to hold sync over a full song. On a 193 s test track the search landed within 0.0024% of the tempo a human charter had entered by hand (about 5 ms of drift end to end) in roughly 6 seconds.

**3. Enter the tempo yourself:**

```bash
python generate.py song.wav --bpm 168 --snap 32
```

### Snapping

`--snap N` rounds every note to the nearest 1/N note. It is only meaningful once the grid is correct, so pair it with one of the options above.

The model emits tokens on a fixed millisecond grid (40 ms for the `40-` checkpoints), so raw output lands on arbitrary ticks. At 168 BPM a 1/32 note is 44.6 ms, which makes `--snap 32` a near-lossless fit: no note moves more than half a grid step (22.3 ms at that tempo). The result reads like a hand-authored chart in Moonscraper instead of scattered off-grid notes.

Snapping can round two adjacent tokens on the same lane onto the same tick; duplicates are dropped, which slightly reduces the note count.

### Options

| flag | default | meaning |
|---|---|---|
| `audio_path` | *(required)* | input audio, must be >= 30 s |
| `--model_name` | `3podi/charter-v1.0-40-M-best-acc` | Hugging Face model id or local path |
| `--temperature` | `0.5` | sampling temperature |
| `--top_k` | `32` | top-k sampling |
| `--sync-from` | none | copy `Resolution` + `[SyncTrack]` from an existing `.chart` |
| `--detect-tempo` | off | estimate tempo from the audio |
| `--snap` | `0` (off) | snap notes to the nearest 1/N note |
| `--bpm` | `200` | fixed BPM, used only without `--sync-from` / `--detect-tempo` |
| `--resolution` | `480` | ticks per quarter note (overridden by `--sync-from`) |
| `--name` `--artist` `--album` `--genre` `--charter` | derived | chart metadata |
| `--output` | `./<song_name>/` | destination **directory** |

Full example:

```bash
python generate.py song.wav --name "Song Title" --artist "Artist" --album "Album" --genre "Metal" --detect-tempo --snap 32 --output ./out
```

### Using the model directly

```python
from inference.engine import Charter

model = Charter.from_pretrained("3podi/charter-v1.0-40-M-best-acc")
seqs = model.generate("path/to/song.wav")
```

---

## Known issues

### Some audio files fail to decode

The loader is `librosa.load`, which tries libsndfile and falls back to `audioread`. Two ways this bites:

- libsndfile's bundled **Ogg Opus** support is incomplete. It will happily read the header and report a correct duration, then fail partway through decoding with `LibsndfileError: Supported file format but file is malformed` on a file that is not actually malformed.
- The `audioread` fallback then needs an external decoder (ffmpeg or gstreamer). Without one you get `audioread.exceptions.NoBackendError` and the run dies.

Also note that a file's **extension is not checked**. A `song.mp3` that is really an Ogg Opus stream is common in Clone Hero song folders and hits exactly this path.

Workaround: transcode to WAV first. With ffmpeg installed:

```bash
ffmpeg -i song.mp3 -ac 1 -ar 24000 song24k.wav
```

Or without installing anything system-wide, in a throwaway virtualenv (PyAV bundles its own FFmpeg):

```bash
pip install av numpy
```

To confirm what a file actually is:

```bash
python -c "import soundfile; print(soundfile.info('song.mp3'))"
```

### Unicode crashes on Windows consoles

`generate.py` contains emoji in the argparse description and in the final success message. On a Windows console using the cp1252 code page both raise `UnicodeEncodeError`:

- `python generate.py --help` fails outright.
- A successful run prints a traceback **after** the chart has already been written. The chart is fine; only the confirmation message failed.

Workaround:

```bash
PYTHONIOENCODING=utf-8 python generate.py song.wav
```

### Chart metadata is partly hardcoded

The template in `chart/chart_writer.py` always writes `MusicStream = "song.ogg"`, `Year = ", 2022"`, `Difficulty = 3` and `Player2 = bass`, regardless of the actual audio filename or the metadata you pass. If your audio is not named `song.ogg`, fix `MusicStream` by hand or rename the file.

### Time signatures are not detected

`--detect-tempo` estimates tempo only. The written time signature is always `4/4`. Only `--sync-from` carries real `TS` events across.

### Tempo detection assumes a roughly constant tempo

`--detect-tempo` fits a single BPM plus phase, searching **100-200 BPM** by default. Songs with genuine mid-song tempo changes, or a true tempo outside that range, need `--sync-from` or a manual `--bpm`. The lead-in before the first beat is absorbed into a short first tempo segment so that tick 0 still maps to t=0 and `Offset` stays 0.

### The frozen-surface test fails on Windows checkouts

`tests/test_architecture_surface.py` pins SHA-1 hashes of the raw bytes of the inference files. With Git's `core.autocrlf=true` (the Windows default) those files are checked out with CRLF line endings, so the hashes never match and `test_frozen_inference_surface_is_byte_identical` fails. The pins are correct for a normal LF checkout.

The test asserts on the first mismatch, so one failing file masks all the others.

Fix, if you want it to pass locally, by adding a `.gitattributes`:

```
*.py text eol=lf
```

---

## Hugging Face models

Pretrained models are published at [huggingface.co/3podi](https://huggingface.co/3podi). Variants differ by:

- **Time resolution:** `20` ms or `40` ms
- **Model size:** `S` (~25 M params) or `M` (~225 M params)
- **Checkpoint type:** `best-acc` / `best-acc-nonpad`

Browse them in the [Charter v1.0 collection](https://huggingface.co/collections/3podi/audio2chart-v10).

The repository default is `3podi/charter-v1.0-40-M-best-acc`.

---

## Training

### Baseline (no audio conditioning)

`baseline.py` is a Transformer decoder that predicts chart tokens symbolically. Dataset:

```bash
huggingface-cli download 3podi/audio2chart-charts charts.zip --repo-type dataset -d <data_path>
```

```bash
unzip <data_path>/charts.zip -d <data_path>/charts
```

```bash
python baseline.py
```

Configuration is handled by **Hydra** from the `configs/` directory. Override any parameter on the command line:

```bash
python baseline.py root_folder=<data_path> is_discrete=True window_seconds=30 grid_ms=40
```

### Audio-conditioned model

`main.py` trains the model that maps encoded audio features to chart tokens. The default uses frozen pretrained Encodec features; a trainable SEANet encoder is selectable through Hydra:

```bash
python main.py model=audio_discrete
```

Both encoder choices use the same discrete charting Lightning module.

---

## Tests and benchmarks

The dependency-light chart, tokenizer, and timing suite needs no ML stack:

```bash
python -m unittest tests.test_chart_pipeline
```

Full suite (see the CRLF caveat under Known issues):

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Dependency-heavy tests skip with a clear reason when the training stack is unavailable. Performance comparisons are kept separate:

```bash
python tests/benchmark_chart_pipeline.py
```

```bash
python tests/benchmark_chart_pipeline.py --max-ratio 1.25
```

The benchmark verifies reference/refactored output equivalence before reporting median timing ratios for small, typical, and large synthetic charts. `--max-ratio` makes it fail when any ratio exceeds the threshold.

---

## Repository structure

```
audio2chart/
├── inference/
│   ├── model_inference.py      # Inference model with KV-cache support
│   ├── layers.py               # Self-contained inference model primitives
│   └── engine.py               # Inference engine
├── chart/
│   ├── tokenizer.py            # Tokenization utilities
│   ├── chart_processor.py      # Chart parsing / preprocessing
│   ├── time_conversion.py      # Note times <-> tick values
│   ├── tempo.py                # SyncTrack parsing and tempo detection
│   └── chart_writer.py         # Writes decoded charts to .chart format
├── dataloader/
│   ├── convert_to_raw.py       # Convert audio to raw format
│   ├── audio_io.py             # Audio reading helpers
│   ├── audio_loader.py         # Dataloader for audio-conditioned training
│   ├── notes_loader.py         # Dataloader for notes-only training
│   └── utils_dataloader.py     # Dataset utils
├── modules/
│   ├── models.py               # torch.nn models
│   ├── transformer_layers.py   # Training model primitives
│   ├── trainer.py              # Lightning training modules
│   ├── training_transformer.py # Audio-conditioned training model
│   ├── scheduler.py            # LR schedulers
│   ├── utils_train.py          # Training utils
│   └── run_utils.py            # Training run setup and lifecycle
├── notebooks/
│   └── audio2chart_charting.ipynb  # Colab notebook for charting
├── baseline.py                 # Baseline training script
├── main.py                     # Audio-conditioned training script
├── generate.py                 # Main inference entry point
└── requirements.txt
```

---

## Citation

```
@misc{tripodi2025audio2chartendendaudio,
      title={audio2chart: End to End Audio Transcription into playable Guitar Hero charts}, 
      author={Riccardo Tripodi},
      year={2025},
      eprint={2511.03337},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2511.03337}, 
}
```

---

## Contact

For questions, discussions, or contributions, open an Issue on GitHub.

## Acknowledgments

This project was supported by **AMD** and the **AMD Developer Cloud**, whose compute resources made training and experimentation possible.

---

⭐ If you find this project useful, please give it a star!
