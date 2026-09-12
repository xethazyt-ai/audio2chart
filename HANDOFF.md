# audio2chart — session handoff

Updated 2026-09-11 evening. **Supersedes everything in this file before today**, which
was written 2026-09-05 and said "Nothing is running. No checkpoint exists." Both are now
false. `MORNING.md` covers the grid change in more depth and is still accurate; this file
is the current state.

Everything below was measured on this machine.

---

## The throughput problem is solved, and the earlier diagnosis of it was wrong

Two days went into the dataloader — filling the dead audio cache, raising worker counts,
writing a hand-rolled loader profiler. Lightning's own profiler over 256 batches of the
real training config:

```
run_training_epoch      91.7 s   (0.358 s/batch, 2.79 it/s)
training_step           0.176 s  49%
backward                0.130 s  36%
optimizer_step          0.311 s  11%
train_dataloader_next   0.0005 s  0.1%
```

**The loader has never been the bottleneck.** Filling the audio cache moved throughput
0.77 → 0.82 it/s, not the 87% the old comment in `dataloader/audio_loader.py` claimed;
that claim is retracted in `31ddf89`.

The real cause was a unit mismatch between two config keys that sit next to each other
and look symmetrical:

- `val_check_interval` counts **batches**
- `ModelCheckpoint`'s `every_n_train_steps` counts **optimizer steps**

Both were launched at 100 with `accumulate_grad_batches=8`, so the run validated once per
training batch and wrote a 2.42 GB checkpoint every 100 batches — onto G:, which measures
**108 MB/s** and is the same drive streaming the songs. C: measures 704 MB/s.

| | before | after |
|---|---|---|
| throughput | 0.82 it/s | **2.79 it/s** |
| GPU utilisation | 16% | **87%** |

`trainer.log_dir` now moves checkpoints and CSV metrics off the corpus drive. **An idle
GPU waiting on disk is indistinguishable from a starved loader in `nvidia-smi`** — that
is why the wrong diagnosis survived so long. Check the profiler before believing either.

---

## Sync: how exact are the ticks?

Robert's requirement was "ticks need to be exact. I dont want any sync issues."

**Exact, with a known boundary.** `tests/test_tick_exactness.py` runs real tick positions
through the whole pipeline — ticks → seconds → `discretize_time` → seconds →
`choose_snap` → ticks — and requires identity, not closeness. Swept over 120–400 BPM and
16th through 96th notes:

- every subdivision whose spacing clears the 10 ms grid step recovers **exactly, 0.0 ms
  error**, down to a 12.5 ms gap (96ths at 200 BPM, 64ths at 300)
- triplets and mid-song tempo changes included; a BPM rounded to `###.#` stays exact
- below the grid step (8.3 ms) discretization **raises and the window is dropped** — it
  refuses rather than shipping a desynced chart
- the one case that drifts is an irregular burst on no common subdivision (ten ticks at
  200 BPM, 15.6 ms), where `choose_snap` has nothing to snap to; residual stays inside
  half a grid step

`tests/test_audio_token_alignment.py` covers the invariant underneath all of that: the
audio slice and the token grid describe the same instant. It puts a click in the audio
where a note sits in the chart and requires the click's position in the returned chunk to
equal the note's grid index times the grid step, for a window at t=0 and one mid-song.

**The `Offset` field matters more than it looks.** 208 of the 1454 songs in the tapping
training split carry a non-zero one, 199 by more than a grid step, median 0.150 s and up
to 5.0 s. Scored against the audio onset envelope over the 46 songs with |offset| ≥ 1.0 s:
as-written **+0.197** > ignored **+0.140** > inverted **+0.089**, paired t = +3.78. That
ordering is what a correct sign looks like. It holds.

Also checked: only 2 of 1454 charts outlast their audio, both by 0.02 s. Silence-padded
windows with real note targets are not a problem in this corpus.

**The grid could still go finer for free.** Measured over the whole 1454-song tapping
split rather than the 120 hand-picked charts the original decision used:

| grid | window | tokens | windows usable | songs 100% | songs dropped |
|---|---|---|---|---|---|
| 20 ms | 30 s | 1502 | 59.9% | 36% | 293 |
| 10 ms | 15 s | 1502 | 97.0% | 80% | 7 |
| **8 ms** | **12 s** | **1502** | **98.8%** | **90%** | **3** |
| 6 ms | 12 s | 2002 | 99.5% | 95% | 1 |

8 ms at 12 s costs the identical 1502 tokens and is strictly better data. Not taken
because changing the grid invalidates the validation cache and wastes any run in
flight — it is a start-of-next-run change. Note also that the "18.2% usable at 20 ms"
figure quoted everywhere came from those 120 charts; on the training split it is 59.9%.
The change was still right, but the smaller number does not describe this corpus.

The loader retries a window that collides and draws another: 197 retries over ~5000
batches in Run A, 189 resolved on the second attempt, zero exhausted. Working as
designed, but it means the 3% of unusable windows are silently skipped rather than
reported, and they concentrate in the fastest 25 songs.

---

## Training runs in flight

Two runs, chained, on the tapping subset (1447 train / 363 val), scored afterwards with
`score_run.py`:

- **Run A** — `model.init_from` the b6000 checkpoint (carries chart knowledge, but was
  trained entirely at 20 ms with the sync error baked in)
- **Run B** — from the released checkpoint, no init

Both: `loader.train_num_pieces=1 trainer.accumulate_grad_batches=8 model.freeze_layers=4
model.input_noise=0.08 trainer.max_steps=1500 trainer.val_check_interval=1200
trainer.limit_val_batches=100 trainer.checkpoint_every_n_steps=200
trainer.log_dir=C:/a2c_runs`

Note `val_check_interval=1200` is 150 optimizer steps, not 1200 of them. Read the warning
in `configs/default.yaml` before changing it.

The falsifiable question for scoring: the model learned note spacing in position units at
20 ms, so at 10 ms every rhythm it knew is wrong by exactly 2×. That should wash out in
fine-tuning. If it did not, note density comes out around double and the inter-onset
histogram clusters at half the right intervals.

**RETRACTED: Run A is not overfitting.** An earlier version of this section said it
was, from three validation points after a peak. The next point made a new high and the
claim did not survive:

| step | val | train | gap |
|---|---|---|---|
| 692 | 0.660 | 0.660 | +0.000 |
| 873 | 0.650 | 0.668 | +0.018 |
| 1054 | 0.639 | 0.672 | +0.033 |
| **1235** | **0.666** | 0.672 | **+0.006** |

The train/val gap widened to +0.033 and then collapsed back to +0.006 as validation hit a
new high. That is a noisy metric, not divergence.

The noise has a cause worth fixing: `limit_val_batches=100` with `val_batch_size=1` and
`val_num_pieces=1` computes the monitored metric over **100 windows**. A swing of ±0.013
around 0.653 is what that sample size produces, and the monitored checkpoint is selected
on it — so which weights get saved as "best" is partly luck. Raising `limit_val_batches`,
or `val_num_pieces`, would cost little (validation is a small share of the run now that
checkpoints are on a fast drive) and would make checkpoint selection mean something.

Score `best-checkpoint.ckpt` rather than `last.ckpt` regardless: it is monitored on
val/acc_nonpad_epoch, so it tracks the peak wherever the peak lands.

---

## What the model actually produces (probe, one song)

Run A's best checkpoint, generating over a 60 s validation clip, scored against the
tapping baseline. Three sampling settings, to separate a sampling artifact from a
property of the model:

| setting | nps | rests/min | pattern lift | coverage |
|---|---|---|---|---|
| default (t 0.5, top_k 32) | 71.26 | **0.0** | +0.579 | 0.840 |
| pad-bias 0.4 | 69.30 | **0.0** | +0.684 | 0.960 |
| t 1.0, top_k 128, pad-bias 2 | 46.66 | **0.0** | +0.538 | 0.804 |
| **human tapping** | **21.01** | **15.1** | +0.416 | ~0.68 |

Two things, and they point in opposite directions.

**The pattern vocabulary is good.** Catalogue coverage 0.80-0.96 against a chance floor
around 0.27, lift at or above the human median in all three settings. Whatever else is
wrong, the model is producing real charting shapes, not noise. That is the thing the
catalogue work was for and it appears to have landed.

**It does not know when to stop.** Density runs 2.2x to 3.4x human and never comes down
to it; rests are *exactly zero* at every setting, against a human 15.1 a minute. The
model fills about 71% of all grid slots at the shipped defaults. Sampling moves density
(71 -> 47) but not phrasing, so this is structural, not a knob that was set wrong.

Two separate faults, measured apart. Teacher-forced on real history the model predicts
pad 79.3% of the time against the data's ~86% (`val/pad_pred_rate` 0.793, pad true
positives 0.936). Generating, it emits pad about 29% of the time. So most of the damage
happens at generation, but not all of it.

Density of the generated chart by position inside a 15 s chunk, 100 ms bins, as a share
of the 10 grid slots each bin contains:

    first 300 ms   4.9/10
    300-1000 ms    6.5/10
    1-2 s          7.3/10
    plateau        7-8/10
    human          ~1.4/10

**The prior is wrong before drift can start.** At the first token the context is nothing
but bos and the audio, and it already fills 54% of slots against a human 14% -- roughly
four times too dense with no opportunity to have drifted yet. Class weighting or a pad
prior is the lever there, not scheduled sampling.

**Then it drifts, and locks in fast.** Density climbs from 54% to about 75% over roughly
130 tokens and stays. That is self-reinforcing: a dense history tells the model it is in
a dense passage. `input_noise=0.08` was aimed at this and is clearly not enough.

A caution about how this was measured, because the first attempt got it backwards. Binned
at 5 s, the profile reads 65.8 nps in the first third of a chunk against 52.0 in the last
-- ratio 0.79, which looks like *no* accumulation and briefly seemed to refute the drift
reading. Saturation completes inside the first bin, so that test could not see it. Use
100 ms bins over the first two seconds, not thirds of a chunk.

The rests figure is the starkest number here: the longest gap anywhere in 60 s of output
is 0.040 s, and the median gap is 0.0099 s -- one grid step. The model does not pause at
all, ever, while human tapping charts rest for a quarter second or more about fifteen
times a minute.

Note what this is *not*: evaluate_chart's docstring predicted the opposite failure -- zero
chords, zero taps, almost no sustains at the shipped defaults. That is not what this
checkpoint does. The docstring describes an older checkpoint and should not be trusted
for this one.

**One song.** The rests=0 result is consistent across three settings, but all three are
the same clip. The 6-chart scoring run is what decides whether it generalises, and no
conclusion here should be repeated until it does.

---

## Two tools were silently broken

**`fit_tier.py` had never run.** `main()` read `args.min_tier`; `parse_args` never declared
it, so every invocation died with `AttributeError` before touching a chart. That is why it
kept reappearing on the to-do list as "never run". `tests/test_cli_arguments.py` now parses
every top-level tool and requires each `args.<name>` to be declared; the rest are clean.

**`audit_catalogue.py` could not see chord patterns.** `parse` resolves one letter at a
time, so `'RB'` matches no fret and `'RB RYB RB'` signs as the empty tuple. All four
chorded entries collided into one reported "duplicate", then fell below the `len(sig) >= 2`
filter and were never scanned against a single chart — while the report said "ABSENT:
none". Distinct bodies went 70 → 74 once fixed.

**`tests/test_tempo.py` imported pytest**, which is installed in neither environment here,
so the whole file raised ImportError under unittest discovery and none of its nine tests
had ever executed — including the BPM-rounding one. Its rounding test also asserted that
the string `"round("` appeared in `detect_tempo`'s source, which would pass with the
rounding commented out. Both fixed.

---

## The tier grader does not work, and more features will not fix it

`fit_tier.py`, first real run:

```
two human label sources disagreeing with each other   5.47 tiers
this model, held out                                  5.43 tiers
predicting a constant                                 6.26 tiers
```

17% of charts within two tiers on a 1–26 scale. **The model is already at the label noise
floor.** Median signed difference between the two sources is 0, so there is nothing to
calibrate away — charters do not agree with each other, and a constant prediction (5.30)
beats one source predicting the other.

The features are fine. Rank correlation *inside one author's setlist* has median
best-feature |rho| **0.50** — 0.86 (nps, Shobas_ Charts), 0.69 (peak10, Community Track
Packs) — against **0.26** pooled across packs. Difficulty is tracked; the tier numbers are
not on one scale.

What would work: a per-pack offset fitted alongside, or pairwise judgements from one
person. Robert has a consistent scale in his head — TTFAF in single digits, Schmoo's World
around 24 — and that is the label source this needs.

---

## Environment traps

- The training interpreter is **`.venv313`** (torch 2.9.1+cu128, RTX 3060 Ti, 8 GB).
  `.venv` is a different, CPU-only install whose `pyvenv.cfg` points at a `C:\Users\rober`
  path that does not exist on this machine. `python` on PATH is a third install with no
  torch. Use `.venv313/Scripts/python.exe`.
- **pytest is not installed anywhere.** Tests run under `python -m unittest discover -s
  tests -t .` — 301 passing.
- `pkill -f` does not reach native Windows processes. Use
  `Get-CimInstance Win32_Process` and `Stop-Process`.
- Writing Python through a bash heredoc mangles backslash escapes (`\a`, `\t`, `\U`).
  Patch from a file, or edit directly.

---

## Open, in rough priority order

1. **Score both runs** with `score_run.py` once they finish, over several charts. One
   chart characterises nothing — across eight from this model sustains ran 0.000–0.655
   and pattern lift −0.172 to +0.383, and a day of conclusions from a single chart had to
   be retracted.
2. **`triangle slide 8-note` and `sweep 8-note` hold the identical sequence**
   `G R Y B O B Y R` under two names, which double-counts in catalogue coverage. Which
   name is right is Robert's call; left alone rather than merged.
3. **Stem separation** — 2,706 overchart songs, ~9 GPU-hours, ~37 GB. Built and ready,
   waiting on a scope decision.
4. **Cross-chunk context** — architectural, and more pressing since the 15 s window
   doubled the number of chunk boundaries in a song.
5. **The frozen Encodec encoder is ~18% of every step** (0.065 s of a 0.358 s batch) and
   recomputes features that cannot change while `freeze_encoder: true`. Precomputing codes
   per window would buy roughly 1.2×, at the cost of foreclosing encoder fine-tuning and
   waveform augmentation. Not done.
6. **`tests/test_grid_window.py::test_engine_default_matches_nothing_by_accident`
   asserts on source text** rather than behaviour — it would pass if someone wrote
   `chunk_sec = 30.0`. Making it behavioural means touching the hash-pinned
   `inference/engine.py`, so it was left for a deliberate pass.
7. Sampling defaults need a multi-clip comparison (GPU, so not while training).
8. Nothing has been pushed. The branch is far ahead of PR #3.
