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
