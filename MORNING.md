# Overnight status

Branch `expressive-vocab-finetune`. Nothing pushed, no PRs opened. Six local commits.

## The headline: two suspects cleared, one real cause found

**`freeze_layers: 8` did not break the audio conditioning.** Fine-tuning *increased*
it. Swapping a batch's audio for a neighbour's costs, on note positions only:

| | own audio | swapped | delta | hurt |
|---|---|---|---|---|
| released, pre-finetune | 4.9165 | 5.0350 | +0.1184 (+2.4%) | 20/20 |
| ours, b6000 | 1.4639 | 1.6530 | +0.1891 (+12.9%) | 20/20 |

**There is no train/inference audio mismatch.** The pre-decoded `.raw` training path
and the engine's librosa+processor path produce the same waveform (cosine ~1.0, gain
1.0, residual 0.3-3% from decoder differences alone).

So the model does use the audio, consistently. It is just weak next to the note-history
prior: per-note perplexity 4.32 with the right audio, 5.22 with the wrong song's. A
model that would rather continue a plausible pattern than follow the music produces
exactly the mechanical charts you saw.

**What was actually broken: we were holding 3.4 GiB of VRAM for a gradient that cannot
exist.** `configs/model/audio_discrete.yaml` had `freeze_encoder: false`, which reads
like it enables encoder training. It cannot -- the discrete path returns quantiser
indices (`int64`, `grad_fn None`), and gradient does not flow through an integer. So
every forward built and retained a full Encodec activation graph nothing backpropagated
through. Measured 4.173 GiB against 0.807 GiB under `no_grad`, byte-identical codes.

That is the VRAM pressure that made me introduce `freeze_layers: 8` in the first place.
It cost decoder capacity to buy nothing.

## Your tempo rule, done and verified

`chart/tempo.py`. Detected 140.006 BPM on the Paparazzi file; first audio note at
0.546s; anchor beat at 0.574s = tick 480, exactly one beat at resolution 480; lead-in
BPM 104.611 so the lead-in lands on the audio's first note. First generated note sits
at tick 480, on the beat line carrying the tempo event.

The anchor is the *tracked beat* nearest the first onset, not the onset itself, so a
note played slightly ahead of the beat does not drag the grid. Beat tracking never
identifies downbeats, so a beat line is the strongest claim the analysis supports --
which is what you specified.

Exact sync is impossible: `.chart` stores BPM as integer milli-BPM. Residual is ~1.5
microseconds, four orders of magnitude under the 20 ms grid.

## New tools

- `evaluate_conditioning.py` -- repeatable audio-ablation, so any future run can be
  checked against the numbers above.
- `chart/responsiveness.py` -- does a chart follow the music? Correlates per-second
  note density against the onset envelope.
- `--guidance` on `generate.py` -- classifier-free guidance, amplifying the one term
  the ablation says exists. Negative branch is real audio, not a zero vector: this
  model never saw audio dropout, so a null input is off-distribution.

## Three mistakes I made and caught

1. First ablation printed `delta +0.0000` for both models. `val_batch_size = 1`, and
   the swap needs two sequences to permute between, so it skipped every batch and
   reported zeros rather than saying it measured nothing.
2. Second version used `torch.randperm`, which returns the identity half the time at
   batch size two -- half the batches compared the audio against itself.
3. I scored loss over all positions. ~86% are padding, which diluted the effect ~7x
   and made me report "fine-tuning changed nothing" when it had roughly doubled it.

Also: I briefly thought the training and inference waveforms were uncorrelated
(cosine -0.001). That was me feeding an MP3 to a raw-PCM reader.

## Metric honesty

`chart/responsiveness` is validated over 60 full human charts: +0.248 against own
audio, +0.054 against a stranger's, winning 46/60. But per-song sd is 0.284, wider
than the effect -- on one song in four a real human chart scores worse against its own
audio than a stranger's. A single song's score is noise. `minimum_songs(0.1) == 33`.

I had started a three-song guidance sweep. By that analysis it could not have concluded
anything, so I stopped it and replaced it with a 36-song paired version.

## Running overnight

1. VRAM sweep -- a real training step at `freeze_layers` 8 / 4 / 0, batch 1 and 2, to
   see what the freed 3.4 GiB buys. Result in `G:\a2c_data\` task output.
2. 36-song paired guidance sweep, resumable, appending to
   `G:\a2c_data\guidance_sweep.jsonl`, log at `G:\a2c_data\guidance_sweep.log`.

## Retracted: precomputing Encodec codes is not the fix

I had this queued as the big throughput win, carried from an old estimate that the
step was ~9.8s against ~0.2s of model math. Profiled properly at freeze_layers 0:

    dataloader fetch (decode + window)   0.292s    2.0%
    encodec encode                       0.246s    1.6%
    transformer fwd+bwd+step            14.436s   96.4%
    total                               14.973s

Precomputing the codes would save 1.6%. Data loading is already fine -- the lazy LRU
cache and four workers did their job. The whole cost is the transformer's own
forward+backward, which is absurd for a 230M model at batch 1.

The config sets no precision at all, so Lightning runs 32-true and TF32 is never
enabled. Benchmarking fp32 / TF32 / bf16 now; that is where the time is.

## Confirmed since the first draft

- `freeze_layers: 0` runs in the *real* pipeline, not just synthetically: 48 batches,
  no OOM, 7938 MiB steady, loss stepping normally. The good b6000 checkpoint is
  untouched.
- `--snap` now works. It rounds against the song's beat grid, and until the tempo fix
  there was no grid to round against. At 140 BPM a generated eighth-note run came out
  246/247/224 ticks against a true 240; snapping fixes that without moving the first
  note off its beat line.
- Second effect of the `freeze_encoder` bug: `audio_encoder.eval()` was never called,
  so Encodec ran in *train* mode for the whole 24-hour run. It is deterministic now.

## Decisions waiting for you

- **The next training run's config.** With the encoder fixed we can likely drop
  `freeze_layers` and/or raise batch size; the sweep says which. This is the run that
  should actually improve conditioning, so it is worth choosing deliberately.
- **Audio dropout for real CFG.** Current guidance uses a wrong-song negative because
  the model has no null-audio concept. Training with ~10% audio dropout would give it
  one and make guidance considerably stronger. Cheap to add, needs a retrain.
- Still open from before: PR #3 is far behind; the pattern questions (tremolo naming,
  `ladder descending split`, what unqualified "trip zig" means).
