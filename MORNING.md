# Status

Branch `expressive-vocab-finetune`. 13 commits over the night and day, nothing pushed,
no PRs opened, `configs/audio.yaml` untouched. 242 tests passing.

## Three faults, separated

The Paparazzi chart was bad for three independent reasons. Each needs a different fix,
and treating them as one problem is why the diagnosis took so long.

**1. It is missing most of the chart vocabulary.** Zero chords, zero taps, 0.1%
sustains, against human medians of 0.128, 0.194 and 0.049. Not the model's fault --
teacher-forced it predicts all three at near-human rates (taps 0.181 against a true
0.197), and the decode/writer path is bit-exact. The cause is exposure bias in the
decoding loop: the tap rate runs away 0.224 -> 0.723 across one 30s chunk as the model
conditions on its own output, and collapses to a flat zero at a lower temperature. No
sampler setting fixes it. *Fix staged: `model.input_noise`, off by default.*

**2. It does not follow the music.** Audio conditioning is real and consistent but
weak: swapping in another song's audio costs +0.1891 nats on note positions and hurts
on 20 of 20 batches, while the note-history prior carries the rest. *Fix staged:
`freeze_layers: 4` gives 45% more trainable parameters in the blocks that read audio.
Classifier-free guidance is being measured now.*

**3. It never rests, and it overspends the hard patterns.** No gap between notes
exceeds 250 ms anywhere in the chart -- not rare, none -- where 118 of 120 human Expert
charts have broader rhythm. And Expert-marker patterns make up 14.9% of its pattern
instances against a human 2.7% (+3.23 sd), where a charter spends one for emphasis.
Real Expert charts are ordinary vocabulary that becomes hard in places. *No fix
staged.*

Two things I claimed here first and had to retract, both from judging a number without
a baseline: the chart's density is *more* variable than 93% of human charts (0.606
against a human mean 0.409), not flat; and its 235-note mean run is normal for Expert,
whose own mean is 232 -- I cited it as a defect against a measurement I had taken that
morning. The long runs are a consequence of never resting, not a separate problem.

## The good news

The model's pattern vocabulary is human-level: lift +0.230 against a human mean +0.209
and median +0.227, with 31 of 60 real charts scoring lower. It has learned to write
guitar. Every fault above is about where notes go, what they are made of, and how they
are paced -- not whether it knows the shapes.

## Ready for the next run

    loader.train_num_pieces: 1          (was 2)   ~10x faster, better gradient diversity
    trainer.accumulate_grad_batches: 8  (was 4)   same effective batch
    model.freeze_layers: 4              (was 8)   +45% trainable, in the audio blocks
    model.input_noise: 0.05-0.1         (was 0)   against fault 1

An epoch drops from ~22 hours to under 3. `evaluate_conditioning.py` scores fault 2 on
a checkpoint, `evaluate_chart.py` scores faults 1 and 3 on a chart.

## Answering the difficulty question

Difficulty is qualitative, not density. Over 120 songs charted at all four levels by
the same charter, notes per second rises only 1.8x across the whole ladder, while mean
run length is flat through Hard and then quadruples at Expert, where the tap rate also
jumps 72%. Chords peak at HARD. Sustains run backwards.

Catalogue patterns are now graded from the corpus rather than by hand -- 165,792
occurrences, in `chart/grades.py`. Read against the null of 2.76, not 2.5: Expert
sections are denser, so every pattern carries an Expert bias for free. Corrected, only
13 of 67 patterns mark difficulty at all; the rest is universal vocabulary.

Together those say difficulty comes from *arrangement*, not from exotic pattern
selection -- which argues for JM's annotated-section registry over a scalar knob.

**One question for Robert:** `trip zig split G-Y-O` grades +1.02 above null and `trip
zig split G-B-O` grades -0.29 below it. Same family, opposite ends. G-Y-O splits evenly
(deltas 2, 2), G-B-O does not (3, 1). Is split regularity what makes these hard, or are
47 occurrences of G-B-O too few to trust?

---

# Detailed record

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

## The throughput answer: we have been paging VRAM all along

`train_num_pieces: 2` puts two 30-second windows in every batch. Measured:

    seqs 1  freeze 8    0.315s/step   peak  5.905 GiB
    seqs 2  freeze 8    6.194s/step   peak  9.476 GiB
    seqs 2  freeze 0   10.424s/step   peak 11.040 GiB

The card has 8 GiB. Two sequences need 9.5 GiB, and Windows' WDDM driver does not
refuse -- it oversubscribes into host RAM, so instead of an out-of-memory error the
step runs 19.7x slower. That is why every component measured fine in isolation: each
one is fine, until the working set crosses 8 GiB and every access starts crossing
PCIe.

Gradient accumulation gives the same effective batch without the cliff:

    now       pieces 2 x accum 4   ->  4 x 6.194s = 24.8s per optimizer step
    proposed  pieces 1 x accum 8   ->  8 x 0.315s =  2.5s per optimizer step

Same eight sequences per update, about 10x faster. An epoch goes from roughly 22
hours to roughly 2.2.

Two genuine throughput bugs fixed on the way, both cheap operations made expensive by
how they touch the GPU, and neither visible in the source:

- The audio encoder ran under bf16 autocast, where cuDNN has no fused LSTM kernel, so
  SEANet's 2250-step recurrence fell back to an unfused path: 0.130s -> 0.712s. Worth
  about 2.3s/batch in practice.
- LogGradientNorm summed `.item()` per parameter, one GPU->CPU sync each, ~470 stalls
  per optimizer step to log a single number: 1.09s, 2.5% of training time.

Things I chased that turned out not to matter, so nobody repeats them: precomputing
Encodec codes (1.6% of a step), the dataloader (0.034%), per-window chart
re-processing (0.004s), and the optimizer (19ms -- the profiler's 8.57s
`optimizer_step` double-counts the closure's forward and backward).

The dead `_audio_cache` is still a real bug -- it is read but never written, so
`max_cache_gb` and the whole chunk-eviction system control nothing. It is not a
throughput problem, and switching it on would risk 6 GiB per worker across 4 workers,
which is the host-RAM blowup from earlier in this project. It should be deleted, not
activated.

## 11 Sep: the charts are missing most of the vocabulary, and it is not the model

Separate fault from the audio-conditioning one, and probably the bigger part of why
the Paparazzi chart looked the way it did.

Profiled against the corpus, the generated chart had **zero chords, zero tap notes and
0.1% sustains** where human charts have 12.7%, 47% and 22.9%, plus 6x too many opens,
2x too many greens, a mean run of 235 notes and 99.7% of its note spacings in two
adjacent buckets. That is "mechanical" made measurable.

The model is not the problem. Teacher-forced it is well calibrated:

    chord    true 0.205   model mass 0.171
    tap      true 0.197   model mass 0.181
    sustain  true 0.071   model mass 0.091

And the output path is lossless -- a human chart round-tripped through
convert_notes_to_ticks -> decode -> fill_expert_single comes back bit-identical
(tap 0.993 -> 0.993).

**It is exposure bias in the decoding loop.** Across one 30s chunk at temperature 1.0
the tap rate climbs 0.224 -> 0.454 -> 0.671 -> 0.723 as the model conditions on more of
its own output. At temperature 0.5 it is a flat 0.000 with no sustains at all:
sharpening suppresses the runaway and the vocabulary together. A pad-logit bias, which
should have separated density from vocabulary, collapses output 22-fold over one unit
and suppresses chords and taps preferentially -- the signature of a self-reinforcing
loop, where what the model emits becomes the history it reads.

So no sampler setting fixes this, and the three I tried are three knobs on one loop.
The fix is in training -- scheduled sampling, or simply more of it now that a run is
~10x faster. `--pad-bias` is kept as a no-op-by-default diagnostic with the numbers in
its docstring.

Also settled: universal rule 1 (no note overlapping) is real -- zero violations in
836,327 human notes across 300 charts. Our output satisfies it, but vacuously, because
it emits almost no sustains.

## Decisions waiting for you

- **The next training run's config.** Measured end to end in the real pipeline:

        loader.train_num_pieces: 1          (was 2)
        trainer.accumulate_grad_batches: 8  (was 4)
        model.freeze_layers: 4              (was 8)

  Same eight sequences per optimizer step, about 9.4x faster (3.2s against 30.4s),
  and 45% more trainable parameters -- layers 4-7, the lower blocks feeding the audio
  cross-attention. Epoch time goes from roughly 22 hours to under 3.

  | pieces 1 x accum 8 | per batch | per update | trainable | VRAM |
  |---|---|---|---|---|
  | freeze 8 | 0.380s | 3.0s  | 121.1M | 6259 MiB |
  | freeze 4 | 0.404s | 3.2s  | 175.7M | ~7950 MiB |
  | freeze 0 | 2.0s   | 16.8s | 230.2M | 7947 MiB, paging |
  | *current* | *7.6s* | *30.4s* | *121.1M* | *7950 MiB, paging* |

  freeze 4 held 1.79 -> 2.00 it/s rising over 25 optimizer steps with flat VRAM, so
  it is not fragmenting. freeze 0 has the most capacity and is 5x slower: past the
  8 GiB ceiling, capacity is paid for in PCIe round-trips rather than parameters.
  If a long run ever does degrade, freeze 8 is the safe fallback with 1.7 GB spare.

  I have not edited configs/audio.yaml. The numbers make the case but committing you
  to a multi-day run is your call.
- **Audio dropout for real CFG.** Current guidance uses a wrong-song negative because
  the model has no null-audio concept. Training with ~10% audio dropout would give it
  one and make guidance considerably stronger. Cheap to add, needs a retrain.
- Still open from before: PR #3 is far behind; the pattern questions (tremolo naming,
  `ladder descending split`, what unqualified "trip zig" means).
