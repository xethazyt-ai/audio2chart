import argparse
import os
import sys
import torch

from inference.engine import Charter
from chart.time_conversion import convert_notes_to_ticks
from chart.tokenizer import SimpleTokenizerGuitar
from chart.chart_writer import fill_expert_single
from chart.tempo import (parse_sync_track, detect_tempo, first_onset,
                         beat_aligned_tempo_events)


def _use_utf8_stdout():
    """Windows consoles default to cp1252, which cannot encode the emoji this script prints.

    The chart is written before those lines run, so the failure is purely cosmetic -- but it
    raises UnicodeEncodeError after a completely successful generation, which reads as a
    crash and invites someone to go looking for a bug in the model.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main():
    _use_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="🎵 Convert an audio file into a Guitar Hero-style chart using Charter."
    )

    # Required argument
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the input audio file (must be >= 30 seconds)."
    )

    # Optional model + sampling args
    parser.add_argument(
        "--model_name",
        type=str,
        default="3podi/charter-v1.0-40-M-best-acc",
        help="Model identifier or path. (default: 3podi/charter-v1.0-40-M-best-acc)"
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=32, help="Top-k sampling parameter.")

    # Optional metadata
    parser.add_argument("--name", type=str, default=None, help="Song title.")
    parser.add_argument("--artist", type=str, default=None, help="Artist name.")
    parser.add_argument("--album", type=str, default=None, help="Album name.")
    parser.add_argument("--genre", type=str, default=None, help="Genre.")
    parser.add_argument("--charter", type=str, default=None, help="Charter name.")
    parser.add_argument("--bpm", type=int, default=200, help="Chart bpm.")
    parser.add_argument("--resolution", type=int, default=480, help="Chart resolution.")

    # Beat grid. Without one of these the chart is audio-aligned but its bar lines are meaningless.
    parser.add_argument(
        "--sync-from",
        type=str,
        default=None,
        help="Copy Resolution and [SyncTrack] from an existing .chart file (exact)."
    )
    parser.add_argument(
        "--style",
        default=None,
        help="Constrain generation to a charting style (e.g. wii, chording, one_hand). "
             "Enforced by masking token ids at sampling time, so it cannot be violated.",
    )
    parser.add_argument(
        "--detect-tempo",
        action="store_true",
        help="Estimate the tempo from the audio instead of trusting --bpm."
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Keep the model's own note timing instead of moving the first note onto "
             "the audio's first note. Only meaningful with --detect-tempo."
    )
    parser.add_argument(
        "--snap",
        type=int,
        default=0,
        help="Snap notes to the nearest 1/SNAP note (e.g. 32). Only sensible with a real beat grid."
    )

    # Output path (optional)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination folder for output. Defaults to ./<song_name>/notes.chart"
    )

    args = parser.parse_args()

    # Resolve the beat grid up front so a bad --sync-from fails before generation
    bpm_events = None
    ts_events = None
    anchor_time = None
    resolution = args.resolution

    if args.sync_from:
        bpm_events, ts_events, resolution = parse_sync_track(args.sync_from)
        print(f"SyncTrack from {args.sync_from}: "
              f"{len(bpm_events)} tempo event(s), resolution {resolution}")
    elif args.detect_tempo:
        import librosa
        y, sr = librosa.load(args.audio_path, sr=22050, mono=True)
        bpm, phase, score = detect_tempo(y, sr)
        onset = first_onset(y, sr)
        bpm_events, anchor_tick, anchor_time = beat_aligned_tempo_events(
            bpm, phase, onset, resolution)
        lead_bpm = bpm_events[0][1] / 1000.0
        print(f"Detected tempo: {bpm:.3f} BPM (phase {phase:.3f}s, score {score:.3f})")
        print(f"First audio note at {onset:.3f}s; beat line at {anchor_time:.3f}s "
              f"(tick {anchor_tick}), lead-in {lead_bpm:.3f} BPM")

    # Load model + tokenizer
    print(f"Loading model: {args.model_name}")
    model = Charter.from_pretrained(args.model_name)
    # A legacy checkpoint emits 32 chord tokens; an expressive one emits 1280.
    _vs = getattr(model.config, "vocab_size", 35)
    tokenizer = SimpleTokenizerGuitar(expressive=_vs > 64)
    print(f"Vocabulary: {'expressive' if tokenizer.expressive else 'legacy'}"
          f" ({tokenizer.vocab_size} tokens, model reports {_vs})")
    ms_resolution = model.config.grid_ms

    # Generate tokens
    print(f"Generating chart for: {args.audio_path}")
    allowed_ids = None
    if args.style:
        from chart.style import allowed_tokens, describe, resolve
        constraint = resolve(args.style)
        allowed_ids = allowed_tokens(tokenizer, constraint)
        print(f"Style {args.style!r}: {describe(tokenizer, constraint)}")

    seqs = model.generate(
        args.audio_path,
        temperature=args.temperature,
        top_k=args.top_k,
        allowed_ids=allowed_ids,
    )
    seqs = torch.cat(seqs).flatten().cpu().tolist()

    # Convert to ticked notes
    time_list = [i * ms_resolution / 1000 for i in range(len(seqs))]

    # Robert's third step: slide the chart so its first note lands on the beat line
    # that the tempo map put on the audio's first note. The notes are placed by
    # seconds and the map decides ticks, so this is the whole of "syncing" -- there
    # is nothing to adjust afterwards.
    if anchor_time is not None and not args.no_sync:
        first = next((i for i, t in enumerate(seqs) if tokenizer.is_note_token(t)), None)
        if first is None:
            print("Nothing generated to sync.")
        else:
            shift = anchor_time - time_list[first]
            time_list = [t + shift for t in time_list]
            print(f"First generated note {time_list[first] - shift:.3f}s "
                  f"-> {anchor_time:.3f}s (shifted {shift:+.3f}s)")
    ticked_notes = convert_notes_to_ticks(seqs, time_list, fixed_bpm=args.bpm,
                                          resolution=resolution, bpm_events=bpm_events,
                                          snap=args.snap, pad_token_id=tokenizer.pad_id,
                                          tokenizer=tokenizer)
    decoded_full = tokenizer.decode(ticked_notes, resolution=resolution)

    # Prepare metadata
    model_tag = args.model_name.split("/")[-1]
    default_charter = args.charter or f"audio2chart/{model_tag}-{args.temperature}-{args.top_k}"

    song_name = args.name or os.path.splitext(os.path.basename(args.audio_path))[0]
    metadata = {
        "name": song_name,
        "artist": args.artist or "audio2chart",
        "album": args.album or "audio2chart",
        "genre": args.genre or "audio2chart",
        "charter": default_charter,
        "bpm": args.bpm,
        "resolution": resolution,
        # Clone Hero loads the audio named here, from the folder holding notes.chart.
        "musicstream": os.path.basename(args.audio_path),
    }

    # Fill and save chart
    filled_text = fill_expert_single(decoded_full, metadata=metadata,
                                     bpm_events=bpm_events, ts_events=ts_events)

    # Determine output folder and file path
    if args.output:
        output_folder = args.output
    else:
        output_folder = os.path.join(os.getcwd(), song_name)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "notes.chart")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(filled_text)

    print(f"✅ Chart saved to: {output_path}")
    print(f"   Copy {os.path.basename(args.audio_path)} next to it for Clone Hero to play the song.")


if __name__ == "__main__":
    main()
