"""Score a trained checkpoint the way the last two days say it has to be scored.

Three things, because no one of them answers "is this good":

*Does it use the audio?* evaluate_conditioning swaps each batch's audio for a
neighbour's and reports what the swap costs. Note accuracy cannot answer this -- ~86%
of slots are padding and an all-pad baseline already scores 85.6%.

*What do its charts look like?* evaluate_chart against human baselines, over SEVERAL
charts. One chart characterises nothing: across eight from this model, sustains ran
0.000 to 0.655 and pattern lift -0.172 to +0.383, and a day of conclusions drawn from a
single chart had to be retracted.

*Did it adapt to the new grid?* The model learned note spacing in position units at
20 ms. At 10 ms every rhythm it knew is wrong by exactly 2x. That should wash out in
fine-tuning, but it is a falsifiable claim: if it did not, note density comes out around
double and the inter-onset histogram clusters at half the right intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CLIP_SECONDS = 180


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--baseline", choices=("all", "tapping"), default="tapping",
                        help="Human population to score charts against. Defaults to "
                             "tapping, because that is what these runs train on: an "
                             "all-styles median marks a correct tapping chart as broken "
                             "on every metric (taps 0.96 against 0.19, density 21 NPS "
                             "against 7.8).")
    parser.add_argument("--label", default=None)
    parser.add_argument("--charts", type=int, default=6)
    parser.add_argument("--export-to", type=Path, default=None)
    parser.add_argument("--config", type=Path,
                        default=Path(r"G:\a2c_data\export\config.json"))
    return parser.parse_args()


def export(checkpoint: Path, config: Path, out: Path):
    """Export with the grid and window the training config actually used."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="audio")

    data = json.loads(config.read_text(encoding="utf-8"))
    data["grid_ms"] = int(cfg.data.grid_ms)
    data["window_seconds"] = float(cfg.data.window_seconds)
    patched = out.parent / "config_patched.json"
    patched.parent.mkdir(parents=True, exist_ok=True)
    patched.write_text(json.dumps(data, indent=2), encoding="utf-8")

    subprocess.run([sys.executable, str(ROOT / "export_checkpoint.py"),
                    str(checkpoint), str(patched), str(out), "--trust"], check=True)
    return out


def main():
    args = parse_args()
    label = args.label or args.checkpoint.parent.parent.name
    out = args.export_to or Path(tempfile.mkdtemp()) / "export"
    print(f"=== {label} ===")
    export(args.checkpoint, args.config, out)

    print("\n-- does it use the audio? --")
    subprocess.run([sys.executable, str(ROOT / "evaluate_conditioning.py"),
                    "--checkpoint", str(args.checkpoint), "--trust", "--batches", "20"])

    import torch
    from inference.engine import Charter
    from chart.chart_writer import fill_expert_single
    from chart.time_conversion import choose_snap, convert_notes_to_ticks
    from chart.tokenizer import SimpleTokenizerGuitar

    entries = [e for e in json.loads(
        Path(r"G:\a2c_data\splits_tapping\val.json").read_text(encoding="utf-8"))]
    import random
    random.Random(0).shuffle(entries)

    model = Charter.from_pretrained(str(out))
    tokenizer = SimpleTokenizerGuitar(expressive=True)
    grid = model.config.grid_ms / 1000.0
    work = Path(tempfile.mkdtemp())
    written = []
    for entry in entries:
        if len(written) >= args.charts:
            break
        clip = work / f"clip{len(written)}.wav"
        if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i",
                           entry["audio_path"], "-t", str(CLIP_SECONDS),
                           "-ar", "24000", "-ac", "1", str(clip)]).returncode:
            continue
        torch.manual_seed(len(written))
        sequence = torch.cat(model.generate(str(clip), temperature=0.5, top_k=32,
                                            max_parallel_chunks=4)).flatten().cpu().tolist()
        torch.cuda.empty_cache()
        times = [i * grid for i in range(len(sequence))]
        ticks = convert_notes_to_ticks(sequence, times, resolution=480,
                                       bpm_events=[(0, 120000)], snap=0,
                                       pad_token_id=tokenizer.pad_id, tokenizer=tokenizer)
        snap = choose_snap([t[0] for t in ticks], 480)
        ticks = convert_notes_to_ticks(sequence, times, resolution=480,
                                       bpm_events=[(0, 120000)], snap=snap,
                                       pad_token_id=tokenizer.pad_id, tokenizer=tokenizer)
        text = fill_expert_single(
            tokenizer.decode(ticks, resolution=480),
            metadata={"name": "s", "artist": "s", "album": "s", "genre": "s",
                      "charter": "s", "bpm": 120, "resolution": 480,
                      "musicstream": "song.ogg"},
            bpm_events=[(0, 120000)], ts_events=None)
        path = work / f"chart{len(written)}" / "notes.chart"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    print(f"\n-- what do its charts look like? ({len(written)} charts) --")
    for path in written:
        subprocess.run([sys.executable, str(ROOT / "evaluate_chart.py"), str(path),
                        "--baseline", args.baseline])


if __name__ == "__main__":
    main()
