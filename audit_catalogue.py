"""Check every catalogue entry against the corpus and against itself.

The catalogue is AI-assisted and Robert flagged it as not ground truth: entries may be
wrong, mislabelled, nonexistent, or real but described incorrectly. Several have
already been caught -- the H pattern was written as single notes when it is tap chords,
two castles never occur, and "triangle slide 8-note" and "sweep 8-note" are the same
notes under two names.

This reports four kinds of problem:

*Duplicates* -- two names for one signature. Transpositions sharing a signature are
correct by design; the same literal sequence twice is not.

*Absent* -- never found in the corpus. For a single-note figure that is evidence the
entry is invented. For a chord figure it means nothing, since the matcher only sees
single notes.

*Rare* -- found, but in so few charts that the entry may be a variant someone inferred
rather than a pattern anyone charts.

*Misnamed* -- the twenty "anchor X-Y-X" entries. Robert: anchoring is holding the
lowest fret through a passage, a technique, not a three-note shape. Those entries
describe real figures under a word that means something else.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.catalogue import CATALOGUE, body_signature, parse, signature
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar

RARE_CHARTS = 5
"""Below this many charts, an entry is more likely inferred than observed."""


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path(r"G:\a2c_data\audio_dataset_with_raw.json"))
    parser.add_argument("--charts", type=int, default=300)
    return parser.parse_args()


def occurrences(charts, tokenizer):
    """How many charts contain each catalogue body, matched inside beats."""
    bodies = {}
    for name, text in CATALOGUE.items():
        sig = body_signature(text)
        if len(sig) >= 2:
            bodies.setdefault(sig, []).append(name)

    found = collections.Counter()
    scanned = 0
    for path in charts:
        try:
            processor = ChartProcessor(["Expert"], ["Single"])
            processor.read_chart(path, target_sections="ExpertSingle")
            notes = processor.notes.get("ExpertSingle")
            if not notes:
                continue
            resolution = int(processor.song_metadata["Resolution"])
            encoded = tokenizer.encode(notes, resolution=resolution)
        except Exception:
            continue
        scanned += 1
        frets = []
        for event in encoded:
            lanes = tokenizer.reverse_chord.get(
                event[1] // (tokenizer.N_FLAG * tokenizer.N_SUSTAIN), ())
            frets.append(lanes[0] if len(lanes) == 1 and lanes[0] <= 4 else None)
        deltas = []
        for a, b in zip(frets, frets[1:]):
            deltas.append(b - a if a is not None and b is not None else None)
        here = set()
        for sig in bodies:
            width = len(sig)
            for start in range(len(deltas) - width + 1):
                chunk = deltas[start:start + width]
                if None not in chunk and tuple(chunk) == sig:
                    here.add(sig)
                    break
        for sig in here:
            found[sig] += 1
    return found, bodies, scanned


def main():
    args = parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = [e for e in entries if e["chart_path"].lower().endswith(".chart")]
    random.Random(7).shuffle(entries)
    charts = [e["chart_path"] for e in entries[:args.charts]]

    tokenizer = SimpleTokenizerGuitar(expressive=True)
    found, bodies, scanned = occurrences(charts, tokenizer)

    print(f"{len(CATALOGUE)} entries, {len(bodies)} distinct bodies, "
          f"{scanned} charts scanned\n")

    print("DUPLICATE NAMES (same body under more than one name)")
    exact = collections.defaultdict(list)
    for name, text in CATALOGUE.items():
        exact[tuple(parse(text))].append(name)
    shown = 0
    for names in exact.values():
        if len(names) > 1:
            print(f"  identical note sequence: {', '.join(names)}")
            shown += 1
    if not shown:
        print("  none with identical notes")
    for sig, names in bodies.items():
        if len(names) > 1:
            print(f"  same shape: {', '.join(names)}")

    print("\nABSENT (never matched)")
    absent = [n for sig, names in bodies.items() if found[sig] == 0 for n in names]
    for name in sorted(absent):
        print(f"  {name}")
    if not absent:
        print("  none")

    print(f"\nRARE (found in fewer than {RARE_CHARTS} of {scanned} charts)")
    rare = [(found[sig], n) for sig, names in bodies.items()
            for n in names if 0 < found[sig] < RARE_CHARTS]
    for count, name in sorted(rare):
        print(f"  {count:>3} charts  {name}")
    if not rare:
        print("  none")

    print("\nMISNAMED")
    anchors = [n for n in CATALOGUE if n.startswith("anchor ")]
    print(f"  {len(anchors)} entries named 'anchor', which is a technique (holding the")
    print("  lowest fret through a passage), not a three-note shape:")
    print("    " + ", ".join(anchors[:6]) + (" ..." if len(anchors) > 6 else ""))


if __name__ == "__main__":
    main()
