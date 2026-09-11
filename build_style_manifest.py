"""Emit a training manifest holding one charting style.

Robert's taxonomy: CSC charts follow the guitar only; overcharts are mostly tapping and
follow the vocals, falling back to the instrumental when the vocals stop; wii charts are
overcharts except that a guitar solo is charted CSC-style and much easier; chording,
technical and strumming are out of scope for now.

Training on all 13,720 charts mixes every one of those. A CSC chart and an overchart of
the same song look completely different -- one tracks guitar throughout, the other
switches -- and the model sees both with no signal telling them apart. That is a
plausible cause of the output inconsistency measured yesterday, where eight generated
charts ranged from 0.000 to 0.655 sustains and -0.172 to +0.383 pattern lift.

Two label sources, and they are complementary:

*song.ini genre* -- 750 charts carry an explicit style ("Tapping", "Variety",
"Technical", "Chording"). High confidence, small slice, and the only source that names
the style directly.

*folder names* -- 2,877 charts from overchart folders and the Fatty Hero and OOS packs,
which Robert says are mostly overcharts. Lower confidence, much larger.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

DEFAULT_STYLES = ("tapping",)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path(r"G:\a2c_data\audio_dataset_with_raw.json"))
    parser.add_argument("--styles", type=Path,
                        default=Path(r"G:\a2c_data\chart_styles.json"),
                        help="Chart path -> style tags, from song.ini genre")
    parser.add_argument("--folders", type=Path,
                        default=Path(r"G:\a2c_data\overchart_list.json"),
                        help="Chart paths identified by folder name instead")
    parser.add_argument("--style", nargs="+", default=list(DEFAULT_STYLES),
                        help="Styles to keep from the genre tags")
    parser.add_argument("--include-folders", action="store_true",
                        help="Also keep charts identified only by folder name. Larger "
                             "and less certain: Robert says those packs are *mostly* "
                             "overcharts, not entirely.")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    tagged = json.loads(args.styles.read_text(encoding="utf-8"))
    wanted = set(args.style)

    by_folder = set()
    if args.include_folders and args.folders.exists():
        by_folder = set(json.loads(args.folders.read_text(encoding="utf-8")))

    sources = Counter()
    kept = []
    for entry in entries:
        path = entry["chart_path"]
        tags = set(tagged.get(path, ()))
        if tags & wanted:
            sources["genre tag"] += 1
        elif path in by_folder:
            sources["folder name"] += 1
        else:
            continue
        row = dict(entry)
        row["style_source"] = "genre" if tags & wanted else "folder"
        kept.append(row)

    out = args.out or args.manifest.with_name(
        args.manifest.stem + "_" + "_".join(sorted(wanted)) + ".json")
    out.write_text(json.dumps(kept, indent=1), encoding="utf-8")

    print(f"wrote {out}  ({len(kept)} of {len(entries)} charts)")
    for name, count in sources.most_common():
        print(f"  {name:<14}{count:>6}")
    if len(kept) < 500:
        print(f"\nNOTE: {len(kept)} charts is small for a 245 M model. Fine-tune from an")
        print("      existing checkpoint rather than training from scratch.")


if __name__ == "__main__":
    main()
