import json
import tempfile
import unittest
from pathlib import Path

from dataloader.utils_dataloader import split_json_entries_by_audio_raw


class GroupedSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_group_safe(self):
        entries = [
            {"raw_path": f"song-{group}.raw", "difficulty": difficulty}
            for group in range(10)
            for difficulty in ("ExpertSingle", "HardSingle")
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(entries), encoding="utf-8")
            first = split_json_entries_by_audio_raw(
                str(source), str(root / "train.json"), str(root / "val.json"),
                val_ratio=0.2, random_seed=7,
            )
            second = split_json_entries_by_audio_raw(
                str(source), str(root / "train2.json"), str(root / "val2.json"),
                val_ratio=0.2, random_seed=7,
            )
            self.assertEqual(first, second)
            self.assertTrue(first[0].isdisjoint(first[1]))
            self.assertEqual(len(first[1]), 2)

    def test_split_rejects_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                split_json_entries_by_audio_raw(
                    str(source), str(root / "train.json"), str(root / "val.json")
                )
