import tempfile
import unittest
from pathlib import Path

from chart.bars import (
    bar_boundaries, bar_length_ticks, bars_for_chart, group_by_bar, parse_time_signatures,
)

CHART = """[Song]
{
  Resolution = 192
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
  1536 = TS 3
}
[ExpertSingle]
{
  0 = N 0 0
}
"""


class BarTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "notes.chart"
        self.path.write_text(CHART, encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def test_parses_time_signature_changes(self):
        self.assertEqual([(0, 4, 4), (1536, 3, 4)], parse_time_signatures(self.path))

    def test_missing_denominator_defaults_to_four(self):
        # ".. = TS 4" with no second value means 4/4, not 4/1.
        self.assertEqual(4, parse_time_signatures(self.path)[0][2])

    def test_explicit_denominator_exponent_is_a_power_of_two(self):
        p = Path(self.dir.name) / "eighths.chart"
        p.write_text("[SyncTrack]\n{\n  0 = TS 6 3\n}\n", encoding="utf-8")
        self.assertEqual((0, 6, 8), parse_time_signatures(p)[0])

    def test_defaults_to_four_four_when_absent(self):
        p = Path(self.dir.name) / "none.chart"
        p.write_text("[SyncTrack]\n{\n  0 = B 120000\n}\n", encoding="utf-8")
        self.assertEqual([(0, 4, 4)], parse_time_signatures(p))

    def test_bar_length(self):
        self.assertEqual(768, bar_length_ticks(4, 4, 192))     # 4/4 at 192 ppq
        self.assertEqual(576, bar_length_ticks(3, 4, 192))     # 3/4
        self.assertEqual(576, bar_length_ticks(6, 8, 192))     # 6/8

    def test_boundaries_follow_a_time_signature_change(self):
        b = bar_boundaries(self.path, resolution=192, last_tick=3000)
        self.assertEqual([0, 768, 1536, 2112, 2688], b[:5])
        # 4/4 bars of 768 until 1536, then 3/4 bars of 576.

    def test_grouping_puts_notes_in_the_right_bar(self):
        encoded = [(0, 1, 0, {}), (700, 2, 0, {}), (800, 3, 0, {}), (1600, 4, 0, {})]
        bars = group_by_bar(encoded, [0, 768, 1536, 2304])
        self.assertEqual([0, 700], [e[0] for e in bars[0]])
        self.assertEqual([800], [e[0] for e in bars[1]])
        self.assertEqual([1600], [e[0] for e in bars[2]])   # 1536 <= 1600 < 2304

    def test_empty_bars_are_preserved_so_index_is_bar_number(self):
        bars = group_by_bar([(0, 1, 0, {}), (2400, 2, 0, {})], [0, 768, 1536, 2304])
        self.assertEqual(4, len(bars))
        self.assertEqual([], bars[1])
        self.assertEqual([], bars[2])

    def test_bars_for_chart_end_to_end(self):
        encoded = [(0, 1, 0, {}), (900, 2, 0, {})]
        bars = bars_for_chart(self.path, encoded, resolution=192)
        self.assertEqual([0], [e[0] for e in bars[0]])
        self.assertEqual([900], [e[0] for e in bars[1]])

    def test_no_notes_gives_no_bars(self):
        self.assertEqual([], bars_for_chart(self.path, [], resolution=192))


if __name__ == "__main__":
    unittest.main()
