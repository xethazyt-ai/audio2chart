"""Tier features have to separate charts that average density cannot.

The difficulty ladder showed notes per second rising only 1.8x from Easy to Expert
while mean run length quadrupled, so a grader built on average rate would put a
sustained shred chart and a sparse one in the same place.
"""

import unittest

from chart.tier import hand_movement, longest_fast_stretch, peak_density


class PeakDensityTests(unittest.TestCase):

    def test_finds_the_busiest_window_not_the_average(self):
        """A quiet chart with one hard passage must score on the passage."""
        burst = [100.0 + i * 0.05 for i in range(200)]      # 20 nps for 10s
        sparse = [i * 2.0 for i in range(50)]               # 0.5 nps for 100s
        self.assertGreater(peak_density(sparse + burst), 15.0)

    def test_empty_input_is_zero(self):
        self.assertEqual(0.0, peak_density([]))
        self.assertEqual(0.0, peak_density([1.0]))


class EnduranceTests(unittest.TestCase):

    def test_measures_the_longest_unbroken_fast_run(self):
        fast = [i * 0.05 for i in range(100)]               # 5s at 20 nps
        later = [50.0 + i * 0.05 for i in range(20)]        # 1s at 20 nps
        self.assertAlmostEqual(longest_fast_stretch(fast + later), 4.95, places=2)

    def test_a_slow_chart_has_no_fast_stretch(self):
        self.assertEqual(0.0, longest_fast_stretch([i * 0.5 for i in range(20)]))

    def test_a_run_reaching_the_end_still_counts(self):
        """The stretch is closed at the final note, not dropped."""
        self.assertGreater(longest_fast_stretch([i * 0.05 for i in range(40)]), 1.8)


class HandMovementTests(unittest.TestCase):

    def test_wide_jumps_need_a_reposition(self):
        _, wide = hand_movement([[0, 4, 0, 4]])
        self.assertEqual(1.0, wide)

    def test_adjacent_frets_are_finger_rolls(self):
        mean, wide = hand_movement([[0, 1, 2, 3, 4]])
        self.assertEqual(0.0, wide)
        self.assertEqual(1.0, mean)

    def test_no_runs_is_zero_not_an_error(self):
        self.assertEqual((0.0, 0.0), hand_movement([]))


if __name__ == "__main__":
    unittest.main()
