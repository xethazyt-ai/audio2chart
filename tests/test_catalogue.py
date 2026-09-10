import unittest

from chart.catalogue import CATALOGUE, collisions, parse, signature, signatures


class CatalogueTest(unittest.TestCase):
    def test_parse_handles_both_separators_and_bar_markers(self):
        self.assertEqual((0, 1, 2), parse("G R Y"))
        self.assertEqual((0, 1, 2), parse("G-R-Y"))
        self.assertEqual((2, 1, 0, 1), parse("Y R | G R"))

    def test_signature_is_transposition_invariant(self):
        self.assertEqual(signature(parse("G R Y")), signature(parse("R Y B")))
        self.assertEqual(signature(parse("Y R G")), signature(parse("B Y R")))

    def test_signature_distinguishes_interval_structure(self):
        # Contiguous and gapped trips are different shapes, not transpositions.
        self.assertNotEqual(signature(parse("G R Y")), signature(parse("G Y B")))

    def test_every_entry_parses_to_at_least_two_notes(self):
        for name, text in CATALOGUE.items():
            self.assertGreaterEqual(len(parse(text)), 2, name)

    def test_every_entry_has_a_signature(self):
        for name, sig in signatures().items():
            self.assertTrue(sig, name)

    def test_known_duplicate_is_detected(self):
        # triangle slide 8-note and sweep 8-note are the same literal sequence.
        self.assertEqual(parse(CATALOGUE["triangle slide 8-note"]),
                         parse(CATALOGUE["sweep 8-note"]))

    def test_transposed_families_collide_as_expected(self):
        found = collisions()
        trip_zigs = signature(parse(CATALOGUE["trip zig G-R-Y"]))
        self.assertIn(trip_zigs, found)
        self.assertIn("trip zig R-Y-B", found[trip_zigs])

    def test_roberts_original_trip_zig_example_is_the_split_variant(self):
        # He first described O B R B O B R B O B R as a "trip zig"; in this catalogue that
        # is the split R-B-O form, not the plain G-R-Y one.
        original = signature(parse("O B R B O B R B O B R"))
        self.assertEqual(original, signature(parse(CATALOGUE["trip zig split R-B-O"])))
        self.assertNotEqual(original, signature(parse(CATALOGUE["trip zig G-R-Y"])))


if __name__ == "__main__":
    unittest.main()
