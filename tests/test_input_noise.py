"""Input corruption must damage the history and nothing else.

Teacher forcing only ever shows a correct history, and free running the model then
drifts -- the generated tap rate climbs 0.224 -> 0.723 across one 30s chunk. Corrupting
a fraction of inputs approximates scheduled sampling without a second forward pass, but
only if it corrupts the right things: targets untouched, padding untouched, and never
an eos or pad injected as noise.
"""

import unittest

import torch


class _Model:
    """The corruption logic, lifted out so the test needs no Encodec download."""

    from modules.trainer import WaveformTransformerDiscrete as _Real
    _corrupt_inputs = _Real._corrupt_inputs

    def __init__(self, noise, pad_token_id=34, eos_token_id=33):
        self.input_noise = noise
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.n_note_tokens = min(pad_token_id, eos_token_id)
        self.training = True


class InputNoiseTests(unittest.TestCase):

    def test_disabled_by_default_returns_the_input_unchanged(self):
        model = _Model(0.0)
        tokens = torch.randint(0, 30, (4, 50))
        self.assertTrue(torch.equal(model._corrupt_inputs(tokens), tokens))

    def test_padding_is_never_corrupted(self):
        """Corrupting pad would teach the model to play past the end of a chart."""
        model = _Model(1.0)
        tokens = torch.full((2, 20), model.pad_token_id)
        self.assertTrue(torch.equal(model._corrupt_inputs(tokens), tokens))

    def test_noise_never_injects_a_control_token(self):
        """An injected eos would teach the model to stop early."""
        torch.manual_seed(0)
        model = _Model(1.0)
        tokens = torch.randint(0, 30, (8, 200))
        out = model._corrupt_inputs(tokens)
        self.assertLess(int(out.max()), model.n_note_tokens)

    def test_corruption_happens_at_roughly_the_requested_rate(self):
        torch.manual_seed(0)
        model = _Model(0.1)
        tokens = torch.randint(0, 30, (16, 500))
        changed = (model._corrupt_inputs(tokens) != tokens).float().mean().item()
        # Some draws land on the same token, so the observed rate is a little under.
        self.assertGreater(changed, 0.05)
        self.assertLess(changed, 0.12)

    def test_eval_mode_is_never_corrupted(self):
        """Validation loss has to stay comparable across runs."""
        model = _Model(1.0)
        model.training = False
        tokens = torch.randint(0, 30, (4, 50))
        self.assertTrue(torch.equal(model._corrupt_inputs(tokens), tokens))


if __name__ == "__main__":
    unittest.main()
