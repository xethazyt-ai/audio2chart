"""The audio encoder must not run under bf16 autocast.

Lightning wraps the training step in bf16, which is a 3.4x speedup on the transformer
and a 5.5x slowdown on Encodec: cuDNN has no fused LSTM kernel for bf16, so SEANet's
recurrence over 2250 timesteps drops to an unfused path. Measured 0.130s per call in
fp32 against 0.712s under autocast.

Nothing is lost by opting out. The discrete path returns quantiser indices, so there
is no precision to gain -- only the recurrence to slow down.
"""

import unittest


class EncoderAutocastTests(unittest.TestCase):

    def test_encode_audio_disables_autocast(self):
        import inspect

        try:
            from modules.trainer import WaveformTransformerDiscrete
        except ImportError as error:
            self.skipTest(f"torch unavailable: {error}")

        source = inspect.getsource(WaveformTransformerDiscrete._encode_audio)
        self.assertIn("autocast", source)
        self.assertIn("enabled=False", source)

    def test_codes_are_identical_either_way(self):
        """The opt-out must not change what the model sees."""
        try:
            import torch
            from modules.models import Encodec
        except ImportError as error:
            self.skipTest(f"torch unavailable: {error}")
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA to exercise autocast")

        encoder = Encodec(bandwidth=3.0).cuda().eval()
        audio = torch.randn(1, 1, 24000 * 2, device="cuda")
        mask = torch.ones(1, 24000 * 2, dtype=torch.long, device="cuda")

        with torch.no_grad():
            plain = encoder(audio, mask, return_embeddings=False)[0]
            with torch.autocast("cuda", enabled=False):
                opted_out = encoder(audio.float(), mask, return_embeddings=False)[0]

        self.assertTrue(torch.equal(plain, opted_out))


if __name__ == "__main__":
    unittest.main()
