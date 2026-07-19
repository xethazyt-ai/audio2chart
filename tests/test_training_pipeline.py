import unittest
from unittest import mock

run_utils = mock.Mock()
try:
    import torch
    from omegaconf import OmegaConf
    from dataloader.audio_loader import AudioChartCollator, _collate_batch_impl_discrete
    from baseline import split_charts
    from export_checkpoint import extract_transformer_state_dict
    from inference.model_inference import (
        TransformerDecoderAudioConditioned as InferenceTransformer,
    )
    from modules import run_utils
    from modules.training_transformer import (
        TransformerDecoderAudioConditioned as TrainingTransformer,
    )
    TRAINING_IMPORT_ERROR = None
except (ImportError, OSError) as error:
    TRAINING_IMPORT_ERROR = error


@unittest.skipIf(TRAINING_IMPORT_ERROR is not None,
                 f"training environment unavailable: {TRAINING_IMPORT_ERROR}")
class CollatorTests(unittest.TestCase):
    def test_discrete_collation_keeps_batch_dimension_one(self):
        sample = {"audio": torch.zeros(1, 8), "note_values": [2, 3], "cond_diff": [0]}
        batch = _collate_batch_impl_discrete([[sample]], 10, 11, True)
        self.assertEqual(tuple(batch["input_values"].shape), (1, 8))
        self.assertEqual(tuple(batch["note_values"].shape), (1, 4))
        self.assertEqual(tuple(batch["cond_diff"].shape), (1, 1))

    @mock.patch("dataloader.audio_loader.AutoProcessor.from_pretrained")
    def test_processor_checkpoint_and_sample_rate(self, from_pretrained):
        from_pretrained.return_value.sampling_rate = 24000
        AudioChartCollator(1, 2, use_processor=True,
                           processor_checkpoint="local/encodec", sample_rate=24000)
        from_pretrained.assert_called_once_with("local/encodec")
        with self.assertRaises(ValueError):
            AudioChartCollator(1, 2, use_processor=True,
                               processor_checkpoint="local/encodec", sample_rate=16000)


@unittest.skipIf(TRAINING_IMPORT_ERROR is not None,
                 f"training environment unavailable: {TRAINING_IMPORT_ERROR}")
class EntrypointTests(unittest.TestCase):
    def _config(self, save_run=False, tracking=True):
        return OmegaConf.create({
            "trainer": {"save_run": save_run, "early_stopping_patience": 5},
            "tracking": {
                "enabled": tracking, "project": "audio2chart", "tags": ["test"]
            },
        })

    def test_checkpoint_callback_follows_save_run(self):
        disabled = run_utils.build_callbacks(self._config(False), "val/acc_epoch")
        enabled = run_utils.build_callbacks(self._config(True), "val/acc_epoch")
        checkpoint_type = run_utils.L.pytorch.callbacks.ModelCheckpoint
        self.assertFalse(any(isinstance(item, checkpoint_type) for item in disabled))
        self.assertTrue(any(isinstance(item, checkpoint_type) for item in enabled))

    @mock.patch.object(run_utils.wandb, "finish")
    @mock.patch.object(run_utils.wandb, "init")
    @mock.patch.object(run_utils, "WandbLogger", return_value="logger")
    def test_wandb_cleanup_after_failure(self, logger, init, finish):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with run_utils.experiment_logger(self._config(tracking=True), "test"):
                raise RuntimeError("boom")
        init.assert_called_once()
        finish.assert_called_once()

    @mock.patch.object(run_utils.wandb, "init")
    def test_disabled_tracking_does_not_initialize_wandb(self, init):
        with run_utils.experiment_logger(self._config(tracking=False), "test") as logger:
            self.assertIsInstance(logger, run_utils.CSVLogger)
        init.assert_not_called()

    def test_baseline_split_uses_validation_ratio(self):
        train, val = split_charts([f"chart-{index}" for index in range(10)], 0.2, 7)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 2)
        self.assertTrue(set(train).isdisjoint(val))


@unittest.skipIf(TRAINING_IMPORT_ERROR is not None,
                 f"training environment unavailable: {TRAINING_IMPORT_ERROR}")
class ExportTests(unittest.TestCase):
    def test_extracts_only_transformer_parameters(self):
        tensor = torch.ones(1)
        result = extract_transformer_state_dict({
            "state_dict": {"transformer.weight": tensor, "audio_encoder.weight": tensor}
        })
        self.assertEqual(set(result), {"weight"})

    def test_rejects_checkpoint_without_transformer(self):
        with self.assertRaisesRegex(ValueError, "no 'transformer"):
            extract_transformer_state_dict({"state_dict": {}})

    def test_training_weights_strict_load_and_match_inference_logits(self):
        arguments = dict(
            vocab_size=35, pad_token_id=-1, eos_token_id=33,
            d_model=32, n_heads=4, num_kv_heads=2, n_layers=2,
            dropout=0.0, audio_drop=0.0, conditional=False,
            use_flash=False, codebook_size=64,
        )
        torch.manual_seed(1)
        training = TrainingTransformer(**arguments).eval()
        inference = InferenceTransformer(**arguments).eval()
        inference.load_state_dict(training.state_dict(), strict=True)
        tokens = torch.randint(0, 35, (2, 5))
        codes = torch.randint(0, 64, (2, 4, 7))
        with torch.no_grad():
            training_logits = training(tokens, codes)
            audio = sum(
                inference.codes_embedding[index](codes[:, index])
                for index in range(4)
            )
            audio = inference.norm_audio(audio)
            inference_logits = inference(tokens, audio)
        torch.testing.assert_close(training_logits, inference_logits)


if __name__ == "__main__":
    unittest.main()
