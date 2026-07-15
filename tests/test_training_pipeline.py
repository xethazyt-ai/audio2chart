import unittest
from unittest import mock

training_main = mock.Mock()
try:
    import torch
    from dataloader.audio_loader import AudioChartCollator, _collate_batch_impl_discrete
    import main as training_main
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
    def _config(self, save_run):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "save_run": save_run,
            "early_stopping_patience": 5,
            "seed": 42,
            "tags": ["test"],
        })

    def test_checkpoint_callback_follows_save_run(self):
        disabled = training_main.build_callbacks(self._config(False))
        enabled = training_main.build_callbacks(self._config(True))
        checkpoint_type = training_main.L.pytorch.callbacks.ModelCheckpoint
        self.assertFalse(any(isinstance(item, checkpoint_type) for item in disabled))
        self.assertTrue(any(isinstance(item, checkpoint_type) for item in enabled))

    @mock.patch.object(training_main, "set_seed_everything")
    @mock.patch.object(training_main, "load_data_splits", side_effect=RuntimeError("boom"))
    @mock.patch.object(training_main.wandb, "finish")
    @mock.patch.object(training_main.wandb, "init")
    @mock.patch.object(training_main, "build_run_name", return_value="test")
    @mock.patch.object(training_main.OmegaConf, "to_container", return_value={})
    def test_wandb_cleanup_after_failure(self, to_container, build_name, init,
                                         finish, load_data, set_seed):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            training_main.run(self._config(False))
        init.assert_called_once()
        finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
