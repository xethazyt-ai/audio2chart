import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    "inference/engine.py": "6c1b61f33a9300db5545997524125a750c4afdef",
    "inference/model_inference.py": "6cd2385275c06e325dd2e8d49a18808ea4d59edc",
    "generate.py": "ddf2d9b1b24aca592e38d8157303531b0f60abd2",
    "notebooks/audio2chart_charting.ipynb": "784fe5ffbbc2aa8efdbf20104ae09591e5ab6bf0",
    "modules/transformer2.py": "59239de999e86eebad22a2565c841e270f24ef14",
}


def class_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


class ArchitectureSurfaceTests(unittest.TestCase):
    def test_only_supported_lightning_modules_remain(self):
        self.assertEqual(
            class_names(ROOT / "modules" / "trainer.py"),
            {"NotesTransformer", "WaveformTransformerDiscrete"},
        )

    def test_only_supported_training_models_remain(self):
        self.assertEqual(
            class_names(ROOT / "modules" / "models.py"),
            {"Encodec", "ResnetBlock2d", "SEANetEncoder2d",
             "TransformerEncoder", "TransformerDecoderOnly"},
        )

    def test_inference_files_are_still_tracked_and_present(self):
        self.assertTrue((ROOT / "inference" / "engine.py").is_file())
        self.assertTrue((ROOT / "inference" / "model_inference.py").is_file())

    def test_frozen_inference_surface_is_byte_identical(self):
        for relative_path, expected in FROZEN_HASHES.items():
            digest = hashlib.sha1((ROOT / relative_path).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relative_path)

    def test_training_configs_use_separate_transformer(self):
        for name in ("audio_codec.yaml", "audio_discrete.yaml"):
            contents = (ROOT / "configs" / "model" / name).read_text(encoding="utf-8")
            self.assertIn("modules.training_transformer.TransformerDecoderAudioConditioned", contents)


if __name__ == "__main__":
    unittest.main()
