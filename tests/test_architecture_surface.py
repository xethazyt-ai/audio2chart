import ast
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_HASHES = {
    "inference/engine.py": "304ee101a6ccc508519fcc05f668a792723ef699",
    "inference/model_inference.py": "cd526cfbb27acd9921fb51a3c240f489bdad80c6",
    "inference/layers.py": "e44bfddff391fe20f482c7152eef11abd9dc88bd",
    "generate.py": "4e193850d855224d8e3f5157abd9ac2ede18936d",
    "notebooks/audio2chart_charting.ipynb": "784fe5ffbbc2aa8efdbf20104ae09591e5ab6bf0",
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
        self.assertTrue((ROOT / "inference" / "layers.py").is_file())

    def test_inference_model_is_self_contained(self):
        contents = (ROOT / "inference" / "model_inference.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from inference.layers import", contents)
        self.assertNotIn("from modules.", contents)
        self.assertFalse((ROOT / "modules" / "transformer2.py").exists())

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
