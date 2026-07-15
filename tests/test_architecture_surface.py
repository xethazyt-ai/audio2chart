import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
