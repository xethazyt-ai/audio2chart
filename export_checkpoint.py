"""Export an Encodec-conditioned Lightning checkpoint for the inference engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from inference.engine import TransformerConfig
from inference.model_inference import TransformerDecoderAudioConditioned


def extract_transformer_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract the decoder state dict and remove Lightning's module prefix."""
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a Lightning 'state_dict'")

    prefix = "transformer."
    transformer_state = {
        name[len(prefix):]: value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }
    if not transformer_state:
        raise ValueError("Checkpoint contains no 'transformer.*' parameters")
    return transformer_state


def export_checkpoint(checkpoint_path: Path, config_path: Path, output_dir: Path) -> None:
    """Validate and export weights in the format consumed by Charter."""
    with config_path.open(encoding="utf-8") as stream:
        config_data = json.load(stream)
    config = TransformerConfig(**config_data)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    transformer_state = extract_transformer_state_dict(checkpoint)

    inference_model = TransformerDecoderAudioConditioned(**config_data)
    try:
        inference_model.load_state_dict(transformer_state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint is not compatible with the current Encodec inference architecture"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(transformer_state, output_dir / "pytorch_model.bin")
    with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config.__dict__, stream, indent=2)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Lightning checkpoint path")
    parser.add_argument("config", type=Path, help="Inference config.json")
    parser.add_argument("output", type=Path, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_checkpoint(arguments.checkpoint, arguments.config, arguments.output)
