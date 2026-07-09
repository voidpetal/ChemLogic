"""
Checkpoint module for saving and loading ChemLogic model checkpoints.

Uses safetensors for efficient, secure weight storage and JSON for metadata.
Two-file format:
  - {name}.safetensors - Model weights as flat tensors
  - {name}.json - Metadata, architecture config, training state, weight names
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

# Checkpoint format version
CHECKPOINT_VERSION = "1.0"


class Checkpoint:
    @staticmethod
    def save(
        evaluator,
        filepath: str | Path,
        *,
        architecture: dict[str, Any] | None = None,
        training_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """Save model checkpoint to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state_dict = evaluator.state_dict()
        weights = state_dict["weights"]
        weight_names = state_dict["weight_names"]

        tensors = Checkpoint._weights_to_tensors(weights)

        json_data = {
            "version": CHECKPOINT_VERSION,
            "created_at": datetime.now().isoformat(),
            "weight_names": {str(k): v for k, v in weight_names.items()},
        }

        if architecture is not None:
            json_data["architecture"] = architecture
        if training_state is not None:
            json_data["training_state"] = training_state
        if metadata is not None:
            json_data["metadata"] = metadata

        safetensors_path = filepath.with_suffix(".safetensors")
        json_path = filepath.with_suffix(".json")

        save_file(tensors, safetensors_path)
        try:
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            safetensors_path.unlink(missing_ok=True)
            raise

        return safetensors_path, json_path

    @staticmethod
    def load(filepath: str | Path) -> dict[str, Any]:
        """Load checkpoint data from disk."""
        safetensors_path, json_path = Checkpoint._resolve_paths(filepath)

        with open(json_path) as f:
            json_data = json.load(f)

        version = json_data.get("version", "unknown")
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version {version} incompatible with {CHECKPOINT_VERSION}"
            )

        tensors = load_file(safetensors_path)
        weights = Checkpoint._tensors_to_weights(tensors)
        weight_names = {int(k): v for k, v in json_data["weight_names"].items()}

        result = {
            "weights": weights,
            "weight_names": weight_names,
            "version": version,
            "created_at": json_data.get("created_at"),
        }

        for key in ("architecture", "training_state", "metadata"):
            if key in json_data:
                result[key] = json_data[key]

        return result

    @staticmethod
    def load_weights_into(evaluator, filepath: str | Path) -> None:
        """Load weights from checkpoint into an existing evaluator."""
        checkpoint_data = Checkpoint.load(filepath)
        state_dict = {
            "weights": checkpoint_data["weights"],
            "weight_names": checkpoint_data["weight_names"],
        }
        evaluator.load_state_dict(state_dict)

    @staticmethod
    def exists(filepath: str | Path) -> bool:
        """Check if a checkpoint exists at the given path."""
        safetensors_path, json_path = Checkpoint._resolve_paths(filepath)
        return safetensors_path.exists() and json_path.exists()

    @staticmethod
    def _resolve_paths(filepath: str | Path) -> tuple[Path, Path]:
        """Resolve checkpoint file paths from base path."""
        filepath = Path(filepath)
        if filepath.suffix in (".safetensors", ".json"):
            filepath = filepath.with_suffix("")
        return filepath.with_suffix(".safetensors"), filepath.with_suffix(".json")

    @staticmethod
    def _weights_to_tensors(weights: dict[int, float | list]) -> dict[str, np.ndarray]:
        """Convert PyNeuraLogic weight format to safetensors format."""
        tensors = {}
        for idx, value in weights.items():
            key = f"weight_{idx}"
            if isinstance(value, (int, float, list, tuple)):
                tensors[key] = np.array(value, dtype=np.float32)
            elif isinstance(value, np.ndarray):
                tensors[key] = value.astype(np.float32)
            else:
                raise TypeError(
                    f"Unsupported weight type for index {idx}: {type(value)}"
                )
        return tensors

    @staticmethod
    def _tensors_to_weights(tensors: dict[str, np.ndarray]) -> dict[int, float | list]:
        """Convert safetensors format back to PyNeuraLogic weight format."""
        weights = {}
        for key, array in tensors.items():
            if not key.startswith("weight_"):
                continue
            idx = int(key.rsplit("_", 1)[1])
            weights[idx] = float(array) if array.ndim == 0 else array.tolist()
        return weights
