"""
Unit tests for the Checkpoint class.

Uses mock evaluator to test without Java/PyNeuraLogic dependency.
"""

import json

import numpy as np
import pytest

from chemlogic.utils.Checkpoint import CHECKPOINT_VERSION, Checkpoint


class MockEvaluator:
    """Mock evaluator for testing checkpoint save/load without Java dependency."""

    def __init__(self, weights: dict, weight_names: dict):
        self._weights = weights
        self._weight_names = weight_names

    def state_dict(self) -> dict:
        return {"weights": self._weights, "weight_names": self._weight_names}

    def load_state_dict(self, state_dict: dict) -> None:
        self._weights = state_dict["weights"]
        self._weight_names = state_dict["weight_names"]


class TestCheckpointSave:
    """Tests for Checkpoint.save()"""

    def test_save_creates_files(self, tmp_path):
        """Save creates .safetensors/.json files with version and timestamp."""
        weights = {0: 0.5, 1: [0.1, 0.2, 0.3]}
        weight_names = {0: "bias", 1: "layer1"}
        evaluator = MockEvaluator(weights, weight_names)

        filepath = tmp_path / "model"
        safetensors_path, json_path = Checkpoint.save(evaluator, filepath)

        assert safetensors_path.exists() and json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["version"] == CHECKPOINT_VERSION
        assert "T" in data["created_at"]  # ISO format

    def test_save_creates_parent_directories(self, tmp_path):
        """Save creates parent directories if needed."""
        evaluator = MockEvaluator({0: 1.0}, {0: "w"})
        filepath = tmp_path / "nested" / "dirs" / "model"
        Checkpoint.save(evaluator, filepath)
        assert filepath.with_suffix(".safetensors").exists()

    def test_save_with_optional_fields(self, tmp_path):
        """Save includes architecture, training_state, metadata when provided."""
        evaluator = MockEvaluator({0: 1.0}, {0: "w"})
        architecture = {"model_name": "gcn", "layers": 3}
        training_state = {"epoch": 10, "loss": 0.05}
        metadata = {"description": "Best model"}

        filepath = tmp_path / "model"
        _, json_path = Checkpoint.save(
            evaluator,
            filepath,
            architecture=architecture,
            training_state=training_state,
            metadata=metadata,
        )

        with open(json_path) as f:
            data = json.load(f)
        assert data["architecture"] == architecture
        assert data["training_state"] == training_state
        assert data["metadata"] == metadata


class TestCheckpointLoad:
    """Tests for Checkpoint.load()"""

    def test_load_returns_weights(self, tmp_path):
        """Load returns weights, weight_names, and optional fields."""
        weights = {0: 0.5, 1: [0.1, 0.2, 0.3]}
        weight_names = {0: "bias", 1: "layer1"}
        architecture = {"model_name": "gcn"}
        evaluator = MockEvaluator(weights, weight_names)

        filepath = tmp_path / "model"
        Checkpoint.save(evaluator, filepath, architecture=architecture)
        loaded = Checkpoint.load(filepath)

        assert loaded["weights"][0] == pytest.approx(0.5)
        assert loaded["weights"][1] == pytest.approx([0.1, 0.2, 0.3])
        assert loaded["weight_names"] == weight_names
        assert loaded["architecture"] == architecture

    def test_load_with_extension(self, tmp_path):
        """Load works when filepath has .safetensors or .json extension."""
        evaluator = MockEvaluator({0: 1.0}, {0: "w"})
        filepath = tmp_path / "model"
        Checkpoint.save(evaluator, filepath)

        loaded1 = Checkpoint.load(filepath.with_suffix(".safetensors"))
        loaded2 = Checkpoint.load(filepath.with_suffix(".json"))
        assert loaded1["weights"][0] == loaded2["weights"][0] == pytest.approx(1.0)

    def test_load_not_found_error(self, tmp_path):
        """Load raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            Checkpoint.load(tmp_path / "nonexistent")

    def test_load_version_error(self, tmp_path):
        """Load raises ValueError for incompatible version."""
        evaluator = MockEvaluator({0: 1.0}, {0: "w"})
        filepath = tmp_path / "model"
        Checkpoint.save(evaluator, filepath)

        # Modify version in JSON
        json_path = filepath.with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)
        data["version"] = "99.0"
        with open(json_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError):
            Checkpoint.load(filepath)


class TestCheckpointLoadWeightsInto:
    """Tests for Checkpoint.load_weights_into()"""

    def test_load_weights_into(self, tmp_path):
        """load_weights_into restores weights to a different evaluator."""
        original_weights = {0: 0.5, 1: [0.1, 0.2, 0.3]}
        original_names = {0: "bias", 1: "layer1"}
        evaluator = MockEvaluator(original_weights, original_names)

        filepath = tmp_path / "model"
        Checkpoint.save(evaluator, filepath)

        # Load into new evaluator with different weights
        new_evaluator = MockEvaluator({0: 999.0, 1: [9.0, 9.0, 9.0]}, original_names)
        Checkpoint.load_weights_into(new_evaluator, filepath)

        state = new_evaluator.state_dict()
        assert state["weights"][0] == pytest.approx(0.5)
        assert state["weights"][1] == pytest.approx([0.1, 0.2, 0.3])


class TestCheckpointRoundTrip:
    """Tests for save/load round-trip with different weight types."""

    def test_vector_and_matrix_roundtrip(self, tmp_path):
        """Vector and 2D matrix weights round-trip correctly."""
        weights = {
            0: [0.1, 0.2, 0.3],
            1: [[1.0, 2.0], [3.0, 4.0]],  # 2D matrix
        }
        weight_names = {0: "vec", 1: "matrix"}
        evaluator = MockEvaluator(weights, weight_names)

        filepath = tmp_path / "model"
        Checkpoint.save(evaluator, filepath)
        loaded = Checkpoint.load(filepath)

        assert loaded["weights"][0] == pytest.approx([0.1, 0.2, 0.3])
        assert np.allclose(loaded["weights"][1], [[1.0, 2.0], [3.0, 4.0]])


class TestCheckpointExists:
    """Tests for Checkpoint.exists()"""

    def test_exists_true(self, tmp_path):
        """exists() returns True for valid checkpoint."""
        evaluator = MockEvaluator({0: 1.0}, {0: "w"})
        filepath = tmp_path / "model"
        Checkpoint.save(evaluator, filepath)
        assert Checkpoint.exists(filepath) is True

    def test_exists_false(self, tmp_path):
        """exists() returns False for missing or partial checkpoint."""
        assert Checkpoint.exists(tmp_path / "nonexistent") is False

        # Partial (only one file)
        evaluator = MockEvaluator({0: 1.0}, {0: "w"})
        filepath = tmp_path / "model"
        Checkpoint.save(evaluator, filepath)
        filepath.with_suffix(".json").unlink()
        assert Checkpoint.exists(filepath) is False
