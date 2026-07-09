"""
Integration tests for Pipeline checkpointing.

These tests require Java/PyNeuraLogic backend and are marked as slow.
They test the full save/load/inference cycle with real evaluators.
"""

import pytest

from chemlogic.utils.Checkpoint import Checkpoint
from chemlogic.utils.Pipeline import Pipeline

# Mark all tests in this module as slow (require Java backend)
pytestmark = pytest.mark.slow


@pytest.fixture
def simple_pipeline():
    """Create a simple pipeline for testing."""
    return Pipeline(
        dataset_name="bbbp",
        model_name="gnn",
        param_size=4,
        layers=1,
        smiles_list=["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"],
        labels=[0, 1, 0, 1, 0, 1],
        task="classification",
    )


@pytest.fixture
def trained_pipeline(simple_pipeline):
    """Create and train a pipeline for testing."""
    simple_pipeline.train_test_cycle(
        epochs=3, early_stopping_rounds=10, split_ratio=0.5
    )
    return simple_pipeline


class TestPipelineSaveCheckpoint:
    """Tests for Pipeline.save_checkpoint()"""

    def test_save_checkpoint_creates_files(self, trained_pipeline, tmp_path):
        """save_checkpoint creates both files."""
        filepath = tmp_path / "model"
        safetensors_path, json_path = trained_pipeline.save_checkpoint(
            filepath, metadata={"description": "Test model"}
        )
        assert safetensors_path.exists() and json_path.exists()

    def test_save_checkpoint_auto_path(self, trained_pipeline):
        """save_checkpoint auto-generates path in checkpoints/{dataset}/."""
        safetensors_path, json_path = trained_pipeline.save_checkpoint()

        assert safetensors_path.exists()
        assert "checkpoints" in str(safetensors_path)
        assert trained_pipeline.dataset.dataset_name in str(safetensors_path)

        # Cleanup
        safetensors_path.unlink()
        json_path.unlink()

    def test_save_checkpoint_untrained_raises(self, simple_pipeline, tmp_path):
        """save_checkpoint raises if model not trained."""
        with pytest.raises(ValueError, match="not been trained"):
            simple_pipeline.save_checkpoint(tmp_path / "model")


class TestPipelineFromCheckpoint:
    """Tests for Pipeline.from_checkpoint()"""

    def test_from_checkpoint_loads_pipeline(self, trained_pipeline, tmp_path):
        """from_checkpoint loads a pipeline with matching architecture."""
        filepath = tmp_path / "model"
        trained_pipeline.save_checkpoint(filepath)

        loaded = Pipeline.from_checkpoint(filepath)

        assert hasattr(loaded, "evaluator")
        assert loaded.task == trained_pipeline.task
        for key in ["dataset_name", "model_name", "param_size", "layers"]:
            assert (
                loaded._architecture_params[key]
                == trained_pipeline._architecture_params[key]
            )

    def test_from_checkpoint_with_custom_smiles(self, trained_pipeline, tmp_path):
        """from_checkpoint accepts custom SMILES list."""
        filepath = tmp_path / "model"
        trained_pipeline.save_checkpoint(filepath)

        loaded = Pipeline.from_checkpoint(
            filepath, smiles_list=["C", "CC", "CCC"], labels=[0, 1, 0]
        )
        assert hasattr(loaded, "evaluator")


class TestPipelineInferenceAfterLoad:
    """Tests for inference consistency after checkpoint load."""

    def test_inference_after_load(self, trained_pipeline, tmp_path):
        """Loaded pipeline produces predictions."""
        test_smiles = ["C", "CC"]
        original_predictions = trained_pipeline.inference(test_smiles)

        filepath = tmp_path / "model"
        trained_pipeline.save_checkpoint(filepath)
        loaded = Pipeline.from_checkpoint(filepath)
        loaded_predictions = loaded.inference(test_smiles)

        assert len(loaded_predictions) == len(original_predictions)
        for pred in loaded_predictions:
            assert isinstance(pred, (int, float))


class TestPipelineCheckpointEvery:
    """Tests for checkpoint_every parameter."""

    def test_checkpoint_every_creates_files_with_state(self, simple_pipeline, tmp_path):
        """checkpoint_every saves checkpoints at intervals with training state."""
        simple_pipeline.train_test_cycle(
            epochs=5,
            early_stopping_rounds=10,
            split_ratio=0.5,
            checkpoint_every=2,
            checkpoint_dir=tmp_path,
        )

        # Should have checkpoints for epochs 2 and 4
        assert (tmp_path / "epoch_2.safetensors").exists()
        assert (tmp_path / "epoch_4.safetensors").exists()

        # Verify training state is saved
        data = Checkpoint.load(tmp_path / "epoch_2")
        assert data["training_state"]["epoch"] == 2
