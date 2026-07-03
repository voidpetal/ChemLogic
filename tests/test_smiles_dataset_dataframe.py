"""Tests for SmilesDataset DataFrame support (Phase 1, Task 1)."""

import pytest
import pandas as pd


class TestSmilesDatasetDataFrameSupport:
    """Test that SmilesDataset accepts DataFrame input."""

    def test_original_api_still_works(self):
        """Original list-based API should work unchanged."""
        from chemlogic.datasets import SmilesDataset

        ds = SmilesDataset(["CCO", "CC"], [0, 1], 8, "test_original")
        assert ds.smiles_list == ["CCO", "CC"]
        assert ds.labels == [0, 1]
        assert ds.graph_features == {}

    def test_dataframe_api_works(self):
        """DataFrame with smiles, target, and extra columns should work."""
        from chemlogic.datasets import SmilesDataset

        df = pd.DataFrame(
            {"smiles": ["CCO", "CC"], "target": [0, 1], "mol_weight": [46.07, 30.07]}
        )
        ds = SmilesDataset(df, None, 8, "test_df")
        assert ds.smiles_list == ["CCO", "CC"]
        assert ds.labels == [0, 1]
        assert "mol_weight" in ds.graph_features
        assert ds.graph_features["mol_weight"] == [46.07, 30.07]

    def test_camelcase_converted_to_snake_case(self):
        """CamelCase column names should be converted to snake_case."""
        from chemlogic.datasets import SmilesDataset

        df = pd.DataFrame(
            {"smiles": ["CCO"], "target": [0], "MolWeight": [46.07], "LogP": [-0.18]}
        )
        ds = SmilesDataset(df, None, 8, "test_camel")
        assert "mol_weight" in ds.graph_features
        assert "log_p" in ds.graph_features

    def test_non_numeric_column_raises_error(self):
        """Non-numeric extra columns should raise ValueError."""
        from chemlogic.datasets import SmilesDataset

        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "target": [0],
                "name": ["ethanol"],  # String column - should fail
            }
        )
        with pytest.raises(ValueError) as excinfo:
            SmilesDataset(df, None, 8, "test_non_numeric")
        assert "numeric" in str(excinfo.value).lower()

    def test_missing_smiles_column_raises_error(self):
        """DataFrame without 'smiles' column should raise ValueError."""
        from chemlogic.datasets import SmilesDataset

        df = pd.DataFrame(
            {
                "molecule": ["CCO"],  # Wrong column name
                "target": [0],
            }
        )
        with pytest.raises(ValueError) as excinfo:
            SmilesDataset(df, None, 8, "test_no_smiles")
        assert "smiles" in str(excinfo.value).lower()

    def test_missing_target_column_raises_error(self):
        """DataFrame without 'target' column should raise ValueError."""
        from chemlogic.datasets import SmilesDataset

        df = pd.DataFrame(
            {
                "smiles": ["CCO"],
                "label": [0],  # Wrong column name
            }
        )
        with pytest.raises(ValueError) as excinfo:
            SmilesDataset(df, None, 8, "test_no_target")
        assert "target" in str(excinfo.value).lower()
