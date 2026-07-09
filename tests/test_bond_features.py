"""Tests for bond feature extraction in smiles_to_pyg."""

import pytest

from chemlogic.datasets.utils.smiles_conversion import smiles_to_pyg


class TestBondFeatureExtraction:
    """Test bond feature extraction in smiles_to_pyg."""

    def test_default_behavior_unchanged(self):
        """bond_features=None should return same structure as before."""
        graph = smiles_to_pyg("CCO", bond_features=None)
        assert hasattr(graph, "edge_attr"), "Should have edge_attr attribute"
        assert hasattr(graph, "x"), "Should have x attribute"

    def test_bond_features_all(self):
        """bond_features='all' should extract all 4 bond features."""
        from chemlogic.datasets.utils.smiles_conversion import BOND_FEATURE_MAP

        graph = smiles_to_pyg("c1ccccc1", bond_features="all")
        # Should have feature tensors for all bond features
        for feat_name in BOND_FEATURE_MAP.keys():
            assert hasattr(graph, feat_name), (
                f"Missing {feat_name} with bond_features='all'"
            )

    def test_bond_features_specific_list(self):
        """bond_features=['is_aromatic'] should only extract that feature."""
        graph = smiles_to_pyg("c1ccccc1", bond_features=["is_aromatic"])
        assert hasattr(graph, "is_aromatic"), "Should have is_aromatic"
        # Should NOT have other features
        assert not hasattr(graph, "is_conjugated"), "Should not have is_conjugated"
        assert not hasattr(graph, "stereo"), "Should not have stereo"

    def test_is_aromatic_on_benzene(self):
        """Benzene bonds should be aromatic (is_aromatic=1)."""
        graph = smiles_to_pyg("c1ccccc1", bond_features=["is_aromatic"])
        # Benzene has aromatic bonds, at least one should be aromatic
        aromatic_values = graph.is_aromatic.tolist()
        assert any(v == 1 for v in aromatic_values), (
            f"Benzene bonds should be aromatic, got: {aromatic_values}"
        )

    def test_is_in_ring_on_benzene(self):
        """Benzene bonds should be in a ring (is_in_ring=1)."""
        graph = smiles_to_pyg("c1ccccc1", bond_features=["is_in_ring"])
        ring_values = graph.is_in_ring.tolist()
        assert any(v == 1 for v in ring_values), (
            f"Benzene bonds should be in ring, got: {ring_values}"
        )

    def test_is_conjugated_on_benzene(self):
        """Benzene bonds should be conjugated."""
        graph = smiles_to_pyg("c1ccccc1", bond_features=["is_conjugated"])
        conj_values = graph.is_conjugated.tolist()
        assert any(v == 1 for v in conj_values), (
            f"Benzene bonds should be conjugated, got: {conj_values}"
        )

    def test_invalid_feature_raises_error(self):
        """Invalid feature name should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            smiles_to_pyg("CCO", bond_features=["invalid_bond_feature"])
        assert "invalid_bond_feature" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_bond_feature_map_exists(self):
        """BOND_FEATURE_MAP should be importable and have expected features."""
        from chemlogic.datasets.utils.smiles_conversion import BOND_FEATURE_MAP

        expected = {"is_aromatic", "is_conjugated", "is_in_ring", "stereo"}
        assert set(BOND_FEATURE_MAP.keys()) == expected
