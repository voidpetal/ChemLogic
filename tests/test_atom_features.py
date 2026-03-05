"""Tests for RDKit atom feature extraction (Phase 2, Task 1)."""

import pytest


class TestAtomFeatureExtraction:
    """Test that smiles_to_pyg extracts RDKit atom features."""

    def test_default_behavior_unchanged(self):
        """Without atom_features, output should be unchanged."""
        from chemlogic.datasets.utils.smiles_conversion import smiles_to_pyg

        g = smiles_to_pyg("CCO")
        assert hasattr(g, "x"), "Should have x attribute"
        assert hasattr(g, "edge_attr"), "Should have edge_attr"

    def test_atom_features_all(self):
        """atom_features='all' should extract all available features."""
        from chemlogic.datasets.utils.smiles_conversion import (
            smiles_to_pyg,
            ATOM_FEATURE_MAP,
        )

        g = smiles_to_pyg("CCO", atom_features="all")
        # Should have all feature attributes
        for feat_name in ATOM_FEATURE_MAP.keys():
            assert hasattr(g, feat_name), f"Should have {feat_name} attribute"

    def test_atom_features_specific_list(self):
        """atom_features=['formal_charge'] should extract only that feature."""
        from chemlogic.datasets.utils.smiles_conversion import smiles_to_pyg

        g = smiles_to_pyg("CCO", atom_features=["formal_charge"])
        assert hasattr(g, "formal_charge"), "Should have formal_charge"
        # Should NOT have other features
        assert not hasattr(g, "hybridization"), "Should not have hybridization"

    def test_formal_charge_on_charged_atom(self):
        """Formal charge should be captured for charged atoms."""
        from chemlogic.datasets.utils.smiles_conversion import smiles_to_pyg

        # NH4+ has nitrogen with +1 formal charge
        g = smiles_to_pyg("[NH4+]", atom_features=["formal_charge"])
        # Find the nitrogen atom (first atom in this SMILES)
        # formal_charge should be 1 for nitrogen
        assert hasattr(g, "formal_charge")
        charges = g.formal_charge.tolist()
        assert 1 in charges, f"Expected +1 charge in {charges}"

    def test_is_aromatic_on_benzene(self):
        """Aromaticity should be captured for aromatic atoms."""
        from chemlogic.datasets.utils.smiles_conversion import smiles_to_pyg

        g = smiles_to_pyg("c1ccccc1", atom_features=["is_aromatic"])  # benzene
        assert hasattr(g, "is_aromatic")
        aromatic = g.is_aromatic.tolist()
        # All 6 carbons should be aromatic (value=1)
        assert aromatic.count(1) >= 6, f"Expected 6 aromatic atoms, got {aromatic}"

    def test_invalid_feature_raises_error(self):
        """Invalid feature name should raise ValueError."""
        from chemlogic.datasets.utils.smiles_conversion import smiles_to_pyg

        with pytest.raises(ValueError) as excinfo:
            smiles_to_pyg("CCO", atom_features=["invalid_feature"])
        assert "invalid_feature" in str(excinfo.value)

    def test_atom_feature_map_exists(self):
        """ATOM_FEATURE_MAP should be exported and contain expected features."""
        from chemlogic.datasets.utils.smiles_conversion import ATOM_FEATURE_MAP

        expected_features = [
            "formal_charge",
            "num_radical_electrons",
            "is_aromatic",
            "hybridization",
            "total_num_hs",
            "degree",
            "is_in_ring",
            "chiral_tag",
        ]
        for feat in expected_features:
            assert feat in ATOM_FEATURE_MAP, f"Expected {feat} in ATOM_FEATURE_MAP"
