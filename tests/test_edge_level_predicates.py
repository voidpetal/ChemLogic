"""Tests for edge-level (bond) predicate generation in smiles_conversion."""

import os
import tempfile

import pytest

from chemlogic.datasets.utils.smiles_conversion import (
    BOND_FEATURE_MAP,
    get_dataset_and_mappings,
)


class TestEdgeLevelPredicates:
    """Test edge-level predicate generation in get_dataset_and_mappings."""

    def test_default_behavior_no_bond_features(self):
        """bond_features=None should produce same output as before (no edge-level predicates)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["CCO"],
                labels=[1],
                file_prefix="test_default",
                output_location=tmpdir,
                bond_features=None,
            )

            with open(f"{tmpdir}/test_default_examples.txt") as f:
                content = f.read()

            # Should not contain any bond feature predicates
            # Note: is_aromatic and is_in_ring are also atom features, so check for bond_id format
            for feat_name in BOND_FEATURE_MAP.keys():
                # Bond predicates have higher IDs (offset by num_atoms)
                # Look for pattern like "feature_name(12)" where ID > typical atom count
                lines = content.split("\n")
                for line in lines:
                    if f"{feat_name}(" in line:
                        # Extract IDs and check if any are bond IDs (high numbers)
                        import re

                        matches = re.findall(rf"{feat_name}\((\d+)\)", line)
                        # CCO has 9 atoms with H, so bond IDs would be >= 9
                        bond_ids = [int(m) for m in matches if int(m) >= 9]
                        assert len(bond_ids) == 0, (
                            f"Should not have bond {feat_name} predicates when bond_features=None"
                        )

    def test_bond_features_is_aromatic_on_benzene(self):
        """Benzene has aromatic bonds, should generate is_aromatic predicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["c1ccccc1"],  # benzene
                labels=[1],
                file_prefix="test_aromatic",
                output_location=tmpdir,
                bond_features=["is_aromatic"],
            )

            with open(f"{tmpdir}/test_aromatic_examples.txt") as f:
                content = f.read()

            # Benzene bonds are aromatic, should have "<1> is_aromatic(bond_id)"
            # Bond IDs are offset by num_atoms (benzene with H has 12 atoms, so bond IDs start at 12)
            assert "is_aromatic(" in content, (
                f"Should have is_aromatic bond predicates. Got: {content}"
            )
            # At least one aromatic bond predicate with value 1
            assert "<1> is_aromatic(" in content, (
                f"Benzene should have aromatic bonds (value=1). Got: {content}"
            )

    def test_bond_features_is_in_ring_on_benzene(self):
        """Benzene bonds are in a ring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["c1ccccc1"],
                labels=[1],
                file_prefix="test_ring",
                output_location=tmpdir,
                bond_features=["is_in_ring"],
            )

            with open(f"{tmpdir}/test_ring_examples.txt") as f:
                content = f.read()

            # Benzene bonds are in a ring
            assert "<1> is_in_ring(" in content, (
                f"Benzene bonds should be in ring. Got: {content}"
            )

    def test_bond_features_all(self):
        """bond_features='all' should include features with non-zero values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["c1ccccc1"],
                labels=[1],
                file_prefix="test_all",
                output_location=tmpdir,
                bond_features="all",
            )

            with open(f"{tmpdir}/test_all_examples.txt") as f:
                content = f.read()

            # Benzene bonds: is_aromatic=1, is_conjugated=1, is_in_ring=1, stereo=0
            # stereo=0 should be omitted
            assert "is_aromatic(" in content, (
                f"Missing is_aromatic. Got: {content[:500]}"
            )
            assert "is_conjugated(" in content, (
                f"Missing is_conjugated. Got: {content[:500]}"
            )
            assert "is_in_ring(" in content, f"Missing is_in_ring. Got: {content[:500]}"
            # stereo is 0 for all bonds, so it should be omitted
            assert "stereo(" not in content, (
                f"stereo should be omitted (all zeros). Got: {content[:500]}"
            )

    def test_bond_features_specific_list(self):
        """bond_features=['is_aromatic', 'is_conjugated'] should only include those."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["c1ccccc1"],
                labels=[1],
                file_prefix="test_specific",
                output_location=tmpdir,
                bond_features=["is_aromatic", "is_conjugated"],
            )

            with open(f"{tmpdir}/test_specific_examples.txt") as f:
                content = f.read()

            # Should have the requested features
            assert "is_aromatic(" in content
            assert "is_conjugated(" in content

            # Check that stereo is NOT present (it's bond-specific, not atom)
            # Note: is_in_ring could appear from atoms, so only check stereo
            import re

            stereo_matches = re.findall(r"stereo\(\d+\)", content)
            assert len(stereo_matches) == 0, (
                f"Should not have stereo predicate. Got: {content}"
            )

    def test_multiple_molecules_correct_predicates(self):
        """Each molecule should have its own bond feature predicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=[
                    "c1ccccc1",
                    "CCO",
                ],  # benzene (aromatic), ethanol (not aromatic)
                labels=[1, 0],
                file_prefix="test_multi",
                output_location=tmpdir,
                bond_features=["is_aromatic"],
            )

            with open(f"{tmpdir}/test_multi_examples.txt") as f:
                lines = f.readlines()

            # First molecule (benzene) should have aromatic bonds (value=1)
            assert "<1> is_aromatic(" in lines[0], (
                f"Benzene should have aromatic bonds. Got: {lines[0]}"
            )

            # Second molecule (ethanol) should NOT have is_aromatic predicates
            # because boolean features with value=0 are omitted
            assert "is_aromatic(" not in lines[1], (
                f"Ethanol should not have is_aromatic predicates (0 values omitted). Got: {lines[1]}"
            )
