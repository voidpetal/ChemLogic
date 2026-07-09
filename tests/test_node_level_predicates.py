"""Tests for node-level (atom) predicate generation in smiles_conversion."""

import os
import tempfile

import pytest

from chemlogic.datasets.utils.smiles_conversion import (
    ATOM_FEATURE_MAP,
    get_dataset_and_mappings,
)


class TestNodeLevelPredicates:
    """Test node-level predicate generation in get_dataset_and_mappings."""

    def test_default_behavior_no_atom_features(self):
        """atom_features=None should produce same output as before (no node-level predicates)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["CCO"],
                labels=[1],
                file_prefix="test_default",
                output_location=tmpdir,
                atom_features=None,
            )

            with open(f"{tmpdir}/test_default_examples.txt") as f:
                content = f.read()

            # Should not contain any atom feature predicates
            for feat_name in ATOM_FEATURE_MAP.keys():
                # Feature predicates have format: "{value} feature_name(atom_id)"
                assert f"{feat_name}(" not in content, (
                    f"Should not have {feat_name} predicate when atom_features=None"
                )

    def test_atom_features_formal_charge_on_charged_molecule(self):
        """NH4+ has nitrogen with formal_charge=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["[NH4+]"],
                labels=[1],
                file_prefix="test_charged",
                output_location=tmpdir,
                atom_features=["formal_charge"],
            )

            with open(f"{tmpdir}/test_charged_examples.txt") as f:
                content = f.read()

            # NH4+ has N atom (idx 0) with formal_charge=1
            # Format: "<value> feature_name(atom_id)"
            assert "formal_charge(0)" in content, (
                f"Should have formal_charge predicate for N atom. Got: {content}"
            )
            # The nitrogen should have charge +1
            assert "<1> formal_charge(0)" in content, (
                f"N in NH4+ should have formal_charge=1. Got: {content}"
            )

    def test_atom_features_is_aromatic_on_benzene(self):
        """Benzene has aromatic carbons."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["c1ccccc1"],  # benzene
                labels=[1],
                file_prefix="test_aromatic",
                output_location=tmpdir,
                atom_features=["is_aromatic"],
            )

            with open(f"{tmpdir}/test_aromatic_examples.txt") as f:
                content = f.read()

            # Benzene carbons are aromatic (is_aromatic=1)
            # At least the first carbon (idx 0) should be aromatic
            assert "<1> is_aromatic(0)" in content, (
                f"Benzene C atoms should be aromatic. Got: {content}"
            )

    def test_atom_features_all(self):
        """atom_features='all' should include features with non-zero values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use benzene - has aromatic atoms so boolean features will appear
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["c1ccccc1"],  # benzene - aromatic
                labels=[1],
                file_prefix="test_all",
                output_location=tmpdir,
                atom_features="all",
            )

            with open(f"{tmpdir}/test_all_examples.txt") as f:
                content = f.read()

            # Should have predicates for features with non-zero values
            # Benzene carbons: is_aromatic=1, hybridization=3(SP2), degree=3, is_in_ring=1
            # Benzene hydrogens: degree=1
            # Features with all-zero values are omitted: formal_charge, num_radical_electrons, chiral_tag, total_num_hs
            assert "is_aromatic(" in content, f"Missing is_aromatic. Got: {content}"
            assert "hybridization(" in content, f"Missing hybridization. Got: {content}"
            assert "degree(" in content, f"Missing degree. Got: {content}"
            assert "is_in_ring(" in content, f"Missing is_in_ring. Got: {content}"

            # These should NOT appear because all values are 0
            assert "formal_charge(" not in content, (
                f"formal_charge should be omitted (all zeros). Got: {content}"
            )
            assert "num_radical_electrons(" not in content, (
                f"num_radical_electrons should be omitted. Got: {content}"
            )
            assert "chiral_tag(" not in content, (
                f"chiral_tag should be omitted. Got: {content}"
            )

    def test_atom_features_specific_list(self):
        """atom_features=['formal_charge', 'degree'] should only include those with non-zero values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["CCO"],
                labels=[1],
                file_prefix="test_specific",
                output_location=tmpdir,
                atom_features=["formal_charge", "degree"],
            )

            with open(f"{tmpdir}/test_specific_examples.txt") as f:
                content = f.read()

            # Should have degree (non-zero for all atoms)
            assert "degree(" in content, f"Missing degree. Got: {content}"

            # formal_charge is 0 for all atoms in CCO, so it should be omitted
            assert "formal_charge(" not in content, (
                f"formal_charge should be omitted (all zeros). Got: {content}"
            )

            # Should NOT have other features
            assert "is_aromatic(" not in content
            assert "hybridization(" not in content

    def test_multiple_molecules_have_correct_predicates(self):
        """Each molecule should have its own atom feature predicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = get_dataset_and_mappings(
                smiles_list=["[NH4+]", "C"],  # charged N, then methane
                labels=[1, 0],
                file_prefix="test_multi",
                output_location=tmpdir,
                atom_features=["formal_charge"],
            )

            with open(f"{tmpdir}/test_multi_examples.txt") as f:
                lines = f.readlines()

            # First molecule (NH4+) should have formal_charge=1 for N
            assert "<1> formal_charge(0)" in lines[0], (
                f"NH4+ should have formal_charge=1. Got: {lines[0]}"
            )

            # Second molecule (CH4) should NOT have formal_charge predicates
            # because all atoms have formal_charge=0, and 0 values are omitted
            assert "formal_charge(" not in lines[1], (
                f"CH4 should not have formal_charge predicates (0 values omitted). Got: {lines[1]}"
            )
