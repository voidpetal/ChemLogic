"""Tests for graph-level predicate generation via synthetic graph node."""

import os
import re
import pytest


class TestGraphLevelPredicates:
    """Test that graph_features generate valued predicates on a synthetic graph node.

    Graph features are represented via a synthetic node connected to all atoms:
    - graph_node(G) marks the synthetic node
    - bond(A, G, B), graph_bond(B) connects each atom to the graph node
    - <value> feature_name(G) holds the feature value on the graph node
    """

    def test_no_graph_features_unchanged(self):
        """Without graph_features, output should have no graph node."""
        from chemlogic.datasets.utils.smiles_conversion import get_dataset_and_mappings

        dataset, _ = get_dataset_and_mappings(
            smiles_list=["CCO"],
            labels=[1],
            file_prefix="test_no_gf",
            output_location="/tmp",
            graph_features=None,
        )

        with open("/tmp/test_no_gf_examples.txt") as f:
            content = f.read()
            # Should have standard predicates but no graph node
            assert "<1>" in content  # Has weighted predicates
            assert "graph_node" not in content
            assert "graph_bond" not in content
            assert "mol_weight" not in content

        # Cleanup
        os.remove("/tmp/test_no_gf_examples.txt")
        os.remove("/tmp/test_no_gf_queries.txt")

    def test_graph_features_added_as_valued_predicates(self):
        """Graph features should appear as valued predicates on graph node."""
        from chemlogic.datasets.utils.smiles_conversion import get_dataset_and_mappings

        graph_features = {"mol_weight": [46.07, 30.07]}
        dataset, _ = get_dataset_and_mappings(
            smiles_list=["CCO", "CC"],
            labels=[1, 0],
            file_prefix="test_gf",
            output_location="/tmp",
            graph_features=graph_features,
        )

        with open("/tmp/test_gf_examples.txt") as f:
            lines = f.readlines()
            # First molecule should have graph node with mol_weight
            assert "graph_node(" in lines[0], f"Expected 'graph_node(' in: {lines[0]}"
            # Pattern: <value> mol_weight(graph_node_id)
            assert re.search(r"<46\.07> mol_weight\(\d+\)", lines[0]), (
                f"Expected '<46.07> mol_weight(N)' in: {lines[0]}"
            )
            # Second molecule
            assert re.search(r"<30\.07> mol_weight\(\d+\)", lines[1]), (
                f"Expected '<30.07> mol_weight(N)' in: {lines[1]}"
            )

        # Cleanup
        os.remove("/tmp/test_gf_examples.txt")
        os.remove("/tmp/test_gf_queries.txt")

    def test_multiple_graph_features(self):
        """Multiple graph features should all be on the same graph node."""
        from chemlogic.datasets.utils.smiles_conversion import get_dataset_and_mappings

        graph_features = {"mol_weight": [46.07], "log_p": [-0.18], "num_atoms": [9]}
        dataset, _ = get_dataset_and_mappings(
            smiles_list=["CCO"],
            labels=[1],
            file_prefix="test_multi_gf",
            output_location="/tmp",
            graph_features=graph_features,
        )

        with open("/tmp/test_multi_gf_examples.txt") as f:
            content = f.read()
            # All features should be on graph node (same ID)
            assert re.search(r"<46\.07> mol_weight\(\d+\)", content)
            assert re.search(r"<-0\.18> log_p\(\d+\)", content)
            assert re.search(r"<9> num_atoms\(\d+\)", content)
            # Should have exactly one graph_node
            assert content.count("graph_node(") == 1

        # Cleanup
        os.remove("/tmp/test_multi_gf_examples.txt")
        os.remove("/tmp/test_multi_gf_queries.txt")

    def test_graph_features_format_angle_brackets(self):
        """Valued predicates should use <value> feature_name(node_id) format."""
        from chemlogic.datasets.utils.smiles_conversion import get_dataset_and_mappings

        graph_features = {"mol_weight": [46.07]}
        dataset, _ = get_dataset_and_mappings(
            smiles_list=["CCO"],
            labels=[1],
            file_prefix="test_format",
            output_location="/tmp",
            graph_features=graph_features,
        )

        with open("/tmp/test_format_examples.txt") as f:
            content = f.read()
            # Should have "<46.07> mol_weight(N)" with angle brackets and node ID
            assert re.search(r"<46\.07> mol_weight\(\d+\)", content)

        # Cleanup
        os.remove("/tmp/test_format_examples.txt")
        os.remove("/tmp/test_format_queries.txt")

    def test_graph_node_connected_to_all_atoms(self):
        """Graph node should be connected to all atoms via graph_bond."""
        from chemlogic.datasets.utils.smiles_conversion import get_dataset_and_mappings

        graph_features = {"mol_weight": [46.07]}
        dataset, _ = get_dataset_and_mappings(
            smiles_list=["CCO"],  # Ethanol: 3 heavy atoms + 6 hydrogens = 9 atoms
            labels=[1],
            file_prefix="test_conn",
            output_location="/tmp",
            graph_features=graph_features,
        )

        with open("/tmp/test_conn_examples.txt") as f:
            content = f.read()
            # Should have 9 graph_bond connections (one per atom)
            graph_bond_count = content.count("graph_bond(")
            assert graph_bond_count == 9, (
                f"Expected 9 graph_bond connections for 9 atoms, got {graph_bond_count}"
            )

        # Cleanup
        os.remove("/tmp/test_conn_examples.txt")
        os.remove("/tmp/test_conn_queries.txt")
