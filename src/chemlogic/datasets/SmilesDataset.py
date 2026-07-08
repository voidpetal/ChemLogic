import logging
import os
import re

import pandas as pd
from neuralogic.core import R, V

from chemlogic.datasets.Dataset import Dataset
from chemlogic.datasets.utils.smiles_conversion import get_dataset_and_mappings


def _to_snake_case(name: str) -> str:
    """Convert column name to valid snake_case predicate name.

    Handles: CamelCase, spaces, hyphens, and other separators.
    """
    # Replace spaces, hyphens, and other separators with underscores
    s = re.sub(r"[\s\-\.]+", "_", name)
    # Insert underscore before uppercase letters (CamelCase handling)
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Lowercase and collapse multiple underscores
    s = re.sub(r"_+", "_", s.lower())
    # Strip leading/trailing underscores
    return s.strip("_")


class SmilesDataset(Dataset):
    def __init__(
        self,
        smiles_list: list[str] | pd.DataFrame,
        labels: list[int] | None = None,
        param_size: int = 8,
        dataset_name: str = "dataset",
        output_directory: str = ".",
        keep: bool = False,
        atom_features: str | list[str] | None = None,
        bond_features: str | list[str] | None = None,
        broadcast_graph_features: bool = False,
        num_outputs: int = 1,
    ):
        """
        Create a custom dataset from SMILES.

        Args:
            smiles_list (list[str] | pd.DataFrame): A list of SMILES strings, or a DataFrame
                with 'smiles' and 'target' columns (plus optional numeric feature columns).
            labels (list[int] | None): A list of labels. Required if smiles_list is a list,
                ignored if smiles_list is a DataFrame.
            param_size (int): The size of the parameter.
            dataset_name (str): The name of the dataset
            output_directory (Optional[str]): The output directory where to dump the dataset. Leave blank if one-time import.
            keep (Optional[bool]): Whether to keep the created files or not.
            atom_features (str | list[str] | None): Atom features to extract as node-level predicates.
                - None: No additional features (default, backward compatible)
                - 'all': Extract all 8 available RDKit atom features
                - list[str]: Extract only specified features (e.g., ['formal_charge', 'is_aromatic'])
                Available features: formal_charge, num_radical_electrons, is_aromatic,
                hybridization, total_num_hs, degree, is_in_ring, chiral_tag
            bond_features (str | list[str] | None): Bond features to extract as edge-level predicates.
                - None: No additional features (default, backward compatible)
                - 'all': Extract all 4 available RDKit bond features
                - list[str]: Extract only specified features (e.g., ['is_aromatic', 'is_conjugated'])
                Available features: is_aromatic, is_conjugated, is_in_ring, stereo
            broadcast_graph_features (bool): If True, broadcast graph features to all atoms
                instead of using a synthetic graph node. This adds graph feature predicates
                to every atom in the molecule, avoiding connectivity issues with synthetic nodes.
                Default: False (use synthetic graph node approach).
        """

        # Handle DataFrame input
        if isinstance(smiles_list, pd.DataFrame):
            df = smiles_list
            if "smiles" not in df.columns:
                raise ValueError("DataFrame must have a 'smiles' column")
            if "target" not in df.columns:
                raise ValueError("DataFrame must have a 'target' column")

            self.smiles_list = df["smiles"].tolist()
            self.labels = df["target"].tolist()

            # Extract extra columns as graph features
            extra_cols = [c for c in df.columns if c not in ("smiles", "target")]
            self.graph_features = {}
            for col in extra_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(
                        f"Column '{col}' must be numeric, got {df[col].dtype}. "
                        "Only numeric columns are supported as graph features."
                    )
                snake_col = _to_snake_case(col)
                self.graph_features[snake_col] = df[col].tolist()
        else:
            # Original list-based input
            if labels is None:
                raise ValueError("labels is required when smiles_list is a list")
            if len(smiles_list) != len(labels):
                raise ValueError(
                    "The params smiles_list and labels must be of same length!"
                )
            self.smiles_list = smiles_list
            self.labels = labels
            self.graph_features = {}

        self.output_directory = output_directory
        self.keep = keep
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.broadcast_graph_features = broadcast_graph_features
        self.num_outputs = num_outputs

        # Placeholder for atom and bond types
        atom_types = ["placeholder"]
        bond_types = ["placeholder"]

        # how to get key atoms?? Non carbon and hydrogen or key atoms always o, s, n, p?
        key_atoms = ["o", "s", "n", "p"]

        super().__init__(
            dataset_name,
            "atom_embed",
            "bond_embed",
            "bond",
            atom_types,
            key_atoms,
            bond_types,
            "b_1",
            "b_2",
            "b_3",
            ["b_1", "b_2", "b_3"],
            ["b_4"],
            "c",
            "o",
            "h",
            "n",
            "s",
            ["f", "cl", "br", "i"],
            param_size,
        )

    def load_data(self):
        (
            dataset,
            (
                atom_types,
                bond_types,
                atom_feature_names,
                bond_feature_names,
                graph_feature_names,
            ),
        ) = get_dataset_and_mappings(
            smiles_list=self.smiles_list,
            labels=self.labels,
            output_location=self.output_directory,
            file_prefix=self.dataset_name,
            graph_features=self.graph_features,
            atom_features=self.atom_features,
            bond_features=self.bond_features,
            broadcast_graph_features=self.broadcast_graph_features,
            num_outputs=self.num_outputs,
        )
        self.atom_types = atom_types
        self.bond_types = bond_types
        # Store feature names for template creation
        self.atom_feature_names = atom_feature_names
        self.bond_feature_names = bond_feature_names
        self.graph_feature_names = graph_feature_names
        return dataset

    def create_template(self):
        """Create template rules including extended features.

        This overrides the parent create_template() to add rules for:
        - Atom features (node-level): contribute to atom_embed (same as atom types)
        - Bond features (edge-level): contribute to bond_embed (same as bond types)
        - Graph features: represented via a synthetic graph node connected to all atoms

        In PyNeuraLogic, multiple rules with the same head predicate are summed,
        so these features naturally combine with the base embeddings.
        """
        # Call parent to create base atom/bond embedding rules
        super().create_template()

        # Add rules for atom features (node-level)
        # Each atom feature contributes to atom_embed, just like atom type predicates
        # e.g., atom_embed(A) <= formal_charge(A) adds to atom_embed(A) <= c(A)
        if hasattr(self, "atom_feature_names") and self.atom_feature_names:
            self.add_rules(
                [
                    (
                        R.get(self.node_embed)(V.A)[self.param_size,]
                        <= R.get(feat_name)(V.A)
                    )
                    for feat_name in self.atom_feature_names
                ]
            )

        # Add rules for bond features (edge-level)
        # Each bond feature contributes to bond_embed, just like bond type predicates
        # e.g., bond_embed(B) <= is_conjugated(B) adds to bond_embed(B) <= b_1(B)
        if hasattr(self, "bond_feature_names") and self.bond_feature_names:
            self.add_rules(
                [
                    (
                        R.get(self.edge_embed)(V.B)[self.param_size,]
                        <= R.get(feat_name)(V.B)
                    )
                    for feat_name in self.bond_feature_names
                ]
            )

        # Add rules for graph features
        # Two modes:
        # 1. Synthetic node (default): graph node connected to all atoms via graph_bond
        # 2. Broadcast mode: graph features added directly to all atoms (no synthetic node)
        if hasattr(self, "graph_feature_names") and self.graph_feature_names:
            # In both modes, graph features contribute to atom_embed
            # The difference is what they're attached to in the examples file
            self.add_rules(
                [
                    (
                        R.get(self.node_embed)(V.A)[self.param_size,]
                        <= R.get(feat_name)(V.A)
                    )
                    for feat_name in self.graph_feature_names
                ]
            )

            # Only add graph_bond rule in synthetic node mode (not broadcast)
            if not getattr(self, "broadcast_graph_features", False):
                self.add_rules(
                    [
                        (
                            R.get(self.edge_embed)(V.B)[self.param_size,]
                            <= R.graph_bond(V.B)
                        )
                    ]
                )

    # TODO: define a dump function
    # TODO: convert to a Dataset, not FileDataset

    def clear(self):
        for file in ["examples", "queries"]:
            file_path = f"{self.output_directory}/{self.dataset_name}_{file}.txt"
            try:
                os.remove(file_path)
            except Exception:
                logging.info(f"Unable to delete {file} file at {file_path}")
