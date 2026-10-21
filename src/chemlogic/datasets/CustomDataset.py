import os

from neuralogic.dataset import FileDataset

from chemlogic.datasets.Dataset import Dataset


class CustomDataset(Dataset):
    """Dataset loaded from user-provided examples and queries files.

    Accepts explicit paths to pre-formatted NeuraLogic text files. When both
    paths are given they are validated and used directly; otherwise falls back
    to the base class behaviour which expects packaged dataset files under
    `src/chemlogic/data/datasets/`.
    """

    def __init__(
        self,
        examples: str,
        queries: str,
        param_size: int,
        dataset_name: str,
        bond_types: list[str] | None = None,
        single_bond: str = "b_1",
        double_bond: str = "b_2",
        triple_bond: str = "b_3",
        aliphatic_bonds: list[str] | None = None,
        aromatic_bonds: list[str] | None = None,
    ):
        """Create a CustomDataset.

        Args:
            examples (str): Path to the examples file.
            queries (str): Path to the queries file.
            param_size (int): Parameter size for embeddings used by the dataset.
            dataset_name (str): Name of the dataset, used as a file prefix.
            bond_types (list[str] | None): Encoded bond predicates in the files.
            single_bond, double_bond, triple_bond (str): Bond-order predicates.
            aliphatic_bonds, aromatic_bonds (list[str] | None): Bond-role predicates.
        """
        self.examples = examples
        self.queries = queries

        atom_types = ["c", "o", "br", "i", "f", "h", "n", "cl", "s"]
        key_atoms = ["o", "s", "n"]
        bond_types = (
            ["b_1", "b_2", "b_3", "b_12"] if bond_types is None else bond_types
        )
        aliphatic_bonds = (
            [single_bond, double_bond, triple_bond]
            if aliphatic_bonds is None
            else aliphatic_bonds
        )
        aromatic_bonds = ["b_12"] if aromatic_bonds is None else aromatic_bonds

        super().__init__(
            dataset_name,
            "atom_embed",
            "bond_embed",
            "bond",
            atom_types,
            key_atoms,
            bond_types,
            single_bond,
            double_bond,
            triple_bond,
            aliphatic_bonds,
            aromatic_bonds,
            "c",
            "o",
            "h",
            "n",
            "s",
            ["f", "cl", "br", "i"],
            param_size,
        )

    def load_data(self):
        """Load dataset from provided file paths.

        If both paths are given they are validated and used to create a
        FileDataset. Otherwise falls back to the base class which expects
        packaged dataset files.

        Returns:
            neuralogic.dataset.FileDataset: The wrapped dataset.
        """
        if self.examples and self.queries:
            if not os.path.isfile(self.examples):
                raise FileNotFoundError(
                    f"Examples file not found at path: {self.examples}"
                )
            if not os.path.isfile(self.queries):
                raise FileNotFoundError(
                    f"Queries file not found at path: {self.queries}"
                )
            return FileDataset(
                examples_file=os.path.abspath(self.examples),
                queries_file=os.path.abspath(self.queries),
            )
        return super().load_data()
