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

    def __init__(self, examples: str, queries: str, param_size: int, dataset_name: str):
        """Create a CustomDataset.

        Args:
            examples (str): Path to the examples file.
            queries (str): Path to the queries file.
            param_size (int): Parameter size for embeddings used by the dataset.
            dataset_name (str): Name of the dataset, used as a file prefix.
        """
        self.examples = examples
        self.queries = queries

        atom_types = ["c", "o", "br", "i", "f", "h", "n", "cl", "s"]
        key_atoms = ["o", "s", "n"]
        bond_types = ["b_1", "b_2", "b_3", "b_4"]

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
            ["b_4", "b_5", "b_6", "b_7"],
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
