import os

from neuralogic.dataset import FileDataset

from chemlogic.datasets.Dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, examples: str, queries: str, param_size: int, dataset_name: str):
        """
        Create a custom dataset for training.

        Args:
            examples (str): The path to the examples file.
            queries (str): The path to the queries file.
            param_size (int): The size of the parameter.
            dataset_name (str): The name of the dataset

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
