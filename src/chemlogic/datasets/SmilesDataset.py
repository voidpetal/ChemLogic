import logging
import os

from chemlogic.datasets.Dataset import Dataset
from chemlogic.datasets.utils.smiles_conversion import get_dataset_and_mappings


class SmilesDataset(Dataset):
    def __init__(
        self,
        smiles_list: list[str],
        labels: list[int],
        param_size: int,
        dataset_name: str,
        output_directory: str = ".",
        keep: bool = False,
    ):
        """
        Create a custom dataset from SMILES.

        Args:
            smiles_list (list[str]): A list of SMILES strings.
            labels (list[int]): A list of labels.
            param_size (int): The size of the parameter.
            dataset_name (str): The name of the dataset
            output_directory (Optional[str]): The output directory where to dump the dataset. Leave blank if one-time import.
            keep (Optional[bool]): Whether to keep the created files or not.
        """

        if len(smiles_list) != len(labels):
            raise ValueError(
                "The params smiles_list and labels must be of same length!"
            )

        self.smiles_list = smiles_list
        self.labels = labels

        self.output_directory = output_directory
        self.keep = keep

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
        dataset, (atom_types, bond_types) = get_dataset_and_mappings(
            smiles_list=self.smiles_list,
            labels=self.labels,
            output_location=self.output_directory,
            file_prefix=self.dataset_name,
        )
        self.atom_types = atom_types
        self.bond_types = bond_types
        return dataset

    # TODO: define a dump function
    # TODO: convert to a Dataset, not FileDataset

    def clear(self):
        for file in ["examples", "queries"]:
            file_path = f"{self.output_directory}/{self.dataset_name}_{file}.txt"
            try:
                os.remove(file_path)
            except Exception:
                logging.info(f"Unable to delete {file} file at {file_path}")
