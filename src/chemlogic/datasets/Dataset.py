import os
from pathlib import Path

from neuralogic.core import R, V
from neuralogic.dataset import FileDataset

from chemlogic.utils.ChemTemplate import ChemTemplate as Template


class Dataset(Template):
    def __init__(
        self,
        dataset_name: str,
        node_embed: str,
        edge_embed: str,
        connection: str,
        atom_types: list,
        key_atom_type: list,
        bond_types: list,
        single_bond: str,
        double_bond: str,
        triple_bond: str,
        aliphatic_bonds: list,
        aromatic_bonds: list,
        carbon: str = "c",
        oxygen: str = "o",
        hydrogen: str = "h",
        nitrogen: str = "n",
        sulfur: str = "s",
        halogens: list = None,
        param_size: int = 1,
    ):
        super().__init__()
        # Validate string inputs
        for name, value in {
            "dataset_name": dataset_name,
            "node_embed": node_embed,
            "edge_embed": edge_embed,
            "connection": connection,
            "carbon": carbon,
            "oxygen": oxygen,
            "hydrogen": hydrogen,
            "nitrogen": nitrogen,
            "sulfur": sulfur,
            "single_bond": single_bond,
            "double_bond": double_bond,
            "triple_bond": triple_bond,
        }.items():
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string.")

        # Validate list inputs
        if not isinstance(key_atom_type, list) or not all(
            isinstance(x, str) for x in key_atom_type
        ):
            raise TypeError("key_atom_type must be a list of strings.")
        if not isinstance(atom_types, list) or not all(
            isinstance(x, str) for x in atom_types
        ):
            raise TypeError("atom_types must be a list of strings.")
        if not isinstance(bond_types, list) or not all(
            isinstance(x, str) for x in bond_types
        ):
            raise TypeError("bond_types must be a list of strings.")
        if not isinstance(aliphatic_bonds, list) or not all(
            isinstance(x, str) for x in aliphatic_bonds
        ):
            raise TypeError("aliphatic_bonds must be a list of strings.")
        if not isinstance(aromatic_bonds, list) or not all(
            isinstance(x, str) for x in aromatic_bonds
        ):
            raise TypeError("aromatic_bonds must be a list of strings.")

        halogens = halogens if halogens is not None else ["f", "cl", "br", "i"]
        if not isinstance(halogens, list) or not all(
            isinstance(x, str) for x in halogens
        ):
            raise TypeError("halogens must be a list of string.")

        # Validate param_size
        if not isinstance(param_size, int) or param_size < 1:
            raise ValueError("param_size must be a positive integer.")

        # Assign values
        self.dataset_name = dataset_name
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.connection = connection
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.key_atom_type = key_atom_type
        self.single_bond = single_bond
        self.double_bond = double_bond
        self.triple_bond = triple_bond
        self.aliphatic_bonds = aliphatic_bonds
        self.aromatic_bonds = aromatic_bonds
        self.carbon = carbon
        self.oxygen = oxygen
        self.hydrogen = hydrogen
        self.nitrogen = nitrogen
        self.sulfur = sulfur
        self.halogens = halogens
        self.param_size = param_size

        self.data = self.load_data()
        self.create_template()

    def load_data(self):
        # Get the path to the current file and navigate to the package root
        src_dir = Path(__file__).resolve().parent.parent
        dataset_path = src_dir / "data" / "datasets" / self.dataset_name
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"The directory '{dataset_path}' does not exist.")

        return FileDataset(
            examples_file=os.path.abspath(f"{dataset_path}/examples.txt"),
            queries_file=os.path.abspath(f"{dataset_path}/queries.txt"),
        )

    def create_template(self):
        self.add_rules(
            [
                (R.get(self.node_embed)(V.A)[self.param_size,] <= R.get(atom)(V.A))
                for atom in self.atom_types
            ]
        )

        self.add_rules(
            [
                (R.get(self.edge_embed)(V.B)[self.param_size,] <= R.get(bond)(V.B))
                for bond in self.bond_types
            ]
        )
