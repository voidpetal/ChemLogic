import os
from pathlib import Path

from neuralogic.core import R, V
from neuralogic.dataset import FileDataset

from chemlogic.utils.ChemTemplate import ChemTemplate as Template


class Dataset(Template):
    """Base class for all ChemLogic datasets.

    Stores atom/bond type metadata, loads the underlying FileDataset via
    `load_data`, and creates embedding rules via `create_template`. Concrete
    subclasses should override `load_data` when the data source is not the
    packaged `data/datasets/` folder.
    """

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
        """Create a dataset configuration.

        Args:
            dataset_name (str): Dataset identifier, also used as the folder name
                under `data/datasets/`.
            node_embed (str): Predicate name for node embeddings.
            edge_embed (str): Predicate name for edge embeddings.
            connection (str): Predicate name for the bond connectivity relation.
            atom_types (list[str]): All atom-type predicate names in this dataset.
            key_atom_type (list[str]): Subset of atom types treated as key atoms
                (heteroatoms) by the chemical rules.
            bond_types (list[str]): All bond-type predicate names.
            single_bond (str): Predicate name for single bonds.
            double_bond (str): Predicate name for double bonds.
            triple_bond (str): Predicate name for triple bonds.
            aliphatic_bonds (list[str]): Predicate names for aliphatic bond types.
            aromatic_bonds (list[str]): Predicate names for aromatic bond types.
            carbon (str): Predicate name for carbon atoms. Default "c".
            oxygen (str): Predicate name for oxygen atoms. Default "o".
            hydrogen (str): Predicate name for hydrogen atoms. Default "h".
            nitrogen (str): Predicate name for nitrogen atoms. Default "n".
            sulfur (str): Predicate name for sulfur atoms. Default "s".
            halogens (list[str]): Predicate names for halogen atoms. Defaults to
                ["f", "cl", "br", "i"].
            param_size (int): Embedding dimension. Must be a positive integer.
        """
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
        """Load dataset files from the packaged `data/datasets/<name>` folder.

        Expects `examples.txt` and `queries.txt` under
        `src/chemlogic/data/datasets/<dataset_name>/`. Override this in
        subclasses that load data from other sources.

        Returns:
            neuralogic.dataset.FileDataset: The wrapped dataset.
        """
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
        """Create embedding rules mapping atom/bond types to embedding predicates.

        Called automatically from `__init__`. Does not need to be invoked manually.
        """
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
