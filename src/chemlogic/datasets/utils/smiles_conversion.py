import os
import re

import networkx
import networkx as nx
from neuralogic.dataset import Data, FileDataset, TensorDataset
from pysmiles import read_smiles
from rdkit import Chem
from rdkit.Chem import AddHs, GetPeriodicTable, MolFromSmiles
from torch_geometric.utils import from_networkx

# Uranium is the heaviest naturally occurring element.
# Beyond that, elements are typically synthesized in laboratories and have short half-lives
# We will use the first 92 elements in the periodic table as possible atom types
# Alternative: use all 118 elements
MAX_ATOM_TYPES = 92
MAX_EDGE_TYPES = len(Chem.rdchem.BondType.values)

# RDKit atom feature extractors
# Each lambda takes an RDKit Atom object and returns a numeric value
ATOM_FEATURE_MAP = {
    "formal_charge": lambda a: a.GetFormalCharge(),
    "num_radical_electrons": lambda a: a.GetNumRadicalElectrons(),
    "is_aromatic": lambda a: int(a.GetIsAromatic()),
    "hybridization": lambda a: int(a.GetHybridization()),
    "total_num_hs": lambda a: a.GetTotalNumHs(),
    "degree": lambda a: a.GetDegree(),
    "is_in_ring": lambda a: int(a.IsInRing()),
    "chiral_tag": lambda a: int(a.GetChiralTag()),
}

# RDKit bond feature extractors
# Each lambda takes an RDKit Bond object and returns a numeric value
BOND_FEATURE_MAP = {
    "is_aromatic": lambda b: int(b.GetIsAromatic()),
    "is_conjugated": lambda b: int(b.GetIsConjugated()),
    "is_in_ring": lambda b: int(b.IsInRing()),
    "stereo": lambda b: int(b.GetStereo()),
}


def smiles_to_pyg(
    smiles: str, explicit_hydrogens=True, atom_features=None, bond_features=None
):
    """
    Converts a SMILES string to a PyTorch Geometric (PyG) graph.

    Args:
        smiles (str): The SMILES representation of the molecule.
        explicit_hydrogens (bool): Add explicit hydrogens, default True
        atom_features (str | list[str] | None): Atom features to extract.
            - None: No additional features (default, backward compatible)
            - 'all': Extract all available features from ATOM_FEATURE_MAP
            - list[str]: Extract only specified features
        bond_features (str | list[str] | None): Bond features to extract.
            - None: No additional features (default, backward compatible)
            - 'all': Extract all available features from BOND_FEATURE_MAP
            - list[str]: Extract only specified features

    Returns:
        torch_geometric.data.Data: A PyG graph representing the molecule, with atom and bond attributes.

    """
    mol = MolFromSmiles(smiles)

    if explicit_hydrogens:
        mol = AddHs(mol)

    # Resolve atom_features to list of feature names
    if atom_features == "all":
        atom_feature_names = list(ATOM_FEATURE_MAP.keys())
    elif atom_features is not None:
        for f in atom_features:
            if f not in ATOM_FEATURE_MAP:
                raise ValueError(
                    f"Unknown atom feature: '{f}'. "
                    f"Available: {list(ATOM_FEATURE_MAP.keys())}"
                )
        atom_feature_names = atom_features
    else:
        atom_feature_names = []

    # Resolve bond_features to list of feature names
    if bond_features == "all":
        bond_feature_names = list(BOND_FEATURE_MAP.keys())
    elif bond_features is not None:
        for f in bond_features:
            if f not in BOND_FEATURE_MAP:
                raise ValueError(
                    f"Unknown bond feature: '{f}'. "
                    f"Available: {list(BOND_FEATURE_MAP.keys())}"
                )
        bond_feature_names = bond_features
    else:
        bond_feature_names = []

    # Convert mol to a NetworkX graph
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        x = [
            0,
        ] * MAX_ATOM_TYPES
        x[atom.GetAtomicNum()] = 1  # Atomic numbers start from 0 (0 for Unknown)

        # Build node data with x and any requested atom features
        node_data = {"x": x}
        for feat_name in atom_feature_names:
            node_data[feat_name] = ATOM_FEATURE_MAP[feat_name](atom)

        graph.add_node(atom.GetIdx(), **node_data)

    for bond in mol.GetBonds():
        edge_attr = [
            0,
        ] * MAX_EDGE_TYPES
        for k, i in Chem.rdchem.BondType.values.items():
            if i == bond.GetBondType():
                edge_attr[k] = 1

        # Build edge data with edge_attr and any requested bond features
        edge_data = {"edge_attr": edge_attr}
        for feat_name in bond_feature_names:
            edge_data[feat_name] = BOND_FEATURE_MAP[feat_name](bond)

        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            **edge_data,
        )

    # Add atom and bond IDs to the graph
    for node in graph.nodes:
        graph.nodes[node]["atom_id"] = node
    for edge in graph.edges:
        graph.edges[edge]["bond_id"] = mol.GetBondBetweenAtoms(
            edge[0], edge[1]
        ).GetIdx()

    return from_networkx(graph)


def add_hydrogens(smiles: str):
    """Add explicit hydrogens to SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    mol_with_hydrogens = AddHs(mol)
    smiles_with_hydrogens = Chem.MolToSmiles(mol_with_hydrogens)
    return smiles_with_hydrogens


def smiles_to_neuralogic(smiles: str):
    """ "Convert SMILES to neuralogic tensor (through pygeometric)"""
    return Data.from_pyg(smiles_to_pyg(smiles))[0]


def smiles_to_networkx(smiles: str):
    """Convert SMILES to networkx network"""
    # smiles_with_hydrogens = add_hydrogens(smiles)
    mol = read_smiles(smiles, explicit_hydrogen=True)
    return mol


def get_atom_mapping(graph: networkx.classes.graph.Graph):
    """Get atom_id<->element mapping for networkx graph"""
    return dict(graph.nodes(data="element"))


def networkx_to_neuralogic(mol):
    """Convert networkx graph to neuralogic tensor (though pygeometric)"""
    pyg_graph = from_networkx(mol)
    return Data.from_pyg(pyg_graph)[0]


def update_predicate(pred, bond_mappings):
    # change bond from `bond_x(a1, a2)` to `bond(a1, a2, B), b_x(B)`
    if pred.startswith("bond"):
        first = int(pred.split("(")[1].split(",")[0])
        second = int(pred.split(",")[1].split(")")[0])
        B = bond_mappings[(first, second)]
        order = int(pred.split("(")[0].split("_")[1])
        new_pred = (
            f"<1> bond({pred.split('(')[1].split(')')[0]}, {B}), <1> b_{order}({B})"
        )
    # change atom from atom_0(0) to the mapping, for example to C(0)
    elif pred.startswith("atom"):
        atomic_number = int(pred.split("(")[0].split("_")[1])
        pt = GetPeriodicTable()
        e_id = pt.GetElementSymbol(atomic_number).lower()
        atom_id = pred.split("(")[1].split(")")[0]
        new_pred = f"<1> {e_id}({atom_id})"
    else:
        new_pred = pred
    return new_pred


def create_queries_file(labels, file_name):
    """ "Manually create the *_queries.txt file using list of labels"""
    with open(file_name, "w") as f:
        for label in labels:
            f.write(f"{label} predict.\n")


def get_unique_atoms_and_bonds(file_path):
    atom_pattern = re.compile(r"<1>\s+([A-Za-z]{1,3})\(\d+\)(?!\s*,?\s*b_\d+\()")
    bond_pattern = re.compile(r"<1>\s+(b_\d+)\(\d+\)")

    unique_atoms = set()
    unique_bonds = set()

    with open(file_path) as f:
        for line in f:
            atoms = atom_pattern.findall(line)
            atoms = [a for a in atom_pattern.findall(line) if not a.startswith("b_")]

            bonds = bond_pattern.findall(line)

            unique_atoms.update(atoms)
            unique_bonds.update(bonds)

    return list(unique_atoms), list(unique_bonds)


def get_dataset_and_mappings(
    smiles_list,
    labels=None,
    file_prefix="",
    output_location=".",
    graph_features=None,
    atom_features=None,
    bond_features=None,
    broadcast_graph_features=False,
):
    """Create the neuralogic dataset from list of smiles and also dump it as text files.

    Args:
        smiles_list: List of SMILES strings.
        labels: List of labels (optional).
        file_prefix: Prefix for output files.
        output_location: Directory for output files.
        graph_features: Dict mapping feature names to lists of values per molecule.
            Example: {'mol_weight': [46.07, 30.07]} for 2 molecules.
            These become valued predicates: "<46.07> mol_weight" in the examples file.
        atom_features: Atom features to extract as node-level predicates.
            - None: No additional features (default, backward compatible)
            - 'all': Extract all available features from ATOM_FEATURE_MAP
            - list[str]: Extract only specified features
            These become valued predicates: "<value> feature_name(atom_id)" in examples.
        bond_features: Bond features to extract as edge-level predicates.
            - None: No additional features (default, backward compatible)
            - 'all': Extract all available features from BOND_FEATURE_MAP
            - list[str]: Extract only specified features
            These become valued predicates: "<value> feature_name(bond_id)" in examples.
        broadcast_graph_features: If True, add graph features to ALL atoms instead of
            a synthetic graph node. This broadcasts the same graph-level values to
            every atom, avoiding synthetic node connectivity issues.
    """
    assert len(smiles_list) == len(labels) if labels is not None else True

    # Resolve atom_features to list of feature names for predicate generation
    if atom_features == "all":
        atom_feature_names = list(ATOM_FEATURE_MAP.keys())
    elif atom_features:
        atom_feature_names = atom_features
    else:
        atom_feature_names = []

    # Resolve bond_features to list of feature names for predicate generation
    if bond_features == "all":
        bond_feature_names = list(BOND_FEATURE_MAP.keys())
    elif bond_features:
        bond_feature_names = bond_features
    else:
        bond_feature_names = []

    # convert the SMILES list to list of neuralogic tensors
    # Pass atom_features and bond_features to smiles_to_pyg so features are stored in PyG graph
    pyg_graphs = [
        smiles_to_pyg(smile, atom_features=atom_features, bond_features=bond_features)
        for smile in smiles_list
    ]
    graphs = [Data.from_pyg(g)[0] for g in pyg_graphs]

    # Add label if available - this behaved bit weird in generation of queries.txt -> "[1.0] predict." instead of "1 predict." so generating this file manually instead

    if labels is not None:
        for graph, label in zip(graphs, labels, strict=False):
            if isinstance(label, (int, float)):
                graph.y = label
            elif isinstance(label, list):
                graph.y = label[0]

    # Get the atom-element mappings - possible to get from networkx so convert from SMILES to networkx
    # networkx_graphs = [smiles_to_networkx(smile) for smile in smiles_list]
    # element_mappings = [get_atom_mapping(graph) for graph in networkx_graphs]
    # Bond mapping: [{(in_node, out_node): bond_id, ...} for G in graphs]
    bond_mappings = [
        {
            (int(g.edge_index[0][i]), int(g.edge_index[1][i])): int(id) + g.x.size(0)
            for i, id in enumerate(g.bond_id)
        }
        for g in pyg_graphs
    ]

    # create the dataset
    dataset = TensorDataset(
        graphs,
        one_hot_encode_labels=False,
        one_hot_decode_edge_features=True,
        one_hot_decode_features=True,
    )
    dataset.edge_name = "bond"
    dataset.feature_name = "atom"

    # dump the dataset to text file
    queries_fp = f"{output_location}/{file_prefix}_queries.txt"
    examples_fp = f"{output_location}/{file_prefix}_examples_bad.txt"  # this file does not have the desired structure, it is used to create the correct formating but can be deleted
    with open(queries_fp, "w") as q_file, open(examples_fp, "w") as e_file:
        dataset.dump(q_file, e_file)

    # update the dataset to have the desired format
    examples_updated_fp = f"{output_location}/{file_prefix}_examples.txt"
    with (
        open(examples_fp) as in_handle,
        open(examples_updated_fp, "w") as out_handle,
    ):
        for idx, (line, mapping) in enumerate(
            zip(in_handle.readlines(), bond_mappings, strict=False)
        ):
            # get the predicates
            predicates = [
                p.strip(" ,\n.") for p in line.split("<1>") if p.strip(" ,\n") != ""
            ]
            # update the predicates
            new_predicates = []
            for predicate in predicates:
                new_predicates.append(update_predicate(predicate, mapping))

            # Add graph-level features
            # Two modes:
            # 1. Synthetic node (default): create a graph node connected to all atoms
            # 2. Broadcast mode: add graph features directly to all atoms
            if graph_features:
                pyg_graph = pyg_graphs[idx]
                num_atoms = pyg_graph.x.size(0)

                if broadcast_graph_features:
                    # BROADCAST MODE: Add graph features to every atom
                    # This avoids synthetic node connectivity issues
                    for feat_name, feat_values in graph_features.items():
                        value = feat_values[idx]
                        if value != 0:
                            for atom_id in range(num_atoms):
                                new_predicates.append(
                                    f"<{value}> {feat_name}({atom_id})"
                                )
                else:
                    # SYNTHETIC NODE MODE (original): Create graph node connected to all atoms
                    num_bonds = len(mapping) // 2  # Each bond appears twice in mapping
                    # Graph node ID is after all atoms and bonds
                    graph_node_id = num_atoms + num_bonds

                    # Add the graph node marker (no atom type - won't trigger KB rules)
                    new_predicates.append(f"<1> graph_node({graph_node_id})")

                    # Connect graph node to all atoms via graph_bond
                    # Each connection gets a unique bond ID starting after graph_node_id
                    # Using a dedicated bond type keeps it separate from molecular bonds
                    for i, atom_id in enumerate(range(num_atoms)):
                        graph_bond_id = graph_node_id + 1 + i
                        new_predicates.append(
                            f"<1> bond({atom_id}, {graph_node_id}, {graph_bond_id}), "
                            f"<1> graph_bond({graph_bond_id})"
                        )

                    # Add graph features as attributes on the graph node
                    # Skip 0 values - they contribute nothing to the prediction
                    for feat_name, feat_values in graph_features.items():
                        value = feat_values[idx]
                        if value != 0:
                            new_predicates.append(
                                f"<{value}> {feat_name}({graph_node_id})"
                            )

            # Add node-level (atom) features as valued predicates: "<value> feature_name(atom_id)"
            # Skip 0 values - they contribute nothing to the prediction
            if atom_feature_names:
                pyg_graph = pyg_graphs[idx]
                num_atoms = pyg_graph.x.size(0)
                for atom_id in range(num_atoms):
                    for feat_name in atom_feature_names:
                        # Get the feature value for this atom
                        # Features are stored as tensors in PyG Data object
                        if hasattr(pyg_graph, feat_name):
                            feat_tensor = getattr(pyg_graph, feat_name)
                            value = (
                                feat_tensor[atom_id].item()
                                if hasattr(feat_tensor[atom_id], "item")
                                else feat_tensor[atom_id]
                            )
                            if value != 0:
                                new_predicates.append(
                                    f"<{value}> {feat_name}({atom_id})"
                                )

            # Add edge-level (bond) features as valued predicates: "<value> feature_name(bond_id)"
            # Skip 0 values - they contribute nothing to the prediction
            if bond_feature_names:
                # Re-create mol to iterate over bonds (need original bond objects for feature extraction)
                mol = MolFromSmiles(smiles_list[idx])
                mol = AddHs(
                    mol
                )  # Match the explicit_hydrogens=True default in smiles_to_pyg

                for bond in mol.GetBonds():
                    # Get the offset bond_id from bond_mappings
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    bond_id = mapping.get((begin_idx, end_idx))
                    if bond_id is None:
                        bond_id = mapping.get((end_idx, begin_idx))

                    if bond_id is not None:
                        for feat_name in bond_feature_names:
                            value = BOND_FEATURE_MAP[feat_name](bond)
                            if value != 0:
                                new_predicates.append(
                                    f"<{value}> {feat_name}({bond_id})"
                                )

            out_handle.write(",".join(new_predicates) + ".\n")

    # Delete the bad examples file
    os.remove(examples_fp)

    # load the dataset from the text file
    dataset = FileDataset(examples_file=examples_updated_fp, queries_file=queries_fp)

    # Get atom/bond types from file
    unique_atoms, unique_bonds = get_unique_atoms_and_bonds(examples_updated_fp)

    # Get graph feature names (keys from graph_features dict)
    graph_feature_names = list(graph_features.keys()) if graph_features else []

    return dataset, (
        unique_atoms,
        unique_bonds,
        atom_feature_names,
        bond_feature_names,
        graph_feature_names,
    )
