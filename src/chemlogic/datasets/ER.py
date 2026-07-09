from chemlogic.datasets.Dataset import Dataset


class ER(Dataset):
    """Estrogen receptor (ER) binding dataset.

    446 compounds labelled by binding affinity to the estrogen receptor alpha,
    relevant to endocrine disruption screening and hormone-related drug design.
    From the Sutherland QSAR benchmark alongside DHFR and COX.

    Atom types encoded as atom_0..atom_9 (10 atom types).
    Bond types: single (b_2), double (b_3), triple (b_4), aromatic (b_0), other (b_1).

    Source: https://chrsmrrs.github.io/datasets/docs/datasets/
    Reference:
        Sutherland et al. "Spline-fitting with a genetic algorithm: A method for
        developing classification structure-activity relationships."
        Journal of Chemical Information and Computer Sciences, 2003.
        https://doi.org/10.1021/ci034143r
    """

    def __init__(self, param_size):
        atom_types = [f"atom_{i}" for i in range(10)]
        key_atoms = ["atom_1", "atom_2", "atom_4", "atom_9"]
        bond_types = ["b_4", "b_2", "b_3", "b_0", "b_1"]

        super().__init__(
            "er",
            "atom_embed",
            "bond_embed",
            "bond",
            atom_types,
            key_atoms,
            bond_types,
            "b_2",
            "b_3",
            "b_4",
            ["b_2", "b_3", "b_4"],
            ["b_0"],
            "atom_0",
            "atom_1",
            "h",
            "atom_2",
            "atom_4",
            ["atom_3", "atom_5", "atom_6", "atom_8"],
            param_size,
        )
