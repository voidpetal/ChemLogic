from chemlogic.datasets.Dataset import Dataset


class COX(Dataset):
    """COX-2 inhibition dataset.

    303 compounds labelled by inhibitory activity against cyclooxygenase-2
    (COX-2), a key enzyme in inflammation and a target for anti-inflammatory
    drugs. From the Sutherland QSAR benchmark alongside DHFR and ER.

    Atom types encoded as atom_0..atom_6 (7 atom types).
    Bond types: single (b_2), double (b_3), triple (b_4), aromatic (b_0), other (b_1).

    Source: https://chrsmrrs.github.io/datasets/docs/datasets/
    Reference:
        Sutherland et al. "Spline-fitting with a genetic algorithm: A method for
        developing classification structure-activity relationships."
        Journal of Chemical Information and Computer Sciences, 2003.
        https://doi.org/10.1021/ci034143r
    """

    def __init__(self, param_size):
        atom_types = [f"atom_{i}" for i in range(7)]
        key_atoms = ["atom_1", "atom_4", "atom_3"]
        bond_types = ["b_4", "b_2", "b_3", "b_0", "b_1"]

        super().__init__(
            "cox",
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
            "atom_4",
            "h",
            "atom_1",
            "atom_3",
            ["atom_2", "atom_5", "atom_6"],
            param_size,
        )
