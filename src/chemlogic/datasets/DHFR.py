from chemlogic.datasets.Dataset import Dataset


class DHFR(Dataset):
    """Dihydrofolate reductase (DHFR) inhibition dataset.

    393 compounds labelled by inhibitory activity against DHFR, an enzyme
    critical to nucleotide synthesis and a target in cancer and antibiotic
    drug design. From the Sutherland QSAR benchmark.

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
        key_atoms = ["atom_0", "atom_5", "atom_3"]
        bond_types = ["b_4", "b_2", "b_3", "b_0", "b_1"]

        super().__init__(
            "dhfr",
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
            "atom_1",
            "atom_3",
            "h",
            "atom_0",
            "atom_5",
            ["atom_2", "atom_4", "atom_6"],
            param_size,
        )
