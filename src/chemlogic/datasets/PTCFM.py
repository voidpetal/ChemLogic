from chemlogic.datasets.Dataset import Dataset


class PTCFM(Dataset):
    """Predictive Toxicology Challenge dataset — Female Mouse (PTC-FM).

    349 compounds labelled by carcinogenicity in female mice. See PTC for
    full reference.

    Source: https://chrsmrrs.github.io/datasets/docs/datasets/
    Reference:
        Helma et al. "The Predictive Toxicology Challenge 2000-2001."
        Bioinformatics, 2001.
        https://doi.org/10.1093/bioinformatics/17.1.107
    """

    def __init__(self, param_size):
        atom_types = [f"atom_{i}" for i in range(18)]
        key_atoms = ["atom_1", "atom_4", "atom_3", "atom_6"]
        bond_types = ["b_1", "b_2", "b_3", "b_0"]

        super().__init__(
            "ptc_fm",
            "atom_embed",
            "bond_embed",
            "bond",
            atom_types,
            key_atoms,
            bond_types,
            "b_1",
            "b_1",
            "b_0",
            ["b_0", "b_1", "b_2"],
            ["b_3"],
            "atom_2",
            "atom_3",
            "h",
            "atom_4",
            "atom_6",
            ["atom_9", "atom_5", "atom_7", "atom_13"],
            param_size,
        )
