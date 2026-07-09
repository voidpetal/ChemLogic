from chemlogic.datasets.Dataset import Dataset


class PTC(Dataset):
    """Predictive Toxicology Challenge dataset — Male Rat (PTC-MR).

    344 chemical compounds labelled by carcinogenicity in male rats,
    from the US National Toxicology Program. Part of the PTC benchmark
    suite; the four variants (MR, FM, MM, FR) differ by the rodent sex
    and species used in the bioassay.

    Atom types encoded as atom_0..atom_17 (18 atom types).
    Bond types: single (b_2), double (b_1), triple (b_0), aromatic (b_3).

    Source: https://chrsmrrs.github.io/datasets/docs/datasets/
    Reference:
        Helma et al. "The Predictive Toxicology Challenge 2000-2001."
        Bioinformatics, 2001.
        https://doi.org/10.1093/bioinformatics/17.1.107
    """

    def __init__(self, param_size):
        atom_types = [f"atom_{i}" for i in range(18)]
        key_atoms = ["atom_1", "atom_2", "atom_3", "atom_7"]
        bond_types = ["b_1", "b_2", "b_3", "b_0"]

        super().__init__(
            "ptc",
            "atom_embed",
            "bond_embed",
            "bond",
            atom_types,
            key_atoms,
            bond_types,
            "b_2",
            "b_1",
            "b_0",
            ["b_0", "b_1", "b_2"],
            ["b_3"],
            "atom_5",
            "atom_2",
            "h",
            "atom_3",
            "atom_7",
            ["atom_9", "atom_6", "atom_8", "atom_13"],
            param_size,
        )
