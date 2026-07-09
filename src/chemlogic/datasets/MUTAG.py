from neuralogic.utils.data import Mutagenesis

from chemlogic.datasets.Dataset import Dataset


# TODO: try the MUTAG from TUD datasets (different bond types and implicit hydrogens)
class MUTAG(Dataset):
    """Mutagenicity dataset (MUTAG / Mutagenesis).

    188 aromatic and heteroaromatic nitro compounds labelled by mutagenic
    activity on Salmonella typhimurium. One of the oldest and most widely
    used benchmarks in graph classification and ILP.

    Atom types: C, O, Br, I, F, H, N, Cl.
    Bond types: single, double, triple, aromatic, and two additional types.

    Note: this variant is loaded from PyNeuraLogic's bundled Mutagenesis
    dataset, which uses explicit hydrogens and a slightly different encoding
    from the TUD MUTAG graph. See the TODO below.

    Source: https://chrsmrrs.github.io/datasets/docs/datasets/
    Reference:
        Debnath et al. "Structure-activity relationship of mutagenic aromatic
        and heteroaromatic nitro compounds."
        Journal of Medicinal Chemistry, 1991.
        https://doi.org/10.1021/jm00106a046
    """

    def __init__(self, param_size):
        atom_types = ["c", "o", "br", "i", "f", "h", "n", "cl"]
        key_atoms = ["o", "s", "n"]
        bond_types = ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"]

        super().__init__(
            "mutag",
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
        _, dataset = Mutagenesis()
        return dataset
