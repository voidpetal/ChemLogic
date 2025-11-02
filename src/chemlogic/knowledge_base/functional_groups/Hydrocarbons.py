from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class Hydrocarbons(KnowledgeBase):
    required_keys = ["layer_name", "param_size", "carbon"]

    def create_template(self):
        # Defining the benzene ring
        self.add_rules(
            [
                R.get(f"{self.layer_name}_benzene_ring")(V.A)
                <= R.get(f"{self.layer_name}_benzene_ring")(V.A, V.B)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_benzene_ring")(V.A, V.B)
                <= R.get(f"{self.layer_name}_benzene_ring")(
                    V.A, V.B, V.C, V.D, V.E, V.F
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_benzene_ring")(V.A, V.B, V.C, V.D, V.E, V.F)
                <= (
                    R.get(f"{self.layer_name}_aromatic_bonded")(V.A, V.B, V.B1),
                    R.get(f"{self.layer_name}_aromatic_bonded")(V.B, V.C, V.B2),
                    R.get(f"{self.layer_name}_aromatic_bonded")(V.C, V.D, V.B3),
                    R.get(f"{self.layer_name}_aromatic_bonded")(V.D, V.E, V.B4),
                    R.get(f"{self.layer_name}_aromatic_bonded")(V.E, V.F, V.B5),
                    R.get(f"{self.layer_name}_aromatic_bonded")(V.F, V.A, V.B6),
                    R.get(self.carbon)(V.A),
                    R.get(self.carbon)(V.B),
                    R.get(self.carbon)(V.C),
                    R.get(self.carbon)(V.D),
                    R.get(self.carbon)(V.E),
                    R.get(self.carbon)(V.F),
                    R.get(f"{self.layer_name}_bond_message")(V.A, V.B, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B, V.C, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.D, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.D, V.E, V.B4),
                    R.get(f"{self.layer_name}_bond_message")(V.E, V.F, V.B5),
                    R.get(f"{self.layer_name}_bond_message")(V.F, V.A, V.B6),
                    R.special.alldiff(V.A, V.B, V.C, V.D, V.E, V.F),
                )
            ]
        )

        # Defining an alkene (R-C=C-R), alkyne group (R-C≡C-R)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_alkene_bond")(V.C1, V.C2)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.carbon)(V.C2),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C1, V.C2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.C1, V.C2, V.B),
                )
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_alkyne_bond")(V.C1, V.C2)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.carbon)(V.C2),
                    R.hidden.get(f"{self.layer_name}_triple_bonded")(V.C1, V.C2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.C1, V.C2, V.B),
                )
            ]
        )

        # Aggregating hydrocarbon groups
        self.add_rules(
            [
                R.get(f"{self.layer_name}_hydrocarbon_groups")(V.C1)
                <= R.get(f"{self.layer_name}_alkene_bond")(V.C1, V.C2)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_hydrocarbon_groups")(V.C1)
                <= R.get(f"{self.layer_name}_alkyne_bond")(V.C1, V.C2)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_hydrocarbon_groups")(V.A)
                <= R.get(f"{self.layer_name}_benzene_ring")(V.A)[self.param_size]
            ]
        )
