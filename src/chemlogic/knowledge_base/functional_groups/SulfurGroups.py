from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class SulfurGroups(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "hydrogen",
        "carbon",
        "sulfur",
        "nitrogen",
    ]

    def create_template(self):
        # Defining thiocyanate group (R-S-C≡N)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_thiocyanate")(V.R)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.sulfur)(V.S),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_triple_bonded")(V.C, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.S, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.S, V.R, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.N, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.S, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.S, V.R, V.B3),
                    R.special.alldiff(V.C, V.N, V.S, V.R),
                )
            ]
        )

        # Defining isothiocyanate group (R-N=C=S)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_isothiocyanate")(V.R)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.sulfur)(V.S),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.S, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.N, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.S, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.N, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.R, V.B3),
                    R.special.alldiff(V.C, V.N, V.S, V.R),
                )
            ]
        )

        # Defining sulfide group (R1-S-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_sulfide")(V.R1, V.R2)
                <= (
                    R.get(self.carbon)(V.R1),
                    R.get(self.sulfur)(V.S),
                    R.get(self.carbon)(V.R2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.R1, V.S, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.S, V.R2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.R1, V.S, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.S, V.R2, V.B2),
                    R.special.alldiff(V.R1, V.R2, V.S),
                )
            ]
        )

        # Defining disulfide group (R1-S-S-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_disulfide")(V.C1, V.C2)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.sulfur)(V.S1),
                    R.get(self.sulfur)(V.S2),
                    R.get(self.carbon)(V.C2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C1, V.S1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.S1, V.S2, V.B12),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.S2, V.C2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.C1, V.S1, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.S2, V.C2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.S1, V.S2, V.B12),
                    R.special.alldiff(V.C1, V.C2, V.S1, V.S2),
                )
            ]
        )

        # Defining thiol group (R-S-H)
        # TODO: this won't work for datasets with implicit hydrogens
        self.add_rules(
            [
                R.get(f"{self.layer_name}_thiol")(V.C)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.sulfur)(V.S),
                    R.get(self.hydrogen)(V.H),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.S, V.B1),
                    R.get(f"{self.layer_name}_single_bonded")(V.S, V.H, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.S, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.S, V.H, V.B2),
                    R.special.alldiff(V.C, V.S, V.H),
                )
            ]
        )

        # Aggregating sulfuric groups
        self.add_rules(
            [
                R.get(f"{self.layer_name}_sulfuric_groups")(V.X)
                <= R.get(f"{self.layer_name}_isothiocyanate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_sulfuric_groups")(V.X)
                <= R.get(f"{self.layer_name}_thiocyanate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_sulfuric_groups")(V.X)
                <= R.get(f"{self.layer_name}_thiol")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_sulfuric_groups")(V.X)
                <= R.get(f"{self.layer_name}_sulfide")(V.X, V.Y)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_sulfuric_groups")(V.X)
                <= R.get(f"{self.layer_name}_disulfide")(V.X, V.Y)[self.param_size]
            ]
        )
