from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class OxygenGroups(KnowledgeBase):
    required_keys = ["layer_name", "param_size", "carbon", "oxygen", "hydrogen"]

    def create_template(self):
        # Defining an alcoholic group (R-O-H)
        # TODO: this won't work for datasets with implicit hydrogens
        self.add_rules(
            [
                R.get(f"{self.layer_name}_alcoholic")(V.C)
                <= (
                    R.get(f"{self.layer_name}_saturated")(V.C),
                    R.get(f"{self.layer_name}_hydroxyl")(V.O),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.O, V.B1),
                )
            ]
        )

        # Defining a ketone (R1-C(=O)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ketone")(V.C)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.R1, V.R2),
                    R.get(self.carbon)(V.R1),
                    R.get(self.carbon)(V.R2),
                )
            ]
        )

        # Defining an aldehyde (R-C(=O)-H)
        # TODO: this won't work for datasets with implicit hydrogens
        self.add_rules(
            [
                R.get(f"{self.layer_name}_aldehyde")(V.C)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.R, V.H),
                    R.get(self.carbon)(V.R),
                    R.get(self.hydrogen)(V.H),
                )
            ]
        )

        # Defining acyl halide group (R-C(=O)-X)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_acyl_halide")(V.C)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.R, V.X),
                    R.get(self.carbon)(V.R),
                    R.get(f"{self.layer_name}_halogen")(V.X),
                )
            ]
        )

        # Defining carboxylic acid (R-C(=O)-OH)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carboxylic_acid")(V.C)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.R, V.O1),
                    R.get(self.carbon)(V.R),
                    R.get(f"{self.layer_name}_hydroxyl")(V.O1),
                )
            ]
        )

        # Defining carboxylic acid anhydride (R1-C(=O)-O-C(=O)-R2)
        # TODO: should this be propagated on C or on R?
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carboxylic_acid_anhydride")(V.C1, V.C2)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C1, V.X1, V.O12, V.R1),
                    R.get(self.oxygen)(V.O12),
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C2, V.X2, V.O12, V.R2),
                    R.special.alldiff(V.C1, V.C2),
                )
            ]
        )

        # Defining an ester group (R1-C(=O)-O-R2)
        # TODO: will fail for HC(=O)-O-CH
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ester")(V.X)
                <= R.get(f"{self.layer_name}_ester")(V.X, V.Y)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ester")(V.R1, V.R2)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.X, V.R1, V.O),
                    R.get(self.carbon)(V.R1),
                    R.get(self.oxygen)(V.O),
                    R.get(self.carbon)(V.R2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.R2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.O, V.R2, V.B),
                )
            ]
        )

        # Defining carbonate ester group (R1-O-C(=O)-O-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonate_ester")(V.X)
                <= R.get(f"{self.layer_name}_carbonate_ester")(V.X, V.Y)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonate_ester")(V.R1, V.R2)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.X, V.O1, V.O2),
                    R.get(self.oxygen)(V.O1),
                    R.get(self.oxygen)(V.O2),
                    R.get(self.carbon)(V.R1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.R1, V.O1, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.O1, V.R1, V.B1),
                    R.get(self.carbon)(V.R2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.R2, V.O2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.O2, V.R2, V.B2),
                )
            ]
        )

        # Defining the ether group (R1-O-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ether")(V.X)
                <= R.get(f"{self.layer_name}_ether")(V.X, V.Y)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ether")(V.C, V.R)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.oxygen)(V.O),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.R, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.O, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.R, V.O, V.B2),
                    R.special.alldiff(V.C, V.R, V.O),
                )
            ]
        )

        # Aggregating oxygen patterns
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_alcoholic")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_ketone")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_aldehyde")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_acyl_halide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_carboxylic_acid")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_carboxylic_acid_anhydride")(V.X, V.Y)[
                    self.param_size
                ]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_ester")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_carbonate_ester")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_ether")(V.X)[self.param_size]
            ]
        )
