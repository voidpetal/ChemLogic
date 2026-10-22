from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class OxygenGroups(KnowledgeBase):
    required_keys = ["layer_name", "param_size", "carbon", "oxygen", "hydrogen"]

    def create_template(self):
        # Defining an alcoholic group (R-O-H)
        # TODO: this won't work for datasets with implicit hydrogens
        self.add_rules(
            [
                R.get(f"{self.layer_name}_alcoholic")(V.O)
                <= (
                    R.get(f"{self.layer_name}_saturated")(V.C),
                    R.get(f"{self.layer_name}_hydroxyl")(V.O),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
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
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_ester_core")(V.C, V.O, V.R1, V.R2)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.X, V.R1, V.O),
                    R.get(self.oxygen)(V.O),
                    R.get(self.carbon)(V.R2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.R2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.C, V.R1, V.O, V.R2),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ester")(V.C)
                <= (
                    R.hidden.get(f"{self.layer_name}_ester_core")(V.C, V.O, V.R1, V.R2),
                    R.get(self.carbon)(V.R1),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ester")(V.C)
                <= (
                    R.hidden.get(f"{self.layer_name}_ester_core")(V.C, V.O, V.R1, V.R2),
                    R.get(self.hydrogen)(V.R1),
                )
            ]
        )

        # Defining carbonate ester group (R1-O-C(=O)-O-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonate_ester")(V.C)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.X, V.O1, V.O2),
                    R.get(self.oxygen)(V.O1),
                    R.get(self.oxygen)(V.O2),
                    R.get(self.carbon)(V.R1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.R1, V.O1, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(self.carbon)(V.R2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.R2, V.O2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                )
            ]
        )

        # Defining the ether group (R1-O-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ether_carbon")(V.C)
                <= R.get(f"{self.layer_name}_saturated")(V.C)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ether_carbon")(V.C)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(f"{self.layer_name}_aromatic_bonded")(V.C, V.Y),
                )
            ]
        )
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_ether_link")(V.C, V.O, V.R)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.oxygen)(V.O),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.R, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.special.alldiff(V.C, V.R, V.O),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ether")(V.O)
                <= (
                    R.get(f"{self.layer_name}_ether_carbon")(V.C),
                    R.hidden.get(f"{self.layer_name}_ether_link")(V.C, V.O, V.R),
                    R.get(f"{self.layer_name}_ether_carbon")(V.R),
                )
            ]
        )

        # Defining phenol (Ar-OH); the hydroxyl oxygen is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_phenol")(V.O)
                <= (
                    R.get(f"{self.layer_name}_hydroxyl")(V.O),
                    R.get(self.carbon)(V.C),
                    R.get(f"{self.layer_name}_aromatic_bonded")(V.C, V.Y),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.C, V.O, V.Y),
                )
            ]
        )

        # Defining an epoxide (three-membered C-C-O ring); oxygen is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_epoxide")(V.O)
                <= (
                    R.hidden.get(f"{self.layer_name}_ether_link")(V.C1, V.O, V.C2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C1, V.C2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.C1, V.C2, V.O),
                )
            ]
        )

        # Defining a peroxide (O-O); each oxygen atom is a carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_peroxide")(V.O1)
                <= (
                    R.get(self.oxygen)(V.O1),
                    R.get(self.oxygen)(V.O2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O1, V.O2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.O1, V.O2),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_peroxide")(V.O2)
                <= (
                    R.get(self.oxygen)(V.O1),
                    R.get(self.oxygen)(V.O2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O1, V.O2, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.special.alldiff(V.O1, V.O2),
                )
            ]
        )

        # Defining an acetal (C(OR)2); the central carbon is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_acetal")(V.C)
                <= (
                    R.get(f"{self.layer_name}_saturated")(V.C),
                    R.get(f"{self.layer_name}_ether")(V.O1),
                    R.get(f"{self.layer_name}_ether")(V.O2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.special.alldiff(V.C, V.O1, V.O2),
                )
            ]
        )

        # Defining a lactone (cyclic ester); the carbonyl carbon is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_lactone")(V.C)
                <= (
                    R.hidden.get(f"{self.layer_name}_ester_core")(
                        V.C, V.O2, V.R1, V.R2
                    ),
                    R.get(self.carbon)(V.R1),
                    R.get(f"{self.layer_name}_cycle")(V.C, V.O2),
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
                R.get(f"{self.layer_name}_oxy_groups")(V.Y)
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
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_phenol")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_epoxide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_peroxide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_acetal")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_oxy_groups")(V.X)
                <= R.get(f"{self.layer_name}_lactone")(V.X)[self.param_size]
            ]
        )
