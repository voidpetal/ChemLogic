from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class GeneralFunctionalGroups(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "node_embed",
        "edge_embed",
        "connection",
        "single_bond",
        "double_bond",
        "triple_bond",
        "aromatic_bond",
        "hydrogen",
        "carbon",
        "oxygen",
    ]

    def create_template(self):
        # Aggregating bond messages
        self.add_rules(
            [
                R.get(f"{self.layer_name}_bond_message")(V.X, V.Y, V.B)
                <= (
                    R.get(self.node_embed)(V.X)[self.param_size],
                    R.get(self.node_embed)(V.Y)[self.param_size],
                    R.get(self.edge_embed)(V.B)[self.param_size],
                )
            ]
        )

        # Defining the predicates when two atoms are single/double/... bonded to each other
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_single_bonded")(V.X, V.Y)
                <= (R.hidden.get(f"{self.layer_name}_single_bonded")(V.X, V.Y, V.B))
            ]
        )
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_single_bonded")(V.X, V.Y, V.B)
                <= (R.get(self.connection)(V.X, V.Y, V.B), R.get(self.single_bond)(V.B))
            ]
        )

        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_double_bonded")(V.X, V.Y)
                <= (R.hidden.get(f"{self.layer_name}_double_bonded")(V.X, V.Y, V.B))
            ]
        )
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_double_bonded")(V.X, V.Y, V.B)
                <= (R.get(self.connection)(V.X, V.Y, V.B), R.get(self.double_bond)(V.B))
            ]
        )

        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_triple_bonded")(V.X, V.Y)
                <= (R.hidden.get(f"{self.layer_name}_triple_bonded")(V.Y, V.X, V.B))
            ]
        )
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_triple_bonded")(V.X, V.Y, V.B)
                <= (R.get(self.connection)(V.Y, V.X, V.B), R.get(self.triple_bond)(V.B))
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_aromatic_bonded")(V.X, V.Y)
                <= (R.get(f"{self.layer_name}_aromatic_bonded")(V.X, V.Y, V.B))
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_aromatic_bonded")(V.X, V.Y, V.B)
                <= (
                    R.get(self.connection)(V.X, V.Y, V.B),
                    R.get(self.aromatic_bond)(V.B),
                )
            ]
        )

        # Defining saturated carbons
        # TODO: this won't work for datasets with implicit hydrogens
        self.add_rules(
            [
                R.get(f"{self.layer_name}_saturated")(V.X)
                <= (
                    R.get(self.carbon)(V.X),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.X, V.Y1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.X, V.Y2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.X, V.Y3),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.X, V.Y4),
                    R.special.alldiff(V.Y1, V.Y2, V.Y3, V.Y4),
                )
            ]
        )

        # Defining a halogen group (R-X)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_halogen_group")(V.R)
                <= (
                    R.get(f"{self.layer_name}_halogen")(V.X),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.X, V.R, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.X, V.R, V.B),
                )
            ]
        )

        # Defining hydroxyl group (O-H)
        # TODO: this won't work for datasets with implicit hydrogens
        self.add_rules(
            [
                R.get(f"{self.layer_name}_hydroxyl")(V.O)
                <= (
                    R.get(self.oxygen)(V.O),
                    R.get(self.hydrogen)(V.H),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.H, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.O, V.H, V.B),
                )
            ]
        )

        # Defining carbonyl group (R1-C(=O)-R2)
        # TODO: this won't work for datasets with implicit hydrogens
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.oxygen)(V.O),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.O, V.C, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.O, V.C, V.B),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.R1, V.R2)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.R1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.R2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.R1, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.R2, V.B2),
                    R.special.alldiff(V.R1, V.R2, V.C, V.O),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonyl_group")(V.C)
                <= (R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O))
            ]
        )

        # Aggregating general patterns
        self.add_rules(
            [
                R.get(f"{self.layer_name}_general_groups")(V.X)
                <= R.get(f"{self.layer_name}_hydroxyl")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_general_groups")(V.X)
                <= R.get(f"{self.layer_name}_carbonyl_group")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_general_groups")(V.X)
                <= R.get(f"{self.layer_name}_halogen_group")(V.X)[self.param_size]
            ]
        )
