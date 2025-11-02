from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class RelaxedFunctionalGroups(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "carbon",
        "connection",
    ]

    def create_template(self):
        # Defining a relaxed aliphatic and aromatic bond messages
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y)
                <= (R.get(f"{self.layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y, V.B))
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y, V.B)
                <= (
                    R.get(self.connection)(V.X, V.Y, V.B),
                    R.get(f"{self.layer_name}_aliphatic_bond")(V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.X, V.Y, V.B)[
                        self.param_size
                    ],
                )
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.X, V.Y)
                <= (R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.X, V.Y, V.B))
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.X, V.Y, V.B)
                <= (
                    R.get(self.connection)(V.X, V.Y, V.B),
                    R.get(f"{self.layer_name}_aromatic_bond")(V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.X, V.Y, V.B)[
                        self.param_size
                    ],
                )
            ]
        )

        # Defining a relaxed carbonyl group (key atom type connected to another using an aliphatic bond)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_carbonyl_group")(V.X, V.Y)
                <= (
                    R.get(f"{self.layer_name}_key_atom")(V.X)[self.param_size],
                    R.get(f"{self.layer_name}_key_atom")(V.Y)[self.param_size],
                    R.get(f"{self.layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y),
                )
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_carbonyl_group")(V.C, V.O, V.R1, V.R2)
                <= (
                    R.get(f"{self.layer_name}_relaxed_carbonyl_group")(V.C, V.O),
                    R.get(f"{self.layer_name}_relaxed_aliphatic_bonded")(V.C, V.R1),
                    R.get(f"{self.layer_name}_relaxed_aliphatic_bonded")(V.C, V.R2),
                    R.special.alldiff(V.C, V.O, V.R1, V.R2),
                )
            ]
        )

        # Defining a relaxed aromatic ring
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_benzene_ring")(V.A)
                <= R.get(f"{self.layer_name}_relaxed_benzene_ring")(V.A, V.B)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_benzene_ring")(V.A, V.B)
                <= (
                    R.get(f"{self.layer_name}_relaxed_benzene_ring")(
                        V.A, V.B, V.C, V.D, V.E, V.F
                    )
                )
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_benzene_ring")(
                    V.A, V.B, V.C, V.D, V.E, V.F
                )
                <= (
                    R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.A, V.B),
                    R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.B, V.C),
                    R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.C, V.D),
                    R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.D, V.E),
                    R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.E, V.F),
                    R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.F, V.A),
                    R.special.alldiff(V.A, V.B, V.C, V.D, V.E, V.F),
                )
            ]
        )

        # Defining a potential group
        self.add_rules(
            [
                R.get(f"{self.layer_name}_potential_group")(V.C)
                <= (
                    R.get(f"{self.layer_name}_relaxed_aliphatic_bonded")(V.C, V.X)[
                        self.param_size
                    ],
                    R.get(f"{self.layer_name}_noncarbon")(V.X)[self.param_size],
                    R.get(self.carbon)(V.C),
                )
            ]
        )

        # Defining a relaxed carbonyl derivative
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonyl_derivatives")(V.X)
                <= (
                    R.get(f"{self.layer_name}_relaxed_carbonyl_group")(
                        V.C, V.O, V.R, V.X
                    ),
                    R.get(f"{self.layer_name}_noncarbon")(V.X)[self.param_size],
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonyl_derivatives")(V.C)
                <= (
                    R.get(f"{self.layer_name}_relaxed_carbonyl_group")(
                        V.C, V.X, V.R, V.R2
                    ),
                    R.get(f"{self.layer_name}_noncarbon")(V.X)[self.param_size],
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbonyl_derivatives")(V.C)
                <= (
                    R.get(f"{self.layer_name}_relaxed_carbonyl_group")(
                        V.C, V.X, V.R, V.R2
                    ),
                    R.get(f"{self.layer_name}_noncarbon")(V.C)[self.param_size],
                )
            ]
        )

        # Aggregating relaxations
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_functional_group")(V.X)
                <= (
                    R.get(f"{self.layer_name}_relaxed_carbonyl_group")(
                        V.X, V.Y, V.R1, V.R2
                    )[self.param_size]
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_functional_group")(V.X)
                <= (
                    R.get(f"{self.layer_name}_relaxed_aliphatic_bonded")(V.X, V.Y)[
                        self.param_size
                    ]
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_functional_group")(V.X)
                <= (
                    R.get(f"{self.layer_name}_relaxed_aromatic_bonded")(V.X, V.Y)[
                        self.param_size
                    ]
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_functional_group")(V.X)
                <= (
                    R.get(f"{self.layer_name}_relaxed_benzene_ring")(V.X)[
                        self.param_size
                    ]
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_functional_group")(V.X)
                <= (R.get(f"{self.layer_name}_potential_group")(V.X)[self.param_size])
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_relaxed_functional_group")(V.X)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_derivatives")(V.X)[
                        self.param_size
                    ]
                )
            ]
        )
