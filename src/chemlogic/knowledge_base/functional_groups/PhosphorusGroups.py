from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class PhosphorusGroups(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "carbon",
        "oxygen",
        "nitrogen",
        "phosphor",
    ]

    def create_template(self):
        # Defining single and double phosphorus-oxygen bonds for resonance forms.
        for bond_predicate in ("single_bonded", "double_bonded"):
            self.add_rules(
                [
                    R.hidden.get(f"{self.layer_name}_phosphorus_oxygen_bond")(V.P, V.O)
                    <= (
                        R.get(self.phosphor)(V.P),
                        R.get(self.oxygen)(V.O),
                        R.hidden.get(f"{self.layer_name}_{bond_predicate}")(
                            V.P, V.O, V.B
                        ),
                        R.get(f"{self.layer_name}_bond_message")(V.B),
                    )
                ]
            )

        # Defining reusable phosphorus-oxygen cores.
        for oxygen_count in (2, 3, 4):
            oxygens = [f"O{i}" for i in range(oxygen_count)]
            self.add_rules(
                [
                    R.hidden.get(
                        f"{self.layer_name}_phosphorus_oxygen_core_{oxygen_count}"
                    )(V.P, *oxygens)
                    <= (
                        *(
                            R.hidden.get(f"{self.layer_name}_phosphorus_oxygen_bond")(
                                V.P, oxygen
                            )
                            for oxygen in oxygens
                        ),
                        R.special.alldiff(V.P, *oxygens),
                    )
                ]
            )

        # Defining phosphate (P bonded to four oxygens); phosphorus is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_phosphate")(V.P)
                <= (
                    R.hidden.get(f"{self.layer_name}_phosphorus_oxygen_core_4")(
                        V.P, V.O1, V.O2, V.O3, V.O4
                    ),
                    R.special.alldiff(V.P, V.O1, V.O2, V.O3, V.O4),
                )
            ]
        )

        # Defining phosphonate (one P-C and three P-O bonds); phosphorus is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_phosphonate")(V.P)
                <= (
                    R.get(self.carbon)(V.C),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.P, V.C, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.hidden.get(f"{self.layer_name}_phosphorus_oxygen_core_3")(
                        V.P, V.O1, V.O2, V.O3
                    ),
                    R.special.alldiff(V.P, V.C, V.O1, V.O2, V.O3),
                )
            ]
        )

        # Defining phosphinate (two P-C and two P-O bonds); phosphorus is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_phosphinate")(V.P)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.carbon)(V.C2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.P, V.C1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.P, V.C2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.B2),
                    R.hidden.get(f"{self.layer_name}_phosphorus_oxygen_core_2")(
                        V.P, V.O1, V.O2
                    ),
                    R.special.alldiff(V.P, V.C1, V.C2, V.O1, V.O2),
                )
            ]
        )

        # Defining phosphoramidate (P-N and at least two P-O bonds); phosphorus is the carrier.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_phosphoramidate")(V.P)
                <= (
                    R.get(self.nitrogen)(V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.P, V.N, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.B),
                    R.hidden.get(f"{self.layer_name}_phosphorus_oxygen_core_2")(
                        V.P, V.O1, V.O2
                    ),
                    R.special.alldiff(V.P, V.N, V.O1, V.O2),
                )
            ]
        )

        # Aggregating phosphorus groups.
        self.add_rules(
            [
                R.get(f"{self.layer_name}_phosphorus_groups")(V.P)
                <= R.get(f"{self.layer_name}_{group}")(V.P)[self.param_size]
                for group in (
                    "phosphate",
                    "phosphonate",
                    "phosphinate",
                    "phosphoramidate",
                )
            ]
        )
