from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class RingGroups(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "node_embed",
        "edge_embed",
        "carbon",
    ]

    ring_sizes = range(5, 8)

    def __init__(self, aromatic_bonds: list, **kwargs):
        if not aromatic_bonds:
            raise ValueError(
                "Ring rules require at least one dataset-configured aromatic bond predicate."
            )
        super().__init__(**kwargs)

    def create_template(self):
        # Defining a non-carbon atom helper for heteroaromatic rings
        self.add_rules(
            [
                R.hidden.get(f"{self.layer_name}_ring_carbon")(V.X)
                <= R.get(self.carbon)(V.X)
            ]
        )

        # Defining aromatic rings and heteroaromatic rings (5- to 7-membered)
        for size in self.ring_sizes:
            atoms = [V.X, *(f"A{i}" for i in range(1, size))]
            bonds = [f"B{i}" for i in range(size)]
            cycle_body = [
                R.get(f"{self.layer_name}_aromatic_bonded")(
                    atoms[i], atoms[(i + 1) % size], bonds[i]
                )
                for i in range(size)
            ]
            cycle_body.extend(
                R.get(self.node_embed)(atom)[self.param_size] for atom in atoms
            )
            cycle_body.extend(
                R.get(self.edge_embed)(bond)[self.param_size] for bond in bonds
            )
            cycle_body.append(R.special.alldiff(*atoms))

            cycle_predicate = R.hidden.get(f"{self.layer_name}_aromatic_cycle_{size}")
            self.add_rules([cycle_predicate(*atoms) <= cycle_body])
            self.add_rules(
                [
                    R.get(f"{self.layer_name}_aromatic_ring")(V.X)
                    <= cycle_predicate(V.X, *atoms[1:])
                ]
            )
            self.add_rules(
                [
                    R.get(f"{self.layer_name}_heteroaromatic_ring")(V.X)
                    <= (
                        cycle_predicate(V.X, *atoms[1:]),
                        ~R.hidden.get(f"{self.layer_name}_ring_carbon")(V.X),
                    )
                ]
            )

        # Defining edge-fused aromatic ring pairs
        fused_rules = []
        for first_size in self.ring_sizes:
            first_atoms = [V.X, V.Y, *(f"A{i}" for i in range(2, first_size))]
            for second_size in range(first_size, self.ring_sizes.stop):
                second_atoms = [V.X, V.Y, *(f"B{i}" for i in range(2, second_size))]
                distinct_atoms = [
                    *first_atoms,
                    *second_atoms[2:],
                ]
                fused_rules.append(
                    R.get(f"{self.layer_name}_fused_aromatic_ring")(V.X)
                    <= (
                        R.hidden.get(f"{self.layer_name}_aromatic_cycle_{first_size}")(
                            *first_atoms
                        ),
                        R.hidden.get(f"{self.layer_name}_aromatic_cycle_{second_size}")(
                            *second_atoms
                        ),
                        R.special.alldiff(*distinct_atoms),
                    )
                )
        self.add_rules(fused_rules)

        # Aggregating ring groups
        self.add_rules(
            [
                R.get(f"{self.layer_name}_ring_groups")(V.X)
                <= R.get(f"{self.layer_name}_{matcher}")(V.X)[self.param_size]
                for matcher in (
                    "aromatic_ring",
                    "heteroaromatic_ring",
                    "fused_aromatic_ring",
                )
            ]
        )
