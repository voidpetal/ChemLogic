from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class CollectivePatterns(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "node_embed",
        "edge_embed",
        "connection",
        "aliphatic_bond",
        "carbon",
    ]

    def create_template(self):
        # Defining when two atoms are NOT in a same cycle
        self.add_rules(
            [
                R.get(f"{self.layer_name}_n_cycle")(V.X, V.Y)
                <= R.get(f"{self.layer_name}_cycle")(V.X, V.Y)
            ]
        )

        # Bridge atom between two cycles
        self.add_rules(
            [
                R.get(f"{self.layer_name}_bridge")(V.X)
                <= (
                    R.get(self.connection)(V.X, V.Y, V.B1),
                    R.get(self.connection)(V.X, V.Z, V.B2),
                    # ~R.get(f"{self.layer_name}_n_cycle")(V.X, V.X1),
                    ~R.get(f"{self.layer_name}_n_cycle")(V.Y, V.Z),
                    R.get(f"{self.layer_name}_cycle")(V.Y, V.Y1)[self.param_size],
                    R.get(f"{self.layer_name}_cycle")(V.Z, V.Z1)[self.param_size],
                    R.get(self.edge_embed)(V.B1)[self.param_size],
                    R.get(self.edge_embed)(V.B2)[self.param_size],
                    R.get(self.node_embed)(V.X)[self.param_size],
                    R.special.alldiff(V.X, V.Y, V.Z),
                )
            ]
        )

        # Shared atom between two cycles
        self.add_rules(
            [
                R.get(f"{self.layer_name}_shared_atom")(V.X)
                <= (
                    R.get(self.connection)(V.X, V.Y, V.B1),
                    R.get(self.connection)(V.X, V.Z, V.B2),
                    R.get(f"{self.layer_name}_cycle")(V.X, V.Y)[self.param_size],
                    R.get(f"{self.layer_name}_cycle")(V.X, V.Z)[self.param_size],
                    ~R.get(f"{self.layer_name}_n_cycle")(V.Y, V.Z),
                    R.get(self.edge_embed)(V.B1)[self.param_size],
                    R.get(self.edge_embed)(V.B2)[self.param_size],
                    R.get(self.node_embed)(V.X)[self.param_size],
                    R.special.alldiff(V.X, V.Y, V.Z),
                )
            ]
        )

        # Chain of carbons connected by a single bond
        self.add_rules(
            [
                R.get(f"{self.layer_name}_aliphatic_chain")(V.X, V.Y)
                <= R.get(f"{self.layer_name}_aliphatic_chain")(V.X, V.Y, self.max_depth)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_aliphatic_chain")(V.X)
                <= R.get(f"{self.layer_name}_aliphatic_chain")(V.X, V.Y)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_aliphatic_chain")(V.X, V.Y, 0)
                <= (
                    R.get(self.connection)(V.X, V.Z, V.B),
                    R.get(self.carbon)(V.X),
                    R.get(self.carbon)(V.Y),
                    R.get(self.aliphatic_bond)(V.B),
                    R.get(self.edge_embed)(V.B)[self.param_size],
                    R.get(self.node_embed)(V.Y)[self.param_size],
                )
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_aliphatic_chain")(V.X, V.Y, V.T)
                <= (
                    R.get(self.carbon)(V.X),
                    R.special.next(V.T1, V.T),
                    R.get(self.connection)(V.X, V.Z, V.B),
                    R.get(f"{self.layer_name}_aliphatic_chain")(V.Z, V.Y, V.T1)[
                        self.param_size
                    ],
                    R.get(self.aliphatic_bond)(V.B)[self.param_size],
                    R.get(self.edge_embed)(V.B)[self.param_size],
                    R.get(self.node_embed)(V.X)[self.param_size],
                )
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_collective_pattern")(V.X)
                <= R.get(f"{self.layer_name}_aliphatic_chain")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_collective_pattern")(V.X)
                <= R.get(f"{self.layer_name}_shared_atom")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_collective_pattern")(V.X)
                <= R.get(f"{self.layer_name}_bridge")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_pattern")(V.X)
                <= R.get(f"{self.layer_name}_collective_pattern")(V.X)[self.param_size]
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_subgraph_pattern")(V.X)
                <= (
                    R.get(f"{self.layer_name}_pattern")(V.X)[self.param_size],
                    R.get(f"{self.layer_name}_pattern")(V.Y)[self.param_size],
                    R.get(f"{self.layer_name}_path")(V.X, V.Y),
                )
            ]
        )
