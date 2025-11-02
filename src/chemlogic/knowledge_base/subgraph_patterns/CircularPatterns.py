from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class CircularPatterns(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "node_embed",
        "edge_embed",
        "connection",
        "single_bond",
        "double_bond",
        "carbon",
    ]

    def create_template(self):
        # Defining Carbon negation helper predicate and heterocycles
        self.add_rules(
            [R.get(f"{self.layer_name}_n_c")(V.X) <= R.get(self.carbon)(V.X)]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_heterocycle")(V.X)
                <= (
                    R.get(self.carbon)(V.C),
                    ~R.hidden.get(f"{self.layer_name}_n_c")(V.X),
                    R.get(f"{self.layer_name}_cycle")(V.X, V.C)[self.param_size],
                )
            ]
        )

        # Defining "brick" substructure (X-Y1=Y2-Y3=X)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_brick")(V.X)
                <= (
                    R.get(self.connection)(V.X, V.Y1, V.B1),
                    R.get(self.connection)(V.Y1, V.Y2, V.B2),
                    R.get(self.connection)(V.Y2, V.Y3, V.B3),
                    R.get(self.connection)(V.Y3, V.X, V.B4),
                    R.get(self.single_bond)(V.B1),
                    R.get(self.double_bond)(V.B2),
                    R.get(self.single_bond)(V.B3),
                    R.get(self.double_bond)(V.B4),
                    R.get(self.node_embed)(V.Y1)[self.param_size],
                    R.get(self.node_embed)(V.Y2)[self.param_size],
                    R.get(self.node_embed)(V.Y3)[self.param_size],
                    R.get(self.node_embed)(V.X)[self.param_size],
                    R.get(self.edge_embed)(V.B1)[self.param_size],
                    R.get(self.edge_embed)(V.B2)[self.param_size],
                    R.get(self.edge_embed)(V.B3)[self.param_size],
                    R.get(self.edge_embed)(V.B4)[self.param_size],
                    R.special.alldiff(V.X, V.Y1, V.Y2, V.Y3),
                )
            ]
        )

        # Aggregating into a common predicate
        self.add_rules(
            [
                R.get(f"{self.layer_name}_circular")(V.X)
                <= R.get(subgraph)(V.X)[self.param_size]
                for subgraph in [
                    f"{self.layer_name}_brick",
                    f"{self.layer_name}_heterocycle",
                ]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_pattern")(V.X)
                <= R.get(f"{self.layer_name}_circular")(V.X)[self.param_size]
            ]
        )
