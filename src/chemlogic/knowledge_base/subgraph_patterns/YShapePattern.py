from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class YShapePattern(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "node_embed",
        "edge_embed",
        "connection",
        "double_bond",
    ]

    def create_template(self):
        # Aggregating messages in a double bond
        self.add_rules(
            [
                R.get(f"{self.layer_name}_double_bond_subgraph")(V.X)
                <= (
                    R.get(self.connection)(V.X, V.Y, V.B),
                    R.get(self.double_bond)(V.B),
                    R.get(self.node_embed)(V.Y)[self.param_size],
                    R.get(self.edge_embed)(V.B)[self.param_size],
                )
            ]
        )

        # Simple 3 neighborhood
        self.add_rules(
            [
                R.get(f"{self.layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4)
                <= (
                    R.get(self.connection)(V.X1, V.X2, V.B1),
                    R.get(self.connection)(V.X1, V.X3, V.B2),
                    R.get(self.connection)(V.X1, V.X4, V.B3),
                    R.get(self.edge_embed)(V.B1)[self.param_size],
                    R.get(self.edge_embed)(V.B2)[self.param_size],
                    R.get(self.edge_embed)(V.B3)[self.param_size],
                    R.get(self.node_embed)(V.X1)[self.param_size],
                    R.get(self.node_embed)(V.X2)[self.param_size],
                    R.get(self.node_embed)(V.X3)[self.param_size],
                    R.get(self.node_embed)(V.X4)[self.param_size],
                    R.special.alldiff(V.X1, V.X2, V.X3, V.X4),
                )
            ]
        )

        # Y subgraph with a double bond
        self.add_rules(
            [
                R.get(f"{self.layer_name}_y_bond")(V.X1, V.X2, V.X3, V.X4)
                <= (
                    R.get(self.connection)(V.X1, V.X2, V.B1),
                    R.get(self.double_bond)(V.B1),
                    R.get(f"{self.layer_name}_y_subgraph")(V.X1, V.X2, V.X3, V.X4),
                    R.special.alldiff(V.X1, V.X2, V.X3, V.X4),
                )
            ]
        )

        # Two Y double bond subgraphs connected with X1 (X1-Y1(=Y2)-X2-Z1(=Z2)-X3)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_y_group")(V.X1, V.X2, V.X3)
                <= (
                    R.get(f"{self.layer_name}_y_bond")(V.Y1, V.Y2, V.X1, V.X2),
                    R.get(f"{self.layer_name}_y_bond")(V.Z1, V.Z2, V.X2, V.X3),
                    R.special.alldiff(V.X1, V.X2, V.X3),
                )
            ]
        )

        # Collecting all Y patterns
        self.add_rules(
            [
                (
                    R.get(f"{self.layer_name}_y_bond_patterns")(V.X)
                    <= R.get(f"{self.layer_name}_double_bond_subgraph")(V.X)[
                        self.param_size
                    ]
                ),
                (
                    R.get(f"{self.layer_name}_y_bond_patterns")(V.X1)
                    <= R.get(f"{self.layer_name}_y_bond")(V.X1, V.X2, V.X3, V.X4)[
                        self.param_size
                    ]
                ),
                (
                    R.get(f"{self.layer_name}_y_bond_patterns")(V.X2)
                    <= R.get(f"{self.layer_name}_y_group")(V.X1, V.X2, V.X3)[
                        self.param_size
                    ]
                ),
                (
                    R.get(f"{self.layer_name}_pattern")(V.X)
                    <= R.get(f"{self.layer_name}_y_bond_patterns")(V.X)[self.param_size]
                ),
            ]
        )
