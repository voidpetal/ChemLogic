from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class NeighborhoodPatterns(KnowledgeBase):
    required_keys = [
        "layer_name",
        "param_size",
        "node_embed",
        "edge_embed",
        "connection",
        "atom_type",
        "carbon",
        "nbh_min_size",
        "nbh_max_size",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(self.nbh_min_size, int) or self.nbh_min_size < 3:
            raise ValueError(
                f"Invalid nbh_min_size={self.nbh_min_size}, must be an integer bigger than 2."
            )

        if (
            not isinstance(self.nbh_max_size, int)
            or self.nbh_max_size <= self.nbh_min_size
        ):
            raise ValueError(
                f"Invalid nbh_max_size={self.nbh_max_size}, must be an integer bigger than {self.nbh_min_size}"
            )

    def create_template(self):
        nbhoods = [f"{self.layer_name}_chiral_center"]

        # n-node neighborhoods
        for n in range(self.nbh_min_size, self.nbh_max_size + 1):
            connections = [(V.X, f"X{i}", f"B{i}") for i in range(n)]
            node_embeddings = [
                R.get(self.node_embed)(f"X{i}")[self.param_size] for i in range(n)
            ]
            edge_embeddings = [
                R.get(self.edge_embed)(f"B{i}")[self.param_size] for i in range(n)
            ]
            nbhood_body = (
                [R.get(self.connection)(*conn) for conn in connections]
                + node_embeddings
                + edge_embeddings
                + [R.special.alldiff(V.X, *(f"X{i}" for i in range(n)))]
            )
            self.add_rules([R.get(f"{self.layer_name}_{n}_nbhood")(V.X) <= nbhood_body])
            nbhoods += [f"{self.layer_name}_{n}_nbhood"]

        # Chiral center is a carbon atom surrounded by
        chiral_connections = [(V.C, f"X{i}", f"B{i}") for i in range(4)]
        chiral_edge_embeddings = [
            R.get(self.edge_embed)(f"B{i}")[self.param_size] for i in range(4)
        ]
        chiral_node_embeddings = [
            R.get(self.atom_type)(f"X{i}")[self.param_size] for i in range(4)
        ] + [R.get(self.node_embed)(f"X{i}")[self.param_size] for i in range(4)]
        chiral_center_body = (
            [R.get(self.carbon)(V.C)]
            + [R.get(self.connection)(*conn) for conn in chiral_connections]
            + chiral_edge_embeddings
            + chiral_node_embeddings
            + [R.special.alldiff(V.C, *(f"X{i}" for i in range(4)))]
        )
        self.add_rules(
            [R.get(f"{self.layer_name}_chiral_center")(V.C) <= chiral_center_body]
        )

        # Neighborhood pattern aggregation
        for nbhood in nbhoods:
            self.add_rules(
                [
                    R.get(f"{self.layer_name}_nbhood")(V.X)
                    <= R.get(nbhood)(V.X)[self.param_size]
                ]
            )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_pattern")(V.X)
                <= R.get(f"{self.layer_name}_nbhood")(V.X)[self.param_size]
            ]
        )
