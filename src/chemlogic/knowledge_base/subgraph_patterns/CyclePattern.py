from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class CyclePattern(KnowledgeBase):
    """Ring/cycle detector for molecular graphs.

    Generates rules that identify cycles of size `min_cycle_size` up to
    `max_cycle_size`. Results are aggregated into a `{layer_name}_cycle`
    predicate and contributed to `{layer_name}_pattern`.

    Args:
        min_cycle_size (int): Smallest ring size to detect. Must be >= 3.
        max_cycle_size (int): Largest ring size to detect. Must be > min_cycle_size.
    """

    required_keys = [
        "layer_name",
        "param_size",
        "node_embed",
        "edge_embed",
        "connection",
        "max_cycle_size",
        "min_cycle_size",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(self.min_cycle_size, int) or self.min_cycle_size < 3:
            raise ValueError(
                f"Invalid min_cycle_size={self.min_cycle_size}, must be an integer bigger than 2."
            )

        if (
            not isinstance(self.max_cycle_size, int)
            or self.max_cycle_size <= self.min_cycle_size
        ):
            raise ValueError(
                f"Invalid max_cycle_size={self.max_cycle_size}, must be an integer bigger than {self.min_cycle_size}"
            )

    def create_template(self):
        """Generate cycle detection rules for sizes min_cycle_size..max_cycle_size.

        Produces `{layer_name}_cycle(X, X0)` for each size, aggregated into
        `{layer_name}_cycle(X)` and contributed to `{layer_name}_pattern(X)`.
        """

        def get_cycle(cycle_size):
            # Cycles are paths from a node to itself, with every node on the path being unique
            # cannot use path predicate here, because all the edges are undirected
            body = [
                R.get(self.connection)(f"X{i}", f"X{(i + 1) % cycle_size}", f"B{i}")
                for i in range(cycle_size)
            ]
            body.extend(
                R.get(self.node_embed)(f"X{i}")[self.param_size]
                for i in range(cycle_size)
            )
            body.extend(
                R.get(self.edge_embed)(f"B{i}")[self.param_size]
                for i in range(cycle_size)
            )
            body.append(
                R.special.alldiff(f"X{i}" for i in range(cycle_size))
            )  # X0....Xmax are different
            body.append(
                R.special._in((V.X,) + tuple(f"X{i}" for i in range(1, cycle_size)))
            )  # X and X0 are in the cycle

            return [R.get(f"{self.layer_name}_cycle")(V.X, V.X0) <= body]

        # Generate cycles of varying sizes
        for i in range(self.min_cycle_size, self.max_cycle_size):
            self.add_rules(get_cycle(i))

        # Aggregating to subgraph patterns
        self.add_rules(
            [
                R.get(f"{self.layer_name}_cycle")(V.X)
                <= R.get(f"{self.layer_name}_cycle")(V.X, V.X0)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_pattern")(V.X)
                <= R.get(f"{self.layer_name}_cycle")(V.X)[self.param_size]
            ]
        )
