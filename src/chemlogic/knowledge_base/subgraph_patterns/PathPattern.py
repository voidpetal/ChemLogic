from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class PathPattern(KnowledgeBase):
    """Path-based reachability pattern detector.

    Builds `{layer_name}_path(X, Y)` predicates encoding reachability up to
    `max_depth` steps, using a timestep counter. Results are contributed to
    `{layer_name}_pattern`.

    Args:
        max_depth (int): Maximum path length. Must be >= 3.
    """
    required_keys = [
        "layer_name",
        "param_size",
        "node_embed",
        "edge_embed",
        "connection",
        "max_depth",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(self.max_depth, int) or self.max_depth < 3:
            raise ValueError(
                f"Invalid max_depth={self.max_depth}, must be an integer bigger than 2."
            )

    def create_template(self):
        """Generate path reachability rules up to max_depth steps.

        Produces `{layer_name}_path(X, Y)` using a timestep counter, aggregated
        into `{layer_name}_pattern(X)`.
        """
        # Defining constants for keeping track
        for i in range(self.max_depth):
            self.add_rules([(R._next(i, i + 1))])

        # Base case
        self.add_rules(
            [
                R.get(f"{self.layer_name}_path")(V.X, V.Y, 0)
                <= (
                    R.get(self.connection)(V.X, V.Y, V.B),
                    R.get(self.edge_embed)(V.B)[self.param_size],
                    R.get(self.node_embed)(V.Y)[self.param_size],
                )
            ]
        )
        # Recursive calls
        self.add_rules(
            [
                R.get(f"{self.layer_name}_path")(V.X, V.Y, V.T)
                <= (
                    R.special.next(V.T1, V.T),
                    R.get(self.connection)(V.X, V.Z, V.B),
                    R.get(f"{self.layer_name}_path")(V.Z, V.Y, V.T1)[self.param_size],
                    R.get(self.edge_embed)(V.B)[self.param_size],
                    R.get(self.node_embed)(V.X)[self.param_size],
                )
            ]
        )

        # If there is a path from X to Y less than or equal to max_depth
        self.add_rules(
            [
                (
                    R.get(f"{self.layer_name}_path")(V.X, V.Y)
                    <= (R.get(f"{self.layer_name}_path")(V.X, V.Y, self.max_depth))
                )
            ]
        )

        # Aggregating for X
        self.add_rules(
            [
                R.get(f"{self.layer_name}_pattern")(V.X)
                <= R.get(f"{self.layer_name}_path")(V.X, V.Y)[self.param_size]
            ]
        )
