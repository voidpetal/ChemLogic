from neuralogic.core import R, V

from chemlogic.models.Model import Model


class EgoGNN(Model):
    """Ego-centric GNN.

    Augments standard message passing with ego-graph structure: each node
    first builds a "multigraph" representation by aggregating edge and
    neighbour embeddings, then aggregates those multigraph representations
    from its neighbours. This two-step aggregation captures richer
    neighbourhood topology than a single round of message passing.

        layer_multigraph(X) <= connection(X, Y, B), edge_embed(B), prev(Y)
        layer(X)            <= connection(X, Y, B), layer_multigraph(Y)

    Reference:
        Sandfelder et al. "Ego-GNNs: Exploiting Ego Structures in Graph Neural Networks."
        ICASSP 2021.
        https://doi.org/10.1109/icassp39728.2021.9414015
    """

    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "ego"
        super().__init__(*args, **kwargs)

    def build_layer(self, current_layer: str, previous_layer: str) -> list:
        template = []
        template += [
            R.get(current_layer + "_multigraph")(V.X)
            <= (
                R.get(self.connection)(V.X, V.Y, V.B),
                R.get(self.edge_embed)(V.B)[self.param_size],
                R.get(previous_layer)(V.Y)[self.param_size],
            )
        ]

        template += [
            R.get(current_layer)(V.X)
            <= (
                R.get(self.connection)(V.X, V.Y, V.B),
                R.get(current_layer + "_multigraph")(V.Y)[self.param_size],
            )
        ]
        return template
