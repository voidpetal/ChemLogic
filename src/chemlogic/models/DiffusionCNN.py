from neuralogic.core import Aggregation, R, Transformation, V

from chemlogic.models.Model import Model


class DiffusionCNN(Model):
    """Diffusion Convolutional Neural Network.

    Models information spread as a diffusion process over the graph.
    Rather than aggregating only direct neighbours, each node accumulates
    signals from all nodes reachable within max_depth steps, weighted by
    the diffusion path. A path predicate tracks depth via a `next` relation,
    and the final representation sums all path-endpoint contributions with
    a sigmoid non-linearity.

        diffusion_path(X, Y, 0)   <= edge_embed(B), connection(X, Y, B)
        diffusion_path(X, Y, T)   <= edge_embed(B), diffusion_path(Z, Y, T-1), connection(X, Z, B)
        layer(X) [sigmoid, sum]   <= diffusion_path(X, Y), prev(Y)

    Captures global structural context without requiring many GNN layers,
    which can help with over-smoothing on small molecular graphs.

    Reference:
        Atwood & Towsley. "Diffusion-Convolutional Neural Networks."
        arXiv:1511.02136, 2016.
        https://arxiv.org/abs/1511.02136
    """

    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "diffusion_cnn"

        self.max_depth = kwargs.pop("max_depth", 1)
        if not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise TypeError("`max_depth` must be a positive integer.")

        super().__init__(*args, **kwargs)

        self.add_rules(self.get_path(f"{self.model_name}_diffusion_path"))

    # Defining a path between nodes to a max depth
    def get_path(self, layer_name: str):
        template = []
        template += [
            (
                R.get(layer_name)(V.X, V.Y, 0)
                <= (
                    R.get(self.edge_embed)(V.B)[self.param_size],
                    R.get(self.connection)(V.X, V.Y, V.B),
                )
            )
        ]

        template += [
            (
                R.get(layer_name)(V.X, V.Y, V.T)
                <= (
                    R.get(self.edge_embed)(V.B)[self.param_size],
                    R.get(layer_name)(V.Z, V.Y, V.T1)[self.param_size],
                    R.get(self.connection)(V.X, V.Z, V.B),
                    R.special.next(V.T1, V.T),
                )
            )
        ]

        # Defining constants for keeping track
        for i in range(self.max_depth):
            template += [(R._next(i, i + 1))]

        template += [
            (
                R.get(layer_name)(V.X, V.Y)
                <= (R.get(layer_name)(V.X, V.Y, self.max_depth))
            )
        ]

        return template

    # Creating a Diffusion CNN layer
    def build_layer(self, current_layer: str, previous_layer: str) -> list:
        template = []
        template += [
            (
                R.get(current_layer + "_Z")(V.X)
                <= (
                    R.get(f"{self.model_name}_diffusion_path")(V.X, V.Y),
                    R.get(previous_layer)(V.Y)[self.param_size],
                )
            )
            | [Aggregation.SUM]
        ]
        template += [
            (
                R.get(current_layer + "_Z")(V.X)
                <= R.get(previous_layer)(V.X)[self.param_size]
            )
        ]
        template += [
            (R.get(current_layer)(V.X) <= (R.get(current_layer + "_Z")(V.X)))
            | [Transformation.SIGMOID, Aggregation.SUM]
        ]

        return template
