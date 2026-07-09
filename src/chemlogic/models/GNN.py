from neuralogic.core import R, V

from chemlogic.models.Model import Model


class GNN(Model):
    """Standard message-passing GNN with edge features.

    Each node aggregates representations from its neighbours and their
    connecting edge embeddings. The update rule per layer is:

        layer_k(X) <= layer_{k-1}(X), layer_{k-1}(Y), connection(X, Y, B), edge_embed(B)

    This subsumes GCN (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017),
    and GIN (Xu et al., 2019) as special cases differing only in the aggregation
    and message function.

    Reference:
        Scarselli et al. "The Graph Neural Network Model."
        IEEE Transactions on Neural Networks, 2009.
        https://doi.org/10.1109/TNN.2008.2005605
    """

    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "gnn"
        super().__init__(*args, **kwargs)

    # gnn_k(X) <=  gnn_k-1(X), gnn_k-1(Y), connection(X, Y, B), edge_embed(B)
    def build_layer(self, current_layer: str, previous_layer: str) -> list:
        return [
            (
                R.get(current_layer)(V.X)
                <= (
                    R.get(previous_layer)(V.X)[self.param_size],
                    R.get(previous_layer)(V.Y)[self.param_size],
                    R.get(self.connection)(
                        V.X, V.Y, V.B
                    ),  # should be first to ground faster?
                    R.get(self.edge_embed)(V.B),
                )
            )
        ]  # why not parametrized?
