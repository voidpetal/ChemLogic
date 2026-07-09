from neuralogic.core import R, V

from chemlogic.models.Model import Model


class RGCN(Model):
    """Relational Graph Convolutional Network.

    Extends GNN to heterogeneous graphs by assigning a separate weight matrix
    to each edge type (relation). Each node aggregates neighbour representations
    grouped by relation type, allowing the model to distinguish, e.g., single
    bonds from aromatic bonds without collapsing them into a shared embedding.

        layer_k(X) <= layer_{k-1}(X), layer_{k-1}(Y), connection(X, Y, B), edge_type_r(B)
                      -- one rule per relation type r

    Requires at least two edge types; with one type it reduces to a standard GNN.

    Reference:
        Schlichtkrull et al. "Modeling Relational Data with Graph Convolutional Networks."
        arXiv:1703.06103, 2017.
        https://arxiv.org/abs/1703.06103
    """

    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "rgcn"

        # Extract and validate edge_types
        # TODO: maybe make this simpler
        if "edge_types" not in kwargs:
            raise KeyError("Missing required keyword argument: `edge_types`")

        self.edge_types = kwargs.pop("edge_types")
        if not isinstance(self.edge_types, list):
            raise TypeError("`edge_types` must be a list of predicates.")
        if len(self.edge_types) < 2:
            raise ValueError(
                "`edge_types` must contain at least two types. "
                "RGCN with one edge type is equivalent to a standard GNN."
            )

        super().__init__(*args, **kwargs)

    # rgcn_k(X) <=  rgcn_k-1(X), rgcn_k-1(Y), connection(X, Y, B), edge_embed(B), edge_type(B) for all edge types
    def build_layer(self, current_layer: str, previous_layer: str) -> list:
        return [
            (
                R.get(current_layer)(V.X)
                <= (
                    R.get(previous_layer)(V.X)[self.param_size],
                    R.get(previous_layer)(V.Y)[self.param_size],
                    R.get(self.connection)(V.X, V.Y, V.B),
                    # R.get(edge_embed)(V.B), # maybe doesnt make sense to have this, as the information is encoded below
                    R.get(t)(V.B),
                )
            )
            for t in self.edge_types
        ]
