from neuralogic.core import R, V

from chemlogic.models.Model import Model


class KGNN(Model):
    """Higher-order GNN implementing the k-dimensional Weisfeiler-Lehman test (k-GNN).

    Standard GNNs are bounded by the expressive power of 1-WL and cannot
    distinguish certain non-isomorphic graphs (e.g. two 3-regular graphs of
    the same size). k-GNN lifts representations from individual nodes to
    k-tuples of nodes, matching the k-WL hierarchy and strictly increasing
    expressiveness with k.

    This implementation builds a chain of tuple representations up to max_depth:

        layer_k_1(X, Y) <= layer_k_0(X), layer_k_0(Y), alldiff(X, Y)
        layer_k_{d+1}(X, Y) <= layer_k_d(X, Z), layer_k_d(Z, Y), alldiff(X, Y, Z)

    The `local` variant (kgnn_local) restricts aggregation to graph-adjacent
    tuples by conditioning on a connection predicate, improving scalability.

    Reference:
        Morris et al. "Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks."
        arXiv:1810.02244, 2021.
        https://arxiv.org/abs/1810.02244
    """

    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "kgnn"

        self.local = kwargs.pop("local", True)
        if not isinstance(self.local, bool):
            raise TypeError("`local` must be a boolean.")

        self.max_depth = kwargs.pop("max_depth", 1)
        if not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise TypeError("`max_depth` must be a positive integer.")

        super().__init__(*args, **kwargs)

    # Creating kGNN sets up to max depth
    def build_layer(self, current_layer: str, previous_layer: str) -> list:
        # Defining the input and aggregating the output
        template = [
            (R.get(f"{current_layer}_0")(V.X) <= (R.get(f"{previous_layer}")(V.X)))
        ]
        template += [
            (
                R.get(f"{current_layer}")(V.X)
                <= (R.get(f"{current_layer}_{self.max_depth}")(V.X, V.Y))
            )
        ]

        # Constructing kGNN from k-1GNN
        for k in range(self.max_depth):
            if k == 0:
                body = [
                    R.get(f"{current_layer}_{k}")(V.X)[self.param_size],
                    R.get(f"{current_layer}_{k}")(V.Y)[self.param_size],
                    R.special.alldiff(V.X, V.Y),
                ]
            else:
                body = [
                    R.get(f"{current_layer}_{k}")(V.X, V.Z)[self.param_size],
                    R.get(f"{current_layer}_{k}")(V.Z, V.Y)[self.param_size],
                    R.special.alldiff(V.X, V.Y, V.Z),
                ]

            if self.local:
                body += [
                    R.get(self.connection)(V.X, V.Y, V.B),
                    R.get(self.edge_embed)(V.B)[self.param_size],
                ]

            template += [(R.get(f"{current_layer}_{k + 1}")(V.X, V.Y) <= body)]

        return template
