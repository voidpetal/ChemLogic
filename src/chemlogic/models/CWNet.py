from neuralogic.core import R, V

from chemlogic.models.Model import Model
from chemlogic.utils.ChemTemplate import ChemTemplate as Template


class CWNet(Model):
    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "cw"

        self.max_ring_size = kwargs.pop("max_ring_size", 7)
        if not isinstance(self.max_ring_size, int) or self.max_ring_size < 4:
            raise TypeError("`max_ring_size` must be an integer larger than 3.")

        super().__init__(*args, **kwargs)

    # TODO: can this be merged with super's function?
    def create_template(self) -> Template:
        previous_layer = f"{self.model_name}_0"
        self.add_rules(
            [(R.get(f"{previous_layer}_node")(V.X) <= (R.get(self.node_embed)(V.X)))]
        )
        self.add_rules(
            [(R.get(f"{previous_layer}_edge")(V.X) <= (R.get(self.edge_embed)(V.X)))]
        )

        for i in range(self.layers):
            current_layer = f"{self.model_name}_{i + 1}"
            self.add_rules(self.build_layer(current_layer, previous_layer))
            previous_layer = current_layer

        # Final aggregation
        self.add_rules(
            [
                (
                    R.get(f"{self.model_name}")(V.X)
                    <= R.get(f"{self.model_name}_{self.layers}")(V.X)
                )
            ]
        )

        # Output layer
        self.add_rules(
            [
                (
                    R.get(self.output_layer_name)[self.output_param_size]
                    <= R.get(f"{self.model_name}")(V.X)
                )
            ]
        )

    # Aggregating bond features
    def bond_features(self, current_layer: str, previous_layer: str):
        template = []
        # atoms aggregate to bonds, bonds to rings
        template += [
            R.get(current_layer + "_edge")(V.B)
            <= (
                R.get(self.connection)(V.X, V.Y, V.B),
                R.get(f"{previous_layer}_node")(V.X)[self.param_size],
                R.get(f"{previous_layer}_node")(V.Y)[self.param_size],
            )
        ]

        # bonds in same cycle
        def get_bond_cycle(n):
            body = [
                R.get(self.connection)(f"X{i}", f"X{(i + 1) % n}", f"B{i}")
                for i in range(n)
            ]
            body.extend(
                R.get(f"{previous_layer}_edge")(f"B{i}")[self.param_size]
                for i in range(n)
            )
            body.append(R.special.alldiff(*(f"X{i}" for i in range(n))))

            return [R.get(current_layer + "_edge")(V.B0) <= body]

        for i in range(3, self.max_ring_size):
            template += get_bond_cycle(i)

        return template

    # Aggregating node features
    def node_features(self, current_layer: str, previous_layer: str):
        template = []

        # atoms sharing a bond share messages, bonds in the same ring
        template += [
            R.get(current_layer + "_node")(V.X)
            <= (
                R.get(self.connection)(V.X, V.Y, V.B),
                R.get(f"{previous_layer}_node")(V.Y)[self.param_size],
                R.get(f"{previous_layer}_edge")(V.B)[self.param_size],
            )
        ]
        return template

    # Constructing a layer of CW net, aggregating node and edge features to layer output
    def build_layer(self, current_layer: str, previous_layer: str) -> list:
        template = []
        template += self.bond_features(current_layer, previous_layer)
        template += self.node_features(current_layer, previous_layer)

        template += [
            R.get(current_layer)(V.X)
            <= (R.get(current_layer + "_node")(V.X)[self.param_size])
        ]
        template += [
            R.get(current_layer)(V.X)
            <= (R.get(current_layer + "_edge")(V.X)[self.param_size])
        ]

        return template
