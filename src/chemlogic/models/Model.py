from neuralogic.core import R, Transformation, V

from chemlogic.utils.ChemTemplate import ChemTemplate as Template


class Model(Template):
    # TODO: add prefix to name (in case where more than 1 GNN computations in parallel - could be needed)
    def __init__(
        self,
        model_name: str,
        layers: int,
        node_embed: str,
        edge_embed: str,
        connection: str,
        param_size: int,
        output_layer_name: str = "predict",
        output_layer_transformation=Transformation.IDENTITY,
        **kwargs,
    ):
        """
        Initializes a Model object.

        Args:
            model_name (str): The name of the model. This will be the predicate name.
            layers (int): The number of layers.
            node_embed (str): The node embedding predicate.
            edge_embed (str): The edge embedding predicate.
            connection (str): The connection type predicate.
            param_size (int): The size of the parameter.
            output_layer_name (str): The name of the output layer. Default "predict"
        """
        super().__init__()
        # Type and value checks
        if not isinstance(model_name, str):
            raise TypeError("`model_name` must be a string.")
        if not isinstance(layers, int) or layers < 1:
            raise ValueError("`layers` must be a positive integer.")
        if not isinstance(node_embed, str):
            raise TypeError("`node_embed` must be a string.")
        if not isinstance(edge_embed, str):
            raise TypeError("`edge_embed` must be a string.")
        if not isinstance(connection, str):
            raise TypeError("`connection` must be a string.")
        if not isinstance(param_size, int) or param_size < 1:
            raise ValueError("`param_size` must be a positive integer.")
        if not isinstance(output_layer_name, str):
            raise TypeError("`output_layer_name` must be a string.")

        self.model_name = model_name
        self.layers = layers
        self.node_embed = node_embed
        self.edge_embed = edge_embed
        self.connection = connection
        self.output_layer_name = output_layer_name
        self.output_layer_transformation = output_layer_transformation

        # Match the parameter size to the output dimension
        if param_size == 1:
            self.output_param_size = (param_size,)
            self.param_size = (param_size,)
        else:
            self.output_param_size = (1, param_size)
            self.param_size = (param_size, param_size)

        self.create_template()

    def build_layer(self, current_layer: str, previous_layer: str) -> list:
        return [
            (R.get(current_layer)(V.X) <= R.get(previous_layer)(V.X)[self.param_size])
        ]

    def create_template(self):
        previous_layer = self.node_embed
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
                | [self.output_layer_transformation]
            ]
        )
