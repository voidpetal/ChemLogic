from chemlogic.models.CWNet import CWNet
from chemlogic.models.DiffusionCNN import DiffusionCNN
from chemlogic.models.EgoGNN import EgoGNN
from chemlogic.models.GNN import GNN
from chemlogic.models.KGNN import KGNN
from chemlogic.models.RGCN import RGCN
from chemlogic.models.SGN import SGN

# Registry of model types
MODEL_REGISTRY = {
    "gnn": GNN,
    "rgcn": RGCN,
    "kgnn": KGNN,
    "ego": EgoGNN,
    "diffusion": DiffusionCNN,
    "cw": CWNet,
    "sgn": SGN,
}


def get_available_models():
    return list(MODEL_REGISTRY.keys())


def get_model(
    model_name: str,
    layers: int,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: int,
    output_layer_name: str,
    num_outputs: int = 1,
    **kwargs,
):
    """
    Instantiates a model class based on its name and configuration.

    Parameters
    ----------
    model_name : str
        Type of the model (e.g., "gnn", "rgcn", etc.)
    layers : int
        Number of layers in the model.
    node_embed : str
        Predicate for node embeddings.
    edge_embed : str
        Predicate for edge embeddings.
    connection : str
        Predicate for connections (expects connection(X, Y, E)).
    param_size : int
        Size of the parameter embeddings.
    output_layer_name : str
        Name of the output layer.
    **kwargs : dict
        Additional model-specific parameters (e.g., max_depth, local).

    Returns
    -------
    An instance of the selected model class.

    Raises
    ------
    ValueError
        If the model name is not recognized.
    """
    model_class = MODEL_REGISTRY.get(model_name)

    if model_class is None:
        raise ValueError(
            f"Invalid model name: '{model_name}'. Available models: {', '.join(get_available_models())}"
        )

    return model_class(
        output_layer_name=output_layer_name,
        layers=layers,
        node_embed=node_embed,
        edge_embed=edge_embed,
        connection=connection,
        param_size=param_size,
        num_outputs=num_outputs,
        **kwargs,
    )
