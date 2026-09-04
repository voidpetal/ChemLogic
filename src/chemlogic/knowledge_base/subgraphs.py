from neuralogic.core import R, Transformation, V

from chemlogic.knowledge_base.subgraph_patterns.CircularPatterns import CircularPatterns
from chemlogic.knowledge_base.subgraph_patterns.CollectivePatterns import (
    CollectivePatterns,
)
from chemlogic.knowledge_base.subgraph_patterns.CyclePattern import CyclePattern
from chemlogic.knowledge_base.subgraph_patterns.NeighborhoodPatterns import (
    NeighborhoodPatterns,
)
from chemlogic.knowledge_base.subgraph_patterns.PathPattern import PathPattern
from chemlogic.knowledge_base.subgraph_patterns.YShapePattern import YShapePattern
from chemlogic.utils.ChemTemplate import ChemTemplate


def get_subgraphs(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: int,
    max_cycle_size: int = 10,
    max_depth: int = 5,
    output_layer_name: str = "predict",
    output_layer_transformation=Transformation.IDENTITY,
    single_bond=None,
    double_bond=None,
    carbon=None,
    atom_types=None,
    aliphatic_bonds=None,
    cycles=False,
    paths=False,
    y_shape=False,
    nbhoods=False,
    circular=False,
    collective=False,
    funnel=False,
):
    """Build a subgraph pattern template for a given layer.

    Composes cycle, path, Y-shape, neighbourhood, circular, and collective
    pattern detectors into a single Template that exposes a unified
    `{layer_name}_subgraph_pattern` predicate wired into the output layer.
    When `funnel=True` the param_size is forced to 1.

    Args:
        layer_name (str): Prefix for all generated predicate names.
        node_embed (str): Node embedding predicate name.
        edge_embed (str): Edge embedding predicate name.
        connection (str): Bond connectivity predicate name.
        param_size (int): Embedding dimension.
        max_cycle_size (int): Maximum ring size for cycle detection. Default 10.
        max_depth (int): Maximum path length. Default 5.
        output_layer_name (str): Output predicate name. Default "predict".
        output_layer_transformation: NeuraLogic output transformation. Default IDENTITY.
        single_bond: Single bond predicate name.
        double_bond: Double bond predicate name.
        carbon (str): Carbon atom predicate name.
        atom_types (list[str]): All atom-type predicate names.
        aliphatic_bonds (list[str]): Aliphatic bond predicate names.
        cycles (bool): Enable cycle patterns.
        paths (bool): Enable path patterns.
        y_shape (bool): Enable Y-shape patterns.
        nbhoods (bool): Enable neighbourhood patterns.
        circular (bool): Enable circular/brick patterns.
        collective (bool): Enable collective (bridge/chain) patterns.
        funnel (bool): Force param_size=1 for scalar interpretable weights.

    Returns:
        neuralogic.core.Model: Model containing subgraph feature extraction rules.
            Produces a `{layer_name}_subgraph_pattern(X)` predicate aggregating all
            enabled pattern types, wired into `output_layer_name`.
    """
    if funnel:
        param_size = 1

    template = ChemTemplate()

    # Aggregating the patterns
    if param_size == 1:
        template.add_rules(
            [
                (
                    R.get(output_layer_name)[param_size,]
                    <= R.get(f"{layer_name}_subgraph_pattern")
                )
                | [output_layer_transformation]
            ]
        )
        param_size = (param_size,)
    else:
        template.add_rules(
            [
                (
                    R.get(output_layer_name)[1, param_size]
                    <= R.get(f"{layer_name}_subgraph_pattern")
                )
                | [output_layer_transformation]
            ]
        )
        param_size = (param_size, param_size)

    # Adding patterns
    if cycles or circular or collective:
        template = (
            CyclePattern(
                layer_name=layer_name,
                node_embed=node_embed,
                edge_embed=edge_embed,
                connection=connection,
                param_size=param_size,
                max_cycle_size=max_cycle_size,
            )
            + template
        )
    if paths or collective:
        template = (
            PathPattern(
                layer_name=layer_name,
                node_embed=node_embed,
                edge_embed=edge_embed,
                connection=connection,
                param_size=param_size,
                max_depth=max_depth,
            )
            + template
        )
    if y_shape:
        template = (
            YShapePattern(
                layer_name=layer_name,
                node_embed=node_embed,
                edge_embed=edge_embed,
                connection=connection,
                param_size=param_size,
                double_bond=double_bond,
            )
            + template
        )
    if nbhoods:
        for t in atom_types:
            # TODO: unstable param size
            template.add_rules(
                [R.get(f"{layer_name}_key_atoms")(V.X)[param_size[0],] <= R.get(t)(V.X)]
            )
        template = (
            NeighborhoodPatterns(
                layer_name=layer_name,
                node_embed=node_embed,
                edge_embed=edge_embed,
                connection=connection,
                param_size=param_size,
                carbon=carbon,
                atom_type=f"{layer_name}_key_atoms",
            )
            + template
        )
    if circular:
        template = (
            CircularPatterns(
                layer_name=layer_name,
                node_embed=node_embed,
                edge_embed=edge_embed,
                connection=connection,
                param_size=param_size,
                carbon=carbon,
                single_bond=single_bond,
                double_bond=double_bond,
            )
            + template
        )
    if collective:
        for t in aliphatic_bonds:
            template.add_rules(
                [
                    R.get(f"{layer_name}_aliphatic_bond")(V.X)[param_size[0],]
                    <= R.get(t)(V.X)
                ]
            )
        template = (
            CollectivePatterns(
                layer_name=layer_name,
                node_embed=node_embed,
                edge_embed=edge_embed,
                connection=connection,
                param_size=param_size,
                carbon=carbon,
                aliphatic_bond=f"{layer_name}_aliphatic_bond",
                max_depth=max_depth,
            )
            + template
        )

    template.add_rules(
        [
            R.get(f"{layer_name}_subgraph_pattern")(V.X)
            <= R.get(f"{layer_name}_pattern")(V.X)[param_size]
        ]
    )

    template.add_rules(
        [
            R.get(f"{layer_name}_subgraph_pattern")
            <= R.get(f"{layer_name}_subgraph_pattern")(V.X)[param_size]
        ]
    )

    return template
