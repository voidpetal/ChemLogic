from neuralogic.core import R, Template, Transformation, V

from chemlogic.knowledge_base.functional_groups.GeneralFunctionalGroups import (
    GeneralFunctionalGroups,
)
from chemlogic.knowledge_base.functional_groups.Hydrocarbons import Hydrocarbons
from chemlogic.knowledge_base.functional_groups.NitrogenGroups import NitrogenGroups
from chemlogic.knowledge_base.functional_groups.OxygenGroups import OxygenGroups
from chemlogic.knowledge_base.functional_groups.RelaxedFunctionalGroups import (
    RelaxedFunctionalGroups,
)
from chemlogic.knowledge_base.functional_groups.SulfurGroups import SulfurGroups


def get_chem_rules(
    layer_name: str,
    node_embed: str,
    edge_embed: str,
    connection: str,
    param_size: int,
    halogens: list,
    output_layer_name: str = "predict",
    output_layer_transformation=Transformation.IDENTITY,
    single_bond=None,
    double_bond=None,
    triple_bond=None,
    aromatic_bonds=None,
    carbon=None,
    hydrogen=None,
    oxygen=None,
    nitrogen=None,
    sulfur=None,
    path=None,
    hydrocarbons=False,
    nitro=False,
    sulfuric=False,
    oxy=False,
    relaxations=False,
    key_atoms: list = None,
    funnel=False,
):
    template = Template()
    if funnel:
        param_size = 1

    for b in aromatic_bonds:
        template.add_rules(
            [(R.get(f"{layer_name}_aromatic_bond")(V.B)[param_size,] <= R.get(b)(V.B))]
        )

    for a in halogens:
        template.add_rules(
            [(R.get(f"{layer_name}_halogen")(V.X)[param_size,] <= R.get(a)(V.X))]
        )
    if relaxations:
        for b in [single_bond, double_bond, triple_bond]:
            template.add_rules(
                [
                    (
                        R.get(f"{layer_name}_aliphatic_bond")(V.B)[param_size,]
                        <= R.get(b)(V.B)
                    )
                ]
            )

        if key_atoms is None:
            key_atoms = []

        for a in key_atoms:
            template.add_rules(
                [(R.get(f"{layer_name}_key_atom")(V.A)[param_size,] <= R.get(a)(V.A))]
            )
            template.add_rules(
                [(R.get(f"{layer_name}_noncarbon")(V.A)[param_size,] <= R.get(a)(V.A))]
            )

        template.add_rules(
            [(R.get(f"{layer_name}_key_atom")(V.A)[param_size,] <= R.get(carbon)(V.A))]
        )

    if param_size == 1:
        template.add_rules(
            [
                (
                    R.get(output_layer_name)[param_size,]
                    <= R.get(f"{layer_name}_chem_rules")
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
                    <= R.get(f"{layer_name}_chem_rules")
                )
                | [output_layer_transformation]
            ]
        )
        param_size = (param_size, param_size)

    template = (
        GeneralFunctionalGroups(
            layer_name=layer_name,
            node_embed=node_embed,
            edge_embed=edge_embed,
            connection=connection,
            param_size=param_size,
            single_bond=single_bond,
            double_bond=double_bond,
            triple_bond=triple_bond,
            aromatic_bond=f"{layer_name}_aromatic_bond",
            hydrogen=hydrogen,
            carbon=carbon,
            oxygen=oxygen,
        )
        + template
    )  # because neuralogic.template + chemlogic.template appends it whole to the list

    template.add_rules(
        [
            R.get(f"{layer_name}_functional_group")(V.X)[param_size]
            <= R.get(f"{layer_name}_general_groups")(V.X)
        ]
    )

    if path:
        template.add_rules(
            [
                R.get(f"{layer_name}_connected_groups")(V.X, V.Y)
                <= (
                    R.get(f"{layer_name}_functional_group")(V.X)[param_size],
                    R.get(f"{layer_name}_functional_group")(V.Y)[param_size],
                    R.get(path)(V.X, V.Y),
                )
            ]
        )
        if relaxations:
            template.add_rules(
                [
                    R.get(f"{layer_name}_connected_groups")(V.X, V.Y)
                    <= (
                        R.get(f"{layer_name}_relaxed_functional_group")(V.X)[
                            param_size
                        ],
                        R.get(f"{layer_name}_relaxed_functional_group")(V.Y)[
                            param_size
                        ],
                        R.get(path)(V.X, V.Y),
                    )
                ]
            )
        template.add_rules(
            [
                R.get(f"{layer_name}_chem_rules")(V.X)[param_size]
                <= R.get(f"{layer_name}_connected_groups")(V.X, V.Y)
            ]
        )

    if hydrocarbons:
        template = (
            Hydrocarbons(layer_name=layer_name, param_size=param_size, carbon=carbon)
            + template
        )
        template.add_rules(
            [
                R.get(f"{layer_name}_functional_group")(V.X)[param_size]
                <= R.get(f"{layer_name}_hydrocarbon_groups")(V.X)
            ]
        )
    if oxy:
        template = (
            OxygenGroups(
                layer_name=layer_name,
                param_size=param_size,
                carbon=carbon,
                oxygen=oxygen,
                hydrogen=hydrogen,
            )
            + template
        )
        template.add_rules(
            [
                R.get(f"{layer_name}_functional_group")(V.X)[param_size]
                <= R.get(f"{layer_name}_oxy_groups")(V.X)
            ]
        )
    if nitro:
        template = (
            NitrogenGroups(
                layer_name=layer_name,
                param_size=param_size,
                carbon=carbon,
                oxygen=oxygen,
                nitrogen=nitrogen,
            )
            + template
        )
        template.add_rules(
            [
                R.get(f"{layer_name}_functional_group")(V.X)[param_size]
                <= R.get(f"{layer_name}_nitrogen_groups")(V.X)
            ]
        )
    if sulfuric:
        template = (
            SulfurGroups(
                layer_name=layer_name,
                param_size=param_size,
                carbon=carbon,
                hydrogen=hydrogen,
                nitrogen=nitrogen,
                sulfur=sulfur,
            )
            + template
        )
        template.add_rules(
            [
                R.get(f"{layer_name}_functional_group")(V.X)[param_size]
                <= R.get(f"{layer_name}_sulfuric_groups")(V.X)
            ]
        )
    if relaxations:
        template = (
            RelaxedFunctionalGroups(
                layer_name=layer_name,
                param_size=param_size,
                connection=connection,
                carbon=carbon,
            )
            + template
        )
        template.add_rules(
            [
                R.get(f"{layer_name}_chem_rules")(V.X)[param_size]
                <= R.get(f"{layer_name}_relaxed_functional_group")(V.X)
            ]
        )

    template.add_rules(
        [
            R.get(f"{layer_name}_chem_rules")(V.X)[param_size]
            <= R.get(f"{layer_name}_functional_group")(V.X)
        ]
    )
    template.add_rules(
        [
            R.get(f"{layer_name}_chem_rules")[param_size]
            <= R.get(f"{layer_name}_chem_rules")(V.X)
        ]
    )

    return template
