import tempfile
import unittest

from neuralogic.core import Model, R, Settings, V

from chemlogic.datasets import (
    SmilesDataset,
)
from chemlogic.knowledge_base.chemrules import get_chem_rules
from chemlogic.knowledge_base.functional_groups import (
    GeneralFunctionalGroups,
    Hydrocarbons,
    NitrogenGroups,
    OxygenGroups,
    RelaxedFunctionalGroups,
    SulfurGroups,
)


class TestFunctionalGroupModules(unittest.TestCase):
    def setUp(self):
        self.common_args = {
            "layer_name": "fg_layer",
            "param_size": (4, 4),
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
            "single_bond": "sb",
            "double_bond": "db",
            "triple_bond": "tb",
            "aromatic_bond": "ar",
            "hydrogen": "H",
            "carbon": "C",
            "oxygen": "O",
        }

    def check_buildable(
        self,
        smiles,
        hydrocarbons=False,
        nitro=False,
        sulfuric=False,
        oxy=False,
        relaxations=False,
        expected_predicate=None,
        expected_groups=None,
        predicate_arity=1,
        carrier_variable=None,
        carrier_atom=None,
        expected_aggregate=None,
        expected_aggregate_groups=None,
    ):
        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True
        ) as output_directory:
            dataset = SmilesDataset(
                smiles_list=smiles,
                labels=[1] * len(smiles),
                param_size=1,
                dataset_name="test_functional_group_grounding",
                output_directory=output_directory,
            )

            kb = get_chem_rules(
                "chem",
                dataset.node_embed,
                dataset.edge_embed,
                dataset.connection,
                1,
                dataset.halogens,
                output_layer_name="predict",
                single_bond=dataset.single_bond,
                double_bond=dataset.double_bond,
                triple_bond=dataset.triple_bond,
                aromatic_bonds=dataset.aromatic_bonds,
                carbon=dataset.carbon,
                hydrogen=dataset.hydrogen,
                oxygen=dataset.oxygen,
                nitrogen=dataset.nitrogen,
                sulfur=dataset.sulfur,
                key_atoms=dataset.key_atom_type,
                hydrocarbons=hydrocarbons,
                nitro=nitro,
                sulfuric=sulfuric,
                oxy=oxy,
                relaxations=relaxations,
            )
            dataset += kb
            dataset.flatten()

            evaluator = dataset.build(Settings())
            groundings = evaluator.ground(dataset.data)
            dataset.clear()

            if expected_predicate is not None:
                if carrier_variable is not None:
                    self.assertIn(
                        f"chem_{expected_predicate}({carrier_variable})", str(kb)
                    )
                expected_groups = expected_groups or [True] * len(smiles)
                aggregate_groups = (
                    expected_groups
                    if expected_aggregate_groups is None
                    else expected_aggregate_groups
                )
                self.assertEqual(len(groundings), len(smiles))
                variables = (V.X, V.Y)[:predicate_arity]
                predicate = R.get(f"chem_{expected_predicate}")(*variables)

                for smile, grounding, expected, aggregate_expected in zip(
                    smiles,
                    groundings,
                    expected_groups,
                    aggregate_groups,
                    strict=True,
                ):
                    grounded = grounding.get_atoms(predicate)
                    self.assertEqual(
                        bool(grounded),
                        expected,
                        f"{expected_predicate} grounding mismatch for {smile}",
                    )

                    if expected and carrier_atom is not None:
                        group_ids = {atom.substitutions["X"] for atom in grounded}
                        carrier_facts = grounding.get_atoms(R.get(carrier_atom)(V.X))
                        carrier_ids = {
                            atom.substitutions["X"] for atom in carrier_facts
                        }
                        self.assertTrue(
                            group_ids <= carrier_ids,
                            f"{expected_predicate} is not anchored on {carrier_atom} for {smile}",
                        )

                    if expected_aggregate is not None:
                        aggregate = grounding.get_atoms(
                            R.get(f"chem_{expected_aggregate}")(V.X)
                        )
                        self.assertEqual(
                            bool(aggregate),
                            aggregate_expected,
                            f"{expected_aggregate} aggregation mismatch for {smile}",
                        )
        fg = GeneralFunctionalGroups(**self.common_args)
        self.assertIsInstance(fg, Model)
        self.assertEqual(fg.layer_name, "fg_layer")

    def test_general_functional_groups_missing_required_param(self):
        for key in GeneralFunctionalGroups.required_keys:
            args = {k: v for k, v in self.common_args.items() if k != key}
            with self.assertRaises((ValueError, TypeError)):
                GeneralFunctionalGroups(**args)

    def test_general_functional_groups_rule_structure(self):
        fg = GeneralFunctionalGroups(**self.common_args)
        rules_str = str(fg)
        expected_predicates = [
            "bond_message",
            "single_bonded",
            "double_bonded",
            "triple_bonded",
            "aromatic_bonded",
            "saturated",
            "halogen_group",
            "hydroxyl",
            "carbonyl_group",
            "general_groups",
        ]
        for name in expected_predicates:
            self.assertIn(f"fg_layer_{name}", rules_str)

    def test_carboxyl(self):
        self.check_buildable(
            ["C=O", "CCO"],  # formaldehyde, ethanol
            expected_predicate="carbonyl_group",
            expected_groups=[True, False],
            carrier_variable="C",
            carrier_atom="c",
            expected_aggregate="general_groups",
            expected_aggregate_groups=[True, True],
        )

    def test_hydroxyl(self):
        self.check_buildable(
            ["C-O", "C=O"],  # methanol, formaldehyde
            expected_predicate="hydroxyl",
            expected_groups=[True, False],
            carrier_variable="O",
            carrier_atom="o",
            expected_aggregate="general_groups",
            expected_aggregate_groups=[True, True],
        )

    def test_halogen(self):
        self.check_buildable(
            ["CF", "C=O"],  # fluormethane, formaldehyde
            expected_predicate="halogen_group",
            expected_groups=[True, False],
            carrier_variable="X",
            carrier_atom="f",
            expected_aggregate="general_groups",
            expected_aggregate_groups=[True, True],
        )

    def test_not_buildable(self):
        self.check_buildable(
            ["C=O"],
            hydrocarbons=True,
            nitro=True,
            sulfuric=True,
            oxy=True,
            expected_predicate="hydrocarbon_groups",
            expected_groups=[False],
        )

    # Hydrocarbons
    def test_hydrocarbons_instantiation(self):
        args = {
            "layer_name": "hydro_layer",
            "param_size": (4, 4),
            "carbon": "C",
        }
        hc = Hydrocarbons(**args)
        self.assertIsInstance(hc, Model)
        self.assertEqual(hc.layer_name, "hydro_layer")

    def test_hydrocarbons_missing_required_param(self):
        for key in Hydrocarbons.required_keys:
            args = {
                "layer_name": "hydro_layer",
                "param_size": (4, 4),
                "carbon": "C",
            }
            args.pop(key, None)
            with self.assertRaises((ValueError, TypeError)):
                Hydrocarbons(**args)

    def test_hydrocarbons_rule_structure(self):
        args = {
            "layer_name": "hydro_layer",
            "param_size": (4, 4),
            "carbon": "C",
        }
        hc = Hydrocarbons(**args)
        rules_str = str(hc)
        expected_predicates = [
            "benzene_ring",
            "alkene_bond",
            "alkyne_bond",
            "hydrocarbon_groups",
        ]
        for name in expected_predicates:
            self.assertIn(f"hydro_layer_{name}", rules_str)

    def test_benzene(self):
        self.check_buildable(
            ["c1ccccc1", "C1CCCCC1C=O"],  # benzene, cyclohexanecarbaldehyde
            hydrocarbons=True,
            expected_predicate="benzene_ring",
            expected_groups=[True, False],
            carrier_variable="A",
            carrier_atom="c",
            expected_aggregate="hydrocarbon_groups",
            expected_aggregate_groups=[True, False],
        )

    def test_alkene(self):
        self.check_buildable(
            ["C=C", "CCO"],  # ethene, ethanol
            hydrocarbons=True,
            expected_predicate="alkene_bond",
            expected_groups=[True, False],
            predicate_arity=2,
            expected_aggregate="hydrocarbon_groups",
            expected_aggregate_groups=[True, False],
        )

    def test_alkyne(self):
        self.check_buildable(
            ["C#C", "CCO"],  # acetylene, ethanol
            hydrocarbons=True,
            expected_predicate="alkyne_bond",
            expected_groups=[True, False],
            predicate_arity=2,
            expected_aggregate="hydrocarbon_groups",
            expected_aggregate_groups=[True, False],
        )

    # Nitrogens
    def test_nitrogens_instantiation(self):
        args = {
            "layer_name": "nitro_layer",
            "param_size": (4, 4),
            "carbon": "c",
            "oxygen": "o",
            "nitrogen": "n",
        }
        hc = NitrogenGroups(**args)
        self.assertIsInstance(hc, Model)
        self.assertEqual(hc.layer_name, "nitro_layer")

    def test_nitrogens_missing_required_param(self):
        for key in NitrogenGroups.required_keys:
            args = {
                "layer_name": "nitro_layer",
                "param_size": (4, 4),
                "carbon": "c",
                "oxygen": "o",
                "nitrogen": "n",
            }
            args.pop(key, None)
            with self.assertRaises((ValueError, TypeError)):
                NitrogenGroups(**args)

    def test_nitrogens_rule_structure(self):
        args = {
            "layer_name": "nitro_layer",
            "param_size": (4, 4),
            "carbon": "c",
            "oxygen": "o",
            "nitrogen": "n",
        }
        hc = NitrogenGroups(**args)
        rules_str = str(hc)
        expected_predicates = [
            "amine",
            "n_carbonyl",
            "n_quaternary",
            "amino_group",
            "quat_ammonion",
            "amide",
            "imine",
            "imide",
            "azide",
            "azo",
            "cyanate",
            "isocyanate",
            "nitro_group",
            "nitro",
            "nitrate",
            "carbamate",
            "aziridine",
            "nitrogen_groups",
        ]
        for name in expected_predicates:
            self.assertIn(f"nitro_layer_{name}", rules_str)

    def test_amine(self):
        self.check_buildable(
            [
                "CN",
                "C[NH2]",
                "CNC",
                "C[NH]C",
                "CN(C)C",
                "C1CCNCC1",
                "CCO",
                "CC(=O)N",
                "C[N+](C)(C)C",
            ],
            nitro=True,
            expected_predicate="amine",
            expected_groups=[True, True, True, True, True, True, False, False, False],
            carrier_variable="N",
            carrier_atom="n",
            expected_aggregate="nitrogen_groups",
            expected_aggregate_groups=[
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
            ],
        )

    def test_quat_ammonion(self):
        self.check_buildable(
            ["C[N+](C)(C)C", "CN"],  # tetramethylammonium, methylamine
            nitro=True,
            expected_predicate="quaternary_ammonium",
            expected_groups=[True, False],
            carrier_variable="N",
            carrier_atom="n",
            expected_aggregate="nitrogen_groups",
            expected_aggregate_groups=[True, True],
        )

    def test_amide(self):
        self.check_buildable(
            ["CC(=O)N", "CCO"],  # acetamide, ethanol
            nitro=True,
            expected_predicate="amide",
            expected_groups=[True, False],
            carrier_variable="N",
            carrier_atom="n",
            expected_aggregate="nitrogen_groups",
            expected_aggregate_groups=[True, False],
        )

    def test_imine(self):
        self.check_buildable(
            ["C=NC", "CCO"],  # ethanimine, ethanol
            nitro=True,
            expected_predicate="imine",
            expected_groups=[True, False],
            carrier_variable="N",
        )

    def test_imide(self):
        self.check_buildable(
            ["O=C1CCC(=O)NC1=O", "CC(=O)O"],  # succinimide, acetic acid
            nitro=True,
            expected_predicate="imide",
            expected_groups=[True, False],
            carrier_variable="N",
        )

    def test_azide(self):
        self.check_buildable(
            ["CN=[N+]=[N-]", "CCO"],  # methyl azide, ethanol
            nitro=True,
            expected_predicate="azide",
            expected_groups=[True, False],
            carrier_variable="N1",
        )

    def test_azo(self):
        self.check_buildable(
            ["CCN=NC", "CCO"],  # dimethylazoethane, ethanol
            nitro=True,
            expected_predicate="azo",
            expected_groups=[True, False],
            predicate_arity=2,
        )

    def test_cyanate(self):
        self.check_buildable(
            ["CCOC#N", "CCO"],  # ethyl cyanate, ethanol
            nitro=True,
            expected_predicate="cyanate",
            expected_groups=[True, False],
            carrier_variable="C",
        )

    def test_isocyanate(self):
        self.check_buildable(
            ["CN=C=O", "CCO"],  # methyl isocyanate, ethanol
            nitro=True,
            expected_predicate="isocyanate",
            expected_groups=[True, False],
            carrier_variable="C",
        )

    def test_nitro_group(self):
        self.check_buildable(
            ["CC[N+](=O)[O-]", "CCO"],  # nitroethane, ethanol
            nitro=True,
            expected_predicate="nitro_group",
            expected_groups=[True, False],
            carrier_variable="N",
        )

    def test_nitro(self):
        self.check_buildable(
            [
                "CC[N+](=O)[O-]",
                "CC[N+]([O-])=O",
                "CCO",
                "CC(=O)N",
            ],
            nitro=True,
            expected_predicate="nitro",
            expected_groups=[True, True, False, False],
            carrier_variable="C",
            expected_aggregate="nitrogen_groups",
            expected_aggregate_groups=[True, True, False, True],
        )

    def test_nitrate(self):
        self.check_buildable(
            ["CO[N+](=O)[O-]", "CCO"],  # methyl nitrate, ethanol
            nitro=True,
            expected_predicate="nitrate",
            expected_groups=[True, False],
            carrier_variable="N",
        )

    def test_carbamate(self):
        self.check_buildable(
            ["COC(=O)N", "CC(=O)N"],  # methyl carbamate, acetamide
            nitro=True,
            expected_predicate="carbamate",
            expected_groups=[True, False],
            carrier_variable="N",
        )

    def test_aziridine(self):
        self.check_buildable(
            ["C1CN1", "C=NC"],  # aziridine, ethanimine
            nitro=True,
            expected_predicate="aziridine",
            expected_groups=[True, False],
            carrier_variable="N",
        )

    # Oxygens
    def test_oxygens_instantiation(self):
        args = {
            "layer_name": "oxy_layer",
            "param_size": (4, 4),
            "carbon": "c",
            "oxygen": "o",
            "hydrogen": "h",
        }
        hc = OxygenGroups(**args)
        self.assertIsInstance(hc, Model)
        self.assertEqual(hc.layer_name, "oxy_layer")

    def test_oxygens_missing_required_param(self):
        for key in OxygenGroups.required_keys:
            args = {
                "layer_name": "oxy_layer",
                "param_size": (4, 4),
                "carbon": "c",
                "oxygen": "o",
                "hydrogen": "h",
            }
            args.pop(key, None)
            with self.assertRaises((ValueError, TypeError)):
                OxygenGroups(**args)

    def test_oxygens_rule_structure(self):
        args = {
            "layer_name": "oxy_layer",
            "param_size": (4, 4),
            "carbon": "c",
            "oxygen": "o",
            "hydrogen": "h",
        }
        hc = OxygenGroups(**args)
        rules_str = str(hc)
        expected_predicates = [
            "alcoholic",
            "ketone",
            "aldehyde",
            "acyl_halide",
            "carboxylic_acid",
            "carboxylic_acid_anhydride",
            "ester",
            "carbonate_ester",
            "ether",
            "oxy_groups",
        ]
        for name in expected_predicates:
            self.assertIn(f"oxy_layer_{name}", rules_str)

    def test_alcoholic(self):
        self.check_buildable(
            ["CCO", "CC(=O)C"],  # ethanol, acetone
            oxy=True,
            expected_predicate="alcoholic",
            expected_groups=[True, False],
            carrier_variable="O",
        )

    def test_ketone(self):
        self.check_buildable(
            ["CC(=O)C", "CC=O"],  # acetone, acetaldehyde
            oxy=True,
            expected_predicate="ketone",
            expected_groups=[True, False],
            carrier_variable="C",
        )

    def test_aldehyde(self):
        self.check_buildable(
            ["CC=O", "CC(=O)C"],  # acetaldehyde, acetone
            oxy=True,
            expected_predicate="aldehyde",
            expected_groups=[True, False],
            carrier_variable="C",
        )

    def test_acyl_halide(self):
        self.check_buildable(
            ["CC(=O)Cl", "CC(=O)C"],  # acetyl chloride, acetone
            oxy=True,
            expected_predicate="acyl_halide",
            expected_groups=[True, False],
            carrier_variable="C",
        )

    def test_carboxylic_acid(self):
        self.check_buildable(
            ["CC(=O)O", "CC(=O)C"],  # acetic acid, acetone
            oxy=True,
            expected_predicate="carboxylic_acid",
            expected_groups=[True, False],
            carrier_variable="C",
        )

    def test_carboxylic_acid_anhydride(self):
        self.check_buildable(
            ["CC(=O)OC(=O)C", "CC(=O)O"],  # acetic anhydride, acetic acid
            oxy=True,
            expected_predicate="carboxylic_acid_anhydride",
            expected_groups=[True, False],
            predicate_arity=2,
        )

    def test_ester(self):
        self.check_buildable(
            ["CC(=O)OC", "CC(=O)O"],  # methyl acetate, acetic acid
            oxy=True,
            expected_predicate="ester",
            expected_groups=[True, False],
            carrier_variable="C",
        )

    def test_carbonate_ester(self):
        self.check_buildable(
            ["COC(=O)OC", "CC(=O)OC"],  # dimethyl carbonate, methyl acetate
            oxy=True,
            expected_predicate="carbonate_ester",
            expected_groups=[True, False],
            carrier_variable="C",
        )

    def test_ether(self):
        self.check_buildable(
            ["COC", "CCO", "CC(=O)OC"],  # ether, alcohol, ester
            oxy=True,
            expected_predicate="ether",
            expected_groups=[True, False, False],
            carrier_variable="O",
            expected_aggregate="oxy_groups",
            carrier_atom="o",
            expected_aggregate_groups=[True, True, True],
        )

    # Relaxations
    def test_relaxations_instantiation(self):
        args = {
            "layer_name": "relaxations",
            "param_size": (4, 4),
            "carbon": "c",
            "connection": "s",
        }
        hc = RelaxedFunctionalGroups(**args)
        self.assertIsInstance(hc, Model)
        self.assertEqual(hc.layer_name, "relaxations")

    def test_relaxations_missing_required_param(self):
        for key in RelaxedFunctionalGroups.required_keys:
            args = {
                "layer_name": "relaxations",
                "param_size": (4, 4),
                "carbon": "c",
                "connection": "s",
            }
            args.pop(key, None)
            with self.assertRaises((ValueError, TypeError)):
                RelaxedFunctionalGroups(**args)

    def test_relaxations_rule_structure(self):
        args = {
            "layer_name": "relaxations",
            "param_size": (4, 4),
            "carbon": "c",
            "connection": "s",
        }
        hc = RelaxedFunctionalGroups(**args)
        rules_str = str(hc)
        expected_predicates = [
            "relaxed_aliphatic_bonded",
            "relaxed_aromatic_bonded",
            "relaxed_carbonyl_group",
            "relaxed_benzene_ring",
            "potential_group",
            "carbonyl_derivatives",
            "relaxed_functional_group",
        ]
        for name in expected_predicates:
            self.assertIn(f"relaxations_{name}", rules_str)

    # Sulfurs
    def test_sulfurs_instantiation(self):
        args = {
            "layer_name": "sulfurs",
            "param_size": (4, 4),
            "carbon": "c",
            "sulfur": "s",
            "nitrogen": "n",
            "hydrogen": "h",
        }
        hc = SulfurGroups(**args)
        self.assertIsInstance(hc, Model)
        self.assertEqual(hc.layer_name, "sulfurs")

    def test_sulfurs_missing_required_param(self):
        for key in SulfurGroups.required_keys:
            args = {
                "layer_name": "sulfurs",
                "param_size": (4, 4),
                "carbon": "c",
                "sulfur": "s",
                "nitrogen": "n",
                "hydrogen": "h",
            }
            args.pop(key, None)
            with self.assertRaises((ValueError, TypeError)):
                SulfurGroups(**args)

    def test_sulfurs_rule_structure(self):
        args = {
            "layer_name": "sulfurs",
            "param_size": (4, 4),
            "carbon": "c",
            "sulfur": "s",
            "nitrogen": "n",
            "hydrogen": "h",
        }
        hc = SulfurGroups(**args)
        rules_str = str(hc)
        expected_predicates = [
            "thiocyanate",
            "isothiocyanate",
            "sulfide",
            "disulfide",
            "thiol",
            "sulfuric_groups",
        ]
        for name in expected_predicates:
            self.assertIn(f"sulfurs_{name}", rules_str)

    def test_thiocyanate(self):
        self.check_buildable(
            ["CSC#N", "CCO"],  # methyl thiocyanate, ethanol
            sulfuric=True,
            expected_predicate="thiocyanate",
            expected_groups=[True, False],
            carrier_variable="C",
            carrier_atom="c",
        )

    def test_isothiocyanate(self):
        self.check_buildable(
            ["CN=C=S", "CCO"],  # methyl isothiocyanate, ethanol
            sulfuric=True,
            expected_predicate="isothiocyanate",
            expected_groups=[True, False],
            carrier_variable="C",
            carrier_atom="c",
        )

    def test_sulfide(self):
        self.check_buildable(
            ["CSC", "CCO"],  # dimethyl sulfide, ethanol
            sulfuric=True,
            expected_predicate="sulfide",
            expected_groups=[True, False],
            carrier_variable="S",
            carrier_atom="s",
        )

    def test_disulfide(self):
        self.check_buildable(
            ["CSSC", "CCO"],  # dimethyl disulfide, ethanol
            sulfuric=True,
            expected_predicate="disulfide",
            expected_groups=[True, False],
            predicate_arity=2,
        )

    def test_thiol(self):
        self.check_buildable(
            ["CS", "CCO"],  # methanethiol, ethanol
            sulfuric=True,
            expected_predicate="thiol",
            expected_groups=[True, False],
            carrier_variable="S",
            carrier_atom="s",
            expected_aggregate="sulfuric_groups",
        )
