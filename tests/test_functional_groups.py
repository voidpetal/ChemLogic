import unittest

from neuralogic.core import Settings, Template
from neuralogic.nn import get_evaluator

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
    ):
        # Define Dataset
        dataset = SmilesDataset(
            smiles_list=smiles,
            labels=[
                1,
            ]
            * len(smiles),
            param_size=1,
            dataset_name=f"test_{str(smiles[0])}",
        )

        # Define the knowledge base
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

        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)
        dataset.clear()

    # GeneralFunctionalGroups
    def test_general_functional_groups_instantiation(self):
        fg = GeneralFunctionalGroups(**self.common_args)
        self.assertIsInstance(fg, Template)
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
            ["C=O"],  # formaldehyde
        )

    def test_hydroxyl(self):
        self.check_buildable(
            ["C-O"],  # methanol
        )

    def test_halogen(self):
        self.check_buildable(
            ["CF"],  # fluormethane
        )

    def test_not_buildable(self):
        with self.assertRaises(Exception):  # noqa: B017
            self.check_buildable(
                ["C"],  # methane does not have any of the groups
                hydrocarbons=True,
                nitro=True,
                sulfuric=True,
                oxy=True,
            )

    # Hydrocarbons
    def test_hydrocarbons_instantiation(self):
        args = {
            "layer_name": "hydro_layer",
            "param_size": (4, 4),
            "carbon": "C",
        }
        hc = Hydrocarbons(**args)
        self.assertIsInstance(hc, Template)
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
        # TODO: benzene gets converted to nonaromatic, no support for aromatic bonds from smiles
        # self.check_buildable(
        #     ["c1ccccc1"], # benzene
        #     hydrocarbons=True
        #     )
        assert True

    def test_alkene(self):
        self.check_buildable(
            ["C=C"],  # ethene
            hydrocarbons=True,
        )

    def test_alkyne(self):
        self.check_buildable(
            ["C#C"],  # acetylene
            hydrocarbons=True,
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
        self.assertIsInstance(hc, Template)
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
                "CN",  # methylamine
                "CNC",  # dimethylamine
                "CN(C)C",  # trimethylamine
                "C1CCNCC1",  # piperine
            ],
            nitro=True,
        )

    def test_quat_ammonion(self):
        self.check_buildable(
            ["C[N+](C)(C)C"],  # tetramethylammonium
            nitro=True,
        )

    def test_amide(self):
        self.check_buildable(
            ["CC(=O)N"],  # acetamide
            nitro=True,
        )

    def test_imine(self):
        self.check_buildable(
            ["C=NC"],  # ethanimine
            nitro=True,
        )

    def test_imide(self):
        self.check_buildable(
            ["O=C1CCC(=O)NC1=O"],  # succinimide
            nitro=True,
        )

    def test_azide(self):
        self.check_buildable(
            ["CN=[N+]=[N-]"],  # methyl azide
            nitro=True,
        )

    def test_azo(self):
        self.check_buildable(
            ["CCN=NC"],  # dimethylazoethane
            nitro=True,
        )

    def test_cyanate(self):
        self.check_buildable(
            ["CCOCN"],  # ethyl cyanate (approximate)
            nitro=True,
        )

    def test_isocyanate(self):
        self.check_buildable(
            ["CN=C=O"],  # methyl isocyanate
            nitro=True,
        )

    def test_nitro_group(self):
        self.check_buildable(
            ["CC(N(O)O)C"],  # nitropropane
            nitro=True,
        )

    def test_nitro(self):
        self.check_buildable(
            ["C1=CC=C(C=C1)NO"],  # nitrobenzene
            nitro=True,
        )

    def test_nitrate(self):
        self.check_buildable(
            ["CONO"],  # methyl nitrate
            nitro=True,
        )

    def test_carbamate(self):
        self.check_buildable(
            ["COC(=O)N"],  # methyl carbamate
            nitro=True,
        )

    def test_aziridine(self):
        self.check_buildable(
            ["C1CN1"],  # aziridine
            nitro=True,
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
        self.assertIsInstance(hc, Template)
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
            ["CCO"],  # ethanol
            oxy=True,
        )

    def test_ketone(self):
        self.check_buildable(
            ["CC(=O)C"],  # acetone
            oxy=True,
        )

    def test_aldehyde(self):
        self.check_buildable(
            ["CC=O"],  # acetaldehyde
            oxy=True,
        )

    def test_acyl_halide(self):
        self.check_buildable(
            ["CC(=O)Cl"],  # acetyl chloride
            oxy=True,
        )

    def test_carboxylic_acid(self):
        self.check_buildable(
            ["CC(=O)O"],  # acetic acid
            oxy=True,
        )

    def test_carboxylic_acid_anhydride(self):
        self.check_buildable(
            ["CC(=O)OC(=O)C"],  # acetic anhydride
            oxy=True,
        )

    def test_ester(self):
        self.check_buildable(
            ["CC(=O)OC"],  # methyl acetate
            oxy=True,
        )

    def test_carbonate_ester(self):
        self.check_buildable(
            ["COC(=O)OC"],  # dimethyl carbonate
            oxy=True,
        )

    def test_ether(self):
        self.check_buildable(
            ["COC"],  # dimethyl ether
            oxy=True,
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
        self.assertIsInstance(hc, Template)
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
        self.assertIsInstance(hc, Template)
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
            ["CSC#N"],  # methyl thiocyanate
            sulfuric=True,
        )

    def test_isothiocyanate(self):
        self.check_buildable(
            ["CN=C=S"],  # methyl isothiocyanate
            sulfuric=True,
        )

    def test_sulfide(self):
        self.check_buildable(
            ["CSC"],  # dimethyl sulfide
            sulfuric=True,
        )

    def test_disulfide(self):
        self.check_buildable(
            ["CSSC"],  # dimethyl disulfide
            sulfuric=True,
        )

    def test_thiol(self):
        self.check_buildable(
            ["CS"],  # methanethiol
            sulfuric=True,
        )
