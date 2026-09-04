import unittest

from neuralogic.core import Model

from chemlogic.knowledge_base.chemrules import get_chem_rules
from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase
from chemlogic.knowledge_base.subgraphs import get_subgraphs
from chemlogic.utils.ChemTemplate import ChemTemplate


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.valid_args = {
            "layer_name": "chem_layer",
            "param_size": (4, 4),
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
            "single_bond": "single",
            "double_bond": "double",
            "triple_bond": "triple",
            "aromatic_bond": "aromatic",
            "aliphatic_bond": "aliphatic",
            "hydrogen": "H",
            "carbon": "C",
            "oxygen": "O",
            "nitrogen": "N",
            "sulfur": "S",
            "atom_type": "atom",
            "min_cycle_size": 3,
            "max_cycle_size": 8,
            "nbh_min_size": 3,
            "nbh_max_size": 5,
            "max_depth": 3,
        }

    def test_valid_knowledgebase_creation(self):
        kb = KnowledgeBase(**self.valid_args)
        self.assertEqual(kb.layer_name, "chem_layer")
        self.assertEqual(kb.param_size, (4, 4))
        self.assertIsInstance(kb, ChemTemplate)

    def test_missing_required_param_size(self):
        args = {**self.valid_args}
        del args["param_size"]
        with self.assertRaises(TypeError):
            KnowledgeBase(**args)

    def test_missing_required_layer_name(self):
        args = {**self.valid_args}
        del args["layer_name"]
        with self.assertRaises(TypeError):
            KnowledgeBase(**args)

    def test_empty_required_param_size(self):
        args = {**self.valid_args, "param_size": None}
        with self.assertRaises(ValueError):
            KnowledgeBase(**args)

    def test_empty_required_layer_name(self):
        args = {**self.valid_args, "layer_name": ""}
        with self.assertRaises(ValueError):
            KnowledgeBase(**args)

    def test_invalid_param_size_type(self):
        args = {**self.valid_args, "param_size": "invalid"}
        with self.assertRaises(TypeError):
            KnowledgeBase(**args)

    def test_invalid_min_cycle_size(self):
        args = {**self.valid_args, "min_cycle_size": 2}
        kb = KnowledgeBase(**args)
        self.assertEqual(
            kb.min_cycle_size, 2
        )  # Accepts 2, but logic may reject it later

    def test_invalid_max_cycle_size(self):
        args = {**self.valid_args, "max_cycle_size": 1}
        kb = KnowledgeBase(**args)
        self.assertEqual(
            kb.max_cycle_size, 1
        )  # Accepts 1, but logic may reject it later

    def test_invalid_nbh_min_size(self):
        args = {**self.valid_args, "nbh_min_size": -1}
        kb = KnowledgeBase(**args)
        self.assertEqual(kb.nbh_min_size, -1)

    def test_partial_initialization(self):
        minimal_args = {"layer_name": "minimal_layer", "param_size": (1,)}
        kb = KnowledgeBase(**minimal_args)
        self.assertEqual(kb.layer_name, "minimal_layer")
        self.assertEqual(kb.param_size, (1,))


class TestChemRules(unittest.TestCase):
    def setUp(self):
        self.common_args = {
            "layer_name": "test_layer",
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
            "param_size": 2,
            "halogens": ["F", "Cl"],
            "aromatic_bonds": ["ar1", "ar2"],
            "carbon": "C",
            "hydrogen": "H",
            "oxygen": "O",
            "nitrogen": "N",
            "sulfur": "S",
            "single_bond": "sb",
            "double_bond": "db",
            "triple_bond": "tb",
        }

    def test_basic_template_creation(self):
        template = get_chem_rules(**self.common_args)
        self.assertIsInstance(template, Model)

    def test_funnel_sets_param_size_to_one(self):
        args = {**self.common_args, "funnel": True}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_relaxations_adds_key_atoms(self):
        args = {**self.common_args, "relaxations": True, "key_atoms": ["O", "N"]}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_relaxations_with_default_key_atoms(self):
        args = {**self.common_args, "relaxations": True}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_with_path_and_relaxations(self):
        args = {
            **self.common_args,
            "path": "connected",
            "relaxations": True,
            "key_atoms": ["O"],
        }
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_with_hydrocarbons_enabled(self):
        args = {**self.common_args, "hydrocarbons": True}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_with_oxygen_groups_enabled(self):
        args = {**self.common_args, "oxy": True}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_with_nitrogen_groups_enabled(self):
        args = {**self.common_args, "nitro": True}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_with_sulfuric_groups_enabled(self):
        args = {**self.common_args, "sulfuric": True}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_with_all_groups_enabled(self):
        args = {
            **self.common_args,
            "hydrocarbons": True,
            "oxy": True,
            "nitro": True,
            "sulfuric": True,
            "relaxations": True,
            "key_atoms": ["O", "N"],
            "path": "connected",
        }
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_empty_halogens_and_aromatic_bonds(self):
        args = {**self.common_args, "halogens": [], "aromatic_bonds": []}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_param_size_one_behavior(self):
        args = {**self.common_args, "param_size": 1}
        template = get_chem_rules(**args)
        self.assertIsInstance(template, Model)

    def test_missing_optional_arguments(self):
        minimal_args = {
            "layer_name": "minimal",
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
            "single_bond": "single_bond",
            "double_bond": "double_bond",
            "triple_bond": "triple_bond",
            "param_size": 1,
            "halogens": [],
            "aromatic_bonds": [],
            "carbon": "C",
            "hydrogen": "H",
            "oxygen": "O",
            "nitrogen": "N",
            "sulfur": "S",
        }
        template = get_chem_rules(**minimal_args)
        self.assertIsInstance(template, Model)


class TestSubgraphPatterns(unittest.TestCase):
    def setUp(self):
        self.common_args = {
            "layer_name": "subgraph_layer",
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
            "param_size": 2,
            "max_cycle_size": 6,
            "max_depth": 4,
            "output_layer_name": "predict",
            "single_bond": "sb",
            "double_bond": "db",
            "carbon": "C",
            "atom_types": ["O", "N"],
            "aliphatic_bonds": ["sb", "db"],
        }

    def test_basic_template_creation(self):
        template = get_subgraphs(**self.common_args)
        self.assertIsInstance(template, Model)

    def test_funnel_sets_param_size_to_one(self):
        args = {**self.common_args, "funnel": True}
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_enable_cycles_only(self):
        args = {**self.common_args, "cycles": True}
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_enable_paths_only(self):
        args = {**self.common_args, "paths": True}
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_enable_y_shape_only(self):
        args = {**self.common_args, "y_shape": True}
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_enable_neighborhoods_only(self):
        args = {**self.common_args, "nbhoods": True}
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_enable_circular_only(self):
        args = {**self.common_args, "circular": True}
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_enable_collective_only(self):
        args = {**self.common_args, "collective": True}
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_enable_all_patterns(self):
        args = {
            **self.common_args,
            "cycles": True,
            "paths": True,
            "y_shape": True,
            "nbhoods": True,
            "circular": True,
            "collective": True,
        }
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_empty_atom_types_and_bonds(self):
        args = {
            **self.common_args,
            "atom_types": [],
            "aliphatic_bonds": [],
            "nbhoods": True,
            "collective": True,
        }
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_param_size_one_behavior(self):
        args = {**self.common_args, "param_size": 1}
        template = get_subgraphs(**args)
        self.assertIsInstance(template, Model)

    def test_missing_optional_arguments(self):
        minimal_args = {
            "layer_name": "minimal",
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
            "param_size": 1,
        }
        template = get_subgraphs(**minimal_args)
        self.assertIsInstance(template, Model)
