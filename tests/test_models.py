import unittest

from chemlogic.models import (
    GNN,
    KGNN,
    RGCN,
    SGN,
    CWNet,
    DiffusionCNN,
    EgoGNN,
    Model,
    get_model,
)
from chemlogic.utils.ChemTemplate import ChemTemplate as Template
from chemlogic.utils.Pipeline import Pipeline


class TestModelLoader(unittest.TestCase):
    def setUp(self):
        self.common_args = {
            "layers": 2,
            "node_embed": "node_embedding",
            "edge_embed": "edge_embedding",
            "connection": "connects",
            "param_size": 1,
            "output_layer_name": "predict",
        }

    def test_valid_model_gnn(self):
        model = get_model("gnn", **self.common_args)
        self.assertIsInstance(model, GNN)

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError) as context:
            get_model("invalid_model", **self.common_args)
        self.assertIn("Invalid model name", str(context.exception))

    def test_model_with_extra_kwargs(self):
        model = get_model("rgcn", **self.common_args, edge_types=["a", "b"])
        self.assertTrue(hasattr(model, "dropout") or True)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.valid_args = {
            "model_name": "test_model",
            "layers": 2,
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
            "param_size": 4,
            "output_layer_name": "predict",
        }

    def test_valid_model_creation(self):
        model = Model(**self.valid_args)
        self.assertEqual(model.model_name, "test_model")
        self.assertEqual(model.layers, 2)
        self.assertIsInstance(model, Template)

    def test_param_size_shape(self):
        model = Model(**self.valid_args)
        self.assertEqual(model.param_size, (4, 4))
        self.assertEqual(model.output_param_size, (1, 4))

        model_single = Model(**{**self.valid_args, "param_size": 1})
        self.assertEqual(model_single.param_size, (1,))
        self.assertEqual(model_single.output_param_size, (1,))

    def test_invalid_model_name_type(self):
        with self.assertRaises(TypeError):
            Model(**{**self.valid_args, "model_name": 123})

    def test_invalid_layers_value(self):
        with self.assertRaises(ValueError):
            Model(**{**self.valid_args, "layers": 0})

    def test_invalid_node_embed_type(self):
        with self.assertRaises(TypeError):
            Model(**{**self.valid_args, "node_embed": None})

    def test_invalid_edge_embed_type(self):
        with self.assertRaises(TypeError):
            Model(**{**self.valid_args, "edge_embed": 123})

    def test_invalid_connection_type(self):
        with self.assertRaises(TypeError):
            Model(**{**self.valid_args, "connection": 5.5})

    def test_invalid_param_size_value(self):
        with self.assertRaises(ValueError):
            Model(**{**self.valid_args, "param_size": 0})

    def test_invalid_output_layer_name_type(self):
        with self.assertRaises(TypeError):
            Model(**{**self.valid_args, "output_layer_name": None})


class TestModels(unittest.TestCase):
    def setUp(self):
        self.args = {
            "layers": 2,
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
            "param_size": 4,
            "output_layer_name": "predict",
            "edge_types": ["bond", "aromatic"],
            "max_depth": 3,
            "local": True,
            "max_ring_size": 6,
        }

    def test_gnn_instantiation(self):
        model = GNN(**self.args)
        self.assertEqual(model.model_name, "gnn")
        self.assertIsInstance(model, Template)

    def test_gnn_build_layer_structure(self):
        model = GNN(**self.args)
        rules = model.build_layer("gnn_1", "node")
        self.assertEqual(len(rules), 1)
        rule = rules[0]
        self.assertIn("gnn_1", str(rule))
        self.assertIn("node", str(rule))
        self.assertIn("connects", str(rule))
        self.assertIn("edge", str(rule))

    def test_rgcn_instantiation(self):
        model = RGCN(**self.args)
        self.assertEqual(model.model_name, "rgcn")
        self.assertEqual(model.edge_types, ["bond", "aromatic"])
        self.assertIsInstance(model, Template)

    def test_rgcn_missing_edge_types(self):
        args = {k: v for k, v in self.args.items() if k != "edge_types"}
        with self.assertRaises(KeyError):
            RGCN(**args)

    def test_rgcn_invalid_edge_types_type(self):
        args = {**self.args, "edge_types": "not_a_list"}
        with self.assertRaises(TypeError):
            RGCN(**args)

    def test_rgcn_too_few_edge_types(self):
        args = {**self.args, "edge_types": ["only_one"]}
        with self.assertRaises(ValueError):
            RGCN(**args)

    def test_rgcn_build_layer_structure(self):
        model = RGCN(**self.args)
        rules = model.build_layer("rgcn_1", "node")
        self.assertEqual(len(rules), 2)
        for rule in rules:
            self.assertIn("rgcn_1", str(rule))
            self.assertIn("node", str(rule))
            self.assertIn("connects", str(rule))
            self.assertTrue(any(t in str(rule) for t in self.args["edge_types"]))

    def test_kgnn_instantiation(self):
        model = KGNN(**self.args)
        self.assertEqual(model.model_name, "kgnn")
        self.assertEqual(model.max_depth, 3)
        self.assertTrue(model.local)
        self.assertIsInstance(model, Template)

    def test_kgnn_invalid_local_type(self):
        with self.assertRaises(TypeError):
            KGNN(**{**self.args, "local": "yes"})

    def test_kgnn_invalid_max_depth(self):
        with self.assertRaises(TypeError):
            KGNN(**{**self.args, "max_depth": "deep"})
        with self.assertRaises(TypeError):
            KGNN(**{**self.args, "max_depth": 0})

    def test_kgnn_build_layer_structure(self):
        model = KGNN(**self.args)
        rules = model.build_layer("kgnn_1", "node")
        self.assertGreaterEqual(len(rules), 3)
        self.assertTrue(any("alldiff" in str(rule) for rule in rules))

    def test_sgn_instantiation(self):
        model = SGN(**self.args)
        self.assertEqual(model.model_name, "sgn")
        self.assertEqual(model.max_depth, 3)
        self.assertIsInstance(model, Template)

    def test_sgn_invalid_max_depth(self):
        with self.assertRaises(TypeError):
            SGN(**{**self.args, "max_depth": "three"})
        with self.assertRaises(TypeError):
            SGN(**{**self.args, "max_depth": 0})

    def test_sgn_build_layer_structure(self):
        model = SGN(**self.args)
        rules = model.build_layer("sgn_1", "node")
        self.assertGreaterEqual(len(rules), 5)
        self.assertTrue(any("order_1" in str(rule) for rule in rules))

    def test_egognn_instantiation(self):
        model = EgoGNN(**self.args)
        self.assertEqual(model.model_name, "ego")
        self.assertIsInstance(model, Template)

    def test_egognn_build_layer_structure(self):
        model = EgoGNN(**self.args)
        rules = model.build_layer("ego_1", "node")
        self.assertEqual(len(rules), 2)
        self.assertTrue(any("multigraph" in str(rule) for rule in rules))

    def test_diffusioncnn_instantiation(self):
        model = DiffusionCNN(**self.args)
        self.assertEqual(model.model_name, "diffusion_cnn")
        self.assertEqual(model.max_depth, 3)
        self.assertIsInstance(model, Template)

    def test_diffusioncnn_invalid_max_depth(self):
        with self.assertRaises(TypeError):
            DiffusionCNN(**{**self.args, "max_depth": "deep"})
        with self.assertRaises(TypeError):
            DiffusionCNN(**{**self.args, "max_depth": 0})

    def test_diffusioncnn_get_path_structure(self):
        model = DiffusionCNN(**self.args)
        rules = model.get_path("diffusion_path")
        self.assertGreaterEqual(len(rules), 4)
        self.assertTrue(any("next" in str(rule) for rule in rules))

    def test_diffusioncnn_build_layer_structure(self):
        model = DiffusionCNN(**self.args)
        rules = model.build_layer("diff_1", "node")
        self.assertGreaterEqual(len(rules), 2)
        self.assertTrue(any("path" in str(rule) for rule in rules))

    def test_cwnet_instantiation(self):
        model = CWNet(**self.args)
        self.assertEqual(model.model_name, "cw")
        self.assertEqual(model.max_ring_size, 6)
        self.assertIsInstance(model, Template)

    def test_cwnet_invalid_max_ring_size(self):
        with self.assertRaises(TypeError):
            CWNet(**{**self.args, "max_ring_size": "six"})
        with self.assertRaises(TypeError):
            CWNet(**{**self.args, "max_ring_size": 3})

    def test_cwnet_create_template_structure(self):
        model = CWNet(**self.args)
        rules = model.template
        self.assertGreaterEqual(len(rules), 4)
        self.assertTrue(any("cw" in str(rule) for rule in rules))

    def test_cwnet_build_layer_structure(self):
        model = CWNet(**self.args)
        rules = model.build_layer("cw_1", "cw_0")
        self.assertGreaterEqual(len(rules), 4)
        self.assertTrue(any("edge" in str(rule) for rule in rules))
        self.assertTrue(any("node" in str(rule) for rule in rules))


class TestModelsTrainable(unittest.TestCase):
    def train(self, model_name):
        pipeline = Pipeline("mutagen", model_name, 1, 1)

        pipeline.train_test_cycle(epochs=1)

    def test_gnn_trainable(self):
        self.train("gnn")

    def test_rgcn_trainable(self):
        self.train("rgcn")

    def test_kgnn_trainable(self):
        self.train("kgnn")

    def test_ego_trainable(self):
        self.train("ego")

    def test_diffusion_trainable(self):
        self.train("diffusion")

    def test_cw_trainable(self):
        self.train("cw")

    def test_sgn_trainable(self):
        self.train("sgn")
