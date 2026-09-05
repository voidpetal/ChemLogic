import unittest
from unittest.mock import MagicMock, patch

from neuralogic.core import R, V

from chemlogic.utils.Pipeline import ArchitectureType, Pipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.default_args = {
            "dataset_name": "mutagen",
            "model_name": "gnn",
            "param_size": 4,
            "layers": 2,
        }

    @patch("chemlogic.utils.Pipeline.get_dataset")
    @patch("chemlogic.models.models.get_model")
    @patch("chemlogic.utils.Pipeline.get_chem_rules")
    def test_pipeline_parallel_architecture(
        self, mock_get_chem_rules, mock_get_model, mock_get_dataset
    ):
        mock_get_dataset.return_value = MagicMock(
            node_embed="node",
            edge_embed="edge",
            connection="connects",
            bond_types=["bond"],
            halogens=[],
            single_bond="sb",
            double_bond="db",
            triple_bond="tb",
            aromatic_bonds=[],
            carbon="C",
            hydrogen="H",
            oxygen="O",
            nitrogen="N",
            sulfur="S",
            key_atom_type=[],
            atom_types=[],
            aliphatic_bonds=[],
            data=[],
        )
        mock_get_model.return_value = []
        pipeline = Pipeline(
            **self.default_args,
            architecture=ArchitectureType.BARE,
            chem_rules=False,
        )
        self.assertIsNotNone(pipeline.template)
        mock_get_chem_rules.assert_not_called()

    @patch("chemlogic.utils.Pipeline.get_dataset")
    @patch("chemlogic.models.models.get_model")
    @patch("chemlogic.utils.Pipeline.get_chem_rules")
    def test_custom_template_is_adapted_to_chem_io(
        self, mock_get_chem_rules, mock_get_model, mock_get_dataset
    ):
        mock_get_dataset.return_value = MagicMock(
            node_embed="node",
            edge_embed="edge",
            connection="connects",
            bond_types=["bond"],
            halogens=[],
            single_bond="sb",
            double_bond="db",
            triple_bond="tb",
            aromatic_bonds=[],
            carbon="C",
            hydrogen="H",
            oxygen="O",
            nitrogen="N",
            sulfur="S",
            key_atom_type=[],
            atom_types=[],
            aliphatic_bonds=[],
            data=[],
        )
        mock_get_dataset.return_value.__add__.side_effect = lambda other: other
        mock_get_model.return_value = []
        mock_get_chem_rules.return_value = []
        custom_rule = R.custom_feature(V.X) <= R.custom_input(V.X)
        custom_template = [custom_rule]

        for architecture, expected_input in (
            (ArchitectureType.BARE, "node"),
            (ArchitectureType.CCE, "node"),
            (ArchitectureType.CCD, "kb_features"),
        ):
            with self.subTest(architecture=architecture):
                pipeline = Pipeline(
                    **self.default_args,
                    architecture=architecture,
                    custom_rules=custom_template,
                    custom_input="custom_input",
                    custom_output="custom_feature",
                )

                rules = [str(rule) for rule in pipeline.template]
                self.assertIn(str(custom_rule), rules)
                self.assertTrue(
                    any(
                        "custom_input" in rule and expected_input in rule
                        for rule in rules
                    )
                )
                expected_output = (
                    "kb_features" if architecture == ArchitectureType.CCE else "predict"
                )
                self.assertTrue(
                    any(
                        expected_output in rule and "custom_feature" in rule
                        for rule in rules
                    )
                )

        self.assertEqual(mock_get_chem_rules.call_count, 3)
        for call in mock_get_chem_rules.call_args_list:
            self.assertEqual(call.kwargs["hydrocarbons"], False)
            self.assertEqual(call.kwargs["oxy"], False)
            self.assertEqual(call.kwargs["nitro"], False)
            self.assertEqual(call.kwargs["sulfuric"], False)
            self.assertEqual(call.kwargs["relaxations"], False)

        Pipeline(
            **self.default_args,
            chem_rules=(True, True, False, False, True),
            custom_rules=custom_template,
            custom_input="custom_input",
            custom_output="custom_feature",
        )
        selected_call = mock_get_chem_rules.call_args
        self.assertEqual(selected_call.kwargs["hydrocarbons"], True)
        self.assertEqual(selected_call.kwargs["oxy"], True)
        self.assertEqual(selected_call.kwargs["relaxations"], True)

    @patch("chemlogic.utils.Pipeline.get_dataset")
    @patch("chemlogic.models.models.get_model")
    def test_custom_rules_require_input_and_output(
        self, mock_get_model, mock_get_dataset
    ):
        mock_get_dataset.return_value = MagicMock(
            node_embed="node",
            edge_embed="edge",
            connection="connects",
            bond_types=[],
            halogens=[],
            single_bond="sb",
            double_bond="db",
            triple_bond="tb",
            aromatic_bonds=[],
            carbon="C",
            hydrogen="H",
            oxygen="O",
            nitrogen="N",
            sulfur="S",
            key_atom_type=[],
            atom_types=[],
            aliphatic_bonds=[],
            data=[],
        )
        mock_get_model.return_value = []

        with self.assertRaises(TypeError):
            Pipeline(**self.default_args, custom_rules=[])

    def test_pipeline_invalid_architecture(self):
        with self.assertRaises(ValueError):
            Pipeline(**self.default_args, architecture="invalid")

    @patch("chemlogic.datasets.datasets.get_dataset")
    @patch("chemlogic.models.models.get_model")
    @patch("chemlogic.utils.Pipeline.ChemTemplate.build")
    def test_train_test_cycle(self, mock_evaluator, mock_model, mock_dataset):
        mock_dataset.return_value = MagicMock(
            node_embed="node",
            edge_embed="edge",
            connection="connects",
            bond_types=["bond"],
            halogens=[],
            single_bond="sb",
            double_bond="db",
            triple_bond="tb",
            aromatic_bonds=[],
            carbon="C",
            hydrogen="H",
            oxygen="O",
            nitrogen="N",
            sulfur="S",
            key_atom_type=[],
            atom_types=[],
            aliphatic_bonds=[],
            data=[],
        )
        mock_model.return_value = []

        mock_eval = MagicMock()
        mock_eval.build_dataset.return_value = [1, 2, 3, 4]
        mock_evaluator.return_value = mock_eval

        pipeline = Pipeline(**self.default_args)
        pipeline._train_model = lambda e, d, ep, es, ed, **kwargs: [0.1, 0.2, 0.15]
        pipeline._evaluate_model = lambda e, d: (0.1, 0.9)

        result = pipeline.train_test_cycle()
        self.assertEqual(len(result), 4)

    def test_train_model_early_stopping(self):
        evaluator = MagicMock()
        evaluator.loss.side_effect = iter(
            [
                0.4,
                0.35,
                0.34,
                0.34,
                0.34,
                0.34,
                0.34,
                0.34,
                0.34,
                0.34,
                0.34,
                0.34,
            ]
        )
        pipeline = Pipeline(**self.default_args)
        losses = pipeline._train_model(
            evaluator, train_dataset=[], epochs=20, early_stopping_rounds=5
        )
        self.assertLessEqual(len(losses), 20)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))

    def test_evaluate_model_accuracy_and_auroc(self):
        evaluator = MagicMock()
        evaluator.test.return_value = [0.9, 0.1, 0.8, 0.2]
        sample_mock = MagicMock()
        sample_mock.java_sample.target.value = 1
        sample_mock_other_class = MagicMock()
        sample_mock_other_class.java_sample.target.value = 0
        test_dataset = [
            sample_mock,
            sample_mock,
            sample_mock_other_class,
            sample_mock_other_class,
        ]

        pipeline = Pipeline(**self.default_args)
        loss, auroc = pipeline._evaluate_model(evaluator, test_dataset)

        self.assertIsInstance(loss, float)
        self.assertIsInstance(auroc, float)

    @patch("chemlogic.datasets.datasets.get_dataset")
    @patch("chemlogic.models.models.get_model")
    def test_inference_raises_if_not_trained(self, mock_get_model, mock_get_dataset):
        """If the pipeline has not been trained (no evaluator), inference should raise ValueError."""
        mock_get_dataset.return_value = MagicMock(
            node_embed="node",
            edge_embed="edge",
            connection="connects",
            bond_types=["bond"],
            halogens=[],
            single_bond="sb",
            double_bond="db",
            triple_bond="tb",
            aromatic_bonds=[],
            carbon="C",
            hydrogen="H",
            oxygen="O",
            nitrogen="N",
            sulfur="S",
            key_atom_type=[],
            atom_types=[],
            aliphatic_bonds=[],
            data=[],
        )
        mock_get_model.return_value = []

        pipeline = Pipeline(**self.default_args)

        # ensure evaluator attribute is not present
        if hasattr(pipeline, "evaluator"):
            delattr(pipeline, "evaluator")

        with self.assertRaises(ValueError):
            pipeline.inference(["C"])

    @patch("chemlogic.datasets.datasets.get_dataset")
    @patch("chemlogic.models.models.get_model")
    def test_inference_returns_predictions(self, mock_get_model, mock_get_dataset):
        """Inference should return the predictions produced by the evaluator in the same order."""
        # Dataset used for Pipeline.__init__
        mock_get_dataset.return_value = MagicMock(
            node_embed="node",
            edge_embed="edge",
            connection="connects",
            bond_types=["bond"],
            halogens=[],
            single_bond="sb",
            double_bond="db",
            triple_bond="tb",
            aromatic_bonds=[],
            carbon="C",
            hydrogen="H",
            oxygen="O",
            nitrogen="N",
            sulfur="S",
            key_atom_type=[],
            atom_types=[],
            aliphatic_bonds=[],
            data=[],
            dataset_name="mutagen",
            param_size=4,
        )
        mock_get_model.return_value = []

        pipeline = Pipeline(**self.default_args)

        # Create a fake evaluator and built dataset for inference
        evaluator = MagicMock()
        built_dataset = MagicMock()
        built_dataset.samples = [1, 2, 3]
        evaluator.build_dataset.return_value = built_dataset
        # evaluator.test will be iterated over; return same-length iterable
        evaluator.test.return_value = [0.1, 0.2, 0.3]

        pipeline.evaluator = evaluator

        smiles = ["C1=CC=CC=C1", "O", "N"]
        preds = pipeline.inference(smiles)

        self.assertEqual(preds, [0.1, 0.2, 0.3])
