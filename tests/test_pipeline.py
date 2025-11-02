import unittest
from unittest.mock import MagicMock, patch

from chemlogic.utils.Pipeline import ArchitectureType, Pipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.default_args = {
            "dataset_name": "mutagen",
            "model_name": "gnn",
            "param_size": 4,
            "layers": 2,
        }

    @patch("chemlogic.datasets.datasets.get_dataset")
    @patch("chemlogic.models.models.get_model")
    def test_pipeline_parallel_architecture(self, mock_get_model, mock_get_dataset):
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
        pipeline = Pipeline(**self.default_args, architecture=ArchitectureType.BARE)
        self.assertIsNotNone(pipeline.template)

    def test_pipeline_invalid_architecture(self):
        with self.assertRaises(ValueError):
            Pipeline(**self.default_args, architecture="invalid")

    @patch("chemlogic.datasets.datasets.get_dataset")
    @patch("chemlogic.models.models.get_model")
    @patch("neuralogic.nn.get_evaluator")
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
        mock_eval.build_dataset.return_value.samples = [1, 2, 3, 4]
        mock_evaluator.return_value = mock_eval

        pipeline = Pipeline(**self.default_args)
        pipeline._train_model = lambda e, d, ep, es, ed: [0.1, 0.2, 0.15]
        pipeline._evaluate_model = lambda e, d: (0.1, 0.9)

        result = pipeline.train_test_cycle()
        self.assertEqual(len(result), 4)

    def test_train_model_early_stopping(self):
        evaluator = MagicMock()
        evaluator.train.return_value = iter(
            [
                (0.4, 1),
                (0.35, 1),
                (0.34, 1),
                (0.34, 1),
                (0.34, 1),
                (0.34, 1),
                (0.34, 1),
                (0.34, 1),
                (0.34, 1),
                (0.34, 1),
                (0.34, 1),
                (0.34, 1),
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
