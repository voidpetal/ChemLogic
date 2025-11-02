import unittest

from neuralogic.core import Settings, Template
from neuralogic.nn import get_evaluator

from chemlogic.datasets import (
    SmilesDataset,
)
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
from chemlogic.knowledge_base.subgraphs import get_subgraphs


class TestSubgraphPatternModules(unittest.TestCase):
    def setUp(self):
        self.common_args = {
            "layer_name": "test_layer",
            "param_size": (4, 4),
            "node_embed": "node",
            "edge_embed": "edge",
            "connection": "connects",
        }

    def check_buildable(
        self,
        smiles,
        cycles=False,
        paths=False,
        y_shape=False,
        nbhoods=False,
        circular=False,
        collective=False,
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
        kb = get_subgraphs(
            "sub",
            dataset.node_embed,
            dataset.edge_embed,
            dataset.connection,
            1,
            single_bond=dataset.single_bond,
            double_bond=dataset.double_bond,
            carbon=dataset.carbon,
            atom_types=dataset.atom_types,
            aliphatic_bonds=dataset.aliphatic_bonds,
            cycles=cycles,
            paths=paths,
            y_shape=y_shape,
            nbhoods=nbhoods,
            circular=circular,
            collective=collective,
        )

        dataset += kb
        dataset.flatten()

        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)
        dataset.clear()

    # CircularPatterns
    def test_circular_instantiation(self):
        args = {
            **self.common_args,
            "single_bond": "sb",
            "double_bond": "db",
            "carbon": "C",
        }
        pattern = CircularPatterns(**args)
        self.assertIsInstance(pattern, Template)

    def test_circular_missing_required_param(self):
        required = CircularPatterns.required_keys
        for key in required:
            args = {
                **self.common_args,
                "single_bond": "sb",
                "double_bond": "db",
                "carbon": "C",
            }
            args.pop(key, None)
            with self.assertRaises((ValueError, TypeError)):
                CircularPatterns(**args)

    def test_brick_buildable(self):
        self.check_buildable(
            [
                "C1=CC=C1"  # cyclobutadiene
            ],
            circular=True,
        )

    # CollectivePatterns
    def test_collective_instantiation(self):
        args = {
            **self.common_args,
            "aliphatic_bond": "sb",
            "carbon": "C",
            "max_depth": 3,
        }
        pattern = CollectivePatterns(**args)
        self.assertIsInstance(pattern, Template)

    def test_collective_missing_required_param(self):
        required = CollectivePatterns.required_keys
        for key in required:
            args = {
                **self.common_args,
                "aliphatic_bond": "sb",
                "carbon": "C",
                "max_depth": 3,
            }
            args.pop(key, None)
            with self.assertRaises((ValueError, TypeError)):
                CollectivePatterns(**args)

    def test_bridge_buildable(self):
        self.check_buildable(
            ["C(C1CCCC1)C1CCCC1"],  # dicyclopentylmethane
            collective=True,
        )

    def test_shared_atom_buildable(self):
        self.check_buildable(
            [
                "C1CC2(C1)CCCC2"  # spiro[3.4]octane
            ],
            collective=True,
        )

    def test_aliphatic_chain_buildable(self):
        self.check_buildable(
            [
                "CCCCCCCCCC"  # decane
            ],
            collective=True,
        )

    # CyclePattern
    def test_cycle_instantiation(self):
        args = {
            **self.common_args,
            "min_cycle_size": 3,
            "max_cycle_size": 6,
        }
        pattern = CyclePattern(**args)
        self.assertIsInstance(pattern, Template)

    def test_cycle_invalid_min_cycle_size(self):
        args = {
            **self.common_args,
            "min_cycle_size": 2,
            "max_cycle_size": 6,
        }
        with self.assertRaises(ValueError):
            CyclePattern(**args)

    def test_cycle_invalid_max_cycle_size(self):
        args = {
            **self.common_args,
            "min_cycle_size": 3,
            "max_cycle_size": 3,
        }
        with self.assertRaises(ValueError):
            CyclePattern(**args)

    def test_cycles_buildable(self):
        self.check_buildable(
            [
                "C1CCCCC1"  # cyclohexane
            ],
            cycles=True,
        )

    # NeighborhoodPatterns
    def test_neighborhood_instantiation(self):
        args = {
            **self.common_args,
            "atom_type": "atom",
            "carbon": "C",
            "nbh_min_size": 3,
            "nbh_max_size": 5,
        }
        pattern = NeighborhoodPatterns(**args)
        self.assertIsInstance(pattern, Template)

    def test_neighborhood_invalid_nbh_min_size(self):
        args = {
            **self.common_args,
            "atom_type": "atom",
            "carbon": "C",
            "nbh_min_size": 2,
            "nbh_max_size": 5,
        }
        with self.assertRaises(ValueError):
            NeighborhoodPatterns(**args)

    def test_neighborhood_invalid_nbh_max_size(self):
        args = {
            **self.common_args,
            "atom_type": "atom",
            "carbon": "C",
            "nbh_min_size": 4,
            "nbh_max_size": 3,
        }
        with self.assertRaises(ValueError):
            NeighborhoodPatterns(**args)

    def test_chiral_center_buildable(self):
        self.check_buildable(
            [
                "O=CC(O)CO"  # glyceraldehyde
            ],
            nbhoods=True,
        )

    # PathPattern
    def test_path_instantiation(self):
        args = {
            **self.common_args,
            "max_depth": 4,
        }
        pattern = PathPattern(**args)
        self.assertIsInstance(pattern, Template)

    def test_path_invalid_max_depth(self):
        args = {
            **self.common_args,
            "max_depth": 2,
        }
        with self.assertRaises(ValueError):
            PathPattern(**args)

    # YShapePattern
    def test_yshape_instantiation(self):
        args = {
            **self.common_args,
            "double_bond": "db",
        }
        pattern = YShapePattern(**args)
        self.assertIsInstance(pattern, Template)

    def test_yshape_missing_required_param(self):
        args = {**self.common_args}
        with self.assertRaises(ValueError):
            YShapePattern(**args)

    def test_yshape_buildable(self):
        self.check_buildable(
            [
                "C=O"  # formaldehyde
            ],
            y_shape=True,
        )
