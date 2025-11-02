import unittest

from neuralogic.core import R, Settings, V
from neuralogic.nn import get_evaluator

from chemlogic.datasets import (
    COX,
    DHFR,
    ER,
    MUTAG,
    PTC,
    PTCFM,
    PTCFR,
    PTCMM,
    CustomDataset,
    Dataset,
    SmilesDataset,
    get_available_datasets,
    get_dataset,
    get_dataset_len,
)


class TestDataset(unittest.TestCase):
    def test_valid_arguments(self):
        Dataset(
            dataset_name="cox",
            node_embed="a",
            edge_embed="a",
            connection="a",
            atom_types=["a"],
            key_atom_type=["a"],
            bond_types=["a"],
            single_bond="a",
            double_bond="a",
            triple_bond="a",
            aliphatic_bonds=["a"],
            aromatic_bonds=["a"],
            carbon="c",
            oxygen="o",
            hydrogen="h",
            nitrogen="n",
            sulfur="s",
            halogens=["f"],
            param_size=1,
        )

    def test_load_data_nonexisting_file(self):
        with self.assertRaises(FileNotFoundError) as context:
            Dataset(
                dataset_name="hello",
                node_embed="a",
                edge_embed="a",
                connection="a",
                atom_types=["a"],
                key_atom_type=["a"],
                bond_types=["a"],
                single_bond="a",
                double_bond="a",
                triple_bond="a",
                aliphatic_bonds=["a"],
                aromatic_bonds=["a"],
                carbon="c",
                oxygen="o",
                hydrogen="h",
                nitrogen="n",
                sulfur="s",
                halogens=["f"],
                param_size=1,
            )
            self.assertIn("does not exist", str(context.exception))

    def test_wrong_param_size(self):
        with self.assertRaises(ValueError) as context:
            Dataset(
                dataset_name="hello",
                node_embed="a",
                edge_embed="a",
                connection="a",
                atom_types=["a"],
                key_atom_type=["a"],
                bond_types=["a"],
                single_bond="a",
                double_bond="a",
                triple_bond="a",
                aliphatic_bonds=["a"],
                aromatic_bonds=["a"],
                carbon="c",
                oxygen="o",
                hydrogen="h",
                nitrogen="n",
                sulfur="s",
                halogens=["f"],
                param_size=-1,
            )
            self.assertIn("positive", str(context.exception))


class TestDatasetClasses(unittest.TestCase):
    def test_anti_sarscov2_activity(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="anti_sarscov2_activity",
        )

    def test_blood_brain_barrier(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="blood_brain_barrier",
        )

    def test_carcinogenous(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="carcinogenous",
        )

    def test_cyp2c9_substrate(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp2c9_substrate",
        )

    def test_cyp2d6_substrate(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp2d6_substrate",
        )

    def test_cyp3a4_substrate(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp3a4_substrate",
        )

    def test_human_intestinal_absorption(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="human_intestinal_absorption",
        )

    def test_oral_bioavailability(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="oral_bioavailability",
        )

    def test_p_glycoprotein_inhibition(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="p_glycoprotein_inhibition",
        )

    def test_pampa_permeability(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="pampa_permeability",
        )

    def test_skin_reaction(self):
        CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="skin_reaction",
        )

    def test_nonexistent_custom(self):
        with self.assertRaises(FileNotFoundError):
            CustomDataset(
                examples="test",
                queries="test",
                param_size=1,
                dataset_name="test",
            )

    def test_cox(self):
        COX(param_size=1)

    def test_dhfr(self):
        DHFR(param_size=1)

    def test_er(self):
        ER(param_size=1)

    def test_mutag(self):
        MUTAG(param_size=1)

    def test_ptc(self):
        PTC(param_size=1)

    def test_ptcfm(self):
        PTCFM(param_size=1)

    def test_ptcfr(self):
        PTCFR(param_size=1)

    def test_ptcmm(self):
        PTCMM(param_size=1)

    def test_smiles_dataset(self):
        dataset = SmilesDataset(
            smiles_list=["O"],
            labels=[1],
            param_size=1,
            dataset_name="test",
        )
        dataset.clear()


class TestDatasetLoader(unittest.TestCase):
    def test_get_available_datasets(self):
        datasets = get_available_datasets()
        self.assertIn("mutagen", datasets)
        self.assertIn("carcinogenous", datasets)

    def test_get_dataset_len(self):
        self.assertEqual(get_dataset_len("mutagen"), 183)
        self.assertEqual(get_dataset_len("nonexistent"), 0)

    def test_get_custom_dataset(self):
        dataset = get_dataset("carcinogenous", param_size=1)
        self.assertIsInstance(dataset, CustomDataset)

    def test_get_predefined_dataset(self):
        dataset = get_dataset("mutagen", param_size=1)
        self.assertIsInstance(dataset, MUTAG)

    def test_invalid_dataset_name(self):
        with self.assertRaises(ValueError):
            get_dataset("invalid_dataset", param_size=1)


class TestDatasetsBuildable(unittest.TestCase):
    def test_anti_sarscov2_activity_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="anti_sarscov2_activity",
        )

        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_blood_brain_barrier_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="blood_brain_barrier",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_carcinogenous_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="carcinogenous",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_cyp2c9_substrate_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp2c9_substrate",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_cyp2d6_substrate_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp2d6_substrate",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_cyp3a4_substrate_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp3a4_substrate",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_human_intestinal_absorption_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="human_intestinal_absorption",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_oral_bioavailability_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="oral_bioavailability",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_p_glycoprotein_inhibition_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="p_glycoprotein_inhibition",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_pampa_permeability_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="pampa_permeability",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_skin_reaction_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="skin_reaction",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_cox_buildable(self):
        dataset = COX(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_dhfr_buildable(self):
        dataset = DHFR(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_er_buildable(self):
        dataset = ER(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_mutag_buildable(self):
        dataset = MUTAG(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_ptc_buildable(self):
        dataset = PTC(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_ptcfm_buildable(self):
        dataset = PTCFM(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_ptcfr_buildable(self):
        dataset = PTCFR(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_ptcmm_buildable(self):
        dataset = PTCMM(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)

    def test_smiles_dataset_buildable(self):
        dataset = SmilesDataset(
            smiles_list=["O"],
            labels=[1],
            param_size=1,
            dataset_name="test",
        )
        assert "h" in dataset.atom_types
        assert "o" in dataset.atom_types
        assert "b_1" in dataset.bond_types
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = get_evaluator(dataset, Settings())
        evaluator.build_dataset(dataset.data)
        dataset.clear()
