import re
import tempfile
import unittest
from pathlib import Path

from neuralogic.core import R, Settings, V

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
        COX(param_size=1)

    def test_load_data_nonexisting_file(self):
        with self.assertRaises(FileNotFoundError) as context:
            Dataset(
                dataset_name="hello",
                node_embed="a",
                edge_embed="a",
                connection="a",
                atom_types=["a"],
                key_atom_type=["a"],
                bond_types=["b_0", "b_1", "b_2", "b_3", "b_4"],
                single_bond="b_2",
                double_bond="b_3",
                triple_bond="b_4",
                aliphatic_bonds=["b_2", "b_3", "b_4"],
                aromatic_bonds=["b_0"],
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
                bond_types=["b_1", "b_2", "b_3", "b_12"],
                single_bond="b_1",
                double_bond="b_2",
                triple_bond="b_3",
                aliphatic_bonds=["b_1", "b_2", "b_3"],
                aromatic_bonds=["b_12"],
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
    def assert_bond_facts_declared(self, dataset):
        examples_file = dataset.data.examples_file
        encoded_bond_types = set(
            re.findall(r"(?<![\w])b_\d+(?=\()", Path(examples_file).read_text())
        )
        self.assertTrue(encoded_bond_types)
        self.assertLessEqual(encoded_bond_types, set(dataset.bond_types))

    def test_anti_sarscov2_activity(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="anti_sarscov2_activity",
        )
        self.assertIn("b_12", dataset.bond_types)
        self.assert_bond_facts_declared(dataset)

    def test_blood_brain_barrier(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="blood_brain_barrier",
        )
        self.assert_bond_facts_declared(dataset)

    def test_carcinogenous(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="carcinogenous",
        )
        self.assert_bond_facts_declared(dataset)

    def test_cyp2c9_substrate(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp2c9_substrate",
        )
        self.assert_bond_facts_declared(dataset)

    def test_cyp2d6_substrate(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp2d6_substrate",
        )
        self.assert_bond_facts_declared(dataset)

    def test_cyp3a4_substrate(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp3a4_substrate",
        )
        self.assert_bond_facts_declared(dataset)

    def test_human_intestinal_absorption(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="human_intestinal_absorption",
        )
        self.assert_bond_facts_declared(dataset)

    def test_oral_bioavailability(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="oral_bioavailability",
        )
        self.assert_bond_facts_declared(dataset)

    def test_p_glycoprotein_inhibition(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="p_glycoprotein_inhibition",
        )
        self.assert_bond_facts_declared(dataset)

    def test_pampa_permeability(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="pampa_permeability",
        )
        self.assert_bond_facts_declared(dataset)

    def test_skin_reaction(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="skin_reaction",
        )
        self.assert_bond_facts_declared(dataset)

    def test_nonexistent_custom(self):
        with self.assertRaises(FileNotFoundError):
            CustomDataset(
                examples="test",
                queries="test",
                param_size=1,
                dataset_name="test",
            )

    def test_cox(self):
        dataset = COX(param_size=1)
        self.assertEqual(
            (dataset.single_bond, dataset.double_bond, dataset.triple_bond),
            ("b_2", "b_3", "b_4"),
        )
        self.assertEqual(dataset.aromatic_bonds, ["b_0"])
        self.assert_bond_facts_declared(dataset)

    def test_dhfr(self):
        self.assert_bond_facts_declared(DHFR(param_size=1))

    def test_er(self):
        self.assert_bond_facts_declared(ER(param_size=1))

    def test_mutag(self):
        dataset = MUTAG(param_size=1)
        self.assertEqual(dataset.single_bond, "b_1")
        self.assertEqual(dataset.double_bond, "b_2")
        self.assertEqual(dataset.triple_bond, "b_3")
        self.assertEqual(dataset.aromatic_bonds, ["b_7"])
        self.assertEqual(dataset.bond_types, ["b_1", "b_2", "b_3", "b_4", "b_5", "b_7"])
        self.assert_bond_facts_declared(dataset)

    def test_ptc(self):
        dataset = PTC(param_size=1)
        self.assertEqual(
            (dataset.single_bond, dataset.double_bond, dataset.triple_bond),
            ("b_2", "b_1", "b_0"),
        )
        self.assertEqual(dataset.aromatic_bonds, ["b_3"])
        self.assert_bond_facts_declared(dataset)

    def test_ptcfm(self):
        dataset = PTCFM(param_size=1)
        self.assertEqual(dataset.single_bond, "b_1")
        self.assertEqual(dataset.double_bond, "b_2")
        self.assertEqual(dataset.triple_bond, "b_0")
        self.assertEqual(dataset.aromatic_bonds, ["b_3"])
        self.assert_bond_facts_declared(dataset)

    def test_ptcfr(self):
        dataset = PTCFR(param_size=1)
        self.assertEqual((dataset.single_bond, dataset.double_bond), ("b_2", "b_1"))
        self.assertEqual(dataset.aromatic_bonds, ["b_3"])
        self.assert_bond_facts_declared(dataset)

    def test_ptcmm(self):
        dataset = PTCMM(param_size=1)
        self.assertEqual((dataset.single_bond, dataset.double_bond), ("b_2", "b_1"))
        self.assertEqual(dataset.aromatic_bonds, ["b_3"])
        self.assert_bond_facts_declared(dataset)

    def test_custom_file_dataset_accepts_dataset_specific_bond_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            examples = Path(directory, "examples.txt")
            queries = Path(directory, "queries.txt")
            examples.write_text("<1> bond(0, 1, 0), <1> b_ar(0), <1> c(0), <1> c(1).\n")
            queries.write_text("1 predict.\n")

            dataset = get_dataset(
                "custom",
                param_size=1,
                examples=str(examples),
                queries=str(queries),
                bond_types=["b_s", "b_d", "b_t", "b_ar"],
                single_bond="b_s",
                double_bond="b_d",
                triple_bond="b_t",
                aliphatic_bonds=["b_s", "b_d", "b_t"],
                aromatic_bonds=["b_ar"],
            )

            self.assertEqual(dataset.aromatic_bonds, ["b_ar"])
            self.assertIn("b_ar", dataset.bond_types)

            examples.write_text(
                "<1> bond(0, 1, 0), <1> b_unlisted(0), <1> c(0), <1> c(1).\n"
            )

    def test_smiles_dataset(self):
        dataset = SmilesDataset(
            smiles_list=["c1ccccc1"],
            labels=[1],
            param_size=1,
            dataset_name="test",
        )
        self.assertEqual(dataset.bond_roles["aromatic"], ("b_12",))
        self.assertIn("b_12", dataset.bond_types)
        self.assert_bond_facts_declared(dataset)
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
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_blood_brain_barrier_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="blood_brain_barrier",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_carcinogenous_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="carcinogenous",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_cyp2c9_substrate_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp2c9_substrate",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_cyp2d6_substrate_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp2d6_substrate",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_cyp3a4_substrate_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="cyp3a4_substrate",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_human_intestinal_absorption_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="human_intestinal_absorption",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_oral_bioavailability_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="oral_bioavailability",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_p_glycoprotein_inhibition_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="p_glycoprotein_inhibition",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_pampa_permeability_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="pampa_permeability",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_skin_reaction_buildable(self):
        dataset = CustomDataset(
            examples=None,
            queries=None,
            param_size=1,
            dataset_name="skin_reaction",
        )
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_cox_buildable(self):
        dataset = COX(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_dhfr_buildable(self):
        dataset = DHFR(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_er_buildable(self):
        dataset = ER(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_mutag_buildable(self):
        dataset = MUTAG(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_ptc_buildable(self):
        dataset = PTC(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_ptcfm_buildable(self):
        dataset = PTCFM(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_ptcfr_buildable(self):
        dataset = PTCFR(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)

    def test_ptcmm_buildable(self):
        dataset = PTCMM(param_size=1)
        dataset.add_rules([R.predict <= R.get(dataset.node_embed)(V.X)])
        evaluator = dataset.build(Settings())
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
        evaluator = dataset.build(Settings())
        evaluator.build_dataset(dataset.data)
        dataset.clear()
