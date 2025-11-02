from chemlogic.datasets.CustomDataset import CustomDataset
from chemlogic.datasets.DHFR import DHFR
from chemlogic.datasets.ER import ER
from chemlogic.datasets.MUTAG import MUTAG
from chemlogic.datasets.PTC import PTC
from chemlogic.datasets.PTCFM import PTCFM
from chemlogic.datasets.PTCFR import PTCFR
from chemlogic.datasets.PTCMM import PTCMM
from chemlogic.datasets.SmilesDataset import SmilesDataset

# Dataset registry
DATASET_CLASSES = {
    "mutagen": MUTAG,
    "ptc": PTC,
    "ptc_fr": PTCFR,
    "ptc_mm": PTCMM,
    "ptc_fm": PTCFM,
    "dhfr": DHFR,
    "er": ER,
    "custom": CustomDataset,
    "smiles": SmilesDataset,
}

# Dataset sizes
DATASET_LENGTHS = {
    "anti_sarscov2_activity": 1484,
    "blood_brain_barrier": 2030,
    "carcinogenous": 280,
    "cox": 303,
    "cyp2c9_substrate": 669,
    "cyp2d6_substrate": 667,
    "cyp3a4_substrate": 670,
    "dhfr": 393,
    "er": 446,
    "human_intestinal_absorption": 578,
    "mutagen": 183,
    "oral_bioavailability": 640,
    "p_glycoprotein_inhibition": 1218,
    "pampa_permeability": 2034,
    "ptc": 344,
    "ptc_fm": 349,
    "ptc_fr": 351,
    "ptc_mm": 336,
    "skin_reaction": 404,
    "splice_ai": 7962,
}

# Custom dataset names
CUSTOM_DATASETS = {
    "anti_sarscov2_activity",
    "blood_brain_barrier",
    "carcinogenous",
    "cyp2c9_substrate",
    "cyp2d6_substrate",
    "cyp3a4_substrate",
    "human_intestinal_absorption",
    "oral_bioavailability",
    "p_glycoprotein_inhibition",
    "pampa_permeability",
    "skin_reaction",
}

# All available datasets
AVAILABLE_DATASETS = sorted(set(DATASET_CLASSES.keys()) - {"custom"} | CUSTOM_DATASETS)


def get_available_datasets():
    """Returns a list of all available dataset names."""
    return AVAILABLE_DATASETS


def get_dataset_len(name):
    """Returns the number of entries in a dataset, or 0 if unknown."""
    return DATASET_LENGTHS.get(name, 0)


def get_dataset(
    dataset_name,
    param_size,
    examples=None,
    queries=None,
    smiles_list: list[str] = None,
    labels: list[int] = None,
):
    """
    Instantiates a dataset class based on its name.

    Args:
        dataset_name (str): Name of the dataset.
        param_size (int): Parameter size.
        examples (str, optional): Path to examples file (for custom datasets).
        queries (str, optional): Path to queries file (for custom datasets).
        smiles_list (list[str], optional): A list of smiles strings to build the dataset with.
        labels (list[int], optional): A list of integer labels to build the dataset with.
    Returns:
        An instance of the dataset class.

    Raises:
        ValueError: If the dataset name is invalid.
    """
    # Dataset from SMILES list
    if smiles_list:
        return SmilesDataset(
            smiles_list=smiles_list,
            labels=labels,
            param_size=param_size,
            dataset_name=dataset_name,
        )

    # Custom dataset with custom examples/queries files, or from custom datasets
    if (examples and queries) or dataset_name in CUSTOM_DATASETS:
        return CustomDataset(examples, queries, param_size, dataset_name)

    dataset_class = DATASET_CLASSES.get(dataset_name)
    if dataset_class is None:
        raise ValueError(
            f"Invalid dataset name: {dataset_name}\nAvailable datasets: {', '.join(get_available_datasets())}"
        )

    return dataset_class(param_size)
