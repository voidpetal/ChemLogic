from enum import Enum

import numpy as np
from neuralogic.core import R, Transformation, Settings, V
from neuralogic.nn import get_evaluator
from neuralogic.nn.loss import MSE, CrossEntropy, ErrorFunction
from neuralogic.optim import Adam, Optimizer
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

from chemlogic.datasets.datasets import get_dataset
from chemlogic.knowledge_base.chemrules import get_chem_rules
from chemlogic.knowledge_base.subgraphs import get_subgraphs
from chemlogic.models.models import get_model
from chemlogic.utils.ChemTemplate import ChemTemplate


class ArchitectureType(Enum):
    BARE = "bare"
    CCE = "CCE"
    CCD = "CCD"

    @staticmethod
    def from_string(name: str):
        try:
            return ArchitectureType[name]
        except KeyError:
            raise ValueError(
                f"Undefined architecture type: {name}. Valid types are: {[e.name for e in ArchitectureType]}"
            ) from KeyError


class Pipeline:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        param_size: int,
        layers: int,
        max_depth: int = 1,
        max_subgraph_depth: int = 5,
        max_cycle_size: int = 10,
        subgraphs: tuple | bool | None = None,
        chem_rules: tuple | bool | None = None,
        architecture: ArchitectureType = ArchitectureType.BARE,
        examples=None,
        queries=None,
        funnel=False,
        smiles_list: list[str] = None,
        labels: list[int] = None,
        task: str = "classification",
    ):
        """
        Initialize the test setup by configuring the dataset and model along with optional chemical rules and subgraphs.

        :param dataset_name: Name of the dataset to use.
        :param model_name: Name of the model to apply.
        :param param_size: The size of the parameters.
        :param layers: Number of layers in the model.
        :param max_depth: Maximum depth for the model.
        :param max_subgraph_depth: Maximum depth for subgraph processing.
        :param max_cycle_size: Maximum size of cycles in subgraphs.
        :param subgraphs: Tuple containing flags for different subgraph types.
        :param chem_rules: Tuple containing chemical rule configurations.
        :param architecture: The architecture to use for the model. - default: ArchitectureType.BARE ["bare", "CCE", "CCD"]
        :param funnel: create an informational funnel in the knowledge base. - default: False
        :param smiles_list: A list of smiles strings to build the dataset with.
        :param labels: A list of integer labels to build the dataset with.
        :param task: The type of task, either "classification" or "regression". - default: "classification"
        :return: A tuple containing the template and dataset.
        """

        if bool(smiles_list) != bool(labels):
            raise ValueError(
                "If building a dataset from SMILES, make sure to provide both `smiles_list` and `labels` params."
            )

        if smiles_list:
            dataset_args = {"smiles_list": smiles_list, "labels": labels}
        else:
            dataset_args = {"examples": examples, "queries": queries}

        dataset = get_dataset(dataset_name, param_size, **dataset_args)

        if task not in ["regression", "classification"]:
            raise ValueError("Task must be either 'regression' or 'classification'.")
        
        transformation = None
        if task == "classification":
            transformation = Transformation.SIGMOID
        elif task == "regression":
            transformation = Transformation.IDENTITY

        template = ChemTemplate()

        if architecture == ArchitectureType.BARE:
            io_layers = {
                "nn_input": dataset.node_embed,
                "nn_output": "predict",
                "chem_input": dataset.node_embed,
                "chem_output": "predict",
                "subg_input": dataset.node_embed,
                "subg_output": "predict",
            }
        elif architecture == ArchitectureType.CCE:
            io_layers = {
                "nn_input": "kb_features",
                "nn_output": "predict",
                "chem_input": dataset.node_embed,
                "chem_output": "predict",
                "subg_input": dataset.node_embed,
                "subg_output": "predict",
            }
            template += [
                (R.get("kb_features")(V.X) <= R.get(dataset.node_embed)(V.X)),
                (R.get("kb_features")(V.X) <= R.get("sub_subgraph_pattern")(V.X)),
                (R.get("kb_features")(V.X) <= R.get("chem_chem_rules")(V.X)),
            ]

        elif architecture == ArchitectureType.CCD:
            io_layers = {
                "nn_input": dataset.node_embed,
                "nn_output": "predict",
                "chem_input": "kb_features",
                "chem_output": "predict",
                "subg_input": "kb_features",
                "subg_output": "predict",
            }
            template += [
                (R.get("kb_features")(V.X) <= R.get(dataset.node_embed)(V.X)),
                (
                    R.get("kb_features")(V.X)
                    <= R.get(f"{model_name.split('_')[0]}")(V.X)
                ),
            ]
        else:
            raise ValueError(
                f"Invalid architecture: {architecture}. Please use one of the following: {[e.value for e in ArchitectureType]}"
            )

        local = False
        if model_name == "kgnn_local":
            local = True
            model_name = "kgnn"

        template += get_model(
            model_name,
            layers,
            io_layers["nn_input"],
            dataset.edge_embed,
            dataset.connection,
            param_size,
            edge_types=dataset.bond_types,
            max_depth=max_depth,
            local=local,
            output_layer_name=io_layers["nn_output"],
            output_layer_transformation=transformation,
        )

        if chem_rules:
            try:
                # TODO: create a generator class
                hydrocarbons, oxy, nitro, sulfuric, relaxations = chem_rules
            except Exception:
                hydrocarbons, oxy, nitro, sulfuric, relaxations = (True,) * 5

            chem_path = (
                "sub_path"
                if subgraphs
                and ((type(subgraphs) in (list, tuple) and subgraphs[1]) or subgraphs)
                else None
            )

            template += get_chem_rules(
                "chem",
                io_layers["chem_input"],
                dataset.edge_embed,
                dataset.connection,
                param_size,
                dataset.halogens,
                output_layer_name=io_layers["chem_output"],
                output_layer_transformation=transformation,
                single_bond=dataset.single_bond,
                double_bond=dataset.double_bond,
                triple_bond=dataset.triple_bond,
                aromatic_bonds=dataset.aromatic_bonds,
                carbon=dataset.carbon,
                hydrogen=dataset.hydrogen,
                oxygen=dataset.oxygen,
                nitrogen=dataset.nitrogen,
                sulfur=dataset.sulfur,
                path=chem_path,
                hydrocarbons=hydrocarbons,
                nitro=nitro,
                sulfuric=sulfuric,
                oxy=oxy,
                relaxations=relaxations,
                key_atoms=dataset.key_atom_type,
                funnel=funnel,
            )

        if subgraphs:
            try:
                cycles, paths, y_shape, nbhoods, circular, collective = subgraphs
            except Exception:
                cycles, paths, y_shape, nbhoods, circular, collective = (True,) * 6

            template += get_subgraphs(
                "sub",
                io_layers["subg_input"],
                dataset.edge_embed,
                dataset.connection,
                param_size,
                max_cycle_size=max_cycle_size,
                max_depth=max_subgraph_depth,
                output_layer_name=io_layers["subg_output"],
                output_layer_transformation=transformation,
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
                funnel=funnel,
            )

        self.dataset = dataset
        self.template = dataset + template
        
        self.task = task

    def train_test_cycle(
        self,
        lr: float = 0.001,
        epochs: int = 100,
        split_ratio: float = 0.75,
        optimizer: Optimizer = Adam,
        error_function: ErrorFunction = None,
        batches: int = 1,
        early_stopping_threshold: float = 0.001,
        early_stopping_rounds: int = 10,
    ):
        """
        Train and test the model based on the provided template and dataset.

        :param lr: Learning rate for the optimizer.
        :param epochs: Number of training epochs.
        :param split_ratio: The ratio to split the dataset into training and testing.
        :param optimizer: The optimizer class to be used.
        :param error_function: The error function to be used.
        :param batches: Number of batches to build the dataset in.
        :param early_stopping_threshold: Minimum improvement threshold to reset early stopping counter.
        :param early_stopping_rounds: Number of rounds without improvement to trigger early stopping.
        :return: The training loss, testing loss, AUROC validation score for classification or R2 for regression tasks and the evaluator object.
        """
        if error_function is None:
            error_function = MSE if self.task == "regression" else CrossEntropy

        settings = Settings(
            optimizer=optimizer(lr=lr), epochs=epochs, error_function=error_function()
        )
        # TODO: log instead of print
        print(f"Building dataset in {batches} batches")
        evaluator = get_evaluator(self.template, settings)
        built_dataset = evaluator.build_dataset(self.dataset.data, batch_size=batches)

        train_dataset, test_dataset = train_test_split(
            built_dataset.samples, train_size=split_ratio, random_state=42
        )
        print("Training model")
        train_losses = self._train_model(evaluator, train_dataset, settings.epochs, early_stopping_rounds, early_stopping_threshold)
        test_loss, other_metric = self._evaluate_model(evaluator, test_dataset)
        
        # Save the trained model
        self.evaluator = evaluator

        return train_losses[-1], test_loss, other_metric, evaluator

    def _train_model(
        self,
        evaluator,
        train_dataset,
        epochs,
        early_stopping_rounds=10,
        early_stopping_threshold=0.001,
    ):
        """
        Train the model on the training dataset.

        :param evaluator: The evaluator object used for training.
        :param train_dataset: The dataset to train on.
        :param epochs: Number of training epochs.
        :param early_stopping_rounds: Number of rounds without improvement to trigger early stopping.
        :param early_stopping_threshold: Minimum improvement threshold to reset early stopping counter.
        :return: List of average training losses per epoch.
        """
        average_losses = []
        best_loss = float("inf")
        rounds_without_improvement = 0

        for epoch in range(epochs):
            current_total_loss, number_of_samples = next(evaluator.train(train_dataset))
            train_loss = current_total_loss / number_of_samples
            average_losses.append(train_loss)

            if train_loss < best_loss - early_stopping_threshold:
                best_loss = train_loss
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            print(
                f"Epoch {epoch + 1}/{epochs} | Train loss: {train_loss} | Best loss: {best_loss} | Difference: {best_loss - train_loss}"
            )

            if rounds_without_improvement >= early_stopping_rounds:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        return average_losses

    def _evaluate_model(self, evaluator, test_dataset):
        """
        Evaluate the model on the test dataset.

        :param evaluator: The evaluator object used for testing.
        :param test_dataset: The dataset to test on.
        :return: The testing loss and the specified metric score.
        """

        predictions = []
        targets = []
        for sample, y_hat in zip(
            test_dataset, evaluator.test(test_dataset, generator=False), strict=False
        ):
            predictions.append(y_hat)
            targets.append(sample.java_sample.target.value)

        metric_score = None
        if self.task == "classification":
            metric_score = roc_auc_score(targets, predictions)
            # Accuracy
            loss = sum(
                round(pred) != target
                for pred, target in zip(predictions, targets, strict=False)
            ) / len(test_dataset)
        elif self.task == "regression":
            metric_score = r2_score(targets, predictions)
            # Mean Squared Error
            loss = sum(
                (pred - target) ** 2
                for pred, target in zip(predictions, targets, strict=False)
            ) / len(test_dataset)

        return loss, metric_score

    def inference(self, smiles_list: list[str]):
        """
        Perform inference on a list of SMILES strings.

        :param smiles_list: A list of SMILES strings to perform inference on.
        :return: A list of predictions corresponding to the input SMILES strings.
        """
        if not hasattr(self, "evaluator"):
            raise ValueError(
                "The model has not been trained yet. Please train the model before performing inference."
            )

        inference_dataset = get_dataset(
            self.dataset.dataset_name,
            self.dataset.param_size,
            smiles_list=smiles_list,
            labels=[0] * len(smiles_list),  # Dummy labels
        )

        built_dataset = self.evaluator.build_dataset(
            inference_dataset.data, batch_size=1
        )

        predictions = []
        for _, y_hat in zip(
            built_dataset.samples,
            self.evaluator.test(built_dataset, generator=False),
            strict=False,
        ):
            predictions.append(y_hat)

        return predictions