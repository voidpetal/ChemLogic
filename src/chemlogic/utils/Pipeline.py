from datetime import datetime
from enum import Enum
from pathlib import Path

from neuralogic.core import R, Settings, Transformation, V
from neuralogic.nn.loss import MSE, CrossEntropy, ErrorFunction
from neuralogic.nn.optim import Adam, Optimizer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from chemlogic.datasets.datasets import get_dataset
from chemlogic.knowledge_base.chemrules import get_chem_rules
from chemlogic.knowledge_base.subgraphs import get_subgraphs
from chemlogic.models.models import get_model
from chemlogic.utils.Checkpoint import Checkpoint
from chemlogic.utils.ChemTemplate import ChemTemplate
from chemlogic.utils.config import PipelineConfig, TrainConfig


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
        task: str | None = None,
        atom_features: str | list[str] | None = None,
        bond_features: str | list[str] | None = None,
        graph_features: dict | None = None,
        num_outputs: int = 1,
        custom_rules=None,
        custom_input: str | None = None,
        custom_output: str | None = None,
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
        :param atom_features: Atom features to extract as node-level predicates.
            - None: No additional features (default, backward compatible)
            - 'all': Extract all 8 available RDKit atom features
            - list[str]: Extract only specified features (e.g., ['formal_charge', 'is_aromatic'])
        :param bond_features: Bond features to extract as edge-level predicates.
            - None: No additional features (default, backward compatible)
            - 'all': Extract all 4 available RDKit bond features
            - list[str]: Extract only specified features (e.g., ['is_aromatic', 'is_conjugated'])
        :param custom_rules: NeuraLogic rules or Model to add as a knowledge-base feature branch.
        :param custom_input: Predicate expected by ``custom_rules`` as its input.
        :param custom_output: Predicate produced by ``custom_rules`` as its output.
        :return: A tuple containing the template and dataset.
        """

        if smiles_list is not None and task is None:
            task, num_outputs = Pipeline._infer_task(labels)

        if task is None:
            task = "classification"

        if (smiles_list is not None) != (labels is not None):
            raise ValueError(
                "If building a dataset from SMILES, provide both `smiles_list` and `labels`."
            )

        cfg = PipelineConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            param_size=param_size,
            layers=layers,
            max_depth=max_depth,
            max_subgraph_depth=max_subgraph_depth,
            max_cycle_size=max_cycle_size,
            subgraphs=subgraphs,
            chem_rules=chem_rules,
            architecture=architecture.value
            if isinstance(architecture, ArchitectureType)
            else architecture,
            funnel=funnel,
            task=task,
            num_outputs=num_outputs,
        )

        if smiles_list is not None:
            dataset_args = {
                "smiles_list": smiles_list,
                "labels": labels,
                "atom_features": atom_features,
                "bond_features": bond_features,
                "graph_features": graph_features,
                "num_outputs": cfg.num_outputs,
            }
        else:
            dataset_args = {"examples": examples, "queries": queries}

        dataset = get_dataset(cfg.dataset_name, cfg.param_size, **dataset_args)

        transformation = None
        if cfg.task in ("classification", "multi_class"):
            transformation = Transformation.SIGMOID
        elif cfg.task in ("regression", "multi_regression"):
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
            cfg.layers,
            io_layers["nn_input"],
            dataset.edge_embed,
            dataset.connection,
            cfg.param_size,
            num_outputs=cfg.num_outputs,
            edge_types=dataset.bond_types,
            max_depth=cfg.max_depth,
            local=local,
            output_layer_name=io_layers["nn_output"],
            output_layer_transformation=transformation,
        )

        if cfg.chem_rules or custom_rules is not None:
            if cfg.chem_rules:
                try:
                    # TODO: create a generator class
                    hydrocarbons, oxy, nitro, sulfuric, relaxations = cfg.chem_rules
                except Exception:
                    hydrocarbons, oxy, nitro, sulfuric, relaxations = (True,) * 5
            else:
                hydrocarbons, oxy, nitro, sulfuric, relaxations = (False,) * 5

            chem_path = (
                "sub_path"
                if cfg.subgraphs
                and (
                    (type(cfg.subgraphs) in (list, tuple) and cfg.subgraphs[1])
                    or cfg.subgraphs
                )
                else None
            )

            template += get_chem_rules(
                "chem",
                io_layers["chem_input"],
                dataset.edge_embed,
                dataset.connection,
                cfg.param_size,
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
                funnel=cfg.funnel,
            )

        if custom_rules is not None:
            if not isinstance(custom_input, str) or not isinstance(custom_output, str):
                raise TypeError(
                    "custom_input and custom_output are required when custom_rules is provided."
                )

            template += custom_rules
            custom_source = io_layers["chem_input"]
            custom_target = (
                "kb_features" if architecture == ArchitectureType.CCE else "predict"
            )
            template += [
                R.get(custom_input)(V.X) <= R.get(custom_source)(V.X),
                R.get(custom_target)(V.X) <= R.get(custom_output)(V.X),
            ]

        if cfg.subgraphs:
            try:
                cycles, paths, y_shape, nbhoods, circular, collective = cfg.subgraphs
            except Exception:
                cycles, paths, y_shape, nbhoods, circular, collective = (True,) * 6

            template += get_subgraphs(
                "sub",
                io_layers["subg_input"],
                dataset.edge_embed,
                dataset.connection,
                cfg.param_size,
                max_cycle_size=cfg.max_cycle_size,
                max_depth=cfg.max_subgraph_depth,
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
                funnel=cfg.funnel,
            )

        self.dataset = dataset
        self.template = dataset + template

        self.task = cfg.task
        self.num_outputs = cfg.num_outputs

        self._architecture_params = cfg.to_architecture_dict()

    @staticmethod
    def _infer_task(labels) -> tuple[str, int]:
        """
        Infer task and num_outputs from the first label.

        Sequence label (list/tuple):
          - all values 0/1 and exactly one 1  → multi_class, len(label)
          - anything else                      → multi_regression, len(label)
            (includes multi-label binary like (1,0,1) and float vectors)
        Scalar:
          - any float value                    → regression, 1
          - integers {0, 1} only              → classification, 1
          - integers with N>2 unique values   → regression (ordinal ambiguous)
        """

        l0 = labels.iloc[0] if hasattr(labels, "iloc") else labels[0]

        if isinstance(l0, (list, tuple)):
            n = len(l0)
            is_one_hot = sum(l0) == 1 and all(v in (0, 1, 0.0, 1.0) for v in l0)
            return ("multi_class" if is_one_hot else "multi_regression"), n

        all_vals = labels.tolist() if hasattr(labels, "tolist") else list(labels)
        if any(isinstance(v, float) and v != int(v) for v in all_vals):
            return "regression", 1

        unique = sorted(set(int(v) for v in all_vals))
        if unique == [0, 1]:
            return "classification", 1

        return "regression", 1

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
        checkpoint_every: int | None = None,
        checkpoint_dir: str | Path | None = None,
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
        :param checkpoint_every: Save checkpoint every N epochs. If None, no checkpoints saved during training.
        :param checkpoint_dir: Directory for checkpoint files. Defaults to checkpoints/{dataset_name}/
        :return: The training loss, testing loss, AUROC validation score for classification or R2 for regression tasks and the evaluator object.
        """
        train_cfg = TrainConfig(
            lr=lr,
            epochs=epochs,
            split_ratio=split_ratio,
            batches=batches,
            early_stopping_threshold=early_stopping_threshold,
            early_stopping_rounds=early_stopping_rounds,
            checkpoint_every=checkpoint_every,
        )

        if error_function is None:
            error_function = MSE if self.task == "regression" else CrossEntropy

        settings = Settings(
            optimizer=optimizer(lr=train_cfg.lr),
            error_function=error_function(),
        )
        # TODO: log instead of print
        print(f"Building dataset in {train_cfg.batches} batches")
        evaluator = self.template.build(settings)
        built_dataset = evaluator.build_dataset(
            self.dataset.data, batch_size=train_cfg.batches
        )

        samples = list(built_dataset)
        train_dataset, test_dataset = train_test_split(
            samples, train_size=train_cfg.split_ratio, random_state=42
        )
        print("Training model")
        train_losses = self._train_model(
            evaluator,
            train_dataset,
            train_cfg.epochs,
            train_cfg.early_stopping_rounds,
            train_cfg.early_stopping_threshold,
            checkpoint_every=train_cfg.checkpoint_every,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir is not None else None,
        )
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
        checkpoint_every: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        """
        Train the model on the training dataset.

        :param evaluator: The evaluator object used for training.
        :param train_dataset: The dataset to train on.
        :param epochs: Number of training epochs.
        :param early_stopping_rounds: Number of rounds without improvement to trigger early stopping.
        :param early_stopping_threshold: Minimum improvement threshold to reset early stopping counter.
        :param checkpoint_every: Save checkpoint every N epochs. If None, no checkpoints saved.
        :param checkpoint_dir: Directory for checkpoint files.
        :return: List of average training losses per epoch.
        """
        average_losses = []
        best_loss = float("inf")
        rounds_without_improvement = 0

        for epoch in range(epochs):
            evaluator.train(train_dataset)
            train_loss = evaluator.loss(train_dataset)
            average_losses.append(train_loss)

            if train_loss < best_loss - early_stopping_threshold:
                best_loss = train_loss
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            print(
                f"Epoch {epoch + 1}/{epochs} | Train loss: {train_loss} | Best loss: {best_loss} | Difference: {best_loss - train_loss}"
            )

            # Save checkpoint if checkpoint_every is set
            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                base_dir = (
                    Path(checkpoint_dir)
                    if checkpoint_dir
                    else Path("checkpoints") / self.dataset.dataset_name
                )
                checkpoint_path = base_dir / f"epoch_{epoch + 1}"
                Checkpoint.save(
                    evaluator,
                    checkpoint_path,
                    architecture=self._architecture_params,
                    training_state={
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "best_loss": best_loss,
                    },
                )
                print(f"Checkpoint saved: {checkpoint_path}")

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
            test_dataset, evaluator.test(test_dataset), strict=False
        ):
            predictions.append(y_hat)
            target = getattr(sample, "target", None)
            if not isinstance(target, (int, float, list, tuple)):
                target = sample.java_sample.target.value
            targets.append(target)

        metric_score = None
        if self.task == "classification":
            metric_score = roc_auc_score(targets, predictions)
            loss = sum(
                round(pred) != target
                for pred, target in zip(predictions, targets, strict=False)
            ) / len(test_dataset)
        elif self.task == "multi_class":
            # targets are one-hot vectors; convert to class indices for metrics
            target_ints = [t.index(max(t)) for t in targets]
            all_classes = list(range(self.num_outputs))
            # Softmax-normalize sigmoid outputs so they sum to 1 (required by roc_auc_score)
            import math

            def softmax(v):
                e = [math.exp(x) for x in v]
                s = sum(e)
                return [x / s for x in e]

            probs = [softmax(p) for p in predictions]
            metric_score = roc_auc_score(
                label_binarize(target_ints, classes=all_classes),
                probs,
                multi_class="ovr",
            )
            loss = sum(
                int(p.index(max(p))) != t
                for p, t in zip(predictions, target_ints, strict=False)
            ) / len(test_dataset)
        elif self.task == "regression":
            metric_score = r2_score(targets, predictions)
            # Mean Squared Error
            loss = sum(
                (pred - target) ** 2
                for pred, target in zip(predictions, targets, strict=False)
            ) / len(test_dataset)
        elif self.task == "multi_regression":
            metric_score = r2_score(targets, predictions, multioutput="uniform_average")
            loss = sum(
                sum((p - t) ** 2 for p, t in zip(pred, target, strict=False))
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
        predictions.extend(self.evaluator.test(built_dataset))

        return predictions

    # -------------------------------------------------------------------------
    # Checkpoint methods
    # -------------------------------------------------------------------------

    def save_checkpoint(
        self,
        filepath: str | Path | None = None,
        *,
        training_state: dict | None = None,
        metadata: dict | None = None,
    ) -> tuple[Path, Path]:
        """
        Save the trained model to a checkpoint file.

        Args:
            filepath: Path for checkpoint files (without extension).
                     If None, auto-generates path in checkpoints/{dataset_name}/{timestamp}
            training_state: Optional dict with training state (epoch, losses, etc.)
                           for resuming training later.
            metadata: Optional dict with additional metadata (description, etc.)

        Returns:
            Tuple of (safetensors_path, json_path) for the created files.

        Raises:
            ValueError: If model hasn't been trained yet.
        """
        if not hasattr(self, "evaluator"):
            raise ValueError(
                "The model has not been trained yet. Please train the model before saving a checkpoint."
            )

        # Auto-generate filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path("checkpoints") / self.dataset.dataset_name / timestamp

        return Checkpoint.save(
            self.evaluator,
            filepath,
            architecture=self._architecture_params,
            training_state=training_state,
            metadata=metadata,
        )

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str | Path,
        *,
        smiles_list: list[str] | None = None,
        labels: list[int] | None = None,
    ) -> "Pipeline":
        """
        Load a Pipeline from a checkpoint file.

        Creates a new Pipeline with the same architecture as the saved model,
        then loads the trained weights. The pipeline is ready for inference.

        Args:
            filepath: Path to checkpoint files (with or without extension).
            smiles_list: Optional SMILES list for building dataset.
                        If not provided, uses minimal dummy data.
            labels: Optional labels for building dataset.
                   Required if smiles_list is provided.

        Returns:
            A Pipeline instance with loaded weights, ready for inference.

        Raises:
            FileNotFoundError: If checkpoint files don't exist.
            ValueError: If checkpoint is missing architecture or incompatible.
        """
        # Load checkpoint data
        checkpoint_data = Checkpoint.load(filepath)

        if "architecture" not in checkpoint_data:
            raise ValueError(
                "Checkpoint does not contain architecture information. "
                "Cannot recreate Pipeline without architecture params."
            )

        arch = checkpoint_data["architecture"]

        # Convert architecture string back to enum if needed
        architecture_type = arch.get("architecture", "bare")
        if isinstance(architecture_type, str):
            architecture_type = ArchitectureType.from_string(architecture_type.upper())

        # Use provided data or create minimal dummy data
        # (needed to initialize the template, will be replaced by checkpoint weights)
        if smiles_list is None:
            # Use minimal dummy SMILES for initialization
            smiles_list = ["C"]  # Methane - simplest molecule
            labels = [0]

        # Create pipeline with same architecture
        pipeline = cls(
            dataset_name=arch["dataset_name"],
            model_name=arch["model_name"],
            param_size=arch["param_size"],
            layers=arch["layers"],
            max_depth=arch.get("max_depth", 1),
            max_subgraph_depth=arch.get("max_subgraph_depth", 5),
            max_cycle_size=arch.get("max_cycle_size", 10),
            subgraphs=arch.get("subgraphs"),
            chem_rules=arch.get("chem_rules"),
            architecture=architecture_type,
            funnel=arch.get("funnel", False),
            smiles_list=smiles_list,
            labels=labels,
            task=arch.get("task", "classification"),
        )

        # Build evaluator (needed to load weights)
        # Use default settings - weights will be overwritten anyway
        error_function = MSE if pipeline.task == "regression" else CrossEntropy
        settings = Settings(
            optimizer=Adam(lr=0.001),
            error_function=error_function(),
        )
        pipeline.evaluator = pipeline.template.build(settings)

        # Build dataset to initialize the evaluator's internal state
        pipeline.evaluator.build_dataset(pipeline.dataset.data, batch_size=1)

        # Load weights from checkpoint
        Checkpoint.load_weights_into(pipeline.evaluator, filepath)

        return pipeline
