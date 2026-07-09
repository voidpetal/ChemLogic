"""
Optional experiments subpackage — requires the 'experiments' extra:

    pip install ChemLogic[experiments]
"""

try:
    import mlflow
    import optuna as _  # noqa: F401 — verify optuna present too
except ImportError as e:
    raise ImportError(
        "chemlogic.experiments requires optional dependencies. "
        "Install them with: pip install ChemLogic[experiments]"
    ) from e

from chemlogic.utils.Pipeline import ArchitectureType, Pipeline


# TODO: do a better job in refactoring here
def main(
    trial,
    dataset_name,
    model_name,
    chemical_rules,
    subgraphs,
    architecture="bare",
    batches=1,
    smiles_list=None,
    labels=None,
    task="classification",
):
    """Run a single experiment/training cycle and log results to MLflow.

    Designed to be called from an Optuna objective function or any HPO
    framework. Reads hyperparameter suggestions from `trial`, constructs a
    Pipeline, runs the training/testing cycle, and logs params and metrics
    to the active MLflow run.

    Args:
        trial: An object exposing the `.suggest_*` API (e.g. Optuna Trial).
        dataset_name (str): Dataset identifier accepted by Pipeline.
        model_name (str): Model architecture key accepted by Pipeline.
        chemical_rules (bool | None): If truthy, chemical rule flags are
            sampled from `trial` and passed to the pipeline.
        subgraphs (bool | None): If truthy, subgraph flags and depth
            parameters are sampled from `trial`.
        architecture (str): Architecture mode string ("bare", "CCE", "CCD").
        batches (int): Number of dataset batches. Default 1.
        smiles_list (list[str] | None): Optional SMILES strings for a custom dataset.
        labels (list | None): Labels corresponding to `smiles_list`.
        task (str): Task type — auto-detected from labels if None.

    Returns:
        tuple: (metric, pipeline) — AUROC or R² and the trained Pipeline.
    """
    with mlflow.start_run():
        max_subgraph_depth = 0
        max_cycle_size = 0
        if chemical_rules:
            chemical_rules = [
                trial.suggest_categorical(i, [True, False])
                for i in ["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]
            ]
        if subgraphs:
            max_subgraph_depth = trial.suggest_int("max_subgraph_depth", 1, 8)
            max_cycle_size = trial.suggest_int("max_cycle_size", 3, 10)
            subgraphs = [
                trial.suggest_categorical(i, [True, False])
                for i in [
                    "cycles",
                    "paths",
                    "y_shape",
                    "nbhoods",
                    "circular",
                    "collective",
                ]
            ]

        param_size = trial.suggest_int("param_size", 1, 4)
        layers = trial.suggest_int("layers", 1, 4)
        max_depth = (
            trial.suggest_int("max_depth", 2, 10)
            if model_name in ["sgn", "diffusion", "cw_net"]
            else 1
        )
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

        pipeline = Pipeline(
            dataset_name,
            model_name,
            param_size,
            layers,
            max_depth,
            max_subgraph_depth=max_subgraph_depth,
            max_cycle_size=max_cycle_size,
            architecture=ArchitectureType.from_string(architecture),
            subgraphs=subgraphs,
            chem_rules=chemical_rules,
            smiles_list=smiles_list,
            labels=labels,
            task=task,
        )

        train_loss, test_loss, metric, _ = pipeline.train_test_cycle(
            lr, 500, 0.7, batches=batches
        )

        mlflow.log_params(
            {
                "dataset": dataset_name,
                "model": model_name,
                "max_depth": max_depth,
                "parameter_size": param_size,
                "num_layers": layers,
                "learning_rate": lr,
                "architecture": architecture,
                "funnel": False,
            }
        )
        if chemical_rules:
            mlflow.log_params(
                dict(
                    zip(
                        ["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"],
                        chemical_rules,
                        strict=False,
                    )
                )
            )
        else:
            mlflow.log_param("chem_rules", None)
        if subgraphs:
            mlflow.log_params(
                {"subgraph_depth": max_subgraph_depth, "cycle_size": max_cycle_size}
            )
            mlflow.log_params(
                dict(
                    zip(
                        [
                            "cycles",
                            "paths",
                            "y_shape",
                            "nbhoods",
                            "circular",
                            "collective",
                        ],
                        subgraphs,
                        strict=False,
                    )
                )
            )
        else:
            mlflow.log_param("subgraphs", None)
        mlflow.log_metrics(
            {"train_loss": train_loss, "test_loss": test_loss, "metric": metric}
        )

    return metric, pipeline


__all__ = ["main"]
