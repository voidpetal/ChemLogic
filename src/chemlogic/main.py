import mlflow

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
        funnel = False
        layers = trial.suggest_int("layers", 1, 4)
        if model_name in ["sgn", "diffusion", "cw_net"]:
            max_depth = trial.suggest_int("max_depth", 2, 10)
        else:
            max_depth = 1

        lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        epochs = 500
        split = 0.7

        architecture_type = ArchitectureType.from_string(architecture)
        pipeline = Pipeline(
            dataset_name,
            model_name,
            param_size,
            layers,
            max_depth,
            max_subgraph_depth=max_subgraph_depth,
            max_cycle_size=max_cycle_size,
            architecture=architecture_type,
            subgraphs=subgraphs,
            chem_rules=chemical_rules,
            funnel=funnel,
            smiles_list=smiles_list,
            labels=labels,
            task=task
        )

        train_loss, test_loss, metric, evaluator = pipeline.train_test_cycle(
            lr, epochs, split, batches=batches
        )

        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("parameter_size", param_size)
        mlflow.log_param("num_layers", layers)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("architecture", architecture)
        mlflow.log_param("funnel", funnel)
        # for chemical rules
        if not chemical_rules:
            mlflow.log_param("chem_rules", None)
        else:
            for i, param in enumerate(
                ["hydrocarbons", "oxy", "nitro", "sulfuric", "relaxations"]
            ):
                mlflow.log_param(param, chemical_rules[i])
        if not subgraphs:
            mlflow.log_param("subgraphs", None)
        else:
            mlflow.log_param("subgraph_depth", max_subgraph_depth)
            mlflow.log_param("cycle_size", max_cycle_size)
            for i, param in enumerate(
                ["cycles", "paths", "y_shape", "nbhoods", "circular", "collective"]
            ):
                mlflow.log_param(param, subgraphs[i])

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("metric", metric)

    return metric
