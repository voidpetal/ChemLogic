"""
Comparative experiment: Test whether extended features (graph/atom/bond) improve accuracy.

This experiment compares:
1. Baseline: No extended features (just SMILES structure)
2. Graph Features: Using DataFrame columns as graph-level predicates
3. Atom Features: Using RDKit atom features as node-level predicates
4. Bond Features: Using RDKit bond features as edge-level predicates
5. All Features: Graph + Atom + Bond features combined

Dataset: Melting point prediction (regression task)
"""

import pandas as pd
import numpy as np
import neuralogic
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from chemlogic.utils.Pipeline import Pipeline, ArchitectureType


def run_experiment(
    name: str,
    train_smiles: list[str],
    train_labels: list[float],
    test_smiles: list[str],
    test_labels: list[float],
    param_size: int = 4,
    layers: int = 2,
    max_depth: int = 2,
    epochs: int = 5000,
    lr: float = 0.01,
    atom_features=None,
    bond_features=None,
    graph_features_df: pd.DataFrame = None,
):
    """Run a single experiment configuration and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"{'=' * 60}")

    # If graph features provided, we need to use DataFrame input
    if graph_features_df is not None:
        # For training: create DataFrame with smiles, target, and graph features
        # Filter to training indices
        train_df = graph_features_df[
            graph_features_df["smiles"].isin(train_smiles)
        ].copy()

        # Create dataset using Pipeline with DataFrame
        # Pipeline doesn't support DataFrame directly, so we use SmilesDataset
        from chemlogic.datasets.SmilesDataset import SmilesDataset

        dataset = SmilesDataset(
            smiles_list=train_df,
            param_size=param_size,
            dataset_name=f"exp_{name}",
            atom_features=atom_features,
            bond_features=bond_features,
        )
    else:
        # Create dataset without graph features
        from chemlogic.datasets.SmilesDataset import SmilesDataset

        dataset = SmilesDataset(
            smiles_list=train_smiles,
            labels=train_labels,
            param_size=param_size,
            dataset_name=f"exp_{name}",
            atom_features=atom_features,
            bond_features=bond_features,
        )

    # Load data
    train_data = dataset.load_data()

    # Create template with model
    dataset.create_template()

    from chemlogic.models.models import get_model
    from neuralogic.core import Transformation

    model_rules = get_model(
        "gnn",
        layers,
        dataset.node_embed,
        dataset.edge_embed,
        dataset.connection,
        param_size,
        edge_types=dataset.bond_types,
        max_depth=max_depth,
        output_layer_name="predict",
        output_layer_transformation=Transformation.IDENTITY,  # Regression
    )
    dataset.add_rules(model_rules)

    # Training settings
    from neuralogic.core import Settings
    from neuralogic.nn import get_evaluator
    from neuralogic.nn.loss import MSE
    from neuralogic.optim import Adam

    settings = Settings(
        optimizer=Adam(lr=lr),
        epochs=epochs,
        error_function=MSE(),
    )

    # Create evaluator
    evaluator = get_evaluator(dataset, settings)

    # Train
    print(f"Training for {epochs} epochs...")
    built_dataset = evaluator.build_dataset(train_data)

    train_losses = []
    for epoch, (loss, _) in enumerate(evaluator.train(built_dataset)):
        train_losses.append(loss)
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.4f}")

    final_train_loss = train_losses[-1]

    # Inference on test set
    print("Running inference on test set...")

    # Create test dataset
    if graph_features_df is not None:
        test_df = graph_features_df[
            graph_features_df["smiles"].isin(test_smiles)
        ].copy()
        test_dataset = SmilesDataset(
            smiles_list=test_df,
            param_size=param_size,
            dataset_name=f"exp_{name}_test",
            atom_features=atom_features,
            bond_features=bond_features,
        )
    else:
        test_dataset = SmilesDataset(
            smiles_list=test_smiles,
            labels=test_labels,
            param_size=param_size,
            dataset_name=f"exp_{name}_test",
            atom_features=atom_features,
            bond_features=bond_features,
        )

    test_data = test_dataset.load_data()
    built_test = evaluator.build_dataset(test_data)

    # Get predictions using evaluator.test()
    predictions = []
    for y_hat in evaluator.test(built_test, generator=False):
        predictions.append(float(y_hat))

    # Calculate metrics
    r2 = r2_score(test_labels, predictions)
    mae = mean_absolute_error(test_labels, predictions)
    rmse = np.sqrt(mean_squared_error(test_labels, predictions))

    print(f"\nResults for {name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Final Train Loss: {final_train_loss:.4f}")

    return {
        "name": name,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "train_loss": final_train_loss,
    }


def main():
    # Set JVM options for larger heap
    neuralogic.set_jvm_options(["-Xms2g", "-Xmx8g"])

    # Load melting point dataset
    print("Loading melting point dataset...")
    train_df = pd.read_csv(
        "/Users/hodzic/Documents/misc/melting-point-prediction-comp/train.csv"
    )

    # Use a smaller subset for faster experimentation
    SAMPLE_SIZE = 300  # Use subset for faster experiments
    if len(train_df) > SAMPLE_SIZE:
        train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)

    print(f"Dataset size: {len(train_df)} molecules")

    # Extract SMILES and labels
    smiles_list = train_df["SMILES"].tolist()
    labels = train_df["Tm"].tolist()

    # Identify Group columns (graph features)
    group_cols = [c for c in train_df.columns if c.startswith("Group")]
    print(f"Found {len(group_cols)} Group columns for graph features")

    # Filter to non-zero columns (many are all zeros)
    non_zero_cols = [c for c in group_cols if train_df[c].sum() > 0]
    print(f"Non-zero Group columns: {len(non_zero_cols)}")

    # Create DataFrame with graph features for experiments that need it
    graph_df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "target": labels,
        }
    )
    # Add top 30 non-zero group columns as graph features
    for col in non_zero_cols[:30]:
        graph_df[col] = train_df[col].values

    # Split data once - use same split for all experiments
    test_size = 0.2
    random_state = 42
    train_smiles, test_smiles, train_labels, test_labels = train_test_split(
        smiles_list, labels, test_size=test_size, random_state=random_state
    )

    print(f"Train size: {len(train_smiles)}, Test size: {len(test_smiles)}")

    # Common parameters
    params = {
        "param_size": 4,
        "layers": 2,
        "max_depth": 2,
        "epochs": 2000,
        "lr": 0.01,
    }

    results = []

    # Experiment 1: Baseline (no extended features)
    results.append(
        run_experiment(
            name="Baseline",
            train_smiles=train_smiles,
            train_labels=train_labels,
            test_smiles=test_smiles,
            test_labels=test_labels,
            **params,
        )
    )

    # Experiment 2: Atom features only
    results.append(
        run_experiment(
            name="AtomFeatures",
            train_smiles=train_smiles,
            train_labels=train_labels,
            test_smiles=test_smiles,
            test_labels=test_labels,
            atom_features="all",
            **params,
        )
    )

    # Experiment 3: Bond features only
    results.append(
        run_experiment(
            name="BondFeatures",
            train_smiles=train_smiles,
            train_labels=train_labels,
            test_smiles=test_smiles,
            test_labels=test_labels,
            bond_features="all",
            **params,
        )
    )

    # Experiment 4: Atom + Bond features
    results.append(
        run_experiment(
            name="Atom+Bond",
            train_smiles=train_smiles,
            train_labels=train_labels,
            test_smiles=test_smiles,
            test_labels=test_labels,
            atom_features="all",
            bond_features="all",
            **params,
        )
    )

    # Experiment 5: Graph features only (via DataFrame)
    results.append(
        run_experiment(
            name="GraphFeatures",
            train_smiles=train_smiles,
            train_labels=train_labels,
            test_smiles=test_smiles,
            test_labels=test_labels,
            graph_features_df=graph_df,
            **params,
        )
    )

    # Experiment 6: All features combined
    results.append(
        run_experiment(
            name="AllFeatures",
            train_smiles=train_smiles,
            train_labels=train_labels,
            test_smiles=test_smiles,
            test_labels=test_labels,
            graph_features_df=graph_df,
            atom_features="all",
            bond_features="all",
            **params,
        )
    )

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<20} {'R²':>10} {'MAE':>10} {'RMSE':>10} {'Train Loss':>12}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['name']:<20} {r['r2']:>10.4f} {r['mae']:>10.2f} {r['rmse']:>10.2f} {r['train_loss']:>12.4f}"
        )

    # Calculate improvements
    baseline_r2 = results[0]["r2"]
    print("\n" + "-" * 62)
    print("Improvement over Baseline (R² difference):")
    for r in results[1:]:
        diff = r["r2"] - baseline_r2
        sign = "+" if diff > 0 else ""
        print(f"  {r['name']:<20}: {sign}{diff:.4f}")

    return results


if __name__ == "__main__":
    results = main()
