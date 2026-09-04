"""
Experiment: Test broadcasting graph features to all atoms vs synthetic node approach.

Background:
- Atom/Bond features improve accuracy (+20-28% R²)
- Graph features via synthetic node HURT accuracy significantly (-0.91 R² avg)

Hypothesis:
The synthetic node approach creates architectural issues:
1. Extra node disrupts GNN message-passing
2. graph_bond edges may not participate correctly in aggregation
3. Information flow is asymmetric

Alternative approach (broadcast):
- Add graph features directly to ALL atoms as node-level features
- Same graph feature value appears on every atom in the molecule
- No synthetic node, no special edges
- Features contribute to atom_embed naturally

This experiment compares:
1. Baseline: No graph features
2. Synthetic Node: Original approach (expected to hurt)
3. Broadcast: New approach (hypothesis: should help or be neutral)
"""

import neuralogic
import numpy as np
import pandas as pd
from neuralogic.core import Settings, Transformation
from neuralogic.nn.loss import MSE
from neuralogic.nn.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from chemlogic.datasets.SmilesDataset import SmilesDataset
from chemlogic.models.models import get_model


def run_experiment(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_labels: list[float],
    param_size: int = 4,
    layers: int = 2,
    max_depth: int = 2,
    epochs: int = 2000,
    lr: float = 0.01,
    broadcast_graph_features: bool = False,
):
    """Run a single experiment configuration and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"{'=' * 60}")

    # Create training dataset
    dataset = SmilesDataset(
        smiles_list=train_df,
        param_size=param_size,
        dataset_name=f"exp_{name}",
        broadcast_graph_features=broadcast_graph_features,
    )

    # Load data
    train_data = dataset.load_data()
    print(
        f"Graph features: {dataset.graph_feature_names[:5]}... ({len(dataset.graph_feature_names)} total)"
    )
    print(f"Broadcast mode: {broadcast_graph_features}")

    # Create template with model
    dataset.create_template()

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
    settings = Settings(
        optimizer=Adam(lr=lr),
        error_function=MSE(),
    )

    # Create evaluator and train
    evaluator = dataset.build(settings)
    print(f"Training for {epochs} epochs...")
    built_dataset = evaluator.build_dataset(train_data)

    train_losses = []
    for epoch in range(epochs):
        evaluator.train(built_dataset)
        loss = evaluator.loss(built_dataset)
        train_losses.append(loss)
        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.4f}")

    final_train_loss = train_losses[-1]

    # Create test dataset with same settings
    test_dataset = SmilesDataset(
        smiles_list=test_df,
        param_size=param_size,
        dataset_name=f"exp_{name}_test",
        broadcast_graph_features=broadcast_graph_features,
    )
    test_data = test_dataset.load_data()
    built_test = evaluator.build_dataset(test_data)

    # Get predictions
    predictions = []
    for y_hat in evaluator.test(built_test):
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
    full_df = pd.read_csv(
        "/Users/hodzic/Documents/misc/melting-point-prediction-comp/train.csv"
    )

    # Use a smaller subset for faster experimentation
    SAMPLE_SIZE = 300

    # Test multiple seeds for robustness
    SEEDS = [42, 123, 456]
    all_results = []

    for seed in SEEDS:
        print(f"\n{'#' * 80}")
        print(f"# SEED: {seed}")
        print(f"{'#' * 80}")

        # Sample with this seed
        sample_df = full_df.sample(n=SAMPLE_SIZE, random_state=seed)

        # Identify Group columns (graph features)
        group_cols = [c for c in sample_df.columns if c.startswith("Group")]
        non_zero_cols = [c for c in group_cols if sample_df[c].sum() > 0]

        # Create DataFrame with graph features
        # Use top 30 non-zero group columns
        graph_df = pd.DataFrame(
            {
                "smiles": sample_df["SMILES"].tolist(),
                "target": sample_df["Tm"].tolist(),
            }
        )
        for col in non_zero_cols[:30]:
            graph_df[col] = sample_df[col].values

        # Also create baseline DataFrame (no graph features)
        baseline_df = pd.DataFrame(
            {
                "smiles": sample_df["SMILES"].tolist(),
                "target": sample_df["Tm"].tolist(),
            }
        )

        # Split data
        test_size = 0.2

        # For graph feature experiments
        train_indices, test_indices = train_test_split(
            range(len(graph_df)), test_size=test_size, random_state=seed
        )

        train_graph_df = graph_df.iloc[train_indices].reset_index(drop=True)
        test_graph_df = graph_df.iloc[test_indices].reset_index(drop=True)

        train_baseline_df = baseline_df.iloc[train_indices].reset_index(drop=True)
        test_baseline_df = baseline_df.iloc[test_indices].reset_index(drop=True)

        test_labels = test_graph_df["target"].tolist()

        print(f"Train size: {len(train_graph_df)}, Test size: {len(test_graph_df)}")

        # Common parameters
        params = {
            "param_size": 4,
            "layers": 2,
            "max_depth": 2,
            "epochs": 2000,
            "lr": 0.01,
        }

        seed_results = {"seed": seed}

        # Experiment 1: Baseline (no graph features)
        r = run_experiment(
            name=f"Baseline_s{seed}",
            train_df=train_baseline_df,
            test_df=test_baseline_df,
            test_labels=test_labels,
            broadcast_graph_features=False,
            **params,
        )
        seed_results["baseline"] = r["r2"]

        # Experiment 2: Synthetic Node approach
        r = run_experiment(
            name=f"Synthetic_s{seed}",
            train_df=train_graph_df,
            test_df=test_graph_df,
            test_labels=test_labels,
            broadcast_graph_features=False,
            **params,
        )
        seed_results["synthetic"] = r["r2"]

        # Experiment 3: Broadcast approach
        r = run_experiment(
            name=f"Broadcast_s{seed}",
            train_df=train_graph_df,
            test_df=test_graph_df,
            test_labels=test_labels,
            broadcast_graph_features=True,
            **params,
        )
        seed_results["broadcast"] = r["r2"]

        all_results.append(seed_results)

    # Multi-seed summary
    print("\n" + "=" * 80)
    print("MULTI-SEED SUMMARY")
    print("=" * 80)
    print(
        f"{'Seed':<10} {'Baseline':>12} {'Synthetic':>12} {'Broadcast':>12} {'Synth Δ':>12} {'Broad Δ':>12}"
    )
    print("-" * 70)

    for r in all_results:
        synth_delta = r["synthetic"] - r["baseline"]
        broad_delta = r["broadcast"] - r["baseline"]
        print(
            f"{r['seed']:<10} {r['baseline']:>12.4f} {r['synthetic']:>12.4f} {r['broadcast']:>12.4f} {synth_delta:>+12.4f} {broad_delta:>+12.4f}"
        )

    # Averages
    avg_baseline = np.mean([r["baseline"] for r in all_results])
    avg_synthetic = np.mean([r["synthetic"] for r in all_results])
    avg_broadcast = np.mean([r["broadcast"] for r in all_results])
    print("-" * 70)
    print(
        f"{'Average':<10} {avg_baseline:>12.4f} {avg_synthetic:>12.4f} {avg_broadcast:>12.4f} {avg_synthetic - avg_baseline:>+12.4f} {avg_broadcast - avg_baseline:>+12.4f}"
    )

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    if avg_synthetic > avg_baseline and avg_broadcast > avg_baseline:
        if avg_broadcast > avg_synthetic:
            print("✓ BROADCAST approach is BEST")
        else:
            print("✓ SYNTHETIC NODE approach is BEST")
    elif avg_synthetic > avg_baseline:
        print("✓ SYNTHETIC NODE helps, BROADCAST hurts")
    elif avg_broadcast > avg_baseline:
        print("✓ BROADCAST helps, SYNTHETIC NODE hurts")
    else:
        print("✗ Both approaches HURT performance")

    return all_results


if __name__ == "__main__":
    results = main()
