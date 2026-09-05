"""Compare a learnable custom nitro rule against a minimal-KB baseline."""

from neuralogic.core import R, V

from chemlogic.utils.Pipeline import Pipeline


# This mirrors the raw structure used by the built-in nitro rule, while keeping
# the built-in nitro category disabled. It emits a learnable four-dimensional
# feature for a carbon attached to R-N(=O)-O.
custom_template = [
    # Nitro: R-N(=O)-O.
    R.custom_feature(V.R)[4,]
    <= (
        R.custom_input(V.R),
        R.c(V.R),
        R.n(V.N),
        R.o(V.O1),
        R.o(V.O2),
        R.bond(V.R, V.N, V.B1),
        R.b_1(V.B1),
        R.bond(V.N, V.O1, V.B2),
        R.b_2(V.B2),
        R.bond(V.N, V.O2, V.B3),
        R.b_1(V.B3),
        R.special.alldiff(V.R, V.N, V.O1, V.O2),
    ),
    # Carbonyl: C=O.
    R.custom_feature(V.C)[4,]
    <= (
        R.custom_input(V.C),
        R.c(V.C),
        R.o(V.O),
        R.bond(V.C, V.O, V.B),
        R.b_2(V.B),
    ),
    # Halogen attachment: R-X. One rule per halogen predicate.
    R.custom_feature(V.R)[4,]
    <= (R.custom_input(V.R), R.bond(V.R, V.X, V.B), R.b_1(V.B), R.f(V.X)),
    R.custom_feature(V.R)[4,]
    <= (R.custom_input(V.R), R.bond(V.R, V.X, V.B), R.b_1(V.B), R.cl(V.X)),
    R.custom_feature(V.R)[4,]
    <= (R.custom_input(V.R), R.bond(V.R, V.X, V.B), R.b_1(V.B), R.br(V.X)),
    R.custom_feature(V.R)[4,]
    <= (R.custom_input(V.R), R.bond(V.R, V.X, V.B), R.b_1(V.B), R.i(V.X)),
]


if __name__ == "__main__":
    common_args = {
        "dataset_name": "mutagen",
        "model_name": "gnn",
        "param_size": 4,
        "layers": 2,
        "chem_rules": (True, False, False, False, False),
        "subgraphs": False,
    }

    baseline = Pipeline(**common_args)
    baseline_result = baseline.train_test_cycle(epochs=30)

    custom = Pipeline(
        **common_args,
        custom_rules=custom_template,
        custom_input="custom_input",
        custom_output="custom_feature",
    )
    custom_result = custom.train_test_cycle(epochs=30)

    print(f"baseline AUROC: {baseline_result[2]:.4f}")
    print(f"custom AUROC:   {custom_result[2]:.4f}")
    print(f"AUROC delta:    {custom_result[2] - baseline_result[2]:+.4f}")
