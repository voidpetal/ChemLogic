# ChemLogic Specification

Technical reference for APIs and configuration. For introduction and installation, see [README](../README.md).

---

## Table of Contents

1. [Datasets](#datasets)
2. [Extended Features](#extended-features)
3. [Models](#models)
4. [Knowledge Base](#knowledge-base)
5. [Pipeline](#pipeline)
6. [Architecture Types](#architecture-types)
7. [Checkpointing](#checkpointing)
8. [Configuration Reference](#configuration-reference)

---

## Datasets

### Supported Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| `mutagen` | [TUD](https://chrsmrrs.github.io/datasets/docs/datasets/) | 183 | Mutagenicity prediction |
| `ptc`, `ptc_fr`, `ptc_mm`, `ptc_fm` | [TUD](https://chrsmrrs.github.io/datasets/docs/datasets/) | 336-351 | Toxicity prediction (various species) |
| `dhfr` | [TUD](https://chrsmrrs.github.io/datasets/docs/datasets/) | 393 | DHFR inhibition |
| `er` | [TUD](https://chrsmrrs.github.io/datasets/docs/datasets/) | 446 | Estrogen receptor binding |
| `blood_brain_barrier` | [TDC](https://tdcommons.ai) | 2030 | Blood-brain barrier penetration |
| `skin_reaction` | [TDC](https://tdcommons.ai) | 404 | Skin sensitization |
| `oral_bioavailability` | [TDC](https://tdcommons.ai) | 640 | Oral bioavailability |
| `carcinogenous` | [TDC](https://tdcommons.ai) | 280 | Carcinogenicity |
| `pampa_permeability` | [TDC](https://tdcommons.ai) | 2034 | Membrane permeability |
| `human_intestinal_absorption` | [TDC](https://tdcommons.ai) | 578 | Intestinal absorption |
| `p_glycoprotein_inhibition` | [TDC](https://tdcommons.ai) | 1218 | P-gp inhibition |
| `cyp2c9_substrate`, `cyp2d6_substrate`, `cyp3a4_substrate` | [TDC](https://tdcommons.ai) | 667-670 | CYP enzyme substrates |
| `anti_sarscov2_activity` | [TDC](https://tdcommons.ai) | 1484 | SARS-CoV-2 activity |

### Custom Datasets via SMILES

```python
from chemlogic.utils.Pipeline import Pipeline

pipeline = Pipeline(
    dataset_name="my_dataset",
    model_name="gnn",
    param_size=2,
    layers=2,
    smiles_list=["CCO", "CC(=O)O", "c1ccccc1"],
    labels=[0, 1, 0],
)
```

### Custom Datasets via DataFrame (with Extended Features)

```python
import pandas as pd
from chemlogic.datasets import SmilesDataset

# DataFrame with SMILES, target, and additional numeric columns
df = pd.DataFrame(
    {
        "smiles": ["CCO", "CC(=O)O", "c1ccccc1"],
        "target": [0, 1, 0],
        "mol_weight": [46.07, 60.05, 78.11],  # Graph-level feature
        "log_p": [-0.18, -0.17, 2.13],  # Graph-level feature
    }
)

dataset = SmilesDataset(
    df,
    atom_features=["formal_charge", "degree"],  # Node-level features
    bond_features=["is_aromatic"],  # Edge-level features
)
```

### Dataset Structure

Each dataset defines:
- **Atom types**: Predicates for elements (e.g., `c`, `o`, `n`, `s`, `h`)
- **Bond types**: Predicates for bond orders (single, double, triple, aromatic)
- **Connectivity**: `bond(X, Y, B)` predicate linking atoms X and Y via bond B
- **Extended features**: Optional atom, bond, and graph-level features as valued predicates

---

## Extended Features

ChemLogic supports three levels of extended features that can enhance model accuracy by incorporating additional chemical information beyond basic atom and bond types.

### Feature Levels

| Level | Scope | Example | Format |
|-------|-------|---------|--------|
| **Graph-level** | Entire molecule | Molecular weight, LogP | `<value> feature(graph_node)` |
| **Node-level** | Individual atoms | Formal charge, degree | `<value> feature(atom_id)` |
| **Edge-level** | Individual bonds | Is aromatic, is conjugated | `<value> feature(bond_id)` |

### Graph-Level Features (via Synthetic Node)

Graph-level features are integrated via a **synthetic graph node** that connects to all atoms:

```
Molecule: C-C-O (ethanol)

     [C]----[C]----[O]        Real atoms (0, 1, 2)
      |      |      |
      +------+------+
             |
           [G]                Synthetic graph node
             |
    mol_weight=46.07          Graph features on synthetic node
    log_p=-0.18
```

**Architecture benefits:**
- Features flow through GNN message-passing naturally
- Graph features don't overpower node-level information
- No atom type on synthetic node → KB chemical rules won't trigger
- Balanced contribution alongside atom embeddings

**Usage:**
```python
# Via DataFrame - extra numeric columns become graph features
df = pd.DataFrame(
    {
        "smiles": ["CCO", "CC"],
        "target": [1, 0],
        "mol_weight": [46.07, 30.07],  # Automatically detected as graph feature
    }
)
dataset = SmilesDataset(df)

# Alternative: Broadcast mode (adds graph features to all atoms, no synthetic node)
dataset = SmilesDataset(df, broadcast_graph_features=True)
```

**Broadcast Mode:**

Instead of a synthetic node, broadcast mode adds graph features directly to all atoms in the molecule. Each atom receives the same graph-level feature values:

```
Molecule: C-C-O (ethanol) with broadcast_graph_features=True

[C]----[C]----[O]
 |      |      |
mol_weight(0)=46.07
mol_weight(1)=46.07
mol_weight(2)=46.07
```

Use broadcast mode when you want simpler architecture without virtual nodes.

### Node-Level Features (Atom Features)

RDKit atom properties extracted as valued predicates on each atom.

| Feature | Description | Type |
|---------|-------------|------|
| `formal_charge` | Formal charge of atom | int |
| `num_radical_electrons` | Number of radical electrons | int |
| `is_aromatic` | Whether atom is aromatic | bool (0/1) |
| `hybridization` | Hybridization state (SP, SP2, SP3, etc.) | int |
| `total_num_hs` | Total number of hydrogens | int |
| `degree` | Number of bonded neighbors | int |
| `is_in_ring` | Whether atom is in a ring | bool (0/1) |
| `chiral_tag` | Chirality tag | int |

**Usage:**
```python
# Enable specific features
dataset = SmilesDataset(smiles_list, labels, atom_features=["formal_charge", "degree"])

# Enable all available features
dataset = SmilesDataset(smiles_list, labels, atom_features="all")
```

### Edge-Level Features (Bond Features)

RDKit bond properties extracted as valued predicates on each bond.

| Feature | Description | Type |
|---------|-------------|------|
| `is_aromatic` | Whether bond is aromatic | bool (0/1) |
| `is_conjugated` | Whether bond is conjugated | bool (0/1) |
| `is_in_ring` | Whether bond is in a ring | bool (0/1) |
| `stereo` | Stereochemistry type | int |

**Usage:**
```python
# Enable specific features
dataset = SmilesDataset(
    smiles_list, labels, bond_features=["is_aromatic", "is_in_ring"]
)

# Enable all available features
dataset = SmilesDataset(smiles_list, labels, bond_features="all")
```

### Zero-Value Optimization

Features with value `0` are automatically omitted from the dataset. Since zero-valued predicates contribute nothing to neural network computation, this optimization:
- Reduces dataset size significantly (especially for sparse features)
- Improves training speed
- Maintains mathematical equivalence

### Combined Example

```python
import pandas as pd
from chemlogic.datasets import SmilesDataset
from chemlogic.models import GNN

# Dataset with all feature levels
df = pd.DataFrame(
    {
        "smiles": ["c1ccccc1", "CCO", "CC(=O)O"],
        "target": [0.5, 1.2, 0.8],
        "mol_weight": [78.11, 46.07, 60.05],
        "tpsa": [0.0, 20.23, 37.30],
    }
)

dataset = SmilesDataset(
    df,
    atom_features=["formal_charge", "is_aromatic", "degree"],
    bond_features=["is_aromatic", "is_in_ring"],
)

# Model rules created:
# - atom_embed(A) <= c(A)           # Atom type
# - atom_embed(A) <= formal_charge(A)  # Atom feature
# - atom_embed(A) <= is_aromatic(A)    # Atom feature
# - atom_embed(A) <= degree(A)         # Atom feature
# - atom_embed(G) <= mol_weight(G)     # Graph feature on synthetic node
# - atom_embed(G) <= tpsa(G)           # Graph feature on synthetic node
# - bond_embed(B) <= b_1(B)            # Bond type
# - bond_embed(B) <= is_aromatic(B)    # Bond feature
# - bond_embed(B) <= is_in_ring(B)     # Bond feature
# - bond_embed(B) <= graph_bond(B)     # Synthetic node connections
```

### Experiment Results

Extended features were validated on a **melting point prediction** regression task using 300 molecules (240 train / 60 test) with 2000 training epochs.

#### Atom & Bond Features

| Configuration | R² | MAE | RMSE | Improvement |
|---------------|-----|-----|------|-------------|
| Baseline (structure only) | 0.232 | 53.4 | 78.9 | — |
| Atom features | 0.279 | 47.6 | 76.5 | **+20.4%** |
| Bond features | 0.297 | 49.7 | 75.5 | **+28.2%** |
| Atom + Bond | 0.287 | 49.5 | 76.1 | **+23.7%** |

#### Graph Features (Multi-seed Validation)

Graph-level features (30 Group contribution columns) were tested with two approaches:

| Seed | Baseline R² | Synthetic Node R² | Broadcast R² |
|------|-------------|-------------------|--------------|
| 42   | 0.231 | 0.430 (+86%) | 0.402 (+74%) |
| 123  | 0.335 | 0.425 (+27%) | 0.425 (+27%) |
| 456  | 0.127 | 0.185 (+46%) | 0.125 (-2%) |
| **Avg** | **0.231** | **0.347 (+50%)** | **0.317 (+37%)** |

- **Synthetic Node**: Adds a virtual node connected to all atoms via `graph_bond` edges
- **Broadcast**: Adds graph features directly to all atoms (no virtual node)

Both approaches show improvement over baseline. Use `broadcast_graph_features=True` in `SmilesDataset` to enable broadcast mode.

**Key findings:**

1. **Node-level features (atoms) improve accuracy** — Adding RDKit atom properties like formal charge, degree, hybridization, etc. provides ~20% improvement in R² score.

2. **Edge-level features (bonds) improve accuracy** — Bond properties like aromaticity, conjugation, and ring membership provide ~28% improvement.

3. **Combined features are effective** — Using both atom and bond features together gives solid improvement (~24%), though not strictly additive.

4. **Graph-level features show promise** — Both synthetic node and broadcast approaches improve accuracy on average. The synthetic node approach shows slightly better results but adds architectural complexity.

**Recommendation:** Start with atom and/or bond features for immediate accuracy gains. Graph-level features can provide additional improvement, with the synthetic node approach showing the best results in our experiments.

---

## Models

### Available Models

| Model | Key | Description |
|-------|-----|-------------|
| GNN | `gnn` | Standard graph neural network with edge features |
| RGCN | `rgcn` | Relational GCN with typed edges |
| KGNN | `kgnn` (`kgnn_local` for local variant) | Higher-order GNN (k-GNN) |
| EgoGNN | `ego` | Ego-centric graph neural network |
| SGN | `sgn` | Subgraph network (requires `max_depth`) |
| DiffusionCNN | `diffusion` | Diffusion convolutional network (requires `max_depth`) |
| CWNet | `cw` | CW-Network (requires `max_depth`) |

### Model Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `layers` | int | Number of message-passing layers |
| `param_size` | int | Embedding dimension |
| `max_depth` | int | Propagation depth (SGN, DiffusionCNN, CWNet only) |

---

## Knowledge Base

### Chemical Rules (Functional Groups)

| Category | Key | Groups Included |
|----------|-----|-----------------|
| General | (always included) | Hydroxyl (-OH), Carbonyl (C=O), Halogen (R-X) |
| Hydrocarbons | `hydrocarbons` | Alkanes, Alkenes, Alkynes, etc. |
| Oxygen Groups | `oxy` | Alcohol, Ether, Ester, Carboxylic acid, Aldehyde, Ketone |
| Nitrogen Groups | `nitro` | Amine, Amide, Nitro, Nitrile, Imine |
| Sulfur Groups | `sulfuric` | Thiol, Sulfide, Sulfoxide, Sulfone |
| Relaxations | `relaxations` | Generalized patterns for approximate matching |

**Configuration:**
```python
chem_rules = True  # Enable all

# Or select categories: (hydrocarbons, oxy, nitro, sulfuric, relaxations)
chem_rules = (True, True, False, False, True)
```

### Custom Knowledge-Base Rules

Pass a NeuraLogic rule list or `Model` to `Pipeline(custom_rules=...)` and
declare its boundary predicates with `custom_input` and `custom_output`.
Pipeline adds adapter rules around the template.

When `custom_rules` is supplied, ChemLogic adds at least the base chemical
rules. If `chem_rules` is also configured, its optional categories and path
rules are preserved unchanged.

- Input: `custom_input` is the predicate expected by the custom template. It is mapped from the KB input: `node_embed` in BARE/CCE or `kb_features` in CCD.
- Output: `custom_output` is the predicate produced by the custom template. It is mapped to `predict` in BARE/CCD or `kb_features` in CCE.

The predicate names are configurable. `custom_input` and `custom_output` are
argument names only; any valid NeuraLogic predicate names may be supplied.

```python
from chemlogic.utils.Pipeline import Pipeline

custom_template = [
    R.custom_feature(V.C)[4,]
    <= (
        R.custom_input(V.C),
        R.c(V.C),
        R.o(V.O),
        R.bond(V.C, V.O, V.B),
        R.b_2(V.B),
    )
]

pipeline = Pipeline(
    "mutagen", "gnn", param_size=4, layers=2,
    chem_rules=True,
    subgraphs=False,
    custom_rules=custom_template,
    custom_input="custom_input",
    custom_output="custom_feature",
)
```

The runnable comparison is in `examples/custom_rules.py`. It disables the
nitro category in the built-in KB, defines raw nitro, carbonyl, and halogen
structures as custom learnable features, and trains baseline and custom
pipelines for comparison.

### Subgraph Patterns

| Pattern | Key | Description |
|---------|-----|-------------|
| Cycles | `cycles` | Ring structures of configurable size |
| Paths | `paths` | Linear chains up to `max_depth` |
| Y-Shape | `y_shape` | Branching patterns |
| Neighborhoods | `nbhoods` | Local atom environments |
| Circular | `circular` | Circular fingerprint-like patterns |
| Collective | `collective` | Combined structural features |

**Configuration:**
```python
subgraphs = True  # Enable all

# Or select patterns: (cycles, paths, y_shape, nbhoods, circular, collective)
subgraphs = (True, True, False, True, False, False)
```

**Subgraph Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_subgraph_depth` | int | 5 | Maximum path length |
| `max_cycle_size` | int | 10 | Maximum ring size to detect |

---

## Pipeline

### Initialization

```python
Pipeline(
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
    funnel: bool = False,
    smiles_list: list[str] = None,
    labels: list = None,
    task: str | None = None,        # auto-detected from labels if not provided
    atom_features: str | list[str] | None = None,
    bond_features: str | list[str] | None = None,
    graph_features: dict | None = None,
    num_outputs: int = 1,           # auto-detected from label shape if task is auto-detected
)
```

`task` and `num_outputs` are auto-detected from label structure when not provided:

| Label | Inferred task | num_outputs |
|---|---|---|
| `[4.56, 3.21, ...]` — scalars with any float | `regression` | 1 |
| `[0, 1, 0, 1, ...]` — integers `{0,1}` only | `classification` | 1 |
| `[0, 1, 2, 3, ...]` — integers N>2 unique | `regression` | 1 |
| `[[1,0,0], [0,1,0], ...]` — sequences summing to 1 with values in `{0,1}` | `multi_class` | `len(label)` |
| everything else sequence — `[(4.5,2.1), ...]`, `[[1.2,2.3], ...]`, `[(1,0,1), ...]` | `multi_regression` | `len(label)` |

Integers with N>2 unique values (ordinal, count) warn and default to `regression` — they are ambiguous. Pass `task=` explicitly.

`graph_features` accepts a `dict` mapping feature name to a list of per-molecule values:
```python
pipeline = Pipeline(
    ...,
    graph_features={"mol_weight": [46.07, 78.11], "log_p": [-0.18, 2.13]},
)
```

`num_outputs` controls the output layer dimension. For `task="multi_class"` set it to the number of classes.

### Training

```python
train_loss, test_loss, metric, model = pipeline.train_test_cycle(
    lr: float = 0.001,
    epochs: int = 100,
    split_ratio: float = 0.75,
    optimizer = Adam,
    error_function = None,           # defaults to CrossEntropy or MSE based on task
    batches: int = 1,
    early_stopping_threshold: float = 0.001,
    early_stopping_rounds: int = 10,
    checkpoint_every: int | None = None,    # save checkpoint every N epochs
    checkpoint_dir: str | None = None,      # defaults to checkpoints/{dataset_name}/
)
```

Returns `(train_loss_last_epoch, test_loss, metric_score, model)`. Metric is AUROC for classification/multi-class and R² for regression. The returned object is the built Neuralogic 0.9 `Model`.

### Inference

```python
predictions = pipeline.inference(smiles_list=["CCO", "CC(=O)O"])
```

Raises `ValueError` if called before `train_test_cycle`.

### Saving and loading checkpoints

```python
# Save after training
safetensors_path, json_path = pipeline.save_checkpoint()

# Save to specific path
pipeline.save_checkpoint("checkpoints/my_model")

# Load and run inference
pipeline = Pipeline.from_checkpoint("checkpoints/my_model")
predictions = pipeline.inference(["CCO", "CC(=O)O"])
```

`save_checkpoint` accepts optional `training_state` and `metadata` dicts for bookkeeping.

### Visualization

```python
pipeline.template.draw()  # Requires graphviz
```

### Example with Extended Features

```python
from chemlogic.utils.Pipeline import Pipeline
import pandas as pd

# Pipeline with atom, bond, and graph features
df = pd.read_csv("molecules.csv")  # must have SMILES and target columns
pipeline = Pipeline(
    dataset_name="my_dataset",
    model_name="gnn",
    param_size=8,
    layers=2,
    smiles_list=df["SMILES"],
    labels=df["target"],
    task="regression",
    atom_features=["formal_charge", "is_aromatic", "degree"],
    bond_features=["is_aromatic", "is_in_ring"],
    graph_features={"num_atoms": df["num_atoms"], "logP": df["logP"]},
)

train_loss, test_loss, r2, model = pipeline.train_test_cycle(epochs=100)
print(f"R²: {r2:.4f}")
```

### Example: Multi-class classification

```python
import pandas as pd
from chemlogic.utils.Pipeline import Pipeline

df = pd.read_csv("molecules.csv")
# Create 3-class labels from a continuous property
labels = pd.cut(df["target"], bins=3, labels=[0, 1, 2]).astype(int).tolist()

pipeline = Pipeline(
    dataset_name="my_dataset",
    model_name="gnn",
    param_size=8,
    layers=2,
    smiles_list=df["SMILES"],
    labels=labels,
    task="multi_class",
    num_outputs=3,
)

train_loss, test_loss, auroc_ovr, model = pipeline.train_test_cycle(epochs=100)
print(f"AUROC (OVR): {auroc_ovr:.4f}")
```

---

## Architecture Types

Controls how the knowledge base integrates with GNN computation.

```
BARE: Independent computation (without knowledge base)
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Features │────▶│   GNN    │────▶│ Predict  │
└──────────┘     └──────────┘     └──────────┘

CCE (Chemical Concept Encoder): KB enhances input features
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Features │────▶│    KB    │────▶│   GNN    │────▶│ Predict  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘

CCD (Chemical Concept Decoder): KB processes GNN output
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ Features │────▶│   GNN    │────▶│    KB    │────▶│ Predict  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
```

| Type | Use Case |
|------|----------|
| `BARE` | Baseline; GNN and KB contribute independently |
| `CCE` | Feature enhancement; KB enriches input representations |
| `CCD` | Explainability; KB interprets GNN outputs |

---

## Checkpointing

Checkpoints save the trained model weights to two files: `{name}.safetensors` (weights) and `{name}.json` (architecture + metadata).

### `Checkpoint` class (low-level)

```python
from chemlogic.utils.Checkpoint import Checkpoint

# Save
safetensors_path, json_path = Checkpoint.save(
    model,
    filepath="checkpoints/run1",
    architecture={"model_name": "gnn", ...},  # optional
    training_state={"epoch": 100, "train_loss": 0.42},  # optional
    metadata={"description": "baseline"},      # optional
)

# Check existence
Checkpoint.exists("checkpoints/run1")  # True if both files present

# Load metadata
data = Checkpoint.load("checkpoints/run1")
# data keys: weights, weight_names, version, created_at, architecture, training_state, metadata

# Load weights into a pre-built Neuralogic model
Checkpoint.load_weights_into(model, "checkpoints/run1")
```

### Via Pipeline (high-level)

```python
# Save after training
pipeline.save_checkpoint()  # auto-path: checkpoints/{dataset_name}/{timestamp}
pipeline.save_checkpoint("checkpoints/my_run")  # explicit path

# Load
pipeline = Pipeline.from_checkpoint("checkpoints/my_run")
predictions = pipeline.inference(["CCO", "CC(=O)O"])

# Save during training (every 10 epochs)
pipeline.train_test_cycle(checkpoint_every=10, checkpoint_dir="checkpoints/")
```

---

## Configuration Reference

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | required | Dataset identifier |
| `model_name` | str | required | Model key (see Models) |
| `param_size` | int | required | Embedding dimension |
| `layers` | int | required | Number of GNN layers |
| `max_depth` | int | 1 | Propagation depth (SGN, DiffusionCNN, CWNet) |
| `max_subgraph_depth` | int | 5 | Maximum subgraph path length |
| `max_cycle_size` | int | 10 | Maximum cycle size |
| `subgraphs` | bool/tuple | None | Enable subgraph patterns |
| `chem_rules` | bool/tuple | None | Enable chemical rules |
| `architecture` | ArchitectureType | BARE | Integration strategy |
| `funnel` | bool | False | Fix KB weight size to 1 for interpretability |
| `smiles_list` | list[str] | None | SMILES strings (required with `labels`) |
| `labels` | list | None | Per-molecule labels (required with `smiles_list`) |
| `task` | str | "classification" | Task type (see Tasks and Metrics) |
| `atom_features` | str/list/None | None | RDKit atom features (`"all"` or list of names) |
| `bond_features` | str/list/None | None | RDKit bond features (`"all"` or list of names) |
| `graph_features` | dict/None | None | Per-molecule scalar features `{"name": [values]}` |
| `num_outputs` | int | 1 | Output dimension; set to number of classes for `multi_class` |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.001 | Learning rate |
| `epochs` | int | 100 | Maximum training epochs |
| `split_ratio` | float | 0.75 | Train/test split ratio |
| `optimizer` | Optimizer | Adam | NeuraLogic optimizer class |
| `error_function` | ErrorFunction | None | Defaults to CrossEntropy (classification) or MSE (regression) |
| `batches` | int | 1 | Number of dataset batches |
| `early_stopping_threshold` | float | 0.001 | Minimum improvement to reset patience |
| `early_stopping_rounds` | int | 10 | Epochs without improvement before stopping |
| `checkpoint_every` | int/None | None | Save checkpoint every N epochs |
| `checkpoint_dir` | str/None | None | Checkpoint directory (default: `checkpoints/{dataset_name}/`) |

### Tasks and Metrics

| Task | `task=` | `num_outputs` | Output Transform | Loss | Metric |
|------|---------|--------------|-----------------|------|--------|
| Binary classification | `"classification"` | 1 | Sigmoid | Cross-Entropy | AUROC |
| Multi-class classification | `"multi_class"` | N classes | Sigmoid + softmax | Cross-Entropy | AUROC (OVR, macro) |
| Regression | `"regression"` | 1 | Identity | MSE | R² |
| Multi-regression | `"multi_regression"` | N targets | Identity | MSE | R² (uniform average) |
