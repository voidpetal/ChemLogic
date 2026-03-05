# ChemLogic Specification

Technical reference for APIs and configuration. For introduction and installation, see [README](../README.md).

---

## Table of Contents

1. [Datasets](#datasets)
2. [Models](#models)
3. [Knowledge Base](#knowledge-base)
4. [Pipeline](#pipeline)
5. [Architecture Types](#architecture-types)
6. [Configuration Reference](#configuration-reference)

---

## Datasets

### Supported Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| `mutagen` | TUD | 183 | Mutagenicity prediction |
| `ptc`, `ptc_fr`, `ptc_mm`, `ptc_fm` | TUD | 336-351 | Toxicity prediction (various species) |
| `dhfr` | TUD | 393 | DHFR inhibition |
| `er` | TUD | 446 | Estrogen receptor binding |
| `blood_brain_barrier` | TDC | 2030 | Blood-brain barrier penetration |
| `skin_reaction` | TDC | 404 | Skin sensitization |
| `oral_bioavailability` | TDC | 640 | Oral bioavailability |
| `carcinogenous` | TDC | 280 | Carcinogenicity |
| `pampa_permeability` | TDC | 2034 | Membrane permeability |
| `human_intestinal_absorption` | TDC | 578 | Intestinal absorption |
| `p_glycoprotein_inhibition` | TDC | 1218 | P-gp inhibition |
| `cyp2c9_substrate`, `cyp2d6_substrate`, `cyp3a4_substrate` | TDC | 667-670 | CYP enzyme substrates |
| `anti_sarscov2_activity` | TDC | 1484 | SARS-CoV-2 activity |

### Custom Datasets via SMILES

```python
from chemlogic.utils.Pipeline import Pipeline

pipeline = Pipeline(
    dataset_name="my_dataset",
    model_name="gnn",
    param_size=2,
    layers=2,
    smiles_list=["CCO", "CC(=O)O", "c1ccccc1"],
    labels=[0, 1, 0]
)
```

### Dataset Structure

Each dataset defines:
- **Atom types**: Predicates for elements (e.g., `c`, `o`, `n`, `s`, `h`)
- **Bond types**: Predicates for bond orders (single, double, triple, aromatic)
- **Connectivity**: `connection(X, Y, B)` predicate linking atoms X and Y via bond B

---

## Models

### Available Models

| Model | Key | Description |
|-------|-----|-------------|
| GNN | `gnn` | Standard graph neural network with edge features |
| RGCN | `rgcn` | Relational GCN with typed edges |
| KGNN | `kgnn` | Knowledge graph neural network (`kgnn_local` for local variant) |
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
    labels: list[int] = None,
    task: str = "classification"
)
```

### Training

```python
train_loss, test_loss, metric, evaluator = pipeline.train_test_cycle(
    lr: float = 0.001,
    epochs: int = 100,
    split_ratio: float = 0.75,
    batches: int = 1,
    early_stopping_threshold: float = 0.001,
    early_stopping_rounds: int = 10
)
```

### Inference

```python
predictions = pipeline.inference(smiles_list=["CCO", "CC(=O)O"])
```

### Visualization

```python
pipeline.template.draw()  # Requires graphviz
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

## Configuration Reference

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | required | Dataset identifier |
| `model_name` | str | required | Model key |
| `param_size` | int | required | Embedding dimension |
| `layers` | int | required | Number of GNN layers |
| `max_depth` | int | 1 | Propagation depth |
| `max_subgraph_depth` | int | 5 | Maximum subgraph path length |
| `max_cycle_size` | int | 10 | Maximum cycle size |
| `subgraphs` | bool/tuple | None | Enable subgraph patterns |
| `chem_rules` | bool/tuple | None | Enable chemical rules |
| `architecture` | ArchitectureType | BARE | Integration strategy |
| `funnel` | bool | False | Fix weight size to 1 for interpretability |
| `task` | str | "classification" | Task type ("classification" or "regression") |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr` | float | 0.001 | Learning rate |
| `epochs` | int | 100 | Maximum training epochs |
| `split_ratio` | float | 0.75 | Train/test split ratio |
| `batches` | int | 1 | Number of batches |
| `early_stopping_threshold` | float | 0.001 | Minimum improvement |
| `early_stopping_rounds` | int | 10 | Patience before stopping |

### Tasks and Metrics

| Task | Output Transformation | Loss Function | Evaluation Metric |
|------|----------------------|---------------|-------------------|
| Classification | Sigmoid | Cross-Entropy | AUROC |
| Regression | Identity | MSE | R² |
