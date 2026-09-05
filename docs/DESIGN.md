# ChemLogic Design Overview

Architectural decisions and rationale. For introduction, see [README](../README.md). For API details, see [SPEC](SPEC.md).

---

## Design Goals

1. **Interpretability**: Model decisions trace back to functional groups and structural patterns
2. **Modularity**: Mix and match GNN architectures, chemical rules, and subgraph patterns
3. **Extensibility**: Add new components without modifying core code
4. **Accessibility**: Configure experiments using chemistry terminology

---

## Core Design Decisions

### 1. Relational Logic as the Foundation

**Decision**: Express all components (data, models, knowledge) in relational logic via PyNeuraLogic.

**Rationale**: 
- Molecular graphs map naturally to relational logic (atoms as entities, bonds as relations)
- Functional groups are inherently relational (e.g., "carbonyl is a carbon double-bonded to oxygen")
- Enables automatic differentiation through logical rules
- Provides built-in visualization of learned weights

**Trade-off**: Requires Java runtime; steeper learning curve for users unfamiliar with logic programming.

### 2. Three-Way Architecture Integration

**Decision**: Provide three architecture types (BARE, CCE, CCD) for knowledge base integration.

**Rationale**: Different research questions require different strategies:
- **BARE**: Establishes baselines; measures KB contribution independently
- **CCE**: Tests whether chemical priors improve learning
- **CCD**: Tests whether learned representations align with chemical concepts

### 3. Separation of Chemical Rules and Subgraph Patterns

**Decision**: Split knowledge base into two independent components.

**Rationale**: This separation enables:
- Using only chemical knowledge (interpretability studies)
- Using only structural patterns (architecture comparisons)
- Combining both (maximum expressiveness)
- Independent ablation studies

### 4. Funnel Mode for Quantitative Interpretability

**Decision**: Option to constrain all KB weights to dimension 1.

**Rationale**: With scalar weights, each functional group's contribution becomes a single interpretable number (e.g., "nitro contributes +0.3 to toxicity, hydroxyl contributes -0.1").

**Trade-off**: Reduced model capacity; use for interpretation, not maximum performance.

### 5. Dataset Abstraction

**Decision**: Datasets define their own atom/bond vocabularies and mappings.

**Rationale**: Different sources use different conventions (TUD has explicit hydrogens, SMILES may have implicit hydrogens). Encapsulating these details keeps the system data-source agnostic.

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            Pipeline                                  │
│  Orchestrates training, evaluation, and inference                   │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                      │
        ▼                    ▼                      ▼
┌──────────────┐    ┌──────────────┐    ┌─────────────────────────────┐
│   Dataset    │    │    Model     │    │      Knowledge Base         │
│              │    │              │    │                             │
│ - Load data  │    │ - GNN layers │    │ ┌─────────┐ ┌────────────┐ │
│ - Define     │    │ - Message    │    │ │Chemical │ │ Subgraph   │ │
│   vocabulary │    │   passing    │    │ │ Rules   │ │ Patterns   │ │
│ - Create     │    │ - Pooling    │    │ └─────────┘ └────────────┘ │
│   template   │    │              │    │                             │
└──────────────┘    └──────────────┘    └─────────────────────────────┘
        │                    │                      │
        └────────────────────┴──────────────────────┘
                             │
                             ▼
                    ┌──────────────┐
                    │ ChemTemplate │
                    │ Base class   │
                    │ for logical  │
                    │ rule sets    │
                    └──────────────┘
```

---

## Extension Points

### Adding a New Model

1. Create a class extending `Model`
2. Implement `build_layer()` to define message passing
3. Register in `MODEL_REGISTRY` (`models/models.py`)

### Adding a New Functional Group Category

1. Create a class extending `KnowledgeBase`
2. Define patterns in `create_template()` using relational predicates
3. Wire into `get_chem_rules()` (`knowledge_base/chemrules.py`)

### Adding Custom Knowledge-Base Features

Custom features are NeuraLogic templates passed through
`Pipeline(custom_rules=...)`. Their architecture-specific input/output adapters
are documented in [the API specification](SPEC.md#custom-knowledge-base-rules).

### Adding a New Subgraph Pattern

1. Create a class extending `KnowledgeBase`
2. Define structural patterns in `create_template()`
3. Wire into `get_subgraphs()` (`knowledge_base/subgraphs.py`)

### Adding a New Dataset

1. For standard formats: extend `Dataset`, implement `load_data()`
2. For SMILES data: use `SmilesDataset` directly
3. Register in `DATASET_CLASSES` if reusable (`datasets/datasets.py`)

---

## Limitations

- **Implicit hydrogens**: Some functional group patterns assume explicit hydrogens
- **Single-task only**: No multi-task or transfer learning support

---

## Future Directions

- Generation tasks (molecular design)
- Multi-task learning
- Attention visualization over functional groups
