# ChemLogic Requirements

Formal requirements specification. For introduction, see [README](../README.md). For API details, see [SPEC](SPEC.md).

---

## Functional Requirements

### FR1: Molecular Property Prediction

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1.1 | Predict binary molecular properties | Must |
| FR1.2 | Predict continuous molecular properties (regression) | Must |
| FR1.3 | Predict multi-class molecular properties | Must |
| FR1.4 | Predict multiple outputs simultaneously (multi-regression) | Must |
| FR1.5 | Accept SMILES strings as input | Must |
| FR1.6 | Auto-detect task type from label structure | Should |
| FR1.7 | Support batch prediction | Should |

### FR2: Model Configuration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR2.1 | Support multiple GNN architectures | Must |
| FR2.2 | Configurable message-passing layers | Must |
| FR2.3 | Configurable embedding dimensions | Must |
| FR2.4 | Support architecture-specific parameters | Must |

### FR3: Knowledge Base Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR3.1 | Encode functional groups as learnable logical rules | Must |
| FR3.2 | Selective enabling of functional group categories | Must |
| FR3.3 | Encode subgraph patterns as learnable rules | Must |
| FR3.4 | Selective enabling of subgraph pattern types | Must |
| FR3.5 | Three integration modes (BARE, CCE, CCD) | Must |

### FR4: Training

| ID | Requirement | Priority |
|----|-------------|----------|
| FR4.1 | Configurable train/test split | Must |
| FR4.2 | Configurable learning rate and epochs | Must |
| FR4.3 | Early stopping | Must |
| FR4.4 | Report loss and evaluation metrics | Must |
| FR4.5 | AUROC for classification/multi-class, R² for regression/multi-regression | Must |

### FR5: Datasets

| ID | Requirement | Priority |
|----|-------------|----------|
| FR5.1 | Support TUD benchmark datasets | Must |
| FR5.2 | Support TDC ADMET datasets | Must |
| FR5.3 | Accept custom SMILES datasets | Must |
| FR5.4 | Automatic SMILES to relational conversion | Must |

### FR6: Interpretability

| ID | Requirement | Priority |
|----|-------------|----------|
| FR6.1 | Visualize learned templates with weights | Must |
| FR6.2 | Funnel mode for scalar weights | Should |
| FR6.3 | Weights traceable to chemical concepts | Should |

### FR7: Inference

| ID | Requirement | Priority |
|----|-------------|----------|
| FR7.1 | Inference on new SMILES after training | Must |
| FR7.2 | Return prediction scores | Must |

### FR8: Extended Features

| ID | Requirement | Priority |
|----|-------------|----------|
| FR8.1 | Atom-level RDKit features as node predicates | Must |
| FR8.2 | Bond-level RDKit features as edge predicates | Must |
| FR8.3 | Graph-level scalar features via synthetic node or broadcast | Must |

### FR9: Checkpointing

| ID | Requirement | Priority |
|----|-------------|----------|
| FR9.1 | Save trained model weights to disk | Must |
| FR9.2 | Load weights and resume inference without retraining | Must |
| FR9.3 | Optionally save checkpoint during training at fixed intervals | Should |

---

## Non-Functional Requirements

### NFR1: Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR1.1 | MUTAG training completes within 5 minutes | Target |
| NFR1.2 | Batched dataset building for large datasets | Must |
| NFR1.3 | Inference under 1 second per molecule | Target |

### NFR2: Usability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR2.1 | Basic pipeline in <10 lines of code | Must |
| NFR2.2 | Sensible parameter defaults | Must |
| NFR2.3 | Auto-detect task and num_outputs from label structure | Should |
| NFR2.4 | Clear error messages | Should |

### NFR3: Compatibility

| ID | Requirement | Target |
|----|-------------|--------|
| NFR3.1 | Python 3.11+ | Must |
| NFR3.2 | Any OS with Java 1.8+ | Must |
| NFR3.3 | Installable via pip | Must |

### NFR4: Extensibility

| ID | Requirement | Target |
|----|-------------|--------|
| NFR4.1 | New models via `Model` extension | Must |
| NFR4.2 | New functional groups via `KnowledgeBase` extension | Must |
| NFR4.3 | New datasets via `Dataset` extension | Must |

### NFR5: Reliability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR5.1 | Input validation with clear errors | Must |
| NFR5.2 | Graceful handling of invalid SMILES | Should |
| NFR5.3 | Reproducible training with fixed seed | Must |

---

## Constraints

| ID | Constraint |
|----|------------|
| C1 | Built on PyNeuraLogic—bound by its capabilities |
| C2 | Requires Java runtime |
| C3 | Some patterns require explicit hydrogens |

