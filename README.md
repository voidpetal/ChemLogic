# ChemLogic

ChemLogic is a neurosymbolic framework that integrates relational logic syntax with graph neural networks (GNNs) to model chemical knowledge. It is designed for interpretable molecular property prediction, combining symbolic reasoning with differentiable learning. ChemLogic was entirely built on the [PyNeuraLogic](https://github.com/LukasZahradnik/PyNeuraLogic) framework.

## 🧬 Introduction

ChemLogic enables binary classification for molecular property prediction tasks on chemistry datasets, such as mutagenicity and toxicity prediction. It supports explainable AI by encoding functional groups and molecular subgraph patterns into logical rules, which are then integrated into GNN architectures. The weights of these rules provide interpretable insights into the model's reasoning process.

## ✨ Features

- Supports well-known GNN architectures from the literature.
- Encodes chemical knowledge using relational logic syntax.
- Integrates functional groups and molecular subgraph patterns into a learnable knowledge base.
- Enables explainable and interpretable predictions.
- Designed for binary classification tasks with future support for regression and more.

## 📦 Installation

ChemLogic is available via PyPI. You can install it using:

```bash
pip install ChemLogic
```

## 📂 Project structure

The project consists off of 3 main modules:

- `datasets` - contain the datasets encoded in relational manner. Includes data from `TUD` and `TDC` datasets, as well as a converter from custom SMILES datasets.
- `models` - contains the GNN architectures.
- `knowledge_base` - contains the functional groups and subgraph patters.

## 🚀 Usage

Basic example of training a GNN on the MUTAG dataset can be found in `notebooks/run_example`.

## 🧩 Dependencies

ChemLogic requires Python 3.11 and Java >=1.8. For visualization `graphviz` is required.

All dependencies are listed in `pyproject.toml`.

## 📖 Documentation

- [SPEC.md](docs/SPEC.md) - API reference and configuration details
- [DESIGN.md](docs/DESIGN.md) - Architecture and design rationale
- [REQUIREMENTS.md](docs/REQUIREMENTS.md) - Functional and non-functional requirements

## 🤝 Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines on how to get started.

## 📄 License

This project is licensed under the MIT License.
