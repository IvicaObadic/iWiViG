# i-WiViG: Interpretable Window Vision GNN

This repository contains the implementation of the **i-WiViG** model.

---

## 🛠️ Requirements

The project relies on the following key libraries:

* **PyTorch**
* **PyTorch Geometric**
* **PyTorch Lightning**
* **wandb** (Weights & Biases)

---

## 🚀 Usage

### Training Models

To train the **i-WiViG** model, as well as the other benchmark models, use the `gnn_model_train.py` script.

The specific models and datasets are selected by providing input parameters to the primary function:

```bash
python gnn_model_train.py 
```

### Generating Explanations

For generating **qualitative explanations** (e.g., visual subgraphs) and performing the **quantitative explanation evaluation**, call the `edge_attribution.py` script:

```bash
python edge_attribution.py