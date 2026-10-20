# Incremental Learning with Neural Networks

Comparison of online / incremental learning models under a **class-incremental** setup on the **MNIST** and **CIFAR-10** datasets.

Four models are evaluated across three phases:

- **FuzzyArtMap** — Fuzzy ARTMAP (adaptive resonance theory classifier)
- **ePNN** — enhanced Probabilistic Neural Network
- **MLP** — Multi-Layer Perceptron (baseline)
- **MiniBatchKMeans** — clustering baseline

### Experimental protocol

The dataset is split into two disjoint groups of classes:

| Phase | What happens |
|-------|--------------|
| **Phase 1** | Train on the first half of the data for classes **0–4** |
| **Phase 2** | Incrementally train on the second half of the data for classes **0–4** |
| **Phase 3** | Incrementally introduce the **new** classes **5–9** and re-evaluate on both the known (0–4) and new (5–9) classes |

For CIFAR-10, raw 32×32 pixels are replaced by **ResNet-50 penultimate-layer embeddings** (ImageNet-pretrained, frozen backbone). For MNIST, features are reduced with **PCA** (40 components).

## Project files

| File | Description |
|------|-------------|
| [mnist_art.ipynb](mnist_art.ipynb) | Full experiment on the MNIST dataset |
| [cifar-10.ipynb](cifar-10.ipynb) | Full experiment on the CIFAR-10 dataset (ResNet-50 embeddings) |
| [requirements.txt](requirements.txt) | Python dependencies |

## Requirements

- Python 3.9+
- The packages listed in [requirements.txt](requirements.txt):
  `scikit-learn`, `numpy`, `matplotlib`, `pandas`, `tensorflow`, `torch`, `torchvision`
- `seaborn` (used for plotting inside the notebooks)

> **Note:** A GPU is recommended for the CIFAR-10 notebook — computing the ResNet-50 embeddings is the heaviest step. The notebooks run on CPU too, just slower.

## How to run

### 1. Clone the repository

```bash
git clone <repository-url>
cd tcc
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
pip install seaborn            # required by the notebooks for plotting
```

### 4. Launch Jupyter and open a notebook

```bash
pip install jupyter            # if not already installed
jupyter notebook
```

Then open either [mnist_art.ipynb](mnist_art.ipynb) or [cifar-10.ipynb](cifar-10.ipynb) and run the cells top to bottom (**Kernel → Restart & Run All**).

Both datasets are downloaded automatically on first run (MNIST via `sklearn.datasets.fetch_openml`, CIFAR-10 via `tensorflow.keras.datasets`), so no manual data download is needed.

## Results

Metrics are reported as **mean ± standard deviation across 10 random seeds**.

### MNIST

| Phase | Group | Model | Accuracy | Precision | Recall | F1-score |
|-------|-------|-------|----------|-----------|--------|----------|
| Phase 1 | Digits 0–4 | FuzzyArtMap | 81.67% ± 2.59 | 83.40% ± 1.70 | 81.27% ± 2.64 | 81.32% ± 2.58 |
| Phase 1 | Digits 0–4 | MLP | 98.68% ± 0.15 | 98.67% ± 0.16 | 98.67% ± 0.15 | 98.67% ± 0.15 |
| Phase 1 | Digits 0–4 | MiniBatchKMeans | 96.55% ± 0.14 | 96.56% ± 0.15 | 96.50% ± 0.14 | 96.52% ± 0.14 |
| Phase 1 | Digits 0–4 | ePNN | 89.11% ± 4.10 | 90.97% ± 2.47 | 89.68% ± 3.71 | 89.02% ± 4.47 |
| Phase 2 | Digits 0–4 | FuzzyArtMap | 82.48% ± 1.80 | 83.49% ± 1.15 | 82.13% ± 1.87 | 82.13% ± 1.72 |
| Phase 2 | Digits 0–4 | MLP | 99.04% ± 0.12 | 99.04% ± 0.12 | 99.04% ± 0.12 | 99.04% ± 0.12 |
| Phase 2 | Digits 0–4 | MiniBatchKMeans | 96.57% ± 0.14 | 96.59% ± 0.14 | 96.52% ± 0.14 | 96.55% ± 0.14 |
| Phase 2 | Digits 0–4 | ePNN | 89.53% ± 5.68 | 91.52% ± 3.22 | 90.07% ± 5.14 | 89.24% ± 6.33 |
| Phase 3 | Known (0–4) | FuzzyArtMap | 71.30% ± 2.89 | 89.87% ± 1.54 | 70.71% ± 2.97 | 78.34% ± 1.79 |
| Phase 3 | Known (0–4) | MLP | 2.97% ± 1.13 | 97.94% ± 6.31 | 2.95% ± 1.07 | 5.58% ± 1.95 |
| Phase 3 | Known (0–4) | MiniBatchKMeans | 90.55% ± 0.30 | 97.53% ± 0.17 | 90.36% ± 0.32 | 93.72% ± 0.21 |
| Phase 3 | Known (0–4) | ePNN | 75.58% ± 4.63 | 94.75% ± 3.30 | 75.82% ± 4.20 | 83.01% ± 5.25 |
| Phase 3 | New (5–9) | FuzzyArtMap | 70.98% ± 1.80 | 82.72% ± 1.17 | 70.80% ± 1.79 | 76.10% ± 1.22 |
| Phase 3 | New (5–9) | MLP | 98.15% ± 0.13 | 98.13% ± 0.13 | 98.14% ± 0.13 | 98.14% ± 0.13 |
| Phase 3 | New (5–9) | MiniBatchKMeans | 89.60% ± 0.21 | 94.77% ± 0.18 | 89.55% ± 0.21 | 92.06% ± 0.17 |
| Phase 3 | New (5–9) | ePNN | 84.94% ± 1.77 | 90.13% ± 0.50 | 85.12% ± 1.77 | 86.82% ± 1.36 |

### CIFAR-10

| Phase | Group | Model | Accuracy | Precision | Recall | F1-score |
|-------|-------|-------|----------|-----------|--------|----------|
| Phase 1 | Classes 0–4 | FuzzyArtMap | 64.89% ± 3.65 | 72.38% ± 4.00 | 64.89% ± 3.65 | 64.47% ± 4.22 |
| Phase 1 | Classes 0–4 | MLP | 94.26% ± 0.33 | 94.26% ± 0.33 | 94.26% ± 0.33 | 94.26% ± 0.33 |
| Phase 1 | Classes 0–4 | MiniBatchKMeans | 87.07% ± 0.46 | 87.11% ± 0.45 | 87.07% ± 0.46 | 87.06% ± 0.45 |
| Phase 1 | Classes 0–4 | ePNN | 88.31% ± 0.67 | 88.84% ± 0.53 | 88.31% ± 0.67 | 88.42% ± 0.64 |
| Phase 2 | Classes 0–4 | FuzzyArtMap | 69.83% ± 4.10 | 74.85% ± 1.55 | 69.83% ± 4.10 | 69.69% ± 3.95 |
| Phase 2 | Classes 0–4 | MLP | 94.29% ± 0.26 | 94.30% ± 0.25 | 94.29% ± 0.26 | 94.29% ± 0.26 |
| Phase 2 | Classes 0–4 | MiniBatchKMeans | 87.07% ± 0.49 | 87.10% ± 0.48 | 87.07% ± 0.49 | 87.05% ± 0.48 |
| Phase 2 | Classes 0–4 | ePNN | 88.42% ± 0.75 | 88.90% ± 0.64 | 88.42% ± 0.75 | 88.51% ± 0.74 |
| Phase 3 | Known (0–4) | FuzzyArtMap | 57.94% ± 3.72 | 80.45% ± 2.92 | 57.94% ± 3.72 | 65.89% ± 2.80 |
| Phase 3 | Known (0–4) | MLP | 12.00% ± 1.03 | 99.80% ± 0.15 | 12.00% ± 1.03 | 20.68% ± 1.57 |
| Phase 3 | Known (0–4) | MiniBatchKMeans | 74.63% ± 0.60 | 89.76% ± 0.66 | 74.63% ± 0.60 | 81.45% ± 0.53 |
| Phase 3 | Known (0–4) | ePNN | 68.85% ± 1.65 | 92.10% ± 0.92 | 68.85% ± 1.65 | 78.21% ± 1.04 |
| Phase 3 | New (5–9) | FuzzyArtMap | 69.63% ± 4.07 | 89.00% ± 1.84 | 69.63% ± 4.07 | 77.26% ± 2.99 |
| Phase 3 | New (5–9) | MLP | 96.14% ± 0.28 | 96.15% ± 0.27 | 96.14% ± 0.28 | 96.14% ± 0.28 |
| Phase 3 | New (5–9) | MiniBatchKMeans | 86.11% ± 0.74 | 95.61% ± 0.41 | 86.11% ± 0.74 | 90.60% ± 0.51 |
| Phase 3 | New (5–9) | ePNN | 88.36% ± 0.73 | 94.04% ± 0.38 | 88.36% ± 0.73 | 91.03% ± 0.49 |

> The sharp drop in MLP accuracy at Phase 3 on the known classes (0–4) is the classic **catastrophic forgetting** effect: after being trained on the new classes, the plain MLP overwrites what it had learned, whereas the incremental models (FuzzyArtMap, ePNN) and the clustering baseline retain far more of the earlier knowledge.
