<meta name="robots" content="noindex">

# Attributed Hypergraph Clustering

This repository contains the implementation of the proposed attributed hypergraph clustering framework. The method jointly learns node representations, clustering assignments, and clustering-adaptive incidence contributions under dual-view normalized-cut optimization. It uses a fixed structure-attribute prior graph as complementary guidance and refines node-hyperedge memberships in the original incidence space.

![Framework](framework_01.png)

Code is available under the root folder.
The dataset archive is provided as `dataset.zip`.

## Requirements

Python 3.10.8  
PyTorch 2.1.2+cu121  
NumPy 1.26.3  
SciPy 1.15.3  
Scikit-learn 1.7.2  
PyYAML 6.0.1  
CUDA 12.1  

For ease of setup, install all required dependencies with the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Dataset Path

Before running the code, unzip `dataset.zip` under this repository:

```bash
unzip dataset.zip
```

By default, datasets are then loaded from `dataset/`. To use another data directory, set:

```bash
export AHC_DATA_ROOT=/path/to/dataset
```

Each dataset directory should contain the corresponding hypergraph, feature, and label files required by `data.py`.

## Usage

Run one dataset with a single random seed:

```bash
python main.py --dataset cocitation/cora --config config.yaml --seed 42
```

Run one dataset with multiple random seeds:

```bash
python main.py --dataset cocitation/cora --config config.yaml --seeds 42,43,44,45,46
```

The dataset name must match one of the entries in `config.yaml`.

## Supported Datasets

| Type of dataset | `--dataset` argument |
| :--- | :--- |
| Co-citation hypergraphs | `cocitation/cora`, `cocitation/citeseer`, `cocitation/pubmed` |
| UCI datasets | `zoo`, `20newsW100`, `Mushroom` |
| Vision and graphics datasets | `NTU2012`, `ModelNet40` |

## Configuration File

Dataset-specific hyperparameters are stored in `config.yaml`. The script automatically loads the configuration corresponding to the selected dataset.

Key parameters include:

| Parameter | Description |
| :--- | :--- |
| `hidden` | Hidden dimension of the HGNN encoder |
| `dropout` | Dropout rate |
| `lr` | Learning rate |
| `epochs` | Number of training epochs |
| `T_attr` | Propagation depth for the structure-attribute prior graph |
| `alpha_attr` | Restart probability for structure-attribute propagation |
| `beta_attr` | Balance factor between incidence-based and hyperedge-semantic diffusion |
| `eta` | Fusion weight between base and learned incidence-induced affinities |
| `gamma_attr` | Weight of the structure-attribute prior Ncut loss |
| `gamma_bal` | Weight of the balance regularization |
| `gamma_rec` | Weight of the reconstruction loss |
| `gamma_kl` | Weight of the KL sharpening loss |

## Reproducing Single-Seed Results

Use the following commands to run the eight datasets with seed 42:

```bash
python main.py --dataset cocitation/cora --config config.yaml --seed 42
python main.py --dataset cocitation/citeseer --config config.yaml --seed 42
python main.py --dataset cocitation/pubmed --config config.yaml --seed 42
python main.py --dataset zoo --config config.yaml --seed 42
python main.py --dataset 20newsW100 --config config.yaml --seed 42
python main.py --dataset Mushroom --config config.yaml --seed 42
python main.py --dataset NTU2012 --config config.yaml --seed 42
python main.py --dataset ModelNet40 --config config.yaml --seed 42
```

Sample output:

```text
Dataset: cocitation/cora | ACC: 0.xxxx | NMI: 0.xxxx | F1: 0.xxxx | ARI: 0.xxxx
Effective Time: xx.xxs
```
