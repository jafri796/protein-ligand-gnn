# Protein-Ligand Binding Affinity Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.0](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A state-of-the-art deep learning system for predicting protein-ligand binding affinities using geometric deep learning and SE(3)-equivariant neural networks.

## 🎯 Overview

This system implements cutting-edge geometric deep learning techniques for accurate binding affinity prediction, built on:

- **PaiNN-inspired SE(3)-equivariant message passing** for 3D molecular geometry
- **Heterogeneous interaction graphs (IGN/GIGN)** for protein-ligand interactions
- **Leak-proof evaluation (LP-PDBBind)** for rigorous benchmarking
- **Production-quality engineering** with comprehensive testing and documentation

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Scientifically Rigorous** | Every design choice justified by peer-reviewed research |
| 🏭 **Production Quality** | Type hints, logging, error handling, comprehensive tests |
| ⚡ **Fully Functional** | Complete end-to-end pipeline from data to predictions |
| 🧪 **Research Extensible** | Modular architecture for easy experimentation |
| 🚀 **GPU Accelerated** | Efficient PyTorch + PyTorch Geometric implementation |

## 📁 Project Structure

```
affinity_system/
├── config/
│   └── painn_config.yaml          # Model hyperparameters and training config
├── data/
│   ├── featurization.py            # RDKit/BioPython molecular features (472 lines)
│   ├── graph_construction.py       # PyG heterogeneous graph building (385 lines)
│   └── dataset.py                  # PyTorch Dataset with caching (251 lines)
├── models/
│   ├── layers/
│   │   └── equivariant_layers.py   # PaiNN equivariant layers (378 lines)
│   └── painn_affinity.py           # Main prediction model (287 lines)
├── experiments/
│   └── train_painn.py              # Training script with metrics (309 lines)
└── README.md                       # This file
```

## 🚀 Quick Start

### Installation

#### 1. Clone Repository
```bash
git clone <repository-url>
cd affinity_system
```

#### 2. Create Conda Environment
```bash
conda create -n affinity python=3.9
conda activate affinity
```

#### 3. Install PyTorch (with CUDA 11.8)
```bash
conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 4. Install PyTorch Geometric
```bash
pip install torch-geometric==2.4.0
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

#### 5. Install Additional Dependencies
```bash
pip install rdkit biopython pandas numpy scipy scikit-learn pyyaml tqdm tensorboard
```

### Data Preparation

1. **Download PDBBind Dataset** (manual download required)
   - Visit [PDBBind Database](http://www.pdbbind.org.cn/)
   - Download the refined set or general set

2. **Extract Dataset**
   ```bash
   # Extract to data/pdbbind/
   tar -xzf PDBbind_v2020.tar.gz -C data/
   ```

3. **Prepare Data Splits**
   Create `train.txt`, `val.txt`, `test.txt` files with format:
   ```
   pdb_id affinity
   1a1e 7.52
   1a30 5.89
   1abc 6.12
   ```

### Training

```bash
# Train model with default configuration
python experiments/train_painn.py --config config/painn_config.yaml --gpu 0

# Monitor training progress with TensorBoard
tensorboard --logdir outputs/logs
```

### Inference

```python
import torch
from models.painn_affinity import PaiNNAffinityPredictor
from data.featurization import featurize_complex
from data.graph_construction import (
    construct_complex_graph, 
    construct_ligand_graph, 
    construct_protein_graph
)

# Load trained model
model = PaiNNAffinityPredictor.from_config('config/painn_config.yaml')
checkpoint = torch.load('outputs/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare protein-ligand complex
complex_data = featurize_complex('protein.pdb', 'ligand.sdf')
ligand_graph = construct_ligand_graph(complex_data['ligand'])
protein_graph = construct_protein_graph(complex_data['protein'])
graph = construct_complex_graph(ligand_graph, protein_graph)

# Predict binding affinity
with torch.no_grad():
    affinity = model(graph)
    print(f"Predicted pKd: {affinity.item():.2f}")
```

## 🧩 Module Descriptions

### 1. `data/featurization.py` (472 lines)

**Molecular Feature Extraction**

- **Ligand Features**: RDKit-based atom features (49-dim) and bond features (10-dim)
  - Atom type, degree, formal charge, hybridization, aromaticity, etc.
  - Bond type, conjugation, ring membership, stereo configuration
  
- **Protein Features**: BioPython residue features (31-dim)
  - Amino acid type, secondary structure (alpha-helix, beta-sheet, coil)
  - Backbone angles, accessibility, charge properties
  
- **3D Geometry**: Atomic coordinates, pairwise distances, directional vectors

- **Binding Pocket Extraction**: 10Å radius cutoff from ligand center

**Scientific Justification**: Features based on GraphDTA, IGN, and molecular ML best practices from recent literature.

### 2. `data/graph_construction.py` (385 lines)

**Heterogeneous Graph Construction**

- **Ligand Graphs**: Covalent bond connectivity + self-loops
- **Protein Graphs**: k-NN (k=8) or radius-based (10Å) connectivity
- **Interaction Edges**: Distance-based protein-ligand interactions (<6Å)
- **RBF Expansion**: Gaussian radial basis functions for distance encoding

**Scientific Justification**: Follows PyTorch Geometric conventions and IGN/GIGN heterogeneous graph design principles.

### 3. `data/dataset.py` (251 lines)

**Efficient Data Loading**

- **Lazy Loading**: On-the-fly graph processing
- **Disk Caching**: Saves processed graphs to avoid recomputation
- **Validation**: Checks file existence and data integrity
- **Memory Efficient**: Loads one complex at a time

**Engineering**: Production-quality with comprehensive error handling and logging.

### 4. `models/layers/equivariant_layers.py` (378 lines)

**SE(3)-Equivariant Layers**

- **RBF Expansion Layer**: Gaussian basis function distance encoding
- **PaiNN Message Layer**: SE(3)-equivariant message passing with scalar and vector features
- **PaiNN Update Layer**: Gated updates preserving equivariance
- **Interaction Layer**: Cross-molecular attention mechanism

**Scientific Justification**: Direct implementation of Schütt et al. (2021) PaiNN architecture.

### 5. `models/painn_affinity.py` (287 lines)

**Main Prediction Model**

- **Ligand Encoder**: 5-layer PaiNN (scalar + vector features)
- **Protein Encoder**: 3-layer Graph Attention Network (GAT)
- **Interaction Module**: Cross-attention between protein and ligand representations
- **Readout Head**: Global pooling + 4-layer MLP for affinity prediction

**Architecture**: Synergizes PaiNN equivariance with IGN interaction modeling for superior performance.

### 6. `experiments/train_painn.py` (309 lines)

**Training Infrastructure**

- **Training Loop**: Epoch-based with progress bars and detailed logging
- **Metrics**: RMSE, MAE, Pearson correlation, Spearman correlation
- **Checkpointing**: Saves best model + periodic checkpoints
- **Early Stopping**: Patience-based termination to prevent overfitting
- **Logging**: TensorBoard integration for visualization

**Engineering**: Complete production training infrastructure with best practices.

## ⚙️ Configuration

Edit `config/painn_config.yaml` to customize hyperparameters:

```yaml
model:
  hidden_dim: 128                    # Feature dimension
  num_message_passing_layers: 5      # PaiNN depth
  num_rbf: 20                        # RBF basis functions
  cutoff: 10.0                       # Interaction cutoff (Angstroms)
  
training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 200
  early_stopping_patience: 20
  weight_decay: 1e-5
  
data:
  train_split: data/splits/train.txt
  val_split: data/splits/val.txt
  test_split: data/splits/test.txt
  cache_dir: data/processed/
```

## 🧪 Testing & Validation

All modules include standalone test code for verification:

```bash
# Test molecular featurization
python data/featurization.py

# Test graph construction
python data/graph_construction.py

# Test equivariant layers
python models/layers/equivariant_layers.py

# Test complete model
python models/painn_affinity.py
```

## 📊 Performance Expectations

Based on literature and architectural design:

| Metric | Expected Performance | Source |
|--------|---------------------|--------|
| **RMSE** | 1.1 - 1.3 | PaiNN + interaction modeling |
| **Pearson R** | 0.76 - 0.80 | IGN/GIGN benchmarks |
| **MAE** | 0.9 - 1.1 | Comparable models |
| **Parameters** | ~2-3M | Efficient architecture |
| **Training Time** | 4-6 hours (V100) | Per 200 epochs |

## 📚 Scientific Foundation

This implementation synthesizes research from:

1. **PaiNN** (Schütt et al., 2021) - Equivariant message passing for 3D molecular modeling
2. **IGN** (Zhang et al., 2022) - Interaction Graph Networks for binding prediction
3. **GIGN** (Yang et al., 2023) - Geometric Interaction Graph Networks with 3D structure
4. **LP-PDBBind** (Li et al., 2023) - Leak-proof evaluation protocols
5. **PyTorch Geometric** - Best practices for heterogeneous graphs

### Key Papers

- Schütt et al. (2021) "Equivariant message passing for the prediction of tensorial properties and molecular spectra" - *ICML*
- Zhang et al. (2022) "InteractionGraphNet: A Novel and Efficient Deep Graph Representation Learning Framework for Accurate Protein-Ligand Interaction Predictions" - *Journal of Chemical Information and Modeling*
- Yang et al. (2023) "Geometric Interaction Graph Neural Network for Predicting Protein-Ligand Binding Affinities"
- Li et al. (2023) "Leak Proof PDBBind: A Reorganized Dataset for More Accurate Binding Affinity Prediction" - *arXiv*

## 🔬 Extensions & Future Work

This codebase serves as a foundation for:

- **Active Learning**: Uncertainty-guided experimental design for efficient drug discovery
- **Multi-Task Learning**: Joint prediction of affinity + selectivity + ADMET properties
- **Transfer Learning**: Pre-training on QM9/MD17 for improved generalization
- **Interpretability**: Attention visualization and binding mode analysis
- **Generative Models**: Structure-based ligand design with target affinity optimization
- **Virtual Screening**: High-throughput screening of compound libraries
- **Ensemble Methods**: Multiple model combinations for improved accuracy

## 🛠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config
batch_size: 16  # or 8

# Or use gradient accumulation
gradient_accumulation_steps: 2
```

**RDKit Import Error**
```bash
# Reinstall via conda (not pip)
conda install -c conda-forge rdkit
```

**PyG Installation Issues**
```bash
# Ensure CUDA version matches
python -c "import torch; print(torch.version.cuda)"

# Install matching PyG version
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions, issues, or collaborations:

- Open an issue on GitHub
- Email: [your-email@domain.com]
- Twitter: [@your-handle]

## 🙏 Acknowledgments

- PDBBind database for curated binding affinity data
- PyTorch Geometric team for excellent graph neural network library
- RDKit and BioPython communities for molecular processing tools
- Research groups who published the foundational papers

---

**Built with ❤️ for accelerating drug discovery through geometric deep learning**
