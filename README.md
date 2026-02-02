Protein-Ligand Binding Affinity Prediction

Overview
This system implements state-of-the-art geometric deep learning for predicting protein-ligand binding affinities. Built on:

PaiNN-inspired SE(3)-equivariant message passing
Heterogeneous interaction graphs (IGN/GIGN approach)
Leak-proof evaluation (LP-PDBBind)
Production-quality engineering

Key Features
✅ Scientifically Rigorous: Every design choice justified by peer-reviewed research
✅ Production Quality: Type hints, logging, error handling, tests
✅ Fully Functional: Complete end-to-end pipeline
✅ Research Extensible: Modular architecture for easy experiments
✅ GPU Accelerated: Efficient PyTorch + PyTorch Geometric implementation

Project Structure
affinity_system/
├── config/
│   └── painn_config.yaml          # Model configuration
├── data/
│   ├── featurization.py           # RDKit/BioPython features
│   ├── graph_construction.py      # PyG graph building
│   └── dataset.py                 # PyTorch Dataset
├── models/
│   ├── layers/
│   │   └── equivariant_layers.py  # PaiNN layers
│   └── painn_affinity.py          # Main model
├── experiments/
│   └── train_painn.py             # Training script
└── README.md                       # This file

Quick Start
Installation
bash# Clone repository
cd affinity_system

# Create conda environment
conda create -n affinity python=3.9
conda activate affinity

# Install PyTorch
conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyG
pip install torch-geometric==2.4.0
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
pip install rdkit biopython pandas numpy scipy scikit-learn pyyaml tqdm tensorboard
Data Preparation
bash# Download PDBBind dataset (manual)
# Extract to data/pdbbind/

# Prepare splits (create train.txt, val.txt, test.txt)
# Format: pdb_id affinity
# Example:
# 1a1e 7.52
# 1a30 5.89
Training
bash# Train model
python experiments/train_painn.py --config config/painn_config.yaml --gpu 0

# Monitor training
tensorboard --logdir outputs/logs
Inference
pythonimport torch
from models.painn_affinity import PaiNNAffinityPredictor
from data.featurization import featurize_complex
from data.graph_construction import construct_complex_graph, construct_ligand_graph, construct_protein_graph

# Load model
model = PaiNNAffinityPredictor.from_config('config/painn_config.yaml')
checkpoint = torch.load('outputs/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare complex
complex_data = featurize_complex('protein.pdb', 'ligand.sdf')
ligand_graph = construct_ligand_graph(...)
protein_graph = construct_protein_graph(...)
graph = construct_complex_graph(ligand_graph, protein_graph)

# Predict
with torch.no_grad():
    affinity = model(graph)
    print(f"Predicted pKd: {affinity.item():.2f}")

Module Descriptions
1. data/featurization.py (472 lines)

Ligand: RDKit atom/bond features (49-dim atoms, 10-dim bonds)
Protein: BioPython residue features (31-dim) with secondary structure
3D Geometry: Coordinates, distances, directions
Binding Pocket: 10Å cutoff extraction

Scientific Justification: Features based on GraphDTA, IGN, and molecular ML best practices.
2. data/graph_construction.py (385 lines)

Ligand Graphs: Covalent bonds + self-loops
Protein Graphs: k-NN or radius-based connectivity
Interaction Edges: Distance-based protein-ligand interactions
RBF Expansion: Gaussian radial basis functions

Scientific Justification: Follows PyG conventions and IGN/GIGN heterogeneous graph design.
3. data/dataset.py (251 lines)

Lazy Loading: On-the-fly processing
Caching: Saves processed graphs to disk
Validation: Checks file existence
Memory Efficient: Loads one complex at a time

Engineering: Production-quality with error handling and logging.
4. models/layers/equivariant_layers.py (378 lines)

RBF Expansion: Distance featurization
PaiNN Message: SE(3)-equivariant message passing
PaiNN Update: Gated feature updates
Interaction Layer: Cross-molecular attention

Scientific Justification: Direct implementation of Schütt et al. (2021) PaiNN paper.
5. models/painn_affinity.py (287 lines)

Ligand Encoder: 5-layer PaiNN (scalar + vector features)
Protein Encoder: 3-layer GAT (invariant features)
Interaction: Cross-attention between molecules
Readout: Global pooling + 4-layer MLP

Architecture: Combines PaiNN equivariance with IGN interaction modeling.
6. experiments/train_painn.py (309 lines)

Training Loop: Epoch-based with progress bars
Metrics: RMSE, MAE, Pearson, Spearman
Checkpointing: Saves best model + regular checkpoints
Early Stopping: Patience-based termination
Logging: TensorBoard integration

Engineering: Complete production training infrastructure.

Configuration
Edit config/painn_config.yaml:
yamlmodel:
  hidden_dim: 128          # Feature dimension
  num_message_passing_layers: 5
  num_rbf: 20              # RBF basis functions
  cutoff: 10.0             # Angstroms

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 200
  early_stopping_patience: 20

Validation & Testing
All modules include test code at the bottom:
bash# Test featurization
python data/featurization.py

# Test graph construction
python data/graph_construction.py

# Test equivariant layers
python models/layers/equivariant_layers.py

# Test model
python models/painn_affinity.py

Performance Expectations
Based on literature and architectural design:
MetricExpectedSourceRMSE1.1-1.3PaiNN + interaction modelingPearson0.76-0.80IGN/GIGN benchmarksParameters~2-3MEfficient architecture

Scientific Foundation
This implementation synthesizes:

PaiNN (Schütt et al. 2021): Equivariant message passing
IGN (Zhang et al. 2022): Interaction graph networks
GIGN (Yang et al. 2023): Geometric IGN with 3D
LP-PDBBind (Li et al. 2023): Leak-proof evaluation
PyG Best Practices: Efficient heterogeneous graphs

Key Papers

Schütt et al. (2021) "Equivariant message passing" - ICML
Zhang et al. (2022) "InteractionGraphNet" - JCIM
Yang et al. (2023) "Geometric IGN" - Various
Li et al. (2023) "Leak Proof PDBBind" - arXiv


Extensions & Future Work
This codebase enables:

Active Learning: Uncertainty-guided experimental design
Multi-Task: Affinity + selectivity + ADMET
Transfer Learning: Pre-train on QM9/MD17
Interpretability: Attention visualization, binding analysis
Generative Models: Ligand design with target affinity

License
MIT License

