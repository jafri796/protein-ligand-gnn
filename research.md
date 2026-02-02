# Scientific Foundation & Research Justification

## Overview

This document provides detailed scientific justification for all architectural decisions in this protein-ligand binding affinity prediction system. Every choice is grounded in peer-reviewed literature and best practices from the computational drug discovery community.

---

## 1. DATASET SELECTION & DATA LEAKAGE MITIGATION

### **Choice: LP-PDBBind (Leak-Proof PDBBind)**

**Scientific Justification:**
- **Problem**: Traditional PDBBind splits (general/refined/core) contain severe data leakage
  - Protein sequence similarity between train/test: up to 99.6%
  - Ligand structural similarity causing memorization, not generalization
  - Inflated performance metrics that don't reflect real-world performance

- **Solution**: Li et al. (2023) "Leak Proof PDBBind" [arxiv:2308.09639]
  - Iterative similarity-based splitting
  - Controls for both protein AND ligand similarity
  - Test set deliberately chosen to be MOST DIFFERENT from training
  - Independent BDB2020+ benchmark for true generalization testing

- **Impact**: Models retrained on LP-PDBBind show:
  - More realistic performance estimates
  - Better generalization to unseen proteins/ligands
  - IGN recommended as best-performing model on leak-proof splits

**Alternative Considered**: PDBBind-Opt (2024)
- Rigorous quality control workflow
- Could be used as additional benchmark
- LP-PDBBind preferred for its explicit anti-leakage design

---

## 2. MOLECULAR REPRESENTATION: GRAPH NEURAL NETWORKS

### **Choice: 3D Geometric Graph Representation**

**Scientific Justification:**

**Why GNNs over sequences/SMILES:**
- Nguyen et al. (2021) GraphDTA: "representing drugs as strings is not a natural way to represent molecules"
- Structural information lost in 1D representations
- Graph structure preserves chemical topology

**Why 3D over 2D graphs:**
- Yang et al. (2023) GIGN: 3D structure-based methods achieve Pearson 0.82 vs 0.75 for 2D
- Binding affinity fundamentally determined by 3D spatial interactions
- Hydrogen bonds, π-π stacking, electrostatics require 3D geometry

**Empirical Evidence** (from literature review):
```
Method              Representation    PDBBind Core (Pearson)
-------------------------------------------------------------
DeepDTA            1D sequences       0.630
GraphDTA (GAT-GCN) 2D graphs         0.664
IGN                3D interaction     0.754
GIGN               3D geometric       0.820
```

---

## 3. EQUIVARIANCE TO E(3) TRANSFORMATIONS

### **Choice: SE(3)-Equivariant Message Passing (PaiNN-inspired)**

**Scientific Justification:**

**Physical Principle**: Molecular binding affinity is invariant to:
- Translations (shifting the complex in space)
- Rotations (rotating the complex)
- Reflections (mirror images of complexes)

**Mathematical Foundation**:
- Schütt et al. (2021) PaiNN: "neural network potentials should encode these constraints"
- Enforcing equivariance → better data efficiency + physical correctness
- Without equivariance: model must learn rotational invariance from data (wasteful)

**Architecture Benefits**:
1. **Scalar features** (s): rotation-invariant (energies, distances)
2. **Vector features** (v): rotation-equivariant (forces, dipole moments)
3. **Message passing**: maintains equivariance through construction

**Performance Evidence**:
- PaiNN: QM9 properties with 600k params vs DimeNet++ 1.8M params
- Better accuracy with 3x fewer parameters
- Generalizes to MD17 force field predictions

**Why not full E(3) (with reflections)?**
- SE(3) (special Euclidean) sufficient for molecular systems
- Molecules don't exhibit parity symmetry in practice
- TorchMD-Net, PaiNN use SE(3) successfully

---

## 4. PROTEIN REPRESENTATION

### **Choice: Residue-Level Graphs with 3D Structure**

**Scientific Justification:**

**Granularity**: Residues instead of all atoms
- Computational efficiency: ~300 residues vs ~3000 atoms per protein
- Captures biological relevant units (amino acids)
- Sufficient for binding pocket modeling

**3D Structure Encoding**:
- Cα coordinates for spatial positioning
- Residue-residue distances (< 10Å cutoff)
- Secondary structure features (helix, sheet, coil)

**Binding Pocket Focus**:
- Extract residues within 10Å of ligand (following PLAIG approach)
- Reduces computational cost 10x
- Focuses model on relevant interactions

**Validation from Literature**:
- Jiang et al. (2020) DGraphDTA: contact maps (2D graph) improve over sequences
- GIGN: full 3D protein graph achieves best results
- Consensus: structural information > sequence-only

---

## 5. LIGAND REPRESENTATION

### **Choice: Atomic Graph with RDKit-Derived Features**

**Scientific Justification:**

**Atom Features** (following GraphDTA, IGN best practices):
1. **Atomic number** (1-hot): Chemical identity
2. **Degree**: Connectivity pattern
3. **Formal charge**: Electronic state
4. **Hybridization**: sp, sp2, sp3 bonding
5. **Aromaticity**: Ring resonance
6. **Num H atoms**: Hydrogen bonding potential
7. **Chirality**: Stereochemistry

**Justification**: Duvenaud et al. (2015) Neural FPs - these features encode:
- Electronic properties (charge, aromaticity)
- Geometric properties (hybridization, degree)
- Chemical reactivity (formal charge, chirality)

**Bond Features**:
- Bond type (single, double, triple, aromatic)
- Conjugation
- Ring membership
- Spatial distance (3D)

**Why RDKit**:
- Industry standard for cheminformatics
- Validated feature extraction
- Integration with PyTorch Geometric

---

## 6. INTERACTION GRAPH CONSTRUCTION

### **Choice: Heterogeneous Graph with Intra/Inter-molecular Edges**

**Scientific Justification:**

**Architecture** (following IGN, GIGN):
```
Graph Components:
- Nodes: Ligand atoms + Protein residues (or Cα atoms)
- Edges:
  * Intra-ligand: Covalent bonds (chemical structure)
  * Intra-protein: Spatial proximity (< 10Å)
  * Inter-molecular: Protein-ligand interactions (< 5Å cutoff)
```

**Rationale**:
- Zhang et al. (2022) IGN: "sequentially learn intramolecular then intermolecular interactions"
- **Intra-molecular**: Captures molecular structure
- **Inter-molecular**: Captures binding interactions
- Separate message passing → better feature learning

**Distance Cutoffs** (from literature consensus):
- Intra-protein: 10Å (residue contact definition)
- Inter-molecular: 5Å (typical non-covalent interaction range)
- Van der Waals: ~4Å, H-bonds: 2.5-3.5Å, π-π: 3.5-4.5Å

**Edge Features**:
- Euclidean distance
- Radial basis function expansion (continuous → discrete)
- Directional information (unit vectors for equivariant layers)

---

## 7. MESSAGE PASSING ARCHITECTURE

### **Choice: Separate Intra/Inter-molecular Message Passing + Equivariant Updates**

**Layer Design**:

1. **Ligand Message Passing** (PaiNN-inspired):
   ```python
   # Equivariant message block
   m_s = MLP(concat([s_i, s_j, edge_features]))  # scalar message
   m_v = edge_vector * MLP(||edge_vector||)        # vector message
   
   # Update block
   s_i' = s_i + m_s
   v_i' = v_i + m_v
   ```

2. **Protein Message Passing**:
   - Can use simpler invariant-only (computational efficiency)
   - OR equivariant (if Cα positions used)
   - Trade-off: accuracy vs computation

3. **Interaction Message Passing**:
   - Cross-attention between ligand and protein
   - Directional interactions (H-bonds, π-π stacking)
   - Only inter-molecular edges

**Justification**:
- IGN (2022): Sequential intra→inter learning superior to single-stage
- GIGN (2023): Heterogeneous interaction layer + invariance
- PaiNN (2021): Equivariant message passing for molecules

---

## 8. MODEL ARCHITECTURE

### **Full Architecture**:

```
Input: Protein-Ligand Complex (PDB structure)
  ↓
[Featurization]
  ├─ Ligand: Atom features (79-dim) + 3D coordinates
  └─ Protein: Residue features (20+30-dim) + Cα coordinates
  ↓
[Graph Construction]
  ├─ Ligand graph: Covalent bonds + self-edges
  ├─ Protein graph: k-NN (k=10) or distance cutoff
  └─ Interaction edges: Distance < 5Å
  ↓
[Message Passing] (5 layers)
  ├─ Ligand: PaiNN equivariant layers
  ├─ Protein: GAT or equivariant layers
  └─ Interaction: Cross-attention
  ↓
[Pooling]
  ├─ Ligand: Global mean/sum
  └─ Protein: Binding pocket mean/sum
  ↓
[Readout] 
  └─ MLP(concat[ligand_features, protein_features]) → affinity
```

**Hyperparameter Choices** (from literature):

| Parameter | Value | Source |
|-----------|-------|--------|
| Hidden dim | 128 | PaiNN, GraphDTA consensus |
| Num layers | 5 | IGN, GIGN optimal |
| Dropout | 0.1-0.2 | Standard for GNNs |
| Batch size | 32 | Memory constraints + stability |
| Learning rate | 1e-4 | Adam optimizer standard |

---

## 9. TRAINING PROCEDURE

### **Loss Function: Mean Squared Error (MSE)**

**Justification**:
- Regression task: predict pKd/pKi (continuous)
- MSE standard for affinity prediction
- Direct optimization of RMSE metric

**Alternative**: Huber loss for robustness to outliers
- Could be explored in ablations

### **Metrics**:

1. **RMSE** (Root Mean Squared Error):
   - Primary metric for affinity prediction
   - Standard in PDBBind benchmarks

2. **MAE** (Mean Absolute Error):
   - Interpretable: average prediction error in log units

3. **Pearson Correlation**:
   - Measures linear relationship
   - Standard in drug discovery (ranking ligands)

4. **Spearman Correlation**:
   - Rank-based (non-parametric)
   - Robust to outliers, monotonic relationships

**Justification**: All four metrics standard in:
- DeepDTA, GraphDTA papers
- IGN, GIGN papers
- PDBBind benchmarking studies

### **Optimization**:

- **Optimizer**: Adam (β1=0.9, β2=0.999)
  - Industry standard for deep learning
  - Adaptive learning rates per parameter

- **Learning Rate Schedule**: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 10 epochs
  - Justification: Prevents overfitting, improves convergence

- **Early Stopping**: Patience 20 epochs
  - Prevents overfitting on small validation set

- **Weight Decay**: 1e-5
  - L2 regularization
  - Prevents overfitting on limited data

---

## 10. BASELINE COMPARISONS

### **Baselines to Implement**:

1. **Random Forest on Fingerprints**:
   - Classical ML baseline
   - RDKit Morgan fingerprints
   - Validates GNN improvement

2. **GraphDTA (GAT-GCN)**:
   - Published sequence+graph method
   - Available code: github.com/thinng/GraphDTA
   - Strong 2D baseline

3. **Simple GNN**:
   - Single-stage message passing
   - No equivariance
   - Validates architectural choices

4. **Our Model** (PaiNN-Interaction):
   - Full equivariant architecture
   - Expected best performance

---

## 11. EVALUATION STRATEGY

### **Dataset Splits** (LP-PDBBind):
- **Training**: ~11,000 complexes
- **Validation**: ~2,000 complexes  
- **Test**: ~4,000 complexes (high dissimilarity)

### **Cross-Validation**:
- 3-fold CV on training set
- Reports mean ± std across folds
- Validates stability of results

### **External Validation**:
- BDB2020+ dataset (if available)
- CSAR-HiQ dataset
- Tests true generalization

---

## 12. ABLATION STUDIES

**Planned Ablations**:

1. **Equivariance**:
   - With vs without vector features
   - Quantifies benefit of SE(3) equivariance

2. **3D vs 2D**:
   - 3D coordinates vs graph-only
   - Validates importance of geometry

3. **Interaction Edges**:
   - With vs without protein-ligand edges
   - Tests interaction modeling

4. **Binding Pocket vs Full Protein**:
   - Local (10Å) vs global protein
   - Computational efficiency trade-off

5. **Number of Message Passing Layers**:
   - {3, 5, 7, 10} layers
   - Finds optimal depth

---

## 13. REFERENCES

### **Key Papers** (chronological):

1. **PDBBind Database**:
   - Wang et al. (2004, 2005) - Original dataset
   - Li et al. (2014, 2016) - Refined/Core sets

2. **Data Leakage & Splits**:
   - Li et al. (2023) "Leak Proof PDBBind" - LP-PDBBind
   - Durant et al. (2023) - Data leakage analysis
   - Buttenschoen et al. (2024) - PDBBind-Opt

3. **Sequence-Based Methods**:
   - Öztürk et al. (2018) DeepDTA - CNN on sequences/SMILES
   - Öztürk et al. (2019) WideDTA - Extended DeepDTA

4. **Graph-Based Methods**:
   - Nguyen et al. (2021) GraphDTA - GNN for ligands
   - Jiang et al. (2020) DGraphDTA - GNN for proteins too

5. **3D Interaction Methods**:
   - Zhang et al. (2022) IGN (InteractionGraphNet)
   - Yang et al. (2023) GIGN (Geometric IGN)
   - Various (2023-2025) PLAIG, EIGN, PIGNet

6. **Equivariant GNNs**:
   - Schütt et al. (2017) SchNet - Continuous-filter convolution
   - Klicpera et al. (2020) DimeNet - Directional message passing
   - Schütt et al. (2021) PaiNN - Equivariant message passing
   - Gasteiger et al. (2021) GemNet - Geometry-enhanced

7. **Recent Reviews**:
   - Durant et al. (2023) "Improving generalisability of 3D binding affinity models"
   - Scantlebury et al. (2023) "A Small Step Toward Generalizability"

---