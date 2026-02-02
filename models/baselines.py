"""
Baseline comparison models for protein-ligand binding affinity prediction.

Implements:
1. GraphDTA (Nguyen et al. 2021) - graph-based approach using 1D sequences
2. Random Forest baseline - simple feature-based model
3. Linear regression baseline - minimal model

These baselines establish performance bounds for the PaiNN model.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE EXTRACTION FROM GRAPHS
# =============================================================================

class GraphDTAFeatureExtractor:
    """
    Extract features from PyG graphs for baseline models.
    
    Converts graph node/edge features into flat feature vectors for
    classical ML models (RF, Linear Regression).
    """
    
    def extract_features(self, graph_data) -> np.ndarray:
        """
        Extract features from a PyG Data object (complex graph).
        
        Args:
            graph_data: PyG Data object with node/edge features
            
        Returns:
            Feature vector suitable for classical ML
        """
        features = []
        
        try:
            # Graph statistics
            num_nodes = graph_data.num_nodes
            num_edges = graph_data.num_edges
            features.append(float(num_nodes))
            features.append(float(num_edges))
            features.append(float(num_edges) / max(num_nodes, 1))  # Edge density
            
            # Node feature statistics
            if hasattr(graph_data, 'x') and graph_data.x is not None:
                x = graph_data.x.numpy() if hasattr(graph_data.x, 'numpy') else graph_data.x
                features.append(float(np.mean(x)))
                features.append(float(np.std(x)))
                features.append(float(np.min(x)))
                features.append(float(np.max(x)))
            else:
                features.extend([0.0] * 4)
            
            # Edge feature statistics
            if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                edge_attr = graph_data.edge_attr.numpy() if hasattr(graph_data.edge_attr, 'numpy') else graph_data.edge_attr
                if edge_attr.size > 0:
                    features.append(float(np.mean(edge_attr)))
                    features.append(float(np.std(edge_attr)))
                    features.append(float(np.min(edge_attr)))
                    features.append(float(np.max(edge_attr)))
                else:
                    features.extend([0.0] * 4)
            else:
                features.extend([0.0] * 4)
            
            # Node type statistics (if heterogeneous)
            if hasattr(graph_data, 'node_type') and graph_data.node_type is not None:
                node_type = graph_data.node_type.numpy() if hasattr(graph_data.node_type, 'numpy') else graph_data.node_type
                features.append(float(np.sum(node_type == 0)))  # Ligand nodes
                features.append(float(np.sum(node_type == 1)))  # Protein nodes
            else:
                features.extend([0.0] * 2)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None


# =============================================================================
# GRAPHDTA BASELINE
# =============================================================================

class GraphDTABlock(nn.Module):
    """Single residual block from GraphDTA."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.conv2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + residual  # Residual connection
        return x


class GraphDTAEncoder(nn.Module):
    """
    GraphDTA encoder for sequences (protein or ligand SMILES).
    
    Encodes sequences via embedding + GRU + attention mechanism.
    Reference: Nguyen et al. (2021) "Graph-Based Tensor Neural Network"
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_blocks = nn.Sequential(
            *[GraphDTABlock(hidden_dim * 2, dropout) for _ in range(2)]
        )
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) token indices
            
        Returns:
            (batch_size, hidden_dim * 2) encoded representation
        """
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)
        
        # BiGRU
        x, _ = self.gru(x)  # (batch, seq_len, hidden_dim * 2)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # (batch, seq_len, hidden_dim * 2)
        x = x + attn_out  # Residual
        
        # FC blocks
        x = self.fc_blocks(x)  # (batch, seq_len, hidden_dim * 2)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, hidden_dim * 2)
        
        return x


class GraphDTA(nn.Module):
    """
    GraphDTA model for binding affinity prediction.
    
    Combines protein sequence encoder + ligand SMILES encoder
    with fusion network for pKd prediction.
    """
    
    def __init__(self, protein_vocab_size: int = 21,  # 20 AAs + padding
                 ligand_vocab_size: int = 65,   # SMILES tokens
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.protein_encoder = GraphDTAEncoder(
            protein_vocab_size, hidden_dim, num_layers=2, dropout=dropout
        )
        self.ligand_encoder = GraphDTAEncoder(
            ligand_vocab_size, hidden_dim, num_layers=2, dropout=dropout
        )
        
        # Fusion network
        fusion_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, 1)
        )
        
    def forward(self, protein_tokens: torch.Tensor, 
                ligand_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            protein_tokens: (batch_size, protein_seq_len)
            ligand_tokens: (batch_size, ligand_seq_len)
            
        Returns:
            (batch_size, 1) predicted pKd values
        """
        protein_feat = self.protein_encoder(protein_tokens)
        ligand_feat = self.ligand_encoder(ligand_tokens)
        
        # Concatenate and fuse
        fused = torch.cat([protein_feat, ligand_feat], dim=1)
        affinity = self.fusion(fused)
        
        return affinity


# =============================================================================
# RANDOM FOREST BASELINE
# =============================================================================

class RFBaseline:
    """
    Random Forest baseline using hand-crafted features.
    
    Features:
    - Molecular properties: MW, logP, HBA, HBD, TPSA, rotatable bonds
    - Protein properties: length, secondary structure content, hydrophobicity
    - Interaction features: estimated by distance-based heuristics
    
    Scientific Rationale:
    RF with engineered features establishes classical ML lower bound.
    Outperforming RF with graph methods validates learned representations.
    """
    
    def __init__(self, n_estimators: int = 100, 
                 max_depth: int = 15,
                 random_state: int = 42):
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def extract_features(self, protein_seq: str, ligand_smiles: str) -> np.ndarray:
        """
        Extract hand-crafted features from protein sequence and ligand SMILES.
        
        Features (14 total):
        - Protein: length, AA composition, hydrophobicity
        - Ligand: MW, logP, HBD, HBA, TPSA, rotatable bonds, aromatic rings
        
        Args:
            protein_seq: Protein sequence string
            ligand_smiles: SMILES string
            
        Returns:
            Feature vector (14,)
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen
        
        features = []
        
        # ===== PROTEIN FEATURES =====
        # Length (continuous, in hundreds)
        protein_len = len(protein_seq) / 100.0
        features.append(protein_len)
        
        # Amino acid composition ratios
        features.append(protein_seq.count('C') / max(len(protein_seq), 1))  # Cysteine (disulfide bridges)
        features.append(protein_seq.count('P') / max(len(protein_seq), 1))  # Proline (structure)
        features.append(protein_seq.count('H') / max(len(protein_seq), 1))  # Histidine (pH-sensitive)
        
        # Hydrophobicity (Kyte-Doolittle scale approximation)
        hydrophobic_aas = set('AILMFVP')
        hydrophobicity = sum(1 for aa in protein_seq if aa in hydrophobic_aas) / max(len(protein_seq), 1)
        features.append(hydrophobicity)
        
        # ===== LIGAND FEATURES =====
        try:
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol is not None:
                # Molecular properties
                mw = Descriptors.MolWt(mol) / 500.0  # Normalized to typical drug range
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHBD(mol)
                hba = Descriptors.NumHBA(mol)
                tpsa = Descriptors.TPSA(mol) / 200.0  # Normalized
                rot_bonds = Descriptors.NumRotatableBonds(mol) / 20.0  # Normalized
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                
                features.extend([mw, logp, hbd, hba, tpsa, rot_bonds, aromatic_rings])
            else:
                # Missing ligand: use defaults
                features.extend([0.0] * 7)
                logger.warning(f"Failed to parse SMILES: {ligand_smiles}")
        except Exception as e:
            logger.warning(f"Exception in ligand feature extraction: {e}")
            features.extend([0.0] * 7)
        
        return np.array(features, dtype=np.float32)
    
    def train(self, protein_sequences: list, ligand_smiles: list, 
              affinities: np.ndarray) -> float:
        """
        Train the Random Forest model.
        
        Args:
            protein_sequences: List of protein sequences
            ligand_smiles: List of ligand SMILES
            affinities: Target pKd values
            
        Returns:
            Training R² score
        """
        # Extract features for all samples
        X = np.array([
            self.extract_features(prot, lig)
            for prot, lig in zip(protein_sequences, ligand_smiles)
        ])
        
        # Standardize features
        X = self.scaler.fit_transform(X)
        
        # Train model
        self.rf.fit(X, affinities)
        self.is_fitted = True
        
        train_score = self.rf.score(X, affinities)
        logger.info(f"Random Forest train R²: {train_score:.4f}")
        
        return train_score
    
    def predict(self, protein_sequences: list, ligand_smiles: list) -> np.ndarray:
        """
        Make predictions on new samples.
        
        Args:
            protein_sequences: List of protein sequences
            ligand_smiles: List of ligand SMILES
            
        Returns:
            Predicted pKd values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        
        X = np.array([
            self.extract_features(prot, lig)
            for prot, lig in zip(protein_sequences, ligand_smiles)
        ])
        
        X = self.scaler.transform(X)
        return self.rf.predict(X)


# =============================================================================
# LINEAR REGRESSION BASELINE
# =============================================================================

class LinearBaseline:
    """
    Simple linear regression baseline.
    
    Minimal model using only sequence length and basic properties.
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_features(self, protein_seq: str, ligand_smiles: str) -> np.ndarray:
        """Minimal feature extraction."""
        features = [
            len(protein_seq),
            len(ligand_smiles),
            protein_seq.count('C') / max(len(protein_seq), 1),
        ]
        return np.array(features, dtype=np.float32)
    
    def train(self, protein_sequences: list, ligand_smiles: list,
              affinities: np.ndarray) -> float:
        """Train the linear model."""
        X = np.array([
            self.extract_features(prot, lig)
            for prot, lig in zip(protein_sequences, ligand_smiles)
        ])
        
        X = self.scaler.fit_transform(X)
        self.model.fit(X, affinities)
        self.is_fitted = True
        
        train_score = self.model.score(X, affinities)
        logger.info(f"Linear baseline train R²: {train_score:.4f}")
        
        return train_score
    
    def predict(self, protein_sequences: list, ligand_smiles: list) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first")
        
        X = np.array([
            self.extract_features(prot, lig)
            for prot, lig in zip(protein_sequences, ligand_smiles)
        ])
        
        X = self.scaler.transform(X)
        return self.model.predict(X)
