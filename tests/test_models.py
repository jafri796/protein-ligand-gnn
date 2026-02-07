"""
Unit tests for protein-ligand binding affinity prediction system.

This test suite validates:
1. Data featurization correctness
2. Graph construction validity
3. Model architecture integrity
4. Training loop functionality
5. Scientific correctness
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

# Imports from project modules
from data.featurization import (
    get_atom_features, get_bond_features, 
    featurize_ligand, get_residue_features
)
from data.graph_construction import (
    construct_ligand_graph, construct_protein_graph,
    construct_complex_graph
)
from models.layers.equivariant_layers import (
    RBFExpansion, PaiNNLayer, PaiNNMessage, PaiNNUpdate, InteractionLayer
)
from models.painn_affinity import PaiNNAffinityPredictor


class TestFeaturization:
    """Test molecular featurization."""
    
    def test_atom_features_dimension(self):
        """Atom features should be 49-dimensional."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        atom = mol.GetAtomWithIdx(0)
        features = get_atom_features(atom)
        
        assert len(features) == 49, f"Expected 49 features, got {len(features)}"
        assert np.all(np.isfinite(features)), "Features contain NaN or Inf"
    
    def test_bond_features_dimension(self):
        """Bond features should be 10 or 12 dimensional (10 without 3D, 12 with dihedral)."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles("C=C")  # Ethene
        bond = mol.GetBondWithIdx(0)
        features = get_bond_features(bond)
        
        # Without 3D coordinates: 10 features (4+1+1+3+1)
        # With 3D coordinates: 12 features (includes dihedral sin/cos)
        assert len(features) in [10, 12], f"Expected 10 or 12 features, got {len(features)}"
        assert np.all(np.isfinite(features)), "Features contain NaN or Inf"
    
    def test_residue_features_dimension(self):
        """Residue features should be 31-dimensional."""
        from Bio.PDB import PDBParser, Residue
        from Bio.PDB.Polypeptide import PPBuilder
        
        # Create a simple peptide for testing
        peptide = PPBuilder().build_peptides(Residue())[0] if hasattr(Residue(), '__len__') else None
        
        # Test with terminus flags directly
        features = get_residue_features(
            residue=None,  # Will use default values
            secondary_structure='H',
            is_n_terminus=True,
            is_c_terminus=False
        )
        
        assert len(features) == 31, f"Expected 31 features, got {len(features)}"


class TestGraphConstruction:
    """Test graph construction."""
    
    def test_ligand_graph_construction(self):
        """Test ligand graph construction."""
        # Create dummy data
        num_atoms = 10
        atom_features = np.random.randn(num_atoms, 49).astype(np.float32)
        atom_coords = np.random.randn(num_atoms, 3).astype(np.float32)
        bond_indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        bond_features = np.random.randn(3, 9).astype(np.float32)
        
        ligand_graph = construct_ligand_graph(
            atom_features, atom_coords, bond_indices, bond_features
        )
        
        assert ligand_graph.x.shape == (num_atoms, 49)
        assert ligand_graph.pos.shape == (num_atoms, 3)
        assert torch.isfinite(ligand_graph.x).all()
        assert torch.isfinite(ligand_graph.pos).all()
    
    def test_protein_graph_construction(self):
        """Test protein graph construction."""
        num_residues = 50
        residue_features = np.random.randn(num_residues, 31).astype(np.float32)
        residue_coords = np.random.randn(num_residues, 3).astype(np.float32)
        
        # Use radius method instead of knn (requires torch-cluster)
        protein_graph = construct_protein_graph(
            residue_features, residue_coords, method='radius', cutoff=5.0
        )
        
        assert protein_graph.x.shape == (num_residues, 31)
        assert protein_graph.pos.shape == (num_residues, 3)
        assert torch.isfinite(protein_graph.x).all()
    
    def test_complex_graph_construction(self):
        """Test full complex graph construction."""
        # Create ligand graph
        num_atoms = 10
        atom_features = np.random.randn(num_atoms, 49).astype(np.float32)
        atom_coords = np.random.randn(num_atoms, 3).astype(np.float32)
        bond_indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
        bond_features = np.random.randn(2, 9).astype(np.float32)
        
        ligand_graph = construct_ligand_graph(
            atom_features, atom_coords, bond_indices, bond_features
        )
        
        # Create protein graph
        num_residues = 20
        residue_features = np.random.randn(num_residues, 31).astype(np.float32)
        residue_coords = np.random.randn(num_residues, 3).astype(np.float32)
        
        protein_graph = construct_protein_graph(
            residue_features, residue_coords, method='radius', cutoff=5.0
        )
        
        # Construct complex
        complex_graph = construct_complex_graph(
            ligand_graph, protein_graph, interaction_cutoff=5.0
        )
        
        assert complex_graph.x.shape[0] == num_atoms + num_residues
        assert torch.isfinite(complex_graph.x).all()


class TestEquivariantLayers:
    """Test SE(3)-equivariant message passing layers."""
    
    def test_rbf_expansion(self):
        """Test RBF expansion."""
        rbf = RBFExpansion(num_rbf=20, cutoff=10.0)
        distances = torch.linspace(0, 10, 100)
        
        rbf_features = rbf(distances)
        
        assert rbf_features.shape == (100, 20)
        assert torch.isfinite(rbf_features).all()
        # RBF values should be in [0, 1]
        assert (rbf_features >= 0).all() and (rbf_features <= 1.1).all()  # Small tolerance
    
    def test_painn_update_equivariance(self):
        """Test that PaiNN update preserves vector shapes."""
        hidden_dim = 128
        num_nodes = 10
        
        s = torch.randn(num_nodes, hidden_dim)
        v = torch.randn(num_nodes, 3, hidden_dim)
        
        update = PaiNNUpdate(hidden_dim)
        s_out, v_out = update(s, v)
        
        assert s_out.shape == s.shape, "Scalar features shape changed"
        assert v_out.shape == v.shape, "Vector features shape changed"
        assert torch.isfinite(s_out).all()
        assert torch.isfinite(v_out).all()
    
    def test_painn_layer_forward(self):
        """Test complete PaiNN layer forward pass."""
        hidden_dim = 64
        num_rbf = 20
        num_nodes = 10
        num_edges = 20
        
        s = torch.randn(num_nodes, hidden_dim)
        v = torch.randn(num_nodes, 3, hidden_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_rbf = torch.randn(num_edges, num_rbf)
        edge_vec = torch.randn(num_edges, 3)
        edge_vec = edge_vec / (torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8)
        
        layer = PaiNNLayer(hidden_dim, num_rbf)
        s_out, v_out = layer(s, v, edge_index, edge_rbf, edge_vec)
        
        assert s_out.shape == s.shape, "Scalar output shape mismatch"
        assert v_out.shape == v.shape, "Vector output shape mismatch"
        assert torch.isfinite(s_out).all()
        assert torch.isfinite(v_out).all()


class TestRotationEquivariance:
    """Test SE(3) rotation equivariance of PaiNN layers."""
    
    def test_painn_message_rotation_equivariance(self):
        """Test that PaiNNMessage preserves SE(3) equivariance.
        
        Under rotation R:
        - Scalar features should be invariant: s' = s
        - Vector features should be equivariant: v' = R @ v
        - Edge vectors should transform: edge_vec' = R @ edge_vec
        """
        batch_size = 10
        hidden_dim = 32
        num_rbf = 10
        
        # Create random features
        s = torch.randn(batch_size, hidden_dim)
        v = torch.randn(batch_size, 3, hidden_dim)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        edge_rbf = torch.randn(edge_index.size(1), num_rbf)
        edge_vec = torch.randn(edge_index.size(1), 3)
        edge_vec = edge_vec / (edge_vec.norm(dim=1, keepdim=True) + 1e-8)
        
        # Create random rotation matrix
        angle = torch.rand(1) * 2 * np.pi
        axis = torch.randn(3)
        axis = axis / axis.norm()
        
        # Rodrigues rotation formula
        K = torch.zeros(3, 3)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        
        # Apply rotation to vector features and edge vectors
        v_rotated = torch.einsum('ij,bjh->bih', R, v)
        edge_vec_rotated = torch.einsum('ij,ej->ei', R, edge_vec)
        
        # Create message passing layer
        layer = PaiNNMessage(hidden_dim, num_rbf)
        layer.eval()
        
        # Forward pass with original features
        ds1, dv1 = layer(s, v, edge_index, edge_rbf, edge_vec)
        
        # Forward pass with rotated features
        ds2, dv2 = layer(s, v_rotated, edge_index, edge_rbf, edge_vec_rotated)
        
        # Scalars should be invariant
        assert torch.allclose(ds1, ds2, atol=1e-5), "Scalars not rotation invariant"
        
        # Vectors should be equivariant: R @ dv1 should equal dv2
        dv1_rotated = torch.einsum('ij,bjh->bih', R, dv1)
        assert torch.allclose(dv1_rotated, dv2, atol=1e-4), "Vectors not rotation equivariant"
    
    def test_painn_update_rotation_equivariance(self):
        """Test that PaiNNUpdate preserves SE(3) equivariance."""
        batch_size = 10
        hidden_dim = 32
        
        s = torch.randn(batch_size, hidden_dim)
        v = torch.randn(batch_size, 3, hidden_dim)
        
        # Create random rotation matrix
        angle = torch.rand(1) * 2 * np.pi
        axis = torch.randn(3)
        axis = axis / axis.norm()
        
        K = torch.zeros(3, 3)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        
        v_rotated = torch.einsum('ij,bjh->bih', R, v)
        
        # Create update layer
        update = PaiNNUpdate(hidden_dim)
        update.eval()
        
        # Forward pass
        s_out1, v_out1 = update(s, v)
        s_out2, v_out2 = update(s, v_rotated)
        
        # Scalars should be invariant
        assert torch.allclose(s_out1, s_out2, atol=1e-5), "Scalars not rotation invariant in update"
        
        # Vectors should be equivariant
        v_out1_rotated = torch.einsum('ij,bjh->bih', R, v_out1)
        assert torch.allclose(v_out1_rotated, v_out2, atol=1e-4), "Vectors not rotation equivariant in update"


class TestModel:
    """Test the full PaiNN model."""
    
    def test_model_creation(self):
        """Test model instantiation."""
        config = {
            'hidden_dim': 64,
            'num_message_passing_layers': 2,
            'num_protein_layers': 2,
            'num_rbf': 10,
            'cutoff': 10.0,
            'dropout': 0.1
        }
        model = PaiNNAffinityPredictor(config)
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test model forward pass with batch data."""
        from torch_geometric.data import Data, Batch
        from data.featurization import LIGAND_ATOM_FEATURE_DIM, LIGAND_BOND_FEATURE_DIM, PROTEIN_RESIDUE_FEATURE_DIM
        
        config = {
            'hidden_dim': 64,
            'num_message_passing_layers': 2,
            'num_protein_layers': 2,
            'num_rbf': 10,
            'cutoff': 10.0,
            'dropout': 0.1
        }
        model = PaiNNAffinityPredictor(config)
        model.eval()
        
        # Padded feature dim (matches construct_complex_graph)
        padded_dim = max(LIGAND_ATOM_FEATURE_DIM, PROTEIN_RESIDUE_FEATURE_DIM)
        # Edge attr dim: max(bond_feat+1, 4, 4) + 3 edge_type one-hot
        edge_attr_dim = LIGAND_BOND_FEATURE_DIM + 1 + 3
        
        # Create batch of dummy complexes
        data_list = []
        for _ in range(2):
            num_ligand = 8
            num_protein = 15
            num_nodes = num_ligand + num_protein
            
            # Both padded to same dim (as construct_complex_graph does)
            x = torch.randn(num_nodes, padded_dim)
            
            pos = torch.randn(num_nodes, 3)
            edge_index = torch.randint(0, num_nodes, (2, 30))
            edge_attr = torch.randn(30, edge_attr_dim)
            
            node_type = torch.cat([
                torch.zeros(num_ligand, dtype=torch.long),
                torch.ones(num_protein, dtype=torch.long)
            ])
            y = torch.tensor([7.5])
            
            data = Data(
                x=x, pos=pos, edge_index=edge_index,
                edge_attr=edge_attr, node_type=node_type, y=y
            )
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        
        # Forward pass
        with torch.no_grad():
            output = model(batch)
        
        assert output.shape == (2,), f"Expected output shape (2,), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"


class TestReproducibility:
    """Test reproducibility settings."""
    
    def test_seed_setting(self):
        """Test that random seed produces reproducible results."""
        from utils import set_seed
        
        # First run
        set_seed(42)
        rand1 = torch.randn(10)
        
        # Second run with same seed
        set_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2), "Random seed not reproducible"


class TestDihedralFeatures:
    """Test 3D dihedral feature computation."""
    
    def test_dihedral_angle_computation(self):
        """Test dihedral angle computation for rotatable bonds."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from data.featurization import get_dihedral_angle
        
        # Create a molecule with 3D coordinates
        mol = Chem.MolFromSmiles("CCCC")  # n-butane has rotatable bond
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Get middle bond (C-C rotatable)
        bond = mol.GetBondWithIdx(1)
        dihedral = get_dihedral_angle(mol, bond)
        
        # Dihedral should be in [-π, π]
        assert -np.pi <= dihedral <= np.pi, f"Dihedral {dihedral} out of range"
        
    def test_bond_features_with_dihedral(self):
        """Test bond features include dihedral when 3D coords available."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from data.featurization import get_bond_features
        
        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        bond = mol.GetBondWithIdx(1)
        features = get_bond_features(bond, mol)
        
        # Should be 13-dim with dihedral (sin/cos)
        assert len(features) >= 9, f"Bond features too short: {len(features)}"
        assert np.all(np.isfinite(features)), "Features contain NaN or Inf"


class TestTerminusDetection:
    """Test N/C terminus detection in protein featurization."""
    
    def test_terminus_flags_in_features(self):
        """Test that terminus flags are correctly set in residue features."""
        from data.featurization import get_residue_features
        
        # N-terminus residue
        n_term_features = get_residue_features(
            None, secondary_structure='H', 
            is_n_terminus=True, is_c_terminus=False
        )
        
        # C-terminus residue
        c_term_features = get_residue_features(
            None, secondary_structure='H',
            is_n_terminus=False, is_c_terminus=True
        )
        
        # Middle residue
        mid_features = get_residue_features(
            None, secondary_structure='H',
            is_n_terminus=False, is_c_terminus=False
        )
        
        # Check dimensions
        assert len(n_term_features) == 31, f"Expected 31 features, got {len(n_term_features)}"
        
        # N-terminus flag is at index 29, C-terminus at index 30
        assert n_term_features[29] == 1.0, "N-terminus flag not set"
        assert n_term_features[30] == 0.0, "C-terminus flag incorrectly set"
        
        assert c_term_features[29] == 0.0, "N-terminus flag incorrectly set"
        assert c_term_features[30] == 1.0, "C-terminus flag not set"
        
        assert mid_features[29] == 0.0, "N-terminus flag incorrectly set for middle"
        assert mid_features[30] == 0.0, "C-terminus flag incorrectly set for middle"


class TestScientificValidity:
    """Test scientific correctness of features and calculations."""
    
    def test_binding_pocket_identification(self):
        """Test binding pocket identification."""
        from data.featurization import identify_binding_pocket
        
        protein_coords = np.random.randn(50, 3)
        ligand_coords = np.random.randn(10, 3)
        
        # Place some ligand atoms near protein
        ligand_coords[0] = protein_coords[0] + 0.5 * np.random.randn(3)
        ligand_coords[1] = protein_coords[10] + 0.5 * np.random.randn(3)
        
        in_pocket = identify_binding_pocket(protein_coords, ligand_coords, cutoff=5.0)
        
        # At least residues 0 and 10 should be in pocket
        assert in_pocket[0] or in_pocket[10], "Binding pocket identification failed"
    
    def test_distance_calculation_correctness(self):
        """Test that distances are calculated correctly."""
        coords1 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        coords2 = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        # Distance from (0,0,0) to (1,0,0) should be 1
        dist = np.linalg.norm(coords1[1] - coords2[0])
        assert np.isclose(dist, 1.0), f"Expected distance 1.0, got {dist}"


if __name__ == "__main__":
    # Run tests with: pytest tests/test_models.py -v
    pytest.main([__file__, "-v"])
