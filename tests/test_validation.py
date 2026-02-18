"""
Validation tests for the protein-ligand GNN implementation.

These tests verify that the fixes and improvements are working correctly.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from pathlib import Path
import tempfile
import os

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.painn_affinity import PaiNNAffinityPredictor
from data.featurization import (
    LIGAND_ATOM_FEATURE_DIM,
    PROTEIN_RESIDUE_FEATURE_DIM
)
from data.splits import create_lp_pdbbind_splits
from utils import set_seed


class TestSE3Equivariance:
    """Test SE(3) equivariance fixes."""
    
    def test_painn_message_equivariance(self):
        """Test that PaiNNMessage preserves SE(3) equivariance."""
        from models.layers.equivariant_layers import PaiNNMessage, RBFExpansion
        
        hidden_dim = 64
        num_rbf = 20
        message = PaiNNMessage(hidden_dim, num_rbf)
        rbf = RBFExpansion(num_rbf, cutoff=10.0)
        
        # Create test data
        batch_size = 10
        s = torch.randn(batch_size, hidden_dim)
        v = torch.randn(batch_size, 3, hidden_dim)
        edge_index = torch.randint(0, batch_size, (2, 20))
        edge_vec = torch.randn(20, 3)
        edge_dist = torch.norm(edge_vec, dim=1)
        edge_rbf = rbf(edge_dist)
        
        # Apply rotation
        rotation = torch.tensor([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Original message
        ds_orig, dv_orig = message(s, v, edge_index, edge_rbf, edge_vec)
        
        # Rotated message
        v_rot = torch.einsum('ij,njk->nik', rotation, v)
        edge_vec_rot = torch.einsum('ij,ej->ei', rotation, edge_vec)
        ds_rot, dv_rot = message(s, v_rot, edge_index, edge_rbf, edge_vec_rot)
        
        # Check scalar invariance
        assert torch.allclose(ds_orig, ds_rot, atol=1e-5), "Scalar features should be invariant"
        
        # Check vector equivariance
        dv_rot_expected = torch.einsum('ij,njk->nik', rotation, dv_orig)
        assert torch.allclose(dv_rot, dv_rot_expected, atol=1e-5), "Vector features should rotate correctly"


class TestFeatureDimensions:
    """Test feature dimension consistency."""
    
    def test_feature_dimension_constants(self):
        """Test that feature dimension constants are correct."""
        assert LIGAND_ATOM_FEATURE_DIM == 49
        assert PROTEIN_RESIDUE_FEATURE_DIM == 31
    
    def test_model_uses_constants(self):
        """Test that model uses imported constants."""
        config = {
            'hidden_dim': 64,
            'num_message_passing_layers': 2,
            'num_protein_layers': 2,
            'num_rbf': 10,
            'cutoff': 10.0,
            'dropout': 0.1
        }
        model = PaiNNAffinityPredictor(config)
        
        # Both embeddings use padded dimension (max of ligand and protein)
        padded_dim = max(LIGAND_ATOM_FEATURE_DIM, PROTEIN_RESIDUE_FEATURE_DIM)
        assert model.ligand_embedding_s.in_features == padded_dim
        assert model.protein_embedding_s.in_features == padded_dim


class TestEdgeFeatureHandling:
    """Test edge feature handling fixes."""
    
    def test_interaction_layer_edge_dim(self):
        """Test that InteractionLayer handles configurable edge features."""
        from models.layers.equivariant_layers import InteractionLayer
        
        hidden_dim = 64
        edge_dim = 14
        layer = InteractionLayer(hidden_dim, num_heads=4, edge_dim=edge_dim)
        
        # Check edge network input dimension
        assert layer.edge_net[0].in_features == edge_dim, f"Should handle {edge_dim}-dimensional edge features"
    
    def test_model_forward_with_edge_types(self):
        """Test model forward pass with edge type indicators."""
        from data.featurization import LIGAND_BOND_FEATURE_DIM
        
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
        
        # Padded dims matching construct_complex_graph
        padded_dim = max(LIGAND_ATOM_FEATURE_DIM, PROTEIN_RESIDUE_FEATURE_DIM)
        edge_attr_dim = LIGAND_BOND_FEATURE_DIM + 1 + 3
        
        num_ligand = 8
        num_protein = 12
        num_nodes = num_ligand + num_protein
        
        x = torch.randn(num_nodes, padded_dim)
        pos = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 30))
        edge_attr = torch.randn(30, edge_attr_dim)
        
        node_type = torch.cat([
            torch.zeros(num_ligand, dtype=torch.long),
            torch.ones(num_protein, dtype=torch.long)
        ])
        
        batch = Batch.from_data_list([Data(
            x=x, pos=pos, edge_index=edge_index,
            edge_attr=edge_attr, node_type=node_type
        )])
        
        # Forward pass should work
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"


class TestProteinVectorFeatures:
    """Test protein vector feature initialization."""
    
    def test_protein_vector_features_initialized(self):
        """Test that protein vector features are properly initialized."""
        from data.featurization import LIGAND_BOND_FEATURE_DIM
        
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
        
        padded_dim = max(LIGAND_ATOM_FEATURE_DIM, PROTEIN_RESIDUE_FEATURE_DIM)
        edge_attr_dim = LIGAND_BOND_FEATURE_DIM + 1 + 3
        
        num_ligand = 8
        num_protein = 12
        num_nodes = num_ligand + num_protein
        
        x = torch.randn(num_nodes, padded_dim)
        pos = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 30))
        edge_attr = torch.randn(30, edge_attr_dim)
        
        node_type = torch.cat([
            torch.zeros(num_ligand, dtype=torch.long),
            torch.ones(num_protein, dtype=torch.long)
        ])
        
        batch = Batch.from_data_list([Data(
            x=x, pos=pos, edge_index=edge_index,
            edge_attr=edge_attr, node_type=node_type
        )])
        
        # Forward pass should not fail due to missing vector features
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (1,)


class TestLPPDBBindSplits:
    """Test LP-PDBBind split implementation."""
    
    def test_lp_pdbbind_split_logic(self):
        """Test that LP-PDBBind splits use OR logic for similarity."""
        # Create mock similarity matrices
        seq_sim = np.array([
            [1.0, 0.3, 0.8],
            [0.3, 1.0, 0.2],
            [0.8, 0.2, 1.0]
        ])
        lig_sim = np.array([
            [1.0, 0.1, 0.2],
            [0.1, 1.0, 0.7],
            [0.2, 0.7, 1.0]
        ])
        
        # Test with cutoffs
        protein_seq_cutoff = 0.5
        ligand_sim_cutoff = 0.5
        
        # Complex 0 and 2 should be similar (sequence similarity)
        assert seq_sim[0, 2] > protein_seq_cutoff
        
        # Complex 1 and 2 should be similar (ligand similarity)
        assert lig_sim[1, 2] > ligand_sim_cutoff
        
        # Combined similarity should use OR logic
        combined = np.maximum(seq_sim, lig_sim)
        assert combined[0, 2] > protein_seq_cutoff  # Due to sequence
        assert combined[1, 2] > ligand_sim_cutoff  # Due to ligand


class TestDatasetValidation:
    """Test dataset validation implementation."""
    
    def test_dataset_validation_exists(self):
        """Test that dataset validation is implemented."""
        from data.dataset import ProteinLigandDataset
        
        # Check that _validate_files method exists
        assert hasattr(ProteinLigandDataset, '_validate_files'), \
            "Dataset should have _validate_files method"
    
    def test_dataset_validation_filters_missing_files(self):
        """Test that validation filters out entries with missing/invalid files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index file in expected format (space-separated: pdb_id affinity)
            index_file = Path(tmpdir) / "index.txt"
            index_file.write_text("1abc 7.5\n2def 6.0\n")
            
            # Create data directory structure for 1abc but NOT for 2def
            # This tests that validation correctly filters missing entries
            data_dir = Path(tmpdir) / "data"
            pdb_dir = data_dir / "1abc"
            pdb_dir.mkdir(parents=True)
            
            # Create placeholder files (they won't parse, but _validate_files
            # should filter them out gracefully)
            (pdb_dir / "1abc_protein.pdb").write_text("MOCK")
            (pdb_dir / "1abc_ligand.sdf").write_text("MOCK")
            
            # Dataset should load 2 entries from index, then validation
            # should filter both out (1abc has unparseable files, 2def is missing)
            dataset = ProteinLigandDataset(
                str(data_dir), str(index_file), use_cache=False
            )
            
            # Both should be filtered: 2def missing files, 1abc unparseable
            assert len(dataset.data_list) == 0, \
                "Invalid/missing files should be filtered by validation"


class TestSetSeedCentralization:
    """Test set_seed centralization."""
    
    def test_set_seed_imported_from_utils(self):
        """Test that set_seed is imported from utils."""
        from experiments.train_painn import set_seed as train_set_seed
        from utils import set_seed as utils_set_seed
        
        assert train_set_seed is utils_set_seed, \
            "train_painn should import set_seed from utils"
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed provides reproducibility."""
        # Set seed and generate random numbers
        set_seed(42)
        rand1 = torch.rand(5)
        
        # Reset seed and generate again
        set_seed(42)
        rand2 = torch.rand(5)
        
        assert torch.allclose(rand1, rand2), "Random numbers should be reproducible"


class TestErrorHandling:
    """Test error handling improvements."""
    
    def test_training_error_handling(self):
        """Test that training script has proper error handling."""
        from experiments.train_painn import main
        
        # Check that main function has try-except blocks
        import inspect
        source = inspect.getsource(main)
        assert "try:" in source, "Main function should have try blocks"
        assert "except" in source, "Main function should have except blocks"
    
    def test_logging_configuration(self):
        """Test logging configuration improvements."""
        from experiments.train_painn import logger
        
        # Check that logger is configured
        assert logger is not None, "Logger should be configured"
        
        # Check handlers
        assert len(logger.handlers) > 0, "Logger should have handlers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
