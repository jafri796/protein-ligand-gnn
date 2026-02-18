"""
PyTorch Dataset for Protein-Ligand Complexes

Implements efficient data loading with:
- On-the-fly processing or pre-processed caching
- Memory-efficient file handling
- Proper error handling and validation
"""

import os
import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable
from torch.utils.data import Dataset
from torch_geometric.data import Data
import logging

from .featurization import featurize_complex
from .graph_construction import construct_complex_graph, construct_ligand_graph, construct_protein_graph

logger = logging.getLogger(__name__)


class ProteinLigandDataset(Dataset):
    """
    Dataset for protein-ligand complexes from PDBBind.
    
    Features:
    - Lazy loading: processes complexes on-the-fly
    - Caching: saves processed graphs to disk
    - Memory efficient: loads one complex at a time
    - Validation: checks file existence and format
    
    Args:
        data_dir: Directory containing PDB and SDF files
        index_file: CSV/text file with [pdb_id, affinity] pairs
        cache_dir: Directory to save/load processed graphs
        binding_pocket_only: Use only binding pocket residues
        pocket_cutoff: Distance cutoff for binding pocket (Angstroms)
        interaction_cutoff: Distance cutoff for interactions (Angstroms)
        transform: Optional transform to apply to graphs
        use_cache: Whether to use cached processed data
        target_stats: Optional target statistics for consistent normalization
    """
    
    def __init__(
        self,
        data_dir: str,
        index_file: str,
        cache_dir: Optional[str] = None,
        binding_pocket_only: bool = True,
        pocket_cutoff: float = 10.0,
        interaction_cutoff: float = 5.0,
        transform: Optional[Callable] = None,
        use_cache: bool = True,
        target_stats: Optional[Dict[str, float]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.index_file = Path(index_file)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.binding_pocket_only = binding_pocket_only
        self.pocket_cutoff = pocket_cutoff
        self.interaction_cutoff = interaction_cutoff
        self.transform = transform
        self.use_cache = use_cache
        self.target_stats = target_stats
        
        # Create cache directory if it doesn't exist
        if self.cache_dir and self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load index file
        self.data_list = self._load_index()
        logger.info(f"Loaded {len(self.data_list)} complexes from {index_file}")
        
        # Validate files
        self._validate_files()
        
        # Target statistics for optional normalization (train stats can be reused)
        if target_stats and {'mean', 'std'} <= target_stats.keys():
            self.target_mean = float(target_stats['mean'])
            self.target_std = float(max(target_stats['std'], 1e-8))
        else:
            self._compute_target_stats()
    
    def _load_index(self) -> List[Dict]:
        """Load index file with PDB IDs and affinities."""
        data_list = []
        
        with open(self.index_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    pdb_id = parts[0]
                    try:
                        affinity = float(parts[1])
                    except ValueError:
                        logger.warning(f"Invalid affinity for {pdb_id}: {parts[1]}")
                        continue
                    
                    data_list.append({
                        'pdb_id': pdb_id,
                        'affinity': affinity,
                        'protein_file': self.data_dir / pdb_id / f"{pdb_id}_protein.pdb",
                        'ligand_file': self.data_dir / pdb_id / f"{pdb_id}_ligand.sdf",
                    })
        
        return data_list
    
    def _validate_files(self):
        """Validate that required files exist AND are readable."""
        valid_indices = []
        from Bio.PDB import PDBParser
        from rdkit import Chem
        
        pdb_parser = PDBParser(QUIET=True)
        
        for idx, item in enumerate(self.data_list):
            protein_exists = item['protein_file'].exists()
            ligand_exists = item['ligand_file'].exists()
            
            if not (protein_exists and ligand_exists):
                logger.warning(f"Missing files for {item['pdb_id']}")
                continue
            
            # Validate PDB file readability
            try:
                pdb_parser.get_structure('test', str(item['protein_file']))
            except Exception as e:
                logger.warning(f"Cannot parse PDB {item['pdb_id']}: {e}")
                continue
            
            # Validate SDF file readability
            try:
                supplier = Chem.SDMolSupplier(str(item['ligand_file']))
                mol = next(supplier)
                if mol is None:
                    logger.warning(f"Cannot read ligand from {item['pdb_id']}")
                    continue
            except Exception as e:
                logger.warning(f"Cannot parse SDF {item['pdb_id']}: {e}")
                continue
            
            # All checks passed
            valid_indices.append(idx)
        
        # Keep only valid entries
        self.data_list = [self.data_list[i] for i in valid_indices]
        logger.info(f"Validated {len(self.data_list)} complexes with complete files")
    
    def _compute_target_stats(self):
        affinities = [item['affinity'] for item in self.data_list]
        if affinities:
            self.target_mean = float(np.mean(affinities))
            std = float(np.std(affinities)) if len(affinities) > 1 else 1.0
            self.target_std = std if std > 0 else 1.0
            logger.info(f"Target stats: mean={self.target_mean:.3f}, std={self.target_std:.3f}")
        else:
            self.target_mean = 0.0
            self.target_std = 1.0
    
    def get_target_stats(self) -> Dict[str, float]:
        """Return target statistics for sharing across dataset splits."""
        return {'mean': self.target_mean, 'std': self.target_std}
    
    def _get_cache_path(self, pdb_id: str) -> Path:
        """Get cache file path for a PDB ID."""
        return self.cache_dir / f"{pdb_id}.pt"
    
    def _process_complex(self, item: Dict) -> Data:
        """
        Process a protein-ligand complex into a PyG Data object.
        
        Args:
            item: Dictionary with file paths and metadata
            
        Returns:
            PyG Data object with complex graph and target affinity
        """
        try:
            # Featurize complex
            complex_data = featurize_complex(
                protein_pdb=str(item['protein_file']),
                ligand_sdf=str(item['ligand_file']),
                binding_pocket_only=self.binding_pocket_only,
                pocket_cutoff=self.pocket_cutoff
            )
            
            # Construct ligand graph
            ligand_graph = construct_ligand_graph(
                atom_features=complex_data['ligand']['atom_features'],
                atom_coords=complex_data['ligand']['coords'],
                bond_indices=complex_data['ligand']['bonds'],
                bond_features=complex_data['ligand']['bond_features']
            )
            
            # Construct protein graph
            protein_graph = construct_protein_graph(
                residue_features=complex_data['protein']['residue_features'],
                residue_coords=complex_data['protein']['coords'],
                method='knn',
                k=10
            )
            
            # Check for empty protein graph (edge case)
            if protein_graph.num_nodes == 0:
                logger.warning(f"Empty protein graph for {item['pdb_id']}, skipping")
                return None
            
            # Construct complex graph
            graph = construct_complex_graph(
                ligand_data=ligand_graph,
                protein_data=protein_graph,
                interaction_cutoff=self.interaction_cutoff,
                use_heterogeneous=False
            )
            
            # Log if no interactions found
            if graph.edge_index.size(1) == 0:
                logger.warning(f"No edges in complex graph for {item['pdb_id']}")
            
            # Add target affinity
            graph.y = torch.tensor([item['affinity']], dtype=torch.float)
            graph.pdb_id = item['pdb_id']
            
            return graph
            
        except Exception as e:
            logger.error(f"Error processing {item['pdb_id']}: {str(e)}")
            return None
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a protein-ligand complex graph.
        
        Args:
            idx: Index of the complex
            
        Returns:
            PyG Data object with complex graph and affinity
        """
        item = self.data_list[idx]
        pdb_id = item['pdb_id']
        
        # Try to load from cache
        if self.use_cache and self.cache_dir:
            cache_path = self._get_cache_path(pdb_id)
            
            if cache_path.exists():
                try:
                    graph = torch.load(cache_path, weights_only=False)  # PyG Data objects need pickle
                    if self.transform:
                        graph = self.transform(graph)
                    return graph
                except Exception as e:
                    logger.warning(f"Failed to load cache for {pdb_id}: {e}")
        
        # Process complex
        graph = self._process_complex(item)
        
        # Skip if processing failed
        if graph is None:
            # Return a dummy graph with NaN label to skip in batch
            # Or raise error - choose based on preference
            raise RuntimeError(f"Failed to process {pdb_id}, check logs for details")
        
        # Save to cache
        if self.use_cache and self.cache_dir:
            try:
                torch.save(graph, self._get_cache_path(pdb_id))
            except Exception as e:
                logger.warning(f"Failed to cache {pdb_id}: {e}")
        
        # Apply transform
        if self.transform:
            graph = self.transform(graph)
        
        return graph
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        affinities = [item['affinity'] for item in self.data_list]
        
        return {
            'num_complexes': len(self.data_list),
            'affinity_mean': np.mean(affinities),
            'affinity_std': np.std(affinities),
            'affinity_min': np.min(affinities),
            'affinity_max': np.max(affinities),
        }


class InMemoryProteinLigandDataset(Dataset):
    """
    In-memory dataset for faster training on small datasets.
    
    Loads all complexes into memory at initialization.
    Use only if dataset fits in RAM.
    """
    
    def __init__(
        self,
        data_dir: str,
        index_file: str,
        binding_pocket_only: bool = True,
        pocket_cutoff: float = 10.0,
        interaction_cutoff: float = 5.0,
        transform: Optional[Callable] = None
    ):
        self.transform = transform
        
        # Create temporary dataset to process all complexes
        temp_dataset = ProteinLigandDataset(
            data_dir=data_dir,
            index_file=index_file,
            binding_pocket_only=binding_pocket_only,
            pocket_cutoff=pocket_cutoff,
            interaction_cutoff=interaction_cutoff,
            use_cache=False
        )
        
        # Load all complexes into memory
        logger.info("Loading all complexes into memory...")
        self.data_list = []
        
        for idx in range(len(temp_dataset)):
            try:
                graph = temp_dataset[idx]
                self.data_list.append(graph)
            except Exception as e:
                logger.error(f"Failed to load complex {idx}: {e}")
        
        logger.info(f"Loaded {len(self.data_list)} complexes into memory")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Data:
        graph = self.data_list[idx]
        
        if self.transform:
            graph = self.transform(graph)
        
        return graph


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset module...")
    
    # This would require actual PDBBind data
    # For now, just verify the class loads
    print("✓ Dataset class defined successfully")
    print("✓ InMemoryDataset class defined successfully")
    print("\n✅ Module validated!")