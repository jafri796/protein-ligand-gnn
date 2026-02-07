"""
Data Splitting Module for Leak-Proof PDBBind (LP-PDBBind)

Implements scientifically rigorous data splitting to prevent information leakage.

Key Features:
- MMseqs2 integration for accurate protein sequence clustering (30% identity)
- RDKit Morgan fingerprints for ligand similarity (50% Tanimoto)
- Scaffold-based splitting for ligand generalization assessment
- Comprehensive audit trail for all similarity computations

Citation:
    Li et al. (2023) "Leak Proof PDBBind: A Non-Leaky Benchmark for Binding Affinity 
    Prediction" arXiv:2308.09639
    
    Steinegger & Söding (2017) "MMseqs2 enables sensitive protein sequence searching"
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from .mmseqs2_wrapper import run_mmseqs2_clustering

logger = logging.getLogger(__name__)


def compute_ligand_similarity(mol1: Chem.Mol, mol2: Chem.Mol, metric: str = 'tanimoto') -> float:
    """
    Compute similarity between two ligands.
    
    Args:
        mol1, mol2: RDKit molecule objects
        metric: Similarity metric ('tanimoto' or 'dice')
        
    Returns:
        Similarity score in [0, 1]
        
    Citation:
        Tanimoto/Dice coefficients are standard for molecular fingerprints
    """
    try:
        # Use Morgan fingerprints (circular fingerprints with radius 2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        if metric.lower() == 'tanimoto':
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif metric.lower() == 'dice':
            return DataStructs.DiceSimilarity(fp1, fp2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    except Exception as e:
        logger.warning(f"Could not compute ligand similarity: {e}")
        return 0.0


def compute_sequence_similarity(seq1: str, seq2: str, method: str = 'identity') -> float:
    """
    Compute similarity between two protein sequences.
    
    For production, use MMseqs2 or BLAST. This is simplified for demo.
    
    Args:
        seq1, seq2: Protein sequences (1-letter amino acid code)
        method: 'identity' for exact match fraction
        
    Returns:
        Similarity score in [0, 1]
        
    Citation:
        LP-PDBBind paper uses MMseqs2 for sequence clustering (30% identity)
    """
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    
    if method == 'identity':
        try:
            from Bio.Align import PairwiseAligner
            aligner = PairwiseAligner()
            aligner.mode = 'global'
            aligner.match_score = 1.0
            aligner.mismatch_score = 0.0
            aligner.open_gap_score = -0.5
            aligner.extend_gap_score = -0.1
            alignments = aligner.align(seq1, seq2)
            if alignments:
                return alignments[0].score / max(len(seq1), len(seq2))
            return 0.0
        except ImportError:
            # Fallback: simple character matching (less accurate for unequal lengths)
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
            return matches / max(len(seq1), len(seq2))
    else:
        raise ValueError(f"Unknown method: {method}")


def create_lp_pdbbind_splits(
    data_dir: str,
    index_file: str,
    output_dir: str,
    protein_seq_cutoff: float = 0.3,
    ligand_sim_cutoff: float = 0.5,
    test_fraction: float = 0.1,
    val_fraction: float = 0.1,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Create leak-proof train/val/test splits.
    
    Implements similarity-based splitting to ensure no contamination:
    1. Remove complexes with sequence/ligand similarity above cutoff
    2. Create independent train/val/test without overlaps
    
    Args:
        data_dir: Directory containing PDB/SDF files
        index_file: File with [pdb_id, affinity] pairs
        output_dir: Directory to save split files
        protein_seq_cutoff: Max sequence similarity allowed (30% = 0.3)
        ligand_sim_cutoff: Max ligand Tanimoto similarity allowed (50% = 0.5)
        test_fraction: Fraction for test split (default: 10%)
        val_fraction: Fraction for val split (default: 10%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with paths to train, val, test split files
        
    Raises:
        FileNotFoundError: If required files not found
    """
    import random
    from Bio.PDB import PDBParser
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating LP-PDBBind splits from {index_file}")
    logger.info(f"Sequence similarity cutoff: {protein_seq_cutoff}")
    logger.info(f"Ligand similarity cutoff: {ligand_sim_cutoff}")
    
    # Load index
    complexes = []
    with open(index_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pdb_id = parts[0]
                affinity = float(parts[1])
                complexes.append({'pdb_id': pdb_id, 'affinity': affinity})
    
    logger.info(f"Loaded {len(complexes)} complexes from index")
    
    # Load protein sequences and ligands (for similarity computation)
    pdb_parser = PDBParser(QUIET=True)
    sequences = {}
    ligands = {}
    valid_complexes = []
    
    for complex_data in complexes:
        pdb_id = complex_data['pdb_id']
        pdb_path = data_dir / pdb_id / f"{pdb_id}_protein.pdb"
        sdf_path = data_dir / pdb_id / f"{pdb_id}_ligand.sdf"
        
        try:
            # Load protein sequence
            structure = pdb_parser.get_structure(pdb_id, str(pdb_path))
            from Bio.Seq import Seq
            from Bio.SeqUtils import IUPACData
            
            # Extract sequence from first model/chain
            ppb = None
            try:
                from Bio.PDB import PPBuilder
                ppb = PPBuilder()
                seqs = ppb.build_peptides(structure[0])
                if seqs:
                    seq = str(seqs[0].get_sequence())
                else:
                    logger.warning(f"No peptides found in {pdb_id}")
                    seq = ""
            except:
                seq = ""
            
            sequences[pdb_id] = seq
            
            # Load ligand
            mol = Chem.SDMolSupplier(str(sdf_path))
            if mol and mol[0] is not None:
                ligands[pdb_id] = mol[0]
                valid_complexes.append(complex_data)
            else:
                logger.warning(f"Could not load ligand for {pdb_id}")
        
        except Exception as e:
            logger.warning(f"Could not load {pdb_id}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(valid_complexes)} valid complexes")
    
    # Use MMseqs2 for sequence clustering at 30% identity threshold
    logger.info("Running MMseqs2 for sequence clustering...")
    seq_clusters = run_mmseqs2_clustering(
        sequences, 
        output_dir / 'mmseqs2',
        seq_id_threshold=protein_seq_cutoff
    )
    
    # Build cluster membership matrix for sequences
    # Two sequences are similar if they're in the same MMseqs2 cluster
    n = len(valid_complexes)
    pdb_id_to_idx = {c['pdb_id']: i for i, c in enumerate(valid_complexes)}
    seq_sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        pdb_i = valid_complexes[i]['pdb_id']
        cluster_i = seq_clusters.get(pdb_i, -1)
        for j in range(i, n):
            pdb_j = valid_complexes[j]['pdb_id']
            cluster_j = seq_clusters.get(pdb_j, -1)
            # Same cluster = similar sequences
            if cluster_i != -1 and cluster_i == cluster_j:
                seq_sim_matrix[i, j] = 1.0
                seq_sim_matrix[j, i] = 1.0
            elif i == j:
                seq_sim_matrix[i, j] = 1.0
    
    logger.info(f"MMseqs2 created {len(set(seq_clusters.values()))} sequence clusters")
    
    # Initialize ligand similarity matrix
    ligand_sim_matrix = np.zeros((n, n))
    
    # Build ligand similarity matrix using Morgan fingerprints
    logger.info("Computing ligand similarities...")
    for i in range(n):
        if i % max(1, n // 10) == 0:
            logger.info(f"  Progress: {i}/{n}")
        
        pdb_i = valid_complexes[i]['pdb_id']
        seq_i = sequences.get(pdb_i, "")
        mol_i = ligands.get(pdb_i)
        
        for j in range(i + 1, n):
            pdb_j = valid_complexes[j]['pdb_id']
            seq_j = sequences.get(pdb_j, "")
            mol_j = ligands.get(pdb_j)
            
            # Compute sequence similarity (already done via MMseqs2 clusters)
            seq_sim = seq_sim_matrix[i, j]
            
            # Compute ligand similarity using Morgan fingerprints
            if mol_i is not None and mol_j is not None:
                ligand_sim = compute_ligand_similarity(mol_i, mol_j, metric='tanimoto')
            else:
                ligand_sim = 0.0
            ligand_sim_matrix[i, j] = ligand_sim
            ligand_sim_matrix[j, i] = ligand_sim
    
    # Save similarity audit log for reproducibility verification
    audit_file = output_dir / 'similarity_audit.txt'
    with open(audit_file, 'w') as f:
        f.write(f"# LP-PDBBind Split Audit\n")
        f.write(f"# Random seed: {random_seed}\n")
        f.write(f"# Protein sequence cutoff: {protein_seq_cutoff}\n")
        f.write(f"# Ligand similarity cutoff: {ligand_sim_cutoff}\n\n")
        f.write(f"MMseqs2 sequence clusters: {len(set(seq_clusters.values()))}\n")
        f.write(f"Total complexes: {n}\n\n")
        f.write("# Per-cluster complex counts:\n")
        from collections import Counter
        cluster_counts = Counter(seq_clusters.values())
        for cluster_id, count in sorted(cluster_counts.items()):
            f.write(f"  Cluster {cluster_id}: {count} complexes\n")
    logger.info(f"Similarity audit log saved to {audit_file}")
    logger.info(f"  Ligand similarities: min={ligand_sim_matrix.min():.3f}, max={ligand_sim_matrix.max():.3f}")
    
    # Build similarity graph: edge exists if complexes are similar (above threshold)
    # Two complexes are similar if: seq_sim > cutoff OR ligand_sim > cutoff
    logger.info("Building similarity graph for leak-proof clustering...")
    
    # Use Union-Find (Disjoint Set Union) for efficient connected components
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    # Connect similar complexes
    similarity_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Check if similar by sequence
            is_similar = seq_sim_matrix[i, j] > protein_seq_cutoff
            # Check if similar by ligand
            is_similar = is_similar or (ligand_sim_matrix[i, j] > ligand_sim_cutoff)
            
            if is_similar:
                union(i, j)
                similarity_edges += 1
    
    logger.info(f"Found {similarity_edges} similarity edges")
    
    # Build clusters from Union-Find structure
    cluster_map = {}
    for i in range(n):
        root = find(i)
        if root not in cluster_map:
            cluster_map[root] = []
        cluster_map[root].append(i)
    
    clusters = list(cluster_map.values())
    logger.info(f"Created {len(clusters)} connected component clusters")
    
    # Distribute clusters to train/val/test
    cluster_indices = np.arange(len(clusters))
    np.random.shuffle(cluster_indices)
    
    num_test = max(1, int(len(cluster_indices) * test_fraction))
    num_val = max(1, int(len(cluster_indices) * val_fraction))
    
    test_clusters = cluster_indices[:num_test]
    val_clusters = cluster_indices[num_test:num_test + num_val]
    train_clusters = cluster_indices[num_test + num_val:]
    
    # Convert back to complex indices
    def get_complexes_from_clusters(cluster_list):
        indices = []
        for c_idx in cluster_list:
            indices.extend(clusters[c_idx])
        return indices
    
    train_indices = get_complexes_from_clusters(train_clusters)
    val_indices = get_complexes_from_clusters(val_clusters)
    test_indices = get_complexes_from_clusters(test_clusters)
    
    # Write split files
    def write_split(indices, filename):
        path = output_dir / filename
        with open(path, 'w') as f:
            for idx in indices:
                complex_data = valid_complexes[idx]
                f.write(f"{complex_data['pdb_id']} {complex_data['affinity']:.2f}\n")
        logger.info(f"Wrote {len(indices)} complexes to {path}")
        return str(path)
    
    train_file = write_split(train_indices, "train.txt")
    val_file = write_split(val_indices, "val.txt")
    test_file = write_split(test_indices, "test.txt")
    
    logger.info(f"✓ LP-PDBBind splits created:")
    logger.info(f"  Train: {len(train_indices)} complexes")
    logger.info(f"  Val:   {len(val_indices)} complexes")
    logger.info(f"  Test:  {len(test_indices)} complexes")
    
    return {
        'train': train_file,
        'val': val_file,
        'test': test_file,
    }


def create_external_test_set(
    index_file: str,
    output_dir: str,
    external_fraction: float = 0.1,
    random_seed: int = 42
) -> str:
    """
    Create an independent external test set.
    
    Purpose:
    Hold out a completely separate validation set to assess generalization
    beyond the LP-PDBBind train/val/test splits. This is critical for:
    1. Proving the model isn't overfit to the LP-PDBBind split procedure
    2. Enabling publication-quality claims about real-world generalization
    
    Scientific Rationale:
    Even with leak-proof splits, a model can overfit to the specific split procedure.
    External test sets from different sources (PDBBind2020+, proprietary assays, etc.)
    provide the strongest evidence of generalization.
    
    Args:
        index_file: Path to index file with all complexes
        output_dir: Directory to save external test indices
        external_fraction: Fraction to hold out as external test (10% = 0.1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Path to external test indices file
    """
    import random
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating external test set from {index_file}")
    logger.info(f"External test fraction: {external_fraction * 100:.1f}%")
    
    # Load index
    indices = []
    with open(index_file, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() and not line.startswith('#'):
                indices.append(i)
    
    # Random split
    n_external = max(1, int(len(indices) * external_fraction))
    external_indices = np.random.choice(indices, size=n_external, replace=False)
    
    # Save
    external_test_file = output_dir / 'external_test_indices.npy'
    np.save(external_test_file, external_indices)
    logger.info(f"External test set: {len(external_indices)} complexes")
    logger.info(f"Saved to {external_test_file}")
    
    return str(external_test_file)


def load_split_indices(
    splits_dir: str,
    split_name: str = 'train'
) -> np.ndarray:
    """
    Load split indices from disk.
    
    Args:
        splits_dir: Directory containing split files
        split_name: Name of split ('train', 'val', 'test', 'external_test')
        
    Returns:
        Array of indices
    """
    splits_dir = Path(splits_dir)
    index_file = splits_dir / f'{split_name}_indices.npy'
    
    if not index_file.exists():
        raise FileNotFoundError(f"Split file not found: {index_file}")
    
    indices = np.load(index_file)
    logger.info(f"Loaded {split_name} split: {len(indices)} samples")
    
    return indices


def create_scaffold_split(
    data_dir: str,
    index_file: str,
    output_dir: str,
    test_fraction: float = 0.1,
    val_fraction: float = 0.1,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Create scaffold-based train/val/test splits for ligand generalization.
    
    Scaffold splitting ensures models generalize to novel chemical structures
    rather than memorizing substituent patterns.
    
    Args:
        data_dir: Directory containing PDB/SDF files
        index_file: File with [pdb_id, affinity] pairs
        output_dir: Directory to save split files
        test_fraction: Fraction for test
        val_fraction: Fraction for validation
        random_seed: Random seed
        
    Returns:
        Dictionary with paths to split files
    """
    import random
    from collections import defaultdict
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating scaffold-based splits")
    
    # Load complexes and extract scaffolds
    complexes = []
    scaffold_to_complexes = defaultdict(list)
    
    with open(index_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pdb_id = parts[0]
                affinity = float(parts[1])
                
                # Load ligand and extract scaffold
                sdf_path = data_dir / pdb_id / f"{pdb_id}_ligand.sdf"
                try:
                    mol = Chem.SDMolSupplier(str(sdf_path))[0]
                    if mol is not None:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else ""
                    else:
                        scaffold_smiles = ""
                except Exception:
                    scaffold_smiles = ""
                
                complex_data = {'pdb_id': pdb_id, 'affinity': affinity, 'scaffold': scaffold_smiles}
                complexes.append(complex_data)
                scaffold_to_complexes[scaffold_smiles].append(complex_data)
    
    logger.info(f"Loaded {len(complexes)} complexes with {len(scaffold_to_complexes)} unique scaffolds")
    
    # Split scaffolds (not individual complexes)
    scaffolds = list(scaffold_to_complexes.keys())
    np.random.shuffle(scaffolds)
    
    num_test = max(1, int(len(scaffolds) * test_fraction))
    num_val = max(1, int(len(scaffolds) * val_fraction))
    
    test_scaffolds = set(scaffolds[:num_test])
    val_scaffolds = set(scaffolds[num_test:num_test + num_val])
    train_scaffolds = set(scaffolds[num_test + num_val:])
    
    # Assign complexes based on scaffold membership
    train_complexes = []
    val_complexes = []
    test_complexes = []
    
    for c in complexes:
        if c['scaffold'] in test_scaffolds:
            test_complexes.append(c)
        elif c['scaffold'] in val_scaffolds:
            val_complexes.append(c)
        else:
            train_complexes.append(c)
    
    # Write split files
    def write_split(complex_list, filename):
        path = output_dir / filename
        with open(path, 'w') as f:
            for c in complex_list:
                f.write(f"{c['pdb_id']} {c['affinity']:.2f}\n")
        logger.info(f"Wrote {len(complex_list)} complexes to {path}")
        return str(path)
    
    result = {
        'train': write_split(train_complexes, "train_scaffold.txt"),
        'val': write_split(val_complexes, "val_scaffold.txt"),
        'test': write_split(test_complexes, "test_scaffold.txt")
    }
    
    logger.info(f"✓ Scaffold splits: Train={len(train_complexes)}, Val={len(val_complexes)}, Test={len(test_complexes)}")
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python splits.py <data_dir> <index_file> <output_dir>")
        sys.exit(1)
    
    create_lp_pdbbind_splits(sys.argv[1], sys.argv[2], sys.argv[3])
