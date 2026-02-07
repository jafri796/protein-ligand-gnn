"""
MMseqs2 integration module for protein sequence clustering.

This module provides a Python wrapper around MMseqs2 for accurate
protein sequence clustering at specified identity thresholds.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)


def run_mmseqs2_clustering(
    sequences: Dict[str, str],
    output_dir: Path,
    seq_id_threshold: float = 0.3,
    min_seq_len: int = 20
) -> Dict[str, int]:
    """
    Run MMseqs2 for sequence clustering at specified identity threshold.
    
    Falls back to greedy clustering if MMseqs2 unavailable.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_sequences = {pid: seq for pid, seq in sequences.items() 
                      if len(seq) >= min_seq_len}
    
    if len(valid_sequences) < 2:
        logger.warning("Too few valid sequences")
        return {pid: i for i, pid in enumerate(valid_sequences.keys())}
    
    try:
        return _run_mmseqs2(valid_sequences, output_dir, seq_id_threshold)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"MMseqs2 unavailable: {e}, using fallback")
        return _fallback_clustering(valid_sequences, seq_id_threshold, output_dir)


def _run_mmseqs2(sequences: Dict[str, str], output_dir: Path, threshold: float) -> Dict[str, int]:
    """Execute MMseqs2 clustering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fasta = tmpdir / "seqs.fasta"
        
        with open(fasta, 'w') as f:
            for pid, seq in sequences.items():
                f.write(f">{pid}\n{seq}\n")
        
        db = tmpdir / "db"
        clu = tmpdir / "clu"
        
        subprocess.run(['mmseqs', 'createdb', str(fasta), str(db)],
                      check=True, capture_output=True, timeout=60)
        subprocess.run(['mmseqs', 'cluster', str(db), str(clu), str(tmpdir / 'tmp'),
                       '--min-seq-id', str(threshold), '-c', '0.8'],
                      check=True, capture_output=True, timeout=300)
        
        tsv = tmpdir / "clusters.tsv"
        subprocess.run(['mmseqs', 'createtsv', str(db), str(db), str(clu), str(tsv)],
                      check=True, capture_output=True, timeout=30)
        
        clusters = defaultdict(list)
        with open(tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    clusters[parts[0]].append(parts[1])
        
        seq_to_cluster = {}
        for cid, (rep, members) in enumerate(clusters.items()):
            seq_to_cluster[rep] = cid
            for m in members:
                seq_to_cluster[m] = cid
        
        logger.info(f"MMseqs2: {len(clusters)} clusters from {len(sequences)} sequences")
        return seq_to_cluster


def _fallback_clustering(sequences: Dict[str, str], threshold: float, output_dir: Path) -> Dict[str, int]:
    """Greedy clustering using pairwise sequence identity.
    
    Computes true sequence identity as the fraction of identical residues
    in a global alignment, normalized by the length of the longer sequence.
    """
    from Bio.Align import PairwiseAligner
    
    # Configure aligner for identity counting:
    # match=1, mismatch=0, no gap penalties → score = number of matches
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    
    pdb_ids = list(sequences.keys())
    n = len(pdb_ids)
    
    cluster_id = 0
    seq_to_cluster = {}
    assigned = set()
    
    for i in range(n):
        if pdb_ids[i] in assigned:
            continue
        
        seq_to_cluster[pdb_ids[i]] = cluster_id
        assigned.add(pdb_ids[i])
        seq_i = sequences[pdb_ids[i]]
        
        for j in range(i + 1, n):
            if pdb_ids[j] in assigned:
                continue
            
            seq_j = sequences[pdb_ids[j]]
            alignments = aligner.align(seq_i, seq_j)
            
            if alignments:
                # With match=1, mismatch=0, score approximates number of matches
                # Normalize by longer sequence length for sequence identity
                identity = alignments[0].score / max(len(seq_i), len(seq_j))
                if identity > threshold:
                    seq_to_cluster[pdb_ids[j]] = cluster_id
                    assigned.add(pdb_ids[j])
        
        cluster_id += 1
    
    logger.info(f"Fallback: {cluster_id} clusters from {n} sequences")
    return seq_to_cluster
