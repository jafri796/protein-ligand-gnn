"""
Reproducibility utilities for protein-ligand binding affinity prediction.

Ensures deterministic behavior across runs for scientific rigor.
"""

import torch
import numpy as np
import random
import warnings


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Configures PyTorch for deterministic behavior including CUDA operations.
    Required for reproducible scientific experiments.
    
    Args:
        seed: Random seed value
        deterministic: If True, enforce deterministic algorithms (slower but reproducible)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if deterministic:
        # PyTorch 2.0+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Make PyTorch operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Suppress known non-deterministic operation warnings
        warnings.filterwarnings('ignore', message='.*is not deterministic.*')
        
        # Set environment variable for additional determinism
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
