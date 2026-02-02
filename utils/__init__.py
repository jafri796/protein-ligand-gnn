"""
Configuration and utilities for reproducibility and validation.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, disable CUDA benchmarking for determinism
        
    Citation:
        PyTorch reproducibility best practices
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"CUDA reproducibility: deterministic={deterministic}")
    
    logger.info(f"Random seed set to {seed}")


def validate_config(config: Dict[str, Any], required_keys: Dict[str, type]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: Dict mapping key names to expected types
        
    Returns:
        True if valid, raises ValueError otherwise
        
    Example:
        required = {'model': dict, 'training': dict, 'batch_size': int}
        validate_config(config, required)
    """
    for key, expected_type in required_keys.items():
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
        
        if not isinstance(config[key], expected_type):
            raise ValueError(
                f"Config key '{key}' has type {type(config[key])}, "
                f"expected {expected_type}"
            )
    
    return True


def create_directories(paths: list) -> None:
    """Create necessary directories."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


class ConfigLoader:
    """Load and validate YAML configuration files."""
    
    @staticmethod
    def load(config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config YAML file
            
        Returns:
            Parsed configuration dictionary
        """
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_path}")
        return config
    
    @staticmethod
    def save(config: Dict, output_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        with open(output_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        logger.info(f"Saved config to {output_path}")


# Common validation schemas
MODEL_SCHEMA = {
    'hidden_dim': int,
    'num_message_passing_layers': int,
    'num_rbf': int,
    'cutoff': float,
}

TRAINING_SCHEMA = {
    'batch_size': int,
    'learning_rate': float,
    'num_epochs': int,
    'early_stopping_patience': int,
}

DATA_SCHEMA = {
    'data_dir': str,
    'train_split': str,
    'val_split': str,
    'binding_pocket_only': bool,
    'pocket_cutoff': float,
}
