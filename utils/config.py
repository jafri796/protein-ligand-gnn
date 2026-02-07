"""
Configuration utilities for protein-ligand binding affinity prediction.

Provides shared config loading, validation, and management across all scripts.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    import yaml
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    import yaml
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved config to {output_path}")


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


# Common validation schemas for PaiNN config
PAINN_CONFIG_SCHEMA = {
    'model': dict,
    'training': dict,
    'data': dict,
}

MODEL_SCHEMA = {
    'hidden_dim': int,
    'num_message_passing_layers': int,
    'num_rbf': int,
    'cutoff': (int, float),
}

TRAINING_SCHEMA = {
    'batch_size': int,
    'learning_rate': (int, float),
    'num_epochs': int,
    'early_stopping_patience': int,
}
