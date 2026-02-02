"""
Hyperparameter optimization using Optuna.

Performs systematic search for optimal hyperparameters of the PaiNN model
using Optuna's TPE sampler with pruning strategies.

Run: python experiments/hpo.py --config config/painn_config.yaml --n-trials 100
"""

import argparse
import copy
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch.utils.tensorboard as tb

from data.dataset import ProteinLigandDataset
from models.painn_affinity import PaiNNAffinityPredictor

# Alias for convenience
PaiNNAffinity = PaiNNAffinityPredictor
from utils import ConfigLoader, set_seed, validate_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PaiNNObjective:
    """Objective function for Optuna optimization."""
    
    def __init__(self, config: Dict[str, Any], data_path: str, 
                 trial_dir: Path, n_trials_per_config: int = 2):
        self.config = config
        self.data_path = Path(data_path)
        self.trial_dir = trial_dir
        self.n_trials_per_config = n_trials_per_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __call__(self, trial: Trial) -> float:
        """Objective function to minimize."""
        
        # Suggest hyperparameters
        config = self.suggest_hyperparameters(trial)
        
        # Setup
        set_seed(config['reproducibility']['seed'])
        trial_name = f"trial_{trial.number:04d}"
        trial_path = self.trial_dir / trial_name
        trial_path.mkdir(exist_ok=True)
        
        # Load data
        try:
            # Use training split index file
            train_split_file = self.data_path.parent / 'splits' / 'train.txt'
            if not train_split_file.exists():
                logger.error(f"Trial {trial.number}: Train split file not found at {train_split_file}")
                return float('inf')
            
            dataset = ProteinLigandDataset(
                data_dir=str(self.data_path),
                index_file=str(train_split_file),
                cache_dir=str(Path(trial_path) / 'cache'),
                binding_pocket_only=self.config['data'].get('binding_pocket_only', True),
                pocket_cutoff=self.config['data'].get('pocket_cutoff', 10.0),
                interaction_cutoff=self.config['data'].get('interaction_cutoff', 5.0),
            )
            
            if len(dataset) < 10:
                logger.warning(f"Trial {trial.number}: Dataset too small ({len(dataset)})")
                return float('inf')
            
            # Use PyG DataLoader
            from torch_geometric.loader import DataLoader as PyGDataLoader
            dataloader = PyGDataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=min(2, config['training'].get('num_workers', 2))
            )
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed to load data: {e}")
            return float('inf')
        
        # Build model
        try:
            model = PaiNNAffinity(config['model']).to(self.device)
        except Exception as e:
            logger.error(f"Trial {trial.number}: Failed to build model: {e}")
            return float('inf')
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Training loop - limited epochs for HPO
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 3
        
        for epoch in range(config['training']['max_epochs'] // 3):  # Reduce epochs for speed
            epoch_loss = 0.0
            num_batches = 0
            
            try:
                for batch in dataloader:
                    # Move batch to device (PyG batch object)
                    batch = batch.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    pred = model(batch)
                    
                    # Loss (use batch.y which contains affinities)
                    targets = batch.y.view(-1, 1) if hasattr(batch, 'y') else batch.affinity.view(-1, 1)
                    loss = torch.nn.functional.mse_loss(
                        pred.view(-1),
                        targets.view(-1)
                    )
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
            except Exception as e:
                logger.error(f"Trial {trial.number} Epoch {epoch}: {e}")
                return float('inf')
            
            avg_loss = epoch_loss / max(num_batches, 1)
            
            # Report to Optuna
            trial.report(avg_loss, epoch)
            
            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
        
        logger.info(f"Trial {trial.number}: Best loss = {best_loss:.6f}")
        
        return best_loss
    
    def suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters to try."""
        
        # Start with base config (deep copy to avoid modifying original)
        config = copy.deepcopy(self.config)
        
        # Model hyperparameters (match actual config keys)
        config['model']['hidden_dim'] = trial.suggest_categorical(
            'hidden_dim', [64, 96, 128, 160]
        )
        config['model']['num_message_passing_layers'] = trial.suggest_int(
            'num_message_passing_layers', 3, 6
        )
        config['model']['num_protein_layers'] = trial.suggest_int(
            'num_protein_layers', 2, 4
        )
        config['model']['dropout'] = trial.suggest_float(
            'dropout', 0.1, 0.5, step=0.1
        )
        
        # Training hyperparameters
        config['training']['batch_size'] = trial.suggest_categorical(
            'batch_size', [16, 32, 64]
        )
        config['training']['learning_rate'] = trial.suggest_loguniform(
            'learning_rate', 1e-5, 1e-3
        )
        config['training']['weight_decay'] = trial.suggest_loguniform(
            'weight_decay', 1e-7, 1e-4
        )
        
        return config


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for PaiNN')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='experiments/hpo_results',
                       help='Output directory for HPO results')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of trials to run')
    parser.add_argument('--timeout', type=float, default=3600.0,
                       help='Timeout per trial in seconds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load config using static method
    config = ConfigLoader.load(args.config)
    
    # Validate config - use schema-based validation
    try:
        from utils import MODEL_SCHEMA
        validate_config(config.get('model', {}), MODEL_SCHEMA)
    except Exception as e:
        logger.warning(f"Config validation warning: {e}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create objective
    objective = PaiNNObjective(
        config=config,
        data_path=args.data_dir,
        trial_dir=output_path,
        n_trials_per_config=2
    )
    
    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5)
    
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction='minimize'
    )
    
    # Optimize
    logger.info(f"Starting HPO with {args.n_trials} trials...")
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
            n_jobs=1  # Single job for reproducibility
        )
    except KeyboardInterrupt:
        logger.info("HPO interrupted by user")
    
    # Results
    logger.info(f"Optimization complete. Best value: {study.best_value:.6f}")
    logger.info(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save results
    results_file = output_path / 'hpo_results.yaml'
    with open(results_file, 'w') as f:
        yaml.dump({
            'best_loss': float(study.best_value),
            'best_hyperparameters': study.best_params,
            'n_trials': len(study.trials),
            'complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        }, f)
    
    logger.info(f"Results saved to {results_file}")
    
    # Save trial dataframe
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_path / 'trials.csv', index=False)
    logger.info(f"Trials data saved to trials.csv")


if __name__ == '__main__':
    main()
