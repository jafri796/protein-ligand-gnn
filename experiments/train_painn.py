"""
Training Script for PaiNN Affinity Prediction

Complete training loop with:
- Data loading
- Model training
- Validation
- Checkpointing
- Logging
- Early stopping
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import ProteinLigandDataset
from models.painn_affinity import PaiNNAffinityPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    if torch.cuda.is_available():
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"CUDA reproducibility: deterministic={deterministic}")


def compute_metrics(predictions, targets):
    """Compute evaluation metrics with proper error handling."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    # Compute correlation metrics with error handling
    pearson = np.nan
    spearman = np.nan
    
    try:
        if len(np.unique(targets)) > 1:  # Need variance in targets
            pearson, _ = pearsonr(predictions, targets)
        else:
            logger.warning("All target values are identical; Pearson correlation undefined")
    except Exception as e:
        logger.warning(f"Pearson correlation computation failed: {e}")
    
    try:
        if len(np.unique(targets)) > 1:
            spearman, _ = spearmanr(predictions, targets)
        else:
            logger.warning("All target values are identical; Spearman correlation undefined")
    except Exception as e:
        logger.warning(f"Spearman correlation computation failed: {e}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson,
        'spearman': spearman
    }


class Trainer:
    """Training manager."""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['lr_scheduler']['factor'],
            patience=config['training']['lr_scheduler']['patience'],
            min_lr=config['training']['lr_scheduler']['min_lr']
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = config['training']['early_stopping_patience']
        
        # Logging
        self.log_dir = Path(config['logging']['log_dir'])
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if config['logging']['use_tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.criterion(output, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(batch.y.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            if self.writer and batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        # Compute epoch metrics
        metrics = compute_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader, epoch):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = batch.to(self.device)
                
                output = self.model(batch)
                loss = self.criterion(output, batch.y)
                
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model (epoch {epoch})")
    
    def fit(self, train_loader, val_loader, num_epochs):
        """Complete training loop."""
        logger.info("Starting training...")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(
                f"Epoch {epoch} | Train Loss: {train_metrics['loss']:.4f} | "
                f"RMSE: {train_metrics['rmse']:.4f} | "
                f"Pearson: {train_metrics['pearson']:.4f}"
            )
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            logger.info(
                f"Epoch {epoch} | Val Loss: {val_metrics['loss']:.4f} | "
                f"RMSE: {val_metrics['rmse']:.4f} | "
                f"Pearson: {val_metrics['pearson']:.4f}"
            )
            
            # Log to TensorBoard
            if self.writer:
                for key, value in train_metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                self.writer.add_scalar('learning_rate', 
                                      self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Checkpointing
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % self.config['logging']['save_every_n_epochs'] == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info("Training completed!")
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed with deterministic flag from config
    reproducibility_config = config.get('reproducibility', {})
    set_seed(
        reproducibility_config.get('seed', 42),
        deterministic=reproducibility_config.get('deterministic', True)
    )
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = ProteinLigandDataset(
        data_dir=config['data']['data_dir'],
        index_file=config['data']['train_split'],
        cache_dir=config['data']['cache_dir'],
        binding_pocket_only=config['data']['binding_pocket_only'],
        pocket_cutoff=config['data']['pocket_cutoff'],
        interaction_cutoff=config['data']['interaction_cutoff']
    )
    
    val_dataset = ProteinLigandDataset(
        data_dir=config['data']['data_dir'],
        index_file=config['data']['val_split'],
        cache_dir=config['data']['cache_dir'],
        binding_pocket_only=config['data']['binding_pocket_only'],
        pocket_cutoff=config['data']['pocket_cutoff'],
        interaction_cutoff=config['data']['interaction_cutoff']
    )
    
    # Create dataloaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = PaiNNAffinityPredictor(config['model'])
    logger.info(f"Model parameters: {model.get_num_params():,}")
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Train
    trainer.fit(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs']
    )


if __name__ == '__main__':
    main()