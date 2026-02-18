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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import ProteinLigandDataset
from models.painn_affinity import PaiNNAffinityPredictor
from utils import set_seed
from utils.config import load_config

# Configure logging only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
logger = logging.getLogger(__name__)


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
    
    # R² score
    r2 = np.nan
    try:
        if len(np.unique(targets)) > 1:
            r2 = r2_score(targets, predictions)
    except Exception as e:
        logger.warning(f"R² computation failed: {e}")

    return {
        'rmse': rmse,
        'mae': mae,
        'pearson': pearson,
        'spearman': spearman,
        'r2': r2
    }


class Trainer:
    """Training manager with gradient accumulation and distributed training support."""
    
    def __init__(self, model, config, device, gradient_accumulation_steps=1):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step = 0  # Global step counter for gradient accumulation
        
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
        """Train for one epoch with gradient accumulation support."""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        self.optimizer.zero_grad()  # Zero gradients at start of epoch
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            
            # Forward pass
            output = self.model(batch)
            loss = self.criterion(output, batch.y)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # NaN/Inf detection
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss detected: {loss.item()}. Skipping batch.")
                continue
            
            # Backward pass
            loss.backward()
            
            # Track metrics (use unscaled loss for reporting)
            total_loss += (loss.item() * self.gradient_accumulation_steps)
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(batch.y.detach().cpu().numpy())
            
            # Gradient update every N steps (gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.step += 1
                
                # Gradient norm monitoring
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                total_norm = total_norm ** 0.5
                
                if total_norm > 1000:
                    logger.warning(f"Exploding gradients detected: norm={total_norm:.2f}")
                elif total_norm < 1e-6:
                    logger.warning(f"Vanishing gradients detected: norm={total_norm:.2e}")
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Log to TensorBoard
                if self.writer and self.step % 10 == 0:
                    global_step = epoch * len(train_loader) + batch_idx
                    self.writer.add_scalar('train/batch_loss', loss.item() * self.gradient_accumulation_steps, global_step)
                    self.writer.add_scalar('train/gradient_norm', total_norm, global_step)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        # Handle remaining gradients if any
        if len(train_loader) % self.gradient_accumulation_steps != 0:
            if self.config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
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
    """Main training function with comprehensive error handling."""
    parser = argparse.ArgumentParser(description='Train PaiNN affinity model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--split-type', type=str, default='random', choices=['random', 'scaffold', 'lp-pdbbind'],
                       help='Split type: random, scaffold-based, or LP-PDBBind (leak-proof)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Number of gradient accumulation steps (default: 1)')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training with DDP')
    parser.add_argument('--local-rank', type=int, default=0,
                       help='Local rank for distributed training')
    args = parser.parse_args()
    
    # Initialize distributed training if requested
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(device)
        logger.info(f"Distributed training: rank {torch.distributed.get_rank()}/{torch.distributed.get_world_size()}")
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load config
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
        
        # Set seed (different for each rank in distributed mode)
        seed = args.seed + (torch.distributed.get_rank() if args.distributed else 0)
        set_seed(seed, deterministic=True)
        logger.info(f"Set random seed to {seed}")
        
        # Load data
        logger.info(f"Loading datasets (split_type={args.split_type})...")
        try:
            # Determine index files based on split type
            if args.split_type == 'scaffold':
                train_index = config['data'].get('train_scaffold_split', config['data']['train_split'].replace('train.txt', 'train_scaffold.txt'))
                val_index = config['data'].get('val_scaffold_split', config['data']['val_split'].replace('val.txt', 'val_scaffold.txt'))
            elif args.split_type == 'lp-pdbbind':
                train_index = config['data']['train_split']
                val_index = config['data']['val_split']
                # Verify LP-PDBBind splits exist
                if not Path(train_index).exists():
                    logger.error(f"LP-PDBBind split not found: {train_index}")
                    logger.error("Run: python data/splits.py to create LP-PDBBind splits")
                    raise FileNotFoundError(f"LP-PDBBind split not found: {train_index}")
            else:  # random
                train_index = config['data']['train_split']
                val_index = config['data']['val_split']
            
            train_dataset = ProteinLigandDataset(
                data_dir=config['data']['data_dir'],
                index_file=train_index,
                cache_dir=config['data'].get('cache_dir'),
                use_cache=config['data'].get('use_cache', True)
            )
            
            val_dataset = ProteinLigandDataset(
                data_dir=config['data']['data_dir'],
                index_file=val_index,
                cache_dir=config['data'].get('cache_dir'),
                use_cache=config['data'].get('use_cache', True)
            )
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise
        
        # Create data loaders
        try:
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
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            raise
        
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
        
        # Create model
        logger.info("Creating model...")
        try:
            model = PaiNNAffinityPredictor(config['model'])
            logger.info(f"Model parameters: {model.get_num_params():,}")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
        
        # Create trainer with gradient accumulation
        try:
            trainer = Trainer(
                model, 
                config, 
                device,
                gradient_accumulation_steps=args.gradient_accumulation
            )
            logger.info(f"Trainer initialized with gradient accumulation: {args.gradient_accumulation}")
        except Exception as e:
            logger.error(f"Failed to create trainer: {e}")
            raise
        
        # Train
        logger.info("Starting training...")
        try:
            trainer.fit(
                train_loader,
                val_loader,
                num_epochs=config['training']['num_epochs']
            )
            logger.info("Training completed successfully!")
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU out of memory. Try reducing batch size.")
            else:
                logger.error(f"Runtime error during training: {e}")
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()