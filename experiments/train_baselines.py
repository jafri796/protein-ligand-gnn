"""
Baseline model training and evaluation on LP-PDBBind splits.

Implements rigorous baseline comparisons:
1. GraphDTA (Nguyen et al. 2021) - sequence-based 2D baseline
2. Random Forest - classical ML with hand-crafted features  
3. Linear Regression - minimal model

All models evaluated on the same LP-PDBBind train/val/test splits as PaiNN.
This enables rigorous comparison and validates that PaiNN performance gains
are real, not artifacts of split design or dataset selection.

Scientific Rationale:
- GraphDTA: Sequence-based, ignores 3D structure - validates value of geometry
- Random Forest: Classical ML baseline with engineered features - establishes lower bound
- Linear: Minimal model - represents zero-learning baseline

Usage:
  python experiments/train_baselines.py --config config/painn_config.yaml --baseline graphdta
  python experiments/train_baselines.py --config config/painn_config.yaml --baseline random_forest
  python experiments/train_baselines.py --config config/painn_config.yaml --baseline all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import ProteinLigandDataset
from data.splits import create_lp_pdbbind_splits
from models.baselines import GraphDTA, GraphDTAFeatureExtractor
from utils import ConfigLoader, set_seed, validate_config

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Compute and report regression metrics."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict:
        """Compute standard regression metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Correlation metrics with error handling
        try:
            pearson_r, _ = pearsonr(y_true, y_pred)
        except (ValueError, FloatingPointError):
            pearson_r = np.nan
        
        try:
            spearman_r, _ = spearmanr(y_true, y_pred)
        except (ValueError, FloatingPointError):
            spearman_r = np.nan
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        results = {
            'model': model_name,
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else None,
            'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else None,
            'r2_score': float(r2_score),
            'n_samples': len(y_true),
        }
        
        return results
    
    @staticmethod
    def log_results(results: Dict):
        """Log metrics in a readable format."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Model: {results['model']}")
        logger.info(f"{'='*80}")
        logger.info(f"RMSE:       {results['rmse']:.4f} pKd")
        logger.info(f"MAE:        {results['mae']:.4f} pKd")
        logger.info(f"R²:         {results['r2_score']:.4f}")
        if results['pearson_r']:
            logger.info(f"Pearson r:  {results['pearson_r']:.4f}")
        if results['spearman_r']:
            logger.info(f"Spearman r: {results['spearman_r']:.4f}")
        logger.info(f"N samples:  {results['n_samples']}")


class BaselineTrainer:
    """Train and evaluate baseline models."""
    
    def __init__(self, config: Dict[str, Any], baseline_type: str = 'graphdta'):
        self.config = config
        self.baseline_type = baseline_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup seed for reproducibility
        set_seed(config['reproducibility']['seed'])
        
        # Output directory
        self.output_dir = Path(config['logging']['checkpoint_dir']) / f'baselines_{baseline_type}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Baseline trainer initialized: {baseline_type}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
    
    def load_datasets_with_lp_split(self) -> Tuple[Subset, Subset, Subset]:
        """Load train/val/test datasets using LP-PDBBind splits."""
        logger.info("Loading datasets with LP-PDBBind splits...")
        
        data_config = self.config['data']
        data_dir = Path(data_config['data_dir'])
        
        # Load full dataset
        full_dataset = ProteinLigandDataset(
            data_dir=str(data_dir),
            index_file=str(data_dir / data_config.get('index_file', 'index.txt')),
            cache_dir=data_config.get('cache_dir'),
            binding_pocket_only=data_config.get('binding_pocket_only', True),
            pocket_cutoff=data_config.get('pocket_cutoff', 10.0),
            interaction_cutoff=data_config.get('interaction_cutoff', 5.0),
            use_cache=data_config.get('use_cache', True)
        )
        
        logger.info(f"Full dataset size: {len(full_dataset)}")
        
        # Get LP-PDBBind splits
        splits_dir = Path(data_config.get('splits_dir', 'data/splits'))
        
        if not splits_dir.exists():
            logger.warning(f"Splits directory not found: {splits_dir}. Creating LP-PDBBind splits...")
            create_lp_pdbbind_splits(
                data_dir=str(data_dir),
                index_file=str(data_dir / data_config.get('index_file', 'index.txt')),
                output_dir=str(splits_dir),
                protein_seq_cutoff=data_config.get('protein_seq_cutoff', 0.3),
                ligand_sim_cutoff=data_config.get('ligand_sim_cutoff', 0.5),
            )
        
        # Load split files (text files written by create_lp_pdbbind_splits)
        train_file = splits_dir / 'train.txt'
        val_file = splits_dir / 'val.txt'
        test_file = splits_dir / 'test.txt'
        
        if not all([train_file.exists(), val_file.exists(), test_file.exists()]):
            raise FileNotFoundError(f"Split files not found in {splits_dir}")
        
        def load_split_pdb_ids(filepath):
            """Load PDB IDs from split text file."""
            pdb_ids = set()
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        pdb_ids.add(parts[0])
            return pdb_ids
        
        train_pdb_ids = load_split_pdb_ids(train_file)
        val_pdb_ids = load_split_pdb_ids(val_file)
        test_pdb_ids = load_split_pdb_ids(test_file)
        
        # Map PDB IDs to dataset indices
        def get_indices_for_pdb_ids(dataset, pdb_ids):
            indices = []
            for i, entry in enumerate(dataset.data_list):
                if entry.get('pdb_id', '') in pdb_ids:
                    indices.append(i)
            return indices
        
        train_indices = get_indices_for_pdb_ids(full_dataset, train_pdb_ids)
        val_indices = get_indices_for_pdb_ids(full_dataset, val_pdb_ids)
        test_indices = get_indices_for_pdb_ids(full_dataset, test_pdb_ids)
        
        logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def extract_graph_features(self, dataset: Subset, split_name: str = "dataset") -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from PyG graphs."""
        logger.info(f"Extracting graph features for {split_name}...")
        
        extractor = GraphDTAFeatureExtractor()
        X_list = []
        y_list = []
        
        for idx in tqdm(range(len(dataset)), desc=f"Extracting {split_name}"):
            try:
                graph_data = dataset[idx]
                
                # Extract features
                features = extractor.extract_features(graph_data)
                
                if features is not None:
                    X_list.append(features)
                    y_list.append(float(graph_data.y.item() if hasattr(graph_data.y, 'item') else graph_data.y))
                else:
                    logger.warning(f"Failed to extract features for index {idx}")
                    
            except Exception as e:
                logger.error(f"Error processing index {idx}: {e}")
                continue
        
        if len(X_list) == 0:
            raise RuntimeError(f"No valid samples extracted from {split_name}")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        logger.info(f"Extracted {len(X)} samples, feature shape: {X.shape}")
        
        return X, y
    
    def train_graphdta(self, train_data: Tuple[np.ndarray, np.ndarray],
                      val_data: Tuple[np.ndarray, np.ndarray],
                      test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Train shallow neural network baseline (GraphDTA-style)."""
        logger.info("=" * 80)
        logger.info("Training GraphDTA Baseline (NN on Graph Features)")
        logger.info("=" * 80)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.FloatTensor(y_test)
        )
        
        batch_size = self.config['training'].get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        input_dim = X_train_scaled.shape[1]
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['training']['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
        
        # Training loop
        best_val_loss = float('inf')
        patience = self.config['training'].get('early_stopping_patience', 20)
        patience_counter = 0
        num_epochs = self.config['training'].get('num_epochs', 100)
        
        for epoch in range(num_epochs):
            # Train epoch
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred.squeeze(), y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.output_dir / 'best_graphdta.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Evaluate on test set
        model.load_state_dict(torch.load(self.output_dir / 'best_graphdta.pt', weights_only=True))
        model.eval()
        test_preds = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = model(X_batch).squeeze()
                test_preds.extend(y_pred.cpu().numpy())
        
        test_preds = np.array(test_preds)
        results = BaselineEvaluator.compute_metrics(y_test, test_preds, 'GraphDTA')
        BaselineEvaluator.log_results(results)
        
        return results
    
    def train_random_forest(self, train_data: Tuple[np.ndarray, np.ndarray],
                           val_data: Tuple[np.ndarray, np.ndarray],
                           test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Train Random Forest baseline."""
        logger.info("=" * 80)
        logger.info("Training Random Forest Baseline")
        logger.info("=" * 80)
        
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        logger.info("Training RF (n_estimators=200, max_depth=20)...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config['reproducibility']['seed'],
            n_jobs=-1,
            verbose=0
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_test_pred = rf_model.predict(X_test_scaled)
        
        results = BaselineEvaluator.compute_metrics(y_test, y_test_pred, 'Random Forest')
        BaselineEvaluator.log_results(results)
        
        # Feature importance
        importance = np.argsort(rf_model.feature_importances_)[::-1][:5]
        logger.info(f"Top 5 important features: indices {importance}")
        logger.info(f"Importances: {rf_model.feature_importances_[importance]}")
        
        return results
    
    def train_linear_regression(self, train_data: Tuple[np.ndarray, np.ndarray],
                               val_data: Tuple[np.ndarray, np.ndarray],
                               test_data: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Train Linear Regression baseline."""
        logger.info("=" * 80)
        logger.info("Training Linear Regression Baseline")
        logger.info("=" * 80)
        
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Linear Regression
        logger.info("Fitting linear regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        # Predict
        y_test_pred = lr_model.predict(X_test_scaled)
        
        results = BaselineEvaluator.compute_metrics(y_test, y_test_pred, 'Linear Regression')
        BaselineEvaluator.log_results(results)
        
        return results
    
    def run(self):
        """Run baseline training pipeline."""
        # Load datasets
        train_dataset, val_dataset, test_dataset = self.load_datasets_with_lp_split()
        
        # Extract features
        X_train, y_train = self.extract_graph_features(train_dataset, 'train')
        X_val, y_val = self.extract_graph_features(val_dataset, 'val')
        X_test, y_test = self.extract_graph_features(test_dataset, 'test')
        
        results = {}
        
        # Train baselines
        if self.baseline_type in ['graphdta', 'all']:
            try:
                results['graphdta'] = self.train_graphdta(
                    (X_train, y_train), (X_val, y_val), (X_test, y_test)
                )
            except Exception as e:
                logger.error(f"GraphDTA training failed: {e}")
                import traceback
                traceback.print_exc()
                results['graphdta'] = None
        
        if self.baseline_type in ['random_forest', 'all']:
            try:
                results['random_forest'] = self.train_random_forest(
                    (X_train, y_train), (X_val, y_val), (X_test, y_test)
                )
            except Exception as e:
                logger.error(f"Random Forest training failed: {e}")
                import traceback
                traceback.print_exc()
                results['random_forest'] = None
        
        if self.baseline_type in ['linear_regression', 'all']:
            try:
                results['linear_regression'] = self.train_linear_regression(
                    (X_train, y_train), (X_val, y_val), (X_test, y_test)
                )
            except Exception as e:
                logger.error(f"Linear Regression training failed: {e}")
                import traceback
                traceback.print_exc()
                results['linear_regression'] = None
        
        # Save results
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE COMPARISON SUMMARY")
        logger.info("=" * 80)
        for model, result in results.items():
            if result:
                logger.info(f"{model:25s}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, R²={result['r2_score']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Train baseline models on LP-PDBBind splits")
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--baseline', type=str, default='graphdta',
                       choices=['graphdta', 'random_forest', 'linear_regression', 'all'],
                       help='Which baseline to train')
    
    args = parser.parse_args()
    
    # Load config using static method
    config = ConfigLoader.load(args.config)
    
    # Validate config with schema
    from utils import MODEL_SCHEMA
    try:
        validate_config(config.get('model', {}), MODEL_SCHEMA)
    except Exception as e:
        logger.warning(f"Config validation warning: {e}")
    
    set_seed(config.get('reproducibility', {}).get('seed', 42))
    
    logger.info(f"Config: {args.config}")
    logger.info(f"Baseline(s): {args.baseline}")
    
    try:
        # Train baselines
        trainer = BaselineTrainer(config, baseline_type=args.baseline)
        results = trainer.run()
        return 0
        
    except Exception as e:
        logger.error(f"Error during baseline training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
