"""
ECG Model Training Script
========================

This script trains 1D CNN and LSTM models on ECG signals for cardiovascular disease prediction.
It includes comprehensive training loop, validation, checkpointing, and evaluation.

Features:
- Multiple ECG model architectures (1D CNN, LSTM, BiLSTM, Transformer, Hybrid)
- Comprehensive loss functions and optimizers
- Advanced metrics and evaluation
- Model checkpointing and early stopping
- Learning rate scheduling
- Data augmentation for ECG signals
- TensorBoard logging
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our modules
from ecg_model_architectures import create_ecg_model, count_parameters, get_model_summary
from ecg_data_loader import ECGDatasetLoader, ECGDataGenerator
from model_utils import ModelEvaluator, ModelAnalyzer, ModelManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ecg_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG signals.
    """
    
    def __init__(self, 
                 signals: List[np.ndarray],
                 labels: List[int],
                 features: Optional[np.ndarray] = None,
                 transform: Optional[callable] = None):
        """
        Initialize the dataset.
        
        Args:
            signals (list): List of ECG signals
            labels (list): List of labels
            features (np.ndarray): Additional features (optional)
            transform (callable): Signal transformations (optional)
        """
        self.signals = signals
        self.labels = labels
        self.features = features
        self.transform = transform
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        label = self.labels[idx]
        
        # Validate signal data
        if np.isnan(signal).any() or np.isinf(signal).any():
            logger.warning(f"NaN or Inf detected in signal at index {idx}, replacing with zeros")
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply transformations if provided
        if self.transform:
            signal = self.transform(signal)
        
        # Convert to tensor
        signal = torch.FloatTensor(signal)
        
        # Validate tensor
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            logger.warning(f"NaN or Inf detected in tensor at index {idx}, replacing with zeros")
            signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.features is not None:
            features = torch.FloatTensor(self.features[idx])
            # Validate features
            if torch.isnan(features).any() or torch.isinf(features).any():
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return signal, label, features
        else:
            return signal, label


class ECGTrainer:
    """
    Comprehensive trainer for ECG models.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 optimizer_config: Dict[str, Any],
                 lr_scheduler_config: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            model_config (dict): Model configuration
            training_config (dict): Training configuration
            optimizer_config (dict): Optimizer configuration
            lr_scheduler_config (dict): Learning rate scheduler configuration
            device (str): Device to use for training
        """
        self.model_config = model_config
        self.training_config = training_config
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.device = device
        
        # Initialize model
        self.model = create_ecg_model(model_config)
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_lr_scheduler()
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'val_auc': []
        }
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter('runs/ecg_training')
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Initialized ECG trainer on device: {device}")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.optimizer_config.get('name', 'adam').lower()
        lr = self.optimizer_config.get('lr', 0.001)
        weight_decay = self.optimizer_config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(self.optimizer_config.get('beta1', 0.9), 
                       self.optimizer_config.get('beta2', 0.999)),
                eps=self.optimizer_config.get('epsilon', 1e-8)
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=self.optimizer_config.get('momentum', 0.9),
                nesterov=self.optimizer_config.get('nesterov', True)
            )
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=self.optimizer_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_name = self.training_config.get('loss_function', 'cross_entropy').lower()
        
        if loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_name == 'focal_loss':
            return FocalLoss(
                alpha=self.training_config.get('focal_alpha', 0.25),
                gamma=self.training_config.get('focal_gamma', 2.0)
            )
        elif loss_name == 'weighted_cross_entropy':
            class_weights = self.training_config.get('class_weights', None)
            if class_weights:
                class_weights = torch.FloatTensor(class_weights).to(self.device)
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def _create_lr_scheduler(self) -> Optional[object]:
        """Create learning rate scheduler based on configuration."""
        scheduler_name = self.lr_scheduler_config.get('name', 'step').lower()
        
        if scheduler_name == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.lr_scheduler_config.get('step_size', 30),
                gamma=self.lr_scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'cosine_annealing':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.lr_scheduler_config.get('T_max', 100),
                eta_min=self.lr_scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.lr_scheduler_config.get('factor', 0.5),
                patience=self.lr_scheduler_config.get('patience', 5),
                min_lr=self.lr_scheduler_config.get('min_lr', 1e-6)
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:
                signals, labels, features = batch_data
                signals, labels, features = signals.to(self.device), labels.to(self.device), features.to(self.device)
            else:
                signals, labels = batch_data
                signals, labels = signals.to(self.device), labels.to(self.device)
                features = None
            
            # Validate input data
            if torch.isnan(signals).any() or torch.isinf(signals).any():
                logger.warning(f"NaN or Inf detected in signals at batch {batch_idx}, skipping batch")
                continue
            
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                logger.warning(f"NaN or Inf detected in labels at batch {batch_idx}, skipping batch")
                continue
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(signals)
            
            # Validate outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.warning(f"NaN or Inf detected in model outputs at batch {batch_idx}")
                # Skip this batch and continue
                continue
            
            loss = self.criterion(outputs, labels)
            
            # Validate loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or Inf loss detected at batch {batch_idx}, skipping batch")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            
            for batch_idx, batch_data in enumerate(pbar):
                if len(batch_data) == 3:
                    signals, labels, features = batch_data
                    signals, labels, features = signals.to(self.device), labels.to(self.device), features.to(self.device)
                else:
                    signals, labels = batch_data
                    signals, labels = signals.to(self.device), labels.to(self.device)
                    features = None
                
                # Forward pass
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
                })
        
        # Calculate metrics
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Calculate AUC
        try:
            epoch_auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except:
            epoch_auc = 0.0
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1,
            'auc': epoch_auc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              save_dir: str = 'ecg_checkpoints') -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
            save_dir (str): Directory to save checkpoints
            
        Returns:
            dict: Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting ECG training for {num_epochs} epochs...")
        logger.info(f"Model: {self.model_config.get('type', 'unknown')}")
        logger.info(f"Optimizer: {self.optimizer_config.get('name', 'unknown')}")
        logger.info(f"Learning rate: {self.optimizer_config.get('lr', 0.001)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.metrics['train_loss'].append(train_metrics['loss'])
            self.metrics['val_loss'].append(val_metrics['loss'])
            self.metrics['train_acc'].append(train_metrics['accuracy'])
            self.metrics['val_acc'].append(val_metrics['accuracy'])
            self.metrics['train_f1'].append(train_metrics['f1'])
            self.metrics['val_f1'].append(val_metrics['f1'])
            self.metrics['val_auc'].append(val_metrics['auc'])
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
            self.writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
            self.writer.add_scalar('AUC/Validation', val_metrics['auc'], epoch)
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                best_model_path = save_path / 'best_ecg_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'model_config': self.model_config,
                    'training_config': self.training_config
                }, best_model_path)
                
                logger.info(f"  New best model saved! Val Acc: {val_metrics['accuracy']:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.training_config.get('checkpoint_frequency', 5) == 0:
                checkpoint_path = save_path / f'ecg_checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_acc': val_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'metrics': self.metrics
                }, checkpoint_path)
            
            # Early stopping
            patience = self.training_config.get('patience', 10)
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        training_time = time.time() - start_time
        logger.info(f"ECG training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final model
        final_model_path = save_path / 'final_ecg_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'metrics': self.metrics,
            'training_time': training_time
        }, final_model_path)
        
        self.writer.close()
        return self.metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc='Evaluating'):
                if len(batch_data) == 3:
                    signals, labels, features = batch_data
                    signals, labels, features = signals.to(self.device), labels.to(self.device), features.to(self.device)
                else:
                    signals, labels = batch_data
                    signals, labels = signals.to(self.device), labels.to(self.device)
                    features = None
                
                outputs = self.model(signals)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities[:, 1])
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        logger.info(f"ECG Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path: str = 'ecg_training_history.png'):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.metrics['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(self.metrics['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.metrics['train_f1'], label='Train F1')
        axes[1, 0].plot(self.metrics['val_f1'], label='Validation F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC
        axes[1, 1].plot(self.metrics['val_auc'], label='Validation AUC')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (float): Weighting factor for rare class
            gamma (float): Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_data_loaders(ecg_loader: ECGDatasetLoader, 
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        ecg_loader (ECGDatasetLoader): ECG data loader
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get data splits
    train_data = ecg_loader.train_data
    val_data = ecg_loader.val_data
    test_data = ecg_loader.test_data
    
    # Create datasets
    train_dataset = ECGDataset(
        signals=train_data['signals'],
        labels=train_data['labels'],
        features=train_data['features']
    )
    
    val_dataset = ECGDataset(
        signals=val_data['signals'],
        labels=val_data['labels'],
        features=val_data['features']
    )
    
    test_dataset = ECGDataset(
        signals=test_data['signals'],
        labels=test_data['labels'],
        features=test_data['features']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ECG Models')
    parser.add_argument('--csv_file', type=str, default='ECGCvdata.csv',
                       help='Path to ECG CSV file')
    parser.add_argument('--model_type', type=str, default='1d_cnn',
                       choices=['1d_cnn', 'lstm', 'bilstm', 'transformer', 'hybrid'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                       help='Optimizer')
    parser.add_argument('--input_length', type=int, default=500,
                       help='Input signal length')
    parser.add_argument('--save_dir', type=str, default='ecg_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create configurations
    model_config = {
        'type': args.model_type,
        'input_length': args.input_length,
        'num_classes': 2,
        'dropout_rate': 0.5
    }
    
    training_config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'loss_function': 'cross_entropy',
        'patience': 10,
        'checkpoint_frequency': 5
    }
    
    optimizer_config = {
        'name': args.optimizer,
        'lr': args.learning_rate,
        'weight_decay': 1e-4
    }
    
    lr_scheduler_config = {
        'name': 'cosine_annealing',
        'T_max': args.epochs,
        'eta_min': 1e-6
    }
    
    # Initialize trainer
    trainer = ECGTrainer(
        model_config=model_config,
        training_config=training_config,
        optimizer_config=optimizer_config,
        lr_scheduler_config=lr_scheduler_config
    )
    
    # Load data
    logger.info("Loading ECG dataset...")
    ecg_loader = ECGDatasetLoader(
        csv_file=args.csv_file,
        target_length=args.input_length,
        sampling_rate=125,
        batch_size=args.batch_size
    )
    
    try:
        dataset_stats = ecg_loader.load_dataset()
        split_info = ecg_loader.create_data_splits()
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            ecg_loader, 
            batch_size=args.batch_size
        )
        
        # Train model
        history = trainer.train(train_loader, val_loader, args.epochs, args.save_dir)
        
        # Evaluate model
        test_metrics = trainer.evaluate(test_loader)
        
        # Plot training history
        trainer.plot_training_history(f'{args.save_dir}/ecg_training_history.png')
        
        # Save training results
        results = {
            'model_config': model_config,
            'training_config': training_config,
            'optimizer_config': optimizer_config,
            'lr_scheduler_config': lr_scheduler_config,
            'dataset_stats': dataset_stats,
            'split_info': split_info,
            'training_history': history,
            'test_metrics': test_metrics
        }
        
        with open(f'{args.save_dir}/ecg_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("ECG training completed successfully!")
        
    except Exception as e:
        logger.error(f"ECG training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
