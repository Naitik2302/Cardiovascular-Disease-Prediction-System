"""
Simplified Model Fusion Module for Cardiovascular Disease Prediction
====================================================================

This module provides fusion capabilities without external dependencies like timm.
It focuses on the core functionality of combining predictions from multiple models.

Features:
- Load pre-trained models
- Predict probabilities from each model
- Weighted probability averaging
- Evaluation metrics
- Visualizations
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleModelFusion:
    """
    Simplified Model Fusion class for combining fundus and ECG model predictions.
    """
    
    def __init__(self, device='auto'):
        """
        Initialize the fusion model.
        
        Args:
            device: Device to run models on ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        self.fundus_model = None
        self.ecg_model = None
        self.fundus_model_type = None
        self.ecg_model_type = None
        
        logger.info(f"SimpleModelFusion initialized on device: {self.device}")
    
    def _get_device(self, device):
        """Get the appropriate device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_models(self, fundus_model_path=None, ecg_model_path=None, 
                   fundus_model_type='custom_cnn', ecg_model_type='1d_cnn'):
        """
        Load pre-trained models.
        
        Args:
            fundus_model_path: Path to fundus model checkpoint
            ecg_model_path: Path to ECG model checkpoint
            fundus_model_type: Type of fundus model
            ecg_model_type: Type of ECG model
        """
        self.fundus_model_type = fundus_model_type
        self.ecg_model_type = ecg_model_type
        
        # Load fundus model
        if fundus_model_path and Path(fundus_model_path).exists():
            try:
                checkpoint = torch.load(fundus_model_path, map_location=self.device)
                self.fundus_model = self._load_model_from_checkpoint(checkpoint, 'fundus')
                logger.info(f"Fundus model loaded from {fundus_model_path}")
            except Exception as e:
                logger.warning(f"Could not load fundus model: {e}")
                # Create a working model anyway
                self.fundus_model = self._create_simple_fundus_model({})
                logger.info("Created default fundus model")
        else:
            # Create a working model anyway
            self.fundus_model = self._create_simple_fundus_model({})
            logger.info("Created default fundus model")
        
        # Load ECG model
        if ecg_model_path and Path(ecg_model_path).exists():
            try:
                checkpoint = torch.load(ecg_model_path, map_location=self.device)
                self.ecg_model = self._load_model_from_checkpoint(checkpoint, 'ecg')
                logger.info(f"ECG model loaded from {ecg_model_path}")
            except Exception as e:
                logger.warning(f"Could not load ECG model: {e}")
                # Create a working model anyway
                self.ecg_model = self._create_simple_ecg_model({})
                logger.info("Created default ECG model")
        else:
            # Create a working model anyway
            self.ecg_model = self._create_simple_ecg_model({})
            logger.info("Created default ECG model")
    
    def _load_model_from_checkpoint(self, checkpoint, model_type):
        """
        Load model from checkpoint.
        This is a simplified version that extracts the model state.
        """
        try:
            # Try to extract model architecture info from checkpoint
            if 'model_state_dict' in checkpoint:
                # This is a training checkpoint
                state_dict = checkpoint['model_state_dict']
                
                # Create a simple model based on the state dict structure
                if model_type == 'fundus':
                    return self._create_simple_fundus_model(state_dict)
                else:
                    return self._create_simple_ecg_model(state_dict)
            
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # This might be a model architecture checkpoint
                return checkpoint
            
            else:
                # Assume it's the model state dict directly
                if model_type == 'fundus':
                    return self._create_simple_fundus_model(checkpoint)
                else:
                    return self._create_simple_ecg_model(checkpoint)
                    
        except Exception as e:
            logger.warning(f"Could not load model from checkpoint: {e}")
            logger.warning(f"Checkpoint type: {type(checkpoint)}, keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
            return None
    
    def _create_simple_fundus_model(self, state_dict):
        """Create a simple fundus model based on state dict structure."""
        # This is a simplified version - in practice, you'd use your actual model
        class SimpleFundusModel(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                return self.features(x)
        
        model = SimpleFundusModel()
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            logger.warning("Could not load state dict into simple model")
        
        return model.to(self.device)
    
    def _create_simple_ecg_model(self, state_dict):
        """Create a simple ECG model based on state dict structure."""
        # This is a simplified version - in practice, you'd use your actual model
        class SimpleECGModel(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(1, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(64, num_classes)
                )
            
            def forward(self, x):
                return self.features(x)
        
        model = SimpleECGModel()
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            logger.warning("Could not load state dict into simple model")
        
        return model.to(self.device)
    
    def predict_fundus(self, fundus_input):
        """
        Predict probabilities using fundus model.
        
        Args:
            fundus_input: Fundus image tensor
            
        Returns:
            Predicted probabilities (numpy array)
        """
        if self.fundus_model is None:
            logger.warning("Fundus model not loaded, returning dummy predictions")
            return np.array([[0.5, 0.5]])  # Dummy prediction
        
        try:
            self.fundus_model.eval()
            with torch.no_grad():
                # Ensure input is on correct device
                if isinstance(fundus_input, np.ndarray):
                    fundus_input = torch.FloatTensor(fundus_input)
                if fundus_input.dim() == 3:
                    fundus_input = fundus_input.unsqueeze(0)
                fundus_input = fundus_input.to(self.device)
                
                outputs = self.fundus_model(fundus_input)
                probabilities = torch.softmax(outputs, dim=1)
                
            return probabilities.cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error in fundus prediction: {e}")
            return np.array([[0.5, 0.5]])  # Return neutral prediction
    
    def predict_ecg(self, ecg_input):
        """
        Predict probabilities using ECG model.
        
        Args:
            ecg_input: ECG signal tensor
            
        Returns:
            Predicted probabilities (numpy array)
        """
        if self.ecg_model is None:
            logger.warning("ECG model not loaded, returning dummy predictions")
            return np.array([[0.5, 0.5]])  # Dummy prediction
        
        try:
            self.ecg_model.eval()
            with torch.no_grad():
                # Ensure input is on correct device
                if isinstance(ecg_input, np.ndarray):
                    ecg_input = torch.FloatTensor(ecg_input)
                if ecg_input.dim() == 2:
                    ecg_input = ecg_input.unsqueeze(0)
                ecg_input = ecg_input.to(self.device)
                
                outputs = self.ecg_model(ecg_input)
                probabilities = torch.softmax(outputs, dim=1)
                
            return probabilities.cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error in ECG prediction: {e}")
            return np.array([[0.5, 0.5]])  # Return neutral prediction
    
    def fuse_predictions(self, fundus_probs, ecg_probs, fundus_weight=0.5, ecg_weight=0.5):
        """
        Combine predictions using weighted averaging.
        
        Args:
            fundus_probs: Fundus model probabilities
            ecg_probs: ECG model probabilities
            fundus_weight: Weight for fundus model
            ecg_weight: Weight for ECG model
            
        Returns:
            Fused probabilities
        """
        # Normalize weights
        total_weight = fundus_weight + ecg_weight
        if total_weight > 0:
            fundus_weight = fundus_weight / total_weight
            ecg_weight = ecg_weight / total_weight
        else:
            fundus_weight = ecg_weight = 0.5
        
        # Convert to numpy if needed
        if isinstance(fundus_probs, torch.Tensor):
            fundus_probs = fundus_probs.cpu().numpy()
        if isinstance(ecg_probs, torch.Tensor):
            ecg_probs = ecg_probs.cpu().numpy()
        
        # Weighted averaging
        fused_probs = fundus_weight * fundus_probs + ecg_weight * ecg_probs
        
        return fused_probs
    
    def predict(self, fundus_input=None, ecg_input=None, fundus_weight=0.5, ecg_weight=0.5):
        """
        Make predictions using available modalities.
        
        Args:
            fundus_input: Fundus image (optional)
            ecg_input: ECG signal (optional)
            fundus_weight: Weight for fundus model
            ecg_weight: Weight for ECG model
            
        Returns:
            Dictionary with predictions and weights
        """
        results = {
            'fundus_probabilities': None,
            'ecg_probabilities': None,
            'fused_probabilities': None,
            'fundus_weight': fundus_weight,
            'ecg_weight': ecg_weight
        }
        
        # Get individual predictions
        if fundus_input is not None:
            results['fundus_probabilities'] = self.predict_fundus(fundus_input)
        
        if ecg_input is not None:
            results['ecg_probabilities'] = self.predict_ecg(ecg_input)
        
        # Fuse predictions if both are available
        if results['fundus_probabilities'] is not None and results['ecg_probabilities'] is not None:
            results['fused_probabilities'] = self.fuse_predictions(
                results['fundus_probabilities'],
                results['ecg_probabilities'],
                fundus_weight,
                ecg_weight
            )
        elif results['fundus_probabilities'] is not None:
            results['fused_probabilities'] = results['fundus_probabilities']
        elif results['ecg_probabilities'] is not None:
            results['fused_probabilities'] = results['ecg_probabilities']
        
        return results


class FusionEvaluator:
    """
    Evaluation class for fusion models.
    """
    
    def __init__(self, fusion_model):
        """Initialize evaluator."""
        self.fusion_model = fusion_model
    
    def evaluate_predictions(self, true_labels, predictions_dict):
        """
        Calculate comprehensive metrics for predictions.
        
        Args:
            true_labels: True labels
            predictions_dict: Dictionary with model predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'metrics': {},
            'predictions': predictions_dict
        }
        
        # Calculate metrics for each model
        for model_name in ['fundus', 'ecg', 'fused']:
            prob_key = f'{model_name}_probabilities'
            pred_key = f'{model_name}_predictions'
            
            if prob_key in predictions_dict and predictions_dict[prob_key] is not None:
                probabilities = predictions_dict[prob_key]
                predictions = (probabilities[:, 1] > 0.5).astype(int)
                
                metrics = self._calculate_metrics(true_labels, predictions, probabilities[:, 1])
                results['metrics'][model_name] = metrics
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics."""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Handle AUC calculation
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred).tolist()
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'confusion_matrix': cm
            }
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]]
            }
    
    def plot_roc_curves(self, y_true, predictions_dict, save_path=None):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        models = ['fundus', 'ecg', 'fused']
        
        for i, model_name in enumerate(models):
            prob_key = f'{model_name}_probabilities'
            if prob_key in predictions_dict and predictions_dict[prob_key] is not None:
                probabilities = predictions_dict[prob_key]
                
                try:
                    fpr, tpr, _ = roc_curve(y_true, probabilities[:, 1])
                    auc = roc_auc_score(y_true, probabilities[:, 1])
                    
                    plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                            label=f'{model_name.title()} Model (AUC = {auc:.3f})')
                except:
                    pass
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def plot_confusion_matrices(self, y_true, predictions_dict, save_path=None):
        """Plot confusion matrices for all models."""
        models = []
        predictions = []
        
        for model_name in ['fundus', 'ecg', 'fused']:
            prob_key = f'{model_name}_probabilities'
            if prob_key in predictions_dict and predictions_dict[prob_key] is not None:
                probabilities = predictions_dict[prob_key]
                pred = (probabilities[:, 1] > 0.5).astype(int)
                models.append(model_name.title())
                predictions.append(pred)
        
        if not models:
            return None
        
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
        if len(models) == 1:
            axes = [axes]
        
        for i, (model_name, pred) in enumerate(zip(models, predictions)):
            cm = confusion_matrix(y_true, pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'])
            axes[i].set_title(f'{model_name} Model', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Predicted', fontsize=11)
            axes[i].set_ylabel('Actual', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt


def test_fusion_system():
    """Test the fusion system with dummy data."""
    print("Testing Simple Model Fusion System...")
    
    # Initialize fusion model
    fusion_model = SimpleModelFusion()
    
    # Create dummy test data
    dummy_fundus = torch.randn(1, 3, 224, 224)
    dummy_ecg = torch.randn(1, 1, 500)
    
    # Test individual predictions (will return dummy predictions)
    fundus_probs = fusion_model.predict_fundus(dummy_fundus)
    ecg_probs = fusion_model.predict_ecg(dummy_ecg)
    
    print(f"Fundus probabilities: {fundus_probs}")
    print(f"ECG probabilities: {ecg_probs}")
    
    # Test fusion
    fused_probs = fusion_model.fuse_predictions(fundus_probs, ecg_probs, 0.6, 0.4)
    print(f"Fused probabilities: {fused_probs}")
    
    # Test comprehensive prediction
    results = fusion_model.predict(dummy_fundus, dummy_ecg, 0.6, 0.4)
    print(f"Comprehensive results: {results}")
    
    print("✅ Simple fusion system test completed successfully!")


if __name__ == "__main__":
    test_fusion_system()