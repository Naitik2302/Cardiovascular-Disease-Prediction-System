"""
Model Fusion Module for Cardiovascular Disease Prediction
=======================================================

This module provides functionality to combine predictions from multiple modalities
(fundus images and ECG signals) for improved cardiovascular disease prediction.

Features:
- Load and manage multiple trained models
- Predict probabilities from individual models
- Weighted probability averaging for fusion
- Comprehensive evaluation metrics
- Visualization tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
from model_architectures import FundusCNN, ResNetFundus, EfficientNetFundus
from ecg_model_architectures import ECG1DCNN, ECGLSTM, ECGTransformer
from fundus_data_loader import FundusDataset
from ecg_data_loader import ECGDataset

logger = logging.getLogger(__name__)

class ModelFusion:
    """
    Model fusion class for combining fundus and ECG model predictions.
    """
    
    def __init__(self, 
                 fundus_model_path: str,
                 ecg_model_path: str,
                 fundus_model_type: str = 'custom_cnn',
                 ecg_model_type: str = '1d_cnn',
                 device: str = 'auto'):
        """
        Initialize the model fusion system.
        
        Args:
            fundus_model_path (str): Path to trained fundus model
            ecg_model_path (str): Path to trained ECG model
            fundus_model_type (str): Type of fundus model ('custom_cnn', 'resnet50', etc.)
            ecg_model_type (str): Type of ECG model ('1d_cnn', 'lstm', etc.)
            device (str): Device to load models on ('auto', 'cpu', 'cuda')
        """
        self.fundus_model_path = fundus_model_path
        self.ecg_model_path = ecg_model_path
        self.fundus_model_type = fundus_model_type
        self.ecg_model_type = ecg_model_type
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.fundus_model = None
        self.ecg_model = None
        self.fundus_scaler = StandardScaler()
        self.ecg_scaler = StandardScaler()
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load both fundus and ECG models."""
        logger.info("Loading fundus model...")
        self.fundus_model = self._load_fundus_model()
        
        logger.info("Loading ECG model...")
        self.ecg_model = self._load_ecg_model()
        
        logger.info("Both models loaded successfully!")
    
    def _load_fundus_model(self):
        """Load the fundus image model."""
        # Create model based on type
        if self.fundus_model_type == 'custom_cnn':
            model = FundusCNN(num_classes=2, dropout_rate=0.5)
        elif self.fundus_model_type == 'resnet50':
            model = ResNetFundus(model_name='resnet50', num_classes=2, pretrained=False)
        elif self.fundus_model_type == 'efficientnet_b0':
            from model_architectures import EfficientNetFundus
            model = EfficientNetFundus(model_name='efficientnet_b0', num_classes=2, pretrained=False)
        else:
            raise ValueError(f"Unsupported fundus model type: {self.fundus_model_type}")
        
        # Load state dict
        checkpoint = torch.load(self.fundus_model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Fundus model loaded: {self.fundus_model_type}")
        return model
    
    def _load_ecg_model(self):
        """Load the ECG signal model."""
        # Create model based on type
        if self.ecg_model_type == '1d_cnn':
            model = ECG1DCNN(input_length=500, num_classes=2, dropout_rate=0.3)
        elif self.ecg_model_type == 'lstm':
            model = ECGLSTM(input_length=500, num_classes=2, dropout_rate=0.5)
        elif self.ecg_model_type == 'transformer':
            model = ECGTransformer(input_length=500, num_classes=2, dropout_rate=0.3)
        else:
            raise ValueError(f"Unsupported ECG model type: {self.ecg_model_type}")
        
        # Load state dict
        checkpoint = torch.load(self.ecg_model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"ECG model loaded: {self.ecg_model_type}")
        return model
    
    def predict_fundus(self, fundus_input: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities from fundus model.
        
        Args:
            fundus_input: Fundus image tensor or numpy array
            
        Returns:
            Predicted probabilities (N x 2 array)
        """
        self.fundus_model.eval()
        
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(fundus_input, np.ndarray):
                fundus_input = torch.FloatTensor(fundus_input)
            
            # Add batch dimension if needed
            if fundus_input.dim() == 3:
                fundus_input = fundus_input.unsqueeze(0)
            
            # Move to device
            fundus_input = fundus_input.to(self.device)
            
            # Get predictions
            outputs = self.fundus_model(fundus_input)
            probabilities = F.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()
    
    def predict_ecg(self, ecg_input: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities from ECG model.
        
        Args:
            ecg_input: ECG signal tensor or numpy array
            
        Returns:
            Predicted probabilities (N x 2 array)
        """
        self.ecg_model.eval()
        
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(ecg_input, np.ndarray):
                ecg_input = torch.FloatTensor(ecg_input)
            
            # Add batch dimension if needed
            if ecg_input.dim() == 1:
                ecg_input = ecg_input.unsqueeze(0)
            if ecg_input.dim() == 2:
                ecg_input = ecg_input.unsqueeze(1)  # Add channel dimension
            
            # Move to device
            ecg_input = ecg_input.to(self.device)
            
            # Get predictions
            outputs = self.ecg_model(ecg_input)
            probabilities = F.softmax(outputs, dim=1)
            
            return probabilities.cpu().numpy()
    
    def predict_fused(self, 
                     fundus_input: Optional[Union[torch.Tensor, np.ndarray]] = None,
                     ecg_input: Optional[Union[torch.Tensor, np.ndarray]] = None,
                     fundus_weight: float = 0.5,
                     ecg_weight: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Predict using weighted fusion of both models.
        
        Args:
            fundus_input: Fundus image input (optional)
            ecg_input: ECG signal input (optional)
            fundus_weight: Weight for fundus model predictions
            ecg_weight: Weight for ECG model predictions
            
        Returns:
            Dictionary containing individual and fused predictions
        """
        results = {}
        
        # Validate inputs
        if fundus_input is None and ecg_input is None:
            raise ValueError("At least one input type must be provided")
        
        # Get individual predictions
        if fundus_input is not None:
            fundus_probs = self.predict_fundus(fundus_input)
            results['fundus_probabilities'] = fundus_probs
            results['fundus_predictions'] = np.argmax(fundus_probs, axis=1)
        else:
            fundus_probs = None
            
        if ecg_input is not None:
            ecg_probs = self.predict_ecg(ecg_input)
            results['ecg_probabilities'] = ecg_probs
            results['ecg_predictions'] = np.argmax(ecg_probs, axis=1)
        else:
            ecg_probs = None
        
        # Calculate fused predictions
        if fundus_probs is not None and ecg_probs is not None:
            # Normalize weights
            total_weight = fundus_weight + ecg_weight
            fundus_weight_norm = fundus_weight / total_weight
            ecg_weight_norm = ecg_weight / total_weight
            
            # Weighted average of probabilities
            fused_probs = (fundus_weight_norm * fundus_probs + 
                          ecg_weight_norm * ecg_probs)
            
            results['fused_probabilities'] = fused_probs
            results['fused_predictions'] = np.argmax(fused_probs, axis=1)
            results['fusion_weights'] = {
                'fundus_weight': fundus_weight_norm,
                'ecg_weight': ecg_weight_norm
            }
        elif fundus_probs is not None:
            # Only fundus available
            results['fused_probabilities'] = fundus_probs
            results['fused_predictions'] = np.argmax(fundus_probs, axis=1)
            results['fusion_weights'] = {'fundus_weight': 1.0, 'ecg_weight': 0.0}
        else:
            # Only ECG available
            results['fused_probabilities'] = ecg_probs
            results['fused_predictions'] = np.argmax(ecg_probs, axis=1)
            results['fusion_weights'] = {'fundus_weight': 0.0, 'ecg_weight': 1.0}
        
        return results
    
    def optimize_fusion_weights(self, 
                               fundus_val_data: np.ndarray,
                               ecg_val_data: np.ndarray,
                               true_labels: np.ndarray,
                               weight_range: Tuple[float, float] = (0.1, 0.9),
                               step: float = 0.1) -> Dict[str, float]:
        """
        Optimize fusion weights using validation data.
        
        Args:
            fundus_val_data: Validation fundus data
            ecg_val_data: Validation ECG data
            true_labels: True labels for validation data
            weight_range: Range of weights to search
            step: Step size for weight search
            
        Returns:
            Optimal fusion weights
        """
        logger.info("Optimizing fusion weights...")
        
        best_weights = {'fundus_weight': 0.5, 'ecg_weight': 0.5}
        best_score = 0.0
        
        # Search over weight combinations
        fundus_weights = np.arange(weight_range[0], weight_range[1] + step, step)
        
        for fundus_w in fundus_weights:
            ecg_w = 1.0 - fundus_w
            
            # Get predictions with current weights
            results = self.predict_fused(
                fundus_val_data, ecg_val_data,
                fundus_weight=fundus_w, ecg_weight=ecg_w
            )
            
            # Calculate F1 score
            predictions = results['fused_predictions']
            f1 = f1_score(true_labels, predictions, average='weighted')
            
            if f1 > best_score:
                best_score = f1
                best_weights = {'fundus_weight': fundus_w, 'ecg_weight': ecg_w}
        
        logger.info(f"Optimal weights found: {best_weights} with F1 score: {best_score:.4f}")
        return best_weights


class FusionEvaluator:
    """
    Evaluation class for model fusion system.
    """
    
    def __init__(self, fusion_model: ModelFusion):
        """
        Initialize the evaluator.
        
        Args:
            fusion_model: Trained ModelFusion instance
        """
        self.fusion_model = fusion_model
        self.results = {}
    
    def evaluate(self, 
                fundus_test_data: Optional[np.ndarray] = None,
                ecg_test_data: Optional[np.ndarray] = None,
                true_labels: np.ndarray = None,
                fundus_weight: float = 0.5,
                ecg_weight: float = 0.5) -> Dict[str, any]:
        """
        Comprehensive evaluation of the fusion system.
        
        Args:
            fundus_test_data: Test fundus data
            ecg_test_data: Test ECG data
            true_labels: True labels
            fundus_weight: Weight for fundus model
            ecg_weight: Weight for ECG model
            
        Returns:
            Evaluation metrics dictionary
        """
        if true_labels is None:
            raise ValueError("True labels must be provided")
        
        # Get predictions
        results = self.fusion_model.predict_fused(
            fundus_test_data, ecg_test_data,
            fundus_weight=fundus_weight, ecg_weight=ecg_weight
        )
        
        # Calculate metrics
        metrics = {}
        
        # Individual model metrics
        if 'fundus_predictions' in results:
            metrics['fundus'] = self._calculate_metrics(
                true_labels, results['fundus_predictions'], results['fundus_probabilities']
            )
        
        if 'ecg_predictions' in results:
            metrics['ecg'] = self._calculate_metrics(
                true_labels, results['ecg_predictions'], results['ecg_probabilities']
            )
        
        # Fused model metrics
        metrics['fused'] = self._calculate_metrics(
            true_labels, results['fused_predictions'], results['fused_probabilities']
        )
        
        self.results = {
            'predictions': results,
            'metrics': metrics,
            'fusion_weights': results.get('fusion_weights', {})
        }
        
        return self.results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (handle binary case)
        try:
            if y_prob.shape[1] == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:  # Multi-class
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except ValueError:
            metrics['auc'] = 0.0  # Handle case where only one class is predicted
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def plot_results(self, save_dir: str = 'fusion_results'):
        """
        Generate comprehensive visualization plots.
        
        Args:
            save_dir: Directory to save plots
        """
        Path(save_dir).mkdir(exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Fusion Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. ROC Curves
        self._plot_roc_curves(axes[0, 0])
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices(axes[0, 1])
        
        # 3. Metrics Comparison
        self._plot_metrics_comparison(axes[1, 0])
        
        # 4. Probability Distributions
        self._plot_probability_distributions(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/fusion_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {save_dir}/fusion_evaluation.png")
    
    def _plot_roc_curves(self, ax):
        """Plot ROC curves for all models."""
        if 'fused_probabilities' not in self.results['predictions']:
            return
            
        y_true = self.results.get('true_labels')
        if y_true is None:
            return
        
        # Plot ROC curves for each model
        colors = ['blue', 'red', 'green']
        models = ['fundus', 'ecg', 'fused']
        
        for i, model_name in enumerate(models):
            if model_name in self.results['metrics']:
                probs = self.results['predictions'].get(f'{model_name}_probabilities')
                if probs is not None and probs.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
                    auc = self.results['metrics'][model_name]['auc']
                    ax.plot(fpr, tpr, color=colors[i], 
                           label=f'{model_name.title()} (AUC = {auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrices(self, ax):
        """Plot confusion matrices for all models."""
        # For now, plot the fused model confusion matrix
        if 'fused' in self.results['metrics']:
            cm = np.array(self.results['metrics']['fused']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Fused Model Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    def _plot_metrics_comparison(self, ax):
        """Plot comparison of metrics across models."""
        models = list(self.results['metrics'].keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        # Prepare data for plotting
        data = []
        for model in models:
            model_metrics = self.results['metrics'][model]
            data.append([model_metrics.get(metric, 0) for metric in metrics_names])
        
        # Create bar plot
        x = np.arange(len(metrics_names))
        width = 0.25
        
        for i, (model, model_data) in enumerate(zip(models, data)):
            ax.bar(x + i * width, model_data, width, label=model.title(), alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_probability_distributions(self, ax):
        """Plot probability distributions."""
        if 'fused_probabilities' not in self.results['predictions']:
            return
        
        probs = self.results['predictions']['fused_probabilities']
        
        # Plot histogram of positive class probabilities
        ax.hist(probs[:, 1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Probability of Positive Class')
        ax.set_ylabel('Frequency')
        ax.set_title('Fused Model Probability Distribution')
        ax.grid(True, alpha=0.3)
    
    def save_results(self, save_path: str = 'fusion_results.json'):
        """Save evaluation results to JSON file."""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=self._json_converter)
        
        logger.info(f"Results saved to {save_path}")
    
    def _json_converter(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    """Example usage of the fusion system."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize fusion system
    fusion = ModelFusion(
        fundus_model_path='checkpoints/best_model.pth',
        ecg_model_path='ecg_checkpoints/best_ecg_model.pth',
        fundus_model_type='custom_cnn',
        ecg_model_type='1d_cnn'
    )
    
    # Example prediction
    # fundus_data = torch.randn(1, 3, 224, 224)  # Example fundus image
    # ecg_data = torch.randn(1, 500)  # Example ECG signal
    
    # results = fusion.predict_fused(fundus_data, ecg_data, fundus_weight=0.6, ecg_weight=0.4)
    # print("Fused prediction results:", results)


if __name__ == "__main__":
    main()