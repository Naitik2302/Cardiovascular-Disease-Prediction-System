"""
Model Utilities and Helper Functions
===================================

This module contains utility functions for model management, evaluation,
and visualization.

Features:
- Model loading and saving
- Evaluation metrics calculation
- Visualization utilities
- Model comparison tools
- Performance analysis
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model management utilities for saving, loading, and comparing models.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the model manager.
        
        Args:
            model_dir (str): Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, 
                   model: nn.Module,
                   model_name: str,
                   epoch: int,
                   metrics: Dict[str, float],
                   config: Dict[str, Any],
                   additional_info: Optional[Dict] = None) -> str:
        """
        Save a model with metadata.
        
        Args:
            model (nn.Module): PyTorch model
            model_name (str): Name for the model
            epoch (int): Training epoch
            metrics (dict): Model metrics
            config (dict): Model configuration
            additional_info (dict): Additional information to save
            
        Returns:
            str: Path to saved model
        """
        model_path = self.model_dir / f"{model_name}_epoch_{epoch}.pth"
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'config': config,
            'model_class': model.__class__.__name__
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return str(model_path)
    
    def load_model(self, 
                   model_path: str,
                   model_class: nn.Module,
                   device: str = 'cpu') -> Tuple[nn.Module, Dict]:
        """
        Load a model from file.
        
        Args:
            model_path (str): Path to model file
            model_class (nn.Module): Model class to instantiate
            device (str): Device to load model on
            
        Returns:
            tuple: (loaded_model, metadata)
        """
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model instance
        model = model_class
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {}),
            'model_class': checkpoint.get('model_class', 'Unknown')
        }
        
        logger.info(f"Model loaded from {model_path}")
        return model, metadata
    
    def compare_models(self, 
                      model_paths: List[str],
                      test_data: Tuple[np.ndarray, np.ndarray],
                      model_classes: List[nn.Module]) -> pd.DataFrame:
        """
        Compare multiple models on test data.
        
        Args:
            model_paths (list): List of model file paths
            test_data (tuple): (X_test, y_test) test data
            model_classes (list): List of model classes
            
        Returns:
            pd.DataFrame: Comparison results
        """
        X_test, y_test = test_data
        results = []
        
        for i, (model_path, model_class) in enumerate(zip(model_paths, model_classes)):
            try:
                model, metadata = self.load_model(model_path, model_class)
                model.eval()
                
                # Make predictions
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test)
                    outputs = model(X_tensor)
                    predictions = torch.argmax(outputs, dim=1).numpy()
                    probabilities = torch.softmax(outputs, dim=1).numpy()
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, predictions, probabilities)
                metrics['model_name'] = Path(model_path).stem
                metrics['epoch'] = metadata['epoch']
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_path}: {str(e)}")
                continue
        
        return pd.DataFrame(results)


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def calculate_metrics(self, 
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Predicted probabilities
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics['precision_class_0'] = precision_per_class[0]
        metrics['precision_class_1'] = precision_per_class[1]
        metrics['recall_class_0'] = recall_per_class[0]
        metrics['recall_class_1'] = recall_per_class[1]
        metrics['f1_class_0'] = f1_per_class[0]
        metrics['f1_class_1'] = f1_per_class[1]
        
        # Probability-based metrics
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
            except:
                metrics['auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: List[str] = ['No CVD', 'CVD'],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (list): Class names
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: Confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray,
                      y_prob: np.ndarray,
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: ROC curve plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, 
                                   y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: Precision-recall curve plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        avg_precision = average_precision_score(y_true, y_prob[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='darkorange', lw=2, 
               label=f'PR curve (AP = {avg_precision:.2f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(self, 
                              y_true: np.ndarray,
                              y_prob: np.ndarray,
                              n_bins: int = 10,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_prob (np.ndarray): Predicted probabilities
            n_bins (int): Number of bins for calibration
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: Calibration curve plot
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob[:, 1], n_bins=n_bins
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
               label=f"Model (n_bins={n_bins})")
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(self, 
                             history: Dict[str, List[float]],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            history (dict): Training history
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: Training history plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        if 'train_loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy
        if 'train_acc' in history and 'val_acc' in history:
            axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
            axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # F1 Score
        if 'train_f1' in history and 'val_f1' in history:
            axes[1, 0].plot(history['train_f1'], label='Train F1')
            axes[1, 0].plot(history['val_f1'], label='Validation F1')
            axes[1, 0].set_title('Training and Validation F1 Score')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # AUC
        if 'val_auc' in history:
            axes[1, 1].plot(history['val_auc'], label='Validation AUC')
            axes[1, 1].set_title('Validation AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ModelAnalyzer:
    """
    Model analysis and interpretation utilities.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
    
    def analyze_predictions(self, 
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_prob: np.ndarray,
                           class_names: List[str] = ['No CVD', 'CVD']) -> Dict[str, Any]:
        """
        Analyze model predictions in detail.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Predicted probabilities
            class_names (list): Class names
            
        Returns:
            dict: Analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_samples'] = len(y_true)
        analysis['correct_predictions'] = np.sum(y_true == y_pred)
        analysis['incorrect_predictions'] = np.sum(y_true != y_pred)
        analysis['accuracy'] = np.mean(y_true == y_pred)
        
        # Confidence analysis
        analysis['mean_confidence'] = np.mean(np.max(y_prob, axis=1))
        analysis['std_confidence'] = np.std(np.max(y_prob, axis=1))
        analysis['min_confidence'] = np.min(np.max(y_prob, axis=1))
        analysis['max_confidence'] = np.max(np.max(y_prob, axis=1))
        
        # Per-class analysis
        for i, class_name in enumerate(class_names):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
                class_confidence = np.mean(np.max(y_prob[class_mask], axis=1))
                analysis[f'{class_name}_accuracy'] = class_accuracy
                analysis[f'{class_name}_confidence'] = class_confidence
                analysis[f'{class_name}_samples'] = np.sum(class_mask)
        
        # Error analysis
        error_mask = y_true != y_pred
        if np.sum(error_mask) > 0:
            analysis['error_confidence'] = np.mean(np.max(y_prob[error_mask], axis=1))
            analysis['error_distribution'] = {
                'false_positives': np.sum((y_true == 0) & (y_pred == 1)),
                'false_negatives': np.sum((y_true == 1) & (y_pred == 0))
            }
        
        return analysis
    
    def plot_prediction_distribution(self, 
                                   y_prob: np.ndarray,
                                   y_true: Optional[np.ndarray] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot prediction probability distribution.
        
        Args:
            y_prob (np.ndarray): Predicted probabilities
            y_true (np.ndarray): True labels (optional)
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: Distribution plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Probability distribution
        axes[0].hist(y_prob[:, 1], bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Predicted Probability (CVD)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Predicted Probabilities')
        axes[0].grid(True)
        
        # Confidence distribution
        confidence = np.max(y_prob, axis=1)
        axes[1].hist(confidence, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_xlabel('Prediction Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Prediction Confidence')
        axes[1].grid(True)
        
        if y_true is not None:
            # Add true label information
            for i, class_name in enumerate(['No CVD', 'CVD']):
                class_mask = y_true == i
                if np.sum(class_mask) > 0:
                    axes[0].hist(y_prob[class_mask, 1], bins=50, alpha=0.5, 
                               label=f'True {class_name}')
            axes[0].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, 
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_prob: np.ndarray,
                       model_name: str = "Model",
                       save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Predicted probabilities
            model_name (str): Model name
            save_path (str): Path to save report
            
        Returns:
            str: Report content
        """
        evaluator = ModelEvaluator()
        analyzer = ModelAnalyzer()
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
        analysis = analyzer.analyze_predictions(y_true, y_pred, y_prob)
        
        # Generate report
        report = f"""
# Model Evaluation Report: {model_name}

## Overview
- Total Samples: {analysis['total_samples']}
- Correct Predictions: {analysis['correct_predictions']}
- Incorrect Predictions: {analysis['incorrect_predictions']}
- Overall Accuracy: {metrics['accuracy']:.4f}

## Performance Metrics
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}
- AUC: {metrics['auc']:.4f}
- Average Precision: {metrics['average_precision']:.4f}

## Per-Class Performance
### No CVD (Class 0)
- Precision: {metrics['precision_class_0']:.4f}
- Recall: {metrics['recall_class_0']:.4f}
- F1-Score: {metrics['f1_class_0']:.4f}
- Samples: {analysis['No CVD_samples']}

### CVD (Class 1)
- Precision: {metrics['precision_class_1']:.4f}
- Recall: {metrics['recall_class_1']:.4f}
- F1-Score: {metrics['f1_class_1']:.4f}
- Samples: {analysis['CVD_samples']}

## Prediction Confidence
- Mean Confidence: {analysis['mean_confidence']:.4f}
- Std Confidence: {analysis['std_confidence']:.4f}
- Min Confidence: {analysis['min_confidence']:.4f}
- Max Confidence: {analysis['max_confidence']:.4f}

## Error Analysis
- False Positives: {analysis['error_distribution']['false_positives']}
- False Negatives: {analysis['error_distribution']['false_negatives']}
- Error Confidence: {analysis['error_confidence']:.4f}

## Classification Report
{classification_report(y_true, y_pred, target_names=['No CVD', 'CVD'])}

---
Report generated on: {pd.Timestamp.now()}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


def main():
    """Example usage of model utilities."""
    # Example usage
    evaluator = ModelEvaluator()
    analyzer = ModelAnalyzer()
    manager = ModelManager()
    
    # Generate dummy data for demonstration
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_prob = np.random.rand(1000, 2)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
    print("Metrics:", metrics)
    
    # Analyze predictions
    analysis = analyzer.analyze_predictions(y_true, y_pred, y_prob)
    print("Analysis:", analysis)
    
    # Generate report
    report = analyzer.generate_report(y_true, y_pred, y_prob, "Example Model")
    print(report)


if __name__ == "__main__":
    main()
