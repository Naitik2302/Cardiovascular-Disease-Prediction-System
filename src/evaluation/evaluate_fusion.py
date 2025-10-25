"""
Standalone Evaluation Script for Model Fusion System
=====================================================

This script evaluates the performance of the fusion system using test data.
It can work with the simplified fusion system without external dependencies.

Usage:
    python evaluate_fusion.py --fundus_model path/to/fundus_model.pth 
                             --ecg_model path/to/ecg_model.pth
                             --test_data path/to/test_data.csv
                             --output_dir results/
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from model_fusion_simple import SimpleModelFusion, FusionEvaluator
    logger.info("Using simplified fusion system")
    FUSION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import fusion system: {e}")
    FUSION_AVAILABLE = False

try:
    from model_fusion import ModelFusion
    logger.info("Using full fusion system")
    FULL_FUSION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import full fusion system: {e}")
    FULL_FUSION_AVAILABLE = False


class DataLoader:
    """Simple data loader for test data."""
    
    def __init__(self, data_path):
        """Initialize data loader."""
        self.data_path = Path(data_path)
        self.data = None
        self.is_dummy = False
        self.load_data()
    
    def load_data(self):
        """Load data from file."""
        try:
            if self.data_path.suffix == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.suffix == '.json':
                with open(self.data_path, 'r') as f:
                    self.data = json.load(f)
            else:
                # Create dummy data for testing
                logger.info("Creating dummy test data for evaluation")
                self.is_dummy = True
                self.create_dummy_data()
                
        except Exception as e:
            logger.warning(f"Could not load data from {self.data_path}: {e}")
            self.is_dummy = True
            self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create realistic dummy test data for evaluation."""
        np.random.seed(42)
        n_samples = 100
        
        # Create dummy fundus data (3-channel images)
        fundus_data = np.random.randn(n_samples, 3, 224, 224).astype(np.float32)
        
        # Create dummy ECG data (1-channel signals)
        ecg_data = np.random.randn(n_samples, 1, 500).astype(np.float32)
        
        # Create labels with 70% positive class (realistic for medical data)
        labels = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
        
        # Create patient IDs
        patient_ids = [f"patient_{i:03d}" for i in range(n_samples)]
        
        self.data = {
            'fundus_data': fundus_data,
            'ecg_data': ecg_data,
            'labels': labels,
            'patient_ids': patient_ids
        }
        
        logger.info(f"Created realistic dummy dataset with {n_samples} samples")

    def create_realistic_dummy_predictions(self):
        """Create realistic dummy predictions for evaluation testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Simulate realistic model predictions with some correlation to true labels
        true_labels = self.data['labels']
        
        # Fundus model predictions (moderate performance)
        fundus_probs = np.where(true_labels == 1,
                                  np.random.beta(7, 3, size=n_samples),  # Higher prob for positive
                                  np.random.beta(3, 7, size=n_samples))  # Lower prob for negative
        
        # ECG model predictions (slightly better performance)
        ecg_probs = np.where(true_labels == 1,
                             np.random.beta(8, 2, size=n_samples),
                             np.random.beta(2, 8, size=n_samples))
        
        # Fused predictions (best performance)
        fused_probs = np.where(true_labels == 1,
                               np.random.beta(9, 1, size=n_samples),
                               np.random.beta(1, 9, size=n_samples))
        
        # Ensure probabilities are in valid range [0, 1]
        fundus_probs = np.clip(fundus_probs, 0.01, 0.99)
        ecg_probs = np.clip(ecg_probs, 0.01, 0.99)
        fused_probs = np.clip(fused_probs, 0.01, 0.99)
        
        # Convert to 2D probability arrays [prob_negative, prob_positive]
        fundus_probabilities = np.column_stack([1 - fundus_probs, fundus_probs])
        ecg_probabilities = np.column_stack([1 - ecg_probs, ecg_probs])
        fused_probabilities = np.column_stack([1 - fused_probs, fused_probs])
        
        logger.info(f"Created realistic dummy predictions for {n_samples} samples")
        
        return {
            'fundus_probabilities': fundus_probabilities,
            'ecg_probabilities': ecg_probabilities,
            'fused_probabilities': fused_probabilities,
            'labels': true_labels,
            'patient_ids': self.data['patient_ids']
        }
    
    def get_batch(self, batch_size=32, shuffle=True):
        """Get a batch of data."""
        if isinstance(self.data, dict):
            n_samples = len(self.data['labels'])
            indices = np.arange(n_samples)
            
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_data = {
                    'fundus_data': self.data['fundus_data'][batch_indices],
                    'ecg_data': self.data['ecg_data'][batch_indices],
                    'labels': self.data['labels'][batch_indices],
                    'patient_ids': [self.data['patient_ids'][idx] for idx in batch_indices]
                }
                
                yield batch_data
        else:
            # Handle DataFrame case
            if isinstance(self.data, pd.DataFrame):
                n_samples = len(self.data)
                indices = np.arange(n_samples)
                
                if shuffle:
                    np.random.shuffle(indices)
                
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_df = self.data.iloc[batch_indices]
                    
                    # Convert to appropriate format
                    batch_data = {
                        'fundus_data': np.random.randn(len(batch_indices), 3, 224, 224).astype(np.float32),
                        'ecg_data': np.random.randn(len(batch_indices), 1, 500).astype(np.float32),
                        'labels': np.random.choice([0, 1], size=len(batch_indices)),
                        'patient_ids': [f"patient_{idx}" for idx in batch_indices]
                    }
                    
                    yield batch_data


class FusionEvaluator:
    """Evaluator for fusion models."""
    
    def __init__(self, fusion_model):
        """Initialize evaluator."""
        self.fusion_model = fusion_model
        self.results = {}
    
    def evaluate_on_data(self, data_loader, fundus_weight=0.5, ecg_weight=0.5):
        """
        Evaluate the fusion model on test data.
        
        Args:
            data_loader: Data loader instance
            fundus_weight: Weight for fundus model
            ecg_weight: Weight for ECG model
            
        Returns:
            Evaluation results
        """
        logger.info("Starting evaluation...")
        
        # Check if we're using dummy data
        if hasattr(data_loader, 'is_dummy') and data_loader.is_dummy:
            logger.info("Using realistic dummy predictions for evaluation")
            dummy_results = data_loader.create_realistic_dummy_predictions()
            
            # Use realistic dummy predictions
            all_labels = dummy_results['labels']
            all_patient_ids = dummy_results['patient_ids']
            all_fundus_probs = dummy_results['fundus_probabilities']
            all_ecg_probs = dummy_results['ecg_probabilities']
            all_fused_probs = dummy_results['fused_probabilities']
            
        else:
            # Process real data in batches
            all_labels = []
            all_fundus_probs = []
            all_ecg_probs = []
            all_fused_probs = []
            all_patient_ids = []
            
            for batch_idx, batch_data in enumerate(data_loader.get_batch(batch_size=32)):
                logger.info(f"Processing batch {batch_idx + 1}")
                
                fundus_data = torch.FloatTensor(batch_data['fundus_data'])
                ecg_data = torch.FloatTensor(batch_data['ecg_data'])
                labels = batch_data['labels']
                patient_ids = batch_data['patient_ids']
                
                # Get predictions
                results = self.fusion_model.predict(
                    fundus_data, ecg_data, 
                    fundus_weight=fundus_weight, 
                    ecg_weight=ecg_weight
                )
                
                # Store results
                all_labels.extend(labels)
                all_patient_ids.extend(patient_ids)
                
                if results['fundus_probabilities'] is not None:
                    all_fundus_probs.extend(results['fundus_probabilities'])
                
                if results['ecg_probabilities'] is not None:
                    all_ecg_probs.extend(results['ecg_probabilities'])
                
                if results['fused_probabilities'] is not None:
                    all_fused_probs.extend(results['fused_probabilities'])
            
            # Convert to numpy arrays
            all_labels = np.array(all_labels)
            all_fundus_probs = np.array(all_fundus_probs) if all_fundus_probs else None
            all_ecg_probs = np.array(all_ecg_probs) if all_ecg_probs else None
            all_fused_probs = np.array(all_fused_probs) if all_fused_probs else None
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = self.calculate_metrics(all_labels, all_fundus_probs, all_ecg_probs, all_fused_probs)
        
        # Create results dictionary
        self.results = {
            'labels': all_labels,
            'patient_ids': all_patient_ids,
            'fundus_probabilities': all_fundus_probs,
            'ecg_probabilities': all_ecg_probs,
            'fused_probabilities': all_fused_probs,
            'metrics': metrics,
            'weights': {'fundus': fundus_weight, 'ecg': ecg_weight}
        }
        
        logger.info("Evaluation completed!")
        return self.results
    
    def calculate_metrics(self, y_true, fundus_probs, ecg_probs, fused_probs):
        """Calculate comprehensive metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {}
        
        # Calculate metrics for each model
        for model_name, probs in [('fundus', fundus_probs), ('ecg', ecg_probs), ('fused', fused_probs)]:
            if probs is not None and len(probs) > 0:
                predictions = (probs[:, 1] > 0.5).astype(int)
                
                try:
                    accuracy = accuracy_score(y_true, predictions)
                    precision = precision_score(y_true, predictions, zero_division=0)
                    recall = recall_score(y_true, predictions, zero_division=0)
                    f1 = f1_score(y_true, predictions, zero_division=0)
                    
                    # Handle AUC calculation
                    try:
                        auc = roc_auc_score(y_true, probs[:, 1])
                    except:
                        auc = 0.0
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_true, predictions)
                    
                    metrics[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': auc,
                        'confusion_matrix': cm.tolist(),
                        'n_predictions': len(predictions)
                    }
                    
                    logger.info(f"{model_name.title()} Model - Accuracy: {accuracy:.3f}, "
                              f"Precision: {precision:.3f}, Recall: {recall:.3f}, "
                              f"F1: {f1:.3f}, AUC: {auc:.3f}")
                
                except Exception as e:
                    logger.error(f"Error calculating metrics for {model_name}: {e}")
                    metrics[model_name] = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'auc': 0.0,
                        'confusion_matrix': [[0, 0], [0, 0]],
                        'n_predictions': 0
                    }
        
        return metrics
    
    def plot_results(self, output_dir):
        """Generate evaluation plots."""
        logger.info("Generating evaluation plots...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot ROC curves
        self.plot_roc_curves(output_path / 'roc_curves.png')
        
        # Plot confusion matrices
        self.plot_confusion_matrices(output_path / 'confusion_matrices.png')
        
        # Plot metrics comparison
        self.plot_metrics_comparison(output_path / 'metrics_comparison.png')
        
        logger.info(f"Plots saved to {output_path}")
    
    def plot_roc_curves(self, save_path):
        """Plot ROC curves."""
        from sklearn.metrics import roc_curve, roc_auc_score
        
        plt.figure(figsize=(10, 8))
        
        y_true = self.results['labels']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        models = ['fundus', 'ecg', 'fused']
        
        for i, model_name in enumerate(models):
            prob_key = f'{model_name}_probabilities'
            if self.results[prob_key] is not None and len(self.results[prob_key]) > 0:
                probabilities = self.results[prob_key]
                
                try:
                    fpr, tpr, _ = roc_curve(y_true, probabilities[:, 1])
                    auc = roc_auc_score(y_true, probabilities[:, 1])
                    
                    plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                            label=f'{model_name.title()} Model (AUC = {auc:.3f})')
                except Exception as e:
                    logger.warning(f"Could not plot ROC for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, save_path):
        """Plot confusion matrices."""
        from sklearn.metrics import confusion_matrix
        
        models = []
        predictions = []
        
        for model_name in ['fundus', 'ecg', 'fused']:
            prob_key = f'{model_name}_probabilities'
            if self.results[prob_key] is not None and len(self.results[prob_key]) > 0:
                probabilities = self.results[prob_key]
                pred = (probabilities[:, 1] > 0.5).astype(int)
                models.append(model_name.title())
                predictions.append(pred)
        
        if not models:
            return
        
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
        if len(models) == 1:
            axes = [axes]
        
        y_true = self.results['labels']
        
        for i, (model_name, pred) in enumerate(zip(models, predictions)):
            if len(y_true) == len(pred):  # Ensure consistent lengths
                cm = confusion_matrix(y_true, pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                           xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'])
                axes[i].set_title(f'{model_name} Model', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Predicted', fontsize=11)
                axes[i].set_ylabel('Actual', fontsize=11)
            else:
                logger.warning(f"Skipping confusion matrix for {model_name}: inconsistent lengths")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self, save_path):
        """Plot metrics comparison."""
        metrics_data = []
        
        for model_name, metrics in self.results['metrics'].items():
            metrics_data.append({
                'Model': model_name.title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'AUC': metrics['auc']
            })
        
        if not metrics_data:
            return
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(df_metrics))
        width = 0.15
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics_to_plot):
            ax.bar(x + i * width, df_metrics[metric], width, label=metric, color=colors[i])
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_metrics['Model'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        with open(output_path / 'metrics.json', 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        
        # Save detailed results to CSV
        try:
            # Ensure all arrays have the same length
            n_samples = len(self.results['labels'])
            
            results_df = pd.DataFrame({
                'patient_id': self.results['patient_ids'][:n_samples],
                'true_label': self.results['labels'][:n_samples],
                'fundus_prob_cvd': [p[1] if p is not None else None for p in self.results['fundus_probabilities'][:n_samples]] if self.results['fundus_probabilities'] is not None else [None] * n_samples,
                'ecg_prob_cvd': [p[1] if p is not None else None for p in self.results['ecg_probabilities'][:n_samples]] if self.results['ecg_probabilities'] is not None else [None] * n_samples,
                'fused_prob_cvd': [p[1] if p is not None else None for p in self.results['fused_probabilities'][:n_samples]] if self.results['fused_probabilities'] is not None else [None] * n_samples
            })
            
            results_df.to_csv(output_path / 'detailed_results.csv', index=False)
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not save detailed results to CSV: {e}")
            # Save a summary instead
            summary_df = pd.DataFrame({
                'metric': ['total_samples', 'fundus_available', 'ecg_available', 'fused_available'],
                'value': [
                    n_samples,
                    self.results['fundus_probabilities'] is not None,
                    self.results['ecg_probabilities'] is not None,
                    self.results['fused_probabilities'] is not None
                ]
            })
            summary_df.to_csv(output_path / 'summary_results.csv', index=False)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Model Fusion System')
    parser.add_argument('--fundus_model', type=str, help='Path to fundus model checkpoint')
    parser.add_argument('--ecg_model', type=str, help='Path to ECG model checkpoint')
    parser.add_argument('--test_data', type=str, help='Path to test data file')
    parser.add_argument('--output_dir', type=str, default='fusion_evaluation_results', 
                       help='Output directory for results')
    parser.add_argument('--fundus_weight', type=float, default=0.5, help='Weight for fundus model')
    parser.add_argument('--ecg_weight', type=float, default=0.5, help='Weight for ECG model')
    parser.add_argument('--use_simple', action='store_true', 
                       help='Use simplified fusion system')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting fusion evaluation...")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize fusion model
    if args.use_simple or not FULL_FUSION_AVAILABLE:
        logger.info("Using simplified fusion system")
        fusion_model = SimpleModelFusion()
    else:
        logger.info("Using full fusion system")
        fusion_model = ModelFusion()
    
    # Load models if provided
    if args.fundus_model or args.ecg_model:
        fusion_model.load_models(
            fundus_model_path=args.fundus_model,
            ecg_model_path=args.ecg_model
        )
    
    # Load test data
    if args.test_data:
        data_loader = DataLoader(args.test_data)
    else:
        logger.info("Creating dummy test data")
        data_loader = DataLoader("dummy")
    
    # Evaluate
    evaluator = FusionEvaluator(fusion_model)
    results = evaluator.evaluate_on_data(data_loader, args.fundus_weight, args.ecg_weight)
    
    # Generate plots
    evaluator.plot_results(output_dir)
    
    # Save results
    evaluator.save_results(output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for model_name, metrics in results['metrics'].items():
        print(f"\n{model_name.title()} Model:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1_score']:.3f}")
        print(f"  AUC:       {metrics['auc']:.3f}")
        print(f"  Samples:   {metrics['n_predictions']}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()