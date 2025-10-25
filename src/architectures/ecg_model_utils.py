"""
ECG Model Utilities and Evaluation
==================================

This module contains utility functions specifically for ECG model evaluation,
visualization, and analysis.

Features:
- ECG-specific evaluation metrics
- Signal visualization tools
- Model performance analysis
- ECG signal preprocessing utilities
- Comparative analysis tools
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
from scipy import signal
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

class ECGEvaluator:
    """
    ECG-specific evaluation utilities.
    """
    
    def __init__(self):
        """Initialize the ECG evaluator."""
        pass
    
    def calculate_ecg_metrics(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_prob: Optional[np.ndarray] = None,
                             ecg_signals: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive ECG evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Predicted probabilities
            ecg_signals (np.ndarray): ECG signals for analysis
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics['precision_normal'] = precision_per_class[0]
        metrics['precision_abnormal'] = precision_per_class[1]
        metrics['recall_normal'] = recall_per_class[0]
        metrics['recall_abnormal'] = recall_per_class[1]
        metrics['f1_normal'] = f1_per_class[0]
        metrics['f1_abnormal'] = f1_per_class[1]
        
        # Probability-based metrics
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
                
                # Sensitivity and Specificity at optimal threshold
                fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                metrics['optimal_threshold'] = optimal_threshold
                metrics['sensitivity'] = tpr[optimal_idx]
                metrics['specificity'] = 1 - fpr[optimal_idx]
                
            except:
                metrics['auc'] = 0.0
                metrics['average_precision'] = 0.0
                metrics['optimal_threshold'] = 0.5
                metrics['sensitivity'] = 0.0
                metrics['specificity'] = 0.0
        
        # ECG-specific metrics
        if ecg_signals is not None:
            ecg_metrics = self._calculate_ecg_specific_metrics(ecg_signals, y_true, y_pred)
            metrics.update(ecg_metrics)
        
        return metrics
    
    def _calculate_ecg_specific_metrics(self, 
                                      ecg_signals: np.ndarray,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate ECG-specific metrics.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            dict: ECG-specific metrics
        """
        metrics = {}
        
        try:
            # Calculate signal quality metrics
            signal_quality = self._assess_signal_quality(ecg_signals)
            metrics.update(signal_quality)
            
            # Calculate prediction confidence by signal quality
            if 'signal_quality_score' in signal_quality:
                quality_scores = signal_quality['signal_quality_score']
                
                # High quality signals
                high_quality_mask = quality_scores > np.percentile(quality_scores, 75)
                if np.sum(high_quality_mask) > 0:
                    high_quality_acc = accuracy_score(
                        y_true[high_quality_mask], 
                        y_pred[high_quality_mask]
                    )
                    metrics['accuracy_high_quality'] = high_quality_acc
                
                # Low quality signals
                low_quality_mask = quality_scores < np.percentile(quality_scores, 25)
                if np.sum(low_quality_mask) > 0:
                    low_quality_acc = accuracy_score(
                        y_true[low_quality_mask], 
                        y_pred[low_quality_mask]
                    )
                    metrics['accuracy_low_quality'] = low_quality_acc
            
        except Exception as e:
            logger.warning(f"Error calculating ECG-specific metrics: {e}")
        
        return metrics
    
    def _assess_signal_quality(self, ecg_signals: np.ndarray) -> Dict[str, float]:
        """
        Assess ECG signal quality.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            
        Returns:
            dict: Signal quality metrics
        """
        metrics = {}
        
        try:
            # Calculate signal-to-noise ratio
            snr_scores = []
            for signal in ecg_signals:
                # Estimate noise as high-frequency components
                noise = signal - signal.rolling(window=5, center=True).mean()
                signal_power = np.mean(signal ** 2)
                noise_power = np.mean(noise ** 2)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                snr_scores.append(snr)
            
            metrics['mean_snr'] = np.mean(snr_scores)
            metrics['std_snr'] = np.std(snr_scores)
            metrics['signal_quality_score'] = np.array(snr_scores)
            
            # Calculate signal variability
            signal_vars = np.var(ecg_signals, axis=1)
            metrics['mean_signal_variance'] = np.mean(signal_vars)
            metrics['std_signal_variance'] = np.std(signal_vars)
            
        except Exception as e:
            logger.warning(f"Error assessing signal quality: {e}")
            metrics['mean_snr'] = 0.0
            metrics['std_snr'] = 0.0
            metrics['signal_quality_score'] = np.zeros(len(ecg_signals))
        
        return metrics


class ECGVisualizer:
    """
    ECG-specific visualization utilities.
    """
    
    def __init__(self):
        """Initialize the ECG visualizer."""
        pass
    
    def plot_ecg_signals(self, 
                        ecg_signals: np.ndarray,
                        labels: Optional[np.ndarray] = None,
                        predictions: Optional[np.ndarray] = None,
                        n_samples: int = 6,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ECG signals with labels and predictions.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            labels (np.ndarray): True labels
            predictions (np.ndarray): Predicted labels
            n_samples (int): Number of samples to plot
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: ECG signals plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select random samples
        if len(ecg_signals) > n_samples:
            indices = np.random.choice(len(ecg_signals), n_samples, replace=False)
        else:
            indices = np.arange(len(ecg_signals))
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            signal = ecg_signals[idx]
            time_axis = np.arange(len(signal))
            
            axes[i].plot(time_axis, signal, 'b-', linewidth=1)
            axes[i].set_title(f'Sample {idx}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True)
            
            # Add labels and predictions if available
            title_parts = [f'Sample {idx}']
            if labels is not None:
                title_parts.append(f'True: {labels[idx]}')
            if predictions is not None:
                title_parts.append(f'Pred: {predictions[idx]}')
            
            axes[i].set_title(' | '.join(title_parts))
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_ecg_spectrogram(self, 
                            ecg_signals: np.ndarray,
                            sampling_rate: int = 125,
                            n_samples: int = 4,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ECG spectrograms.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            sampling_rate (int): Sampling rate
            n_samples (int): Number of samples to plot
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: Spectrogram plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Select random samples
        if len(ecg_signals) > n_samples:
            indices = np.random.choice(len(ecg_signals), n_samples, replace=False)
        else:
            indices = np.arange(len(ecg_signals))
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            signal = ecg_signals[idx]
            
            # Calculate spectrogram
            f, t, Sxx = signal.spectrogram(signal, fs=sampling_rate, nperseg=64)
            
            # Plot spectrogram
            im = axes[i].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            axes[i].set_title(f'ECG Spectrogram - Sample {idx}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Frequency (Hz)')
            axes[i].set_ylim(0, sampling_rate/2)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], label='Power (dB)')
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_ecg_features(self, 
                         ecg_signals: np.ndarray,
                         labels: np.ndarray,
                         feature_names: List[str] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ECG features distribution.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            labels (np.ndarray): Labels
            feature_names (list): Feature names
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: Features plot
        """
        # Extract basic features
        features = self._extract_basic_features(ecg_signals)
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(features.shape[1])]
        
        # Create subplots
        n_features = min(features.shape[1], 6)  # Limit to 6 features
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_features):
            feature = features[:, i]
            
            # Plot distribution by class
            for class_label in np.unique(labels):
                class_mask = labels == class_label
                class_feature = feature[class_mask]
                
                axes[i].hist(class_feature, alpha=0.7, 
                           label=f'Class {class_label}', bins=20)
            
            axes[i].set_title(feature_names[i])
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _extract_basic_features(self, ecg_signals: np.ndarray) -> np.ndarray:
        """
        Extract basic features from ECG signals.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            
        Returns:
            np.ndarray: Extracted features
        """
        features = []
        
        for signal in ecg_signals:
            signal_features = [
                np.mean(signal),           # Mean
                np.std(signal),            # Standard deviation
                np.var(signal),            # Variance
                np.max(signal),            # Maximum
                np.min(signal),            # Minimum
                np.ptp(signal),            # Peak-to-peak
                np.median(signal),         # Median
                np.percentile(signal, 25), # 25th percentile
                np.percentile(signal, 75), # 75th percentile
                np.sum(np.abs(signal))     # Total variation
            ]
            features.append(signal_features)
        
        return np.array(features)
    
    def plot_model_comparison(self, 
                             model_results: Dict[str, Dict[str, float]],
                             metric: str = 'accuracy',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model comparison.
        
        Args:
            model_results (dict): Model results
            metric (str): Metric to compare
            save_path (str): Path to save plot
            
        Returns:
            plt.Figure: Comparison plot
        """
        models = list(model_results.keys())
        values = [model_results[model].get(metric, 0) for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(f'Model Comparison - {metric.title()}')
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.title())
        ax.set_ylim(0, max(values) * 1.1)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ECGAnalyzer:
    """
    ECG signal analysis utilities.
    """
    
    def __init__(self):
        """Initialize the ECG analyzer."""
        pass
    
    def analyze_signal_characteristics(self, 
                                     ecg_signals: np.ndarray,
                                     sampling_rate: int = 125) -> Dict[str, Any]:
        """
        Analyze ECG signal characteristics.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            sampling_rate (int): Sampling rate
            
        Returns:
            dict: Signal characteristics
        """
        analysis = {}
        
        try:
            # Basic statistics
            analysis['mean_amplitude'] = np.mean(ecg_signals)
            analysis['std_amplitude'] = np.std(ecg_signals)
            analysis['min_amplitude'] = np.min(ecg_signals)
            analysis['max_amplitude'] = np.max(ecg_signals)
            
            # Signal length
            analysis['signal_length'] = ecg_signals.shape[1]
            analysis['duration_seconds'] = ecg_signals.shape[1] / sampling_rate
            
            # Frequency analysis
            freqs, psd = signal.welch(ecg_signals[0], fs=sampling_rate, nperseg=256)
            analysis['dominant_frequency'] = freqs[np.argmax(psd)]
            analysis['mean_power'] = np.mean(psd)
            
            # Signal quality metrics
            quality_metrics = self._assess_signal_quality(ecg_signals)
            analysis.update(quality_metrics)
            
        except Exception as e:
            logger.warning(f"Error analyzing signal characteristics: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _assess_signal_quality(self, ecg_signals: np.ndarray) -> Dict[str, float]:
        """
        Assess ECG signal quality.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            
        Returns:
            dict: Signal quality metrics
        """
        metrics = {}
        
        try:
            # Calculate signal-to-noise ratio
            snr_scores = []
            for signal in ecg_signals:
                # Simple SNR estimation
                signal_power = np.mean(signal ** 2)
                noise_estimate = np.std(np.diff(signal))
                noise_power = noise_estimate ** 2
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                snr_scores.append(snr)
            
            metrics['mean_snr'] = np.mean(snr_scores)
            metrics['std_snr'] = np.std(snr_scores)
            metrics['min_snr'] = np.min(snr_scores)
            metrics['max_snr'] = np.max(snr_scores)
            
            # Signal variability
            signal_vars = np.var(ecg_signals, axis=1)
            metrics['mean_variance'] = np.mean(signal_vars)
            metrics['std_variance'] = np.std(signal_vars)
            
            # Signal range
            signal_ranges = np.ptp(ecg_signals, axis=1)
            metrics['mean_range'] = np.mean(signal_ranges)
            metrics['std_range'] = np.std(signal_ranges)
            
        except Exception as e:
            logger.warning(f"Error assessing signal quality: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def detect_artifacts(self, 
                        ecg_signals: np.ndarray,
                        threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect artifacts in ECG signals.
        
        Args:
            ecg_signals (np.ndarray): ECG signals
            threshold (float): Threshold for artifact detection
            
        Returns:
            dict: Artifact detection results
        """
        results = {}
        
        try:
            # Calculate signal statistics
            signal_means = np.mean(ecg_signals, axis=1)
            signal_stds = np.std(ecg_signals, axis=1)
            
            # Detect outliers based on standard deviation
            outlier_mask = np.abs(signal_means) > threshold * np.std(signal_means)
            outlier_count = np.sum(outlier_mask)
            
            results['total_signals'] = len(ecg_signals)
            results['outlier_count'] = outlier_count
            results['outlier_percentage'] = (outlier_count / len(ecg_signals)) * 100
            results['outlier_indices'] = np.where(outlier_mask)[0].tolist()
            
            # Detect flat signals (no variation)
            flat_mask = signal_stds < 0.01
            flat_count = np.sum(flat_mask)
            
            results['flat_signals'] = flat_count
            results['flat_percentage'] = (flat_count / len(ecg_signals)) * 100
            results['flat_indices'] = np.where(flat_mask)[0].tolist()
            
        except Exception as e:
            logger.warning(f"Error detecting artifacts: {e}")
            results['error'] = str(e)
        
        return results


def main():
    """Example usage of ECG utilities."""
    # Generate dummy ECG data for testing
    np.random.seed(42)
    n_signals = 100
    signal_length = 1000
    
    # Generate synthetic ECG-like signals
    ecg_signals = []
    for i in range(n_signals):
        # Create a synthetic ECG-like signal
        t = np.linspace(0, 8, signal_length)  # 8 seconds
        heart_rate = 60 + np.random.normal(0, 10)  # BPM with variation
        
        # Generate R-peaks
        r_peaks = np.arange(0, 8, 60/heart_rate)
        
        # Create signal
        signal = np.zeros_like(t)
        for peak_time in r_peaks:
            if peak_time < 8:
                # Add R-peak
                peak_idx = int(peak_time * signal_length / 8)
                if peak_idx < signal_length:
                    signal[peak_idx] = 1.0
        
        # Add noise
        noise = np.random.normal(0, 0.1, signal_length)
        signal = signal + noise
        
        # Smooth the signal
        from scipy.ndimage import gaussian_filter1d
        signal = gaussian_filter1d(signal, sigma=2)
        
        ecg_signals.append(signal)
    
    ecg_signals = np.array(ecg_signals)
    labels = np.random.randint(0, 2, n_signals)
    predictions = np.random.randint(0, 2, n_signals)
    probabilities = np.random.rand(n_signals, 2)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Test ECG evaluator
    evaluator = ECGEvaluator()
    metrics = evaluator.calculate_ecg_metrics(labels, predictions, probabilities, ecg_signals)
    print("ECG Metrics:", metrics)
    
    # Test ECG visualizer
    visualizer = ECGVisualizer()
    
    # Plot ECG signals
    fig1 = visualizer.plot_ecg_signals(ecg_signals, labels, predictions, n_samples=6)
    plt.show()
    
    # Plot spectrograms
    fig2 = visualizer.plot_ecg_spectrogram(ecg_signals, n_samples=4)
    plt.show()
    
    # Test ECG analyzer
    analyzer = ECGAnalyzer()
    characteristics = analyzer.analyze_signal_characteristics(ecg_signals)
    print("Signal Characteristics:", characteristics)
    
    artifacts = analyzer.detect_artifacts(ecg_signals)
    print("Artifact Detection:", artifacts)


if __name__ == "__main__":
    main()
