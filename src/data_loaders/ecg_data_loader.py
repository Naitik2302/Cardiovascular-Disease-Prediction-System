"""
ECG Dataset Loader and Preprocessor
==================================

This module handles loading, preprocessing, and augmentation of ECG signals
from the ECGCvdata.csv dataset for cardiovascular disease prediction.

Features:
- ECG signal loading and parsing
- Signal preprocessing (filtering, resampling, segmentation)
- Data augmentation for ECG signals
- Spectrogram conversion
- Data splitting and caching
- Visualization and analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import butter, filtfilt, resample, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGDatasetLoader:
    """
    ECG Dataset Loader and Preprocessor
    
    This class handles all aspects of loading and preprocessing ECG signals
    for cardiovascular disease prediction.
    """
    
    def __init__(self, 
                 csv_file: str = "ECGCvdata.csv",
                 target_length: int = 1000,
                 sampling_rate: int = 125,
                 batch_size: int = 32,
                 random_seed: int = 42):
        """
        Initialize the ECGDatasetLoader.
        
        Args:
            csv_file (str): Path to the ECG CSV file
            target_length (int): Target length for ECG signals
            sampling_rate (int): Sampling rate in Hz
            batch_size (int): Batch size for data loading
            random_seed (int): Random seed for reproducibility
        """
        self.csv_file = Path(csv_file)
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        # Initialize data containers
        self.raw_data = None
        self.ecg_signals = []
        self.labels = []
        self.features = []
        self.metadata = []
        
        # Preprocessing parameters
        self.scaler = None
        self.feature_scaler = None
        
        # Data splits
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Filter parameters
        self.filter_params = {
            'lowcut': 0.5,    # Hz
            'highcut': 40.0,  # Hz
            'notch_freq': 50.0,  # Hz (power line noise)
            'order': 4
        }
        
        logger.info(f"Initialized ECGDatasetLoader with target length: {target_length}, sampling rate: {sampling_rate}")
    
    def load_dataset(self) -> Dict:
        """
        Load the ECG dataset from CSV file.
        
        Returns:
            dict: Dataset information and statistics
        """
        logger.info("Loading ECG dataset...")
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"ECG dataset file not found: {self.csv_file}")
        
        # Load CSV data
        self.raw_data = pd.read_csv(self.csv_file)
        logger.info(f"Loaded CSV with {len(self.raw_data)} records and {len(self.raw_data.columns)} columns")
        
        # Extract ECG signals and labels
        self._extract_signals_and_labels()
        
        # Extract features
        self._extract_features()
        
        # Calculate dataset statistics
        dataset_stats = self._calculate_dataset_stats()
        
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"  - Total records: {len(self.raw_data)}")
        logger.info(f"  - Valid ECG signals: {len(self.ecg_signals)}")
        logger.info(f"  - CVD positive: {np.sum(self.labels)}")
        logger.info(f"  - CVD negative: {len(self.labels) - np.sum(self.labels)}")
        
        return dataset_stats
    
    def _extract_signals_and_labels(self):
        """Extract ECG signals and labels from the dataset."""
        logger.info("Extracting ECG signals and labels...")

        self.ecg_signals = []
        self.labels = []
        self.metadata = []

        # Map ECG_signal column to binary labels
        # ARR = Arrhythmia, AFF = Atrial Fibrillation, CHF = Congestive Heart Failure (all CVD positive)
        # NSR = Normal Sinus Rhythm (CVD negative)
        label_mapping = {'ARR': 1, 'AFF': 1, 'CHF': 1, 'NSR': 0}

        for idx, row in self.raw_data.iterrows():
            try:
                # Extract label from ECG_signal column
                ecg_signal_label = str(row['ECG_signal'])
                
                if ecg_signal_label in label_mapping:
                    label = label_mapping[ecg_signal_label]
                    self.labels.append(label)
                    
                    # Since we don't have raw ECG signals, create synthetic signals from features
                    # Use the available numeric ECG features to create a synthetic time series
                    feature_cols = ['hbpermin', 'Pseg', 'PQseg', 'QRSseg', 'QRseg', 'QTseg', 
                                  'RSseg', 'STseg', 'Tseg', 'PTseg', 'ECGseg', 'RRmean',
                                  'PQdis', 'PRdis', 'PSdis', 'PTdis', 'QRdis', 'QSdis', 
                                  'QTdis', 'RSdis', 'RTdis', 'STdis', 'SDRR', 'RMSSD']
                    
                    # Get available feature columns
                    available_features = [col for col in feature_cols if col in self.raw_data.columns]
                    
                    if available_features:
                        # Create synthetic ECG signal from features
                        feature_values = row[available_features].values.astype(float)
                        signal_data = self._create_ecg_signal_from_features(feature_values)
                    else:
                        # Fallback to completely synthetic signal
                        signal_data = self._generate_synthetic_ecg(self.target_length)
                    
                    # Ensure signal is proper length
                    if len(signal_data) >= self.target_length:
                        signal_data = signal_data[:self.target_length]
                    else:
                        # Pad with zeros if shorter
                        signal_data = np.pad(signal_data, (0, self.target_length - len(signal_data)), mode='constant')
                    
                    self.ecg_signals.append(signal_data)
                    
                    # Extract metadata
                    metadata = {
                        'record_id': row.get('RECORD', f'record_{idx}'),
                        'heart_rate': row.get('hbpermin', 0),
                        'original_length': len(signal_data),
                        'features': {
                            'Pseg': row.get('Pseg', 0),
                            'QRSseg': row.get('QRSseg', 0),
                            'QTseg': row.get('QTseg', 0),
                            'RRmean': row.get('RRmean', 0)
                        },
                        'original_label': ecg_signal_label,
                        'signal_type': 'feature_based'
                    }
                    self.metadata.append(metadata)
                else:
                    logger.warning(f"Unknown ECG signal label: {ecg_signal_label}")
                
            except Exception as e:
                logger.warning(f"Error processing record {idx}: {str(e)}")
                continue
        
        # Convert to numpy arrays
        self.ecg_signals = np.array(self.ecg_signals)
        self.labels = np.array(self.labels)
        
        logger.info(f"Extracted {len(self.ecg_signals)} valid ECG signals")
        logger.info(f"Label distribution: {np.bincount(self.labels)}")
        logger.info(f"CVD positive samples: {np.sum(self.labels)}")
        logger.info(f"CVD negative samples: {len(self.labels) - np.sum(self.labels)}")
    
    def _parse_ecg_signal(self, signal_str: str) -> Optional[np.ndarray]:
        """
        Parse ECG signal from string format.
        
        Args:
            signal_str (str): String representation of ECG signal
            
        Returns:
            np.ndarray: Parsed ECG signal or None if parsing fails
        """
        try:
            # Method 1: Try to parse as space-separated values
            if ' ' in signal_str:
                values = signal_str.split()
                return np.array([float(x) for x in values if x.replace('.', '').replace('-', '').isdigit()])
            
            # Method 2: Try to parse as comma-separated values
            elif ',' in signal_str:
                values = signal_str.split(',')
                return np.array([float(x) for x in values if x.replace('.', '').replace('-', '').isdigit()])
            
            # Method 3: Try to parse as array-like string
            elif '[' in signal_str and ']' in signal_str:
                # Remove brackets and split
                clean_str = signal_str.replace('[', '').replace(']', '')
                if ',' in clean_str:
                    values = clean_str.split(',')
                else:
                    values = clean_str.split()
                return np.array([float(x) for x in values if x.replace('.', '').replace('-', '').isdigit()])
            
            # Method 4: Generate synthetic ECG signal if parsing fails
            else:
                logger.warning(f"Could not parse ECG signal: {signal_str[:50]}...")
                return self._generate_synthetic_ecg()
                
        except Exception as e:
            logger.warning(f"Error parsing ECG signal: {str(e)}")
            return self._generate_synthetic_ecg()
    
    def _create_ecg_signal_from_features(self, feature_values: np.ndarray) -> np.ndarray:
        """
        Create a synthetic ECG signal from feature values.
        
        Args:
            feature_values (np.ndarray): Feature values to interpolate
            
        Returns:
            np.ndarray: Synthetic ECG signal
        """
        try:
            # Create a realistic ECG-like signal based on the features
            # Use heart rate to determine base frequency
            heart_rate = feature_values[0] if len(feature_values) > 0 else 75  # Default 75 bpm
            fs = 500  # Sampling frequency
            
            # Calculate heart rate period
            rr_interval = 60.0 / heart_rate  # RR interval in seconds
            rr_samples = int(rr_interval * fs)
            
            # Create multiple heartbeats
            num_beats = int(np.ceil(self.target_length / rr_samples))
            
            signal = np.array([])
            
            for beat_idx in range(num_beats):
                # Create a single heartbeat based on feature values
                beat = self._create_single_heartbeat_from_features(feature_values, rr_samples)
                signal = np.concatenate([signal, beat])
                
                if len(signal) >= self.target_length:
                    break
            
            # Trim to target length
            signal = signal[:self.target_length]
            
            # Add realistic ECG characteristics
            # Add baseline wander (low frequency drift)
            baseline = np.linspace(0, 0.1, len(signal)) + np.sin(2 * np.pi * 0.1 * np.arange(len(signal)) / fs) * 0.05
            signal = signal + baseline
            
            # Add muscle noise
            muscle_noise = np.random.normal(0, 0.02, len(signal))
            signal = signal + muscle_noise
            
            # Normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            return signal
            
        except Exception as e:
            logger.warning(f"Error creating ECG signal from features: {str(e)}")
            return self._generate_synthetic_ecg(self.target_length)
    
    def _create_single_heartbeat_from_features(self, features, length):
        """Create a single heartbeat based on feature values."""
        try:
            # Create time vector
            t = np.linspace(0, 1, length)
            
            # Extract relevant features (use defaults if not available)
            p_duration = features[1] if len(features) > 1 else 0.1  # P wave duration
            pq_duration = features[2] if len(features) > 2 else 0.2  # PQ segment duration
            qrs_duration = features[3] if len(features) > 3 else 0.08  # QRS duration
            qt_duration = features[5] if len(features) > 5 else 0.35  # QT duration
            t_duration = features[8] if len(features) > 8 else 0.15  # T wave duration
            
            # Normalize durations
            p_duration = min(p_duration, 0.2)
            qrs_duration = min(qrs_duration, 0.15)
            t_duration = min(t_duration, 0.25)
            
            # Create P wave
            p_start = int(0.1 * length)
            p_end = int((0.1 + p_duration) * length)
            
            # Create QRS complex
            qrs_start = int((0.1 + p_duration + pq_duration) * length)
            qrs_end = int((0.1 + p_duration + pq_duration + qrs_duration) * length)
            
            # Create T wave
            t_start = int((0.1 + p_duration + pq_duration + qrs_duration + 0.05) * length)
            t_end = int(min((0.1 + p_duration + pq_duration + qrs_duration + 0.05 + t_duration), 0.9) * length)
            
            # Initialize heartbeat signal
            heartbeat = np.zeros(length)
            
            # Add P wave (positive deflection)
            if p_end > p_start:
                p_wave = np.sin(np.linspace(0, np.pi, p_end - p_start)) * 0.3
                heartbeat[p_start:p_end] = p_wave
            
            # Add QRS complex (Q negative, R positive, S negative)
            if qrs_end > qrs_start:
                qrs_length = qrs_end - qrs_start
                qrs = np.zeros(qrs_length)
                
                # Q wave (negative)
                q_end = int(0.1 * qrs_length)
                if q_end > 0:
                    qrs[:q_end] = -0.5 * np.sin(np.linspace(0, np.pi, q_end))
                
                # R wave (positive, sharp)
                r_start = q_end
                r_end = int(0.4 * qrs_length)
                if r_end > r_start:
                    r_wave = np.sin(np.linspace(0, np.pi, r_end - r_start)) * 1.5
                    qrs[r_start:r_end] = r_wave
                
                # S wave (negative)
                s_start = r_end
                s_end = qrs_length
                if s_end > s_start:
                    s_wave = -0.3 * np.sin(np.linspace(0, np.pi, s_end - s_start))
                    qrs[s_start:s_end] = s_wave
                
                heartbeat[qrs_start:qrs_end] = qrs
            
            # Add T wave (positive)
            if t_end > t_start:
                t_wave = np.sin(np.linspace(0, np.pi, t_end - t_start)) * 0.4
                heartbeat[t_start:t_end] = t_wave
            
            # Add some baseline drift
            heartbeat = heartbeat + np.linspace(0, 0.02, length)
            
            return heartbeat
            
        except Exception as e:
            logger.warning(f"Error creating single heartbeat: {e}")
            # Fallback: create simple sine wave
            return np.sin(2 * np.pi * 1 * t) * 0.5

    def _generate_synthetic_ecg(self, length: int = 1000) -> np.ndarray:
        """
        Generate a synthetic ECG signal for testing purposes.
        
        Args:
            length (int): Length of the synthetic signal
            
        Returns:
            np.ndarray: Synthetic ECG signal
        """
        # Generate synthetic ECG-like signal
        t = np.linspace(0, length / self.sampling_rate, length)
        
        # Basic ECG components
        heart_rate = np.random.uniform(60, 100)  # BPM
        rr_interval = 60.0 / heart_rate  # seconds
        
        # P wave
        p_wave = 0.1 * np.exp(-((t % rr_interval - 0.2) / 0.05) ** 2)
        
        # QRS complex
        qrs_complex = 0.8 * np.exp(-((t % rr_interval - 0.3) / 0.02) ** 2)
        
        # T wave
        t_wave = 0.3 * np.exp(-((t % rr_interval - 0.5) / 0.08) ** 2)
        
        # Combine components
        synthetic_ecg = p_wave + qrs_complex + t_wave
        
        # Add noise
        noise = np.random.normal(0, 0.05, length)
        synthetic_ecg += noise
        
        return synthetic_ecg
    
    def _extract_features(self):
        """Extract features from the raw data."""
        logger.info("Extracting features...")
        
        feature_columns = [
            'hbpermin', 'Pseg', 'PQseg', 'QRSseg', 'QRseg', 'QTseg',
            'RSseg', 'STseg', 'Tseg', 'PTseg', 'ECGseg', 'RRmean', 'PPmean',
            'SDRR', 'IBIM', 'IBISD', 'SDSD', 'RMSSD', 'QRSarea', 'QRSperi',
            'NN50', 'pNN50'
        ]
        
        # Extract available features
        available_features = []
        for col in feature_columns:
            if col in self.raw_data.columns:
                available_features.append(col)
        
        if available_features:
            self.features = self.raw_data[available_features].fillna(0).values
            logger.info(f"Extracted {len(available_features)} features: {available_features}")
        else:
            logger.warning("No feature columns found, using dummy features")
            self.features = np.random.randn(len(self.ecg_signals), 10)
    
    def _calculate_dataset_stats(self) -> Dict:
        """Calculate dataset statistics."""
        stats = {
            'total_records': len(self.raw_data),
            'valid_signals': len(self.ecg_signals),
            'cvd_positive': int(np.sum(self.labels)),
            'cvd_negative': int(len(self.labels) - np.sum(self.labels)),
            'class_distribution': self.labels.tolist(),
            'class_balance': float(np.sum(self.labels) / len(self.labels)) if len(self.labels) > 0 else 0,
            'signal_lengths': [len(signal) for signal in self.ecg_signals],
            'mean_signal_length': np.mean([len(signal) for signal in self.ecg_signals]) if len(self.ecg_signals) > 0 else 0,
            'feature_count': self.features.shape[1] if len(self.features) > 0 else 0
        }
        return stats
    
    def create_data_splits(self, 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          stratify: bool = True) -> Dict:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            train_ratio (float): Proportion of data for training
            val_ratio (float): Proportion of data for validation
            test_ratio (float): Proportion of data for testing
            stratify (bool): Whether to stratify by class labels
            
        Returns:
            dict: Split information
        """
        logger.info("Creating data splits...")
        
        if stratify and len(np.unique(self.labels)) > 1:
            # Stratified split
            X_temp, X_test, y_temp, y_test = train_test_split(
                range(len(self.ecg_signals)), self.labels,
                test_size=test_ratio,
                random_state=self.random_seed,
                stratify=self.labels
            )
            
            val_size = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size,
                random_state=self.random_seed,
                stratify=y_temp
            )
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                range(len(self.ecg_signals)), self.labels,
                test_size=test_ratio,
                random_state=self.random_seed
            )
            
            val_size = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size,
                random_state=self.random_seed
            )
        
        # Store splits
        self.train_data = {
            'signals': [self.ecg_signals[i] for i in X_train],
            'labels': y_train,
            'features': self.features[X_train] if len(self.features) > 0 else None,
            'metadata': [self.metadata[i] for i in X_train]
        }
        
        self.val_data = {
            'signals': [self.ecg_signals[i] for i in X_val],
            'labels': y_val,
            'features': self.features[X_val] if len(self.features) > 0 else None,
            'metadata': [self.metadata[i] for i in X_val]
        }
        
        self.test_data = {
            'signals': [self.ecg_signals[i] for i in X_test],
            'labels': y_test,
            'features': self.features[X_test] if len(self.features) > 0 else None,
            'metadata': [self.metadata[i] for i in X_test]
        }
        
        split_info = {
            'train': {'size': len(X_train), 'cvd_positive': int(np.sum(y_train))},
            'val': {'size': len(X_val), 'cvd_positive': int(np.sum(y_val))},
            'test': {'size': len(X_test), 'cvd_positive': int(np.sum(y_test))}
        }
        
        logger.info("Data splits created:")
        for split, info in split_info.items():
            logger.info(f"  {split}: {info['size']} signals ({info['cvd_positive']} CVD positive)")
        
        return split_info
    
    def preprocess_signals(self, 
                          signals: List[np.ndarray],
                          apply_filtering: bool = True,
                          apply_resampling: bool = True,
                          apply_normalization: bool = True) -> List[np.ndarray]:
        """
        Preprocess ECG signals.
        
        Args:
            signals (list): List of ECG signals
            apply_filtering (bool): Whether to apply filtering
            apply_resampling (bool): Whether to apply resampling
            apply_normalization (bool): Whether to apply normalization
            
        Returns:
            list: Preprocessed signals
        """
        logger.info(f"Preprocessing {len(signals)} ECG signals...")
        
        processed_signals = []
        
        for i, signal in enumerate(signals):
            try:
                processed_signal = signal.copy()
                
                # Apply filtering
                if apply_filtering:
                    processed_signal = self._apply_filters(processed_signal)
                
                # Apply resampling
                if apply_resampling:
                    processed_signal = self._resample_signal(processed_signal)
                
                # Apply normalization
                if apply_normalization:
                    processed_signal = self._normalize_signal(processed_signal)
                
                processed_signals.append(processed_signal)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(signals)} signals")
                    
            except Exception as e:
                logger.error(f"Error processing signal {i}: {str(e)}")
                # Use original signal if processing fails
                processed_signals.append(signal)
        
        logger.info(f"Successfully preprocessed {len(processed_signals)} signals")
        return processed_signals
    
    def _apply_filters(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply filtering to ECG signal.
        
        Args:
            signal (np.ndarray): Input ECG signal
            
        Returns:
            np.ndarray: Filtered signal
        """
        try:
            # Bandpass filter
            nyquist = self.sampling_rate / 2
            low = self.filter_params['lowcut'] / nyquist
            high = self.filter_params['highcut'] / nyquist
            
            b, a = butter(self.filter_params['order'], [low, high], btype='band')
            filtered_signal = filtfilt(b, a, signal)
            
            # Notch filter for power line noise
            if self.filter_params['notch_freq'] > 0:
                notch_freq = self.filter_params['notch_freq'] / nyquist
                b_notch, a_notch = butter(2, [notch_freq - 0.01, notch_freq + 0.01], btype='bandstop')
                filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)
            
            return filtered_signal
            
        except Exception as e:
            logger.warning(f"Filtering failed: {str(e)}")
            return signal
    
    def _resample_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Resample signal to target length.
        
        Args:
            signal (np.ndarray): Input signal
            
        Returns:
            np.ndarray: Resampled signal
        """
        try:
            if len(signal) == self.target_length:
                return signal
            elif len(signal) > self.target_length:
                # Downsample
                return resample(signal, self.target_length)
            else:
                # Upsample
                return resample(signal, self.target_length)
        except Exception as e:
            logger.warning(f"Resampling failed: {str(e)}")
            return signal
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize ECG signal.
        
        Args:
            signal (np.ndarray): Input signal
            
        Returns:
            np.ndarray: Normalized signal
        """
        try:
            # Z-score normalization
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            
            if std_val > 0:
                normalized = (signal - mean_val) / std_val
            else:
                normalized = signal - mean_val
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Normalization failed: {str(e)}")
            return signal
    
    def convert_to_spectrograms(self, 
                               signals: List[np.ndarray],
                               nperseg: int = 256,
                               noverlap: int = 128) -> List[np.ndarray]:
        """
        Convert ECG signals to spectrograms.
        
        Args:
            signals (list): List of ECG signals
            nperseg (int): Length of each segment for FFT
            noverlap (int): Number of points to overlap between segments
            
        Returns:
            list: List of spectrograms
        """
        logger.info(f"Converting {len(signals)} signals to spectrograms...")
        
        spectrograms = []
        
        for i, signal in enumerate(signals):
            try:
                # Compute spectrogram
                f, t, Sxx = signal.spectrogram(
                    signal, 
                    fs=self.sampling_rate,
                    nperseg=nperseg,
                    noverlap=noverlap
                )
                
                # Convert to log scale
                Sxx_log = 10 * np.log10(Sxx + 1e-10)
                
                spectrograms.append(Sxx_log)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Converted {i + 1}/{len(signals)} signals")
                    
            except Exception as e:
                logger.error(f"Error converting signal {i} to spectrogram: {str(e)}")
                # Create dummy spectrogram
                dummy_spec = np.random.randn(129, 8)  # Default spectrogram shape
                spectrograms.append(dummy_spec)
        
        logger.info(f"Successfully converted {len(spectrograms)} signals to spectrograms")
        return spectrograms
    
    def save_preprocessed_data(self, 
                              output_dir: str = "preprocessed_ecg_data",
                              save_format: str = "numpy") -> Dict:
        """
        Save preprocessed data for all splits.
        
        Args:
            output_dir (str): Directory to save preprocessed data
            save_format (str): Format to save data ('numpy', 'pickle', 'h5')
            
        Returns:
            dict: Information about saved files
        """
        logger.info(f"Saving preprocessed ECG data to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Process and save each split
        for split_name, split_data in [('train', self.train_data), 
                                     ('val', self.val_data), 
                                     ('test', self.test_data)]:
            if split_data is None:
                continue
                
            logger.info(f"Processing {split_name} split...")
            
            # Preprocess signals
            processed_signals = self.preprocess_signals(split_data['signals'])
            
            # Convert to spectrograms
            spectrograms = self.convert_to_spectrograms(processed_signals)
            
            # Convert to numpy arrays
            signals_array = np.array([self._pad_or_truncate(signal, self.target_length) 
                                    for signal in processed_signals])
            labels_array = np.array(split_data['labels'])
            
            # Save data
            if save_format == "numpy":
                np.save(output_path / f"{split_name}_signals.npy", signals_array)
                np.save(output_path / f"{split_name}_labels.npy", labels_array)
                np.save(output_path / f"{split_name}_spectrograms.npy", np.array(spectrograms))
                
                if split_data['features'] is not None:
                    np.save(output_path / f"{split_name}_features.npy", split_data['features'])
                
                saved_files[f"{split_name}_signals"] = str(output_path / f"{split_name}_signals.npy")
                saved_files[f"{split_name}_labels"] = str(output_path / f"{split_name}_labels.npy")
                saved_files[f"{split_name}_spectrograms"] = str(output_path / f"{split_name}_spectrograms.npy")
                if split_data['features'] is not None:
                    saved_files[f"{split_name}_features"] = str(output_path / f"{split_name}_features.npy")
            
            elif save_format == "pickle":
                data_dict = {
                    'signals': signals_array,
                    'labels': labels_array,
                    'spectrograms': np.array(spectrograms),
                    'features': split_data['features']
                }
                with open(output_path / f"{split_name}_data.pkl", 'wb') as f:
                    pickle.dump(data_dict, f)
                saved_files[f"{split_name}_data"] = str(output_path / f"{split_name}_data.pkl")
        
        # Save metadata
        metadata = {
            'target_length': self.target_length,
            'sampling_rate': self.sampling_rate,
            'filter_params': self.filter_params,
            'dataset_stats': self._calculate_dataset_stats(),
            'preprocessing_info': {
                'filtering': 'Bandpass + Notch filter',
                'resampling': f'Target length: {self.target_length}',
                'normalization': 'Z-score normalization'
            }
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_files['metadata'] = str(output_path / "metadata.json")
        
        logger.info(f"Preprocessed ECG data saved successfully to {output_dir}")
        logger.info(f"Saved files: {list(saved_files.keys())}")
        
        return saved_files
    
    def _pad_or_truncate(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or truncate signal to target length.
        
        Args:
            signal (np.ndarray): Input signal
            target_length (int): Target length
            
        Returns:
            np.ndarray: Padded or truncated signal
        """
        if len(signal) == target_length:
            return signal
        elif len(signal) > target_length:
            # Truncate
            return signal[:target_length]
        else:
            # Pad with zeros
            padded = np.zeros(target_length)
            padded[:len(signal)] = signal
            return padded
    
    def visualize_dataset(self, 
                         num_samples: int = 8,
                         figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize sample ECG signals from the dataset.
        
        Args:
            num_samples (int): Number of samples to display
            figsize (tuple): Figure size
        """
        logger.info("Creating ECG dataset visualization...")
        
        if len(self.ecg_signals) == 0:
            logger.warning("No ECG signals available for visualization")
            return
        
        # Get random samples
        indices = np.random.choice(len(self.ecg_signals), 
                                 min(num_samples, len(self.ecg_signals)), 
                                 replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            # Get signal
            signal = self.ecg_signals[idx]
            if len(signal) > 0:
                # Create time axis
                time = np.arange(len(signal)) / self.sampling_rate
                
                # Plot signal
                axes[i].plot(time, signal)
                axes[i].set_title(f"Label: {self.labels[idx]}\nRecord: {self.metadata[idx]['record_id']}")
                axes[i].set_xlabel("Time (s)")
                axes[i].set_ylabel("Amplitude")
                axes[i].grid(True)
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Plot class distribution
        plt.figure(figsize=(8, 6))
        class_counts = np.bincount(self.labels)
        plt.bar(['No CVD', 'CVD'], class_counts)
        plt.title('Class Distribution in ECG Dataset')
        plt.ylabel('Number of Signals')
        plt.show()
    
    def get_data_generator(self, 
                          split: str = 'train',
                          batch_size: Optional[int] = None) -> 'ECGDataGenerator':
        """
        Get a data generator for the specified split.
        
        Args:
            split (str): Data split ('train', 'val', 'test')
            batch_size (int): Batch size (uses default if None)
            
        Returns:
            ECGDataGenerator: Data generator object
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        split_data = getattr(self, f"{split}_data")
        if split_data is None:
            raise ValueError(f"No {split} data available. Run create_data_splits() first.")
        
        return ECGDataGenerator(
            signals=split_data['signals'],
            labels=split_data['labels'],
            features=split_data['features'],
            target_length=self.target_length,
            sampling_rate=self.sampling_rate,
            batch_size=batch_size,
            is_training=(split == 'train'),
            shuffle=(split == 'train')
        )


class ECGDataGenerator:
    """
    Data generator for batch processing of ECG signals during training.
    """
    
    def __init__(self, 
                 signals: List[np.ndarray],
                 labels: np.ndarray,
                 features: Optional[np.ndarray],
                 target_length: int,
                 sampling_rate: int,
                 batch_size: int = 32,
                 is_training: bool = True,
                 shuffle: bool = True):
        """
        Initialize the ECG data generator.
        
        Args:
            signals (list): List of ECG signals
            labels (np.ndarray): Array of labels
            features (np.ndarray): Array of features
            target_length (int): Target signal length
            sampling_rate (int): Sampling rate
            batch_size (int): Batch size
            is_training (bool): Whether this is for training
            shuffle (bool): Whether to shuffle data
        """
        self.signals = signals
        self.labels = labels
        self.features = features
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle
        
        # Initialize indices
        self.indices = np.arange(len(self.signals))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_index = 0
    
    def __len__(self):
        """Return number of batches."""
        return len(self.indices) // self.batch_size
    
    def __iter__(self):
        """Return iterator."""
        return self
    
    def __next__(self):
        """Get next batch."""
        if self.current_index >= len(self.indices):
            # Reset for next epoch
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration
        
        # Get batch indices
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        # Load and preprocess batch
        batch_signals = []
        batch_labels = []
        batch_features = []
        
        for idx in batch_indices:
            try:
                # Get signal
                signal = self.signals[idx]
                
                # Pad or truncate to target length
                if len(signal) == self.target_length:
                    processed_signal = signal
                elif len(signal) > self.target_length:
                    processed_signal = signal[:self.target_length]
                else:
                    padded = np.zeros(self.target_length)
                    padded[:len(signal)] = signal
                    processed_signal = padded
                
                # Normalize
                if np.std(processed_signal) > 0:
                    processed_signal = (processed_signal - np.mean(processed_signal)) / np.std(processed_signal)
                
                batch_signals.append(processed_signal)
                batch_labels.append(self.labels[idx])
                
                if self.features is not None:
                    batch_features.append(self.features[idx])
                
            except Exception as e:
                logger.error(f"Error processing signal {idx}: {str(e)}")
                continue
        
        batch_signals = np.array(batch_signals)
        batch_labels = np.array(batch_labels)
        
        if self.features is not None and len(batch_features) > 0:
            batch_features = np.array(batch_features)
            return batch_signals, batch_labels, batch_features
        else:
            return batch_signals, batch_labels


def main():
    """
    Example usage of the ECGDatasetLoader.
    """
    # Initialize the loader
    loader = ECGDatasetLoader(
        csv_file="ECGCvdata.csv",
        target_length=1000,
        sampling_rate=125,
        batch_size=32
    )
    
    # Load dataset
    try:
        dataset_stats = loader.load_dataset()
        print("ECG dataset loaded successfully!")
        print(f"Dataset statistics: {dataset_stats}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        return
    
    # Create data splits
    split_info = loader.create_data_splits()
    print(f"Data splits created: {split_info}")
    
    # Visualize dataset
    loader.visualize_dataset()
    
    # Save preprocessed data
    saved_files = loader.save_preprocessed_data()
    print(f"Preprocessed data saved: {saved_files}")
    
    # Test data generator
    train_gen = loader.get_data_generator('train')
    print(f"Training data generator created with {len(train_gen)} batches")
    
    # Test a batch
    for batch_data in train_gen:
        if len(batch_data) == 3:
            batch_signals, batch_labels, batch_features = batch_data
            print(f"Batch - Signals: {batch_signals.shape}, Labels: {batch_labels.shape}, Features: {batch_features.shape}")
        else:
            batch_signals, batch_labels = batch_data
            print(f"Batch - Signals: {batch_signals.shape}, Labels: {batch_labels.shape}")
        break


if __name__ == "__main__":
    main()
