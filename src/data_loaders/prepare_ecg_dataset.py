"""
ECG Dataset Preparation Script
=============================

This script prepares the ECG dataset from ECGCvdata.csv for cardiovascular disease prediction.
It handles dataset loading, preprocessing, and preparation for training.

Usage:
    python prepare_ecg_dataset.py --csv_file ECGCvdata.csv --output_dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from ecg_data_loader import ECGDatasetLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to prepare the ECG dataset."""
    parser = argparse.ArgumentParser(description='Prepare ECG dataset from ECGCvdata.csv')
    parser.add_argument('--csv_file', type=str, default='ECGCvdata.csv',
                       help='Path to the ECG CSV file')
    parser.add_argument('--output_dir', type=str, default='preprocessed_ecg_data',
                       help='Path to save preprocessed data')
    parser.add_argument('--target_length', type=int, default=1000,
                       help='Target length for ECG signals')
    parser.add_argument('--sampling_rate', type=int, default=125,
                       help='Sampling rate in Hz')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--visualize', action='store_true',
                       help='Create dataset visualizations')
    parser.add_argument('--save_format', type=str, default='numpy',
                       choices=['numpy', 'pickle', 'h5'],
                       help='Format to save preprocessed data')
    parser.add_argument('--create_spectrograms', action='store_true',
                       help='Create spectrograms from ECG signals')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        logger.error(f"ECG CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    # Initialize the dataset loader
    logger.info("Initializing ECG dataset loader...")
    loader = ECGDatasetLoader(
        csv_file=args.csv_file,
        target_length=args.target_length,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size
    )
    
    try:
        # Load dataset
        logger.info("Loading ECG dataset...")
        dataset_stats = loader.load_dataset()
        logger.info(f"Dataset loaded successfully: {dataset_stats}")
        
        # Create data splits
        logger.info("Creating data splits...")
        split_info = loader.create_data_splits()
        logger.info(f"Data splits created: {split_info}")
        
        # Visualize dataset if requested
        if args.visualize:
            logger.info("Creating dataset visualizations...")
            loader.visualize_dataset()
        
        # Save preprocessed data
        logger.info("Saving preprocessed data...")
        saved_files = loader.save_preprocessed_data(
            output_dir=args.output_dir,
            save_format=args.save_format
        )
        logger.info(f"Preprocessed data saved: {saved_files}")
        
        # Test data generator
        logger.info("Testing data generator...")
        train_gen = loader.get_data_generator('train')
        logger.info(f"Training data generator created with {len(train_gen)} batches")
        
        # Test a batch
        for batch_data in train_gen:
            if len(batch_data) == 3:
                batch_signals, batch_labels, batch_features = batch_data
                logger.info(f"Test batch - Signals: {batch_signals.shape}, Labels: {batch_labels.shape}, Features: {batch_features.shape}")
            else:
                batch_signals, batch_labels = batch_data
                logger.info(f"Test batch - Signals: {batch_signals.shape}, Labels: {batch_labels.shape}")
            break
        
        logger.info("ECG dataset preparation completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("ECG DATASET PREPARATION SUMMARY")
        print("="*60)
        print(f"CSV file: {args.csv_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Target signal length: {args.target_length}")
        print(f"Sampling rate: {args.sampling_rate}")
        print(f"Batch size: {args.batch_size}")
        print(f"Save format: {args.save_format}")
        print(f"Total records: {dataset_stats['total_records']}")
        print(f"Valid signals: {dataset_stats['valid_signals']}")
        print(f"CVD positive: {dataset_stats['cvd_positive']}")
        print(f"CVD negative: {dataset_stats['cvd_negative']}")
        print(f"Class balance: {dataset_stats['class_balance']:.3f}")
        print(f"Mean signal length: {dataset_stats['mean_signal_length']:.1f}")
        print(f"Feature count: {dataset_stats['feature_count']}")
        print("\nData splits:")
        for split, info in split_info.items():
            print(f"  {split}: {info['size']} signals ({info['cvd_positive']} CVD positive)")
        print("\nSaved files:")
        for name, path in saved_files.items():
            print(f"  {name}: {path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during ECG dataset preparation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
