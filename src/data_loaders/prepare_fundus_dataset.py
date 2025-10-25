"""
Dataset Preparation Script for China-Fundus-CIMT
===============================================

This script prepares the China-Fundus-CIMT dataset for cardiovascular disease prediction.
It handles dataset organization, preprocessing, and preparation for training.

Usage:
    python prepare_fundus_dataset.py --data_dir /path/to/dataset --output_dir /path/to/output
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from fundus_data_loader import FundusDatasetLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_dataset_structure(data_dir: str):
    """
    Create a sample dataset structure if the dataset doesn't exist.
    This is for demonstration purposes.
    """
    logger.info("Creating sample dataset structure...")
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = data_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create sample labels.csv
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    sample_files = [f"sample_{i:04d}.jpg" for i in range(100)]
    sample_labels = np.random.randint(0, 2, 100)
    
    labels_df = pd.DataFrame({
        'filename': sample_files,
        'label': sample_labels,
        'patient_id': [f"P{i:04d}" for i in range(100)],
        'age': np.random.randint(40, 80, 100),
        'gender': np.random.choice(['M', 'F'], 100)
    })
    
    labels_df.to_csv(data_path / "labels.csv", index=False)
    logger.info(f"Created sample labels.csv with {len(labels_df)} entries")
    
    # Create sample metadata
    import json
    metadata = {
        "dataset_name": "China-Fundus-CIMT",
        "description": "Sample dataset for cardiovascular disease prediction",
        "total_images": len(sample_files),
        "image_format": "JPEG",
        "image_size": "Variable",
        "classes": ["No CVD", "CVD"],
        "class_distribution": {
            "No CVD": int(np.sum(sample_labels == 0)),
            "CVD": int(np.sum(sample_labels == 1))
        }
    }
    
    with open(data_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Created sample metadata.json")
    
    # Create README
    readme_content = """
# China-Fundus-CIMT Dataset

This is a sample dataset structure for the China-Fundus-CIMT dataset.

## Directory Structure
```
data/
├── images/           # Fundus images
├── labels.csv        # Image labels and metadata
├── metadata.json     # Dataset metadata
└── README.md         # This file
```

## Dataset Information
- **Total Images**: 100 (sample)
- **Classes**: 2 (No CVD, CVD)
- **Format**: JPEG
- **Purpose**: Cardiovascular disease prediction from retinal fundus images

## Usage
Use the `fundus_data_loader.py` module to load and preprocess this dataset.

## Note
This is a sample dataset structure. Replace with actual China-Fundus-CIMT data.
"""
    
    with open(data_path / "README.md", 'w') as f:
        f.write(readme_content)
    
    logger.info("Created README.md")
    logger.info(f"Sample dataset structure created in {data_dir}")
    logger.info("Note: This is a sample structure. Replace with actual dataset files.")

def main():
    """Main function to prepare the dataset."""
    parser = argparse.ArgumentParser(description='Prepare China-Fundus-CIMT dataset')
    parser.add_argument('--data_dir', type=str, default='data/china_fundus_cimt',
                       help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data',
                       help='Path to save preprocessed data')
    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                       help='Target image size (height width)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset structure if dataset not found')
    parser.add_argument('--visualize', action='store_true',
                       help='Create dataset visualizations')
    parser.add_argument('--save_format', type=str, default='numpy',
                       choices=['numpy', 'pickle', 'h5'],
                       help='Format to save preprocessed data')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        if args.create_sample:
            logger.info("Dataset directory not found. Creating sample structure...")
            create_sample_dataset_structure(args.data_dir)
        else:
            logger.error(f"Dataset directory not found: {args.data_dir}")
            logger.error("Use --create_sample to create a sample structure")
            sys.exit(1)
    
    # Initialize the dataset loader
    logger.info("Initializing dataset loader...")
    loader = FundusDatasetLoader(
        data_dir=args.data_dir,
        target_size=tuple(args.target_size),
        batch_size=args.batch_size
    )
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
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
        for batch_images, batch_labels in train_gen:
            logger.info(f"Test batch - Images shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")
            break
        
        logger.info("Dataset preparation completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("DATASET PREPARATION SUMMARY")
        print("="*60)
        print(f"Dataset directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Target image size: {args.target_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Save format: {args.save_format}")
        print(f"Total images: {dataset_stats['total_images']}")
        print(f"CVD positive: {dataset_stats['cvd_positive']}")
        print(f"CVD negative: {dataset_stats['cvd_negative']}")
        print(f"Class balance: {dataset_stats['class_balance']:.3f}")
        print("\nData splits:")
        for split, info in split_info.items():
            print(f"  {split}: {info['size']} images ({info['cvd_positive']} CVD positive)")
        print("\nSaved files:")
        for name, path in saved_files.items():
            print(f"  {name}: {path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during dataset preparation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
