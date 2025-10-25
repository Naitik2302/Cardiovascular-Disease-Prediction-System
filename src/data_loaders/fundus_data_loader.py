"""
China-Fundus-CIMT Dataset Loader and Preprocessor
=================================================

This module handles loading, preprocessing, and augmentation of the China-Fundus-CIMT dataset
for cardiovascular disease prediction using retinal fundus images.

Features:
- Dataset loading and organization
- Image preprocessing (resize, normalize)
- Data augmentation for training
- Data splitting (train/validation/test)
- Preprocessed data caching
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundusDatasetLoader:
    """
    China-Fundus-CIMT Dataset Loader and Preprocessor
    
    This class handles all aspects of loading and preprocessing the fundus dataset
    for cardiovascular disease prediction.
    """
    
    def __init__(self, 
                 data_dir: str,
                 target_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32,
                 random_seed: int = 42):
        """
        Initialize the FundusDatasetLoader.
        
        Args:
            data_dir (str): Path to the dataset directory
            target_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for data loading
            random_seed (int): Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
        # Initialize data containers
        self.image_paths = []
        self.labels = []
        self.metadata = []
        
        # Preprocessing parameters
        self.mean = None
        self.std = None
        
        # Data splits
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        logger.info(f"Initialized FundusDatasetLoader with target size: {target_size}")
    
    def load_dataset(self, 
                    image_folder: str = "images",
                    label_file: str = "labels.csv",
                    metadata_file: str = "metadata.json") -> Dict:
        """
        Load the China-Fundus-CIMT dataset.
        
        Args:
            image_folder (str): Name of the folder containing images
            label_file (str): Name of the label file
            metadata_file (str): Name of the metadata file
            
        Returns:
            dict: Dataset information and statistics
        """
        logger.info("Loading China-Fundus-CIMT dataset...")
        
        # Construct paths
        image_dir = self.data_dir / image_folder
        label_path = self.data_dir / label_file
        metadata_path = self.data_dir / metadata_file
        
        # Check if paths exist
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Load labels if available
        if label_path.exists():
            labels_df = pd.read_csv(label_path)
            logger.info(f"Loaded labels from {label_path}")
        else:
            logger.warning(f"Label file not found: {label_path}. Creating dummy labels.")
            labels_df = None
        
        # Load metadata if available
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            metadata = {}
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
            image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {image_dir}")
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Process images and labels
        self.image_paths = []
        self.labels = []
        self.metadata = []
        
        for img_path in image_files:
            self.image_paths.append(str(img_path))
            
            # Extract label if available
            if labels_df is not None:
                # Try to match by filename
                filename = img_path.stem
                matching_rows = labels_df[labels_df['filename'] == filename]
                
                if not matching_rows.empty:
                    label = matching_rows.iloc[0]['label']  # Assuming 'label' column
                    self.labels.append(label)
                else:
                    # Default label if not found
                    self.labels.append(0)
                    logger.warning(f"No label found for {filename}")
            else:
                # Create dummy labels (0: no CVD, 1: CVD)
                self.labels.append(np.random.randint(0, 2))
            
            # Add metadata
            self.metadata.append({
                'filename': img_path.name,
                'path': str(img_path),
                'size': None  # Will be filled during preprocessing
            })
        
        # Convert to numpy arrays
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        # Calculate dataset statistics
        dataset_stats = self._calculate_dataset_stats()
        
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"  - Total images: {len(self.image_paths)}")
        logger.info(f"  - CVD positive: {np.sum(self.labels)}")
        logger.info(f"  - CVD negative: {len(self.labels) - np.sum(self.labels)}")
        logger.info(f"  - Class distribution: {np.bincount(self.labels)}")
        
        return dataset_stats
    
    def _calculate_dataset_stats(self) -> Dict:
        """Calculate dataset statistics."""
        stats = {
            'total_images': len(self.image_paths),
            'cvd_positive': int(np.sum(self.labels)),
            'cvd_negative': int(len(self.labels) - np.sum(self.labels)),
            'class_distribution': self.labels.tolist(),
            'class_balance': float(np.sum(self.labels) / len(self.labels))
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
                self.image_paths, self.labels,
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
                self.image_paths, self.labels,
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
        self.train_data = {'images': X_train, 'labels': y_train}
        self.val_data = {'images': X_val, 'labels': y_val}
        self.test_data = {'images': X_test, 'labels': y_test}
        
        split_info = {
            'train': {'size': len(X_train), 'cvd_positive': int(np.sum(y_train))},
            'val': {'size': len(X_val), 'cvd_positive': int(np.sum(y_val))},
            'test': {'size': len(X_test), 'cvd_positive': int(np.sum(y_test))}
        }
        
        logger.info("Data splits created:")
        for split, info in split_info.items():
            logger.info(f"  {split}: {info['size']} images ({info['cvd_positive']} CVD positive)")
        
        return split_info
    
    def get_preprocessing_transforms(self, 
                                   is_training: bool = True,
                                   include_augmentation: bool = True) -> A.Compose:
        """
        Get preprocessing transforms for images.
        
        Args:
            is_training (bool): Whether this is for training
            include_augmentation (bool): Whether to include data augmentation
            
        Returns:
            albumentations.Compose: Preprocessing pipeline
        """
        transforms = []
        
        # Resize to target size
        transforms.append(A.Resize(height=self.target_size[0], width=self.target_size[1]))
        
        if is_training and include_augmentation:
            # Data augmentation for training
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=15, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.05, p=0.2)
            ])
        
        # Normalization
        transforms.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
            max_pixel_value=255.0
        ))
        
        # Convert to tensor
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def preprocess_images(self, 
                         image_paths: List[str],
                         labels: List[int],
                         is_training: bool = True,
                         save_preprocessed: bool = False,
                         output_dir: str = "preprocessed_data") -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess images with the specified transforms.
        
        Args:
            image_paths (list): List of image paths
            labels (list): List of corresponding labels
            is_training (bool): Whether this is for training
            save_preprocessed (bool): Whether to save preprocessed images
            output_dir (str): Directory to save preprocessed images
            
        Returns:
            tuple: (preprocessed_images, labels)
        """
        logger.info(f"Preprocessing {len(image_paths)} images...")
        
        # Get preprocessing transforms
        transform = self.get_preprocessing_transforms(is_training=is_training)
        
        # Create output directory if saving
        if save_preprocessed:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        preprocessed_images = []
        valid_labels = []
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Could not load image: {img_path}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                transformed = transform(image=image)
                processed_image = transformed['image']
                
                preprocessed_images.append(processed_image.numpy())
                valid_labels.append(label)
                
                # Save preprocessed image if requested
                if save_preprocessed:
                    save_path = output_path / f"processed_{i:06d}.npy"
                    np.save(save_path, processed_image.numpy())
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
        
        preprocessed_images = np.array(preprocessed_images)
        valid_labels = np.array(valid_labels)
        
        logger.info(f"Successfully preprocessed {len(preprocessed_images)} images")
        
        return preprocessed_images, valid_labels
    
    def save_preprocessed_data(self, 
                              output_dir: str = "preprocessed_data",
                              save_format: str = "numpy") -> Dict:
        """
        Save preprocessed data for all splits.
        
        Args:
            output_dir (str): Directory to save preprocessed data
            save_format (str): Format to save data ('numpy', 'pickle', 'h5')
            
        Returns:
            dict: Information about saved files
        """
        logger.info(f"Saving preprocessed data to {output_dir}...")
        
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
            
            # Preprocess images
            images, labels = self.preprocess_images(
                split_data['images'], 
                split_data['labels'],
                is_training=(split_name == 'train')
            )
            
            # Save data
            if save_format == "numpy":
                np.save(output_path / f"{split_name}_images.npy", images)
                np.save(output_path / f"{split_name}_labels.npy", labels)
                saved_files[f"{split_name}_images"] = str(output_path / f"{split_name}_images.npy")
                saved_files[f"{split_name}_labels"] = str(output_path / f"{split_name}_labels.npy")
            
            elif save_format == "pickle":
                with open(output_path / f"{split_name}_data.pkl", 'wb') as f:
                    pickle.dump({'images': images, 'labels': labels}, f)
                saved_files[f"{split_name}_data"] = str(output_path / f"{split_name}_data.pkl")
            
            elif save_format == "h5":
                import h5py
                with h5py.File(output_path / f"{split_name}_data.h5", 'w') as f:
                    f.create_dataset('images', data=images)
                    f.create_dataset('labels', data=labels)
                saved_files[f"{split_name}_data"] = str(output_path / f"{split_name}_data.h5")
        
        # Save metadata
        metadata = {
            'target_size': self.target_size,
            'dataset_stats': self._calculate_dataset_stats(),
            'preprocessing_info': {
                'normalization': 'ImageNet mean/std',
                'augmentation': 'Applied to training set only'
            }
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_files['metadata'] = str(output_path / "metadata.json")
        
        logger.info(f"Preprocessed data saved successfully to {output_dir}")
        logger.info(f"Saved files: {list(saved_files.keys())}")
        
        return saved_files
    
    def visualize_dataset(self, 
                         num_samples: int = 8,
                         figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Visualize sample images from the dataset.
        
        Args:
            num_samples (int): Number of samples to display
            figsize (tuple): Figure size
        """
        logger.info("Creating dataset visualization...")
        
        # Get random samples
        indices = np.random.choice(len(self.image_paths), 
                                 min(num_samples, len(self.image_paths)), 
                                 replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Load and display image
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(image)
            axes[i].set_title(f"Label: {self.labels[idx]}\n{Path(self.image_paths[idx]).name}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Plot class distribution
        plt.figure(figsize=(8, 6))
        class_counts = np.bincount(self.labels)
        plt.bar(['No CVD', 'CVD'], class_counts)
        plt.title('Class Distribution in Dataset')
        plt.ylabel('Number of Images')
        plt.show()
    
    def get_data_generator(self, 
                          split: str = 'train',
                          batch_size: Optional[int] = None) -> 'DataGenerator':
        """
        Get a data generator for the specified split.
        
        Args:
            split (str): Data split ('train', 'val', 'test')
            batch_size (int): Batch size (uses default if None)
            
        Returns:
            DataGenerator: Data generator object
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        split_data = getattr(self, f"{split}_data")
        if split_data is None:
            raise ValueError(f"No {split} data available. Run create_data_splits() first.")
        
        return DataGenerator(
            image_paths=split_data['images'],
            labels=split_data['labels'],
            target_size=self.target_size,
            batch_size=batch_size,
            is_training=(split == 'train'),
            shuffle=(split == 'train')
        )


class DataGenerator:
    """
    Data generator for batch processing of images during training.
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 target_size: Tuple[int, int],
                 batch_size: int = 32,
                 is_training: bool = True,
                 shuffle: bool = True):
        """
        Initialize the data generator.
        
        Args:
            image_paths (list): List of image paths
            labels (list): List of labels
            target_size (tuple): Target image size
            batch_size (int): Batch size
            is_training (bool): Whether this is for training
            shuffle (bool): Whether to shuffle data
        """
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.target_size = target_size
        self.batch_size = batch_size
        self.is_training = is_training
        self.shuffle = shuffle
        
        # Create preprocessing transforms
        self.transform = self._get_transforms()
        
        # Initialize indices
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_index = 0
    
    def _get_transforms(self):
        """Get preprocessing transforms."""
        transforms = []
        
        # Resize
        transforms.append(A.Resize(height=self.target_size[0], width=self.target_size[1]))
        
        # Augmentation for training
        if self.is_training:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Rotate(limit=15, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
            ])
        
        # Normalization
        transforms.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
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
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            try:
                # Load image
                image = cv2.imread(self.image_paths[idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                transformed = self.transform(image=image)
                processed_image = transformed['image']
                
                batch_images.append(processed_image)
                batch_labels.append(self.labels[idx])
                
            except Exception as e:
                logger.error(f"Error processing image {self.image_paths[idx]}: {str(e)}")
                continue
        
        return torch.stack(batch_images), torch.LongTensor(batch_labels)


def main():
    """
    Example usage of the FundusDatasetLoader.
    """
    # Initialize the loader
    loader = FundusDatasetLoader(
        data_dir="data/china_fundus_cimt",
        target_size=(224, 224),
        batch_size=32
    )
    
    # Load dataset
    try:
        dataset_stats = loader.load_dataset()
        print("Dataset loaded successfully!")
        print(f"Dataset statistics: {dataset_stats}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please ensure the dataset is in the correct directory structure:")
        print("data/china_fundus_cimt/")
        print("├── images/")
        print("├── labels.csv")
        print("└── metadata.json")
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
    for batch_images, batch_labels in train_gen:
        print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")
        break


if __name__ == "__main__":
    main()
