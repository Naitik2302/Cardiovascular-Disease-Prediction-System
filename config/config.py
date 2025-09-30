"""
Configuration file for Cardiovascular Disease Prediction Project
==============================================================

This file contains all configuration parameters for the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "fundus": {
        "name": "China-Fundus-CIMT",
        "data_dir": DATA_DIR / "china_fundus_cimt",
        "image_folder": "images",
        "label_file": "labels.csv",
        "metadata_file": "metadata.json",
        "target_size": (224, 224),
        "channels": 3,
        "classes": ["No CVD", "CVD"],
        "class_weights": None  # Will be calculated based on class distribution
    },
    "ecg": {
        "name": "Cardiac Ailments",
        "data_dir": DATA_DIR / "ecg_dataset",
        "signal_length": 1000,  # Number of samples per ECG signal
        "sampling_rate": 125,   # Hz
        "channels": 1,
        "classes": ["No CVD", "CVD"]
    }
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    "fundus": {
        "resize_method": "bilinear",
        "normalization": "imagenet",  # or "custom"
        "augmentation": {
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation": True,
            "brightness_contrast": True,
            "hue_saturation": True,
            "gaussian_noise": True,
            "gaussian_blur": True,
            "elastic_transform": True,
            "grid_distortion": True,
            "optical_distortion": True
        },
        "augmentation_probability": 0.5
    },
    "ecg": {
        "filtering": {
            "lowpass_freq": 40,  # Hz
            "highpass_freq": 0.5,  # Hz
            "notch_freq": 50  # Hz (for power line noise)
        },
        "normalization": "z_score",  # or "min_max"
        "segmentation": {
            "window_size": 1000,
            "overlap": 0.5
        },
        "augmentation": {
            "time_warping": True,
            "amplitude_scaling": True,
            "gaussian_noise": True,
            "baseline_drift": True
        }
    }
}

# Model configuration
MODEL_CONFIG = {
    "fundus_cnn": {
        "architecture": "resnet50",  # or "efficientnet", "vgg16", "custom"
        "pretrained": True,
        "freeze_backbone": False,
        "dropout_rate": 0.5,
        "num_classes": 2,
        "input_shape": (224, 224, 3)
    },
    "ecg_cnn": {
        "architecture": "1d_cnn",  # or "lstm", "transformer"
        "filters": [64, 128, 256, 512],
        "kernel_sizes": [3, 3, 3, 3],
        "pool_sizes": [2, 2, 2, 2],
        "dropout_rate": 0.5,
        "num_classes": 2,
        "input_shape": (1000, 1)
    },
    "fusion": {
        "method": "weighted_average",  # or "concatenation", "attention"
        "weights": [0.6, 0.4],  # [fundus_weight, ecg_weight]
        "temperature": 2.0  # For temperature scaling
    }
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "patience": 10,  # Early stopping patience
    "min_delta": 0.001,  # Minimum change for early stopping
    "validation_split": 0.2,
    "test_split": 0.1,
    "random_seed": 42,
    "use_class_weights": True,
    "use_data_augmentation": True,
    "save_best_model": True,
    "save_checkpoints": True,
    "checkpoint_frequency": 5  # Save checkpoint every N epochs
}

# Optimizer configuration
OPTIMIZER_CONFIG = {
    "name": "adam",  # or "sgd", "adamw", "rmsprop"
    "lr": 0.001,
    "weight_decay": 1e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "momentum": 0.9,  # For SGD
    "nesterov": True  # For SGD
}

# Learning rate scheduler configuration
LR_SCHEDULER_CONFIG = {
    "name": "cosine_annealing",  # or "step", "exponential", "plateau"
    "T_max": 100,  # For cosine annealing
    "eta_min": 1e-6,  # For cosine annealing
    "step_size": 30,  # For step scheduler
    "gamma": 0.1,  # For step/exponential scheduler
    "patience": 5,  # For plateau scheduler
    "factor": 0.5,  # For plateau scheduler
    "min_lr": 1e-6  # For plateau scheduler
}

# Loss function configuration
LOSS_CONFIG = {
    "name": "binary_crossentropy",  # or "focal_loss", "weighted_bce"
    "from_logits": False,
    "label_smoothing": 0.0,
    "focal_alpha": 0.25,  # For focal loss
    "focal_gamma": 2.0,   # For focal loss
    "class_weights": None  # Will be calculated
}

# Metrics configuration
METRICS_CONFIG = {
    "primary_metric": "val_auc",
    "monitor_metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc",
        "specificity",
        "sensitivity"
    ]
}

# Data loading configuration
DATA_LOADING_CONFIG = {
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "shuffle_train": True,
    "shuffle_val": False,
    "shuffle_test": False
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "threshold": 0.5,
    "metrics": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc",
        "confusion_matrix",
        "classification_report"
    ],
    "save_predictions": True,
    "save_probabilities": True,
    "create_plots": True
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    "page_title": "Cardiovascular Disease Prediction",
    "page_icon": "❤️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "upload_max_size": 200,  # MB
    "allowed_file_types": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".csv", ".txt"]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOG_DIR / "training.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Hardware configuration
HARDWARE_CONFIG = {
    "use_gpu": True,
    "gpu_memory_growth": True,
    "mixed_precision": True,
    "num_gpus": 1,
    "cpu_threads": None  # None for auto-detection
}

# File paths
FILE_PATHS = {
    "fundus_data_loader": PROJECT_ROOT / "fundus_data_loader.py",
    "ecg_data_loader": PROJECT_ROOT / "ecg_data_loader.py",
    "model_architectures": PROJECT_ROOT / "model_architectures.py",
    "training_utils": PROJECT_ROOT / "training_utils.py",
    "evaluation_utils": PROJECT_ROOT / "evaluation_utils.py",
    "streamlit_app": PROJECT_ROOT / "streamlit_app.py",
    "requirements": PROJECT_ROOT / "requirements.txt",
    "readme": PROJECT_ROOT / "README.md"
}

# Default values
DEFAULT_VALUES = {
    "image_size": (224, 224),
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "random_seed": 42,
    "validation_split": 0.2,
    "test_split": 0.1
}

# Environment variables
ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": "0",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "PYTHONPATH": str(PROJECT_ROOT)
}

# Set environment variables
for key, value in ENV_VARS.items():
    os.environ[key] = value

# Validation functions
def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Validate paths
    for name, path in FILE_PATHS.items():
        if not path.exists():
            errors.append(f"File not found: {name} at {path}")
    
    # Validate dataset configuration
    if DATASET_CONFIG["fundus"]["target_size"][0] <= 0 or DATASET_CONFIG["fundus"]["target_size"][1] <= 0:
        errors.append("Invalid target size for fundus images")
    
    # Validate training configuration
    if TRAINING_CONFIG["batch_size"] <= 0:
        errors.append("Invalid batch size")
    
    if TRAINING_CONFIG["learning_rate"] <= 0:
        errors.append("Invalid learning rate")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Initialize configuration
if __name__ == "__main__":
    try:
        validate_config()
        print("Configuration validated successfully!")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        exit(1)
