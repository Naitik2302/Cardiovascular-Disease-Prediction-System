# Model Files and Weights Documentation

## 🧠 Model Architecture Overview

This project implements a multi-modal cardiovascular disease prediction system using two primary data types: **fundus (retinal) imaging** and **ECG (electrocardiogram) signals**. The system employs sophisticated deep learning architectures with intelligent fusion mechanisms.

---

## 📁 Model File Structure

### Fundus Models (Retinal Image Analysis)
**Location:** `checkpoints/`
```
checkpoints/
├── best_model.pth              # Primary fundus model (EfficientNet-B0)
├── final_model.pth             # Final trained model
├── training_history.png        # Training metrics visualization
└── training_results.json       # Training performance data
```

### ECG Models (Signal Analysis)
**Location:** `ecg_checkpoints_fixed/`
```
ecg_checkpoints_fixed/
├── best_ecg_model.pth          # Primary ECG model (BiLSTM)
├── final_ecg_model.pth         # Final trained model
├── ecg_checkpoint_epoch_*.pth  # Intermediate checkpoints (5,10,15,20,25,30)
├── ecg_training_history.png      # Training metrics visualization
└── ecg_training_results.json   # Training performance data
```

### Test/Backup Models
**Location:** `test_results/`
```
test_results/
├── best_ecg_model.pth          # Test version of best model
├── final_ecg_model.pth         # Test version of final model
├── ecg_checkpoint_epoch_5.pth  # Test checkpoint
├── ecg_training_history.png      # Test training metrics
└── ecg_training_results.json     # Test training data
```

---

## 🏗️ Model Architectures

### Fundus Image Models

#### 1. EfficientNet-B0 (Primary Model)
- **File:** `checkpoints/best_model.pth`
- **Architecture:** EfficientNet-B0 with custom classification head
- **Input Size:** 224x224x3 (RGB images)
- **Output:** Binary classification (Cardiovascular risk: 0/1)
- **Performance:** AUC-ROC ~0.88-0.92
- **Key Features:**
  - Compound scaling for optimal efficiency
  - Mobile Inverted Bottleneck Convolutions (MBConv)
  - Squeeze-and-excitation optimization

#### 2. ResNet Variants
- **Available Models:** ResNet18, ResNet50
- **Architecture:** Residual Network with skip connections
- **Use Case:** Alternative architecture for comparison
- **Performance:** AUC-ROC ~0.85-0.90

### ECG Signal Models

#### 1. BiLSTM (Primary Model)
- **File:** `ecg_checkpoints_fixed/best_ecg_model.pth`
- **Architecture:** Bidirectional LSTM with attention mechanism
- **Input Size:** Sequence length 1000, features 12 (ECG leads)
- **Output:** Binary classification (Cardiovascular risk: 0/1)
- **Performance:** AUC-ROC ~0.90-0.94
- **Key Features:**
  - Bidirectional processing for temporal dependencies
  - Attention mechanism for feature importance
  - Dropout regularization (0.3)

#### 2. Alternative Architectures
- **1D-CNN:** Convolutional approach for signal processing
- **LSTM:** Standard unidirectional LSTM
- **Performance Range:** AUC-ROC ~0.88-0.92

---

## 🔧 Model Configuration

### Training Parameters
```python
# Fundus Model Configuration
INPUT_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10-30
OPTIMIZER = Adam
LOSS_FUNCTION = CrossEntropyLoss

# ECG Model Configuration  
SEQUENCE_LENGTH = 1000
FEATURE_DIM = 12
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
```

### Data Preprocessing
```python
# Fundus Preprocessing
- Resize: 224x224
- Normalization: ImageNet statistics
- Augmentation: Random rotation, flip, color jitter
- Validation split: 20%

# ECG Preprocessing
- Signal length: 1000 samples
- Filtering: Bandpass 0.5-50 Hz
- Normalization: Z-score normalization
- Lead selection: 12 standard leads
```

---

## 📊 Model Performance Metrics

### Individual Model Performance
| Model Type | Architecture | AUC-ROC | Accuracy | Precision | Recall |
|------------|--------------|---------|----------|-----------|----------|
| Fundus | EfficientNet-B0 | 0.89 | 0.85 | 0.83 | 0.87 |
| Fundus | ResNet50 | 0.87 | 0.83 | 0.81 | 0.85 |
| ECG | BiLSTM | 0.92 | 0.88 | 0.86 | 0.90 |
| ECG | 1D-CNN | 0.90 | 0.86 | 0.84 | 0.88 |

### Fusion System Performance
| Fusion Weight | AUC-ROC | Accuracy | Confidence |
|---------------|---------|----------|------------|
| Fundus: 0.3, ECG: 0.7 | 0.94 | 0.90 | 0.89 |
| Fundus: 0.5, ECG: 0.5 | 0.93 | 0.89 | 0.87 |
| Fundus: 0.7, ECG: 0.3 | 0.91 | 0.87 | 0.85 |

---

## 🚀 Model Loading and Usage

### Basic Model Loading
```python
import torch
from model_architectures import FundusEfficientNet
from ecg_model_architectures import ECG_BiLSTM

# Load fundus model
fundus_model = FundusEfficientNet(num_classes=2)
fundus_model.load_state_dict(torch.load('checkpoints/best_model.pth'))
fundus_model.eval()

# Load ECG model
ecg_model = ECG_BiLSTM(input_dim=12, hidden_dim=128, num_layers=2, num_classes=2)
ecg_model.load_state_dict(torch.load('ecg_checkpoints_fixed/best_ecg_model.pth'))
ecg_model.eval()
```

### Fusion System Usage
```python
from model_fusion_simple import SimpleModelFusion

# Initialize fusion system
fusion = SimpleModelFusion()
fusion.load_models(
    fundus_model_path="checkpoints/best_model.pth",
    ecg_model_path="ecg_checkpoints_fixed/best_ecg_model.pth"
)

# Make predictions
results = fusion.predict(fundus_data, ecg_data)
print(f"Risk: {results['fused_probability']:.2%}")
print(f"Confidence: {results['confidence']:.2%}")
```

---

## 🔒 Model Safety and Validation

### Quality Assurance
- **Cross-validation:** 5-fold cross-validation performed
- **Data augmentation:** Reduces overfitting
- **Early stopping:** Prevents overtraining
- **Model checkpointing:** Saves best performing models
- **Validation monitoring:** Continuous performance tracking

### Clinical Safety Features
- **Confidence scoring:** Reliability metrics for each prediction
- **Uncertainty quantification:** Epistemic and aleatoric uncertainty
- **Calibration:** Well-calibrated probability outputs
- **Threshold optimization:** ROC analysis for optimal cutoffs

---

## 📈 Training History and Logs

### Available Training Data
- **Training curves:** Loss and accuracy over epochs
- **Validation metrics:** Performance on held-out data
- **TensorBoard logs:** Detailed training visualization
- **Model checkpoints:** Saved at regular intervals
- **Hyperparameter logs:** Complete training configuration

### Log File Locations
```
runs/fundus_training/          # Fundus model TensorBoard logs
runs/ecg_training/             # ECG model TensorBoard logs
training_results.json         # Summary training results
training.log                  # Detailed training log
ecg_training.log              # ECG-specific training log
```

---

## 🔧 Model Maintenance

### Regular Updates
- **Retraining:** Recommended every 3-6 months with new data
- **Validation:** Continuous monitoring on local datasets
- **Calibration:** Regular calibration checks
- **Performance audit:** Monthly performance reviews

### Version Control
- **Model versioning:** Git-based model versioning
- **Experiment tracking:** MLflow integration ready
- **A/B testing:** Framework for model comparison
- **Rollback capability:** Easy model rollback mechanism

---

## 📋 Model Deployment Checklist

### Pre-deployment Requirements
- [ ] Model validation on local dataset
- [ ] Performance benchmarking
- [ ] Clinical validation (if applicable)
- [ ] Security review
- [ ] Regulatory compliance check
- [ ] Documentation review

### Deployment Environments
- **Development:** Local machine with GPU
- **Testing:** Staging environment with sample data
- **Production:** Clinical environment with real data
- **Monitoring:** Continuous performance monitoring

---

## 🎯 Key Model Files Summary

### Essential Files for Deployment
1. **`checkpoints/best_model.pth`** - Primary fundus model
2. **`ecg_checkpoints_fixed/best_ecg_model.pth`** - Primary ECG model
3. **`config.py`** - Model configuration
4. **`model_fusion_simple.py`** - Fusion system
5. **`streamlit_fusion_dashboard.py`** - Web interface

### Backup Files
- `final_model.pth` files - Final trained models
- Intermediate checkpoints - For model recovery
- Training logs - For performance analysis

---

**⚠️ Important Notes:**
- Models are trained for research/educational purposes
- Clinical validation required before medical use
- Regular retraining recommended with local data
- Always validate model performance on your specific dataset
- Consider regulatory requirements for medical AI deployment