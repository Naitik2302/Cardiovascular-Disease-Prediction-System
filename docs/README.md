# Cardiovascular Disease Prediction System

An advanced AI-powered system for cardiovascular disease prediction using multi-modal data fusion of fundus imaging and ECG signal analysis.

## 🏥 Overview

This project implements a comprehensive machine learning pipeline that combines predictions from **fundus (retinal) imaging** and **ECG (electrocardiogram) signals** to provide accurate cardiovascular disease risk assessment. The system uses state-of-the-art deep learning models with an intelligent fusion mechanism.

## 🚀 Key Features

- **Multi-Modal Fusion**: Combines fundus and ECG predictions using weighted fusion
- **Interactive Dashboard**: Web-based interface for easy data upload and visualization
- **Multiple Model Architectures**: Support for ResNet, EfficientNet, CNN, LSTM, and Transformer models
- **Real-time Predictions**: Fast inference with pre-trained models
- **Comprehensive Evaluation**: Detailed metrics, ROC curves, and confusion matrices
- **Medical-Grade Interface**: Professional design suitable for healthcare environments

## 📁 Project Structure

```
cardiovascular-disease-prediction/
├── 📊 Core Components
│   ├── streamlit_fusion_dashboard.py    # Interactive web dashboard
│   ├── model_fusion_simple.py           # Simplified fusion system
│   ├── model_fusion.py                  # Advanced fusion system
│   └── evaluate_fusion.py               # Model evaluation script
│
├── 🧠 Model Architectures
│   ├── model_architectures.py           # Fundus CNN models (ResNet, EfficientNet)
│   ├── ecg_model_architectures.py     # ECG models (1D-CNN, LSTM, BiLSTM)
│   └── model_utils.py                   # Model utilities and evaluators
│
├── 📊 Data Processing
│   ├── fundus_data_loader.py            # Fundus image data loader
│   ├── ecg_data_loader.py              # ECG signal data loader
│   ├── prepare_fundus_dataset.py       # Fundus dataset preparation
│   ├── prepare_ecg_dataset.py          # ECG dataset preparation
│   └── create_fundus_labels.py         # Label generation utility
│
├── 🏋️ Training Scripts
│   ├── train_fundus_cnn.py             # Fundus model training
│   ├── train_ecg_models.py              # ECG model training
│   ├── train_all_models.py              # Comprehensive training orchestrator
│   └── config.py                        # Configuration parameters
│
├── 💾 Model Checkpoints
│   ├── checkpoints/                     # Fundus model weights
│   ├── ecg_checkpoints/                 # ECG model weights
│   └── ecg_checkpoints_fixed/           # Corrected ECG weights
│
├── 📈 Results & Evaluation
│   ├── fusion_evaluation_results/       # Evaluation outputs
│   ├── training_results.json            # Training summary
│   └── runs/                           # TensorBoard logs
│
└── 🛠️ Setup & Utilities
    ├── requirements.txt                  # Python dependencies
    ├── install_libraries.py              # Library installation
    ├── validate_datasets.py              # Data validation
    └── setup_environment.bat            # Environment setup
```

## 🎯 System Capabilities

### Fundus Imaging Analysis
- **Retinal Vessel Analysis**: Extracts cardiovascular risk indicators from retinal blood vessels
- **Image Preprocessing**: Automatic resizing, normalization, and augmentation
- **Model Support**: ResNet18, ResNet50, EfficientNet-B0, Custom CNN architectures

### ECG Signal Processing
- **Signal Analysis**: Advanced filtering and feature extraction from ECG signals
- **Temporal Modeling**: LSTM, BiLSTM, and 1D-CNN architectures for time-series analysis
- **Real-time Processing**: Efficient signal processing for clinical workflows

### Fusion System
- **Weighted Fusion**: Adjustable importance weighting for fundus vs ECG predictions
- **Confidence Scoring**: Reliability metrics for each prediction
- **Multi-model Integration**: Seamless combination of different model outputs

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or use the automated installer
python install_libraries.py
```

### 2. Data Preparation
```bash
# Prepare fundus dataset
python prepare_fundus_dataset.py --data_dir "Fundus_CIMT_2903/Fundus_CIMT_2903 Dataset"

# Prepare ECG dataset  
python prepare_ecg_dataset.py --csv_file ECGCvdata.csv
```

### 3. Model Training
```bash
# Train individual models
python train_fundus_cnn.py --model_type efficientnet_b0 --epochs 10
python train_ecg_models.py --model_type bilstm --epochs 15

# Or train all models comprehensively
python train_all_models.py
```

### 4. Launch Dashboard
```bash
# Start the interactive web interface
streamlit run streamlit_fusion_dashboard.py
```

### 5. Model Evaluation
```bash
# Evaluate fusion system performance
python evaluate_fusion.py --fundus_model checkpoints/best_model.pth --ecg_model ecg_checkpoints_fixed/best_ecg_model.pth
```

## 📊 Usage Examples

### Dashboard Interface
1. **Upload Data**: Drag-and-drop fundus images and ECG files
2. **Adjust Weights**: Use sliders to set fundus vs ECG importance
3. **Generate Predictions**: Click to get cardiovascular risk assessment
4. **View Results**: Interactive visualizations with confidence scores
5. **Export Reports**: Save results for clinical documentation

### Programmatic Usage
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
print(f"Risk Probability: {results['fused_probability']:.2%}")
print(f"Confidence: {results['confidence']:.2%}")
```

## 🏥 Medical Interpretation

### Risk Categories
- **High Risk (>70%)**: Immediate clinical attention recommended
- **Moderate Risk (30-70%)**: Further evaluation suggested
- **Low Risk (<30%)**: Routine monitoring adequate

### Confidence Levels
- **High Confidence (>80%)**: Reliable prediction, high clinical value
- **Moderate Confidence (50-80%)**: Consider additional tests
- **Low Confidence (<50%)**: Prediction uncertain, recommend specialist review

## 📈 Performance Metrics

The system achieves competitive performance on cardiovascular disease prediction:

- **Fundus Models**: AUC-ROC ~0.85-0.90
- **ECG Models**: AUC-ROC ~0.88-0.92  
- **Fusion System**: AUC-ROC ~0.92-0.95

*Note: Performance may vary based on dataset quality and model configuration*

## 🔧 Configuration

Key configuration parameters are managed through `config.py`:

- **Model Settings**: Architecture selection, hyperparameters
- **Training Parameters**: Learning rates, batch sizes, epochs
- **Data Processing**: Image sizes, signal lengths, preprocessing options
- **Fusion Weights**: Default importance weighting between modalities

## 📋 Requirements

### Core Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
streamlit>=1.20.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
plotly>=5.0.0
Pillow>=8.0.0
opencv-python>=4.5.0
scipy>=1.7.0
albumentations>=1.1.0
timm>=0.6.0
```

### Optional Dependencies
```
tensorboard>=2.8.0
tqdm>=4.62.0
jupyter>=1.0.0
```

## 🏥 Clinical Workflow Integration

### Recommended Usage
1. **Data Collection**: Obtain fundus images and ECG recordings
2. **Quality Check**: Validate data quality using built-in validation tools
3. **Prediction Generation**: Use dashboard or API for risk assessment
4. **Clinical Review**: Interpret results with medical expertise
5. **Patient Communication**: Explain risk levels and recommendations

### Safety Considerations
- This system is for **decision support only**, not diagnostic replacement
- Always combine with clinical judgment and additional tests
- Consider patient-specific factors beyond the model inputs
- Regular model validation on local data is recommended

## 🔍 Troubleshooting

### Common Issues

**Model Loading Errors**
- Verify checkpoint files exist in correct directories
- Check PyTorch version compatibility
- Ensure sufficient GPU memory if using CUDA

**Data Processing Issues**
- Validate input data formats (images: JPG/PNG, ECG: CSV/JSON)
- Check data dimensions match model expectations
- Verify file paths are correct

**Dashboard Problems**
- Ensure Streamlit is properly installed
- Check port availability (default: 8501)
- Verify all dependencies are installed

**Performance Issues**
- Consider using GPU acceleration for faster inference
- Reduce batch sizes if memory issues occur
- Use pre-trained models for faster startup

### Getting Help
- Check the evaluation results in `fusion_evaluation_results/` directory
- Review training logs in `runs/` for TensorBoard visualization
- Validate datasets using `validate_datasets.py`

## 🚀 Advanced Features

### Custom Model Integration
- Add new architectures by extending base classes
- Implement custom fusion strategies
- Integrate with existing clinical systems

### Batch Processing
- Process multiple patients simultaneously
- Generate comprehensive reports
- Export results in various formats

### Model Comparison
- Compare different model architectures
- Evaluate fusion vs individual model performance
- Select optimal models for specific use cases

## 📊 Evaluation & Validation

The system includes comprehensive evaluation tools:

- **ROC Curves**: Receiver Operating Characteristic analysis
- **Confusion Matrices**: Detailed classification performance
- **Calibration Plots**: Prediction reliability assessment
- **Feature Importance**: Understanding model decision-making
- **Cross-Validation**: Robust performance estimation

## 🔒 Security & Privacy

- All processing is done locally (no external API calls)
- Patient data remains on your system
- No data transmission to external servers
- Compliant with healthcare data protection requirements

## 📈 Future Enhancements

- **Additional Modalities**: Integration with other biomarkers
- **Real-time Processing**: Live ECG analysis capabilities
- **Mobile Deployment**: Mobile-optimized interface
- **API Development**: RESTful API for system integration
- **Advanced Visualizations**: Enhanced medical imaging displays

## 📞 Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review the evaluation results and logs
3. Validate your datasets using provided tools
4. Ensure all dependencies are properly installed

---

**⚠️ Important Medical Disclaimer**

This system is designed for **research and educational purposes** and should be used as a **decision support tool** only. It is **not intended to replace professional medical judgment, diagnosis, or treatment**. Always consult qualified healthcare professionals for medical decisions. The accuracy of predictions depends on data quality and may not be suitable for all patient populations or clinical scenarios.

**🏥 Clinical Use Guidelines**
- Use as supplementary information only
- Combine with clinical assessment and additional tests
- Consider patient-specific factors beyond model inputs
- Regular validation on local data is recommended
- Follow institutional protocols for AI-assisted healthcare tools