"""
Streamlit Dashboard for Model Fusion System
==========================================

Interactive web interface for cardiovascular disease prediction using
fundus images and ECG signals with model fusion.

Usage:
    streamlit run streamlit_fusion_dashboard.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import json
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import cv2
from scipy import signal
from scipy.signal import resample

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Prediction - Fusion System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .low-risk {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #236B8E;
    }
</style>
""", unsafe_allow_html=True)

# Import fusion modules
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model_fusion_simple import SimpleModelFusion
    FUSION_AVAILABLE = True
    logger.info("Simple fusion system loaded successfully")
except ImportError as e:
    logger.error(f"Could not load fusion system: {e}")
    FUSION_AVAILABLE = False


class DashboardApp:
    """Main dashboard application class."""
    
    def __init__(self):
        """Initialize the dashboard."""
        # Initialize session state
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'current_patient' not in st.session_state:
            st.session_state.current_patient = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'evaluation_results' not in st.session_state:
            st.session_state.evaluation_results = None
        if 'fundus_data' not in st.session_state:
            st.session_state.fundus_data = None
        if 'ecg_data' not in st.session_state:
            st.session_state.ecg_data = None
        if 'fusion_model' not in st.session_state:
            st.session_state.fusion_model = None
    
    def load_models(self, fundus_model_path=None, ecg_model_path=None):
        """Load fusion models."""
        try:
            if FUSION_AVAILABLE:
                st.session_state.fusion_model = SimpleModelFusion()
                
                # Set default paths if not provided
                if fundus_model_path is None:
                    fundus_model_path = "checkpoints/best_model.pth"
                if ecg_model_path is None:
                    ecg_model_path = "ecg_checkpoints_fixed/best_ecg_model.pth"
                
                # Load models with the provided or default paths
                st.session_state.fusion_model.load_models(
                    fundus_model_path=fundus_model_path,
                    ecg_model_path=ecg_model_path
                )
                
                st.session_state.model_loaded = True
                return True
            else:
                st.error("Fusion system not available. Please check the model files.")
                return False
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def create_dummy_data(self, n_samples=1):
        """Create dummy data for testing."""
        np.random.seed(42)
        
        # Dummy fundus data (3-channel images)
        fundus_data = np.random.randn(n_samples, 3, 224, 224).astype(np.float32)
        
        # Dummy ECG data (1-channel signals, 1000 samples to match model expectations)
        ecg_data = np.random.randn(n_samples, 1, 1000).astype(np.float32)
        
        # Create patient info
        patient_info = {
            'id': f"patient_{np.random.randint(1000, 9999)}",
            'age': np.random.randint(30, 80),
            'gender': np.random.choice(['Male', 'Female']),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_source': 'dummy'
        }
        
        return fundus_data, ecg_data, patient_info
    
    def process_fundus_image(self, image_file):
        """Process uploaded fundus image into model-ready format."""
        try:
            # Read image file
            image_bytes = image_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image file")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image = cv2.resize(image, (224, 224))
            
            # Convert to float32 and normalize
            image = image.astype(np.float32) / 255.0
            
            # Convert to tensor format (C, H, W)
            image = np.transpose(image, (2, 0, 1))
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            logger.info(f"Processed fundus image: shape {image.shape}, range [{image.min():.3f}, {image.max():.3f}]")
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing fundus image: {e}")
            st.error(f"Error processing fundus image: {str(e)}")
            return None
    
    def process_ecg_file(self, ecg_file):
        """Process uploaded ECG file into model-ready format."""
        try:
            # Read file content
            content = ecg_file.read()
            
            if ecg_file.name.endswith('.csv'):
                # Process CSV file
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                
                # Try to find ECG signal columns
                signal_columns = [col for col in df.columns if any(keyword in col.lower() 
                                   for keyword in ['ecg', 'signal', 'voltage', 'amplitude'])]
                
                if signal_columns:
                    # Use the first signal column found
                    signal = df[signal_columns[0]].values
                else:
                    # Use the first numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        signal = df[numeric_cols[0]].values
                    else:
                        raise ValueError("No numeric data found in CSV file")
                        
            elif ecg_file.name.endswith('.json'):
                # Process JSON file
                data = json.loads(content.decode('utf-8'))
                
                # Try to extract signal from JSON structure
                if isinstance(data, dict):
                    # Look for signal in common keys
                    for key in ['ecg', 'signal', 'data', 'values']:
                        if key in data and isinstance(data[key], list):
                            signal = np.array(data[key])
                            break
                    else:
                        # Try to find any list of numbers
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) > 10:
                                signal = np.array(value)
                                break
                        else:
                            raise ValueError("No signal data found in JSON file")
                elif isinstance(data, list):
                    signal = np.array(data)
                else:
                    raise ValueError("Invalid JSON structure")
            else:
                raise ValueError("Unsupported file format. Please use CSV or JSON.")
            
            # Ensure signal is 1D
            if signal.ndim > 1:
                signal = signal.flatten()
            
            # Remove NaN and infinite values
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize signal
            signal_min, signal_max = signal.min(), signal.max()
            if signal_max > signal_min:
                signal = (signal - signal_min) / (signal_max - signal_min)
            else:
                signal = np.zeros_like(signal)
            
            # Resample to target length (1000 samples)
            if len(signal) != 1000:
                signal = resample(signal, 1000)
            
            # Add channel and batch dimensions
            signal = signal.reshape(1, 1, -1)
            
            logger.info(f"Processed ECG signal: shape {signal.shape}, range [{signal.min():.3f}, {signal.max():.3f}]")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error processing ECG file: {e}")
            st.error(f"Error processing ECG file: {str(e)}")
            return None
    
    def make_prediction(self, fundus_data, ecg_data, fundus_weight=0.5, ecg_weight=0.5):
        """Make prediction using fusion model."""
        # Initialize fusion model if not already loaded
        if st.session_state.fusion_model is None:
            logger.info("Initializing fusion model for prediction...")
            try:
                # Try to load models if not already loaded
                if not st.session_state.model_loaded:
                    self.load_models()
                # If still no fusion model, create one
                if st.session_state.fusion_model is None:
                    st.session_state.fusion_model = SimpleModelFusion()
                    logger.info("New fusion model created for prediction")
                logger.info("Fusion model ready for prediction")
            except Exception as e:
                logger.error(f"Failed to initialize fusion model: {e}")
                return None
        
        try:
            # Log input data details for debugging
            logger.info(f"Making prediction with fundus_data: {fundus_data.shape if fundus_data is not None else None}, ecg_data: {ecg_data.shape if ecg_data is not None else None}")
            
            results = st.session_state.fusion_model.predict(
                fundus_data, ecg_data,
                fundus_weight=fundus_weight,
                ecg_weight=ecg_weight
            )
            
            logger.info(f"Prediction successful, results: {type(results)}")
            return results
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            logger.error(f"Input data - fundus: {fundus_data.shape if fundus_data is not None else None}, ecg: {ecg_data.shape if ecg_data is not None else None}")
            return None
    
    def display_prediction_results(self, results):
        """Display prediction results in a user-friendly format."""
        if results is None:
            st.error("Could not generate predictions. Please check the input data.")
            return
        
        # Extract probabilities
        fundus_prob = results['fundus_probabilities'][0][1] if results['fundus_probabilities'] is not None else None
        ecg_prob = results['ecg_probabilities'][0][1] if results['ecg_probabilities'] is not None else None
        fused_prob = results['fused_probabilities'][0][1] if results['fused_probabilities'] is not None else None
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if fundus_prob is not None:
                st.metric(
                    label="Fundus Model",
                    value=f"{fundus_prob:.1%}",
                    delta=f"{fundus_prob - 0.5:.1%}" if fundus_prob > 0.5 else None,
                    delta_color="inverse"
                )
            else:
                st.metric(label="Fundus Model", value="N/A")
        
        with col2:
            if ecg_prob is not None:
                st.metric(
                    label="ECG Model",
                    value=f"{ecg_prob:.1%}",
                    delta=f"{ecg_prob - 0.5:.1%}" if ecg_prob > 0.5 else None,
                    delta_color="inverse"
                )
            else:
                st.metric(label="ECG Model", value="N/A")
        
        with col3:
            if fused_prob is not None:
                risk_level = "HIGH RISK" if fused_prob > 0.5 else "LOW RISK"
                risk_class = "high-risk" if fused_prob > 0.5 else "low-risk"
                
                st.metric(
                    label="Fused Model",
                    value=f"{fused_prob:.1%}",
                    delta=f"{fused_prob - 0.5:.1%}" if fused_prob > 0.5 else None,
                    delta_color="inverse"
                )
                
                # Display risk level
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    {risk_level}<br>
                    <small>Confidence: {abs(fused_prob - 0.5) * 200:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.metric(label="Fused Model", value="N/A")
        
        # Display weights used
        st.info(f"**Fusion Weights:** Fundus: {results['fundus_weight']:.1f}, ECG: {results['ecg_weight']:.1f}")
    
    def plot_probability_distribution(self, results):
        """Plot probability distribution visualization."""
        if results is None:
            return
        
        # Extract probabilities
        fundus_prob = results['fundus_probabilities'][0][1] if results['fundus_probabilities'] is not None else None
        ecg_prob = results['ecg_probabilities'][0][1] if results['ecg_probabilities'] is not None else None
        fused_prob = results['fused_probabilities'][0][1] if results['fused_probabilities'] is not None else None
        
        # Create plot
        fig = go.Figure()
        
        models = []
        probabilities = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        if fundus_prob is not None:
            models.append('Fundus')
            probabilities.append(fundus_prob)
        
        if ecg_prob is not None:
            models.append('ECG')
            probabilities.append(ecg_prob)
        
        if fused_prob is not None:
            models.append('Fused')
            probabilities.append(fused_prob)
        
        if models:
            fig.add_trace(go.Bar(
                x=models,
                y=probabilities,
                marker_color=colors[:len(models)],
                text=[f'{p:.1%}' for p in probabilities],
                textposition='auto',
                name='CVD Probability'
            ))
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         annotation_text="Decision Threshold")
            
            fig.update_layout(
                title='Model Probability Comparison',
                xaxis_title='Model',
                yaxis_title='CVD Probability',
                yaxis_range=[0, 1],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_confidence_heatmap(self, results):
        """Plot confidence heatmap."""
        if results is None:
            return
        
        # Extract probabilities
        fundus_prob = results['fundus_probabilities'][0][1] if results['fundus_probabilities'] is not None else None
        ecg_prob = results['ecg_probabilities'][0][1] if results['ecg_probabilities'] is not None else None
        fused_prob = results['fused_probabilities'][0][1] if results['fused_probabilities'] is not None else None
        
        if fundus_prob is None and ecg_prob is None:
            return
        
        # Create confidence matrix
        confidence_data = []
        
        if fundus_prob is not None:
            confidence_data.append(['Fundus', 'No CVD', 1 - fundus_prob])
            confidence_data.append(['Fundus', 'CVD', fundus_prob])
        
        if ecg_prob is not None:
            confidence_data.append(['ECG', 'No CVD', 1 - ecg_prob])
            confidence_data.append(['ECG', 'CVD', ecg_prob])
        
        if fused_prob is not None:
            confidence_data.append(['Fused', 'No CVD', 1 - fused_prob])
            confidence_data.append(['Fused', 'CVD', fused_prob])
        
        df_confidence = pd.DataFrame(confidence_data, columns=['Model', 'Class', 'Confidence'])
        
        # Create heatmap
        fig = px.imshow(
            df_confidence.pivot(index='Model', columns='Class', values='Confidence'),
            text_auto='.1%',
            aspect="auto",
            color_continuous_scale="RdYlBu_r"
        )
        
        fig.update_layout(
            title='Model Confidence Heatmap',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def load_evaluation_results(self):
        """Load the latest evaluation results from the fusion_evaluation_results directory."""
        try:
            results_dir = Path("fusion_evaluation_results")
            if not results_dir.exists():
                return None
            
            # Get all evaluation directories
            eval_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("evaluation_")]
            if not eval_dirs:
                return None
            
            # Get the latest evaluation directory
            latest_dir = max(eval_dirs, key=lambda x: x.name)
            
            # Load metrics.json
            metrics_file = latest_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return {
                    'metrics': metrics,
                    'directory': str(latest_dir),
                    'timestamp': latest_dir.name.split('_')[1] + '_' + latest_dir.name.split('_')[2]
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading evaluation results: {e}")
            return None
    
    def display_evaluation_results(self, eval_results):
        """Display evaluation results in the dashboard."""
        if eval_results is None:
            st.warning("No evaluation results found. Run the evaluation script first.")
            return
        
        st.markdown('<h2 class="sub-header">📈 Model Evaluation Results</h2>', unsafe_allow_html=True)
        
        # Display timestamp
        timestamp = eval_results['timestamp']
        st.info(f"**Results from:** {timestamp}")
        
        # Extract metrics
        metrics = eval_results['metrics']
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔍 Fundus Model")
            if 'fundus' in metrics:
                fundus_metrics = metrics['fundus']
                st.metric("Accuracy", f"{fundus_metrics['accuracy']:.3f}")
                st.metric("Precision", f"{fundus_metrics['precision']:.3f}")
                st.metric("Recall", f"{fundus_metrics['recall']:.3f}")
                st.metric("F1 Score", f"{fundus_metrics['f1_score']:.3f}")
                st.metric("AUC", f"{fundus_metrics['auc']:.3f}")
            else:
                st.write("No data available")
        
        with col2:
            st.markdown("### ⚡ ECG Model")
            if 'ecg' in metrics:
                ecg_metrics = metrics['ecg']
                st.metric("Accuracy", f"{ecg_metrics['accuracy']:.3f}")
                st.metric("Precision", f"{ecg_metrics['precision']:.3f}")
                st.metric("Recall", f"{ecg_metrics['recall']:.3f}")
                st.metric("F1 Score", f"{ecg_metrics['f1_score']:.3f}")
                st.metric("AUC", f"{ecg_metrics['auc']:.3f}")
            else:
                st.write("No data available")
        
        with col3:
            st.markdown("### 🎯 Fused Model")
            if 'fused' in metrics:
                fused_metrics = metrics['fused']
                st.metric("Accuracy", f"{fused_metrics['accuracy']:.3f}")
                st.metric("Precision", f"{fused_metrics['precision']:.3f}")
                st.metric("Recall", f"{fused_metrics['recall']:.3f}")
                st.metric("F1 Score", f"{fused_metrics['f1_score']:.3f}")
                st.metric("AUC", f"{fused_metrics['auc']:.3f}")
            else:
                st.write("No data available")
        
        # Model comparison chart
        if all(model in metrics for model in ['fundus', 'ecg', 'fused']):
            st.markdown("### 📊 Model Performance Comparison")
            
            # Prepare data for comparison chart
            comparison_data = []
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
                comparison_data.append({
                    'Metric': metric.upper().replace('_', ' '),
                    'Fundus': metrics['fundus'][metric],
                    'ECG': metrics['ecg'][metric],
                    'Fused': metrics['fused'][metric]
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Create comparison chart
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, model in enumerate(['Fundus', 'ECG', 'Fused']):
                fig.add_trace(go.Scatter(
                    x=df_comparison['Metric'],
                    y=df_comparison[model],
                    mode='lines+markers',
                    name=model,
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Metrics',
                yaxis_title='Score',
                yaxis_range=[0, 1.05],
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display results directory
        with st.expander("📁 Results Location"):
            st.write(f"Results saved in: `{eval_results['directory']}`")
            st.write("Files available:")
            results_dir = Path(eval_results['directory'])
            for file in results_dir.iterdir():
                if file.is_file():
                    st.write(f"- {file.name}")
    
    def run(self):
        """Run the dashboard application."""
        # Header
        st.markdown('<h1 class="main-header">🏥 Cardiovascular Disease Prediction</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666;">AI-Powered Fusion System</h3>', unsafe_allow_html=True)
        
        # Auto-load models on startup if not already loaded
        if not st.session_state.model_loaded and FUSION_AVAILABLE:
            with st.spinner("Loading pre-trained models..."):
                success = self.load_models()
                if success:
                    st.success("Models loaded successfully!")
                else:
                    st.warning("Could not load models. Using dummy predictions.")
        
        # Sidebar
        with st.sidebar:
            st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
            
            # Model loading
            st.markdown("### Model Configuration")
            
            fundus_model_path = st.text_input("Fundus Model Path (optional):", "")
            ecg_model_path = st.text_input("ECG Model Path (optional):", "")
            
            if st.button("Load Models"):
                with st.spinner("Loading models..."):
                    success = self.load_models(fundus_model_path, ecg_model_path)
                    if success:
                        st.success("Models loaded successfully!")
                    else:
                        st.error("Failed to load models. Using dummy predictions.")
            
            # Fusion weights
            st.markdown("### Fusion Weights")
            fundus_weight = st.slider("Fundus Model Weight", 0.0, 1.0, 0.5, 0.1)
            ecg_weight = st.slider("ECG Model Weight", 0.0, 1.0, 0.5, 0.1)
            
            # Data input options
            st.markdown("### Data Input")
            input_method = st.radio("Input Method:", ["Upload Files", "Use Dummy Data", "Manual Input"])
            
            # Add clear data button
            if st.button("🗑️ Clear All Data"):
                st.session_state.fundus_data = None
                st.session_state.ecg_data = None
                st.session_state.predictions = None
                st.session_state.current_patient = None
                st.success("All data cleared!")
                st.rerun()
            
            # Data status indicator
            st.markdown("### Data Status")
            fundus_status = "✅ Available" if st.session_state.fundus_data is not None else "❌ Not Available"
            ecg_status = "✅ Available" if st.session_state.ecg_data is not None else "❌ Not Available"
            st.write(f"Fundus Data: {fundus_status}")
            st.write(f"ECG Data: {ecg_status}")
            
            # Patient information
            st.markdown("### Patient Information")
            patient_id = st.text_input("Patient ID:", f"patient_{np.random.randint(1000, 9999)}")
            patient_age = st.number_input("Age:", min_value=18, max_value=100, value=45)
            patient_gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
            
            # Evaluation Results Section
            st.markdown("### 📈 Evaluation Results")
            if st.button("Load Latest Results"):
                with st.spinner("Loading evaluation results..."):
                    eval_results = self.load_evaluation_results()
                    if eval_results:
                        st.session_state.evaluation_results = eval_results
                        st.success("Results loaded successfully!")
                    else:
                        st.warning("No evaluation results found. Run the evaluation script first.")
        
        # Main content area
        st.markdown('<h2 class="sub-header">🔍 Prediction Interface</h2>', unsafe_allow_html=True)
        
        # Data input section
        fundus_data = st.session_state.fundus_data
        ecg_data = st.session_state.ecg_data
        patient_info = None
        
        if input_method == "Upload Files":
            col1, col2 = st.columns(2)
            
            with col1:
                fundus_file = st.file_uploader("Upload Fundus Image", type=['jpg', 'jpeg', 'png'])
                if fundus_file is not None:
                    try:
                        fundus_data = self.process_fundus_image(fundus_file)
                        st.session_state.fundus_data = fundus_data
                        st.success("Fundus image uploaded and processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing fundus image: {str(e)}")
                        logger.error(f"Fundus processing error: {e}")
            
            with col2:
                ecg_file = st.file_uploader("Upload ECG Signal", type=['csv', 'json'])
                if ecg_file is not None:
                    try:
                        ecg_data = self.process_ecg_file(ecg_file)
                        st.session_state.ecg_data = ecg_data
                        st.success("ECG signal uploaded and processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing ECG file: {str(e)}")
                        logger.error(f"ECG processing error: {e}")
        
        elif input_method == "Use Dummy Data":
            if st.button("Generate Dummy Data"):
                fundus_data, ecg_data, patient_info = self.create_dummy_data()
                st.session_state.fundus_data = fundus_data
                st.session_state.ecg_data = ecg_data
                st.success("Dummy data generated successfully!")
        
        else:  # Manual Input
            st.markdown("### Manual Data Input")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fundus_available = st.checkbox("Fundus Data Available")
                if fundus_available:
                    fundus_data, _, _ = self.create_dummy_data()
                    st.session_state.fundus_data = fundus_data
            
            with col2:
                ecg_available = st.checkbox("ECG Data Available")
                if ecg_available:
                    _, ecg_data, _ = self.create_dummy_data()
                    st.session_state.ecg_data = ecg_data
        
        # Prediction button
        if st.button("🎯 Generate Prediction", key="predict_button"):
            if fundus_data is None and ecg_data is None:
                st.warning("Please provide at least one type of data (Fundus or ECG)")
            else:
                with st.spinner("Analyzing data and generating predictions..."):
                    # Make prediction
                    results = self.make_prediction(fundus_data, ecg_data, fundus_weight, ecg_weight)
                    
                    if results is not None:
                        # Store in session state
                        st.session_state.predictions = results
                        st.session_state.current_patient = {
                            'id': patient_id,
                            'age': patient_age,
                            'gender': patient_gender
                        }
                        
                        # Display results
                        st.success("Prediction completed successfully!")
                    else:
                        st.error("Failed to generate predictions. Please check your input data.")
        
        # Results section
        if st.session_state.predictions is not None:
            st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
            
            # Patient information
            if st.session_state.current_patient:
                patient = st.session_state.current_patient
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Patient ID:</strong> {patient['id']}<br>
                    <strong>Age:</strong> {patient['age']} years<br>
                    <strong>Gender:</strong> {patient['gender']}
                </div>
                """, unsafe_allow_html=True)
            
            # Display prediction results
            self.display_prediction_results(st.session_state.predictions)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                self.plot_probability_distribution(st.session_state.predictions)
            
            with col2:
                self.plot_confidence_heatmap(st.session_state.predictions)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Generate Report"):
                    st.info("Report generation feature coming soon!")
            
            with col2:
                if st.button("💾 Save Results"):
                    st.info("Results saved to session (demo feature)")
            
            with col3:
                if st.button("🔄 New Prediction"):
                    # Clear current predictions and data
                    st.session_state.predictions = None
                    st.session_state.current_patient = None
                    st.session_state.fundus_data = None
                    st.session_state.ecg_data = None
                    st.rerun()
        
        # Evaluation Results Section
        if st.session_state.evaluation_results is not None:
            self.display_evaluation_results(st.session_state.evaluation_results)
        
        # Information section
        with st.expander("ℹ️ About This System"):
            st.markdown("""
            ### Cardiovascular Disease Prediction System
            
            This AI-powered system combines predictions from **fundus imaging** and **ECG analysis** 
            to provide comprehensive cardiovascular disease risk assessment.
            
            #### Key Features:
            - **Multi-modal Fusion**: Combines fundus and ECG predictions
            - **Adjustable Weights**: Customize the importance of each modality
            - **Interactive Visualizations**: Clear probability displays and confidence metrics
            - **Medical Interface**: Designed for healthcare professionals
            
            #### How to Use:
            1. Load the pre-trained models (optional)
            2. Upload patient data or use dummy data for testing
            3. Adjust fusion weights if needed
            4. Generate predictions
            5. Review results and visualizations
            
            #### Interpretation:
            - **Probability > 50%**: High risk of cardiovascular disease
            - **Probability < 50%**: Low risk of cardiovascular disease
            - **Confidence**: Higher values indicate more reliable predictions
            
            **Note**: This is a demonstration system. Always consult with healthcare 
            professionals for medical decisions.
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>🏥 Cardiovascular Disease Prediction System | Powered by AI Fusion Technology</p>
            <p><small>For demonstration purposes only. Not for clinical use.</small></p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the dashboard."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()