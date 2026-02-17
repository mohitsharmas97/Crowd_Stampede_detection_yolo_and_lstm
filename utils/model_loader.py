"""
Model loading utilities with caching for YOLOv8 and LSTM models
"""

import streamlit as st
import torch
from ultralytics import YOLO
import os
import config


@st.cache_resource
def load_yolo_model(model_path=None):
    """
    Load YOLOv8 model with Streamlit caching
    
    Args:
        model_path: Path to YOLOv8 model file
        
    Returns:
        YOLO model instance
    """
    if model_path is None:
        model_path = config.YOLO_MODEL_PATH
    
    if not os.path.exists(model_path):
        st.error(f"YOLOv8 model not found at {model_path}")
        return None
    
    try:
        model = YOLO(model_path)
        st.success(f"✅ YOLOv8 model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv8 model: {str(e)}")
        return None


@st.cache_resource
def load_lstm_model(model_path=None):
    """
    Load LSTM model with Streamlit caching
    
    Args:
        model_path: Path to LSTM model file
        
    Returns:
        Tuple of (LSTM model instance, normalization params dict)
    """
    if model_path is None:
        model_path = config.LSTM_MODEL_PATH
    
    if not os.path.exists(model_path):
        st.error(f"LSTM model not found at {model_path}")
        return None
    
    try:
        from .lstm_model import StampedeRiskLSTM
        
        # Load PyTorch model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use weights_only=False for PyTorch 2.6+ compatibility with older saved models
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Initialize model
        model = StampedeRiskLSTM(input_size=6, hidden_size=64, num_layers=2)
        
        # Check the structure of checkpoint
        if isinstance(checkpoint, dict):
            # Check if it contains 'model_state' key (saved with normalization params)
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                # Store normalization parameters if available
                if hasattr(model, 'register_buffer'):
                    if 'X_mean' in checkpoint:
                        model.register_buffer('X_mean', torch.FloatTensor(checkpoint['X_mean']))
                    if 'X_std' in checkpoint:
                        model.register_buffer('X_std', torch.FloatTensor(checkpoint['X_std']))
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
        else:
            # It's a full model object
            model = checkpoint
        
        model = model.to(device)
        
        # Set to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        
        st.success(f"✅ LSTM model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading LSTM model: {str(e)}")
        st.error("If model architecture doesn't match, check lstm_model.py parameters")
        return None


def predict_risk_score(lstm_model, features):
    """
    Predict stampede risk score using LSTM model
    
    Args:
        lstm_model: Loaded LSTM model
        features: Extracted temporal features
        
    Returns:
        Risk score (0-100)
    """
    try:
        with torch.no_grad():
            # Convert features to tensor
            if not isinstance(features, torch.Tensor):
                features = torch.FloatTensor(features)
            
            # Add batch dimension if needed
            if features.dim() == 2:
                features = features.unsqueeze(0)
            
            # Get prediction
            # Check if model is a dict (state_dict) or actual model
            if isinstance(lstm_model, dict):
                # Model was saved as state_dict only, cannot predict
                st.warning("LSTM model is a state dict. Need model architecture to predict.")
                return 0.0
            
            # Get device - handle both model objects and potential issues
            try:
                device = next(lstm_model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device('cpu')
            
            features = features.to(device)
            output = lstm_model(features)
            
            # Convert to risk score (0-100)
            risk_score = float(output.cpu().numpy()[0][0]) * 100
            risk_score = max(0, min(100, risk_score))  # Clip to 0-100
            
            return risk_score
    except Exception as e:
        st.warning(f"Error predicting risk score: {str(e)}")
        return 0.0
