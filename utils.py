import os
import torch
import numpy as np
import streamlit as st

def validate_model_file(uploaded_file):
    """
    Validate uploaded model file
    """
    if uploaded_file is None:
        st.error("No file uploaded")
        return False
    
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension not in ['.pth', '.pt']:
        st.error(f"Unsupported file type: {file_extension}. Please upload .pth or .pt files.")
        return False
    
    return True

def load_pytorch_model(uploaded_file, algorithm):
    """
    Load PyTorch model with error handling
    """
    try:
        # Temporarily save uploaded file
        with open(os.path.join("uploaded_models", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load model based on algorithm
        model = torch.load(os.path.join("uploaded_models", uploaded_file.name))
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def analyze_model_architecture(model):
    """
    Analyze and display model architecture details
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        st.subheader("Model Architecture Analysis")
        st.write(f"Total Parameters: {total_params:,}")
        st.write(f"Trainable Parameters: {trainable_params:,}")
        st.write(f"Model Type: {type(model).__name__}")
        
        return True
    except Exception as e:
        st.error(f"Could not analyze model architecture: {e}")
        return False