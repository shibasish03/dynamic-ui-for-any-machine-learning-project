import streamlit as st
import torch
from config import SUPPORTED_ALGORITHMS
from utils import validate_model_file, load_pytorch_model, analyze_model_architecture
from model_processor import ModelProcessor

def main():
    st.title("🤖 PyTorch Model Analysis Tool")
    
    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload PyTorch Model (.pth/.pt)", 
        type=['pth', 'pt']
    )
    
    # Algorithm selection
    selected_algorithm = st.sidebar.selectbox(
        "Select Model Algorithm", 
        SUPPORTED_ALGORITHMS
    )
    
    # Dataset labels input
    dataset_labels = st.sidebar.text_input(
        "Enter Dataset Labels (comma-separated)", 
        placeholder="e.g., CIFAR10, ImageNet"
    ).split(',') if st.sidebar.checkbox("Add Dataset Labels") else []
    
    # Process model when file is uploaded
    if uploaded_file is not None:
        if validate_model_file(uploaded_file):
            # Load model
            model = load_pytorch_model(uploaded_file, selected_algorithm)
            
            if model is not None:
                # Analyze model architecture
                if analyze_model_architecture(model):
                    # Process model based on selected algorithm
                    ModelProcessor.process_model(
                        model, 
                        selected_algorithm, 
                        [label.strip() for label in dataset_labels]
                    )

if __name__ == "__main__":
    main()