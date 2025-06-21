import os

# Configuration settings for the model analysis application
MODEL_UPLOAD_DIR = "uploaded_models"
SUPPORTED_EXTENSIONS = ['.pth', '.pt']
SUPPORTED_ALGORITHMS = [
    'CNN', 
    'ResNet', 
    'VGG', 
    'InceptionNet', 
    'DenseNet', 
    'EfficientNet'
]

# Ensure upload directory exists
os.makedirs(MODEL_UPLOAD_DIR, exist_ok=True)