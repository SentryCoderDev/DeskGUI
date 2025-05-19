# Vision modülleri için paket tanımlayıcı
import os

# Vision modülleri için paket tanımlayıcı

# Define the common models directory
MODELS_DIR = r"C:\test\models"  # Update this path as needed

# Check if directory exists
if not os.path.exists(MODELS_DIR):
    print(f"WARNING: Models directory not found: {MODELS_DIR}")
    print("Vision modules may not function correctly.")

# Utility function to get model path
def get_model_path(model_name):
    """Get the full path for a model file"""
    return os.path.join(MODELS_DIR, model_name)
# 