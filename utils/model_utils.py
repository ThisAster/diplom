import torch
import numpy as np
import cv2
import onnxruntime as ort
import os

def load_model(model_path):
    """Load model from either PyTorch or ONNX format."""
    if model_path.endswith('.onnx'):
        return load_onnx_model(model_path)
    else:
        return load_pytorch_model(model_path)

def load_pytorch_model(model_path):
    """Load PyTorch model from checkpoint."""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if it's a Lightning checkpoint
        if 'state_dict' in checkpoint:
            # Remove 'model.' prefix from keys
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint
            
        # Create model instance (you'll need to import your model architecture here)
        from models.denoiser import Denoiser  # Adjust import path as needed
        model = Denoiser()
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None

def load_onnx_model(model_path):
    """Load ONNX model."""
    try:
        # Create ONNX Runtime session
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None

def preprocess_image(image):
    """Preprocess image for model input."""
    # Convert to float32 and normalize
    image = image.astype(np.float32) / 255.0
    
    # Convert to NCHW format
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def postprocess_image(image):
    """Postprocess model output to image."""
    # Remove batch dimension
    image = np.squeeze(image, axis=0)
    
    # Convert to HWC format
    image = np.transpose(image, (1, 2, 0))
    
    # Denormalize and clip
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    return image

def denoise_frame(model, frame):
    """Denoise a single frame using either PyTorch or ONNX model."""
    # Preprocess
    input_tensor = preprocess_image(frame)
    
    if isinstance(model, ort.InferenceSession):
        # ONNX inference
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        denoised = model.run([output_name], {input_name: input_tensor})[0]
    else:
        # PyTorch inference
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_tensor)
            denoised = model(input_tensor).numpy()
    
    # Postprocess
    return postprocess_image(denoised) 