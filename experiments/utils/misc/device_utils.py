"""
Device utilities for cross-platform compatibility.
Provides helper functions for selecting appropriate compute devices.
"""

import torch


def get_optimal_device():
    """
    Returns the best available device for PyTorch operations.
    
    Priority order:
    1. CUDA (NVIDIA GPUs) - if available
    2. MPS (Apple Silicon) - if available  
    3. CPU - fallback
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_with_id(device_id=0):
    """
    Returns the best available device with optional device ID.
    
    Args:
        device_id (int): Device ID for CUDA devices (ignored for MPS/CPU)
        
    Returns:
        str: Device string with ID ('cuda:0', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return f"cuda:{device_id}"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_map_auto():
    """
    Returns appropriate device_map for HuggingFace models.
    
    Returns:
        str or None: 'auto' for CUDA systems, None for MPS/CPU systems
    """
    if torch.cuda.is_available():
        return "auto"
    else:
        return None


def move_model_to_device(model, device=None):
    """
    Moves a model to the optimal device.
    
    Args:
        model: PyTorch model to move
        device (str, optional): Specific device to use. If None, uses optimal device.
        
    Returns:
        model: Model moved to the specified device
    """
    if device is None:
        device = get_optimal_device()
    
    return model.to(device)


def print_device_info():
    """
    Prints information about available compute devices.
    """
    print("=== Device Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
    
    print(f"Optimal device: {get_optimal_device()}")
    print("========================")
