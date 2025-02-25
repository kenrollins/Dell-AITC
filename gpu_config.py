"""GPU Configuration for Dell-AITC

This module provides utilities for managing dual A6000 GPUs.
"""
import torch

def setup_gpus():
    """Configure system for optimal dual A6000 usage."""
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return False
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"Warning: Expected 2 GPUs, found {gpu_count}")
        return False
    
    # Print GPU information
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return True

def get_optimal_device():
    """Get the GPU with the most available memory."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    max_free_memory = 0
    optimal_device = 0
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        # Clear cache to get accurate memory reading
        torch.cuda.empty_cache()
        free_memory = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            optimal_device = i
    
    return torch.device(f'cuda:{optimal_device}')

def parallel_config():
    """Get configuration for DataParallel/DistributedDataParallel."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None
    
    return {
        'device_ids': list(range(torch.cuda.device_count())),
        'output_device': 0
    }
