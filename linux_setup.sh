#!/bin/bash

# Install core dependencies (excluding Neo4j-specific ones)
echo "Installing core dependencies..."
pip install fastapi==0.115.8 \
    uvicorn==0.34.0 \
    python-dotenv==1.0.1 \
    pydantic==2.10.6 \
    pydantic-settings==2.7.1

# Install AI and data science packages with GPU support
echo "Installing AI and data science packages..."
pip install sentence-transformers==3.4.1 \
    numpy==2.2.2 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    transformers==4.48.2 \
    ollama==0.4.7 \
    openai==1.61.0

# Install PyTorch with CUDA 12.2 support
echo "Installing PyTorch with CUDA 12.2 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# Install testing and utility packages
echo "Installing testing and utility packages..."
pip install pytest==8.3.4 \
    pytest-asyncio==0.25.3 \
    python-multipart==0.0.20 \
    python-jose[cryptography] \
    passlib[bcrypt] \
    httpx==0.28.1

# Install visualization packages (if needed)
echo "Installing visualization packages..."
pip install matplotlib==3.10.0 \
    seaborn==0.13.2

# Create GPU configuration file
echo "Creating GPU configuration..."
cat > gpu_config.py << EOL
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
EOL

# Verify GPU support and configuration
echo "Verifying PyTorch GPU support and configuration..."
python - << EOL
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('\nGPU Information:')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    props = torch.cuda.get_device_properties(i)
    print(f'  Memory: {props.total_memory / 1024**3:.1f}GB')
    print(f'  Compute Capability: {props.major}.{props.minor}')
    print(f'  Multi-Processor Count: {props.multi_processor_count}')
EOL

# Add environment variables for optimal multi-GPU setup
echo "Adding multi-GPU environment variables..."
cat >> ~/.bashrc << EOL

# CUDA and GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
EOL

echo "Setup complete! Remember to:"
echo "1. Configure your .env file"
echo "2. Run tests to verify installation"
echo "3. Verify GPU support is working correctly"
echo "4. Source your .bashrc or restart your terminal for GPU environment variables"
echo "5. Import gpu_config in your Python code for optimal multi-GPU usage" 
