#!/bin/bash

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

# Verify existing CUDA installation
echo "Verifying existing CUDA installation..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA driver not found. Please install NVIDIA driver 535 first."
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA toolkit not found. Please install CUDA 12.2 first."
    exit 1
fi

echo "Found NVIDIA installation:"
nvidia-smi
echo "CUDA version:"
nvcc --version

# Check for multiple GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPU(s)"
if [ "$GPU_COUNT" -eq 2 ]; then
    echo "Detected dual A6000s - will configure for multi-GPU support"
fi

# Create and activate conda environment
echo "Creating conda environment..."
conda create -n Dell-AITC python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate Dell-AITC

