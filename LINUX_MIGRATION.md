# Linux Migration Checklist

## Pre-Migration Steps
1. Backup your data:
   - [ ] Copy all data from `data/` directory
   - [ ] Backup `.env` files (but don't transfer sensitive credentials directly)
   - [ ] Note: Neo4j database will be handled separately with recover_from_nuke script

## System Requirements
- Ubuntu (recommended 20.04 LTS or newer)
- Miniconda or Anaconda installed
- Git installed
- For GPU Support (✓ Already Installed):
  - [x] 2x NVIDIA A6000 GPUs (48GB each)
  - [x] NVIDIA driver 535
  - [x] CUDA toolkit 12.2
- Database (✓ Already Installed):
  - [x] Neo4j with blank database

## Pre-Installation GPU Check
1. Verify NVIDIA driver and GPUs:
   ```bash
   nvidia-smi
   ```
   Expected output should show both A6000s with driver version 535

2. Verify CUDA installation:
   ```bash
   nvcc --version
   ```
   Should show CUDA version 12.2

3. Verify GPU memory:
   ```bash
   nvidia-smi --query-gpu=memory.total,memory.free --format=csv
   ```
   Should show ~48GB per GPU

## Installation Steps

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Dell-AITC
   ```

2. Make the setup script executable:
   ```bash
   chmod +x linux_setup.sh
   ```

3. Run the setup script:
   ```bash
   ./linux_setup.sh
   ```
   Note: The script will configure for dual A6000 GPUs automatically

4. Configure environment:
   - [ ] Copy and modify `.env` file for Linux paths
   - [ ] Update any Windows-specific paths in configuration files
   - [ ] Verify gpu_config.py was created

## Post-Installation Verification

1. Test the environment:
   ```bash
   conda activate Dell-AITC
   pytest backend/tests/
   ```

2. Verify critical components:
   - [ ] FastAPI server starts
   - [ ] AI models load correctly
   - [ ] File paths work correctly
   - [ ] Both GPUs are recognized:
     ```python
     from gpu_config import setup_gpus
     setup_gpus()  # Should show both A6000s
     ```

3. Check specific functionalities:
   - [ ] AI model inference (on both GPUs)
   - [ ] File I/O operations
   - [ ] API endpoints
   - [ ] Multi-GPU model distribution

4. Database Setup (Separate Process):
   - [ ] Run recover_from_nuke script to populate database
   - [ ] Verify database connectivity
   - [ ] Test database operations

## Multi-GPU Usage

1. Basic GPU Selection:
   ```python
   from gpu_config import get_optimal_device
   device = get_optimal_device()  # Automatically selects GPU with most free memory
   ```

2. Parallel Processing:
   ```python
   from gpu_config import parallel_config
   import torch.nn as nn
   
   config = parallel_config()
   if config:
       model = nn.DataParallel(model, **config)
   ```

3. Memory Management:
   ```python
   import torch
   torch.cuda.empty_cache()  # Clear GPU memory if needed
   ```

## Known Differences from Windows

1. Path separators:
   - Windows uses backslashes (`\`)
   - Linux uses forward slashes (`/`)
   - Update any hardcoded paths in your code

2. Package considerations:
   - torch: Now installed with CUDA 12.2 support for A6000s
   - GPU memory management is more critical
   - Multi-GPU support enabled by default

3. Environment variables:
   - Update any Windows-specific paths
   - Use Linux path format
   - GPU-specific variables are set in .bashrc:
     ```bash
     export CUDA_VISIBLE_DEVICES=0,1
     export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
     export CUDA_LAUNCH_BLOCKING=0
     ```

## Troubleshooting

1. If AI models fail to load:
   - Check model paths
   - Verify PyTorch CUDA compatibility:
     ```python
     python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
     ```
   - Common GPU issues:
     - Memory fragmentation: Use `torch.cuda.empty_cache()`
     - Load balancing: Check both GPUs with `nvidia-smi -l 1`
     - CUDA version mismatch: Verify PyTorch CUDA version matches system
     - Environment issues: Check CUDA paths and environment variables

2. If file operations fail:
   - Check file permissions
   - Verify path separators
   - Ensure directories exist

3. If database operations fail:
   - Note: Handle database issues with your existing database management scripts

## Additional Notes

- The setup creates a `gpu_config.py` utility for managing dual A6000s
- A6000s have 48GB each - utilize this for larger batch sizes and models
- For optimal performance with your dual A6000 setup:
  ```bash
  # Memory management (in ~/.bashrc)
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  export CUDA_LAUNCH_BLOCKING=0
  
  # Monitor both GPUs
  watch -n 1 "nvidia-smi"
  ```
- For distributed training:
  ```python
  import torch.distributed as dist
  from torch.nn.parallel import DistributedDataParallel
  # See gpu_config.py for helper functions
  ```
- Consider using Docker with NVIDIA Container Toolkit for isolation:
  ```bash
  # Your system supports nvidia-docker2 with multi-GPU
  ``` 