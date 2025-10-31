#!/bin/bash

# H100 Safe Training Script - Disables all compilation and problematic features

echo "🚀 Starting H100 Safe Training Mode..."

# Disable all PyTorch compilation
export TORCH_COMPILE_DISABLE=1
export TORCH_DYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Disable optimizations that can cause issues
export TORCH_CUDNN_V8_API_DISABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TORCH_BACKENDS_CUDNN_BENCHMARK=0

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Disable problematic features
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false

echo "Environment configured for H100 safe mode"
echo "TORCH_COMPILE_DISABLE=$TORCH_COMPILE_DISABLE"
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"

# Run training with safe configuration
python train.py --config configs/h100_no_compile.json "$@"