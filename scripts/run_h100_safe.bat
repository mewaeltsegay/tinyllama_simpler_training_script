@echo off
REM H100 Safe Training Script - Disables all compilation and problematic features

echo 🚀 Starting H100 Safe Training Mode...

REM Disable all PyTorch compilation
set TORCH_COMPILE_DISABLE=1
set TORCH_DYNAMO_DISABLE=1
set TORCHINDUCTOR_DISABLE=1

REM Enable CUDA debugging
set CUDA_LAUNCH_BLOCKING=1
set TORCH_USE_CUDA_DSA=1

REM Disable optimizations that can cause issues
set TORCH_CUDNN_V8_API_DISABLED=1
set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
set TORCH_BACKENDS_CUDNN_BENCHMARK=0

REM Memory management
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

REM Disable problematic features
set TRANSFORMERS_NO_ADVISORY_WARNINGS=1
set TOKENIZERS_PARALLELISM=false

echo Environment configured for H100 safe mode
echo TORCH_COMPILE_DISABLE=%TORCH_COMPILE_DISABLE%
echo CUDA_LAUNCH_BLOCKING=%CUDA_LAUNCH_BLOCKING%

REM Run training with safe configuration
python train.py --config configs/h100_no_compile.json %*