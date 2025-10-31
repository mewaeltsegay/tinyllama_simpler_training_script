#!/bin/bash

echo "🚀 Starting H100 Optimized Training"
echo "=================================="

# Check GPU memory before starting
echo "GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "Starting training with optimized H100 configuration..."

# Run with optimized config
python train.py --config configs/h100_optimized.json

echo ""
echo "Training completed!"