# RTX 4050 Training Configurations Guide

This guide provides optimized configurations for training TinyLlama on RTX 4050 (6GB VRAM) with different training objectives.

## 🚀 Quick Start

Choose the configuration that best fits your needs:

### 1. Safe Training (`rtx4050_safe.json`)
**Recommended for: First-time users, testing, short experiments**

```bash
python train.py --config configs/rtx4050_safe.json
```

**Features:**
- ✅ Guaranteed to fit in 6GB VRAM
- ✅ Stable training with minimal memory usage
- ✅ 25,000 training steps (moderate training)
- ✅ 256 token sequence length
- ✅ BF16 mixed precision for stability

**Memory Usage:** ~3.5GB VRAM

### 2. Production Training (`rtx4050_production.json`)
**Recommended for: Balanced performance and memory efficiency**

```bash
python train.py --config configs/rtx4050_production.json
```

**Features:**
- ✅ Optimized for RTX 4050 performance
- ✅ 10,000 training steps with quality focus
- ✅ 1024 token sequence length
- ✅ Advanced optimization settings
- ✅ Comprehensive logging and checkpointing

**Memory Usage:** ~4.8GB VRAM

### 3. Full Training (`rtx4050_full_training.json`)
**Recommended for: Complete model fine-tuning, production deployment**

```bash
python train.py --config configs/rtx4050_full_training.json
```

**Features:**
- ✅ Complete 50,000 step training
- ✅ Knowledge preservation with English data
- ✅ Advanced learning rate scheduling
- ✅ Comprehensive validation and metrics
- ✅ Production-ready checkpointing

**Memory Usage:** ~5.5GB VRAM

## 📊 Configuration Comparison

| Feature | Safe | Production | Full Training |
|---------|------|------------|---------------|
| Max Steps | 25,000 | 10,000 | 50,000 |
| Sequence Length | 256 | 1024 | 512 |
| Batch Size | 1 | 1 | 1 |
| Gradient Accumulation | 32 | 64 | 128 |
| Effective Batch Size | 32 | 64 | 128 |
| Knowledge Preservation | ❌ | ❌ | ✅ (10% English) |
| Mixed Precision | BF16 | BF16 | BF16 |
| Memory Usage | ~3.5GB | ~4.8GB | ~5.5GB |
| Training Time | ~8 hours | ~6 hours | ~24 hours |

## 🔧 Memory Optimization Features

All configurations include:

### Automatic Memory Management
- **Gradient Checkpointing**: Reduces activation memory by 50%
- **BF16 Mixed Precision**: Halves model memory usage
- **Dynamic Batch Sizing**: Automatic OOM recovery
- **Memory Monitoring**: Real-time VRAM usage tracking

### Error Recovery System
- **Automatic Checkpoint Recovery**: Resume from failures
- **FP16 Gradient Handling**: Graceful mixed precision error recovery
- **Data Loading Resilience**: Skip corrupted samples
- **Emergency Checkpointing**: Save progress during crashes

### Performance Optimizations
- **Fused Optimizers**: Faster gradient updates
- **Efficient Data Loading**: Optimized batch processing
- **Smart Scheduling**: Cosine annealing with restarts
- **Validation Efficiency**: Streamlined evaluation

## 🎯 Training Recommendations

### For RTX 4050 (6GB VRAM):

1. **Start with Safe Config** to verify everything works
2. **Use Production Config** for balanced training
3. **Use Full Training Config** for final model

### Memory Tips:
- Close other GPU applications before training
- Use `nvidia-smi` to monitor VRAM usage
- Enable Windows GPU scheduling for better memory management
- Consider training during off-peak hours for stability

### Performance Tips:
- Use SSD storage for datasets (faster I/O)
- Ensure adequate cooling (training generates heat)
- Monitor GPU temperature during long training runs
- Use `--auto-resume` flag for automatic recovery

## 📈 Expected Results

### Training Progress:
- **Initial Loss**: ~9-12 (typical for language models)
- **Target Loss**: ~2-4 (good convergence)
- **Perplexity**: Should decrease from 1000+ to 10-50

### Validation Metrics:
- **Tigrinya Perplexity**: Target < 20
- **Validation Loss**: Should track training loss
- **GPU Memory**: Stable throughout training

## 🚨 Troubleshooting

### Common Issues:

1. **Out of Memory Error**:
   ```bash
   # Use safer config
   python train.py --config configs/rtx4050_safe.json
   ```

2. **Training Stops**:
   ```bash
   # Resume automatically
   python train.py --config configs/rtx4050_production.json --auto-resume
   ```

3. **Slow Training**:
   - Reduce `dataloader_workers` to 0
   - Disable `pin_memory` if system RAM is low
   - Use shorter sequences (reduce `max_length`)

4. **FP16 Gradient Issues**:
   - Switch to BF16 (already configured)
   - Reduce learning rate
   - Increase gradient clipping

## 📝 Monitoring Training

### Real-time Monitoring:
```bash
# TensorBoard (recommended)
tensorboard --logdir logs/rtx4050_production

# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f output/logs/training.log
```

### Key Metrics to Watch:
- **Loss**: Should decrease over time
- **Learning Rate**: Should follow schedule
- **GPU Memory**: Should remain stable
- **Gradient Norm**: Should be < 10

## 🎉 Success Indicators

Your training is successful when you see:
- ✅ Consistent loss reduction
- ✅ Stable GPU memory usage
- ✅ Regular checkpoint creation
- ✅ Validation perplexity improving
- ✅ No repeated error recovery

## 💡 Advanced Usage

### Custom Configuration:
```bash
# Override specific settings
python train.py --config configs/rtx4050_production.json --override-config my_overrides.json

# Use recovery plan from previous failure
python train.py --config configs/rtx4050_production.json --recovery-plan output/recovery_plan.json

# Debug mode with limited samples
python train.py --config configs/rtx4050_production.json --debug
```

### Production Deployment:
```bash
# Full training with all features
python train.py --config configs/rtx4050_full_training.json --auto-resume --log-level INFO
```

---

**Happy Training! 🚀**

For issues or questions, check the logs in `output/logs/training.log` or review the recovery plan in `output/recovery_plan.json`.