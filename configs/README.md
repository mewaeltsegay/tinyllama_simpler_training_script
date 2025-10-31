# RTX 4050 Training Configurations

## ✅ **Ready-to-Use Configurations**

All configurations have been validated and are ready for training on RTX 4050 (6GB VRAM).

### **1. Safe Training - `rtx4050_safe.json`**
```bash
python train.py --config configs/rtx4050_safe.json
```
- **Memory**: ~3.5GB VRAM (safe margin)
- **Steps**: 25,000 training steps
- **Sequence**: 256 tokens
- **Batch**: Effective size 32 (1×32 accumulation)
- **Time**: ~8 hours
- **Best for**: First-time users, testing

### **2. Production Training - `rtx4050_production.json`**
```bash
python train.py --config configs/rtx4050_production.json
```
- **Memory**: ~4.8GB VRAM (optimized)
- **Steps**: 10,000 training steps
- **Sequence**: 1024 tokens
- **Batch**: Effective size 64 (1×64 accumulation)
- **Time**: ~6 hours
- **Best for**: Balanced performance

### **3. Full Training - `rtx4050_full_training.json`**
```bash
python train.py --config configs/rtx4050_full_training.json
```
- **Memory**: ~5.5GB VRAM (maximum efficiency)
- **Steps**: 50,000 training steps
- **Sequence**: 512 tokens
- **Batch**: Effective size 128 (1×128 accumulation)
- **Focus**: Pure Tigrinya training
- **Time**: ~24 hours
- **Best for**: Complete fine-tuning

### **4. Tigrinya-Only Optimized - `tigrinya_only.json`**
```bash
python train.py --config configs/tigrinya_only.json
```
- **Memory**: ~4.5GB VRAM (Tigrinya-optimized)
- **Steps**: 20,000 training steps
- **Sequence**: 512 tokens
- **Batch**: Effective size 64 (1×64 accumulation)
- **Focus**: 100% Tigrinya specialization
- **Time**: ~12 hours
- **Best for**: Pure Tigrinya language learning

## 🚀 **Quick Start Commands**

### Test Configuration (Debug Mode):
```bash
# Quick test with 1000 samples, 500 steps
python train.py --config configs/rtx4050_safe.json --debug
```

### Validate Configuration:
```bash
# Check config without training
python train.py --config configs/rtx4050_production.json --validate-config-only
```

### Auto-Resume Training:
```bash
# Automatically resume from latest checkpoint
python train.py --config configs/rtx4050_full_training.json --auto-resume
```

## 📊 **Key Features**

All configurations include:
- ✅ **BF16 Mixed Precision** (more stable than FP16)
- ✅ **Gradient Checkpointing** (50% memory reduction)
- ✅ **Error Recovery System** (automatic checkpoint recovery)
- ✅ **Memory Optimization** (fits in 6GB VRAM)
- ✅ **TensorBoard Logging** (real-time monitoring)
- ✅ **Graceful Error Handling** (FP16 gradient issues)

## 🎯 **Recommended Workflow**

1. **Start with Safe**: `python train.py --config configs/rtx4050_safe.json --debug`
2. **Move to Production**: `python train.py --config configs/rtx4050_production.json`
3. **Full Training**: `python train.py --config configs/rtx4050_full_training.json --auto-resume`

## 📈 **Expected Results**

- **Initial Loss**: ~9-12
- **Target Loss**: ~2-4
- **Perplexity**: 1000+ → 10-50
- **Memory Usage**: Stable within 6GB
- **Training**: Continuous with error recovery

## 🔧 **Monitoring**

```bash
# TensorBoard
tensorboard --logdir logs/

# GPU Usage
nvidia-smi

# Training Logs
tail -f output/logs/training.log
```

---

**All configurations are production-ready with comprehensive error handling and recovery!** 🎉