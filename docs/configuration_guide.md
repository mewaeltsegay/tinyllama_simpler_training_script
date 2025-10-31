# Configuration Guide for Tigrinya TinyLlama Training

This guide provides comprehensive documentation for configuring the Tigrinya TinyLlama continuous pretraining system.

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Model Configuration](#model-configuration)
3. [Training Parameters](#training-parameters)
4. [Data Configuration](#data-configuration)
5. [Hardware Configuration](#hardware-configuration)
6. [Knowledge Preservation](#knowledge-preservation)
7. [Logging Configuration](#logging-configuration)
8. [Hardware-Specific Configurations](#hardware-specific-configurations)
9. [Inference Configuration](#inference-configuration)
10. [Configuration Examples](#configuration-examples)

## Configuration File Structure

The training system uses JSON configuration files with the following top-level sections:

```json
{
  "model": { ... },
  "training": { ... },
  "data": { ... },
  "hardware": { ... },
  "knowledge_preservation": { ... },
  "logging": { ... }
}
```

## Model Configuration

Controls model loading and tokenizer setup.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `checkpoint_path` | string | Yes | - | Path to TinyLlama model checkpoint or HuggingFace model name |
| `tokenizer_path` | string | Yes | - | Path to SentencePiece tokenizer directory |
| `vocab_size` | integer | No | 32000 | Vocabulary size (auto-detected from tokenizer) |

### Example

```json
{
  "model": {
    "checkpoint_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tokenizer_path": "tokenizer/",
    "vocab_size": 32000
  }
}
```

## Training Parameters

Controls the training process and optimization.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `learning_rate` | float | No | 2e-5 | Learning rate for AdamW optimizer |
| `batch_size` | integer | No | 4 | Training batch size per GPU |
| `gradient_accumulation_steps` | integer | No | 8 | Steps to accumulate gradients before update |
| `max_steps` | integer | No | 10000 | Maximum training steps |
| `warmup_steps` | integer | No | 1000 | Linear warmup steps |
| `save_steps` | integer | No | 500 | Steps between checkpoint saves |
| `eval_steps` | integer | No | 100 | Steps between validation runs |
| `mixed_precision` | string | No | "fp16" | Mixed precision mode: "fp16", "bf16", or "fp32" |
| `gradient_checkpointing` | boolean | No | true | Enable gradient checkpointing for memory savings |

### Hardware-Specific Recommendations

#### RTX 4050 (8GB VRAM)
```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 32,
    "mixed_precision": "fp16",
    "gradient_checkpointing": true
  }
}
```

#### H100 80GB
```json
{
  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "bf16",
    "gradient_checkpointing": false
  }
}
```

## Data Configuration

Controls dataset loading and preprocessing.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tigrinya_dataset` | string | Yes | - | Path to Tigrinya training dataset (JSONL format) |
| `validation_dataset` | string | Yes | - | Path to validation dataset (JSONL format) |
| `english_validation` | string | No | null | Path to English validation dataset for knowledge preservation |
| `max_length` | integer | No | 2048 | Maximum sequence length for training |
| `debug_samples` | integer | No | null | Limit dataset to N samples for debugging |

### Dataset Format

Training datasets should be in JSONL format with each line containing:

```json
{"text": "Your training text here in Tigrinya or English"}
```

### Example

```json
{
  "data": {
    "tigrinya_dataset": "dataset/train.jsonl",
    "validation_dataset": "dataset/validation.jsonl",
    "english_validation": "dataset/english_validation.jsonl",
    "max_length": 2048,
    "debug_samples": 1000
  }
}
```

## Hardware Configuration

Controls hardware utilization and performance optimization.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `device` | string | No | "auto" | Device selection: "auto", "cpu", "cuda" |
| `num_gpus` | integer | No | 1 | Number of GPUs to use |
| `dataloader_workers` | integer | No | 4 | Number of data loading workers |
| `pin_memory` | boolean | No | true | Pin memory for faster GPU transfer |

### Example

```json
{
  "hardware": {
    "device": "auto",
    "num_gpus": 1,
    "dataloader_workers": 4,
    "pin_memory": true
  }
}
```

## Knowledge Preservation

Controls techniques to prevent catastrophic forgetting of English capabilities.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `enabled` | boolean | No | true | Enable knowledge preservation techniques |
| `english_weight` | float | No | 0.3 | Weight for English samples in mixed batches (0.0-1.0) |
| `regularization_strength` | float | No | 0.01 | Strength of regularization penalty |
| `validation_frequency` | integer | No | 100 | Steps between English validation checks |

### Example

```json
{
  "knowledge_preservation": {
    "enabled": true,
    "english_weight": 0.3,
    "regularization_strength": 0.01,
    "validation_frequency": 100
  }
}
```

## Logging Configuration

Controls logging, monitoring, and experiment tracking.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `log_level` | string | No | "INFO" | Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" |
| `wandb_project` | string | No | null | Weights & Biases project name (null to disable) |
| `tensorboard_dir` | string | No | "logs/" | TensorBoard log directory |
| `save_metrics` | boolean | No | true | Save training metrics to files |

### Example

```json
{
  "logging": {
    "log_level": "INFO",
    "wandb_project": "tigrinya-tinyllama-pretraining",
    "tensorboard_dir": "logs/",
    "save_metrics": true
  }
}
```

## Hardware-Specific Configurations

### Debug Configuration (RTX 4050)

Optimized for consumer GPU with limited VRAM:

```json
{
  "model": {
    "checkpoint_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tokenizer_path": "tokenizer/"
  },
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 32,
    "max_steps": 500,
    "mixed_precision": "fp16",
    "gradient_checkpointing": true
  },
  "data": {
    "tigrinya_dataset": "dataset/train.jsonl",
    "validation_dataset": "dataset/validation.jsonl",
    "max_length": 512,
    "debug_samples": 1000
  },
  "hardware": {
    "dataloader_workers": 2
  }
}
```

### Production Configuration (H100 80GB)

Optimized for high-end server GPU:

```json
{
  "model": {
    "checkpoint_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tokenizer_path": "tokenizer/"
  },
  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "max_steps": 10000,
    "mixed_precision": "bf16",
    "gradient_checkpointing": false
  },
  "data": {
    "tigrinya_dataset": "dataset/train.jsonl",
    "validation_dataset": "dataset/validation.jsonl",
    "english_validation": "dataset/english_validation.jsonl",
    "max_length": 2048
  },
  "hardware": {
    "dataloader_workers": 8
  }
}
```

## Inference Configuration

Configuration for text generation inference.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature (0.1-2.0) |
| `top_k` | integer | 50 | Top-k sampling parameter |
| `top_p` | float | 0.9 | Top-p (nucleus) sampling parameter |
| `max_length` | integer | 100 | Maximum generation length |
| `min_length` | integer | 1 | Minimum generation length |
| `repetition_penalty` | float | 1.1 | Repetition penalty factor |
| `do_sample` | boolean | true | Use sampling vs greedy decoding |
| `num_return_sequences` | integer | 1 | Number of sequences to generate |

### Example

```json
{
  "generation": {
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9,
    "max_length": 100,
    "repetition_penalty": 1.1
  }
}
```

## Configuration Examples

### Minimal Configuration

The absolute minimum required configuration:

```json
{
  "model": {
    "checkpoint_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tokenizer_path": "tokenizer/"
  },
  "data": {
    "tigrinya_dataset": "dataset/train.jsonl",
    "validation_dataset": "dataset/validation.jsonl"
  }
}
```

### Quick Test Configuration

For rapid testing and validation:

```json
{
  "model": {
    "checkpoint_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tokenizer_path": "tokenizer/"
  },
  "training": {
    "max_steps": 100,
    "save_steps": 50,
    "eval_steps": 25
  },
  "data": {
    "tigrinya_dataset": "dataset/train.jsonl",
    "validation_dataset": "dataset/validation.jsonl",
    "debug_samples": 100
  }
}
```

### Multi-GPU Configuration

For distributed training across multiple GPUs:

```json
{
  "model": {
    "checkpoint_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tokenizer_path": "tokenizer/"
  },
  "training": {
    "batch_size": 4,
    "gradient_accumulation_steps": 2
  },
  "data": {
    "tigrinya_dataset": "dataset/train.jsonl",
    "validation_dataset": "dataset/validation.jsonl"
  },
  "hardware": {
    "num_gpus": 4,
    "dataloader_workers": 16
  }
}
```

## Configuration Validation

The system automatically validates configurations and provides helpful error messages:

- **File existence**: Checks that model, tokenizer, and dataset paths exist
- **Parameter ranges**: Validates that numeric parameters are within acceptable ranges
- **Logical consistency**: Ensures parameters make sense together (e.g., warmup_steps < max_steps)
- **Hardware compatibility**: Warns about potential memory issues

## Configuration Override

You can override configuration parameters using:

1. **Override files**: Use `--override-config` to merge additional parameters
2. **Command line flags**: Use `--debug` to apply debug-specific settings
3. **Environment variables**: Some parameters can be set via environment variables

## Best Practices

1. **Start with templates**: Use provided hardware-specific templates as starting points
2. **Test with debug mode**: Always test new configurations with `debug_samples` first
3. **Monitor memory usage**: Watch GPU memory consumption and adjust batch sizes accordingly
4. **Save configurations**: Keep successful configurations for reproducibility
5. **Version control**: Track configuration changes alongside code changes

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce `batch_size`, increase `gradient_accumulation_steps`, enable `gradient_checkpointing`
2. **Slow training**: Increase `batch_size`, reduce `gradient_accumulation_steps`, use more `dataloader_workers`
3. **Poor convergence**: Adjust `learning_rate`, check `warmup_steps`, verify data quality
4. **Validation errors**: Check dataset format, verify file paths, ensure proper encoding

### Memory Optimization

For limited GPU memory:

```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 32,
    "mixed_precision": "fp16",
    "gradient_checkpointing": true
  },
  "data": {
    "max_length": 512
  }
}
```

### Performance Optimization

For maximum training speed:

```json
{
  "training": {
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "gradient_checkpointing": false
  },
  "hardware": {
    "dataloader_workers": 8,
    "pin_memory": true
  }
}
```