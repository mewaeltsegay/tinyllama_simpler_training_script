# Tigrinya TinyLlama Continuous Pretraining

This project implements continuous pretraining of TinyLlama models for the Tigrinya language while preserving existing English capabilities. The system supports flexible hardware configurations from consumer GPUs to enterprise setups through JSON-based configuration management.

## Features

- **Bilingual Training**: Continuous pretraining for Tigrinya while preserving English knowledge
- **Hardware Adaptability**: Automatic configuration for RTX 4050 (8GB) to H100 80GB setups
- **Knowledge Preservation**: Techniques to prevent catastrophic forgetting
- **Mixed Precision Training**: FP16/BF16 support for memory optimization
- **Comprehensive Monitoring**: Integration with Weights & Biases and TensorBoard
- **Flexible Configuration**: JSON-based parameter management
- **Quality Validation**: Bilingual text generation quality assessment
- **Interactive Inference**: Real-time text generation with parameter tuning
- **Batch Processing**: Efficient processing of multiple prompts

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```bash
# Debug training on consumer GPU
python train.py --config configs/debug_rtx4050.json --debug

# Production training on server GPU
python train.py --config configs/production_h100.json

# Resume from checkpoint
python train.py --config configs/production_h100.json --resume-from checkpoints/step_5000
```

### Inference

```bash
# Interactive text generation
python inference.py -m checkpoints/step_1000 -t tokenizer/ --mode interactive

# Single generation
python inference.py -m checkpoints/step_1000 -t tokenizer/ --mode single --prompt "ሰላም! ከመይ ኣሎኻ?"

# Batch processing
python inference.py -m checkpoints/step_1000 -t tokenizer/ --mode batch -i prompts.txt -o results.txt

# Parameter optimization
python inference.py -m checkpoints/step_1000 -t tokenizer/ --optimize-parameters --test-prompts test.txt
```

## Project Structure

```
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── data/              # Data loading and preprocessing
│   ├── model/             # Model and tokenizer management
│   ├── training/          # Training engine and monitoring
│   ├── inference/         # Text generation and quality validation
│   └── utils/             # Utilities and logging
├── configs/               # Configuration templates
│   ├── debug_rtx4050.json      # Debug config for RTX 4050
│   ├── production_h100.json    # Production config for H100
│   ├── inference_config.json   # Inference parameters
│   └── minimal_config.json     # Minimal required config
├── docs/                  # Documentation
│   ├── configuration_guide.md  # Comprehensive config docs
│   └── usage_guide.md          # Step-by-step usage guide
├── dataset/               # Training data (JSONL format)
├── tokenizer/             # Tigrinya SentencePiece tokenizer
├── train.py              # Main training script
├── inference.py          # Text generation script
└── README.md             # This file
```

## Configuration Templates

The system provides several configuration templates for different use cases:

### Debug Configuration (RTX 4050)
Optimized for consumer GPU with limited VRAM:
```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 32,
    "max_steps": 500,
    "mixed_precision": "fp16",
    "gradient_checkpointing": true
  },
  "data": {
    "max_length": 512,
    "debug_samples": 1000
  }
}
```

### Production Configuration (H100)
Optimized for high-end server GPU:
```json
{
  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "max_steps": 10000,
    "mixed_precision": "bf16",
    "gradient_checkpointing": false
  },
  "data": {
    "max_length": 2048
  }
}
```

### Inference Configuration
Parameters for text generation:
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

## Training Scripts

### Main Training Script (`train.py`)

Comprehensive training script with:
- JSON configuration loading and validation
- Hardware detection and optimization
- Automatic checkpoint saving and resumption
- Real-time monitoring and logging
- Error handling and recovery

**Key Features:**
- Command-line argument parsing
- Configuration override support
- Debug mode for quick testing
- Comprehensive error handling
- Integration with monitoring systems

### Usage Examples

```bash
# Basic training
python train.py --config configs/debug_rtx4050.json

# Training with overrides
python train.py --config configs/base.json --override-config configs/overrides.json

# Debug mode with custom output directory
python train.py --config configs/production_h100.json --debug --output-dir ./debug_run

# Resume training from checkpoint
python train.py --config configs/production_h100.json --resume-from ./output/checkpoints/step_5000

# Validate configuration only
python train.py --config configs/my_config.json --validate-config-only
```

## Inference Scripts

### Main Inference Script (`inference.py`)

Flexible inference script supporting:
- Interactive text generation
- Single prompt processing
- Batch processing from files
- Quality validation
- Parameter optimization

**Generation Modes:**
1. **Interactive**: Real-time generation with parameter tuning
2. **Single**: Generate from one prompt
3. **Batch**: Process multiple prompts from file

**Key Features:**
- Configurable generation parameters
- Quality assessment and validation
- Language detection and consistency checking
- Parameter optimization for better quality
- Support for both JSON and text output formats

### Usage Examples

```bash
# Interactive mode with quality validation
python inference.py -m checkpoints/step_1000 -t tokenizer/ --mode interactive --validate-quality

# Single generation with custom parameters
python inference.py -m checkpoints/step_1000 -t tokenizer/ --mode single \
  --prompt "ሰላም! ከመይ ኣሎኻ?" --temperature 0.8 --max-length 150

# Batch processing with quality validation
python inference.py -m checkpoints/step_1000 -t tokenizer/ --mode batch \
  --input-file prompts.txt --output-file results.json --validate-quality

# Parameter optimization
python inference.py -m checkpoints/step_1000 -t tokenizer/ \
  --optimize-parameters --test-prompts test_prompts.txt
```

## Documentation

- **[Configuration Guide](docs/configuration_guide.md)** - Comprehensive documentation of all configuration parameters, hardware-specific settings, and best practices
- **[Usage Guide](docs/usage_guide.md)** - Step-by-step instructions for training and inference, troubleshooting, and performance optimization

## Hardware Requirements

### Minimum (Debug/Development)
- **GPU**: RTX 4050 (8GB VRAM) or equivalent
- **RAM**: 16GB system memory
- **Storage**: 50GB free space
- **CUDA**: 11.8+ or compatible

### Recommended (Production)
- **GPU**: H100 80GB, A100 80GB, or RTX 4090
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ NVMe SSD
- **CUDA**: 12.0+ for optimal performance

### Supported Hardware Configurations

The system automatically detects and optimizes for:
- **Consumer GPUs**: RTX 3080, RTX 3090, RTX 4080, RTX 4090
- **Professional GPUs**: A100, H100, V100, A6000
- **Multi-GPU setups**: Distributed training support

## Dataset Format

Training datasets should be in JSONL format with each line containing:

```json
{"text": "Your training text here in Tigrinya or English"}
```

Example:
```json
{"text": "ሰላም! ከመይ ኣሎኻ? ጽቡቕ እየ።"}
{"text": "Hello! How are you? I am fine."}
{"text": "ትግርኛ ቋንቋ ኤርትራን ሰሜን ኢትዮጵያን እዩ።"}
```

## Monitoring and Logging

### Built-in Monitoring
- Real-time console output with training metrics
- Structured logging to files
- GPU memory and utilization tracking
- Validation metrics for both languages

### External Integration
- **Weights & Biases**: Experiment tracking and visualization
- **TensorBoard**: Training metrics and loss curves
- **Custom metrics**: JSON/JSONL format for analysis

### Log Files
- `training.log`: Detailed training logs
- `training_metrics.jsonl`: Step-by-step metrics
- `validation_metrics.jsonl`: Validation results
- `training_summary.json`: Final training summary

## Quality Validation

The system includes comprehensive quality validation:

### Metrics
- **Language consistency**: Input vs output language matching
- **Repetition analysis**: Detection of repetitive patterns
- **Lexical diversity**: Vocabulary richness assessment
- **Coherence scoring**: Sentence structure evaluation
- **Perplexity calculation**: Model confidence measurement

### Quality Ratings
- **Excellent** (0.8-1.0): High-quality, coherent generation
- **Good** (0.6-0.8): Acceptable quality with minor issues
- **Fair** (0.4-0.6): Usable but needs improvement
- **Poor** (0.2-0.4): Significant quality issues
- **Very Poor** (0.0-0.2): Unusable output

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training
   - Reduce sequence length

2. **Slow Training**
   - Increase batch size if memory allows
   - Use more data loader workers
   - Disable gradient checkpointing if memory permits
   - Check GPU utilization

3. **Poor Generation Quality**
   - Adjust temperature (try 0.8-1.2)
   - Increase repetition penalty
   - Use parameter optimization
   - Ensure sufficient training

4. **Configuration Errors**
   - Use `--validate-config-only` flag
   - Check file paths exist
   - Verify parameter ranges
   - Compare with working templates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TinyLlama team for the base model architecture
- Tigrinya language community for dataset contributions
- HuggingFace for the transformers library
- PyTorch team for the deep learning framework