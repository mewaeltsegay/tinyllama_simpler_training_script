# Usage Guide for Tigrinya TinyLlama Training and Inference

This guide provides step-by-step instructions for training and using the Tigrinya TinyLlama model.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training](#training)
3. [Inference](#inference)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

1. Python 3.8+ with required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Tigrinya tokenizer in `tokenizer/` directory
3. Training data in JSONL format in `dataset/` directory

### Basic Training

1. **Debug training on RTX 4050:**
   ```bash
   python train.py --config configs/debug_rtx4050.json --debug
   ```

2. **Production training on H100:**
   ```bash
   python train.py --config configs/production_h100.json
   ```

### Basic Inference

1. **Interactive mode:**
   ```bash
   python inference.py -m checkpoints/step_1000 -t tokenizer/ --mode interactive
   ```

2. **Single generation:**
   ```bash
   python inference.py -m checkpoints/step_1000 -t tokenizer/ --mode single --prompt "ሰላም! ከመይ ኣሎኻ?"
   ```

## Training

### Command Line Interface

```bash
python train.py [OPTIONS]
```

#### Required Arguments

- `--config, -c`: Path to JSON configuration file

#### Optional Arguments

- `--override-config`: Path to configuration override file
- `--resume-from`: Path to checkpoint directory to resume from
- `--output-dir`: Output directory for checkpoints and logs (default: ./output)
- `--debug`: Enable debug mode with reduced dataset
- `--device`: Device selection (auto, cpu, cuda)
- `--validate-config-only`: Only validate configuration and exit
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-file`: Path to log file

### Training Examples

#### 1. Debug Training

Quick validation on consumer hardware:

```bash
python train.py \
  --config configs/debug_rtx4050.json \
  --debug \
  --output-dir ./debug_output \
  --log-level DEBUG
```

#### 2. Production Training

Full training on server hardware:

```bash
python train.py \
  --config configs/production_h100.json \
  --output-dir ./production_output \
  --log-level INFO
```

#### 3. Resume Training

Continue from a previous checkpoint:

```bash
python train.py \
  --config configs/production_h100.json \
  --resume-from ./output/checkpoints/step_5000 \
  --output-dir ./resumed_output
```

#### 4. Configuration Override

Use base config with overrides:

```bash
python train.py \
  --config configs/production_h100.json \
  --override-config configs/custom_overrides.json
```

### Training Process

1. **Configuration Loading**: System loads and validates configuration
2. **Hardware Detection**: Automatically detects GPU capabilities
3. **Model Loading**: Loads TinyLlama model and Tigrinya tokenizer
4. **Data Preparation**: Sets up training and validation data loaders
5. **Training Loop**: Executes training with monitoring and checkpointing
6. **Validation**: Periodic evaluation on validation sets
7. **Checkpointing**: Saves model state at configured intervals

### Monitoring Training

#### Console Output

Training progress is displayed in real-time:

```
Step 100: Loss=2.1234, LR=1.50e-05, GPU=6.2GB
Step 200: Loss=2.0987, LR=1.48e-05, GPU=6.2GB
Validation - Loss: 2.0543, Perplexity: 7.82
```

#### Log Files

Detailed logs are saved to `output_dir/logs/training.log`:

```
2024-10-31 10:30:15 INFO Starting training loop...
2024-10-31 10:30:16 INFO Step 1: Loss=3.2145, LR=2.00e-06
2024-10-31 10:30:17 INFO GPU Memory: 5.8GB / 8.0GB
```

#### Metrics Files

Training metrics are saved in structured format:

- `output_dir/metrics/training_metrics.jsonl`: Step-by-step training metrics
- `output_dir/metrics/validation_metrics.jsonl`: Validation results
- `output_dir/metrics/training_summary.json`: Final training summary

### Checkpoints

Checkpoints are saved to `output_dir/checkpoints/step_N/`:

```
checkpoints/step_1000/
├── pytorch_model.bin      # Model weights
├── optimizer.pt           # Optimizer state
├── scheduler.pt           # Learning rate scheduler state
├── scaler.pt             # Mixed precision scaler (if used)
└── training_metadata.json # Training metadata
```

## Inference

### Command Line Interface

```bash
python inference.py [OPTIONS]
```

#### Required Arguments

- `--model-path, -m`: Path to trained model checkpoint
- `--tokenizer-path, -t`: Path to tokenizer directory

#### Mode Selection

- `--mode`: Generation mode (interactive, batch, single)

#### Generation Parameters

- `--temperature`: Sampling temperature (0.1-2.0, default: 1.0)
- `--top-k`: Top-k sampling (default: 50)
- `--top-p`: Top-p sampling (default: 0.9)
- `--max-length`: Maximum generation length (default: 100)
- `--repetition-penalty`: Repetition penalty (default: 1.1)
- `--num-return-sequences`: Number of sequences to generate (default: 1)

#### Quality and Optimization

- `--validate-quality`: Enable quality validation
- `--optimize-parameters`: Optimize generation parameters
- `--test-prompts`: File with test prompts for optimization

### Inference Modes

#### 1. Interactive Mode

Real-time text generation with user interaction:

```bash
python inference.py \
  -m checkpoints/step_5000 \
  -t tokenizer/ \
  --mode interactive \
  --validate-quality
```

Interactive commands:
- `help`: Show available commands
- `config`: Display current generation parameters
- `set <param> <value>`: Change generation parameter
- `quit`/`exit`/`q`: Exit interactive mode

#### 2. Single Generation

Generate text from a single prompt:

```bash
python inference.py \
  -m checkpoints/step_5000 \
  -t tokenizer/ \
  --mode single \
  --prompt "ሰላም! ከመይ ኣሎኻ?" \
  --temperature 0.8 \
  --max-length 150
```

#### 3. Batch Generation

Process multiple prompts from a file:

```bash
python inference.py \
  -m checkpoints/step_5000 \
  -t tokenizer/ \
  --mode batch \
  --input-file prompts.txt \
  --output-file results.txt \
  --validate-quality
```

Input file format (`prompts.txt`):
```
ሰላም! ከመይ ኣሎኻ?
Hello, how are you?
# This is a comment and will be ignored
Tell me about Tigrinya language
```

### Generation Parameters

#### Temperature
Controls randomness in generation:
- `0.1-0.5`: More focused, deterministic
- `0.8-1.2`: Balanced creativity and coherence
- `1.5-2.0`: More creative, less predictable

#### Top-k and Top-p
Control vocabulary selection:
- `top_k=30`: Consider only top 30 tokens
- `top_p=0.9`: Consider tokens comprising 90% probability mass

#### Example Parameter Combinations

**Conservative (coherent, safe):**
```bash
--temperature 0.7 --top-k 30 --top-p 0.8 --repetition-penalty 1.2
```

**Balanced (default):**
```bash
--temperature 1.0 --top-k 50 --top-p 0.9 --repetition-penalty 1.1
```

**Creative (diverse, experimental):**
```bash
--temperature 1.3 --top-k 80 --top-p 0.95 --repetition-penalty 1.05
```

### Quality Validation

Enable quality assessment with `--validate-quality`:

```bash
python inference.py \
  -m checkpoints/step_5000 \
  -t tokenizer/ \
  --mode single \
  --prompt "ሰላም!" \
  --validate-quality
```

Quality metrics include:
- Overall quality rating (excellent, good, fair, poor)
- Quality score (0.0-1.0)
- Language consistency
- Repetition analysis
- Lexical diversity
- Coherence assessment

### Parameter Optimization

Automatically find optimal generation parameters:

```bash
python inference.py \
  -m checkpoints/step_5000 \
  -t tokenizer/ \
  --optimize-parameters \
  --test-prompts test_prompts.txt \
  --output-file optimization_results.json
```

## Configuration

### Using Configuration Files

#### Training Configuration

```bash
python train.py --config my_config.json
```

#### Inference Configuration

```bash
python inference.py \
  -m checkpoints/step_5000 \
  -t tokenizer/ \
  --config inference_config.json \
  --mode interactive
```

### Configuration Templates

Use provided templates as starting points:

- `configs/debug_rtx4050.json`: Consumer GPU debugging
- `configs/production_h100.json`: Server GPU production
- `configs/minimal_config.json`: Minimal required parameters
- `configs/inference_config.json`: Inference parameters

### Custom Configurations

Create custom configurations by copying and modifying templates:

```bash
cp configs/debug_rtx4050.json configs/my_config.json
# Edit my_config.json as needed
python train.py --config configs/my_config.json
```

## Monitoring

### Real-time Monitoring

#### Console Output
Monitor training progress in real-time through console output.

#### TensorBoard
View training metrics in TensorBoard:

```bash
tensorboard --logdir logs/
```

#### Weights & Biases
Configure W&B in your training config:

```json
{
  "logging": {
    "wandb_project": "my-tigrinya-project"
  }
}
```

### Log Analysis

#### Training Logs
Analyze training progress:

```bash
tail -f output/logs/training.log
grep "Validation" output/logs/training.log
```

#### Metrics Files
Process metrics programmatically:

```python
import json

# Load training metrics
with open('output/metrics/training_metrics.jsonl', 'r') as f:
    metrics = [json.loads(line) for line in f]

# Analyze loss progression
losses = [m['loss'] for m in metrics]
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `batch_size` in configuration
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Reduce `max_length`
- Use `mixed_precision: "fp16"`

#### 2. Slow Training

**Symptoms:**
- Very low tokens/second
- High GPU memory usage but low utilization

**Solutions:**
- Increase `batch_size`
- Reduce `gradient_accumulation_steps`
- Increase `dataloader_workers`
- Disable `gradient_checkpointing` if memory allows

#### 3. Poor Generation Quality

**Symptoms:**
- Repetitive text
- Incoherent output
- Wrong language

**Solutions:**
- Adjust `temperature` (try 0.8-1.2)
- Increase `repetition_penalty`
- Use `--optimize-parameters` to find better settings
- Check if model needs more training

#### 4. Configuration Errors

**Symptoms:**
```
Configuration validation failed: ...
```

**Solutions:**
- Check file paths exist
- Verify parameter ranges
- Use `--validate-config-only` to test configuration
- Compare with working template configurations

### Getting Help

#### Verbose Output
Use verbose flags for detailed information:

```bash
python train.py --config my_config.json --log-level DEBUG
python inference.py -m model -t tokenizer --verbose
```

#### Configuration Validation
Test configurations without training:

```bash
python train.py --config my_config.json --validate-config-only
```

#### Hardware Information
Check hardware compatibility:

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

### Performance Tips

#### Training Performance
1. Use appropriate batch size for your GPU
2. Enable mixed precision (`fp16` or `bf16`)
3. Use multiple data loader workers
4. Pin memory for faster GPU transfers
5. Use gradient checkpointing only when necessary

#### Inference Performance
1. Use `do_sample=False` for faster greedy decoding
2. Reduce `max_length` for shorter generations
3. Use batch processing for multiple prompts
4. Consider model quantization for deployment

#### Memory Optimization
1. Start with small batch sizes and increase gradually
2. Monitor GPU memory usage during training
3. Use gradient accumulation to simulate larger batches
4. Enable gradient checkpointing for memory-constrained setups

### Best Practices

1. **Always test with debug configuration first**
2. **Monitor training metrics and validation loss**
3. **Save configurations that work well**
4. **Use version control for reproducibility**
5. **Validate generation quality regularly**
6. **Keep backups of successful checkpoints**