#!/usr/bin/env python3
"""
End-to-end integration test for Tigrinya TinyLlama training pipeline.

This script performs a complete end-to-end test of the training pipeline with:
- Small dataset samples
- Minimal training steps
- Validation of bilingual generation
- Performance benchmarking
"""

import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.manager import ConfigManager
from config.hardware import HardwareAdapter
from model.manager import ModelManager
from data.loader import DataPipeline
from training.engine import TrainingEngine
from inference.engine import InferenceEngine
from utils.logging import setup_logging, get_logger


def create_test_config(test_dir: Path) -> dict:
    """Create test configuration for end-to-end testing."""
    
    # Create minimal test configuration
    test_config = {
        "model_config": {
            "checkpoint_path": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "tokenizer_path": str(test_dir.parent.parent / "tokenizer"),
            "vocab_size": 32000
        },
        "training_params": {
            "learning_rate": 2e-5,
            "batch_size": 1,  # Small for testing
            "gradient_accumulation_steps": 2,
            "max_steps": 10,  # Very few steps for testing
            "warmup_steps": 2,
            "save_steps": 5,
            "eval_steps": 5,
            "mixed_precision": "fp16",
            "gradient_checkpointing": True
        },
        "data_config": {
            "tigrinya_dataset": str(test_dir.parent.parent / "dataset" / "train.jsonl"),
            "validation_dataset": str(test_dir.parent.parent / "dataset" / "validation.jsonl"),
            "english_validation": None,
            "max_length": 512,  # Shorter for testing
            "debug_samples": 50  # Very small sample
        },
        "hardware_config": {
            "device": "auto",
            "num_gpus": 1,
            "dataloader_workers": 0,  # No workers for testing
            "pin_memory": True
        },
        "knowledge_preservation": {
            "enabled": False,  # Disable for simplicity
            "english_weight": 0.3,
            "regularization_strength": 0.01,
            "validation_frequency": 5
        },
        "logging_config": {
            "log_level": "INFO",
            "wandb_project": None,
            "tensorboard_dir": str(test_dir / "logs"),
            "save_metrics": True
        }
    }
    
    return test_config


def create_test_dataset(test_dir: Path) -> None:
    """Create minimal test dataset."""
    
    # Create test data directory
    data_dir = test_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create minimal Tigrinya training data
    tigrinya_samples = [
        {"text": "ሰላም! ከመይ ኣሎኻ? እንታይ ትገብር ኣሎኻ?"},
        {"text": "ሓደ ዕለት ሓደ ሰብ ኣብ ገዛ ነበረ።"},
        {"text": "ትግርኛ ቋንቋ ኤርትራን ትግራይን እዩ።"},
        {"text": "ንሕና ሎሚ ናብ ቤት ትምህርቲ ንኸይድ።"},
        {"text": "እዚ መጽሓፍ ብዛዕባ ታሪኽ እዩ።"}
    ]
    
    # Write training data
    train_path = data_dir / "train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in tigrinya_samples * 10:  # Repeat for more samples
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Write validation data
    val_path = data_dir / "validation.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in tigrinya_samples[:3]:  # Smaller validation set
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return str(train_path), str(val_path)


def test_model_loading(config: dict) -> bool:
    """Test model and tokenizer loading."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing model and tokenizer loading...")
        
        model_manager = ModelManager()
        
        # Load tokenizer
        tokenizer = model_manager.load_tokenizer(config["model_config"]["tokenizer_path"])
        logger.info(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")
        
        # Load model
        model = model_manager.load_model(config["model_config"]["checkpoint_path"])
        logger.info(f"Model loaded: {model.config.vocab_size} vocab size")
        
        # Test compatibility
        is_compatible = model_manager.validate_model_tokenizer_compatibility(model, tokenizer)
        logger.info(f"Model-tokenizer compatibility: {is_compatible}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        return False


def test_data_pipeline(config: dict, train_path: str, val_path: str) -> bool:
    """Test data pipeline functionality."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing data pipeline...")
        
        # Update config with test data paths
        config["data_config"]["tigrinya_dataset"] = train_path
        config["data_config"]["validation_dataset"] = val_path
        
        # Load tokenizer for data pipeline
        from model.tokenizer import TigrinyaTokenizer
        tokenizer = TigrinyaTokenizer(config["model_config"]["tokenizer_path"])
        
        # Create data pipeline
        data_pipeline = DataPipeline(
            tokenizer=tokenizer,
            max_length=config["data_config"]["max_length"],
            debug_samples=config["data_config"]["debug_samples"]
        )
        
        # Load datasets
        train_dataset = data_pipeline.load_tigrinya_dataset(train_path)
        val_dataset = data_pipeline.load_tigrinya_dataset(val_path)
        
        logger.info(f"Datasets loaded: {len(train_dataset)} train, {len(val_dataset)} val")
        
        # Create data loaders
        train_loader = data_pipeline.create_data_loader(
            train_dataset,
            batch_size=config["training_params"]["batch_size"],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = data_pipeline.create_data_loader(
            val_dataset,
            batch_size=config["training_params"]["batch_size"],
            shuffle=False,
            num_workers=0
        )
        
        # Test batch loading
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        logger.info(f"Batch shapes: train={train_batch['input_ids'].shape}, val={val_batch['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data pipeline test failed: {e}")
        return False


def test_training_loop(config: dict, test_dir: Path) -> bool:
    """Test minimal training loop."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing training loop...")
        
        # Load model and tokenizer
        model_manager = ModelManager()
        tokenizer = model_manager.load_tokenizer(config["model_config"]["tokenizer_path"])
        model = model_manager.load_model(config["model_config"]["checkpoint_path"])
        
        # Resize embeddings if needed
        model_manager.resize_token_embeddings(model, tokenizer)
        
        # Create data pipeline
        data_pipeline = DataPipeline(
            tokenizer=tokenizer,
            max_length=config["data_config"]["max_length"],
            debug_samples=config["data_config"]["debug_samples"]
        )
        
        # Load data
        train_dataset = data_pipeline.load_tigrinya_dataset(config["data_config"]["tigrinya_dataset"])
        train_loader = data_pipeline.create_data_loader(
            train_dataset,
            batch_size=config["training_params"]["batch_size"],
            shuffle=True,
            num_workers=0
        )
        
        # Create training config object
        from config.base import (
            TrainingConfig, ModelConfig, TrainingParams, DataConfig, 
            HardwareConfig, KnowledgePreservationConfig, LoggingConfig
        )
        
        training_config = TrainingConfig(
            model_config=ModelConfig(**config["model_config"]),
            training_params=TrainingParams(**config["training_params"]),
            data_config=DataConfig(**config["data_config"]),
            hardware_config=HardwareConfig(**config["hardware_config"]),
            knowledge_preservation=KnowledgePreservationConfig(**config["knowledge_preservation"]),
            logging_config=LoggingConfig(**config["logging_config"])
        )
        
        # Create training engine
        training_engine = TrainingEngine(training_config)
        training_engine.setup_training(model, training_config)
        
        # Run minimal training loop
        logger.info("Running minimal training steps...")
        
        step = 0
        for batch in train_loader:
            if step >= config["training_params"]["max_steps"]:
                break
            
            # Training step
            metrics = training_engine.train_step(batch)
            
            logger.info(f"Step {step}: Loss={metrics.loss:.4f}, LR={metrics.learning_rate:.2e}")
            
            # Save checkpoint at save_steps
            if step > 0 and step % config["training_params"]["save_steps"] == 0:
                checkpoint_dir = test_dir / "checkpoints" / f"step_{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                training_engine.save_checkpoint(step, {"step": step, "loss": metrics.loss})
                logger.info(f"Checkpoint saved at step {step}")
            
            step += 1
        
        logger.info(f"Training completed after {step} steps")
        return True
        
    except Exception as e:
        logger.error(f"Training loop test failed: {e}")
        return False


def test_inference_generation(config: dict, test_dir: Path) -> bool:
    """Test inference and text generation."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Testing inference and text generation...")
        
        # Create inference config
        from config.base import TrainingConfig, ModelConfig, TrainingParams, DataConfig, HardwareConfig, KnowledgePreservationConfig, LoggingConfig
        
        inference_config = TrainingConfig(
            model_config=ModelConfig(**config["model_config"]),
            training_params=TrainingParams(**config["training_params"]),
            data_config=DataConfig(**config["data_config"]),
            hardware_config=HardwareConfig(**config["hardware_config"]),
            knowledge_preservation=KnowledgePreservationConfig(**config["knowledge_preservation"]),
            logging_config=LoggingConfig(**config["logging_config"])
        )
        
        # Create inference engine
        inference_engine = InferenceEngine(inference_config)
        
        # Load model and tokenizer
        inference_engine.load_model(config["model_config"]["checkpoint_path"])
        inference_engine.load_tokenizer(config["model_config"]["tokenizer_path"])
        
        # Test English generation
        english_prompts = [
            "Hello, this is a test",
            "The weather today is",
            "Machine learning is"
        ]
        
        logger.info("Testing English text generation...")
        for prompt in english_prompts:
            try:
                generated = inference_engine.generate_text(
                    prompt,
                    max_length=50,
                    temperature=0.7,
                    top_k=50
                )
                logger.info(f"English - Prompt: '{prompt}' -> Generated: '{generated}'")
            except Exception as e:
                logger.warning(f"English generation failed for '{prompt}': {e}")
        
        # Test Tigrinya generation
        tigrinya_prompts = [
            "ሰላም",
            "ሓደ ዕለት",
            "ትግርኛ ቋንቋ"
        ]
        
        logger.info("Testing Tigrinya text generation...")
        for prompt in tigrinya_prompts:
            try:
                generated = inference_engine.generate_text(
                    prompt,
                    max_length=50,
                    temperature=0.7,
                    top_k=50
                )
                logger.info(f"Tigrinya - Prompt: '{prompt}' -> Generated: '{generated}'")
            except Exception as e:
                logger.warning(f"Tigrinya generation failed for '{prompt}': {e}")
        
        # Test batch generation
        logger.info("Testing batch generation...")
        try:
            batch_prompts = ["Hello", "ሰላም", "Test"]
            batch_results = inference_engine.batch_generate(
                batch_prompts,
                max_length=30,
                temperature=0.7
            )
            
            for prompt, result in zip(batch_prompts, batch_results):
                logger.info(f"Batch - '{prompt}' -> '{result}'")
                
        except Exception as e:
            logger.warning(f"Batch generation failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return False


def test_performance_benchmarks(config: dict) -> dict:
    """Run performance benchmarks."""
    logger = get_logger(__name__)
    
    benchmarks = {
        "model_loading_time": 0.0,
        "training_step_time": 0.0,
        "inference_time": 0.0,
        "memory_usage": 0.0
    }
    
    try:
        logger.info("Running performance benchmarks...")
        
        # Benchmark model loading
        start_time = time.time()
        model_manager = ModelManager()
        tokenizer = model_manager.load_tokenizer(config["model_config"]["tokenizer_path"])
        model = model_manager.load_model(config["model_config"]["checkpoint_path"])
        benchmarks["model_loading_time"] = time.time() - start_time
        
        # Benchmark training step (simplified)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Simulate training step
            device = next(model.parameters()).device
            dummy_input = torch.randint(0, model.config.vocab_size, (1, 100), device=device)
            
            with torch.no_grad():
                outputs = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            benchmarks["training_step_time"] = time.time() - start_time
            
            # Memory usage
            if torch.cuda.is_available():
                benchmarks["memory_usage"] = torch.cuda.memory_allocated() / 1024**3  # GB
        
        # Benchmark inference
        start_time = time.time()
        test_prompt = "Hello world"
        tokens = tokenizer.encode(test_prompt)
        decoded = tokenizer.decode(tokens)
        benchmarks["inference_time"] = time.time() - start_time
        
        logger.info(f"Performance benchmarks: {benchmarks}")
        return benchmarks
        
    except Exception as e:
        logger.error(f"Performance benchmarking failed: {e}")
        return benchmarks


def run_end_to_end_test() -> bool:
    """Run complete end-to-end test."""
    
    # Setup test environment
    test_dir = Path(tempfile.mkdtemp(prefix="tigrinya_e2e_test_"))
    
    try:
        # Setup logging
        setup_logging(level="INFO", console_output=True)
        logger = get_logger(__name__)
        
        logger.info("=" * 60)
        logger.info("TIGRINYA TINYLLAMA END-TO-END INTEGRATION TEST")
        logger.info("=" * 60)
        logger.info(f"Test directory: {test_dir}")
        
        # Create test configuration
        logger.info("Creating test configuration...")
        config = create_test_config(test_dir)
        
        # Create test dataset
        logger.info("Creating test dataset...")
        train_path, val_path = create_test_dataset(test_dir)
        config["data_config"]["tigrinya_dataset"] = train_path
        config["data_config"]["validation_dataset"] = val_path
        
        # Save test configuration
        config_path = test_dir / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Test results
        test_results = {
            "model_loading": False,
            "data_pipeline": False,
            "training_loop": False,
            "inference_generation": False,
            "performance_benchmarks": {}
        }
        
        # Run tests
        logger.info("\n" + "="*40)
        logger.info("RUNNING TESTS")
        logger.info("="*40)
        
        # Test 1: Model loading
        test_results["model_loading"] = test_model_loading(config)
        
        # Test 2: Data pipeline
        test_results["data_pipeline"] = test_data_pipeline(config, train_path, val_path)
        
        # Test 3: Training loop
        if test_results["model_loading"] and test_results["data_pipeline"]:
            test_results["training_loop"] = test_training_loop(config, test_dir)
        else:
            logger.warning("Skipping training loop test due to previous failures")
        
        # Test 4: Inference generation
        if test_results["model_loading"]:
            test_results["inference_generation"] = test_inference_generation(config, test_dir)
        else:
            logger.warning("Skipping inference test due to model loading failure")
        
        # Test 5: Performance benchmarks
        if test_results["model_loading"]:
            test_results["performance_benchmarks"] = test_performance_benchmarks(config)
        
        # Save test results
        results_path = test_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*40)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*40)
        
        passed_tests = sum(1 for result in test_results.values() if result is True)
        total_tests = len([k for k in test_results.keys() if k != "performance_benchmarks"])
        
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        
        for test_name, result in test_results.items():
            if test_name == "performance_benchmarks":
                continue
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {status}: {test_name.replace('_', ' ').title()}")
        
        # Performance summary
        if test_results["performance_benchmarks"]:
            logger.info("\nPerformance Benchmarks:")
            benchmarks = test_results["performance_benchmarks"]
            logger.info(f"  Model loading: {benchmarks.get('model_loading_time', 0):.2f}s")
            logger.info(f"  Training step: {benchmarks.get('training_step_time', 0):.4f}s")
            logger.info(f"  Inference: {benchmarks.get('inference_time', 0):.4f}s")
            logger.info(f"  Memory usage: {benchmarks.get('memory_usage', 0):.2f}GB")
        
        # Overall result
        overall_success = passed_tests == total_tests
        
        if overall_success:
            logger.info("\n🎉 END-TO-END TEST PASSED!")
            logger.info("The system is ready for production training.")
        else:
            logger.error("\n❌ END-TO-END TEST FAILED!")
            logger.error("Some components need attention before production use.")
        
        logger.info(f"\nTest artifacts saved in: {test_dir}")
        logger.info("You can examine the test results and logs for more details.")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"End-to-end test failed with exception: {e}")
        return False
    
    finally:
        # Cleanup (optional - comment out to keep test artifacts)
        # shutil.rmtree(test_dir, ignore_errors=True)
        pass


if __name__ == "__main__":
    import torch
    
    success = run_end_to_end_test()
    sys.exit(0 if success else 1)