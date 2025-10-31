#!/usr/bin/env python3
"""
Main training script for Tigrinya TinyLlama continuous pretraining.

This script orchestrates the complete training pipeline with JSON configuration support,
hardware detection, and comprehensive error handling.
"""

import os
import sys
import time
import argparse
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.manager import ConfigManager
from src.config.hardware import HardwareAdapter
from src.model.manager import ModelManager
from src.data.dataset import TigrinyaDataset
from src.data.loader import DataLoaderFactory
from src.training.engine import TrainingEngine
from src.training.monitoring import TrainingMonitor
from src.training.distributed import DistributedTrainingManager, auto_detect_distributed_config
from src.utils.logging import setup_logging, get_logger
from src.utils.error_handling import ErrorHandler, GracefulShutdownHandler, OutOfMemoryError, TrainingError


def apply_recovery_plan(config: 'TrainingConfig', recovery_plan_path: str) -> 'TrainingConfig':
    """Apply recovery plan adjustments to configuration.
    
    Args:
        config: Original training configuration
        recovery_plan_path: Path to recovery plan JSON file
        
    Returns:
        Modified configuration with recovery adjustments
    """
    logger = get_logger(__name__)
    
    try:
        with open(recovery_plan_path, 'r') as f:
            recovery_plan = json.load(f)
        
        adjustments = recovery_plan.get("config_adjustments", {})
        
        if adjustments:
            logger.info(f"Applying recovery plan adjustments: {adjustments}")
            
            # Apply batch size adjustments
            if "batch_size" in adjustments:
                config.training_params.batch_size = adjustments["batch_size"]
                logger.info(f"Adjusted batch size to: {config.training_params.batch_size}")
            
            # Apply gradient accumulation adjustments
            if "gradient_accumulation_steps" in adjustments:
                config.training_params.gradient_accumulation_steps = adjustments["gradient_accumulation_steps"]
                logger.info(f"Adjusted gradient accumulation steps to: {config.training_params.gradient_accumulation_steps}")
            
            # Apply gradient checkpointing
            if "gradient_checkpointing" in adjustments:
                config.training_params.gradient_checkpointing = adjustments["gradient_checkpointing"]
                logger.info(f"Set gradient checkpointing to: {config.training_params.gradient_checkpointing}")
            
            # Apply mixed precision adjustments
            if "mixed_precision" in adjustments:
                config.training_params.mixed_precision = adjustments["mixed_precision"]
                logger.info(f"Adjusted mixed precision to: {config.training_params.mixed_precision}")
            
            # Apply data handling adjustments
            if "skip_corrupted_samples" in adjustments:
                # This would need to be added to the config structure
                logger.info("Recovery plan requests enabling corrupted sample skipping")
            
            if "max_data_errors" in adjustments:
                # This would need to be added to the config structure
                logger.info(f"Recovery plan sets max data errors to: {adjustments['max_data_errors']}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to apply recovery plan: {e}")
        logger.info("Continuing with original configuration")
        return config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TinyLlama model for Tigrinya language",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--override-config",
        type=str,
        help="Path to JSON configuration file with parameter overrides"
    )
    
    # Training control arguments
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint directory to resume training from"
    )
    
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from the latest checkpoint if available"
    )
    
    parser.add_argument(
        "--recovery-plan",
        type=str,
        help="Path to recovery plan JSON file to apply configuration adjustments"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save checkpoints and logs"
    )
    
    # Hardware and debugging arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced dataset and logging"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Only validate configuration and exit"
    )
    
    # Distributed training arguments
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="Local rank for distributed training (set by torch.distributed.launch)"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (default: output_dir/training.log)"
    )
    
    return parser.parse_args()


def setup_output_directory(output_dir: str) -> Path:
    """Setup output directory for checkpoints and logs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "metrics").mkdir(exist_ok=True)
    
    return output_path


def load_and_validate_config(args: argparse.Namespace) -> 'TrainingConfig':
    """Load and validate training configuration."""
    logger = get_logger(__name__)
    
    try:
        config_manager = ConfigManager()
        
        # Load base configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = config_manager.load_config(args.config)
        
        # Apply override configuration if provided
        if args.override_config:
            logger.info(f"Applying configuration overrides from: {args.override_config}")
            config = config_manager.merge_configs(args.config, args.override_config)
        
        # Apply recovery plan if specified
        if args.recovery_plan:
            logger.info(f"Applying recovery plan from: {args.recovery_plan}")
            config = apply_recovery_plan(config, args.recovery_plan)
        
        # Apply command line overrides
        if args.debug:
            logger.info("Debug mode enabled - applying debug configuration")
            config.data_config.debug_samples = 1000
            config.training_params.max_steps = min(config.training_params.max_steps, 500)
            config.training_params.warmup_steps = min(config.training_params.warmup_steps, 50)
            config.training_params.save_steps = min(config.training_params.save_steps, 100)
            config.training_params.eval_steps = min(config.training_params.eval_steps, 50)
            config.logging_config.log_level = "DEBUG"
        
        # Override device if specified
        if args.device != "auto":
            config.hardware_config.device = args.device
        
        # Override log level if specified
        if args.log_level:
            config.logging_config.log_level = args.log_level
        
        # Validate final configuration
        logger.info("Validating configuration...")
        config_manager.validate_config(config)
        logger.info("Configuration validation passed")
        
        return config
        
    except Exception as e:
        logger.error(f"Configuration loading/validation failed: {str(e)}")
        raise


def setup_hardware_and_config(config: 'TrainingConfig', distributed_manager: Optional[DistributedTrainingManager] = None) -> 'TrainingConfig':
    """Setup hardware detection and optimize configuration."""
    logger = get_logger(__name__)
    
    try:
        # Initialize hardware adapter
        hardware_adapter = HardwareAdapter()
        
        # Detect hardware
        logger.info("Detecting hardware configuration...")
        hardware_info = hardware_adapter.detect_hardware()
        
        # Log hardware info (only from main process in distributed training)
        if not distributed_manager or not distributed_manager.is_distributed or distributed_manager.is_main_process():
            logger.info(f"Detected hardware: {hardware_info.gpu_name} "
                       f"({hardware_info.gpu_memory_gb:.1f}GB VRAM, "
                       f"{hardware_info.gpu_count} GPU(s))")
        
        # Get optimal configuration for detected hardware
        optimal_config = hardware_adapter.get_optimal_config(hardware_info)
        
        # Apply hardware-specific optimizations
        logger.info("Applying hardware-specific optimizations...")
        config = hardware_adapter.adjust_config_for_hardware(config, hardware_info)
        
        # Validate hardware compatibility
        is_compatible, compatibility_message = hardware_adapter.validate_hardware_compatibility(config)
        if not is_compatible:
            logger.warning(f"Hardware compatibility issues: {compatibility_message}")
        
        # Estimate memory usage
        memory_estimate = hardware_adapter.estimate_memory_usage(config)
        if not distributed_manager or not distributed_manager.is_distributed or distributed_manager.is_main_process():
            logger.info(f"Estimated memory usage: {memory_estimate.model_memory_gb:.1f}GB model, "
                       f"{memory_estimate.optimizer_memory_gb:.1f}GB optimizer, "
                       f"{memory_estimate.activation_memory_gb:.1f}GB activations")
        
        if not memory_estimate.fits_in_memory:
            logger.warning(f"Estimated memory usage ({memory_estimate.total_memory_gb:.1f}GB) "
                          f"exceeds available GPU memory ({hardware_info.gpu_memory_gb:.1f}GB)")
        
        return config
        
    except Exception as e:
        logger.error(f"Hardware setup failed: {str(e)}")
        raise


def setup_model_and_tokenizer(config: 'TrainingConfig') -> tuple:
    """Setup model and tokenizer."""
    logger = get_logger(__name__)
    
    try:
        # Initialize model manager
        model_manager = ModelManager(device=config.hardware_config.device)
        
        # Load tokenizer
        logger.info("Loading Tigrinya tokenizer...")
        tokenizer = model_manager.load_tokenizer(config.model_config.tokenizer_path)
        
        # Load model
        logger.info("Loading TinyLlama model...")
        model = model_manager.load_model(
            config.model_config.checkpoint_path,
            config_overrides={"vocab_size": config.model_config.vocab_size}
        )
        
        # Resize token embeddings if needed
        model_manager.resize_token_embeddings(model, tokenizer)
        
        # Validate compatibility
        logger.info("Validating model-tokenizer compatibility...")
        is_compatible = model_manager.validate_model_tokenizer_compatibility(model, tokenizer)
        if not is_compatible:
            raise RuntimeError("Model-tokenizer compatibility validation failed")
        
        logger.info("Model and tokenizer setup completed successfully")
        return model, tokenizer, model_manager
        
    except Exception as e:
        logger.error(f"Model/tokenizer setup failed: {str(e)}")
        raise


def setup_data_loaders(config: 'TrainingConfig', tokenizer, distributed_manager: Optional[DistributedTrainingManager] = None) -> tuple:
    """Setup training and validation data loaders."""
    logger = get_logger(__name__)
    
    try:
        # Initialize data loader factory
        loader_factory = DataLoaderFactory(tokenizer, config)
        
        # Create training data loader with distributed support
        logger.info("Setting up training data loader...")
        if distributed_manager and distributed_manager.is_distributed:
            # Use distributed data loader
            from src.data.dataset import TigrinyaDataset
            
            # Create dataset
            train_dataset = TigrinyaDataset(
                data_path=config.data_config.tigrinya_dataset,
                tokenizer=tokenizer,
                max_length=config.data_config.max_length,
                debug_samples=config.data_config.debug_samples
            )
            
            # Create distributed data loader
            train_loader = distributed_manager.create_distributed_dataloader(
                train_dataset,
                batch_size=config.training_params.batch_size,
                shuffle=True,
                num_workers=config.hardware_config.dataloader_workers,
                pin_memory=config.hardware_config.pin_memory,
                drop_last=True
            )
        else:
            # Use regular data loader
            train_loader = loader_factory.create_training_loader(
                config.data_config.tigrinya_dataset,
                batch_size=config.training_params.batch_size,
                debug_samples=config.data_config.debug_samples if config.data_config.debug_samples else None
            )
        
        # Create validation data loader
        logger.info("Setting up validation data loader...")
        val_loader = loader_factory.create_validation_loader(
            config.data_config.validation_dataset,
            batch_size=config.training_params.batch_size
        )
        
        # Create English validation loader if specified
        english_val_loader = None
        if config.data_config.english_validation:
            logger.info("Setting up English validation data loader...")
            english_val_loader = loader_factory.create_validation_loader(
                config.data_config.english_validation,
                batch_size=config.training_params.batch_size
            )
        
        # Log data loader info (only from main process in distributed training)
        if not distributed_manager or not distributed_manager.is_distributed or distributed_manager.is_main_process():
            logger.info(f"Data loaders created: {len(train_loader)} training batches, "
                       f"{len(val_loader)} validation batches")
            
            if distributed_manager and distributed_manager.is_distributed:
                effective_batch_size = distributed_manager.calculate_effective_batch_size(
                    config.training_params.batch_size,
                    config.training_params.gradient_accumulation_steps
                )
                logger.info(f"Effective batch size (distributed): {effective_batch_size}")
        
        return train_loader, val_loader, english_val_loader
        
    except Exception as e:
        logger.error(f"Data loader setup failed: {str(e)}")
        raise


def setup_training_engine(config: 'TrainingConfig', model, original_model=None) -> 'TrainingEngine':
    """Setup training engine with all components."""
    logger = get_logger(__name__)
    
    try:
        # Initialize training engine
        training_engine = TrainingEngine(config, original_model=original_model)
        
        # Setup training components
        logger.info("Setting up training engine...")
        training_engine.setup_training(model, config)
        
        # Validate training state
        if not validate_training_state(training_engine, model):
            logger.warning("Training state validation failed, but continuing...")
        
        logger.info("Training engine setup completed")
        return training_engine
        
    except Exception as e:
        logger.error(f"Training engine setup failed: {str(e)}")
        raise


def validate_training_state(training_engine: 'TrainingEngine', model) -> bool:
    """Validate training state for consistency and recovery capability.
    
    Args:
        training_engine: Training engine to validate
        model: Model to validate
        
    Returns:
        True if validation passes
    """
    logger = get_logger(__name__)
    
    try:
        logger.info("Validating training state...")
        
        # Validate model state
        if not validate_model_state(model):
            logger.error("Model state validation failed")
            return False
        
        # Validate optimizer state
        if training_engine.optimizer and not validate_optimizer_state(training_engine.optimizer):
            logger.error("Optimizer state validation failed")
            return False
        
        # Validate training engine state
        if training_engine.state_manager:
            if not training_engine.state_manager.validate_training_state(
                model, training_engine.optimizer, training_engine.current_step
            ):
                logger.error("Training engine state validation failed")
                return False
        
        logger.info("Training state validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Training state validation error: {e}")
        return False


def validate_model_state(model) -> bool:
    """Validate model state for training readiness.
    
    Args:
        model: Model to validate
        
    Returns:
        True if model state is valid
    """
    logger = get_logger(__name__)
    
    try:
        # Check for NaN or infinite parameters
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"NaN values found in parameter: {name}")
                return False
            
            if torch.isinf(param).any():
                logger.error(f"Infinite values found in parameter: {name}")
                return False
        
        # Test forward pass
        device = next(model.parameters()).device
        test_input = torch.randint(0, model.config.vocab_size, (1, 10), device=device)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        if outputs.logits is None:
            logger.error("Model forward pass failed")
            return False
        
        if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
            logger.error("Model outputs contain NaN or infinite values")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Model state validation failed: {e}")
        return False


def validate_optimizer_state(optimizer) -> bool:
    """Validate optimizer state.
    
    Args:
        optimizer: Optimizer to validate
        
    Returns:
        True if optimizer state is valid
    """
    logger = get_logger(__name__)
    
    try:
        # Check optimizer state for NaN/inf values
        for group in optimizer.param_groups:
            for param in group['params']:
                if param in optimizer.state:
                    state = optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            if torch.isnan(value).any() or torch.isinf(value).any():
                                logger.error(f"Invalid values in optimizer state: {key}")
                                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Optimizer state validation failed: {e}")
        return False


def create_training_recovery_plan(config: 'TrainingConfig', 
                                output_dir: Path,
                                error_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a recovery plan for training failures.
    
    Args:
        config: Training configuration
        output_dir: Output directory
        error_info: Information about the error that occurred
        
    Returns:
        Recovery plan dictionary
    """
    logger = get_logger(__name__)
    
    recovery_plan = {
        "timestamp": time.time(),
        "config_adjustments": {},
        "recovery_actions": [],
        "fallback_options": []
    }
    
    if error_info:
        error_type = error_info.get("error_type", "unknown")
        
        if "OutOfMemory" in error_type or "CUDA" in error_type:
            # Memory-related recovery
            recovery_plan["config_adjustments"] = {
                "batch_size": max(1, config.training_params.batch_size // 2),
                "gradient_accumulation_steps": config.training_params.gradient_accumulation_steps * 2,
                "gradient_checkpointing": True,
                "mixed_precision": "fp16" if config.training_params.mixed_precision != "fp16" else "bf16"
            }
            recovery_plan["recovery_actions"].append("reduce_memory_usage")
            
        elif "Data" in error_type:
            # Data-related recovery
            recovery_plan["config_adjustments"] = {
                "skip_corrupted_samples": True,
                "max_data_errors": 200
            }
            recovery_plan["recovery_actions"].append("enable_data_error_handling")
            
        elif "Checkpoint" in error_type:
            # Checkpoint-related recovery
            recovery_plan["recovery_actions"].append("find_alternative_checkpoint")
            recovery_plan["fallback_options"].append("start_from_pretrained_model")
    
    # General recovery actions
    recovery_plan["recovery_actions"].extend([
        "validate_training_state",
        "cleanup_gpu_memory",
        "reduce_learning_rate"
    ])
    
    recovery_plan["fallback_options"].extend([
        "restart_with_smaller_model",
        "use_cpu_training",
        "restart_from_scratch"
    ])
    
    # Save recovery plan
    recovery_plan_path = output_dir / "recovery_plan.json"
    with open(recovery_plan_path, 'w') as f:
        json.dump(recovery_plan, f, indent=2, default=str)
    
    logger.info(f"Recovery plan created: {recovery_plan_path}")
    return recovery_plan


def run_training_loop(training_engine: 'TrainingEngine', 
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     english_val_loader: Optional[DataLoader],
                     config: 'TrainingConfig',
                     output_dir: Path) -> None:
    """Run the main training loop with comprehensive error handling."""
    logger = get_logger(__name__)
    
    # Initialize error handling
    error_handler = ErrorHandler({
        'max_retries': 3,
        'retry_delay': 2.0,
        'max_data_errors': 50
    })
    
    shutdown_handler = GracefulShutdownHandler()
    
    try:
        logger.info("Starting training loop with error handling...")
        
        # Set validation loaders for monitoring
        training_engine.set_validation_loaders(val_loader, english_val_loader)
        
        # Setup knowledge preservation if enabled
        if config.knowledge_preservation.enabled and english_val_loader:
            logger.info("Setting up knowledge preservation...")
            training_engine.setup_knowledge_preservation(english_val_loader)
        
        # Training loop with error recovery
        step = 0
        best_val_loss = float('inf')
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        for epoch in range(1000):  # Large number, will break based on max_steps
            logger.info(f"Starting epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(train_loader):
                # Check for shutdown request
                if shutdown_handler.is_shutdown_requested():
                    logger.info("Graceful shutdown requested")
                    break
                
                try:
                    # Get English batch for mixed training if available
                    english_batch = None
                    if english_val_loader and config.knowledge_preservation.enabled:
                        try:
                            english_batch = next(iter(english_val_loader))
                        except StopIteration:
                            pass
                    
                    # Training step with error handling
                    metrics = training_engine.train_step(batch, english_batch)
                    consecutive_failures = 0  # Reset failure counter on success
                    
                    # Log progress
                    if step % 10 == 0:
                        logger.info(f"Step {step}: Loss={metrics.loss:.4f}, "
                                   f"LR={metrics.learning_rate:.2e}, "
                                   f"GPU={metrics.gpu_memory_used:.1f}GB")
                    
                    # Validation with error handling
                    if step % config.training_params.eval_steps == 0 and step > 0:
                        try:
                            avg_val_loss = run_validation_with_error_handling(
                                training_engine, val_loader, step, error_handler
                            )
                            
                            # Save best model
                            if avg_val_loss < best_val_loss:
                                best_val_loss = avg_val_loss
                                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                                
                        except Exception as val_error:
                            logger.error(f"Validation failed at step {step}: {val_error}")
                            # Continue training even if validation fails
                    
                    # Save checkpoint with error handling
                    if step % config.training_params.save_steps == 0 and step > 0:
                        try:
                            logger.info(f"Saving checkpoint at step {step}")
                            checkpoint_metrics = {
                                "step": step,
                                "train_loss": metrics.loss,
                                "val_loss": best_val_loss,
                                "learning_rate": metrics.learning_rate,
                                "recovery_info": training_engine.get_recovery_info()
                            }
                            training_engine.save_checkpoint_with_recovery(step, checkpoint_metrics)
                            
                        except Exception as checkpoint_error:
                            logger.error(f"Checkpoint saving failed at step {step}: {checkpoint_error}")
                            # Try basic checkpoint save as fallback
                            try:
                                training_engine.save_checkpoint(step, {"step": step, "emergency": True})
                            except Exception as fallback_error:
                                logger.error(f"Emergency checkpoint save also failed: {fallback_error}")
                    
                    step += 1
                    
                    # Check if training is complete
                    if step >= config.training_params.max_steps:
                        logger.info(f"Training completed after {step} steps")
                        return
                
                except OutOfMemoryError as oom_error:
                    logger.error(f"Out of memory error at step {step}: {oom_error}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive OOM failures ({consecutive_failures})")
                        raise TrainingError("Training failed due to persistent memory issues")
                    
                    # Continue with next batch after OOM handling
                    continue
                
                except TrainingError as train_error:
                    logger.error(f"Training error at step {step}: {train_error}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive training failures ({consecutive_failures})")
                        raise
                    
                    # Try to recover and continue
                    logger.info("Attempting to recover from training error...")
                    time.sleep(2.0)  # Brief pause before retry
                    continue
                
                except Exception as unexpected_error:
                    logger.error(f"Unexpected error at step {step}: {unexpected_error}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise TrainingError(f"Training failed due to repeated unexpected errors: {unexpected_error}")
                    
                    # Create emergency checkpoint
                    try:
                        error_info = {
                            "error_type": type(unexpected_error).__name__,
                            "error_message": str(unexpected_error),
                            "step": step,
                            "consecutive_failures": consecutive_failures
                        }
                        training_engine.error_handler.create_error_recovery_checkpoint(
                            training_engine.model, training_engine.optimizer, step, error_info, str(output_dir)
                        )
                    except Exception as checkpoint_error:
                        logger.error(f"Failed to create error recovery checkpoint: {checkpoint_error}")
                    
                    continue
            
            # Break if shutdown requested
            if shutdown_handler.is_shutdown_requested():
                break
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        training_engine.handle_training_interruption()
        
    except Exception as e:
        logger.error(f"Training loop failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Attempt to save emergency state
        try:
            training_engine.handle_training_interruption()
        except Exception as cleanup_error:
            logger.error(f"Emergency cleanup failed: {cleanup_error}")
        
        raise
    
    finally:
        # Ensure cleanup is performed
        try:
            shutdown_handler.cleanup()
        except Exception as cleanup_error:
            logger.error(f"Final cleanup failed: {cleanup_error}")


def run_validation_with_error_handling(training_engine: 'TrainingEngine',
                                     val_loader: DataLoader,
                                     step: int,
                                     error_handler: ErrorHandler) -> float:
    """Run validation with error handling and recovery.
    
    Args:
        training_engine: Training engine instance
        val_loader: Validation data loader
        step: Current training step
        error_handler: Error handler instance
        
    Returns:
        Average validation loss
        
    Raises:
        Exception: If validation fails completely
    """
    logger = get_logger(__name__)
    
    logger.info(f"Running validation at step {step}")
    
    val_metrics = []
    failed_batches = 0
    max_failed_batches = len(val_loader) // 4  # Allow up to 25% of batches to fail
    
    for val_batch_idx, val_batch in enumerate(val_loader):
        try:
            val_metric = training_engine.validation_step(val_batch)
            val_metrics.append(val_metric)
            
        except Exception as val_error:
            failed_batches += 1
            logger.warning(f"Validation batch {val_batch_idx} failed: {val_error}")
            
            # Handle validation error
            if not error_handler.handle_data_loading_error(
                val_error, val_batch_idx, "validation", skip_corrupted=True
            ):
                logger.error("Too many validation errors, stopping validation")
                break
            
            if failed_batches > max_failed_batches:
                logger.error(f"Too many failed validation batches ({failed_batches})")
                break
    
    if not val_metrics:
        raise Exception("All validation batches failed")
    
    avg_val_loss = sum(m.tigrinya_loss for m in val_metrics) / len(val_metrics)
    avg_val_perplexity = sum(m.tigrinya_perplexity for m in val_metrics) / len(val_metrics)
    
    logger.info(f"Validation - Loss: {avg_val_loss:.4f}, "
               f"Perplexity: {avg_val_perplexity:.2f}, "
               f"Valid batches: {len(val_metrics)}/{len(val_loader)}")
    
    return avg_val_loss


def main():
    """Main training function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup output directory
        output_dir = setup_output_directory(args.output_dir)
        
        # Setup logging
        log_file = args.log_file or "training.log"
        setup_logging(
            log_level=args.log_level,
            log_file=log_file,
            log_dir=str(output_dir / "logs")
        )
        
        logger = get_logger(__name__)
        
        # Initialize distributed training if requested or auto-detected
        distributed_manager = None
        if args.distributed or args.local_rank is not None:
            distributed_manager = DistributedTrainingManager()
            distributed_success = distributed_manager.setup_distributed_training()
            
            if distributed_success:
                logger.info(f"Distributed training initialized: "
                           f"rank {distributed_manager.rank}/{distributed_manager.world_size}")
            else:
                logger.warning("Distributed training initialization failed, using single GPU")
                distributed_manager = None
        else:
            # Auto-detect distributed training capability
            distributed_config = auto_detect_distributed_config()
            if distributed_config["can_use_distributed"]:
                logger.info(f"Multi-GPU setup detected ({distributed_config['num_gpus']} GPUs) "
                           f"but distributed training not explicitly enabled")
                logger.info("Use --distributed flag or launch with torch.distributed.launch for multi-GPU training")
        
        # Log startup info (only from main process in distributed training)
        if not distributed_manager or not distributed_manager.is_distributed or distributed_manager.is_main_process():
            logger.info("=" * 80)
            logger.info("Starting Tigrinya TinyLlama Training")
            logger.info("=" * 80)
            logger.info(f"Arguments: {vars(args)}")
            
            if distributed_manager and distributed_manager.is_distributed:
                dist_info = distributed_manager.get_distributed_info()
                logger.info(f"Distributed training info: {dist_info}")
        
        # Load and validate configuration
        config = load_and_validate_config(args)
        
        # Exit early if only validating config
        if args.validate_config_only:
            logger.info("Configuration validation completed successfully")
            return 0
        
        # Setup hardware and optimize configuration
        config = setup_hardware_and_config(config, distributed_manager)
        
        # Save final configuration (only from main process)
        if not distributed_manager or not distributed_manager.is_distributed or distributed_manager.is_main_process():
            config_path = output_dir / "final_config.json"
            config_manager = ConfigManager()
            config_manager.save_config(config, str(config_path))
            logger.info(f"Final configuration saved to: {config_path}")
        
        # Synchronize all processes before proceeding
        if distributed_manager and distributed_manager.is_distributed:
            distributed_manager.barrier()
        
        # Setup model and tokenizer
        model, tokenizer, model_manager = setup_model_and_tokenizer(config)
        
        # Setup data loaders with distributed support
        train_loader, val_loader, english_val_loader = setup_data_loaders(config, tokenizer, distributed_manager)
        
        # Setup training engine
        training_engine = setup_training_engine(config, model)
        
        # Resume from checkpoint with enhanced recovery
        resume_successful = False
        
        # Handle explicit resume request
        if args.resume_from:
            logger.info(f"Resuming training from: {args.resume_from}")
            try:
                metadata = training_engine.load_checkpoint_with_recovery(args.resume_from)
                resume_successful = True
                logger.info(f"Successfully resumed from step {metadata.get('step', 0)}")
            except Exception as resume_error:
                logger.error(f"Failed to resume from specified checkpoint: {resume_error}")
                if not args.auto_resume:
                    logger.info("Attempting automatic checkpoint discovery...")
        
        # Handle auto-resume or fallback from failed explicit resume
        if (args.auto_resume or (args.resume_from and not resume_successful)) and not resume_successful:
            try:
                latest_checkpoint = training_engine.state_manager.find_resumable_checkpoint() if training_engine.state_manager else None
                if latest_checkpoint:
                    logger.info(f"Found resumable checkpoint: {latest_checkpoint}")
                    metadata = training_engine.load_checkpoint_with_recovery(latest_checkpoint)
                    resume_successful = True
                    logger.info(f"Automatically resumed from step {metadata.get('step', 0)}")
                else:
                    logger.info("No resumable checkpoints found, starting fresh training")
            except Exception as auto_resume_error:
                logger.warning(f"Automatic checkpoint recovery failed: {auto_resume_error}")
                logger.info("Starting fresh training")
        
        # Log resume status
        if resume_successful:
            logger.info("Training will continue from resumed checkpoint")
        else:
            logger.info("Training will start from the beginning")
        
        # Run training
        run_training_loop(
            training_engine, 
            train_loader, 
            val_loader, 
            english_val_loader,
            config, 
            output_dir
        )
        
        # Save final checkpoint (only from main process)
        if not distributed_manager or not distributed_manager.is_distributed or distributed_manager.is_main_process():
            logger.info("Saving final checkpoint...")
            final_metrics = {"step": config.training_params.max_steps, "final": True}
            training_engine.save_checkpoint(config.training_params.max_steps, final_metrics)
        
        # Cleanup distributed training
        if distributed_manager and distributed_manager.is_distributed:
            distributed_manager.cleanup_distributed()
        
        logger.info("Training completed successfully!")
        return 0
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Training failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create recovery plan
            try:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                if 'config' in locals() and 'output_dir' in locals():
                    recovery_plan = create_training_recovery_plan(config, output_dir, error_info)
                    logger.info("Recovery plan created for future training attempts")
                
                # Attempt emergency state save if training engine exists
                if 'training_engine' in locals():
                    try:
                        training_engine.handle_training_interruption()
                        logger.info("Emergency state saved successfully")
                    except Exception as save_error:
                        logger.error(f"Emergency state save failed: {save_error}")
                
            except Exception as recovery_error:
                logger.error(f"Recovery plan creation failed: {recovery_error}")
        else:
            print(f"Training failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())