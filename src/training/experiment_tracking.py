"""Experiment tracking integration for Weights & Biases and TensorBoard."""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
from pathlib import Path

from ..config.base import TrainingConfig, TrainingMetrics, ValidationMetrics
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Optional imports for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")


class WandBTracker:
    """Weights & Biases experiment tracking integration."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize Weights & Biases tracker.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.enabled = WANDB_AVAILABLE and config.logging_config.wandb_project is not None
        self.run = None
        
        if self.enabled:
            self._initialize_wandb()
        else:
            logger.info("Weights & Biases tracking disabled")
    
    def _initialize_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        try:
            # Create run configuration
            wandb_config = {
                "model": {
                    "checkpoint_path": self.config.model_config.checkpoint_path,
                    "vocab_size": self.config.model_config.vocab_size
                },
                "training": {
                    "learning_rate": self.config.training_params.learning_rate,
                    "batch_size": self.config.training_params.batch_size,
                    "gradient_accumulation_steps": self.config.training_params.gradient_accumulation_steps,
                    "max_steps": self.config.training_params.max_steps,
                    "warmup_steps": self.config.training_params.warmup_steps,
                    "mixed_precision": self.config.training_params.mixed_precision,
                    "gradient_checkpointing": self.config.training_params.gradient_checkpointing
                },
                "data": {
                    "tigrinya_dataset": self.config.data_config.tigrinya_dataset,
                    "max_length": self.config.data_config.max_length,
                    "debug_samples": self.config.data_config.debug_samples
                },
                "hardware": {
                    "num_gpus": self.config.hardware_config.num_gpus,
                    "dataloader_workers": self.config.hardware_config.dataloader_workers
                },
                "knowledge_preservation": {
                    "enabled": self.config.knowledge_preservation.enabled,
                    "english_weight": self.config.knowledge_preservation.english_weight,
                    "regularization_strength": self.config.knowledge_preservation.regularization_strength
                }
            }
            
            # Initialize run
            self.run = wandb.init(
                project=self.config.logging_config.wandb_project,
                config=wandb_config,
                name=f"tigrinya-tinyllama-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["tigrinya", "tinyllama", "continuous-pretraining"],
                notes="Continuous pretraining of TinyLlama for Tigrinya language"
            )
            
            logger.info(f"Weights & Biases initialized - Project: {self.config.logging_config.wandb_project}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weights & Biases: {e}")
            self.enabled = False
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics to Weights & Biases.
        
        Args:
            metrics: Training metrics to log
        """
        if not self.enabled or not self.run:
            return
        
        try:
            wandb_metrics = {
                "train/loss": metrics.loss,
                "train/learning_rate": metrics.learning_rate,
                "train/gpu_memory_used": metrics.gpu_memory_used,
                "train/tokens_per_second": metrics.tokens_per_second,
                "train/tigrinya_perplexity": metrics.tigrinya_perplexity,
                "train/gradient_norm": metrics.gradient_norm,
                "step": metrics.step
            }
            
            if metrics.english_perplexity is not None:
                wandb_metrics["train/english_perplexity"] = metrics.english_perplexity
            
            wandb.log(wandb_metrics, step=metrics.step)
            
        except Exception as e:
            logger.warning(f"Failed to log training metrics to W&B: {e}")
    
    def log_validation_metrics(self, metrics: ValidationMetrics) -> None:
        """Log validation metrics to Weights & Biases.
        
        Args:
            metrics: Validation metrics to log
        """
        if not self.enabled or not self.run:
            return
        
        try:
            wandb_metrics = {
                "val/tigrinya_loss": metrics.tigrinya_loss,
                "val/tigrinya_perplexity": metrics.tigrinya_perplexity,
                "step": metrics.step
            }
            
            if metrics.english_loss is not None:
                wandb_metrics["val/english_loss"] = metrics.english_loss
            if metrics.english_perplexity is not None:
                wandb_metrics["val/english_perplexity"] = metrics.english_perplexity
            
            wandb.log(wandb_metrics, step=metrics.step)
            
        except Exception as e:
            logger.warning(f"Failed to log validation metrics to W&B: {e}")
    
    def log_hardware_metrics(self, hardware_stats: Dict[str, Any]) -> None:
        """Log hardware metrics to Weights & Biases.
        
        Args:
            hardware_stats: Hardware statistics to log
        """
        if not self.enabled or not self.run:
            return
        
        try:
            gpu_stats = hardware_stats.get('gpu', {})
            cpu_stats = hardware_stats.get('cpu', {})
            
            wandb_metrics = {
                "hardware/gpu_memory_used_gb": gpu_stats.get('memory_reserved_gb', 0),
                "hardware/gpu_memory_percent": gpu_stats.get('memory_used_percent', 0),
                "hardware/gpu_utilization": gpu_stats.get('utilization_percent', 0),
                "hardware/cpu_utilization": cpu_stats.get('cpu_utilization_percent', 0),
                "hardware/system_memory_used_gb": cpu_stats.get('system_memory_used_gb', 0),
                "hardware/system_memory_percent": cpu_stats.get('system_memory_used_percent', 0)
            }
            
            wandb.log(wandb_metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log hardware metrics to W&B: {e}")
    
    def log_model_checkpoint(self, checkpoint_path: str, step: int, metrics: Dict[str, Any]) -> None:
        """Log model checkpoint as artifact to Weights & Biases.
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Training step
            metrics: Associated metrics
        """
        if not self.enabled or not self.run:
            return
        
        try:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-step-{step}",
                type="model",
                description=f"Model checkpoint at step {step}",
                metadata=metrics
            )
            
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            
            logger.info(f"Model checkpoint logged to W&B: step {step}")
            
        except Exception as e:
            logger.warning(f"Failed to log checkpoint to W&B: {e}")
    
    def finish(self) -> None:
        """Finish Weights & Biases run."""
        if self.enabled and self.run:
            try:
                wandb.finish()
                logger.info("Weights & Biases run finished")
            except Exception as e:
                logger.warning(f"Error finishing W&B run: {e}")


class TensorBoardTracker:
    """TensorBoard experiment tracking integration."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize TensorBoard tracker.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.enabled = TENSORBOARD_AVAILABLE
        self.writer = None
        
        if self.enabled:
            self._initialize_tensorboard()
        else:
            logger.info("TensorBoard tracking disabled")
    
    def _initialize_tensorboard(self) -> None:
        """Initialize TensorBoard SummaryWriter."""
        try:
            # Create log directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path(self.config.logging_config.tensorboard_dir) / f"tigrinya_tinyllama_{timestamp}"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=str(log_dir))
            
            # Log configuration as text
            config_text = self._format_config_for_tensorboard()
            self.writer.add_text("config", config_text, 0)
            
            logger.info(f"TensorBoard initialized - Log dir: {log_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard: {e}")
            self.enabled = False
    
    def _format_config_for_tensorboard(self) -> str:
        """Format configuration for TensorBoard display.
        
        Returns:
            Formatted configuration string
        """
        config_dict = {
            "Model": {
                "Checkpoint": self.config.model_config.checkpoint_path,
                "Vocab Size": self.config.model_config.vocab_size
            },
            "Training": {
                "Learning Rate": self.config.training_params.learning_rate,
                "Batch Size": self.config.training_params.batch_size,
                "Gradient Accumulation": self.config.training_params.gradient_accumulation_steps,
                "Max Steps": self.config.training_params.max_steps,
                "Mixed Precision": self.config.training_params.mixed_precision
            },
            "Data": {
                "Dataset": self.config.data_config.tigrinya_dataset,
                "Max Length": self.config.data_config.max_length,
                "Debug Samples": self.config.data_config.debug_samples
            },
            "Knowledge Preservation": {
                "Enabled": self.config.knowledge_preservation.enabled,
                "English Weight": self.config.knowledge_preservation.english_weight,
                "Regularization": self.config.knowledge_preservation.regularization_strength
            }
        }
        
        # Format as markdown table
        lines = ["| Parameter | Value |", "|-----------|-------|"]
        for section, params in config_dict.items():
            lines.append(f"| **{section}** | |")
            for key, value in params.items():
                lines.append(f"| {key} | {value} |")
        
        return "\n".join(lines)
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics to TensorBoard.
        
        Args:
            metrics: Training metrics to log
        """
        if not self.enabled or not self.writer:
            return
        
        try:
            # Training metrics
            self.writer.add_scalar("Training/Loss", metrics.loss, metrics.step)
            self.writer.add_scalar("Training/Learning_Rate", metrics.learning_rate, metrics.step)
            self.writer.add_scalar("Training/GPU_Memory_Used", metrics.gpu_memory_used, metrics.step)
            self.writer.add_scalar("Training/Tokens_Per_Second", metrics.tokens_per_second, metrics.step)
            self.writer.add_scalar("Training/Tigrinya_Perplexity", metrics.tigrinya_perplexity, metrics.step)
            self.writer.add_scalar("Training/Gradient_Norm", metrics.gradient_norm, metrics.step)
            
            if metrics.english_perplexity is not None:
                self.writer.add_scalar("Training/English_Perplexity", metrics.english_perplexity, metrics.step)
            
            # Flush to ensure immediate writing
            self.writer.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log training metrics to TensorBoard: {e}")
    
    def log_validation_metrics(self, metrics: ValidationMetrics) -> None:
        """Log validation metrics to TensorBoard.
        
        Args:
            metrics: Validation metrics to log
        """
        if not self.enabled or not self.writer:
            return
        
        try:
            # Validation metrics
            self.writer.add_scalar("Validation/Tigrinya_Loss", metrics.tigrinya_loss, metrics.step)
            self.writer.add_scalar("Validation/Tigrinya_Perplexity", metrics.tigrinya_perplexity, metrics.step)
            
            if metrics.english_loss is not None:
                self.writer.add_scalar("Validation/English_Loss", metrics.english_loss, metrics.step)
            if metrics.english_perplexity is not None:
                self.writer.add_scalar("Validation/English_Perplexity", metrics.english_perplexity, metrics.step)
            
            # Flush to ensure immediate writing
            self.writer.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log validation metrics to TensorBoard: {e}")
    
    def log_hardware_metrics(self, hardware_stats: Dict[str, Any]) -> None:
        """Log hardware metrics to TensorBoard.
        
        Args:
            hardware_stats: Hardware statistics to log
        """
        if not self.enabled or not self.writer:
            return
        
        try:
            gpu_stats = hardware_stats.get('gpu', {})
            cpu_stats = hardware_stats.get('cpu', {})
            
            # Use current timestamp as step for hardware metrics
            step = int(hardware_stats.get('unix_timestamp', 0))
            
            # GPU metrics
            self.writer.add_scalar("Hardware/GPU_Memory_Used_GB", gpu_stats.get('memory_reserved_gb', 0), step)
            self.writer.add_scalar("Hardware/GPU_Memory_Percent", gpu_stats.get('memory_used_percent', 0), step)
            self.writer.add_scalar("Hardware/GPU_Utilization", gpu_stats.get('utilization_percent', 0), step)
            
            # CPU metrics
            self.writer.add_scalar("Hardware/CPU_Utilization", cpu_stats.get('cpu_utilization_percent', 0), step)
            self.writer.add_scalar("Hardware/System_Memory_Used_GB", cpu_stats.get('system_memory_used_gb', 0), step)
            self.writer.add_scalar("Hardware/System_Memory_Percent", cpu_stats.get('system_memory_used_percent', 0), step)
            
            # Flush to ensure immediate writing
            self.writer.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log hardware metrics to TensorBoard: {e}")
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        """Log model graph to TensorBoard.
        
        Args:
            model: Model to visualize
            input_sample: Sample input tensor
        """
        if not self.enabled or not self.writer:
            return
        
        try:
            self.writer.add_graph(model, input_sample)
            self.writer.flush()
            logger.info("Model graph logged to TensorBoard")
            
        except Exception as e:
            logger.warning(f"Failed to log model graph to TensorBoard: {e}")
    
    def log_text_samples(self, step: int, samples: Dict[str, str]) -> None:
        """Log text generation samples to TensorBoard.
        
        Args:
            step: Training step
            samples: Dictionary of sample texts (language -> text)
        """
        if not self.enabled or not self.writer:
            return
        
        try:
            for language, text in samples.items():
                self.writer.add_text(f"Samples/{language}", text, step)
            
            self.writer.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log text samples to TensorBoard: {e}")
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.enabled and self.writer:
            try:
                self.writer.close()
                logger.info("TensorBoard writer closed")
            except Exception as e:
                logger.warning(f"Error closing TensorBoard writer: {e}")


class ExperimentTracker:
    """Unified experiment tracking interface."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize experiment tracker.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Initialize trackers
        self.wandb_tracker = WandBTracker(config)
        self.tensorboard_tracker = TensorBoardTracker(config)
        
        # Track which trackers are enabled
        self.enabled_trackers = []
        if self.wandb_tracker.enabled:
            self.enabled_trackers.append("wandb")
        if self.tensorboard_tracker.enabled:
            self.enabled_trackers.append("tensorboard")
        
        logger.info(f"Experiment tracking initialized - Enabled: {', '.join(self.enabled_trackers) or 'None'}")
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics to all enabled trackers.
        
        Args:
            metrics: Training metrics to log
        """
        self.wandb_tracker.log_training_metrics(metrics)
        self.tensorboard_tracker.log_training_metrics(metrics)
    
    def log_validation_metrics(self, metrics: ValidationMetrics) -> None:
        """Log validation metrics to all enabled trackers.
        
        Args:
            metrics: Validation metrics to log
        """
        self.wandb_tracker.log_validation_metrics(metrics)
        self.tensorboard_tracker.log_validation_metrics(metrics)
    
    def log_hardware_metrics(self, hardware_stats: Dict[str, Any]) -> None:
        """Log hardware metrics to all enabled trackers.
        
        Args:
            hardware_stats: Hardware statistics to log
        """
        self.wandb_tracker.log_hardware_metrics(hardware_stats)
        self.tensorboard_tracker.log_hardware_metrics(hardware_stats)
    
    def log_model_checkpoint(self, checkpoint_path: str, step: int, metrics: Dict[str, Any]) -> None:
        """Log model checkpoint to enabled trackers.
        
        Args:
            checkpoint_path: Path to checkpoint file
            step: Training step
            metrics: Associated metrics
        """
        self.wandb_tracker.log_model_checkpoint(checkpoint_path, step, metrics)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor) -> None:
        """Log model graph to TensorBoard.
        
        Args:
            model: Model to visualize
            input_sample: Sample input tensor
        """
        self.tensorboard_tracker.log_model_graph(model, input_sample)
    
    def log_text_samples(self, step: int, samples: Dict[str, str]) -> None:
        """Log text generation samples.
        
        Args:
            step: Training step
            samples: Dictionary of sample texts
        """
        self.tensorboard_tracker.log_text_samples(step, samples)
    
    def create_training_summary(self, final_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive training summary for experiment tracking.
        
        Args:
            final_metrics: Final training metrics
            
        Returns:
            Training summary dictionary
        """
        summary = {
            "experiment_tracking": {
                "enabled_trackers": self.enabled_trackers,
                "wandb_project": self.config.logging_config.wandb_project,
                "tensorboard_dir": self.config.logging_config.tensorboard_dir
            },
            "final_metrics": final_metrics,
            "configuration": {
                "model_config": {
                    "checkpoint_path": self.config.model_config.checkpoint_path,
                    "vocab_size": self.config.model_config.vocab_size
                },
                "training_params": {
                    "learning_rate": self.config.training_params.learning_rate,
                    "batch_size": self.config.training_params.batch_size,
                    "max_steps": self.config.training_params.max_steps,
                    "mixed_precision": self.config.training_params.mixed_precision
                },
                "knowledge_preservation": {
                    "enabled": self.config.knowledge_preservation.enabled,
                    "english_weight": self.config.knowledge_preservation.english_weight
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def finalize(self) -> None:
        """Finalize all experiment trackers."""
        logger.info("Finalizing experiment tracking...")
        
        self.wandb_tracker.finish()
        self.tensorboard_tracker.close()
        
        logger.info("Experiment tracking finalized")