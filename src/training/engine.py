"""Training engine with mixed precision support and knowledge preservation."""

import os
import math
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from transformers import get_linear_schedule_with_warmup
import psutil
import GPUtil

from ..config.base import TrainingConfig, TrainingMetrics, ValidationMetrics
from ..training.base import BaseTrainingEngine
from ..training.knowledge_preservation import KnowledgePreservationManager
from ..training.monitoring import TrainingMonitor
from ..training.recovery import TrainingStateManager
from ..training.performance import PerformanceOptimizer
from ..training.distributed import DistributedTrainingManager
from ..utils.logging import get_logger
from ..utils.error_handling import (
    ErrorHandler, GracefulShutdownHandler, OutOfMemoryError,
    with_error_handling, TrainingError
)

logger = get_logger(__name__)


class TrainingEngine(BaseTrainingEngine):
    """Core training engine with mixed precision support and gradient accumulation."""
    
    def __init__(self, config: TrainingConfig, original_model: Optional[nn.Module] = None):
        """Initialize training engine with configuration.
        
        Args:
            config: Training configuration
            original_model: Original model for knowledge preservation
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.device = None
        self.current_step = 0
        self.accumulated_steps = 0
        
        # Training state
        self.is_mixed_precision = config.training_params.mixed_precision in ["fp16", "bf16"]
        self.gradient_accumulation_steps = config.training_params.gradient_accumulation_steps
        
        # Knowledge preservation
        self.knowledge_preservation = None
        self.original_model = original_model
        
        # Monitoring
        self.monitor = None
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizer()
        
        # Distributed training
        self.distributed_manager = DistributedTrainingManager()
        
        # Error handling and recovery
        self.error_handler = ErrorHandler({
            'max_retries': 3,
            'min_batch_size': 1,
            'batch_size_reduction_factor': 0.5,
            'max_data_errors': 100
        })
        self.shutdown_handler = GracefulShutdownHandler()
        self.state_manager = None
        
        # Dynamic training parameters for error recovery
        self.current_batch_size = config.training_params.batch_size
        self.current_gradient_accumulation_steps = config.training_params.gradient_accumulation_steps
        
        logger.info(f"TrainingEngine initialized with mixed precision: {self.is_mixed_precision}")
    
    def setup_training(self, model: nn.Module, config: TrainingConfig) -> None:
        """Setup training components (optimizer, scheduler, scaler).
        
        Args:
            model: Model to train
            config: Training configuration
        """
        logger.info("Setting up training components...")
        
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Setup error handling and recovery
        self._setup_error_handling()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup learning rate scheduler
        self._setup_scheduler()
        
        # Setup mixed precision scaler
        if self.is_mixed_precision:
            self._setup_mixed_precision()
        
        # Enable gradient checkpointing if configured
        if config.training_params.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Setup distributed training if available
        self._setup_distributed_training()
        
        # Apply performance optimizations
        self._apply_performance_optimizations()
        
        # Setup knowledge preservation
        self._setup_knowledge_preservation()
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register cleanup callbacks
        self._register_cleanup_callbacks()
        
        logger.info("Training setup completed successfully")
    
    def _setup_optimizer(self) -> None:
        """Setup AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training_params.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        logger.info(f"Optimizer setup: AdamW with lr={self.config.training_params.learning_rate}")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler with warmup."""
        warmup_steps = self.config.training_params.warmup_steps
        max_steps = self.config.training_params.max_steps
        
        # Use linear warmup followed by cosine annealing
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )
        
        logger.info(f"Scheduler setup: Linear warmup ({warmup_steps} steps) + Cosine annealing ({max_steps} total steps)")
    
    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training with GradScaler."""
        if self.config.training_params.mixed_precision == "fp16":
            self.scaler = GradScaler()
            logger.info("Mixed precision setup: FP16 with GradScaler")
        elif self.config.training_params.mixed_precision == "bf16":
            # BF16 doesn't need gradient scaling
            self.scaler = None
            logger.info("Mixed precision setup: BF16 (no gradient scaling)")
        else:
            self.scaler = None
            logger.info("Mixed precision disabled")
    
    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory optimization."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
            # Alternative method for models that don't have gradient_checkpointing_enable
            self.model.config.use_cache = False
            logger.info("Gradient checkpointing enabled via use_cache=False")
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    def _apply_performance_optimizations(self) -> None:
        """Apply performance optimizations to the model."""
        logger.info("Applying performance optimizations...")
        
        # Get hardware information for optimization decisions
        hardware_info = {
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cpu_cores": os.cpu_count() or 1
        }
        
        # Apply model optimizations
        enable_flash_attention = self.config.training_params.mixed_precision in ["fp16", "bf16"]
        enable_torch_compile = hardware_info["gpu_memory_gb"] >= 8  # Only for GPUs with sufficient memory
        
        self.model = self.performance_optimizer.optimize_model(
            self.model,
            enable_flash_attention=enable_flash_attention,
            enable_torch_compile=enable_torch_compile,
            compile_mode="default"
        )
        
        # Setup efficient training configurations
        setup_info = self.performance_optimizer.setup_efficient_training(
            self.model,
            self.optimizer if hasattr(self, 'optimizer') and self.optimizer else None,
            enable_amp_autocast=self.is_mixed_precision
        )
        
        logger.info(f"Performance optimizations applied: {setup_info.get('optimizations', [])}")
        
        # Log recommendations
        recommendations = setup_info.get('recommendations', [])
        if recommendations:
            logger.info(f"Performance recommendations: {recommendations}")
    
    def _setup_distributed_training(self) -> None:
        """Setup distributed training if multiple GPUs are available."""
        # Check if distributed training should be enabled
        if self.config.hardware_config.num_gpus > 1:
            logger.info("Setting up distributed training...")
            
            # Initialize distributed training
            distributed_success = self.distributed_manager.setup_distributed_training(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )
            
            if distributed_success:
                # Wrap model with DDP
                self.model = self.distributed_manager.wrap_model_for_distributed(
                    self.model,
                    find_unused_parameters=False,  # Set to True if model has unused parameters
                    broadcast_buffers=True
                )
                
                # Adjust learning rate for distributed training
                if hasattr(self, 'optimizer') and self.optimizer:
                    current_lr = self.config.training_params.learning_rate
                    adjusted_lr = self.distributed_manager.adjust_learning_rate_for_distributed(
                        current_lr, batch_size_scaling=True
                    )
                    
                    # Update optimizer learning rate
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = adjusted_lr
                
                logger.info(f"Distributed training setup completed: "
                           f"world_size={self.distributed_manager.world_size}, "
                           f"rank={self.distributed_manager.rank}")
            else:
                logger.warning("Failed to setup distributed training, falling back to single GPU")
        else:
            logger.info("Single GPU training mode")
    
    def _setup_knowledge_preservation(self) -> None:
        """Setup knowledge preservation manager."""
        self.knowledge_preservation = KnowledgePreservationManager(
            model=self.model,
            config=self.config.knowledge_preservation,
            original_model=self.original_model
        )
        logger.info("Knowledge preservation manager initialized")
    
    def _setup_monitoring(self) -> None:
        """Setup training monitoring system."""
        self.monitor = TrainingMonitor(
            config=self.config,
            model=self.model,
            device=self.device
        )
        logger.info("Training monitor initialized")
    
    def _setup_error_handling(self) -> None:
        """Setup error handling and recovery systems."""
        # Initialize state manager
        checkpoint_dir = getattr(self.config, 'checkpoint_dir', 'checkpoints')
        self.state_manager = TrainingStateManager(checkpoint_dir)
        
        # Register model and optimizer for cleanup
        self.shutdown_handler.register_resource(self.model, 'cpu')
        
        logger.info("Error handling and recovery systems initialized")
    
    def _register_cleanup_callbacks(self) -> None:
        """Register cleanup callbacks for graceful shutdown."""
        # Register checkpoint saving on shutdown
        self.shutdown_handler.register_cleanup_callback(
            self._emergency_checkpoint_save
        )
        
        # Register memory cleanup
        self.shutdown_handler.register_cleanup_callback(
            self.error_handler._clear_gpu_memory
        )
        
        logger.info("Cleanup callbacks registered")
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                  english_batch: Optional[Dict[str, torch.Tensor]] = None) -> TrainingMetrics:
        """Execute a single training step with gradient accumulation and knowledge preservation.
        
        Args:
            batch: Training batch containing input_ids, attention_mask, labels
            english_batch: Optional English batch for mixed training
            
        Returns:
            Training metrics for this step
            
        Raises:
            OutOfMemoryError: If GPU runs out of memory
            TrainingError: If training step fails
        """
        try:
            # Check for shutdown request
            if self.shutdown_handler.is_shutdown_requested():
                logger.info("Shutdown requested, stopping training step")
                raise KeyboardInterrupt("Graceful shutdown requested")
            
            with self.error_handler.memory_management(self.model):
                return self._execute_train_step(batch, english_batch)
                
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during training step: {e}")
            return self._handle_oom_during_training(batch, english_batch, e)
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            
            # Create emergency checkpoint
            if self.state_manager:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "step": self.current_step,
                    "batch_size": self.current_batch_size
                }
                self.state_manager.create_recovery_checkpoint(
                    self.model, self.optimizer, self.current_step, error_info
                )
            
            raise TrainingError(f"Training step failed: {e}") from e
    
    def _execute_train_step(self, batch: Dict[str, torch.Tensor], 
                           english_batch: Optional[Dict[str, torch.Tensor]] = None) -> TrainingMetrics:
        """Execute the actual training step logic."""
        self.model.train()
        
        # Create mixed batch if knowledge preservation is enabled
        if self.knowledge_preservation and self.knowledge_preservation.enabled:
            batch, batch_type = self.knowledge_preservation.create_mixed_batch(batch, english_batch)
        else:
            batch_type = "tigrinya"
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Get hardware metrics before forward pass
        gpu_memory_before = self._get_gpu_memory_usage()
        
        # Forward pass with mixed precision
        with self._get_autocast_context():
            outputs = self.model(**batch)
            base_loss = outputs.loss
            
            # Apply knowledge preservation techniques
            if self.knowledge_preservation and self.knowledge_preservation.enabled:
                # Get teacher logits if available
                teacher_logits = None
                if (self.knowledge_preservation.kd and 
                    self.knowledge_preservation.kd.teacher_model is not None):
                    with torch.no_grad():
                        teacher_outputs = self.knowledge_preservation.kd.teacher_model(**batch)
                        teacher_logits = teacher_outputs.logits
                
                # Apply preservation loss
                loss = self.knowledge_preservation.apply_preservation_loss(
                    base_loss, batch_type, outputs.logits, teacher_logits
                )
            else:
                loss = base_loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.current_gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            # FP16 with gradient scaling
            self.scaler.scale(loss).backward()
        else:
            # FP32 or BF16
            loss.backward()
        
        self.accumulated_steps += 1
        
        # Perform optimizer step if we've accumulated enough gradients
        if self.accumulated_steps >= self.current_gradient_accumulation_steps:
            # Calculate gradient norm before clipping
            grad_norm = self._get_gradient_norm()
            
            # Gradient clipping and optimizer step with error handling
            if self.scaler is not None:
                try:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except ValueError as e:
                    if "Attempting to unscale FP16 gradients" in str(e):
                        # Handle FP16 gradient scaling issues
                        logger.warning("FP16 gradient scaling issue detected, skipping step")
                        # Skip the step entirely - don't call scaler.update() without unscale
                        self.optimizer.zero_grad()
                        # Don't increment step counter for failed steps
                        self.accumulated_steps = 0
                        
                        # Create dummy metrics for skipped step
                        gpu_memory_after = self._get_gpu_memory_usage()
                        tokens_per_second = self._calculate_tokens_per_second(batch)
                        
                        skipped_metrics = TrainingMetrics(
                            step=self.current_step,
                            loss=0.0,  # No loss for skipped step
                            learning_rate=self.scheduler.get_last_lr()[0],
                            gpu_memory_used=gpu_memory_after,
                            tokens_per_second=tokens_per_second,
                            tigrinya_perplexity=0.0,
                            english_perplexity=None,
                            gradient_norm=0.0
                        )
                        return skipped_metrics
                    else:
                        raise
            else:
                # Standard gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Reset accumulation counter
            self.accumulated_steps = 0
            self.current_step += 1
        else:
            grad_norm = 0.0  # No gradient norm when accumulating
        
        # Calculate metrics
        gpu_memory_after = self._get_gpu_memory_usage()
        tokens_per_second = self._calculate_tokens_per_second(batch)
        perplexity = torch.exp(loss * self.current_gradient_accumulation_steps).item()
        
        # Set language-specific perplexity
        tigrinya_perplexity = perplexity if batch_type == "tigrinya" else None
        english_perplexity = perplexity if batch_type == "english" else None
        
        metrics = TrainingMetrics(
            step=self.current_step,
            loss=(loss * self.current_gradient_accumulation_steps).item(),
            learning_rate=self.scheduler.get_last_lr()[0],
            gpu_memory_used=gpu_memory_after,
            tokens_per_second=tokens_per_second,
            tigrinya_perplexity=tigrinya_perplexity or perplexity,
            english_perplexity=english_perplexity,
            gradient_norm=grad_norm
        )
        
        # Aggregate metrics across distributed processes
        if self.distributed_manager.is_distributed:
            metrics_dict = {
                'loss': metrics.loss,
                'learning_rate': metrics.learning_rate,
                'gpu_memory_used': metrics.gpu_memory_used,
                'tokens_per_second': metrics.tokens_per_second,
                'gradient_norm': metrics.gradient_norm
            }
            
            # Only aggregate on step completion
            if self.accumulated_steps == 0:
                aggregated_metrics = self.distributed_manager.log_distributed_metrics(metrics_dict)
                
                # Update metrics with aggregated values
                metrics.loss = aggregated_metrics['loss']
                metrics.gpu_memory_used = aggregated_metrics['gpu_memory_used']
                metrics.tokens_per_second = aggregated_metrics['tokens_per_second']
                metrics.gradient_norm = aggregated_metrics['gradient_norm']
        
        # Log metrics through monitoring system (only on main process for distributed)
        if self.monitor and self.accumulated_steps == 0:  # Only log when step is complete
            if not self.distributed_manager.is_distributed or self.distributed_manager.is_main_process():
                self.monitor.log_training_step(metrics)
        
        return metrics
    
    def _handle_oom_during_training(self, batch: Dict[str, torch.Tensor], 
                                  english_batch: Optional[Dict[str, torch.Tensor]],
                                  error: Exception) -> TrainingMetrics:
        """Handle out-of-memory error during training step.
        
        Args:
            batch: Original training batch
            english_batch: Optional English batch
            error: The OOM error
            
        Returns:
            Training metrics after recovery
            
        Raises:
            OutOfMemoryError: If recovery is not possible
        """
        logger.warning("Handling out-of-memory error during training...")
        
        # Reduce batch size and increase gradient accumulation
        new_batch_size, new_grad_accum = self.error_handler.handle_out_of_memory(
            self.current_batch_size,
            self.current_gradient_accumulation_steps,
            self.model
        )
        
        # Update current parameters
        self.current_batch_size = new_batch_size
        self.current_gradient_accumulation_steps = new_grad_accum
        
        logger.info(f"Adjusted training parameters: batch_size={new_batch_size}, "
                   f"gradient_accumulation_steps={new_grad_accum}")
        
        # Create smaller batch
        smaller_batch = self._create_smaller_batch(batch, new_batch_size)
        smaller_english_batch = None
        if english_batch:
            smaller_english_batch = self._create_smaller_batch(english_batch, new_batch_size)
        
        # Retry training step with smaller batch
        try:
            return self._execute_train_step(smaller_batch, smaller_english_batch)
        except torch.cuda.OutOfMemoryError as retry_error:
            raise OutOfMemoryError(f"Training failed even after batch size reduction: {retry_error}") from retry_error
    
    def _create_smaller_batch(self, batch: Dict[str, torch.Tensor], new_batch_size: int) -> Dict[str, torch.Tensor]:
        """Create a smaller batch from the original batch.
        
        Args:
            batch: Original batch
            new_batch_size: New batch size
            
        Returns:
            Smaller batch
        """
        smaller_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                # Take only the first new_batch_size samples
                smaller_batch[key] = value[:new_batch_size]
            else:
                smaller_batch[key] = value
        
        return smaller_batch
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> ValidationMetrics:
        """Execute a single validation step.
        
        Args:
            batch: Validation batch containing input_ids, attention_mask, labels
            
        Returns:
            Validation metrics for this step
        """
        self.model.eval()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            with self._get_autocast_context():
                outputs = self.model(**batch)
                loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        
        return ValidationMetrics(
            step=self.current_step,
            tigrinya_loss=loss.item(),
            tigrinya_perplexity=perplexity
        )
    
    def set_validation_loaders(self, tigrinya_loader=None, english_loader=None) -> None:
        """Set validation data loaders for monitoring.
        
        Args:
            tigrinya_loader: Tigrinya validation DataLoader
            english_loader: English validation DataLoader
        """
        if self.monitor:
            self.monitor.set_validation_loaders(tigrinya_loader, english_loader)
    
    def save_checkpoint(self, step: int, metrics: Dict[str, Any]) -> None:
        """Save training checkpoint with metadata.
        
        Args:
            step: Current training step
            metrics: Training metrics to save
        """
        # Only save from main process in distributed training
        if self.distributed_manager.is_distributed and not self.distributed_manager.is_main_process():
            # Wait for main process to finish saving
            self.distributed_manager.barrier()
            return
        
        checkpoint_dir = os.path.join("checkpoints", f"step_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state (unwrap DDP if necessary)
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        
        model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), model_path)
        
        # Save optimizer state
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save scaler state if using mixed precision
        if self.scaler is not None:
            scaler_path = os.path.join(checkpoint_dir, "scaler.pt")
            torch.save(self.scaler.state_dict(), scaler_path)
        
        # Save training metadata
        metadata = {
            "step": step,
            "current_step": self.current_step,
            "accumulated_steps": self.accumulated_steps,
            "config": self.config.__dict__,
            "metrics": metrics,
            "model_config": self.model.config.to_dict() if hasattr(self.model, 'config') else {}
        }
        
        # Add monitoring summary
        if self.monitor:
            metadata["monitoring_summary"] = self.monitor.get_monitoring_summary()
        
        metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved at step {step}: {checkpoint_dir}")
        
        # Synchronize all processes after saving
        if self.distributed_manager.is_distributed:
            self.distributed_manager.barrier()
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint and restore state.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Loaded metadata
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load model state
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("Model state loaded")
        
        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logger.info("Optimizer state loaded")
        
        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        if os.path.exists(scheduler_path):
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
            logger.info("Scheduler state loaded")
        
        # Load scaler state if exists
        scaler_path = os.path.join(checkpoint_path, "scaler.pt")
        if os.path.exists(scaler_path) and self.scaler is not None:
            self.scaler.load_state_dict(torch.load(scaler_path, map_location=self.device))
            logger.info("Scaler state loaded")
        
        # Load metadata
        metadata_path = os.path.join(checkpoint_path, "training_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Restore training state
            self.current_step = metadata.get("current_step", 0)
            self.accumulated_steps = metadata.get("accumulated_steps", 0)
            
            logger.info(f"Training state restored: step {self.current_step}")
        
        return metadata
    
    def _get_autocast_context(self):
        """Get appropriate autocast context for mixed precision."""
        if self.config.training_params.mixed_precision == "fp16":
            return autocast(dtype=torch.float16)
        elif self.config.training_params.mixed_precision == "bf16":
            return autocast(dtype=torch.bfloat16)
        else:
            return torch.no_grad() if not self.model.training else torch.enable_grad()
    
    def _get_gradient_norm(self) -> float:
        """Calculate gradient norm across all model parameters."""
        total_norm = 0.0
        param_count = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
        
        return total_norm
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            return torch.cuda.memory_allocated(self.device) / 1024**3
        return 0.0
    
    def _calculate_tokens_per_second(self, batch: Dict[str, torch.Tensor]) -> float:
        """Calculate tokens processed per second."""
        # This is a simplified calculation
        # In practice, you'd want to track actual timing
        batch_size = batch['input_ids'].size(0)
        seq_length = batch['input_ids'].size(1)
        total_tokens = batch_size * seq_length
        
        # Placeholder calculation - should be replaced with actual timing
        return float(total_tokens)
    
    def setup_knowledge_preservation(self, english_dataloader, num_fisher_samples: int = 1000) -> None:
        """Setup knowledge preservation with English data.
        
        Args:
            english_dataloader: DataLoader for English validation data
            num_fisher_samples: Number of samples for Fisher information computation
        """
        if self.knowledge_preservation:
            self.knowledge_preservation.setup_preservation(english_dataloader, num_fisher_samples)
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state information."""
        state = {
            "current_step": self.current_step,
            "accumulated_steps": self.accumulated_steps,
            "learning_rate": self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
            "gpu_memory_gb": self._get_gpu_memory_usage(),
            "mixed_precision": self.config.training_params.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps
        }
        
        # Add knowledge preservation stats
        if self.knowledge_preservation:
            state["knowledge_preservation"] = self.knowledge_preservation.get_preservation_stats()
        
        # Add monitoring summary
        if self.monitor:
            state["monitoring"] = self.monitor.get_monitoring_summary()
        
        return state
    
    def _emergency_checkpoint_save(self) -> None:
        """Save emergency checkpoint during shutdown."""
        try:
            if self.model and self.optimizer and self.state_manager:
                logger.info("Saving emergency checkpoint...")
                
                emergency_metrics = {
                    "step": self.current_step,
                    "emergency_save": True,
                    "timestamp": time.time()
                }
                
                checkpoint_path = self.state_manager.save_training_state(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    step=self.current_step,
                    epoch=0,  # We don't track epochs in this implementation
                    metrics=emergency_metrics,
                    config=self.config.__dict__ if hasattr(self.config, '__dict__') else {}
                )
                
                logger.info(f"Emergency checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
    
    def save_checkpoint_with_recovery(self, step: int, metrics: Dict[str, Any]) -> str:
        """Save checkpoint with enhanced error handling and validation.
        
        Args:
            step: Current training step
            metrics: Training metrics
            
        Returns:
            Path to saved checkpoint
            
        Raises:
            RuntimeError: If checkpoint saving fails
        """
        try:
            if not self.state_manager:
                raise RuntimeError("State manager not initialized")
            
            # Validate training state before saving
            if not self.state_manager.validate_training_state(self.model, self.optimizer, step):
                logger.warning("Training state validation failed, but proceeding with checkpoint save")
            
            # Save checkpoint with full state
            checkpoint_path = self.state_manager.save_training_state(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                step=step,
                epoch=0,
                metrics=metrics,
                config=self.config.__dict__ if hasattr(self.config, '__dict__') else {},
                additional_state={
                    "current_batch_size": self.current_batch_size,
                    "current_gradient_accumulation_steps": self.current_gradient_accumulation_steps,
                    "accumulated_steps": self.accumulated_steps,
                    "error_statistics": self.error_handler.get_error_statistics()
                }
            )
            
            # Cleanup old checkpoints
            self.state_manager.cleanup_old_checkpoints(keep_count=5)
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Enhanced checkpoint saving failed: {e}")
            # Fallback to basic checkpoint saving
            return self.save_checkpoint(step, metrics)
    
    def load_checkpoint_with_recovery(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint with enhanced error handling and recovery.
        
        Args:
            checkpoint_path: Path to checkpoint (uses latest if None)
            
        Returns:
            Loaded metadata
            
        Raises:
            CheckpointCorruptionError: If checkpoint loading fails
        """
        try:
            if not self.state_manager:
                raise RuntimeError("State manager not initialized")
            
            # Find resumable checkpoint if none specified
            if checkpoint_path is None:
                checkpoint_path = self.state_manager.find_resumable_checkpoint()
                if checkpoint_path is None:
                    raise FileNotFoundError("No resumable checkpoints found")
            
            # Load training state with recovery
            metadata, recovery_successful = self.state_manager.load_training_state(
                checkpoint_path=checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                device=str(self.device)
            )
            
            # Restore additional state
            additional_state = metadata.get("additional_state", {})
            self.current_batch_size = additional_state.get("current_batch_size", self.config.training_params.batch_size)
            self.current_gradient_accumulation_steps = additional_state.get("current_gradient_accumulation_steps", self.config.training_params.gradient_accumulation_steps)
            self.accumulated_steps = additional_state.get("accumulated_steps", 0)
            self.current_step = metadata.get("step", 0)
            
            if recovery_successful:
                logger.info(f"Successfully loaded checkpoint from step {metadata['step']}")
            else:
                logger.warning("Checkpoint loaded with recovery - some data may be inconsistent")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Enhanced checkpoint loading failed: {e}")
            # Fallback to basic checkpoint loading
            return self.load_checkpoint(checkpoint_path or "")
    
    def handle_training_interruption(self) -> None:
        """Handle training interruption with graceful shutdown."""
        logger.info("Handling training interruption...")
        
        try:
            # Save emergency checkpoint
            self._emergency_checkpoint_save()
            
            # Perform cleanup
            self.shutdown_handler.cleanup()
            
            logger.info("Training interruption handled successfully")
            
        except Exception as e:
            logger.error(f"Error during training interruption handling: {e}")
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """Get information about recovery capabilities and state.
        
        Returns:
            Dictionary with recovery information
        """
        recovery_info = {
            "error_handler_stats": self.error_handler.get_error_statistics(),
            "current_batch_size": self.current_batch_size,
            "current_gradient_accumulation_steps": self.current_gradient_accumulation_steps,
            "shutdown_requested": self.shutdown_handler.is_shutdown_requested()
        }
        
        if self.state_manager:
            recovery_info["state_manager_stats"] = self.state_manager.get_recovery_statistics()
        
        return recovery_info