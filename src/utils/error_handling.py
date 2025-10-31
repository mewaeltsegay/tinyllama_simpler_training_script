"""Comprehensive error handling and recovery utilities for training pipeline."""

import os
import gc
import time
import logging
import traceback
from typing import Dict, Any, Optional, Callable, List, Tuple
from functools import wraps
from contextlib import contextmanager
import torch
import psutil
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass


class OutOfMemoryError(TrainingError):
    """Exception for out-of-memory errors during training."""
    pass


class CheckpointCorruptionError(TrainingError):
    """Exception for checkpoint corruption errors."""
    pass


class DataLoadingError(TrainingError):
    """Exception for data loading errors."""
    pass


class ModelStateError(TrainingError):
    """Exception for model state validation errors."""
    pass


class CudaMemoryError(TrainingError):
    """Exception for CUDA memory access errors."""
    pass


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize error handler.
        
        Args:
            config: Configuration for error handling behavior
        """
        self.config = config or {}
        self.error_counts = {}
        self.recovery_attempts = {}
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # Memory management settings
        self.min_batch_size = self.config.get('min_batch_size', 1)
        self.batch_size_reduction_factor = self.config.get('batch_size_reduction_factor', 0.5)
        self.memory_cleanup_threshold = self.config.get('memory_cleanup_threshold', 0.9)
        
        logger.info("ErrorHandler initialized")
    
    def handle_out_of_memory(self, 
                           current_batch_size: int,
                           gradient_accumulation_steps: int,
                           model: Optional[torch.nn.Module] = None) -> Tuple[int, int]:
        """Handle CUDA out of memory errors with automatic batch size reduction.
        
        Args:
            current_batch_size: Current batch size
            gradient_accumulation_steps: Current gradient accumulation steps
            model: Model instance for memory cleanup
            
        Returns:
            Tuple of (new_batch_size, new_gradient_accumulation_steps)
            
        Raises:
            OutOfMemoryError: If batch size cannot be reduced further
        """
        logger.warning("Handling out of memory error...")
        
        # Clear GPU cache
        self._clear_gpu_memory()
        
        # Perform garbage collection
        gc.collect()
        
        # Calculate new batch size
        new_batch_size = max(
            self.min_batch_size,
            int(current_batch_size * self.batch_size_reduction_factor)
        )
        
        if new_batch_size >= current_batch_size:
            # Cannot reduce batch size further
            raise OutOfMemoryError(
                f"Cannot reduce batch size below {current_batch_size}. "
                f"Consider using gradient checkpointing or a smaller model."
            )
        
        # Increase gradient accumulation to maintain effective batch size
        effective_batch_size = current_batch_size * gradient_accumulation_steps
        new_gradient_accumulation_steps = max(
            gradient_accumulation_steps,
            int(effective_batch_size / new_batch_size)
        )
        
        logger.info(f"Reduced batch size: {current_batch_size} -> {new_batch_size}")
        logger.info(f"Increased gradient accumulation: {gradient_accumulation_steps} -> {new_gradient_accumulation_steps}")
        
        # Log memory usage after cleanup
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory after cleanup: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        
        return new_batch_size, new_gradient_accumulation_steps
    
    def handle_checkpoint_corruption(self, 
                                   checkpoint_path: str,
                                   available_checkpoints: List[str]) -> Optional[str]:
        """Handle checkpoint corruption by finding a valid fallback.
        
        Args:
            checkpoint_path: Path to corrupted checkpoint
            available_checkpoints: List of available checkpoint paths
            
        Returns:
            Path to valid checkpoint or None if no valid checkpoint found
        """
        logger.error(f"Checkpoint corruption detected: {checkpoint_path}")
        
        # Sort checkpoints by modification time (newest first)
        valid_checkpoints = []
        for cp_path in available_checkpoints:
            if cp_path != checkpoint_path and os.path.exists(cp_path):
                try:
                    # Basic validation - check if required files exist
                    if self._validate_checkpoint_files(cp_path):
                        valid_checkpoints.append((cp_path, os.path.getmtime(cp_path)))
                except Exception as e:
                    logger.warning(f"Checkpoint validation failed for {cp_path}: {e}")
        
        if not valid_checkpoints:
            logger.error("No valid checkpoints found for fallback")
            return None
        
        # Sort by modification time (newest first)
        valid_checkpoints.sort(key=lambda x: x[1], reverse=True)
        fallback_checkpoint = valid_checkpoints[0][0]
        
        logger.info(f"Using fallback checkpoint: {fallback_checkpoint}")
        return fallback_checkpoint
    
    def handle_data_loading_error(self, 
                                error: Exception,
                                sample_index: int,
                                dataset_path: str,
                                skip_corrupted: bool = True) -> bool:
        """Handle data loading errors with sample skipping.
        
        Args:
            error: The data loading error
            sample_index: Index of the problematic sample
            dataset_path: Path to the dataset
            skip_corrupted: Whether to skip corrupted samples
            
        Returns:
            True if error was handled and training can continue, False otherwise
        """
        error_key = f"data_loading_{dataset_path}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        logger.warning(f"Data loading error at sample {sample_index} in {dataset_path}: {error}")
        
        if skip_corrupted:
            logger.info(f"Skipping corrupted sample {sample_index}")
            
            # Log error details for debugging
            self._log_data_error_details(error, sample_index, dataset_path)
            
            # Check if too many errors occurred
            max_data_errors = self.config.get('max_data_errors', 100)
            if self.error_counts[error_key] > max_data_errors:
                logger.error(f"Too many data loading errors ({self.error_counts[error_key]}) in {dataset_path}")
                return False
            
            return True
        else:
            logger.error(f"Data loading failed and skip_corrupted=False")
            return False
    
    def handle_model_state_error(self, 
                               model: torch.nn.Module,
                               error: Exception,
                               checkpoint_path: Optional[str] = None) -> bool:
        """Handle model state validation errors.
        
        Args:
            model: Model with state issues
            error: The validation error
            checkpoint_path: Path to checkpoint being loaded
            
        Returns:
            True if error was handled, False otherwise
        """
        logger.error(f"Model state validation error: {error}")
        
        try:
            # Try to reset model to a clean state
            if hasattr(model, 'reset_parameters'):
                logger.info("Attempting to reset model parameters")
                model.reset_parameters()
                return True
            
            # Try to reinitialize specific layers that might be corrupted
            if hasattr(model, 'init_weights'):
                logger.info("Attempting to reinitialize model weights")
                model.init_weights()
                return True
            
            logger.error("Cannot recover from model state error")
            return False
            
        except Exception as recovery_error:
            logger.error(f"Model state recovery failed: {recovery_error}")
            return False
    
    def handle_cuda_memory_error(self, 
                                model: torch.nn.Module,
                                error: Exception) -> bool:
        """Handle CUDA memory access errors.
        
        Args:
            model: Model that encountered the error
            error: The CUDA error
            
        Returns:
            True if error was handled, False otherwise
        """
        logger.error(f"CUDA memory access error detected: {error}")
        
        try:
            # Immediate CUDA cleanup
            if torch.cuda.is_available():
                logger.info("Performing emergency CUDA cleanup...")
                
                # Clear all CUDA caches
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Reset CUDA context if possible
                try:
                    torch.cuda.reset_peak_memory_stats()
                    logger.info("CUDA memory stats reset")
                except Exception as reset_error:
                    logger.warning(f"Could not reset CUDA memory stats: {reset_error}")
                
                # Move model to CPU temporarily
                try:
                    logger.info("Moving model to CPU for safety...")
                    model.cpu()
                    torch.cuda.empty_cache()
                    
                    # Move back to CUDA
                    logger.info("Moving model back to CUDA...")
                    model.cuda()
                    torch.cuda.synchronize()
                    
                    logger.info("CUDA memory error recovery successful")
                    return True
                    
                except Exception as move_error:
                    logger.error(f"Failed to move model during CUDA recovery: {move_error}")
                    return False
            
            return False
            
        except Exception as recovery_error:
            logger.error(f"CUDA memory error recovery failed: {recovery_error}")
            return False
    
    def with_retry(self, 
                  max_retries: Optional[int] = None,
                  delay: Optional[float] = None,
                  exceptions: Tuple = (Exception,),
                  backoff_factor: float = 2.0):
        """Decorator for automatic retry with exponential backoff.
        
        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries
            exceptions: Tuple of exceptions to catch and retry
            backoff_factor: Factor to multiply delay by after each retry
        """
        max_retries = max_retries or self.max_retries
        delay = delay or self.retry_delay
        
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay:.1f}s...")
                            time.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(f"All {max_retries + 1} attempts failed")
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @contextmanager
    def memory_management(self, 
                         model: Optional[torch.nn.Module] = None,
                         cleanup_threshold: Optional[float] = None):
        """Context manager for automatic memory management.
        
        Args:
            model: Model for memory monitoring
            cleanup_threshold: Memory usage threshold for cleanup (0.0-1.0)
        """
        cleanup_threshold = cleanup_threshold or self.memory_cleanup_threshold
        
        try:
            # Monitor memory before operation
            initial_memory = self._get_memory_usage()
            yield
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {e}")
            self._clear_gpu_memory()
            raise OutOfMemoryError(f"GPU out of memory: {e}") from e
            
        except Exception as e:
            # Check if memory-related
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                self._clear_gpu_memory()
            raise
            
        finally:
            # Check memory usage and cleanup if needed
            current_memory = self._get_memory_usage()
            if current_memory.get('gpu_usage_ratio', 0) > cleanup_threshold:
                logger.info(f"Memory usage high ({current_memory['gpu_usage_ratio']:.1%}), performing cleanup")
                self._clear_gpu_memory()
    
    def create_error_recovery_checkpoint(self, 
                                       model: torch.nn.Module,
                                       optimizer: torch.optim.Optimizer,
                                       step: int,
                                       error_info: Dict[str, Any],
                                       output_dir: str) -> str:
        """Create an emergency checkpoint when errors occur.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            error_info: Information about the error
            output_dir: Directory to save checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"error_recovery_step_{step}_{timestamp}"
        checkpoint_path = os.path.join(output_dir, "checkpoints", checkpoint_name)
        
        try:
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model state
            model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            torch.save(model.state_dict(), model_path)
            
            # Save optimizer state
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            torch.save(optimizer.state_dict(), optimizer_path)
            
            # Save error information
            error_info_path = os.path.join(checkpoint_path, "error_info.json")
            import json
            with open(error_info_path, 'w') as f:
                json.dump(error_info, f, indent=2, default=str)
            
            logger.info(f"Error recovery checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to create error recovery checkpoint: {e}")
            raise
    
    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory cache with error handling."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU memory cache cleared")
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    logger.error(f"CUDA error during memory cleanup: {e}")
                    # Don't attempt further CUDA operations if context is corrupted
                    return
                else:
                    raise
            
            # Force garbage collection
            gc.collect()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_used_gb'] = system_memory.used / 1024**3
        memory_info['system_total_gb'] = system_memory.total / 1024**3
        memory_info['system_usage_ratio'] = system_memory.percent / 100.0
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            memory_info['gpu_allocated_gb'] = gpu_memory_allocated
            memory_info['gpu_reserved_gb'] = gpu_memory_reserved
            memory_info['gpu_total_gb'] = gpu_memory_total
            memory_info['gpu_usage_ratio'] = gpu_memory_reserved / gpu_memory_total
        
        return memory_info
    
    def _validate_checkpoint_files(self, checkpoint_path: str) -> bool:
        """Validate that checkpoint contains required files.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            True if checkpoint appears valid
        """
        required_files = ['pytorch_model.bin', 'config.json']
        
        for file_name in required_files:
            file_path = os.path.join(checkpoint_path, file_name)
            if not os.path.exists(file_path):
                return False
            
            # Check file size (should not be empty)
            if os.path.getsize(file_path) == 0:
                return False
        
        return True
    
    def _log_data_error_details(self, 
                              error: Exception,
                              sample_index: int,
                              dataset_path: str) -> None:
        """Log detailed information about data loading errors.
        
        Args:
            error: The error that occurred
            sample_index: Index of problematic sample
            dataset_path: Path to dataset
        """
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'sample_index': sample_index,
            'dataset_path': dataset_path,
            'traceback': traceback.format_exc()
        }
        
        logger.debug(f"Data loading error details: {error_details}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors encountered.
        
        Returns:
            Dictionary with error statistics
        """
        return {
            'error_counts': dict(self.error_counts),
            'recovery_attempts': dict(self.recovery_attempts),
            'total_errors': sum(self.error_counts.values()),
            'memory_usage': self._get_memory_usage()
        }


class GracefulShutdownHandler:
    """Handler for graceful shutdown and resource cleanup."""
    
    def __init__(self):
        """Initialize graceful shutdown handler."""
        self.shutdown_requested = False
        self.cleanup_callbacks = []
        self.resources_to_cleanup = []
        
        # Register signal handlers
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("GracefulShutdownHandler initialized")
    
    def register_cleanup_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Register a callback to be called during shutdown.
        
        Args:
            callback: Function to call during cleanup
            *args: Arguments for callback
            **kwargs: Keyword arguments for callback
        """
        self.cleanup_callbacks.append((callback, args, kwargs))
    
    def register_resource(self, resource: Any, cleanup_method: str = 'close') -> None:
        """Register a resource for automatic cleanup.
        
        Args:
            resource: Resource object to cleanup
            cleanup_method: Method name to call for cleanup
        """
        self.resources_to_cleanup.append((resource, cleanup_method))
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def cleanup(self) -> None:
        """Perform cleanup of all registered resources and callbacks."""
        logger.info("Performing graceful cleanup...")
        
        # Execute cleanup callbacks
        for callback, args, kwargs in self.cleanup_callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        # Cleanup registered resources
        for resource, cleanup_method in self.resources_to_cleanup:
            try:
                if hasattr(resource, cleanup_method):
                    getattr(resource, cleanup_method)()
            except Exception as e:
                logger.error(f"Resource cleanup failed: {e}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Graceful cleanup completed")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested.
        
        Returns:
            True if shutdown was requested
        """
        return self.shutdown_requested


def with_error_handling(error_handler: ErrorHandler):
    """Decorator to add error handling to functions.
    
    Args:
        error_handler: ErrorHandler instance to use
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory in {func.__name__}: {e}")
                raise OutOfMemoryError(f"GPU memory error in {func.__name__}: {e}") from e
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                raise
        
        return wrapper
    return decorator