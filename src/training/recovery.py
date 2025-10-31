"""Training resumption and state recovery mechanisms."""

import os
import json
import time
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import torch
from transformers import LlamaForCausalLM
from ..utils.logging import get_logger
from ..utils.error_handling import CheckpointCorruptionError, ModelStateError

logger = get_logger(__name__)


class TrainingStateManager:
    """Manages training state persistence and recovery."""
    
    def __init__(self, checkpoint_dir: str, max_recovery_attempts: int = 3):
        """Initialize training state manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            max_recovery_attempts: Maximum attempts for state recovery
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_recovery_attempts = max_recovery_attempts
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.current_state = {}
        self.recovery_attempts = 0
        
        logger.info(f"TrainingStateManager initialized with checkpoint_dir: {self.checkpoint_dir}")
    
    def save_training_state(self,
                          model: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          scheduler: Optional[Any],
                          scaler: Optional[torch.cuda.amp.GradScaler],
                          step: int,
                          epoch: int,
                          metrics: Dict[str, Any],
                          config: Dict[str, Any],
                          additional_state: Optional[Dict[str, Any]] = None) -> str:
        """Save complete training state with validation.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            scaler: Gradient scaler for mixed precision
            step: Current training step
            epoch: Current epoch
            metrics: Training metrics
            config: Training configuration
            additional_state: Additional state information
            
        Returns:
            Path to saved checkpoint
            
        Raises:
            RuntimeError: If state saving fails
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"training_state_step_{step}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving training state to: {checkpoint_path}")
            
            # Save model state with validation
            self._save_model_state(model, checkpoint_path)
            
            # Save optimizer state
            self._save_optimizer_state(optimizer, checkpoint_path)
            
            # Save scheduler state
            if scheduler is not None:
                self._save_scheduler_state(scheduler, checkpoint_path)
            
            # Save scaler state
            if scaler is not None:
                self._save_scaler_state(scaler, checkpoint_path)
            
            # Create comprehensive training metadata
            training_metadata = {
                "step": step,
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "config": config,
                "model_info": self._get_model_info(model),
                "optimizer_info": self._get_optimizer_info(optimizer),
                "scheduler_info": self._get_scheduler_info(scheduler),
                "system_info": self._get_system_info(),
                "checkpoint_version": "1.0"
            }
            
            # Add additional state if provided
            if additional_state:
                training_metadata["additional_state"] = additional_state
            
            # Save metadata
            metadata_path = checkpoint_path / "training_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(training_metadata, f, indent=2, default=str)
            
            # Validate saved checkpoint
            self._validate_saved_checkpoint(checkpoint_path, training_metadata)
            
            # Create symlink to latest checkpoint
            self._update_latest_checkpoint_link(checkpoint_name)
            
            # Update current state tracking
            self.current_state = {
                "checkpoint_path": str(checkpoint_path),
                "step": step,
                "epoch": epoch,
                "timestamp": training_metadata["timestamp"]
            }
            
            logger.info(f"Training state saved successfully: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")
            # Clean up partial checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise RuntimeError(f"Training state saving failed: {e}") from e
    
    def load_training_state(self,
                          checkpoint_path: Optional[str] = None,
                          model: Optional[torch.nn.Module] = None,
                          optimizer: Optional[torch.optim.Optimizer] = None,
                          scheduler: Optional[Any] = None,
                          scaler: Optional[torch.cuda.amp.GradScaler] = None,
                          device: str = 'auto') -> Tuple[Dict[str, Any], bool]:
        """Load training state with comprehensive validation and recovery.
        
        Args:
            checkpoint_path: Path to checkpoint (uses latest if None)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            scaler: Scaler to load state into
            device: Device to load tensors on
            
        Returns:
            Tuple of (training_metadata, recovery_successful)
            
        Raises:
            CheckpointCorruptionError: If checkpoint is corrupted
            ModelStateError: If model state validation fails
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found for recovery")
        
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading training state from: {checkpoint_path}")
        
        try:
            # Load and validate metadata
            metadata = self._load_and_validate_metadata(checkpoint_path)
            
            # Load model state with validation
            if model is not None:
                self._load_model_state(model, checkpoint_path, device)
                self._validate_model_state(model, metadata)
            
            # Load optimizer state
            if optimizer is not None:
                self._load_optimizer_state(optimizer, checkpoint_path, device)
            
            # Load scheduler state
            if scheduler is not None:
                self._load_scheduler_state(scheduler, checkpoint_path)
            
            # Load scaler state
            if scaler is not None:
                self._load_scaler_state(scaler, checkpoint_path)
            
            # Update current state tracking
            self.current_state = {
                "checkpoint_path": str(checkpoint_path),
                "step": metadata["step"],
                "epoch": metadata["epoch"],
                "timestamp": metadata["timestamp"]
            }
            
            logger.info(f"Training state loaded successfully from step {metadata['step']}")
            return metadata, True
            
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            
            # Attempt recovery with fallback checkpoints
            if self.recovery_attempts < self.max_recovery_attempts:
                return self._attempt_recovery_with_fallback(
                    checkpoint_path, model, optimizer, scheduler, scaler, device
                )
            else:
                raise CheckpointCorruptionError(f"Training state loading failed after {self.max_recovery_attempts} attempts: {e}") from e
    
    def find_resumable_checkpoint(self) -> Optional[str]:
        """Find the most recent resumable checkpoint.
        
        Returns:
            Path to resumable checkpoint or None if none found
        """
        try:
            # First try the latest symlink
            latest_path = self._get_latest_checkpoint()
            if latest_path and self._is_checkpoint_valid(latest_path):
                return latest_path
            
            # Search for valid checkpoints
            valid_checkpoints = []
            for checkpoint_dir in self.checkpoint_dir.iterdir():
                if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('training_state_'):
                    if self._is_checkpoint_valid(checkpoint_dir):
                        metadata_path = checkpoint_dir / "training_metadata.json"
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                valid_checkpoints.append((str(checkpoint_dir), metadata.get('step', 0)))
                            except Exception:
                                continue
            
            if valid_checkpoints:
                # Return checkpoint with highest step number
                valid_checkpoints.sort(key=lambda x: x[1], reverse=True)
                return valid_checkpoints[0][0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding resumable checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_count: int = 5) -> None:
        """Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of checkpoints to keep
        """
        try:
            checkpoints = []
            for checkpoint_dir in self.checkpoint_dir.iterdir():
                if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('training_state_'):
                    metadata_path = checkpoint_dir / "training_metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            checkpoints.append((checkpoint_dir, metadata.get('step', 0)))
                        except Exception:
                            # If metadata is corrupted, consider for deletion
                            checkpoints.append((checkpoint_dir, -1))
            
            # Sort by step number (descending)
            checkpoints.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old checkpoints
            for checkpoint_dir, step in checkpoints[keep_count:]:
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Removed old checkpoint: {checkpoint_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_dir.name}: {e}")
            
            logger.info(f"Checkpoint cleanup completed, kept {min(len(checkpoints), keep_count)} checkpoints")
            
        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")
    
    def validate_training_state(self,
                              model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              step: int) -> bool:
        """Validate current training state for consistency.
        
        Args:
            model: Model to validate
            optimizer: Optimizer to validate
            step: Expected training step
            
        Returns:
            True if state is valid
        """
        try:
            # Validate model state
            if not self._validate_model_parameters(model):
                logger.error("Model parameter validation failed")
                return False
            
            # Validate optimizer state
            if not self._validate_optimizer_state(optimizer):
                logger.error("Optimizer state validation failed")
                return False
            
            # Check step consistency
            if self.current_state.get('step', 0) != step:
                logger.warning(f"Step mismatch: expected {step}, got {self.current_state.get('step', 0)}")
            
            logger.info("Training state validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Training state validation failed: {e}")
            return False
    
    def create_recovery_checkpoint(self,
                                 model: torch.nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 step: int,
                                 error_info: Dict[str, Any]) -> str:
        """Create a recovery checkpoint when errors occur.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current step
            error_info: Information about the error
            
        Returns:
            Path to recovery checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recovery_name = f"recovery_checkpoint_step_{step}_{timestamp}"
        recovery_path = self.checkpoint_dir / recovery_name
        
        try:
            recovery_path.mkdir(parents=True, exist_ok=True)
            
            # Save minimal state for recovery
            torch.save(model.state_dict(), recovery_path / "model_state.pt")
            torch.save(optimizer.state_dict(), recovery_path / "optimizer_state.pt")
            
            # Save error information
            recovery_metadata = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "error_info": error_info,
                "recovery_checkpoint": True
            }
            
            with open(recovery_path / "recovery_metadata.json", 'w') as f:
                json.dump(recovery_metadata, f, indent=2, default=str)
            
            logger.info(f"Recovery checkpoint created: {recovery_path}")
            return str(recovery_path)
            
        except Exception as e:
            logger.error(f"Failed to create recovery checkpoint: {e}")
            raise
    
    def _save_model_state(self, model: torch.nn.Module, checkpoint_path: Path) -> None:
        """Save model state with validation."""
        model_path = checkpoint_path / "pytorch_model.bin"
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Save model configuration if available
        if hasattr(model, 'config'):
            config_path = checkpoint_path / "config.json"
            model.config.save_pretrained(checkpoint_path)
        
        logger.debug("Model state saved")
    
    def _save_optimizer_state(self, optimizer: torch.optim.Optimizer, checkpoint_path: Path) -> None:
        """Save optimizer state."""
        optimizer_path = checkpoint_path / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        logger.debug("Optimizer state saved")
    
    def _save_scheduler_state(self, scheduler: Any, checkpoint_path: Path) -> None:
        """Save scheduler state."""
        scheduler_path = checkpoint_path / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_path)
        logger.debug("Scheduler state saved")
    
    def _save_scaler_state(self, scaler: torch.cuda.amp.GradScaler, checkpoint_path: Path) -> None:
        """Save gradient scaler state."""
        scaler_path = checkpoint_path / "scaler.pt"
        torch.save(scaler.state_dict(), scaler_path)
        logger.debug("Scaler state saved")
    
    def _load_model_state(self, model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
        """Load model state with validation."""
        model_path = checkpoint_path / "pytorch_model.bin"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model state file not found: {model_path}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Validate state dict compatibility
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        if model_keys != checkpoint_keys:
            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys
            
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        logger.debug("Model state loaded")
    
    def _load_optimizer_state(self, optimizer: torch.optim.Optimizer, checkpoint_path: Path, device: str) -> None:
        """Load optimizer state."""
        optimizer_path = checkpoint_path / "optimizer.pt"
        
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)
            logger.debug("Optimizer state loaded")
        else:
            logger.warning("Optimizer state file not found")
    
    def _load_scheduler_state(self, scheduler: Any, checkpoint_path: Path) -> None:
        """Load scheduler state."""
        scheduler_path = checkpoint_path / "scheduler.pt"
        
        if scheduler_path.exists():
            scheduler_state = torch.load(scheduler_path, map_location='cpu')
            scheduler.load_state_dict(scheduler_state)
            logger.debug("Scheduler state loaded")
        else:
            logger.warning("Scheduler state file not found")
    
    def _load_scaler_state(self, scaler: torch.cuda.amp.GradScaler, checkpoint_path: Path) -> None:
        """Load scaler state."""
        scaler_path = checkpoint_path / "scaler.pt"
        
        if scaler_path.exists():
            scaler_state = torch.load(scaler_path, map_location='cpu')
            scaler.load_state_dict(scaler_state)
            logger.debug("Scaler state loaded")
        else:
            logger.warning("Scaler state file not found")
    
    def _load_and_validate_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load and validate checkpoint metadata."""
        metadata_path = checkpoint_path / "training_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate required fields
            required_fields = ['step', 'epoch', 'timestamp']
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Missing required field in metadata: {field}")
            
            return metadata
            
        except json.JSONDecodeError as e:
            raise CheckpointCorruptionError(f"Corrupted metadata file: {e}") from e
    
    def _validate_saved_checkpoint(self, checkpoint_path: Path, metadata: Dict[str, Any]) -> None:
        """Validate that checkpoint was saved correctly."""
        # Check required files exist
        required_files = ['pytorch_model.bin', 'training_metadata.json']
        for file_name in required_files:
            file_path = checkpoint_path / file_name
            if not file_path.exists():
                raise RuntimeError(f"Required checkpoint file missing: {file_name}")
            
            # Check file is not empty
            if file_path.stat().st_size == 0:
                raise RuntimeError(f"Checkpoint file is empty: {file_name}")
        
        logger.debug("Checkpoint validation passed")
    
    def _validate_model_state(self, model: torch.nn.Module, metadata: Dict[str, Any]) -> None:
        """Validate loaded model state."""
        try:
            # Check parameter count
            param_count = sum(p.numel() for p in model.parameters())
            expected_param_count = metadata.get('model_info', {}).get('parameter_count')
            
            if expected_param_count and param_count != expected_param_count:
                logger.warning(f"Parameter count mismatch: expected {expected_param_count}, got {param_count}")
            
            # Test forward pass
            device = next(model.parameters()).device
            test_input = torch.randint(0, model.config.vocab_size, (1, 10), device=device)
            
            with torch.no_grad():
                outputs = model(test_input)
            
            if outputs.logits is None:
                raise ModelStateError("Model forward pass failed after loading")
            
            logger.debug("Model state validation passed")
            
        except Exception as e:
            raise ModelStateError(f"Model state validation failed: {e}") from e
    
    def _validate_model_parameters(self, model: torch.nn.Module) -> bool:
        """Validate model parameters for NaN or infinite values."""
        try:
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    logger.error(f"NaN values found in parameter: {name}")
                    return False
                
                if torch.isinf(param).any():
                    logger.error(f"Infinite values found in parameter: {name}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            return False
    
    def _validate_optimizer_state(self, optimizer: torch.optim.Optimizer) -> bool:
        """Validate optimizer state."""
        try:
            # Check if optimizer has state
            if not optimizer.state:
                logger.warning("Optimizer has no state")
                return True  # This might be normal for fresh optimizers
            
            # Check for NaN/inf in optimizer state
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
    
    def _is_checkpoint_valid(self, checkpoint_path: Path) -> bool:
        """Check if checkpoint appears to be valid."""
        try:
            # Check if directory exists
            if not checkpoint_path.is_dir():
                return False
            
            # Check for required files
            required_files = ['pytorch_model.bin', 'training_metadata.json']
            for file_name in required_files:
                file_path = checkpoint_path / file_name
                if not file_path.exists() or file_path.stat().st_size == 0:
                    return False
            
            # Try to load metadata
            metadata_path = checkpoint_path / "training_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check required fields
            required_fields = ['step', 'epoch', 'timestamp']
            for field in required_fields:
                if field not in metadata:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_link = self.checkpoint_dir / "latest"
        
        if latest_link.exists():
            if latest_link.is_symlink():
                # Handle symlink
                target = latest_link.resolve()
                if target.exists():
                    return str(target)
            else:
                # Handle text file fallback
                try:
                    with open(latest_link, 'r') as f:
                        checkpoint_name = f.read().strip()
                    checkpoint_path = self.checkpoint_dir / checkpoint_name
                    if checkpoint_path.exists():
                        return str(checkpoint_path)
                except Exception as e:
                    logger.debug(f"Failed to read latest checkpoint file: {e}")
        
        return None
    
    def _update_latest_checkpoint_link(self, checkpoint_name: str) -> None:
        """Update the 'latest' symlink to point to the new checkpoint."""
        latest_link = self.checkpoint_dir / "latest"
        
        try:
            # Remove existing link
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            
            # Create new link (try symlink first, fallback to text file on Windows)
            try:
                latest_link.symlink_to(checkpoint_name, target_is_directory=True)
            except (OSError, NotImplementedError):
                # Fallback for Windows without admin privileges
                with open(latest_link, 'w') as f:
                    f.write(checkpoint_name)
                logger.debug("Created latest checkpoint reference file (symlink not available)")
        except Exception as e:
            logger.warning(f"Failed to create latest checkpoint link: {e}")
            # Continue without the link - not critical for training
    
    def _attempt_recovery_with_fallback(self,
                                      failed_checkpoint: Path,
                                      model: Optional[torch.nn.Module],
                                      optimizer: Optional[torch.optim.Optimizer],
                                      scheduler: Optional[Any],
                                      scaler: Optional[torch.cuda.amp.GradScaler],
                                      device: str) -> Tuple[Dict[str, Any], bool]:
        """Attempt recovery using fallback checkpoints."""
        self.recovery_attempts += 1
        logger.warning(f"Attempting recovery with fallback checkpoints (attempt {self.recovery_attempts})")
        
        # Find alternative checkpoints
        fallback_checkpoints = []
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if (checkpoint_dir.is_dir() and 
                checkpoint_dir != failed_checkpoint and
                checkpoint_dir.name.startswith('training_state_')):
                if self._is_checkpoint_valid(checkpoint_dir):
                    fallback_checkpoints.append(checkpoint_dir)
        
        # Sort by modification time (newest first)
        fallback_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for fallback_checkpoint in fallback_checkpoints:
            try:
                logger.info(f"Trying fallback checkpoint: {fallback_checkpoint}")
                return self.load_training_state(
                    str(fallback_checkpoint), model, optimizer, scheduler, scaler, device
                )
            except Exception as e:
                logger.warning(f"Fallback checkpoint {fallback_checkpoint} also failed: {e}")
                continue
        
        raise CheckpointCorruptionError("All available checkpoints are corrupted")
    
    def _get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get model information for metadata."""
        info = {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_type": type(model).__name__
        }
        
        if hasattr(model, 'config'):
            info["config"] = model.config.to_dict()
        
        return info
    
    def _get_optimizer_info(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Get optimizer information for metadata."""
        return {
            "optimizer_type": type(optimizer).__name__,
            "param_groups_count": len(optimizer.param_groups),
            "state_count": len(optimizer.state)
        }
    
    def _get_scheduler_info(self, scheduler: Optional[Any]) -> Dict[str, Any]:
        """Get scheduler information for metadata."""
        if scheduler is None:
            return {"scheduler_type": None}
        
        return {
            "scheduler_type": type(scheduler).__name__,
            "last_epoch": getattr(scheduler, 'last_epoch', -1)
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for metadata."""
        import psutil
        
        info = {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "system_memory_gb": psutil.virtual_memory().total / 1024**3
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return info
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts.
        
        Returns:
            Dictionary with recovery statistics
        """
        return {
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts,
            "current_state": dict(self.current_state),
            "checkpoint_dir": str(self.checkpoint_dir),
            "available_checkpoints": len(list(self.checkpoint_dir.glob("training_state_*")))
        }