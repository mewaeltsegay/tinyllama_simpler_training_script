"""Checkpoint management for model saving and loading with metadata support."""

import os
import json
import time
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages model checkpoint saving and loading with metadata support."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CheckpointManager initialized with dir: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: LlamaForCausalLM,
        tokenizer: Optional[LlamaTokenizer] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """Save model checkpoint with metadata.
        
        Args:
            model: Model to save
            tokenizer: Optional tokenizer to save
            optimizer: Optional optimizer state to save
            scheduler: Optional scheduler state to save
            step: Training step number
            epoch: Training epoch number
            loss: Current loss value
            metrics: Additional metrics to save
            config: Training configuration
            checkpoint_name: Custom checkpoint name (auto-generated if None)
            
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_step_{step}_epoch_{epoch}_{timestamp}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint to: {checkpoint_path}")
        
        try:
            # Save model and tokenizer
            model.save_pretrained(checkpoint_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint_path)
            
            # Prepare checkpoint metadata
            metadata = {
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "timestamp": datetime.now().isoformat(),
                "model_config": {
                    "vocab_size": model.config.vocab_size,
                    "hidden_size": model.config.hidden_size,
                    "num_layers": model.config.num_hidden_layers,
                    "num_attention_heads": model.config.num_attention_heads,
                    "parameter_count": sum(p.numel() for p in model.parameters())
                }
            }
            
            # Add metrics if provided
            if metrics:
                metadata["metrics"] = metrics
            
            # Add training config if provided
            if config:
                metadata["training_config"] = config
            
            # Save optimizer state
            if optimizer is not None:
                optimizer_path = checkpoint_path / "optimizer.pt"
                torch.save(optimizer.state_dict(), optimizer_path)
                metadata["has_optimizer"] = True
            else:
                metadata["has_optimizer"] = False
            
            # Save scheduler state
            if scheduler is not None:
                scheduler_path = checkpoint_path / "scheduler.pt"
                torch.save(scheduler.state_dict(), scheduler_path)
                metadata["has_scheduler"] = True
            else:
                metadata["has_scheduler"] = False
            
            # Save metadata
            metadata_path = checkpoint_path / "checkpoint_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Create a symlink to latest checkpoint
            latest_path = self.checkpoint_dir / "latest"
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(checkpoint_name, target_is_directory=True)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint saved successfully: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            # Clean up partial checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise RuntimeError(f"Checkpoint saving failed: {str(e)}") from e
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Optional[LlamaForCausalLM] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'auto'
    ) -> Tuple[LlamaForCausalLM, Dict[str, Any]]:
        """Load model checkpoint with metadata.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Optional existing model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load model on
            
        Returns:
            Tuple of (loaded_model, metadata)
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Handle 'latest' symlink
        if checkpoint_path.name == "latest":
            checkpoint_path = checkpoint_path.resolve()
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Load metadata
            metadata_path = checkpoint_path / "checkpoint_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                logger.warning("No metadata found, creating minimal metadata")
                metadata = {"step": 0, "epoch": 0, "loss": 0.0}
            
            # Load model
            if model is None:
                model = LlamaForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
                    device_map='auto' if device.startswith('cuda') else None,
                    low_cpu_mem_usage=True
                )
            else:
                # Load state dict into existing model
                model_state_path = checkpoint_path / "pytorch_model.bin"
                if model_state_path.exists():
                    state_dict = torch.load(model_state_path, map_location=device)
                    model.load_state_dict(state_dict)
                else:
                    # Try loading with from_pretrained
                    loaded_model = LlamaForCausalLM.from_pretrained(checkpoint_path)
                    model.load_state_dict(loaded_model.state_dict())
            
            # Load optimizer state
            if optimizer is not None and metadata.get("has_optimizer", False):
                optimizer_path = checkpoint_path / "optimizer.pt"
                if optimizer_path.exists():
                    optimizer_state = torch.load(optimizer_path, map_location=device)
                    optimizer.load_state_dict(optimizer_state)
                    logger.info("Optimizer state loaded")
                else:
                    logger.warning("Optimizer state not found")
            
            # Load scheduler state
            if scheduler is not None and metadata.get("has_scheduler", False):
                scheduler_path = checkpoint_path / "scheduler.pt"
                if scheduler_path.exists():
                    scheduler_state = torch.load(scheduler_path, map_location=device)
                    scheduler.load_state_dict(scheduler_state)
                    logger.info("Scheduler state loaded")
                else:
                    logger.warning("Scheduler state not found")
            
            # Validate checkpoint integrity
            self._validate_checkpoint_integrity(model, metadata)
            
            logger.info(f"Checkpoint loaded successfully from step {metadata.get('step', 0)}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Checkpoint loading failed: {str(e)}") from e
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if checkpoint_dir.is_dir() and checkpoint_dir.name != "latest":
                metadata_path = checkpoint_dir / "checkpoint_metadata.json"
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        checkpoint_info = {
                            "name": checkpoint_dir.name,
                            "path": str(checkpoint_dir),
                            "step": metadata.get("step", 0),
                            "epoch": metadata.get("epoch", 0),
                            "loss": metadata.get("loss", 0.0),
                            "timestamp": metadata.get("timestamp", "unknown"),
                            "size_mb": self._get_directory_size(checkpoint_dir) / (1024 * 1024)
                        }
                        
                        checkpoints.append(checkpoint_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {checkpoint_dir.name}: {e}")
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x["step"], reverse=True)
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        latest_path = self.checkpoint_dir / "latest"
        
        if latest_path.exists() and latest_path.is_symlink():
            return str(latest_path.resolve())
        
        # Fallback: find checkpoint with highest step number
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]["path"]
        
        return None
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_name}")
            return False
        
        try:
            shutil.rmtree(checkpoint_path)
            logger.info(f"Deleted checkpoint: {checkpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_name}: {e}")
            return False
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.max_checkpoints:
            # Keep the most recent checkpoints
            checkpoints_to_delete = checkpoints[self.max_checkpoints:]
            
            for checkpoint in checkpoints_to_delete:
                checkpoint_name = Path(checkpoint["path"]).name
                self.delete_checkpoint(checkpoint_name)
                logger.info(f"Cleaned up old checkpoint: {checkpoint_name}")
    
    def _validate_checkpoint_integrity(self, model: LlamaForCausalLM, metadata: Dict[str, Any]) -> None:
        """Validate checkpoint integrity.
        
        Args:
            model: Loaded model
            metadata: Checkpoint metadata
            
        Raises:
            RuntimeError: If validation fails
        """
        try:
            # Check parameter count
            current_params = sum(p.numel() for p in model.parameters())
            expected_params = metadata.get("model_config", {}).get("parameter_count")
            
            if expected_params and current_params != expected_params:
                logger.warning(f"Parameter count mismatch: expected {expected_params}, got {current_params}")
            
            # Basic forward pass test
            vocab_size = model.config.vocab_size
            test_input = torch.randint(0, vocab_size, (1, 10))
            
            # Move to model device
            model_device = next(model.parameters()).device
            test_input = test_input.to(model_device)
            
            with torch.no_grad():
                outputs = model(test_input)
            
            if outputs.logits is None:
                raise RuntimeError("Model forward pass failed after loading")
            
            logger.info("Checkpoint integrity validation passed")
            
        except Exception as e:
            logger.error(f"Checkpoint integrity validation failed: {str(e)}")
            raise RuntimeError(f"Checkpoint validation failed: {str(e)}") from e
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes.
        
        Args:
            directory: Directory path
            
        Returns:
            Size in bytes
        """
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def get_checkpoint_info(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint
            
        Returns:
            Checkpoint information or None if not found
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            return None
        
        metadata_path = checkpoint_path / "checkpoint_metadata.json"
        
        if not metadata_path.exists():
            return {"name": checkpoint_name, "path": str(checkpoint_path), "metadata": "missing"}
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return {
                "name": checkpoint_name,
                "path": str(checkpoint_path),
                "metadata": metadata,
                "size_mb": self._get_directory_size(checkpoint_path) / (1024 * 1024),
                "files": [f.name for f in checkpoint_path.iterdir() if f.is_file()]
            }
            
        except Exception as e:
            logger.error(f"Failed to read checkpoint info for {checkpoint_name}: {e}")
            return None