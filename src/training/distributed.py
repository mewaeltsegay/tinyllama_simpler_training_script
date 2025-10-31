"""Distributed training support with DistributedDataParallel (DDP)."""

import os
import logging
import socket
from typing import Dict, Any, Optional, Tuple
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..utils.logging import get_logger

logger = get_logger(__name__)


class DistributedTrainingManager:
    """Manages distributed training setup and coordination."""
    
    def __init__(self):
        """Initialize distributed training manager."""
        self.is_distributed = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = None
        self.backend = "nccl"
        
        # Check if distributed training is available
        self.distributed_available = torch.distributed.is_available()
        
        logger.info(f"DistributedTrainingManager initialized. "
                   f"Distributed available: {self.distributed_available}")
    
    def setup_distributed_training(self, 
                                 backend: str = "nccl",
                                 init_method: Optional[str] = None,
                                 timeout_minutes: int = 30) -> bool:
        """Setup distributed training environment.
        
        Args:
            backend: Distributed backend ('nccl', 'gloo', 'mpi')
            init_method: Initialization method (auto-detected if None)
            timeout_minutes: Timeout for initialization
            
        Returns:
            True if distributed training was successfully initialized
        """
        if not self.distributed_available:
            logger.warning("Distributed training not available")
            return False
        
        # Check for distributed environment variables
        if not self._check_distributed_env():
            logger.info("Distributed environment not detected, using single GPU")
            return False
        
        try:
            # Get distributed parameters from environment
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.rank = int(os.environ.get('RANK', 0))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Set device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = f"cuda:{self.local_rank}"
            else:
                self.device = "cpu"
                backend = "gloo"  # Use gloo for CPU
            
            # Initialize process group
            if init_method is None:
                init_method = "env://"
            
            timeout = torch.distributed.default_pg_timeout
            if timeout_minutes > 0:
                timeout = torch.timedelta(minutes=timeout_minutes)
            
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=self.world_size,
                rank=self.rank,
                timeout=timeout
            )
            
            self.is_distributed = True
            self.backend = backend
            
            logger.info(f"Distributed training initialized: "
                       f"rank={self.rank}/{self.world_size}, "
                       f"local_rank={self.local_rank}, "
                       f"device={self.device}, "
                       f"backend={backend}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            return False
    
    def _check_distributed_env(self) -> bool:
        """Check if distributed environment variables are set."""
        required_vars = ['WORLD_SIZE', 'RANK']
        return all(var in os.environ for var in required_vars)
    
    def wrap_model_for_distributed(self, model: torch.nn.Module,
                                  find_unused_parameters: bool = False,
                                  broadcast_buffers: bool = True) -> torch.nn.Module:
        """Wrap model with DistributedDataParallel.
        
        Args:
            model: Model to wrap
            find_unused_parameters: Whether to find unused parameters
            broadcast_buffers: Whether to broadcast buffers
            
        Returns:
            DDP-wrapped model or original model if not distributed
        """
        if not self.is_distributed:
            return model
        
        try:
            # Move model to appropriate device
            model = model.to(self.device)
            
            # Wrap with DDP
            ddp_model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=broadcast_buffers
            )
            
            logger.info(f"Model wrapped with DDP on device {self.device}")
            return ddp_model
            
        except Exception as e:
            logger.error(f"Failed to wrap model with DDP: {e}")
            return model
    
    def create_distributed_dataloader(self, dataset,
                                    batch_size: int,
                                    shuffle: bool = True,
                                    num_workers: int = 0,
                                    pin_memory: bool = True,
                                    drop_last: bool = True,
                                    **kwargs) -> DataLoader:
        """Create DataLoader with distributed sampling.
        
        Args:
            dataset: Dataset to create loader for
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes per GPU
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            **kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader with distributed sampling
        """
        sampler = None
        
        if self.is_distributed:
            # Create distributed sampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=drop_last
            )
            # Don't shuffle in DataLoader when using DistributedSampler
            shuffle = False
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
        
        if self.is_distributed:
            logger.info(f"Created distributed DataLoader: "
                       f"batch_size={batch_size}, "
                       f"num_workers={num_workers}, "
                       f"world_size={self.world_size}")
        
        return dataloader
    
    def all_reduce_tensor(self, tensor: torch.Tensor, 
                         op: str = "mean") -> torch.Tensor:
        """All-reduce tensor across all processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('sum', 'mean', 'max', 'min')
            
        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor
        
        try:
            # Clone tensor to avoid in-place operations
            reduced_tensor = tensor.clone()
            
            # Perform all-reduce
            if op == "sum":
                dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
            elif op == "mean":
                dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
                reduced_tensor /= self.world_size
            elif op == "max":
                dist.all_reduce(reduced_tensor, op=dist.ReduceOp.MAX)
            elif op == "min":
                dist.all_reduce(reduced_tensor, op=dist.ReduceOp.MIN)
            else:
                raise ValueError(f"Unsupported reduction operation: {op}")
            
            return reduced_tensor
            
        except Exception as e:
            logger.error(f"All-reduce failed: {e}")
            return tensor
    
    def all_gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather tensor from all processes.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            Concatenated tensor from all processes
        """
        if not self.is_distributed:
            return tensor
        
        try:
            # Create list to store tensors from all processes
            tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            
            # All-gather tensors
            dist.all_gather(tensor_list, tensor)
            
            # Concatenate tensors
            gathered_tensor = torch.cat(tensor_list, dim=0)
            
            return gathered_tensor
            
        except Exception as e:
            logger.error(f"All-gather failed: {e}")
            return tensor
    
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast object from source rank to all ranks.
        
        Args:
            obj: Object to broadcast
            src: Source rank
            
        Returns:
            Broadcasted object
        """
        if not self.is_distributed:
            return obj
        
        try:
            # Use torch.distributed.broadcast_object_list
            obj_list = [obj]
            dist.broadcast_object_list(obj_list, src=src)
            return obj_list[0]
            
        except Exception as e:
            logger.error(f"Broadcast failed: {e}")
            return obj
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_distributed:
            try:
                dist.barrier()
            except Exception as e:
                logger.error(f"Barrier failed: {e}")
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
    
    def cleanup_distributed(self) -> None:
        """Clean up distributed training."""
        if self.is_distributed:
            try:
                dist.destroy_process_group()
                logger.info("Distributed training cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup distributed training: {e}")
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """Get distributed training information.
        
        Returns:
            Dictionary with distributed training info
        """
        return {
            "is_distributed": self.is_distributed,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device": str(self.device),
            "backend": self.backend,
            "is_main_process": self.is_main_process()
        }
    
    def save_checkpoint_distributed(self, state_dict: Dict[str, Any],
                                   checkpoint_path: str) -> None:
        """Save checkpoint only from main process.
        
        Args:
            state_dict: State dictionary to save
            checkpoint_path: Path to save checkpoint
        """
        if self.is_main_process():
            try:
                torch.save(state_dict, checkpoint_path)
                logger.info(f"Checkpoint saved by main process: {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
        
        # Synchronize all processes
        self.barrier()
    
    def load_checkpoint_distributed(self, checkpoint_path: str,
                                   map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint in distributed setting.
        
        Args:
            checkpoint_path: Path to checkpoint
            map_location: Device to map tensors to
            
        Returns:
            Loaded state dictionary
        """
        if map_location is None:
            map_location = self.device
        
        try:
            # Load checkpoint
            state_dict = torch.load(checkpoint_path, map_location=map_location)
            logger.info(f"Checkpoint loaded on rank {self.rank}: {checkpoint_path}")
            return state_dict
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint on rank {self.rank}: {e}")
            return {}
    
    def calculate_effective_batch_size(self, per_gpu_batch_size: int,
                                     gradient_accumulation_steps: int = 1) -> int:
        """Calculate effective batch size across all GPUs.
        
        Args:
            per_gpu_batch_size: Batch size per GPU
            gradient_accumulation_steps: Gradient accumulation steps
            
        Returns:
            Effective batch size
        """
        return per_gpu_batch_size * self.world_size * gradient_accumulation_steps
    
    def adjust_learning_rate_for_distributed(self, base_lr: float,
                                            batch_size_scaling: bool = True) -> float:
        """Adjust learning rate for distributed training.
        
        Args:
            base_lr: Base learning rate
            batch_size_scaling: Whether to scale LR with batch size
            
        Returns:
            Adjusted learning rate
        """
        if not self.is_distributed or not batch_size_scaling:
            return base_lr
        
        # Linear scaling rule: scale LR proportionally to world size
        adjusted_lr = base_lr * self.world_size
        
        logger.info(f"Learning rate adjusted for distributed training: "
                   f"{base_lr} -> {adjusted_lr} (world_size={self.world_size})")
        
        return adjusted_lr
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information for current device.
        
        Returns:
            Dictionary with memory information in GB
        """
        if not torch.cuda.is_available():
            return {"total": 0.0, "allocated": 0.0, "cached": 0.0}
        
        device_id = self.local_rank if self.is_distributed else 0
        
        return {
            "total": torch.cuda.get_device_properties(device_id).total_memory / 1024**3,
            "allocated": torch.cuda.memory_allocated(device_id) / 1024**3,
            "cached": torch.cuda.memory_reserved(device_id) / 1024**3
        }
    
    def log_distributed_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Log and aggregate metrics across all processes.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Aggregated metrics (only on main process)
        """
        if not self.is_distributed:
            return metrics
        
        aggregated_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Convert to tensor for all-reduce
                tensor_value = torch.tensor(value, dtype=torch.float32, device=self.device)
                
                # All-reduce with mean
                reduced_value = self.all_reduce_tensor(tensor_value, op="mean")
                
                aggregated_metrics[key] = reduced_value.item()
            else:
                # Non-numeric values are not aggregated
                aggregated_metrics[key] = value
        
        return aggregated_metrics


def setup_distributed_environment(local_rank: Optional[int] = None,
                                 world_size: Optional[int] = None,
                                 rank: Optional[int] = None,
                                 master_addr: str = "localhost",
                                 master_port: str = "12355") -> None:
    """Setup distributed environment variables.
    
    Args:
        local_rank: Local rank of the process
        world_size: Total number of processes
        rank: Global rank of the process
        master_addr: Master node address
        master_port: Master node port
    """
    if local_rank is not None:
        os.environ['LOCAL_RANK'] = str(local_rank)
    
    if world_size is not None:
        os.environ['WORLD_SIZE'] = str(world_size)
    
    if rank is not None:
        os.environ['RANK'] = str(rank)
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    logger.info(f"Distributed environment setup: "
               f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}, "
               f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
               f"RANK={os.environ.get('RANK')}, "
               f"MASTER_ADDR={master_addr}, "
               f"MASTER_PORT={master_port}")


def find_free_port() -> int:
    """Find a free port for distributed training.
    
    Returns:
        Free port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def auto_detect_distributed_config() -> Dict[str, Any]:
    """Auto-detect distributed training configuration.
    
    Returns:
        Dictionary with detected configuration
    """
    config = {
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "distributed_available": torch.distributed.is_available(),
        "nccl_available": torch.distributed.is_nccl_available() if torch.distributed.is_available() else False,
        "gloo_available": torch.distributed.is_gloo_available() if torch.distributed.is_available() else False,
        "recommended_backend": "nccl" if torch.cuda.is_available() else "gloo",
        "can_use_distributed": False
    }
    
    # Check if distributed training is feasible
    if config["num_gpus"] > 1 and config["distributed_available"]:
        config["can_use_distributed"] = True
        config["recommended_world_size"] = config["num_gpus"]
    
    return config