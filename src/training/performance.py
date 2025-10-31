"""Performance optimization utilities for training."""

import os
import logging
import warnings
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceOptimizer:
    """Handles various performance optimizations for training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize performance optimizer.
        
        Args:
            config: Configuration dictionary for optimizations
        """
        self.config = config or {}
        self.optimizations_applied = []
        
        # Check for available optimizations
        self.flash_attention_available = self._check_flash_attention()
        self.torch_compile_available = self._check_torch_compile()
        
        logger.info(f"PerformanceOptimizer initialized. Available optimizations: "
                   f"Flash Attention: {self.flash_attention_available}, "
                   f"Torch Compile: {self.torch_compile_available}")
    
    def _check_flash_attention(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            logger.info("Flash Attention 2 not available. Install with: pip install flash-attn")
            return False
    
    def _check_torch_compile(self) -> bool:
        """Check if torch.compile is available (PyTorch 2.0+)."""
        try:
            return hasattr(torch, 'compile') and torch.__version__ >= "2.0"
        except:
            return False
    
    def optimize_model(self, model: LlamaForCausalLM, 
                      enable_flash_attention: bool = True,
                      enable_torch_compile: bool = True,
                      compile_mode: str = "default") -> LlamaForCausalLM:
        """Apply model-level optimizations.
        
        Args:
            model: Model to optimize
            enable_flash_attention: Whether to enable Flash Attention 2
            enable_torch_compile: Whether to enable torch.compile
            compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            
        Returns:
            Optimized model
        """
        logger.info("Applying model optimizations...")
        
        # Apply Flash Attention 2 if available and requested
        if enable_flash_attention and self.flash_attention_available:
            model = self._apply_flash_attention(model)
        
        # Apply torch.compile if available and requested
        if enable_torch_compile and self.torch_compile_available:
            model = self._apply_torch_compile(model, compile_mode)
        
        # Apply memory optimizations
        model = self._apply_memory_optimizations(model)
        
        logger.info(f"Model optimizations applied: {self.optimizations_applied}")
        return model
    
    def _apply_flash_attention(self, model: LlamaForCausalLM) -> LlamaForCausalLM:
        """Apply Flash Attention 2 optimization."""
        try:
            # Check if model already uses Flash Attention
            if hasattr(model.config, '_flash_attn_2_enabled') and model.config._flash_attn_2_enabled:
                logger.info("Flash Attention 2 already enabled")
                return model
            
            # Enable Flash Attention 2
            if hasattr(model.config, 'use_flash_attention_2'):
                model.config.use_flash_attention_2 = True
                logger.info("Flash Attention 2 enabled via config")
                self.optimizations_applied.append("flash_attention_2")
            else:
                # Try to manually replace attention modules
                model = self._replace_attention_modules(model)
                
        except Exception as e:
            logger.warning(f"Failed to apply Flash Attention 2: {e}")
        
        return model
    
    def _replace_attention_modules(self, model: LlamaForCausalLM) -> LlamaForCausalLM:
        """Replace standard attention with Flash Attention modules."""
        try:
            from flash_attn.modules.mha import MHA
            from flash_attn.modules.mlp import MLP
            
            # This is a simplified example - actual implementation would need
            # to properly replace LlamaAttention modules with Flash Attention equivalents
            logger.info("Flash Attention module replacement not fully implemented")
            logger.info("Consider using a model that natively supports Flash Attention 2")
            
        except ImportError:
            logger.warning("Flash Attention modules not available for replacement")
        
        return model
    
    def _apply_torch_compile(self, model: LlamaForCausalLM, mode: str = "default") -> LlamaForCausalLM:
        """Apply torch.compile optimization."""
        try:
            # Compile the model for faster execution
            compiled_model = torch.compile(model, mode=mode)
            logger.info(f"torch.compile applied with mode: {mode}")
            self.optimizations_applied.append(f"torch_compile_{mode}")
            return compiled_model
            
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}")
            return model
    
    def _apply_memory_optimizations(self, model: LlamaForCausalLM) -> LlamaForCausalLM:
        """Apply memory-specific optimizations."""
        try:
            # Enable memory efficient attention if available
            if hasattr(model.config, 'use_memory_efficient_attention'):
                model.config.use_memory_efficient_attention = True
                logger.info("Memory efficient attention enabled")
                self.optimizations_applied.append("memory_efficient_attention")
            
            # Set model to use less memory for caching
            if hasattr(model.config, 'use_cache'):
                # Only disable cache if gradient checkpointing is enabled
                if getattr(model.config, 'gradient_checkpointing', False):
                    model.config.use_cache = False
                    logger.info("Model cache disabled for gradient checkpointing")
            
        except Exception as e:
            logger.warning(f"Failed to apply memory optimizations: {e}")
        
        return model
    
    def optimize_dataloader(self, dataloader: DataLoader,
                          enable_prefetch: bool = True,
                          prefetch_factor: int = 2,
                          persistent_workers: bool = True) -> DataLoader:
        """Optimize DataLoader for better performance.
        
        Args:
            dataloader: DataLoader to optimize
            enable_prefetch: Whether to enable prefetching
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Whether to keep workers alive between epochs
            
        Returns:
            Optimized DataLoader
        """
        logger.info("Optimizing DataLoader...")
        
        # Get current DataLoader parameters
        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        shuffle = hasattr(dataloader, 'shuffle') and dataloader.shuffle
        num_workers = dataloader.num_workers
        pin_memory = dataloader.pin_memory
        collate_fn = dataloader.collate_fn
        
        # Apply optimizations
        optimized_params = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': max(num_workers, 1),  # Ensure at least 1 worker
            'pin_memory': pin_memory and torch.cuda.is_available(),
            'collate_fn': collate_fn,
            'drop_last': True,  # Drop last incomplete batch for consistent performance
        }
        
        # Add prefetching if enabled and workers > 0
        if enable_prefetch and optimized_params['num_workers'] > 0:
            optimized_params['prefetch_factor'] = prefetch_factor
            optimized_params['persistent_workers'] = persistent_workers
            logger.info(f"DataLoader prefetching enabled: factor={prefetch_factor}, "
                       f"persistent_workers={persistent_workers}")
        
        # Create optimized DataLoader
        try:
            optimized_dataloader = DataLoader(**optimized_params)
            logger.info("DataLoader optimization completed")
            return optimized_dataloader
            
        except Exception as e:
            logger.warning(f"DataLoader optimization failed: {e}")
            return dataloader
    
    def setup_efficient_training(self, model: LlamaForCausalLM, 
                                optimizer: torch.optim.Optimizer,
                                enable_amp_autocast: bool = True) -> Dict[str, Any]:
        """Setup efficient training configurations.
        
        Args:
            model: Model for training
            optimizer: Optimizer
            enable_amp_autocast: Whether to enable automatic mixed precision
            
        Returns:
            Dictionary with training setup information
        """
        logger.info("Setting up efficient training configurations...")
        
        setup_info = {
            "optimizations": [],
            "recommendations": []
        }
        
        # Enable efficient attention patterns
        if hasattr(model, 'set_default_attn_implementation'):
            try:
                model.set_default_attn_implementation('flash_attention_2')
                setup_info["optimizations"].append("flash_attention_2_default")
            except:
                pass
        
        # Configure optimizer for efficiency
        if hasattr(optimizer, 'param_groups'):
            for group in optimizer.param_groups:
                # Enable fused optimizer operations if available
                if 'fused' in group and torch.cuda.is_available():
                    group['fused'] = True
                    setup_info["optimizations"].append("fused_optimizer")
        
        # Setup automatic mixed precision
        if enable_amp_autocast and torch.cuda.is_available():
            setup_info["use_amp"] = True
            setup_info["optimizations"].append("automatic_mixed_precision")
        
        # Memory optimization recommendations
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 12:
                setup_info["recommendations"].extend([
                    "Enable gradient checkpointing",
                    "Use smaller batch sizes with gradient accumulation",
                    "Consider reducing sequence length"
                ])
            elif gpu_memory >= 24:
                setup_info["recommendations"].extend([
                    "Consider larger batch sizes for better throughput",
                    "Flash Attention 2 recommended for long sequences"
                ])
        
        logger.info(f"Efficient training setup completed: {setup_info['optimizations']}")
        return setup_info
    
    def benchmark_optimizations(self, model: LlamaForCausalLM,
                              sample_batch: Dict[str, torch.Tensor],
                              num_iterations: int = 10) -> Dict[str, float]:
        """Benchmark different optimization configurations.
        
        Args:
            model: Model to benchmark
            sample_batch: Sample batch for benchmarking
            num_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Benchmarking optimization configurations...")
        
        results = {}
        
        # Benchmark baseline (no optimizations)
        baseline_time = self._benchmark_forward_pass(model, sample_batch, num_iterations)
        results["baseline"] = baseline_time
        
        # Benchmark with torch.compile if available
        if self.torch_compile_available:
            try:
                compiled_model = torch.compile(model, mode="default")
                compile_time = self._benchmark_forward_pass(compiled_model, sample_batch, num_iterations)
                results["torch_compile"] = compile_time
                speedup = baseline_time / compile_time if compile_time > 0 else 0
                logger.info(f"torch.compile speedup: {speedup:.2f}x")
            except Exception as e:
                logger.warning(f"torch.compile benchmark failed: {e}")
        
        # Benchmark with different precision modes
        if torch.cuda.is_available():
            # FP16 benchmark
            try:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    fp16_time = self._benchmark_forward_pass(model, sample_batch, num_iterations)
                results["fp16"] = fp16_time
                speedup = baseline_time / fp16_time if fp16_time > 0 else 0
                logger.info(f"FP16 speedup: {speedup:.2f}x")
            except Exception as e:
                logger.warning(f"FP16 benchmark failed: {e}")
            
            # BF16 benchmark if supported
            if torch.cuda.is_bf16_supported():
                try:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        bf16_time = self._benchmark_forward_pass(model, sample_batch, num_iterations)
                    results["bf16"] = bf16_time
                    speedup = baseline_time / bf16_time if bf16_time > 0 else 0
                    logger.info(f"BF16 speedup: {speedup:.2f}x")
                except Exception as e:
                    logger.warning(f"BF16 benchmark failed: {e}")
        
        logger.info(f"Benchmark results: {results}")
        return results
    
    def _benchmark_forward_pass(self, model: nn.Module,
                               batch: Dict[str, torch.Tensor],
                               num_iterations: int) -> float:
        """Benchmark forward pass timing.
        
        Args:
            model: Model to benchmark
            batch: Input batch
            num_iterations: Number of iterations
            
        Returns:
            Average time per iteration in seconds
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(**batch)
        
        # Synchronize GPU if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(**batch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        return avg_time
    
    def get_optimization_recommendations(self, hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization recommendations based on hardware.
        
        Args:
            hardware_info: Hardware information dictionary
            
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = {
            "model_optimizations": [],
            "training_optimizations": [],
            "dataloader_optimizations": []
        }
        
        gpu_memory = hardware_info.get("gpu_memory_gb", 0)
        gpu_name = hardware_info.get("gpu_name", "").lower()
        
        # Model optimization recommendations
        if self.torch_compile_available:
            recommendations["model_optimizations"].append({
                "name": "torch.compile",
                "description": "Compile model for faster execution",
                "expected_speedup": "1.2-2.0x"
            })
        
        if self.flash_attention_available:
            recommendations["model_optimizations"].append({
                "name": "Flash Attention 2",
                "description": "Memory-efficient attention implementation",
                "expected_speedup": "1.5-3.0x for long sequences"
            })
        
        # Training optimization recommendations
        if gpu_memory < 12:
            recommendations["training_optimizations"].extend([
                {
                    "name": "Gradient Checkpointing",
                    "description": "Trade computation for memory",
                    "memory_savings": "30-50%"
                },
                {
                    "name": "FP16 Mixed Precision",
                    "description": "Reduce memory usage and increase speed",
                    "memory_savings": "40-50%"
                }
            ])
        elif gpu_memory >= 24:
            recommendations["training_optimizations"].extend([
                {
                    "name": "BF16 Mixed Precision",
                    "description": "Better numerical stability than FP16",
                    "benefits": "Improved training stability"
                },
                {
                    "name": "Larger Batch Sizes",
                    "description": "Better GPU utilization",
                    "throughput_improvement": "20-40%"
                }
            ])
        
        # DataLoader optimization recommendations
        cpu_cores = hardware_info.get("cpu_cores", 1)
        recommended_workers = min(cpu_cores, 8)
        
        recommendations["dataloader_optimizations"].extend([
            {
                "name": "Multi-worker Data Loading",
                "description": f"Use {recommended_workers} workers for parallel data loading",
                "recommended_workers": recommended_workers
            },
            {
                "name": "Prefetching",
                "description": "Prefetch batches to overlap data loading with training",
                "prefetch_factor": 2
            },
            {
                "name": "Pin Memory",
                "description": "Faster GPU memory transfers",
                "enabled": torch.cuda.is_available()
            }
        ])
        
        return recommendations
    
    def apply_all_optimizations(self, model: LlamaForCausalLM,
                               dataloader: DataLoader,
                               hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all available optimizations based on hardware.
        
        Args:
            model: Model to optimize
            dataloader: DataLoader to optimize
            hardware_info: Hardware information
            
        Returns:
            Dictionary with applied optimizations and results
        """
        logger.info("Applying all available optimizations...")
        
        results = {
            "model_optimizations": [],
            "dataloader_optimizations": [],
            "performance_impact": {}
        }
        
        # Apply model optimizations
        gpu_memory = hardware_info.get("gpu_memory_gb", 0)
        
        # Enable torch.compile for GPUs with sufficient memory
        enable_compile = gpu_memory >= 8 and self.torch_compile_available
        
        # Enable Flash Attention for all compatible GPUs
        enable_flash_attn = self.flash_attention_available
        
        optimized_model = self.optimize_model(
            model,
            enable_flash_attention=enable_flash_attn,
            enable_torch_compile=enable_compile,
            compile_mode="default"
        )
        
        results["model_optimizations"] = self.optimizations_applied.copy()
        
        # Apply DataLoader optimizations
        cpu_cores = hardware_info.get("cpu_cores", 1)
        num_workers = min(cpu_cores, 8)
        
        optimized_dataloader = self.optimize_dataloader(
            dataloader,
            enable_prefetch=True,
            prefetch_factor=2,
            persistent_workers=num_workers > 0
        )
        
        results["dataloader_optimizations"] = [
            "multi_worker_loading",
            "prefetching",
            "persistent_workers"
        ]
        
        logger.info(f"All optimizations applied: {results}")
        return results