"""Hardware detection and profile management."""

import torch
import psutil
import subprocess
import re
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from .base import HardwareInfo, HardwareConfig, TrainingConfig, MemoryEstimate
from ..utils.helpers import get_hardware_info


@dataclass
class HardwareProfile:
    """Hardware-specific training profile."""
    name: str
    batch_size: int
    gradient_accumulation_steps: int
    mixed_precision: str
    gradient_checkpointing: bool
    dataloader_workers: int
    max_memory_gb: float
    recommended_max_length: int


class HardwareAdapter:
    """Hardware detection and automatic configuration adjustment."""
    
    def __init__(self):
        """Initialize hardware adapter with predefined profiles."""
        self._profiles = self._create_hardware_profiles()
    
    def _create_hardware_profiles(self) -> Dict[str, HardwareProfile]:
        """Create predefined hardware profiles."""
        return {
            "rtx_4050": HardwareProfile(
                name="RTX 4050 (8GB VRAM)",
                batch_size=1,
                gradient_accumulation_steps=32,
                mixed_precision="fp16",
                gradient_checkpointing=True,
                dataloader_workers=2,
                max_memory_gb=6.0,  # Leave 2GB for system
                recommended_max_length=1024
            ),
            "rtx_4060": HardwareProfile(
                name="RTX 4060 (8GB VRAM)",
                batch_size=2,
                gradient_accumulation_steps=16,
                mixed_precision="fp16",
                gradient_checkpointing=True,
                dataloader_workers=2,
                max_memory_gb=6.5,
                recommended_max_length=1024
            ),
            "rtx_4070": HardwareProfile(
                name="RTX 4070 (12GB VRAM)",
                batch_size=3,
                gradient_accumulation_steps=12,
                mixed_precision="fp16",
                gradient_checkpointing=True,
                dataloader_workers=4,
                max_memory_gb=10.0,
                recommended_max_length=1536
            ),
            "rtx_4080": HardwareProfile(
                name="RTX 4080 (16GB VRAM)",
                batch_size=4,
                gradient_accumulation_steps=8,
                mixed_precision="fp16",
                gradient_checkpointing=False,
                dataloader_workers=4,
                max_memory_gb=14.0,
                recommended_max_length=2048
            ),
            "rtx_4090": HardwareProfile(
                name="RTX 4090 (24GB VRAM)",
                batch_size=6,
                gradient_accumulation_steps=6,
                mixed_precision="fp16",
                gradient_checkpointing=False,
                dataloader_workers=6,
                max_memory_gb=20.0,
                recommended_max_length=2048
            ),
            "h100_80gb": HardwareProfile(
                name="H100 80GB",
                batch_size=16,
                gradient_accumulation_steps=4,
                mixed_precision="bf16",
                gradient_checkpointing=False,
                dataloader_workers=8,
                max_memory_gb=70.0,
                recommended_max_length=2048
            ),
            "a100_40gb": HardwareProfile(
                name="A100 40GB",
                batch_size=8,
                gradient_accumulation_steps=6,
                mixed_precision="bf16",
                gradient_checkpointing=False,
                dataloader_workers=6,
                max_memory_gb=35.0,
                recommended_max_length=2048
            ),
            "a100_80gb": HardwareProfile(
                name="A100 80GB",
                batch_size=12,
                gradient_accumulation_steps=4,
                mixed_precision="bf16",
                gradient_checkpointing=False,
                dataloader_workers=8,
                max_memory_gb=70.0,
                recommended_max_length=2048
            ),
            "generic_8gb": HardwareProfile(
                name="Generic 8GB GPU",
                batch_size=1,
                gradient_accumulation_steps=32,
                mixed_precision="fp16",
                gradient_checkpointing=True,
                dataloader_workers=2,
                max_memory_gb=6.0,
                recommended_max_length=1024
            ),
            "generic_16gb": HardwareProfile(
                name="Generic 16GB GPU",
                batch_size=4,
                gradient_accumulation_steps=8,
                mixed_precision="fp16",
                gradient_checkpointing=False,
                dataloader_workers=4,
                max_memory_gb=14.0,
                recommended_max_length=2048
            ),
            "generic_24gb": HardwareProfile(
                name="Generic 24GB GPU",
                batch_size=6,
                gradient_accumulation_steps=6,
                mixed_precision="fp16",
                gradient_checkpointing=False,
                dataloader_workers=6,
                max_memory_gb=20.0,
                recommended_max_length=2048
            )
        }
    
    def detect_hardware(self) -> HardwareInfo:
        """
        Detect available hardware and return detailed information.
        
        Returns:
            HardwareInfo object with detected hardware details
        """
        hardware_data = get_hardware_info()
        
        # Get primary GPU information
        gpu_name = ""
        gpu_memory_gb = 0.0
        cuda_version = ""
        
        if hardware_data["cuda_available"] and hardware_data["gpu_count"] > 0:
            gpu_name = hardware_data["gpu_names"][0]
            gpu_memory_gb = hardware_data["gpu_memory_gb"][0]
            cuda_version = hardware_data["cuda_version"] or "Unknown"
        
        return HardwareInfo(
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            gpu_count=hardware_data["gpu_count"],
            cpu_cores=hardware_data["cpu_cores"],
            system_memory_gb=hardware_data["system_memory_gb"],
            cuda_version=cuda_version
        )
    
    def _identify_gpu_profile(self, gpu_name: str, gpu_memory_gb: float) -> str:
        """
        Identify the best matching hardware profile for a GPU.
        
        Args:
            gpu_name: Name of the GPU
            gpu_memory_gb: GPU memory in GB
            
        Returns:
            Profile key string
        """
        gpu_name_lower = gpu_name.lower()
        
        # Specific GPU model detection
        if "rtx 4050" in gpu_name_lower or "rtx4050" in gpu_name_lower:
            return "rtx_4050"
        elif "rtx 4060" in gpu_name_lower or "rtx4060" in gpu_name_lower:
            return "rtx_4060"
        elif "rtx 4070" in gpu_name_lower or "rtx4070" in gpu_name_lower:
            return "rtx_4070"
        elif "rtx 4080" in gpu_name_lower or "rtx4080" in gpu_name_lower:
            return "rtx_4080"
        elif "rtx 4090" in gpu_name_lower or "rtx4090" in gpu_name_lower:
            return "rtx_4090"
        elif "h100" in gpu_name_lower:
            return "h100_80gb"
        elif "a100" in gpu_name_lower:
            if gpu_memory_gb >= 70:
                return "a100_80gb"
            else:
                return "a100_40gb"
        
        # Fallback to memory-based detection
        if gpu_memory_gb >= 70:
            return "h100_80gb"
        elif gpu_memory_gb >= 35:
            return "a100_40gb"
        elif gpu_memory_gb >= 20:
            return "generic_24gb"
        elif gpu_memory_gb >= 14:
            return "generic_16gb"
        else:
            return "generic_8gb"
    
    def get_optimal_config(self, hardware_info: HardwareInfo) -> HardwareConfig:
        """
        Get optimal hardware configuration based on detected hardware.
        
        Args:
            hardware_info: Detected hardware information
            
        Returns:
            Optimized HardwareConfig
        """
        if hardware_info.gpu_count == 0:
            # CPU-only fallback
            return HardwareConfig(
                device="cpu",
                num_gpus=0,
                dataloader_workers=min(4, hardware_info.cpu_cores),
                pin_memory=False
            )
        
        # Get profile for primary GPU
        profile_key = self._identify_gpu_profile(
            hardware_info.gpu_name, 
            hardware_info.gpu_memory_gb
        )
        profile = self._profiles[profile_key]
        
        return HardwareConfig(
            device="auto",
            num_gpus=hardware_info.gpu_count,  # Support multi-GPU
            dataloader_workers=min(profile.dataloader_workers, hardware_info.cpu_cores),
            pin_memory=True
        )
    
    def get_hardware_profile(self, hardware_info: HardwareInfo) -> HardwareProfile:
        """
        Get the hardware profile for detected hardware.
        
        Args:
            hardware_info: Detected hardware information
            
        Returns:
            Matching HardwareProfile
        """
        if hardware_info.gpu_count == 0:
            # Return a CPU profile (very conservative)
            return HardwareProfile(
                name="CPU Only",
                batch_size=1,
                gradient_accumulation_steps=64,
                mixed_precision="fp32",
                gradient_checkpointing=True,
                dataloader_workers=2,
                max_memory_gb=hardware_info.system_memory_gb * 0.5,
                recommended_max_length=512
            )
        
        profile_key = self._identify_gpu_profile(
            hardware_info.gpu_name,
            hardware_info.gpu_memory_gb
        )
        return self._profiles[profile_key]
    
    def adjust_config_for_hardware(self, config: TrainingConfig, 
                                 hardware_info: HardwareInfo) -> TrainingConfig:
        """
        Automatically adjust training configuration based on hardware.
        
        Args:
            config: Original training configuration
            hardware_info: Detected hardware information
            
        Returns:
            Hardware-optimized training configuration
        """
        if hardware_info.gpu_count == 0:
            # CPU-only adjustments
            config.training_params.mixed_precision = "fp32"
            config.training_params.batch_size = 1
            config.training_params.gradient_accumulation_steps = 64
            config.training_params.gradient_checkpointing = True
            config.hardware_config.device = "cpu"
            config.hardware_config.num_gpus = 0
            config.hardware_config.pin_memory = False
            config.data_config.max_length = min(config.data_config.max_length, 512)
            return config
        
        # Get hardware profile
        profile = self.get_hardware_profile(hardware_info)
        
        # Adjust training parameters based on profile
        config.training_params.batch_size = min(
            config.training_params.batch_size, 
            profile.batch_size
        )
        config.training_params.gradient_accumulation_steps = max(
            config.training_params.gradient_accumulation_steps,
            profile.gradient_accumulation_steps
        )
        config.training_params.mixed_precision = profile.mixed_precision
        config.training_params.gradient_checkpointing = profile.gradient_checkpointing
        
        # Adjust hardware configuration
        config.hardware_config = self.get_optimal_config(hardware_info)
        
        # Adjust data configuration
        config.data_config.max_length = min(
            config.data_config.max_length,
            profile.recommended_max_length
        )
        
        return config
    
    def estimate_memory_usage(self, config: TrainingConfig) -> MemoryEstimate:
        """
        Estimate memory usage for a given configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            Memory usage estimate
        """
        # TinyLlama 1.1B parameter estimates
        model_params = 1.1e9  # 1.1 billion parameters
        
        # Memory calculations (rough estimates)
        bytes_per_param = 2 if config.training_params.mixed_precision in ["fp16", "bf16"] else 4
        
        # Model memory (parameters + gradients + optimizer states)
        model_memory_gb = (model_params * bytes_per_param) / (1024**3)
        optimizer_memory_gb = model_memory_gb * 2  # Adam optimizer states
        
        # Activation memory (depends on batch size and sequence length)
        batch_size = config.training_params.batch_size
        seq_length = config.data_config.max_length
        hidden_size = 2048  # TinyLlama hidden size
        num_layers = 22  # TinyLlama layers
        
        activation_memory_gb = (
            batch_size * seq_length * hidden_size * num_layers * bytes_per_param
        ) / (1024**3)
        
        # Add gradient checkpointing savings
        if config.training_params.gradient_checkpointing:
            activation_memory_gb *= 0.5  # Rough estimate of savings
        
        total_memory_gb = model_memory_gb + optimizer_memory_gb + activation_memory_gb
        
        # Check if it fits in available GPU memory
        hardware_info = self.detect_hardware()
        fits_in_memory = (
            hardware_info.gpu_count > 0 and 
            total_memory_gb <= hardware_info.gpu_memory_gb * 0.9  # 90% utilization
        )
        
        return MemoryEstimate(
            model_memory_gb=round(model_memory_gb, 2),
            optimizer_memory_gb=round(optimizer_memory_gb, 2),
            activation_memory_gb=round(activation_memory_gb, 2),
            total_memory_gb=round(total_memory_gb, 2),
            fits_in_memory=fits_in_memory
        )
    
    def validate_hardware_compatibility(self, config: TrainingConfig) -> Tuple[bool, str]:
        """
        Validate if configuration is compatible with current hardware.
        
        Args:
            config: Training configuration to validate
            
        Returns:
            Tuple of (is_compatible, error_message)
        """
        hardware_info = self.detect_hardware()
        
        # Check CUDA availability
        if config.hardware_config.device != "cpu" and hardware_info.gpu_count == 0:
            return False, "GPU training requested but no CUDA GPUs available"
        
        # Check mixed precision support
        if config.training_params.mixed_precision == "bf16":
            if hardware_info.gpu_count == 0:
                return False, "BF16 mixed precision requires GPU"
            # Check if GPU supports BF16 (Ampere and newer)
            gpu_name = hardware_info.gpu_name.lower()
            if not any(arch in gpu_name for arch in ["a100", "h100", "rtx 30", "rtx 40"]):
                return False, f"BF16 not supported on {hardware_info.gpu_name}"
        
        # Check memory requirements
        memory_estimate = self.estimate_memory_usage(config)
        if not memory_estimate.fits_in_memory and hardware_info.gpu_count > 0:
            return False, (
                f"Estimated memory usage ({memory_estimate.total_memory_gb:.1f}GB) "
                f"exceeds available GPU memory ({hardware_info.gpu_memory_gb:.1f}GB)"
            )
        
        return True, "Configuration is compatible with current hardware"
    
    def get_profile_recommendations(self, hardware_info: HardwareInfo) -> Dict[str, Any]:
        """
        Get training recommendations based on hardware.
        
        Args:
            hardware_info: Hardware information
            
        Returns:
            Dictionary with recommendations
        """
        if hardware_info.gpu_count == 0:
            return {
                "device": "cpu",
                "recommendations": [
                    "Consider using a GPU for faster training",
                    "Use very small batch sizes and long gradient accumulation",
                    "Expect significantly slower training times"
                ]
            }
        
        profile = self.get_hardware_profile(hardware_info)
        
        recommendations = []
        
        if hardware_info.gpu_memory_gb < 12:
            recommendations.extend([
                "Enable gradient checkpointing to save memory",
                "Use FP16 mixed precision",
                "Consider reducing sequence length if OOM occurs"
            ])
        
        if hardware_info.gpu_memory_gb >= 24:
            recommendations.extend([
                "You can use larger batch sizes for faster training",
                "Consider disabling gradient checkpointing for speed",
                "BF16 mixed precision may provide better stability"
            ])
        
        if hardware_info.gpu_count > 1:
            recommendations.extend([
                f"Multi-GPU training available with {hardware_info.gpu_count} GPUs",
                "Use DistributedDataParallel (DDP) for optimal performance",
                "Consider linear learning rate scaling with number of GPUs"
            ])
        
        return {
            "profile": asdict(profile),
            "recommendations": recommendations,
            "estimated_training_time": self._estimate_training_time(profile)
        }
    
    def _estimate_training_time(self, profile: HardwareProfile) -> str:
        """
        Rough estimate of training time based on hardware profile.
        
        Args:
            profile: Hardware profile
            
        Returns:
            Estimated training time string
        """
        # Very rough estimates based on profile performance
        if "H100" in profile.name:
            return "~2-4 hours for 10K steps"
        elif "A100" in profile.name:
            return "~4-8 hours for 10K steps"
        elif "RTX 4090" in profile.name:
            return "~8-12 hours for 10K steps"
        elif "RTX 4080" in profile.name:
            return "~12-16 hours for 10K steps"
        elif "RTX 4070" in profile.name:
            return "~16-24 hours for 10K steps"
        elif "RTX 4060" in profile.name or "RTX 4050" in profile.name:
            return "~24-48 hours for 10K steps"
        else:
            return "Varies significantly based on hardware"
    
    def list_available_profiles(self) -> Dict[str, str]:
        """
        List all available hardware profiles.
        
        Returns:
            Dictionary mapping profile keys to profile names
        """
        return {key: profile.name for key, profile in self._profiles.items()}