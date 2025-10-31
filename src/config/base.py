"""Base configuration classes and data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


@dataclass
class ModelConfig:
    """Configuration for model loading and setup."""
    checkpoint_path: str
    tokenizer_path: str
    vocab_size: int = 32000


@dataclass
class TrainingParams:
    """Training hyperparameters."""
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 10000
    warmup_steps: int = 1000
    save_steps: int = 500
    eval_steps: int = 100
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    tigrinya_dataset: str
    validation_dataset: str
    english_validation: Optional[str] = None
    max_length: int = 2048
    debug_samples: Optional[int] = None


@dataclass
class HardwareConfig:
    """Hardware-specific configuration."""
    device: str = "auto"
    num_gpus: int = 1
    dataloader_workers: int = 4
    pin_memory: bool = True


@dataclass
class KnowledgePreservationConfig:
    """Configuration for knowledge preservation techniques."""
    enabled: bool = True
    english_weight: float = 0.3
    regularization_strength: float = 0.01
    validation_frequency: int = 100


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    tensorboard_dir: str = "logs/"
    save_metrics: bool = True


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    model_config: ModelConfig
    training_params: TrainingParams
    data_config: DataConfig
    hardware_config: HardwareConfig
    knowledge_preservation: KnowledgePreservationConfig
    logging_config: LoggingConfig


@dataclass
class HardwareInfo:
    """Information about available hardware."""
    gpu_name: str
    gpu_memory_gb: float
    gpu_count: int
    cpu_cores: int
    system_memory_gb: float
    cuda_version: str


@dataclass
class TrainingMetrics:
    """Training step metrics."""
    step: int
    loss: float
    learning_rate: float
    gpu_memory_used: float
    tokens_per_second: float
    tigrinya_perplexity: float
    english_perplexity: Optional[float]
    gradient_norm: float


@dataclass
class ValidationMetrics:
    """Validation metrics."""
    step: int
    tigrinya_loss: float
    tigrinya_perplexity: float
    english_loss: Optional[float] = None
    english_perplexity: Optional[float] = None


@dataclass
class MemoryEstimate:
    """Memory usage estimation."""
    model_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    total_memory_gb: float
    fits_in_memory: bool


class BaseConfigManager(ABC):
    """Abstract base class for configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> TrainingConfig:
        """Load configuration from JSON file."""
        pass
    
    @abstractmethod
    def validate_config(self, config: TrainingConfig) -> bool:
        """Validate configuration parameters."""
        pass
    
    @abstractmethod
    def save_config(self, config: TrainingConfig, config_path: str) -> None:
        """Save configuration to JSON file."""
        pass