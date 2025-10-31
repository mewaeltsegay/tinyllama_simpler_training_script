"""Base classes for training components."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from ..config.base import TrainingConfig, TrainingMetrics, ValidationMetrics


class BaseModelManager(ABC):
    """Abstract base class for model management."""
    
    @abstractmethod
    def load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        pass
    
    @abstractmethod
    def load_tokenizer(self, tokenizer_path: str) -> Any:
        """Load tokenizer."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, model: torch.nn.Module, path: str, metadata: Dict[str, Any]) -> None:
        """Save model checkpoint with metadata."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load model checkpoint and metadata."""
        pass


class BaseDataPipeline(ABC):
    """Abstract base class for data pipeline."""
    
    @abstractmethod
    def load_tigrinya_dataset(self, path: str) -> Dataset:
        """Load Tigrinya dataset from JSONL file."""
        pass
    
    @abstractmethod
    def load_english_validation(self, path: str) -> Optional[Dataset]:
        """Load English validation dataset."""
        pass
    
    @abstractmethod
    def create_dataloader(self, dataset: Dataset, batch_size: int, **kwargs) -> DataLoader:
        """Create DataLoader with appropriate settings."""
        pass
    
    @abstractmethod
    def tokenize_batch(self, texts: list) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        pass


class BaseTrainingEngine(ABC):
    """Abstract base class for training engine."""
    
    @abstractmethod
    def setup_training(self, model: torch.nn.Module, config: TrainingConfig) -> None:
        """Setup training components (optimizer, scheduler, etc.)."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Execute a single training step."""
        pass
    
    @abstractmethod
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> ValidationMetrics:
        """Execute a single validation step."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, step: int, metrics: Dict[str, Any]) -> None:
        """Save training checkpoint."""
        pass


class BaseHardwareAdapter(ABC):
    """Abstract base class for hardware adaptation."""
    
    @abstractmethod
    def detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware."""
        pass
    
    @abstractmethod
    def get_optimal_config(self, hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal configuration for detected hardware."""
        pass
    
    @abstractmethod
    def estimate_memory_usage(self, config: TrainingConfig) -> Dict[str, float]:
        """Estimate memory usage for given configuration."""
        pass


class BaseMonitoringSystem(ABC):
    """Abstract base class for monitoring and logging."""
    
    @abstractmethod
    def setup_logging(self, config: TrainingConfig) -> None:
        """Setup logging configuration."""
        pass
    
    @abstractmethod
    def log_training_metrics(self, step: int, metrics: TrainingMetrics) -> None:
        """Log training metrics."""
        pass
    
    @abstractmethod
    def log_validation_metrics(self, step: int, metrics: ValidationMetrics) -> None:
        """Log validation metrics."""
        pass
    
    @abstractmethod
    def log_hardware_usage(self, gpu_memory: float, gpu_utilization: float) -> None:
        """Log hardware usage metrics."""
        pass