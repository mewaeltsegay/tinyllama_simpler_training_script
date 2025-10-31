"""Base classes for inference components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch


class BaseInferenceEngine(ABC):
    """Abstract base class for inference engine."""
    
    @abstractmethod
    def load_trained_model(self, checkpoint_path: str) -> None:
        """Load trained model from checkpoint."""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, **generation_kwargs) -> str:
        """Generate text from a single prompt."""
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Generate text from multiple prompts."""
        pass
    
    @abstractmethod
    def configure_generation(self, temperature: float = 1.0, top_k: int = 50, 
                           max_length: int = 100, **kwargs) -> None:
        """Configure generation parameters."""
        pass


class BaseLanguageDetector(ABC):
    """Abstract base class for language detection."""
    
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        pass
    
    @abstractmethod
    def is_tigrinya(self, text: str) -> bool:
        """Check if text is in Tigrinya."""
        pass
    
    @abstractmethod
    def is_english(self, text: str) -> bool:
        """Check if text is in English."""
        pass