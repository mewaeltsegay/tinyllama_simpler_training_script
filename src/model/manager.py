"""Model manager for TinyLlama model loading and device placement."""

import os
import logging
from typing import Optional, Dict, Any, Tuple
import torch
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    LlamaConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
from ..utils.logging import get_logger
from .tokenizer import TigrinyaTokenizer, TokenizerUtils

logger = get_logger(__name__)


class ModelManager:
    """Manages TinyLlama model loading, tokenizer integration, and device placement."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize ModelManager with device configuration.
        
        Args:
            device: Target device ('cuda', 'cpu', or 'auto' for automatic detection)
        """
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup and validate device configuration.
        
        Args:
            device: Device specification or 'auto' for automatic detection
            
        Returns:
            Validated device string
        """
        if device is None or device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                logger.info("CUDA not available, using CPU")
        
        # Validate device
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA device requested but not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def load_model(self, checkpoint_path: str, config_overrides: Optional[Dict[str, Any]] = None) -> LlamaForCausalLM:
        """Load TinyLlama model from checkpoint with proper device placement.
        
        Args:
            checkpoint_path: Path to model checkpoint or HuggingFace model name
            config_overrides: Optional configuration overrides
            
        Returns:
            Loaded LlamaForCausalLM model
            
        Raises:
            FileNotFoundError: If checkpoint path doesn't exist
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading TinyLlama model from: {checkpoint_path}")
        
        try:
            # Check if path exists for local models
            if not checkpoint_path.startswith('TinyLlama') and not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            # Load model configuration
            if os.path.isdir(checkpoint_path):
                config = LlamaConfig.from_pretrained(checkpoint_path)
            else:
                # For HuggingFace model names
                config = LlamaConfig.from_pretrained(checkpoint_path)
            
            # Apply configuration overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        logger.info(f"Applied config override: {key} = {value}")
                    else:
                        logger.warning(f"Unknown config parameter: {key}")
            
            # Load model with configuration
            model = LlamaForCausalLM.from_pretrained(
                checkpoint_path,
                config=config,
                torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
                device_map='auto' if self.device.startswith('cuda') else None,
                low_cpu_mem_usage=True
            )
            
            # Move to device if not using device_map
            if not self.device.startswith('cuda') or 'device_map' not in locals():
                model = model.to(self.device)
            
            # Validate model state
            self._validate_model_state(model)
            
            self.model = model
            logger.info(f"Successfully loaded model with {model.num_parameters():,} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def load_tokenizer(self, tokenizer_path: str) -> TigrinyaTokenizer:
        """Load and configure Tigrinya SentencePiece tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer directory
            
        Returns:
            Configured TigrinyaTokenizer
            
        Raises:
            FileNotFoundError: If tokenizer path doesn't exist
            RuntimeError: If tokenizer loading fails
        """
        logger.info(f"Loading Tigrinya tokenizer from: {tokenizer_path}")
        
        try:
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
            
            # Validate tokenizer files
            is_valid, missing_files = TokenizerUtils.validate_tokenizer_files(tokenizer_path)
            if not is_valid:
                raise FileNotFoundError(f"Missing tokenizer files: {missing_files}")
            
            # Load Tigrinya tokenizer
            tokenizer = TigrinyaTokenizer(tokenizer_path)
            
            # Validate tokenizer functionality
            self._validate_tigrinya_tokenizer(tokenizer)
            
            self.tokenizer = tokenizer
            logger.info(f"Successfully loaded Tigrinya tokenizer with vocab size: {tokenizer.get_vocab_size()}")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {str(e)}")
            raise RuntimeError(f"Tokenizer loading failed: {str(e)}") from e
    
    def resize_token_embeddings(self, model: LlamaForCausalLM, tokenizer: TigrinyaTokenizer) -> None:
        """Resize model token embeddings to match tokenizer vocabulary.
        
        Args:
            model: LlamaForCausalLM model
            tokenizer: TigrinyaTokenizer
        """
        model_vocab_size = model.config.vocab_size
        tokenizer_vocab_size = tokenizer.get_vocab_size()
        
        if model_vocab_size != tokenizer_vocab_size:
            logger.info(f"Resizing token embeddings from {model_vocab_size} to {tokenizer_vocab_size}")
            model.resize_token_embeddings(tokenizer_vocab_size)
            
            # Update config
            model.config.vocab_size = tokenizer_vocab_size
            
            logger.info("Token embeddings resized successfully")
        else:
            logger.info("Token embeddings already match tokenizer vocabulary size")
    
    def validate_model_tokenizer_compatibility(self, model: LlamaForCausalLM, tokenizer: TigrinyaTokenizer) -> bool:
        """Validate compatibility between model and tokenizer.
        
        Args:
            model: LlamaForCausalLM model
            tokenizer: TigrinyaTokenizer
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Check vocabulary sizes
            model_vocab_size = model.config.vocab_size
            tokenizer_vocab_size = tokenizer.get_vocab_size()
            
            is_compatible, message = tokenizer.check_compatibility(model_vocab_size)
            if not is_compatible:
                logger.warning(message)
                return False
            
            # Test tokenization and model input
            test_texts = [
                "Hello, this is a test.",
                "ሰላም! ከመይ ኣሎኻ?",  # Tigrinya test
                "This is a mixed test ሰላም"  # Mixed language test
            ]
            
            for test_text in test_texts:
                # Encode with Tigrinya tokenizer
                token_ids = tokenizer.encode(test_text, add_bos=True, add_eos=False)
                
                # Convert to tensor and move to model device
                input_tensor = torch.tensor([token_ids]).to(model.device)
                
                # Test forward pass
                with torch.no_grad():
                    outputs = model(input_tensor)
                
                if outputs.logits is None:
                    logger.error(f"Model forward pass failed for text: {test_text}")
                    return False
            
            logger.info("Model-tokenizer compatibility validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model-tokenizer compatibility check failed: {str(e)}")
            return False
    
    def _validate_model_state(self, model: LlamaForCausalLM) -> None:
        """Validate model state and configuration.
        
        Args:
            model: Model to validate
            
        Raises:
            RuntimeError: If model validation fails
        """
        try:
            # Check if model is in expected state
            if not isinstance(model, LlamaForCausalLM):
                raise RuntimeError(f"Expected LlamaForCausalLM, got {type(model)}")
            
            # Check model parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                raise RuntimeError("Model has no parameters")
            
            # Check model device
            model_device = next(model.parameters()).device
            logger.info(f"Model loaded on device: {model_device}")
            
            # Basic forward pass test
            test_input = torch.randint(0, model.config.vocab_size, (1, 10)).to(model_device)
            with torch.no_grad():
                outputs = model(test_input)
            
            if outputs.logits is None:
                raise RuntimeError("Model forward pass failed")
            
            logger.info("Model state validation passed")
            
        except Exception as e:
            logger.error(f"Model state validation failed: {str(e)}")
            raise RuntimeError(f"Model validation failed: {str(e)}") from e
    
    def _validate_tigrinya_tokenizer(self, tokenizer: TigrinyaTokenizer) -> None:
        """Validate Tigrinya tokenizer functionality.
        
        Args:
            tokenizer: TigrinyaTokenizer to validate
            
        Raises:
            RuntimeError: If tokenizer validation fails
        """
        try:
            # Test basic tokenization with English
            test_text_en = "Hello, this is a test."
            tokens_en = tokenizer.encode(test_text_en, add_bos=False, add_eos=False)
            decoded_en = tokenizer.decode(tokens_en)
            
            if not tokens_en or len(tokens_en) == 0:
                raise RuntimeError("Tokenizer failed to encode English text")
            
            # Test Tigrinya tokenization
            test_text_ti = "ሰላም! ከመይ ኣሎኻ?"
            tokens_ti = tokenizer.encode(test_text_ti, add_bos=False, add_eos=False)
            decoded_ti = tokenizer.decode(tokens_ti)
            
            if not tokens_ti or len(tokens_ti) == 0:
                raise RuntimeError("Tokenizer failed to encode Tigrinya text")
            
            # Test batch processing
            batch_texts = [test_text_en, test_text_ti]
            batch_result = tokenizer.encode_batch(batch_texts, padding=True)
            
            if 'input_ids' not in batch_result or 'attention_mask' not in batch_result:
                raise RuntimeError("Batch tokenization failed")
            
            # Check special tokens
            special_tokens = tokenizer.get_special_tokens()
            required_tokens = ['unk_id', 'bos_id', 'eos_id']
            
            for token_name in required_tokens:
                if token_name not in special_tokens:
                    logger.warning(f"Special token {token_name} not found")
            
            # Validate vocabulary size
            vocab_size = tokenizer.get_vocab_size()
            if vocab_size <= 0:
                raise RuntimeError("Invalid vocabulary size")
            
            logger.info(f"Tigrinya tokenizer validation passed (vocab_size: {vocab_size})")
            
        except Exception as e:
            logger.error(f"Tigrinya tokenizer validation failed: {str(e)}")
            raise RuntimeError(f"Tigrinya tokenizer validation failed: {str(e)}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_type": type(self.model).__name__,
            "vocab_size": self.model.config.vocab_size,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_hidden_layers,
            "num_attention_heads": self.model.config.num_attention_heads,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype)
        }
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get information about the loaded tokenizer.
        
        Returns:
            Dictionary containing tokenizer information
        """
        if self.tokenizer is None:
            return {"status": "No tokenizer loaded"}
        
        return self.tokenizer.get_tokenizer_info()