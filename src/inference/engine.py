"""Bilingual text generation inference engine for Tigrinya-English TinyLlama model."""

import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple
from transformers import LlamaForCausalLM, GenerationConfig
from ..model.manager import ModelManager
from ..model.tokenizer import TigrinyaTokenizer
from ..utils.logging import get_logger
from .base import BaseInferenceEngine, BaseLanguageDetector

logger = get_logger(__name__)


class LanguageDetector(BaseLanguageDetector):
    """Language detection for Tigrinya and English text."""
    
    def __init__(self):
        """Initialize language detector."""
        # Tigrinya Unicode ranges and common patterns
        self.tigrinya_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
        ]
        
        # Common Tigrinya words and patterns
        self.tigrinya_patterns = [
            r'[ሀ-ሿ]',  # Ethiopic syllables
            r'[ጀ-ጿ]',  # Ethiopic syllables
            r'[ፀ-ፚ]',  # Ethiopic syllables
        ]
        
        # Common English patterns
        self.english_patterns = [
            r'[a-zA-Z]',  # Latin letters
            r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b',  # Common English words
        ]
        
        logger.info("LanguageDetector initialized")
    
    def detect_language(self, text: str) -> str:
        """Detect the primary language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            'tigrinya', 'english', or 'mixed'
        """
        if not text or not text.strip():
            return 'unknown'
        
        tigrinya_score = self._calculate_tigrinya_score(text)
        english_score = self._calculate_english_score(text)
        
        # Determine primary language based on scores
        if tigrinya_score > english_score * 1.5:
            return 'tigrinya'
        elif english_score > tigrinya_score * 1.5:
            return 'english'
        else:
            return 'mixed'
    
    def is_tigrinya(self, text: str) -> bool:
        """Check if text is primarily in Tigrinya.
        
        Args:
            text: Input text
            
        Returns:
            True if text is primarily Tigrinya
        """
        return self.detect_language(text) == 'tigrinya'
    
    def is_english(self, text: str) -> bool:
        """Check if text is primarily in English.
        
        Args:
            text: Input text
            
        Returns:
            True if text is primarily English
        """
        return self.detect_language(text) == 'english'
    
    def _calculate_tigrinya_score(self, text: str) -> float:
        """Calculate Tigrinya language score for text.
        
        Args:
            text: Input text
            
        Returns:
            Tigrinya score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        tigrinya_chars = 0
        total_chars = len(text)
        
        for char in text:
            char_code = ord(char)
            for start, end in self.tigrinya_ranges:
                if start <= char_code <= end:
                    tigrinya_chars += 1
                    break
        
        # Pattern-based scoring
        pattern_score = 0
        for pattern in self.tigrinya_patterns:
            matches = len(re.findall(pattern, text))
            pattern_score += matches
        
        # Combine character-based and pattern-based scores
        char_score = tigrinya_chars / total_chars if total_chars > 0 else 0
        normalized_pattern_score = min(pattern_score / total_chars, 0.5) if total_chars > 0 else 0
        
        return char_score + normalized_pattern_score
    
    def _calculate_english_score(self, text: str) -> float:
        """Calculate English language score for text.
        
        Args:
            text: Input text
            
        Returns:
            English score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        # Count Latin characters
        latin_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        total_chars = len(text)
        
        # Pattern-based scoring
        pattern_score = 0
        for pattern in self.english_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            pattern_score += matches
        
        # Combine scores
        char_score = latin_chars / total_chars if total_chars > 0 else 0
        normalized_pattern_score = min(pattern_score / total_chars, 0.3) if total_chars > 0 else 0
        
        return char_score + normalized_pattern_score


class BilingualInferenceEngine(BaseInferenceEngine):
    """Bilingual text generation inference engine for Tigrinya-English model."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize bilingual inference engine.
        
        Args:
            device: Target device ('cuda', 'cpu', or 'auto')
        """
        self.model_manager = ModelManager(device)
        self.language_detector = LanguageDetector()
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Default generation parameters
        self.default_params = {
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.9,
            'max_length': 100,
            'min_length': 1,
            'do_sample': True,
            'num_return_sequences': 1,
            'repetition_penalty': 1.1,
            'length_penalty': 1.0,
            'early_stopping': True,
            'pad_token_id': None,  # Will be set from tokenizer
            'eos_token_id': None,  # Will be set from tokenizer
            'bos_token_id': None,  # Will be set from tokenizer
        }
        
        logger.info(f"BilingualInferenceEngine initialized on device: {self.model_manager.device}")
    
    def load_trained_model(self, checkpoint_path: str, tokenizer_path: str) -> None:
        """Load trained model and tokenizer from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer directory
            
        Raises:
            RuntimeError: If model or tokenizer loading fails
        """
        logger.info(f"Loading trained model from: {checkpoint_path}")
        
        try:
            # Load tokenizer first
            self.tokenizer = self.model_manager.load_tokenizer(tokenizer_path)
            
            # Load model
            self.model = self.model_manager.load_model(checkpoint_path)
            
            # Ensure model-tokenizer compatibility
            if not self.model_manager.validate_model_tokenizer_compatibility(self.model, self.tokenizer):
                logger.warning("Model-tokenizer compatibility issues detected")
            
            # Set up generation configuration
            self._setup_generation_config()
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def _setup_generation_config(self) -> None:
        """Setup generation configuration with tokenizer special tokens."""
        special_tokens = self.tokenizer.get_special_tokens()
        
        # Update default parameters with special tokens
        self.default_params.update({
            'pad_token_id': special_tokens.get('pad_id', 0),
            'eos_token_id': special_tokens.get('eos_id', 2),
            'bos_token_id': special_tokens.get('bos_id', 1),
        })
        
        # Create generation configuration
        self.generation_config = GenerationConfig(**self.default_params)
        
        logger.info("Generation configuration setup complete")
    
    def configure_generation(
        self, 
        temperature: float = 1.0, 
        top_k: int = 50, 
        top_p: float = 0.9,
        max_length: int = 100,
        min_length: int = 1,
        repetition_penalty: float = 1.1,
        length_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        early_stopping: bool = True,
        **kwargs
    ) -> None:
        """Configure generation parameters.
        
        Args:
            temperature: Sampling temperature (0.1 to 2.0)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            max_length: Maximum generation length
            min_length: Minimum generation length
            repetition_penalty: Repetition penalty factor
            length_penalty: Length penalty factor
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            early_stopping: Whether to stop early when EOS is generated
            **kwargs: Additional generation parameters
        """
        # Validate parameters
        temperature = max(0.1, min(2.0, temperature))
        top_k = max(1, min(100, top_k))
        top_p = max(0.1, min(1.0, top_p))
        max_length = max(1, min(2048, max_length))
        min_length = max(1, min(max_length, min_length))
        repetition_penalty = max(1.0, min(2.0, repetition_penalty))
        length_penalty = max(0.1, min(2.0, length_penalty))
        num_return_sequences = max(1, min(10, num_return_sequences))
        
        # Update generation configuration
        generation_params = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'max_length': max_length,
            'min_length': min_length,
            'repetition_penalty': repetition_penalty,
            'length_penalty': length_penalty,
            'do_sample': do_sample,
            'num_return_sequences': num_return_sequences,
            'early_stopping': early_stopping,
            **kwargs
        }
        
        # Preserve special token IDs
        generation_params.update({
            'pad_token_id': self.default_params['pad_token_id'],
            'eos_token_id': self.default_params['eos_token_id'],
            'bos_token_id': self.default_params['bos_token_id'],
        })
        
        self.generation_config = GenerationConfig(**generation_params)
        
        logger.info(f"Generation parameters updated: temp={temperature}, top_k={top_k}, top_p={top_p}")
    
    def generate_text(self, prompt: str, **generation_kwargs) -> str:
        """Generate text from a single prompt.
        
        Args:
            prompt: Input text prompt
            **generation_kwargs: Override generation parameters
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation")
        
        try:
            # Detect input language for context
            detected_language = self.language_detector.detect_language(prompt)
            logger.debug(f"Detected language: {detected_language} for prompt: {prompt[:50]}...")
            
            # Preprocess prompt
            processed_prompt = self._preprocess_prompt(prompt, detected_language)
            
            # Tokenize input
            input_ids = self.tokenizer.encode(processed_prompt, add_bos=True, add_eos=False)
            input_tensor = torch.tensor([input_ids]).to(self.model.device)
            
            # Create generation config with overrides
            gen_config = self._create_generation_config(**generation_kwargs)
            
            # Generate text
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_tensor,
                    generation_config=gen_config,
                    attention_mask=torch.ones_like(input_tensor)
                )
            
            # Decode generated text
            generated_ids = output_ids[0][len(input_ids):].tolist()  # Remove input tokens
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Post-process output
            final_text = self._postprocess_output(generated_text, detected_language)
            
            logger.debug(f"Generated text: {final_text[:100]}...")
            return final_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}") from e
    
    def batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Generate text from multiple prompts.
        
        Args:
            prompts: List of input text prompts
            **generation_kwargs: Override generation parameters
            
        Returns:
            List of generated texts
            
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generation")
        
        if not prompts:
            return []
        
        try:
            # Process each prompt individually for now (can be optimized for true batching)
            results = []
            for prompt in prompts:
                generated_text = self.generate_text(prompt, **generation_kwargs)
                results.append(generated_text)
            
            logger.info(f"Batch generation completed for {len(prompts)} prompts")
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            raise RuntimeError(f"Batch generation failed: {str(e)}") from e
    
    def _preprocess_prompt(self, prompt: str, detected_language: str) -> str:
        """Preprocess prompt based on detected language.
        
        Args:
            prompt: Input prompt
            detected_language: Detected language
            
        Returns:
            Preprocessed prompt
        """
        # Basic preprocessing
        processed = prompt.strip()
        
        # Language-specific preprocessing
        if detected_language == 'tigrinya':
            # Tigrinya-specific preprocessing
            processed = self._preprocess_tigrinya_text(processed)
        elif detected_language == 'english':
            # English-specific preprocessing
            processed = self._preprocess_english_text(processed)
        
        return processed
    
    def _preprocess_tigrinya_text(self, text: str) -> str:
        """Preprocess Tigrinya text.
        
        Args:
            text: Tigrinya text
            
        Returns:
            Preprocessed text
        """
        # Basic Tigrinya text normalization
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize punctuation spacing
        text = re.sub(r'\s*([።፣፤፥፦፧፨])\s*', r'\1 ', text)
        
        return text.strip()
    
    def _preprocess_english_text(self, text: str) -> str:
        """Preprocess English text.
        
        Args:
            text: English text
            
        Returns:
            Preprocessed text
        """
        # Basic English text normalization
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize punctuation spacing
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def _postprocess_output(self, generated_text: str, input_language: str) -> str:
        """Post-process generated output.
        
        Args:
            generated_text: Generated text
            input_language: Detected input language
            
        Returns:
            Post-processed text
        """
        # Basic post-processing
        processed = generated_text.strip()
        
        # Remove incomplete sentences at the end
        if input_language == 'tigrinya':
            # For Tigrinya, look for sentence-ending punctuation
            sentences = re.split(r'[።]', processed)
            if len(sentences) > 1 and sentences[-1].strip() and not sentences[-1].strip().endswith(('።', '፣', '፤')):
                processed = '።'.join(sentences[:-1]) + '።'
        else:
            # For English, look for sentence-ending punctuation
            sentences = re.split(r'[.!?]', processed)
            if len(sentences) > 1 and sentences[-1].strip() and not sentences[-1].strip().endswith(('.', '!', '?')):
                processed = '.'.join(sentences[:-1]) + '.'
        
        # Clean up extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        return processed.strip()
    
    def _create_generation_config(self, **kwargs) -> GenerationConfig:
        """Create generation config with parameter overrides.
        
        Args:
            **kwargs: Parameter overrides
            
        Returns:
            GenerationConfig object
        """
        # Start with current configuration
        config_dict = self.generation_config.to_dict()
        
        # Apply overrides
        config_dict.update(kwargs)
        
        # Validate critical parameters
        if 'max_length' in kwargs:
            config_dict['max_length'] = max(1, min(2048, kwargs['max_length']))
        
        if 'temperature' in kwargs:
            config_dict['temperature'] = max(0.1, min(2.0, kwargs['temperature']))
        
        if 'top_k' in kwargs:
            config_dict['top_k'] = max(1, min(100, kwargs['top_k']))
        
        if 'top_p' in kwargs:
            config_dict['top_p'] = max(0.1, min(1.0, kwargs['top_p']))
        
        return GenerationConfig(**config_dict)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        model_info = self.model_manager.get_model_info()
        tokenizer_info = self.model_manager.get_tokenizer_info()
        
        return {
            "model": model_info,
            "tokenizer": tokenizer_info,
            "generation_config": self.generation_config.to_dict() if self.generation_config else None,
            "device": self.model_manager.device
        }
    
    def get_generation_stats(self, prompt: str, generated_text: str) -> Dict[str, Any]:
        """Get statistics about the generation process.
        
        Args:
            prompt: Input prompt
            generated_text: Generated text
            
        Returns:
            Generation statistics
        """
        prompt_tokens = len(self.tokenizer.encode(prompt, add_bos=False, add_eos=False))
        generated_tokens = len(self.tokenizer.encode(generated_text, add_bos=False, add_eos=False))
        
        input_language = self.language_detector.detect_language(prompt)
        output_language = self.language_detector.detect_language(generated_text)
        
        return {
            "prompt_length": len(prompt),
            "generated_length": len(generated_text),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": prompt_tokens + generated_tokens,
            "input_language": input_language,
            "output_language": output_language,
            "language_consistency": input_language == output_language
        }


def create_inference_engine(device: Optional[str] = None) -> BilingualInferenceEngine:
    """Factory function to create bilingual inference engine.
    
    Args:
        device: Target device ('cuda', 'cpu', or 'auto')
        
    Returns:
        BilingualInferenceEngine instance
    """
    return BilingualInferenceEngine(device)