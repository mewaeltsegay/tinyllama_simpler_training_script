"""SentencePiece tokenizer integration for Tigrinya language processing."""

import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import sentencepiece as spm
from transformers import LlamaTokenizer, PreTrainedTokenizer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TigrinyaTokenizer:
    """Wrapper for Tigrinya SentencePiece tokenizer with batch processing utilities."""
    
    def __init__(self, tokenizer_path: str):
        """Initialize Tigrinya tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer directory containing sentencepiece.model
        """
        self.tokenizer_path = Path(tokenizer_path)
        self.sp_model = None
        self.hf_tokenizer = None
        self._vocab_size = None
        self._special_tokens = {}
        
        self._load_tokenizer()
        logger.info(f"TigrinyaTokenizer initialized from: {tokenizer_path}")
    
    def _load_tokenizer(self) -> None:
        """Load SentencePiece model and HuggingFace tokenizer."""
        # Load SentencePiece model
        sp_model_path = self.tokenizer_path / "sentencepiece.model"
        if not sp_model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(str(sp_model_path))
        self._vocab_size = self.sp_model.get_piece_size()
        
        # Load HuggingFace tokenizer if available
        try:
            self.hf_tokenizer = LlamaTokenizer.from_pretrained(
                str(self.tokenizer_path),
                legacy=False,
                use_fast=False
            )
            logger.info("HuggingFace tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace tokenizer: {e}")
            self.hf_tokenizer = None
        
        # Extract special tokens
        self._extract_special_tokens()
        
        logger.info(f"Tokenizer loaded with vocabulary size: {self._vocab_size}")
    
    def _extract_special_tokens(self) -> None:
        """Extract special tokens from the tokenizer."""
        # Common special tokens
        special_token_names = ['<unk>', '<s>', '</s>', '<pad>', '<mask>']
        
        for token_name in special_token_names:
            try:
                token_id = self.sp_model.piece_to_id(token_name)
                if token_id != self.sp_model.unk_id() or token_name == '<unk>':
                    self._special_tokens[token_name] = token_id
            except:
                pass
        
        # Add standard special token IDs
        self._special_tokens.update({
            'unk_id': self.sp_model.unk_id(),
            'bos_id': self.sp_model.bos_id(),
            'eos_id': self.sp_model.eos_id(),
            'pad_id': self.sp_model.pad_id() if hasattr(self.sp_model, 'pad_id') else 3
        })
        
        logger.info(f"Special tokens: {self._special_tokens}")
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Encode with SentencePiece
        token_ids = self.sp_model.encode_as_ids(text)
        
        # Add special tokens
        if add_bos and self.sp_model.bos_id() >= 0:
            token_ids = [self.sp_model.bos_id()] + token_ids
        
        if add_eos and self.sp_model.eos_id() >= 0:
            token_ids = token_ids + [self.sp_model.eos_id()]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not isinstance(token_ids, list):
            raise ValueError("token_ids must be a list")
        
        # Filter special tokens if requested
        if skip_special_tokens:
            filtered_ids = []
            special_ids = set(self._special_tokens.values())
            for token_id in token_ids:
                if token_id not in special_ids:
                    filtered_ids.append(token_id)
            token_ids = filtered_ids
        
        return self.sp_model.decode_ids(token_ids)
    
    def encode_batch(
        self, 
        texts: List[str], 
        add_bos: bool = True, 
        add_eos: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False
    ) -> Dict[str, List[List[int]]]:
        """Encode batch of texts to token IDs.
        
        Args:
            texts: List of input texts
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            max_length: Maximum sequence length (truncate if longer)
            padding: Whether to pad sequences to same length
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if not isinstance(texts, list):
            raise ValueError("texts must be a list")
        
        batch_ids = []
        attention_masks = []
        
        for text in texts:
            token_ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            
            # Truncate if necessary
            if max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                # Ensure EOS token at the end if it was requested
                if add_eos and self.sp_model.eos_id() >= 0:
                    token_ids[-1] = self.sp_model.eos_id()
            
            batch_ids.append(token_ids)
            attention_masks.append([1] * len(token_ids))
        
        # Padding
        if padding:
            max_len = max(len(ids) for ids in batch_ids) if batch_ids else 0
            pad_id = self._special_tokens.get('pad_id', 0)
            
            for i in range(len(batch_ids)):
                padding_length = max_len - len(batch_ids[i])
                batch_ids[i].extend([pad_id] * padding_length)
                attention_masks[i].extend([0] * padding_length)
        
        return {
            'input_ids': batch_ids,
            'attention_mask': attention_masks
        }
    
    def decode_batch(self, batch_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of token IDs to texts.
        
        Args:
            batch_ids: List of token ID lists
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            List of decoded texts
        """
        return [self.decode(token_ids, skip_special_tokens) for token_ids in batch_ids]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into pieces.
        
        Args:
            text: Input text
            
        Returns:
            List of token pieces
        """
        return self.sp_model.encode_as_pieces(text)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns:
            Vocabulary size
        """
        return self._vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special tokens mapping.
        
        Returns:
            Dictionary mapping special token names to IDs
        """
        return self._special_tokens.copy()
    
    def check_compatibility(self, model_vocab_size: int) -> Tuple[bool, str]:
        """Check compatibility with model vocabulary size.
        
        Args:
            model_vocab_size: Model's vocabulary size
            
        Returns:
            Tuple of (is_compatible, message)
        """
        if self._vocab_size == model_vocab_size:
            return True, "Tokenizer and model vocabulary sizes match"
        else:
            return False, f"Vocabulary size mismatch: tokenizer={self._vocab_size}, model={model_vocab_size}"
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get tokenizer information.
        
        Returns:
            Dictionary with tokenizer information
        """
        return {
            "tokenizer_type": "SentencePiece",
            "model_type": "unigram",
            "vocab_size": self._vocab_size,
            "special_tokens": self._special_tokens,
            "model_path": str(self.tokenizer_path / "sentencepiece.model"),
            "has_hf_tokenizer": self.hf_tokenizer is not None
        }
    
    def sample_tokenization(self, text: str, num_samples: int = 5) -> List[List[str]]:
        """Generate multiple tokenization samples (for data augmentation).
        
        Args:
            text: Input text
            num_samples: Number of samples to generate
            
        Returns:
            List of tokenization samples
        """
        samples = []
        for _ in range(num_samples):
            # Use sampling-based tokenization
            pieces = self.sp_model.sample_encode_as_pieces(text, nbest_size=-1, alpha=0.1)
            samples.append(pieces)
        return samples
    
    def get_hf_tokenizer(self) -> Optional[LlamaTokenizer]:
        """Get HuggingFace tokenizer if available.
        
        Returns:
            HuggingFace tokenizer or None
        """
        return self.hf_tokenizer


class TokenizerUtils:
    """Utility functions for tokenizer operations."""
    
    @staticmethod
    def validate_tokenizer_files(tokenizer_path: str) -> Tuple[bool, List[str]]:
        """Validate that all required tokenizer files exist.
        
        Args:
            tokenizer_path: Path to tokenizer directory
            
        Returns:
            Tuple of (is_valid, missing_files)
        """
        tokenizer_path = Path(tokenizer_path)
        required_files = [
            "sentencepiece.model",
            "sentencepiece.vocab",
            "tokenizer_config.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = tokenizer_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        is_valid = len(missing_files) == 0
        return is_valid, missing_files
    
    @staticmethod
    def compare_tokenizers(tokenizer1: TigrinyaTokenizer, tokenizer2: TigrinyaTokenizer, test_texts: List[str]) -> Dict[str, Any]:
        """Compare two tokenizers on test texts.
        
        Args:
            tokenizer1: First tokenizer
            tokenizer2: Second tokenizer
            test_texts: List of test texts
            
        Returns:
            Comparison results
        """
        results = {
            "vocab_size_match": tokenizer1.get_vocab_size() == tokenizer2.get_vocab_size(),
            "tokenization_matches": 0,
            "total_tests": len(test_texts),
            "differences": []
        }
        
        for i, text in enumerate(test_texts):
            tokens1 = tokenizer1.encode(text)
            tokens2 = tokenizer2.encode(text)
            
            if tokens1 == tokens2:
                results["tokenization_matches"] += 1
            else:
                results["differences"].append({
                    "text": text,
                    "tokenizer1": tokens1,
                    "tokenizer2": tokens2
                })
        
        results["match_rate"] = results["tokenization_matches"] / results["total_tests"]
        return results
    
    @staticmethod
    def analyze_tokenization_efficiency(tokenizer: TigrinyaTokenizer, texts: List[str]) -> Dict[str, float]:
        """Analyze tokenization efficiency metrics.
        
        Args:
            tokenizer: Tokenizer to analyze
            texts: List of texts to analyze
            
        Returns:
            Efficiency metrics
        """
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(len(tokenizer.encode(text, add_bos=False, add_eos=False)) for text in texts)
        
        return {
            "compression_ratio": total_chars / total_tokens if total_tokens > 0 else 0,
            "avg_tokens_per_text": total_tokens / len(texts) if texts else 0,
            "avg_chars_per_token": total_chars / total_tokens if total_tokens > 0 else 0,
            "total_texts": len(texts),
            "total_characters": total_chars,
            "total_tokens": total_tokens
        }
    
    @staticmethod
    def create_tokenizer_config(
        tokenizer_path: str,
        model_max_length: int = 2048,
        padding_side: str = "right",
        truncation_side: str = "right"
    ) -> None:
        """Create or update tokenizer configuration file.
        
        Args:
            tokenizer_path: Path to tokenizer directory
            model_max_length: Maximum model input length
            padding_side: Side to pad sequences ("left" or "right")
            truncation_side: Side to truncate sequences ("left" or "right")
        """
        tokenizer_path = Path(tokenizer_path)
        config_path = tokenizer_path / "tokenizer_config.json"
        
        # Load existing config or create new one
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update configuration
        config.update({
            "tokenizer_class": "LlamaTokenizer",
            "model_max_length": model_max_length,
            "padding_side": padding_side,
            "truncation_side": truncation_side,
            "clean_up_tokenization_spaces": False,
            "legacy": False,
            "use_fast": False,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>"
        })
        
        # Save updated configuration
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Tokenizer configuration saved to: {config_path}")


def load_tigrinya_tokenizer(tokenizer_path: str) -> TigrinyaTokenizer:
    """Convenience function to load Tigrinya tokenizer.
    
    Args:
        tokenizer_path: Path to tokenizer directory
        
    Returns:
        Loaded TigrinyaTokenizer
    """
    return TigrinyaTokenizer(tokenizer_path)