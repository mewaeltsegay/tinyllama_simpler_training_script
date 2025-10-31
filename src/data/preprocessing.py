"""Text preprocessing and dataset sampling utilities."""

import re
import random
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import unicodedata
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities for Tigrinya and mixed-language data."""
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_extra_whitespace: bool = True,
        min_length: int = 10,
        max_length: int = 10000
    ):
        """Initialize text preprocessor.
        
        Args:
            normalize_unicode: Whether to normalize Unicode characters
            remove_extra_whitespace: Whether to clean up whitespace
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep
        """
        self.normalize_unicode = normalize_unicode
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        self.max_length = max_length
        
        # Tigrinya Unicode ranges
        self.tigrinya_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F),  # Ethiopic Extended-A
        ]
        
        logger.info("TextPreprocessor initialized")
    
    def preprocess_text(self, text: str) -> Optional[str]:
        """Preprocess a single text sample.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text or None if text should be filtered out
        """
        if not isinstance(text, str):
            return None
        
        # Basic cleaning
        text = text.strip()
        if not text:
            return None
        
        # Length filtering
        if len(text) < self.min_length or len(text) > self.max_length:
            return None
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        # Final length check after cleaning
        if len(text.strip()) < self.min_length:
            return None
        
        return text
    
    def detect_language(self, text: str) -> str:
        """Detect if text contains Tigrinya characters.
        
        Args:
            text: Input text
            
        Returns:
            'tigrinya', 'english', or 'mixed'
        """
        if not text:
            return 'unknown'
        
        tigrinya_chars = 0
        english_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                char_code = ord(char)
                
                # Check if character is in Tigrinya ranges
                is_tigrinya = any(start <= char_code <= end for start, end in self.tigrinya_ranges)
                
                if is_tigrinya:
                    tigrinya_chars += 1
                elif char.isascii():
                    english_chars += 1
        
        if total_chars == 0:
            return 'unknown'
        
        tigrinya_ratio = tigrinya_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if tigrinya_ratio > 0.5:
            return 'tigrinya'
        elif english_ratio > 0.5:
            return 'english'
        elif tigrinya_ratio > 0.1 and english_ratio > 0.1:
            return 'mixed'
        else:
            return 'unknown'
    
    def filter_by_language(self, texts: List[str], target_language: str) -> List[str]:
        """Filter texts by detected language.
        
        Args:
            texts: List of input texts
            target_language: Target language ('tigrinya', 'english', 'mixed')
            
        Returns:
            Filtered list of texts
        """
        filtered_texts = []
        
        for text in texts:
            detected_lang = self.detect_language(text)
            if detected_lang == target_language or target_language == 'all':
                filtered_texts.append(text)
        
        logger.info(f"Filtered {len(filtered_texts)} texts for language: {target_language}")
        return filtered_texts
    
    def get_text_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Get statistics about text data.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with text statistics
        """
        if not texts:
            return {"error": "No texts provided"}
        
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Language detection
        language_counts = {'tigrinya': 0, 'english': 0, 'mixed': 0, 'unknown': 0}
        for text in texts:
            lang = self.detect_language(text)
            language_counts[lang] += 1
        
        return {
            "total_texts": len(texts),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "language_distribution": language_counts,
            "total_characters": sum(lengths)
        }
    
    def clean_batch(self, texts: List[str]) -> List[Optional[str]]:
        """Clean a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of cleaned texts (None for filtered out texts)
        """
        return [self.preprocess_text(text) for text in texts]


class DatasetSampler:
    """Utilities for sampling and debugging dataset operations."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize dataset sampler.
        
        Args:
            random_seed: Random seed for reproducible sampling
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        logger.info(f"DatasetSampler initialized with seed: {random_seed}")
    
    def sample_dataset(
        self, 
        data: List[Dict[str, Any]], 
        sample_size: int,
        sampling_strategy: str = 'random'
    ) -> List[Dict[str, Any]]:
        """Sample data for debug or testing purposes.
        
        Args:
            data: List of data samples
            sample_size: Number of samples to select
            sampling_strategy: 'random', 'first', 'last', or 'distributed'
            
        Returns:
            Sampled data
        """
        if sample_size >= len(data):
            logger.info(f"Sample size ({sample_size}) >= data size ({len(data)}), returning all data")
            return data
        
        if sampling_strategy == 'random':
            sampled_data = random.sample(data, sample_size)
        elif sampling_strategy == 'first':
            sampled_data = data[:sample_size]
        elif sampling_strategy == 'last':
            sampled_data = data[-sample_size:]
        elif sampling_strategy == 'distributed':
            # Sample evenly distributed across the dataset
            step = len(data) / sample_size
            indices = [int(i * step) for i in range(sample_size)]
            sampled_data = [data[i] for i in indices]
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        logger.info(f"Sampled {len(sampled_data)} samples using '{sampling_strategy}' strategy")
        return sampled_data
    
    def create_debug_subset(
        self,
        data: List[Dict[str, Any]],
        debug_samples: int,
        preserve_distribution: bool = True
    ) -> List[Dict[str, Any]]:
        """Create a debug subset that preserves data characteristics.
        
        Args:
            data: Original data
            debug_samples: Number of samples for debug subset
            preserve_distribution: Whether to preserve language/length distribution
            
        Returns:
            Debug subset
        """
        if debug_samples >= len(data):
            return data
        
        if not preserve_distribution:
            return self.sample_dataset(data, debug_samples, 'random')
        
        # Analyze data distribution
        preprocessor = TextPreprocessor()
        
        # Group by language
        language_groups = {'tigrinya': [], 'english': [], 'mixed': [], 'unknown': []}
        
        for sample in data:
            text = sample.get('text', '')
            lang = preprocessor.detect_language(text)
            language_groups[lang].append(sample)
        
        # Calculate proportional samples per language
        debug_subset = []
        total_samples = len(data)
        
        for lang, samples in language_groups.items():
            if not samples:
                continue
            
            # Calculate proportional sample size
            proportion = len(samples) / total_samples
            lang_debug_samples = max(1, int(debug_samples * proportion))
            
            # Don't exceed available samples
            lang_debug_samples = min(lang_debug_samples, len(samples))
            
            # Sample from this language group
            lang_subset = random.sample(samples, lang_debug_samples)
            debug_subset.extend(lang_subset)
        
        # If we haven't reached the target, fill with random samples
        if len(debug_subset) < debug_samples:
            remaining_samples = debug_samples - len(debug_subset)
            remaining_data = [s for s in data if s not in debug_subset]
            if remaining_data:
                additional_samples = random.sample(
                    remaining_data, 
                    min(remaining_samples, len(remaining_data))
                )
                debug_subset.extend(additional_samples)
        
        # Shuffle the final subset
        random.shuffle(debug_subset)
        
        logger.info(f"Created debug subset with {len(debug_subset)} samples preserving distribution")
        return debug_subset
    
    def validate_sample_quality(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of sampled data.
        
        Args:
            samples: List of data samples
            
        Returns:
            Quality validation results
        """
        if not samples:
            return {"error": "No samples provided"}
        
        preprocessor = TextPreprocessor()
        
        # Extract texts
        texts = [sample.get('text', '') for sample in samples]
        
        # Get text statistics
        text_stats = preprocessor.get_text_statistics(texts)
        
        # Check for duplicates
        unique_texts = set(texts)
        duplicate_count = len(texts) - len(unique_texts)
        
        # Check for empty or invalid texts
        empty_count = sum(1 for text in texts if not text.strip())
        
        return {
            "total_samples": len(samples),
            "unique_samples": len(unique_texts),
            "duplicate_count": duplicate_count,
            "empty_count": empty_count,
            "text_statistics": text_stats,
            "quality_score": (len(unique_texts) - empty_count) / len(samples) if samples else 0
        }
    
    def create_balanced_sample(
        self,
        datasets: List[List[Dict[str, Any]]],
        sample_size: int,
        balance_ratios: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Create a balanced sample from multiple datasets.
        
        Args:
            datasets: List of datasets to sample from
            sample_size: Total number of samples
            balance_ratios: Ratios for balancing (None for equal balance)
            
        Returns:
            Balanced sample
        """
        if not datasets:
            return []
        
        if balance_ratios is None:
            balance_ratios = [1.0 / len(datasets)] * len(datasets)
        
        if len(balance_ratios) != len(datasets):
            raise ValueError("Number of balance ratios must match number of datasets")
        
        if abs(sum(balance_ratios) - 1.0) > 1e-6:
            raise ValueError("Balance ratios must sum to 1.0")
        
        balanced_sample = []
        
        for dataset, ratio in zip(datasets, balance_ratios):
            if not dataset:
                continue
            
            # Calculate samples for this dataset
            dataset_samples = int(sample_size * ratio)
            dataset_samples = min(dataset_samples, len(dataset))
            
            # Sample from dataset
            if dataset_samples > 0:
                sampled = random.sample(dataset, dataset_samples)
                balanced_sample.extend(sampled)
        
        # Shuffle the final balanced sample
        random.shuffle(balanced_sample)
        
        logger.info(f"Created balanced sample with {len(balanced_sample)} samples from {len(datasets)} datasets")
        return balanced_sample


def create_debug_sample(
    data_path: str,
    output_path: str,
    sample_size: int = 1000,
    sampling_strategy: str = 'distributed'
) -> None:
    """Create a debug sample file from a larger dataset.
    
    Args:
        data_path: Path to original JSONL file
        output_path: Path to save debug sample
        sample_size: Number of samples to include
        sampling_strategy: Sampling strategy to use
    """
    import json
    
    # Load original data
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Create sampler and sample data
    sampler = DatasetSampler()
    sampled_data = sampler.sample_dataset(data, sample_size, sampling_strategy)
    
    # Save debug sample
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in sampled_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Created debug sample with {len(sampled_data)} samples at {output_path}")


def analyze_dataset_languages(data_path: str) -> Dict[str, Any]:
    """Analyze language distribution in a dataset.
    
    Args:
        data_path: Path to JSONL dataset file
        
    Returns:
        Language analysis results
    """
    import json
    
    preprocessor = TextPreprocessor()
    
    # Load and analyze data
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    if 'text' in sample:
                        texts.append(sample['text'])
                except json.JSONDecodeError:
                    continue
    
    # Get statistics
    stats = preprocessor.get_text_statistics(texts)
    
    logger.info(f"Analyzed {len(texts)} texts from {data_path}")
    return stats