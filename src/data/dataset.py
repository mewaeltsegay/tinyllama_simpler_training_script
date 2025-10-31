"""Dataset classes for Tigrinya and mixed-language training data."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union
import torch
from torch.utils.data import Dataset
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TigrinyaDataset(Dataset):
    """Dataset for loading and processing Tigrinya JSONL data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        debug_samples: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True
    ):
        """Initialize Tigrinya dataset.
        
        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer instance (TigrinyaTokenizer)
            max_length: Maximum sequence length
            debug_samples: Number of samples for debug mode (None for all)
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug_samples = debug_samples
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Load and validate data
        self.data = self._load_data()
        self.language = "tigrinya"
        
        logger.info(f"TigrinyaDataset initialized with {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file with error handling.
        
        Returns:
            List of data samples
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        data = []
        line_count = 0
        error_count = 0
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON line
                        sample = json.loads(line)
                        
                        # Validate required fields
                        if 'text' not in sample:
                            logger.warning(f"Line {line_num}: Missing 'text' field, skipping")
                            error_count += 1
                            continue
                        
                        # Validate text content
                        text = sample['text']
                        if not isinstance(text, str) or not text.strip():
                            logger.warning(f"Line {line_num}: Invalid or empty text, skipping")
                            error_count += 1
                            continue
                        
                        # Add metadata
                        sample['line_number'] = line_num
                        sample['source_file'] = str(self.data_path)
                        
                        data.append(sample)
                        line_count += 1
                        
                        # Apply debug sampling limit
                        if self.debug_samples and len(data) >= self.debug_samples:
                            logger.info(f"Debug mode: Limited to {self.debug_samples} samples")
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: JSON decode error - {e}, skipping")
                        error_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Line {line_num}: Unexpected error - {e}, skipping")
                        error_count += 1
                        continue
        
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.data_path}: {e}")
            raise ValueError(f"Dataset loading failed: {e}") from e
        
        if not data:
            raise ValueError(f"No valid samples found in {self.data_path}")
        
        logger.info(f"Loaded {len(data)} valid samples, skipped {error_count} invalid lines")
        return data
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing tokenized sample
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        sample = self.data[idx]
        text = sample['text']
        
        # Tokenize text
        try:
            token_ids = self.tokenizer.encode(
                text, 
                add_bos=self.add_bos, 
                add_eos=self.add_eos
            )
            
            # Truncate if necessary
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
                # Ensure EOS token at the end if it was requested
                if self.add_eos and self.tokenizer.sp_model.eos_id() >= 0:
                    token_ids[-1] = self.tokenizer.sp_model.eos_id()
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(token_ids, dtype=torch.long),  # For causal LM
                'text': text,
                'language': self.language,
                'metadata': {
                    'line_number': sample.get('line_number'),
                    'source_file': sample.get('source_file'),
                    'original_length': len(text),
                    'token_count': len(token_ids)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to tokenize sample {idx}: {e}")
            # Return a minimal valid sample to avoid breaking training
            fallback_tokens = [self.tokenizer.sp_model.unk_id()]
            return {
                'input_ids': torch.tensor(fallback_tokens, dtype=torch.long),
                'attention_mask': torch.tensor([1], dtype=torch.long),
                'labels': torch.tensor(fallback_tokens, dtype=torch.long),
                'text': "<UNK>",
                'language': self.language,
                'metadata': {
                    'line_number': sample.get('line_number'),
                    'source_file': sample.get('source_file'),
                    'original_length': 0,
                    'token_count': 1,
                    'error': str(e)
                }
            }
    
    def get_sample_texts(self, num_samples: int = 5) -> List[str]:
        """Get sample texts for inspection.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of sample texts
        """
        samples = min(num_samples, len(self.data))
        return [self.data[i]['text'] for i in range(samples)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.data:
            return {"error": "No data loaded"}
        
        text_lengths = [len(sample['text']) for sample in self.data]
        
        return {
            "total_samples": len(self.data),
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "total_characters": sum(text_lengths),
            "language": self.language,
            "source_file": str(self.data_path),
            "debug_mode": self.debug_samples is not None,
            "debug_samples": self.debug_samples
        }


class MixedLanguageDataset(Dataset):
    """Dataset that combines multiple language datasets for mixed training."""
    
    def __init__(
        self,
        datasets: List[Dataset],
        mixing_ratios: Optional[List[float]] = None,
        shuffle_seed: int = 42
    ):
        """Initialize mixed language dataset.
        
        Args:
            datasets: List of language-specific datasets
            mixing_ratios: Ratios for mixing datasets (None for equal mixing)
            shuffle_seed: Random seed for shuffling
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        
        self.datasets = datasets
        self.mixing_ratios = mixing_ratios or [1.0 / len(datasets)] * len(datasets)
        self.shuffle_seed = shuffle_seed
        
        # Validate mixing ratios
        if len(self.mixing_ratios) != len(self.datasets):
            raise ValueError("Number of mixing ratios must match number of datasets")
        
        if abs(sum(self.mixing_ratios) - 1.0) > 1e-6:
            raise ValueError("Mixing ratios must sum to 1.0")
        
        # Create mixed indices
        self.mixed_indices = self._create_mixed_indices()
        
        logger.info(f"MixedLanguageDataset initialized with {len(self.datasets)} datasets, "
                   f"total samples: {len(self.mixed_indices)}")
    
    def _create_mixed_indices(self) -> List[tuple]:
        """Create mixed indices based on mixing ratios.
        
        Returns:
            List of (dataset_idx, sample_idx) tuples
        """
        import random
        random.seed(self.shuffle_seed)
        
        # Calculate target samples per dataset
        total_samples = sum(len(dataset) for dataset in self.datasets)
        target_samples = [int(ratio * total_samples) for ratio in self.mixing_ratios]
        
        # Adjust for rounding errors
        diff = total_samples - sum(target_samples)
        if diff > 0:
            target_samples[0] += diff
        
        mixed_indices = []
        
        for dataset_idx, (dataset, target_count) in enumerate(zip(self.datasets, target_samples)):
            dataset_size = len(dataset)
            
            if target_count <= dataset_size:
                # Sample without replacement
                sample_indices = random.sample(range(dataset_size), target_count)
            else:
                # Sample with replacement
                sample_indices = random.choices(range(dataset_size), k=target_count)
            
            for sample_idx in sample_indices:
                mixed_indices.append((dataset_idx, sample_idx))
        
        # Shuffle the mixed indices
        random.shuffle(mixed_indices)
        
        return mixed_indices
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.mixed_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a mixed sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample from one of the datasets
        """
        if idx >= len(self.mixed_indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.mixed_indices)}")
        
        dataset_idx, sample_idx = self.mixed_indices[idx]
        sample = self.datasets[dataset_idx][sample_idx]
        
        # Add mixing metadata
        sample['metadata']['dataset_idx'] = dataset_idx
        sample['metadata']['mixed_idx'] = idx
        
        return sample
    
    def get_language_distribution(self) -> Dict[str, int]:
        """Get distribution of languages in the mixed dataset.
        
        Returns:
            Dictionary mapping language names to sample counts
        """
        distribution = {}
        
        for dataset_idx, _ in self.mixed_indices:
            # Get language from dataset
            sample = self.datasets[dataset_idx][0]  # Get first sample to check language
            language = sample.get('language', f'dataset_{dataset_idx}')
            
            distribution[language] = distribution.get(language, 0) + 1
        
        return distribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mixed dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_samples": len(self.mixed_indices),
            "num_datasets": len(self.datasets),
            "mixing_ratios": self.mixing_ratios,
            "language_distribution": self.get_language_distribution(),
            "dataset_statistics": []
        }
        
        # Add individual dataset statistics
        for i, dataset in enumerate(self.datasets):
            if hasattr(dataset, 'get_statistics'):
                dataset_stats = dataset.get_statistics()
                dataset_stats['dataset_index'] = i
                stats["dataset_statistics"].append(dataset_stats)
        
        return stats


class EnglishDataset(Dataset):
    """Dataset for English text data (for knowledge preservation)."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        debug_samples: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True
    ):
        """Initialize English dataset.
        
        Args:
            data_path: Path to JSONL file with English text
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            debug_samples: Number of samples for debug mode
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug_samples = debug_samples
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Load and validate data
        self.data = self._load_data()
        self.language = "english"
        
        logger.info(f"EnglishDataset initialized with {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load English data from JSONL file.
        
        Returns:
            List of data samples
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"English dataset file not found: {self.data_path}")
        
        data = []
        error_count = 0
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        
                        if 'text' not in sample:
                            logger.warning(f"Line {line_num}: Missing 'text' field, skipping")
                            error_count += 1
                            continue
                        
                        text = sample['text']
                        if not isinstance(text, str) or not text.strip():
                            logger.warning(f"Line {line_num}: Invalid or empty text, skipping")
                            error_count += 1
                            continue
                        
                        # Add metadata
                        sample['line_number'] = line_num
                        sample['source_file'] = str(self.data_path)
                        
                        data.append(sample)
                        
                        # Apply debug sampling limit
                        if self.debug_samples and len(data) >= self.debug_samples:
                            logger.info(f"Debug mode: Limited to {self.debug_samples} samples")
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: JSON decode error - {e}, skipping")
                        error_count += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Line {line_num}: Unexpected error - {e}, skipping")
                        error_count += 1
                        continue
        
        except Exception as e:
            logger.error(f"Failed to load English dataset from {self.data_path}: {e}")
            raise ValueError(f"English dataset loading failed: {e}") from e
        
        if not data:
            raise ValueError(f"No valid samples found in {self.data_path}")
        
        logger.info(f"Loaded {len(data)} valid English samples, skipped {error_count} invalid lines")
        return data
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single English sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing tokenized sample
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        sample = self.data[idx]
        text = sample['text']
        
        # Tokenize text
        try:
            token_ids = self.tokenizer.encode(
                text, 
                add_bos=self.add_bos, 
                add_eos=self.add_eos
            )
            
            # Truncate if necessary
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
                # Ensure EOS token at the end if it was requested
                if self.add_eos and self.tokenizer.sp_model.eos_id() >= 0:
                    token_ids[-1] = self.tokenizer.sp_model.eos_id()
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(token_ids, dtype=torch.long),
                'text': text,
                'language': self.language,
                'metadata': {
                    'line_number': sample.get('line_number'),
                    'source_file': sample.get('source_file'),
                    'original_length': len(text),
                    'token_count': len(token_ids)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to tokenize English sample {idx}: {e}")
            # Return a minimal valid sample
            fallback_tokens = [self.tokenizer.sp_model.unk_id()]
            return {
                'input_ids': torch.tensor(fallback_tokens, dtype=torch.long),
                'attention_mask': torch.tensor([1], dtype=torch.long),
                'labels': torch.tensor(fallback_tokens, dtype=torch.long),
                'text': "<UNK>",
                'language': self.language,
                'metadata': {
                    'line_number': sample.get('line_number'),
                    'source_file': sample.get('source_file'),
                    'original_length': 0,
                    'token_count': 1,
                    'error': str(e)
                }
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get English dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.data:
            return {"error": "No data loaded"}
        
        text_lengths = [len(sample['text']) for sample in self.data]
        
        return {
            "total_samples": len(self.data),
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "total_characters": sum(text_lengths),
            "language": self.language,
            "source_file": str(self.data_path),
            "debug_mode": self.debug_samples is not None,
            "debug_samples": self.debug_samples
        }