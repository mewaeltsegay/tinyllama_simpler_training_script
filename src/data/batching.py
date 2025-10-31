"""Advanced batching utilities for mixed-language training with dynamic batching and memory efficiency."""

import math
import random
from typing import List, Dict, Any, Optional, Tuple, Iterator
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Sampler
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DynamicBatchSampler(Sampler):
    """Dynamic batch sampler that groups sequences by length for memory efficiency."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        max_tokens: Optional[int] = None,
        length_tolerance: float = 0.1,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42
    ):
        """Initialize dynamic batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Target batch size
            max_tokens: Maximum tokens per batch (overrides batch_size if specified)
            length_tolerance: Tolerance for grouping sequences by length
            shuffle: Whether to shuffle batches
            drop_last: Whether to drop last incomplete batch
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.length_tolerance = length_tolerance
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        # Analyze dataset to create length-based groups
        self.length_groups = self._create_length_groups()
        self.batches = self._create_batches()
        
        logger.info(f"DynamicBatchSampler initialized with {len(self.batches)} batches")
    
    def _create_length_groups(self) -> Dict[int, List[int]]:
        """Group dataset indices by sequence length.
        
        Returns:
            Dictionary mapping length ranges to lists of indices
        """
        length_groups = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                seq_length = len(sample['input_ids'])
                
                # Group by length ranges to allow some flexibility
                length_bucket = int(seq_length / (1 + self.length_tolerance))
                length_groups[length_bucket].append(idx)
                
            except Exception as e:
                logger.warning(f"Failed to get length for sample {idx}: {e}")
                # Put in a default bucket
                length_groups[0].append(idx)
        
        logger.info(f"Created {len(length_groups)} length groups")
        return dict(length_groups)
    
    def _create_batches(self) -> List[List[int]]:
        """Create batches from length groups.
        
        Returns:
            List of batches (each batch is a list of indices)
        """
        batches = []
        
        for length_bucket, indices in self.length_groups.items():
            if self.shuffle:
                random.Random(self.seed).shuffle(indices)
            
            # Create batches from this length group
            if self.max_tokens:
                # Dynamic batching based on token count
                batches.extend(self._create_token_based_batches(indices, length_bucket))
            else:
                # Fixed batch size
                batches.extend(self._create_fixed_size_batches(indices))
        
        # Shuffle batches if requested
        if self.shuffle:
            random.Random(self.seed + 1).shuffle(batches)
        
        return batches
    
    def _create_token_based_batches(self, indices: List[int], length_bucket: int) -> List[List[int]]:
        """Create batches based on maximum token count.
        
        Args:
            indices: List of sample indices
            length_bucket: Length bucket identifier
            
        Returns:
            List of batches
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        # Estimate tokens per sample based on length bucket
        estimated_length = int(length_bucket * (1 + self.length_tolerance))
        
        for idx in indices:
            # Add sample if it fits within token limit
            if current_tokens + estimated_length <= self.max_tokens or not current_batch:
                current_batch.append(idx)
                current_tokens += estimated_length
            else:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_tokens = estimated_length
        
        # Add final batch if not empty
        if current_batch and (not self.drop_last or len(current_batch) >= self.batch_size // 2):
            batches.append(current_batch)
        
        return batches
    
    def _create_fixed_size_batches(self, indices: List[int]) -> List[List[int]]:
        """Create fixed-size batches.
        
        Args:
            indices: List of sample indices
            
        Returns:
            List of batches
        """
        batches = []
        
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        return batches
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        return iter(self.batches)
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.batches)


class MixedLanguageBatchSampler(Sampler):
    """Batch sampler for mixed-language training with controlled language distribution."""
    
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int,
        mixing_ratios: List[float],
        language_batch_strategy: str = "mixed",
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42
    ):
        """Initialize mixed-language batch sampler.
        
        Args:
            datasets: List of language-specific datasets
            batch_size: Batch size
            mixing_ratios: Ratios for mixing languages in batches
            language_batch_strategy: 'mixed', 'separate', or 'alternating'
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop last incomplete batch
            seed: Random seed
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.mixing_ratios = mixing_ratios
        self.language_batch_strategy = language_batch_strategy
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        # Validate inputs
        if len(datasets) != len(mixing_ratios):
            raise ValueError("Number of datasets must match number of mixing ratios")
        
        if abs(sum(mixing_ratios) - 1.0) > 1e-6:
            raise ValueError("Mixing ratios must sum to 1.0")
        
        # Create batches based on strategy
        self.batches = self._create_mixed_batches()
        
        logger.info(f"MixedLanguageBatchSampler initialized with {len(self.batches)} batches "
                   f"using '{language_batch_strategy}' strategy")
    
    def _create_mixed_batches(self) -> List[List[Tuple[int, int]]]:
        """Create mixed-language batches.
        
        Returns:
            List of batches, where each batch contains (dataset_idx, sample_idx) tuples
        """
        if self.language_batch_strategy == "mixed":
            return self._create_mixed_language_batches()
        elif self.language_batch_strategy == "separate":
            return self._create_separate_language_batches()
        elif self.language_batch_strategy == "alternating":
            return self._create_alternating_language_batches()
        else:
            raise ValueError(f"Unknown language batch strategy: {self.language_batch_strategy}")
    
    def _create_mixed_language_batches(self) -> List[List[Tuple[int, int]]]:
        """Create batches with mixed languages within each batch.
        
        Returns:
            List of mixed-language batches
        """
        # Calculate samples per language per batch
        samples_per_lang = [max(1, int(ratio * self.batch_size)) for ratio in self.mixing_ratios]
        
        # Adjust for rounding errors
        total_samples = sum(samples_per_lang)
        if total_samples != self.batch_size:
            diff = self.batch_size - total_samples
            samples_per_lang[0] += diff
        
        # Create sample pools for each dataset
        sample_pools = []
        for dataset_idx, dataset in enumerate(self.datasets):
            indices = list(range(len(dataset)))
            if self.shuffle:
                random.Random(self.seed + dataset_idx).shuffle(indices)
            sample_pools.append([(dataset_idx, idx) for idx in indices])
        
        # Create mixed batches
        batches = []
        pool_positions = [0] * len(sample_pools)
        
        while any(pos < len(pool) for pos, pool in zip(pool_positions, sample_pools)):
            batch = []
            
            # Add samples from each language according to ratios
            for lang_idx, (num_samples, pool) in enumerate(zip(samples_per_lang, sample_pools)):
                start_pos = pool_positions[lang_idx]
                end_pos = min(start_pos + num_samples, len(pool))
                
                batch.extend(pool[start_pos:end_pos])
                pool_positions[lang_idx] = end_pos
            
            # Only add batch if it has enough samples
            if len(batch) >= self.batch_size or not self.drop_last:
                # Shuffle samples within batch
                if self.shuffle:
                    random.Random(self.seed + len(batches)).shuffle(batch)
                batches.append(batch)
        
        return batches
    
    def _create_separate_language_batches(self) -> List[List[Tuple[int, int]]]:
        """Create separate batches for each language.
        
        Returns:
            List of language-specific batches
        """
        all_batches = []
        
        for dataset_idx, dataset in enumerate(self.datasets):
            indices = list(range(len(dataset)))
            if self.shuffle:
                random.Random(self.seed + dataset_idx).shuffle(indices)
            
            # Create batches for this language
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    batch = [(dataset_idx, idx) for idx in batch_indices]
                    all_batches.append(batch)
        
        # Shuffle all batches
        if self.shuffle:
            random.Random(self.seed).shuffle(all_batches)
        
        return all_batches
    
    def _create_alternating_language_batches(self) -> List[List[Tuple[int, int]]]:
        """Create batches that alternate between languages.
        
        Returns:
            List of alternating language batches
        """
        # Create separate batches for each language first
        language_batches = [[] for _ in self.datasets]
        
        for dataset_idx, dataset in enumerate(self.datasets):
            indices = list(range(len(dataset)))
            if self.shuffle:
                random.Random(self.seed + dataset_idx).shuffle(indices)
            
            # Create batches for this language
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    batch = [(dataset_idx, idx) for idx in batch_indices]
                    language_batches[dataset_idx].append(batch)
        
        # Alternate between languages based on mixing ratios
        all_batches = []
        batch_counts = [0] * len(self.datasets)
        total_batches = sum(len(batches) for batches in language_batches)
        
        # Calculate target batches per language
        target_batches = [int(ratio * total_batches) for ratio in self.mixing_ratios]
        
        # Alternate adding batches
        while any(batch_counts[i] < len(language_batches[i]) for i in range(len(self.datasets))):
            for lang_idx in range(len(self.datasets)):
                if (batch_counts[lang_idx] < len(language_batches[lang_idx]) and
                    batch_counts[lang_idx] < target_batches[lang_idx]):
                    all_batches.append(language_batches[lang_idx][batch_counts[lang_idx]])
                    batch_counts[lang_idx] += 1
        
        return all_batches
    
    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        """Iterate over batches."""
        return iter(self.batches)
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.batches)


class StreamingBatchLoader:
    """Memory-efficient streaming batch loader for large datasets."""
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        batch_size: int = 4,
        max_length: int = 2048,
        buffer_size: int = 10000,
        mixing_ratios: Optional[List[float]] = None
    ):
        """Initialize streaming batch loader.
        
        Args:
            data_paths: List of paths to JSONL files
            tokenizer: Tokenizer instance
            batch_size: Batch size
            max_length: Maximum sequence length
            buffer_size: Size of streaming buffer
            mixing_ratios: Ratios for mixing datasets
        """
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.mixing_ratios = mixing_ratios or [1.0 / len(data_paths)] * len(data_paths)
        
        # Validate mixing ratios
        if len(self.mixing_ratios) != len(data_paths):
            raise ValueError("Number of mixing ratios must match number of data paths")
        
        if abs(sum(self.mixing_ratios) - 1.0) > 1e-6:
            raise ValueError("Mixing ratios must sum to 1.0")
        
        logger.info(f"StreamingBatchLoader initialized for {len(data_paths)} datasets")
    
    def stream_batches(self, shuffle: bool = True, seed: int = 42) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream batches from multiple datasets.
        
        Args:
            shuffle: Whether to shuffle samples
            seed: Random seed for shuffling
            
        Yields:
            Batched and padded tensors
        """
        import json
        
        # Open file handles
        file_handles = []
        for path in self.data_paths:
            try:
                file_handles.append(open(path, 'r', encoding='utf-8'))
            except Exception as e:
                logger.error(f"Failed to open {path}: {e}")
                raise
        
        try:
            # Initialize buffers for each dataset
            buffers = [[] for _ in self.data_paths]
            buffer_positions = [0] * len(self.data_paths)
            
            # Calculate samples per dataset per batch
            samples_per_dataset = [max(1, int(ratio * self.batch_size)) for ratio in self.mixing_ratios]
            
            # Adjust for rounding errors
            total_samples = sum(samples_per_dataset)
            if total_samples != self.batch_size:
                diff = self.batch_size - total_samples
                samples_per_dataset[0] += diff
            
            while True:
                # Fill buffers if needed
                for i, (file_handle, buffer) in enumerate(zip(file_handles, buffers)):
                    while len(buffer) - buffer_positions[i] < samples_per_dataset[i]:
                        line = file_handle.readline()
                        if not line:
                            # End of file reached
                            if all(len(buf) - pos == 0 for buf, pos in zip(buffers, buffer_positions)):
                                return  # All files exhausted
                            break
                        
                        try:
                            sample = json.loads(line.strip())
                            if 'text' in sample:
                                # Tokenize and add to buffer
                                processed_sample = self._process_sample(sample, i)
                                if processed_sample:
                                    buffer.append(processed_sample)
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing sample: {e}")
                            continue
                
                # Create batch from buffers
                batch_samples = []
                
                for i, (buffer, num_samples) in enumerate(zip(buffers, samples_per_dataset)):
                    start_pos = buffer_positions[i]
                    end_pos = min(start_pos + num_samples, len(buffer))
                    
                    batch_samples.extend(buffer[start_pos:end_pos])
                    buffer_positions[i] = end_pos
                
                # Check if we have enough samples for a batch
                if len(batch_samples) < self.batch_size:
                    # Not enough samples, try to get more or finish
                    if all(len(buf) - pos == 0 for buf, pos in zip(buffers, buffer_positions)):
                        if batch_samples:  # Yield partial batch if we have samples
                            yield self._collate_batch(batch_samples, shuffle, seed)
                        return
                    continue
                
                # Shuffle batch if requested
                if shuffle:
                    random.Random(seed).shuffle(batch_samples)
                
                # Yield collated batch
                yield self._collate_batch(batch_samples, shuffle, seed)
                
                # Clean up buffers periodically
                for i in range(len(buffers)):
                    if buffer_positions[i] > self.buffer_size // 2:
                        buffers[i] = buffers[i][buffer_positions[i]:]
                        buffer_positions[i] = 0
        
        finally:
            # Close file handles
            for handle in file_handles:
                handle.close()
    
    def _process_sample(self, sample: Dict[str, Any], dataset_idx: int) -> Optional[Dict[str, Any]]:
        """Process a single sample from streaming data.
        
        Args:
            sample: Raw sample from JSONL
            dataset_idx: Index of source dataset
            
        Returns:
            Processed sample or None if invalid
        """
        try:
            text = sample['text']
            if not text or not text.strip():
                return None
            
            # Tokenize
            token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            
            # Truncate if necessary
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
                if self.tokenizer.sp_model.eos_id() >= 0:
                    token_ids[-1] = self.tokenizer.sp_model.eos_id()
            
            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'attention_mask': torch.tensor([1] * len(token_ids), dtype=torch.long),
                'labels': torch.tensor(token_ids, dtype=torch.long),
                'text': text,
                'dataset_idx': dataset_idx
            }
            
        except Exception as e:
            logger.warning(f"Failed to process sample: {e}")
            return None
    
    def _collate_batch(
        self, 
        batch_samples: List[Dict[str, Any]], 
        shuffle: bool, 
        seed: int
    ) -> Dict[str, torch.Tensor]:
        """Collate batch samples with padding.
        
        Args:
            batch_samples: List of processed samples
            shuffle: Whether samples were shuffled
            seed: Random seed used
            
        Returns:
            Collated batch
        """
        if not batch_samples:
            return {}
        
        # Get maximum length in batch
        max_length = max(len(sample['input_ids']) for sample in batch_samples)
        
        # Pad sequences
        pad_token_id = self.tokenizer.get_special_tokens().get('pad_id', 0)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for sample in batch_samples:
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            labels = sample['labels']
            
            # Calculate padding
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids
            padded_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), pad_token_id, dtype=torch.long)
            ])
            padded_input_ids.append(padded_ids)
            
            # Pad attention_mask
            padded_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            padded_attention_masks.append(padded_mask)
            
            # Pad labels (use -100 for ignored tokens)
            padded_label = torch.cat([
                labels,
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
            padded_labels.append(padded_label)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_labels),
            'texts': [sample['text'] for sample in batch_samples],
            'dataset_indices': [sample['dataset_idx'] for sample in batch_samples]
        }
    
    def estimate_batches(self) -> int:
        """Estimate total number of batches across all datasets.
        
        Returns:
            Estimated number of batches
        """
        total_lines = 0
        
        for path in self.data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for line in f if line.strip())
                    total_lines += lines
            except Exception as e:
                logger.warning(f"Failed to count lines in {path}: {e}")
        
        estimated_batches = math.ceil(total_lines / self.batch_size)
        logger.info(f"Estimated {estimated_batches} batches from {total_lines} total samples")
        
        return estimated_batches


class MemoryEfficientCollator:
    """Memory-efficient collate function with advanced padding strategies."""
    
    def __init__(
        self,
        tokenizer,
        padding_strategy: str = "longest",
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None
    ):
        """Initialize memory-efficient collator.
        
        Args:
            tokenizer: Tokenizer instance
            padding_strategy: 'longest', 'max_length', or 'bucket'
            max_length: Maximum sequence length
            pad_to_multiple_of: Pad to multiple of this value
        """
        self.tokenizer = tokenizer
        self.padding_strategy = padding_strategy
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
        self.pad_token_id = tokenizer.get_special_tokens().get('pad_id', 0)
        
        logger.info(f"MemoryEfficientCollator initialized with strategy: {padding_strategy}")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with efficient padding.
        
        Args:
            batch: List of samples
            
        Returns:
            Collated batch
        """
        if not batch:
            return {}
        
        # Determine target length
        if self.padding_strategy == "longest":
            target_length = max(len(sample['input_ids']) for sample in batch)
        elif self.padding_strategy == "max_length":
            target_length = self.max_length or max(len(sample['input_ids']) for sample in batch)
        elif self.padding_strategy == "bucket":
            target_length = self._get_bucket_length(batch)
        else:
            raise ValueError(f"Unknown padding strategy: {self.padding_strategy}")
        
        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of:
            target_length = math.ceil(target_length / self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad sequences
        return self._pad_sequences(batch, target_length)
    
    def _get_bucket_length(self, batch: List[Dict[str, Any]]) -> int:
        """Get bucket length for efficient padding.
        
        Args:
            batch: List of samples
            
        Returns:
            Target length for padding
        """
        lengths = [len(sample['input_ids']) for sample in batch]
        max_length = max(lengths)
        
        # Define buckets (powers of 2 for memory efficiency)
        buckets = [64, 128, 256, 512, 1024, 2048, 4096]
        
        for bucket in buckets:
            if max_length <= bucket:
                return bucket
        
        return max_length
    
    def _pad_sequences(self, batch: List[Dict[str, Any]], target_length: int) -> Dict[str, torch.Tensor]:
        """Pad sequences to target length.
        
        Args:
            batch: List of samples
            target_length: Target sequence length
            
        Returns:
            Padded batch
        """
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for sample in batch:
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            labels = sample['labels']
            
            current_length = len(input_ids)
            
            if current_length > target_length:
                # Truncate if longer than target
                input_ids = input_ids[:target_length]
                attention_mask = attention_mask[:target_length]
                labels = labels[:target_length]
            elif current_length < target_length:
                # Pad if shorter than target
                padding_length = target_length - current_length
                
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
                ])
                
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                
                labels = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=torch.long)
                ])
            
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)
            padded_labels.append(labels)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_labels),
            'texts': [sample['text'] for sample in batch],
            'languages': [sample.get('language', 'unknown') for sample in batch],
            'metadata': [sample.get('metadata', {}) for sample in batch]
        }