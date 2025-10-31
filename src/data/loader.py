"""Data pipeline and loader components for mixed-language training."""

import os
import time
from typing import List, Dict, Any, Optional, Union, Iterator
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from ..utils.logging import get_logger
from ..utils.error_handling import ErrorHandler, DataLoadingError
from .dataset import TigrinyaDataset, EnglishDataset, MixedLanguageDataset
from .preprocessing import TextPreprocessor, DatasetSampler
from .batching import (
    DynamicBatchSampler, 
    MixedLanguageBatchSampler, 
    StreamingBatchLoader,
    MemoryEfficientCollator
)

logger = get_logger(__name__)


class DataPipeline:
    """Main data pipeline for loading and preprocessing mixed-language training data."""
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        debug_samples: Optional[int] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize data pipeline.
        
        Args:
            tokenizer: Tokenizer instance (TigrinyaTokenizer)
            max_length: Maximum sequence length
            debug_samples: Number of samples for debug mode
            preprocessing_config: Configuration for text preprocessing
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.debug_samples = debug_samples
        
        # Initialize preprocessor
        preprocessing_config = preprocessing_config or {}
        self.preprocessor = TextPreprocessor(**preprocessing_config)
        
        # Initialize sampler
        self.sampler = DatasetSampler()
        
        # Initialize error handler
        self.error_handler = ErrorHandler({
            'max_data_errors': 100,
            'skip_corrupted_samples': True
        })
        
        # Storage for loaded datasets
        self.tigrinya_dataset = None
        self.english_dataset = None
        self.mixed_dataset = None
        
        logger.info(f"DataPipeline initialized with max_length={max_length}, debug_samples={debug_samples}")
    

    def load_tigrinya_dataset(
        self,
        data_path: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> TigrinyaDataset:
        """Load Tigrinya dataset with error handling and preprocessing.
        
        Args:
            data_path: Path to Tigrinya JSONL file
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            Loaded TigrinyaDataset
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            DataLoadingError: If dataset loading fails
        """
        logger.info(f"Loading Tigrinya dataset from: {data_path}")
        
        try:
            # Validate file exists and is readable
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Tigrinya dataset file not found: {data_path}")
            
            if not os.access(data_path, os.R_OK):
                raise PermissionError(f"Cannot read Tigrinya dataset file: {data_path}")
            
            # Check file size
            file_size = os.path.getsize(data_path)
            if file_size == 0:
                raise ValueError(f"Tigrinya dataset file is empty: {data_path}")
            
            logger.info(f"Dataset file size: {file_size / 1024 / 1024:.1f} MB")
            
            # Create dataset with error handling
            dataset = self._create_dataset_with_error_handling(
                TigrinyaDataset,
                data_path=data_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                debug_samples=self.debug_samples,
                add_bos=add_bos,
                add_eos=add_eos
            )
            
            # Validate dataset
            self._validate_dataset_with_error_handling(dataset, "Tigrinya", data_path)
            
            self.tigrinya_dataset = dataset
            logger.info(f"Successfully loaded Tigrinya dataset with {len(dataset)} samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load Tigrinya dataset: {e}")
            raise DataLoadingError(f"Tigrinya dataset loading failed: {e}") from e
    
    def load_english_dataset(
        self,
        data_path: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> EnglishDataset:
        """Load English dataset for knowledge preservation.
        
        Args:
            data_path: Path to English JSONL file
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            Loaded EnglishDataset
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset loading fails
        """
        logger.info(f"Loading English dataset from: {data_path}")
        
        try:
            # Validate file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"English dataset file not found: {data_path}")
            
            # Create dataset
            dataset = EnglishDataset(
                data_path=data_path,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                debug_samples=self.debug_samples,
                add_bos=add_bos,
                add_eos=add_eos
            )
            
            # Validate dataset
            self._validate_dataset(dataset, "English")
            
            self.english_dataset = dataset
            logger.info(f"Successfully loaded English dataset with {len(dataset)} samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load English dataset: {e}")
            raise ValueError(f"English dataset loading failed: {e}") from e
    
    def create_mixed_dataset(
        self,
        tigrinya_path: str,
        english_path: Optional[str] = None,
        mixing_ratios: Optional[List[float]] = None,
        shuffle_seed: int = 42
    ) -> MixedLanguageDataset:
        """Create mixed-language dataset for bilingual training.
        
        Args:
            tigrinya_path: Path to Tigrinya JSONL file
            english_path: Path to English JSONL file (optional)
            mixing_ratios: Ratios for mixing datasets [tigrinya_ratio, english_ratio]
            shuffle_seed: Random seed for shuffling
            
        Returns:
            Mixed language dataset
            
        Raises:
            ValueError: If dataset creation fails
        """
        logger.info("Creating mixed-language dataset")
        
        try:
            datasets = []
            
            # Load Tigrinya dataset
            tigrinya_dataset = self.load_tigrinya_dataset(tigrinya_path)
            datasets.append(tigrinya_dataset)
            
            # Load English dataset if provided
            if english_path:
                english_dataset = self.load_english_dataset(english_path)
                datasets.append(english_dataset)
            
            # Set default mixing ratios
            if mixing_ratios is None:
                if len(datasets) == 1:
                    mixing_ratios = [1.0]
                else:
                    # Default: 70% Tigrinya, 30% English for knowledge preservation
                    mixing_ratios = [0.7, 0.3]
            
            # Create mixed dataset
            mixed_dataset = MixedLanguageDataset(
                datasets=datasets,
                mixing_ratios=mixing_ratios,
                shuffle_seed=shuffle_seed
            )
            
            # Validate mixed dataset
            self._validate_mixed_dataset(mixed_dataset)
            
            self.mixed_dataset = mixed_dataset
            logger.info(f"Successfully created mixed dataset with {len(mixed_dataset)} samples")
            
            return mixed_dataset
            
        except Exception as e:
            logger.error(f"Failed to create mixed dataset: {e}")
            raise ValueError(f"Mixed dataset creation failed: {e}") from e
    
    def create_mixed_language_loader(
        self,
        datasets: List[Dataset],
        batch_size: int = 4,
        mixing_ratios: Optional[List[float]] = None,
        language_batch_strategy: str = "mixed",
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True
    ) -> DataLoader:
        """Create data loader with mixed-language batch sampling.
        
        Args:
            datasets: List of language-specific datasets
            batch_size: Batch size
            mixing_ratios: Ratios for mixing languages
            language_batch_strategy: 'mixed', 'separate', or 'alternating'
            shuffle: Whether to shuffle samples
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            
        Returns:
            DataLoader with mixed-language batching
        """
        logger.info(f"Creating mixed-language data loader with strategy: {language_batch_strategy}")
        
        try:
            # Set default mixing ratios
            if mixing_ratios is None:
                mixing_ratios = [1.0 / len(datasets)] * len(datasets)
            
            # Create mixed-language batch sampler
            batch_sampler = MixedLanguageBatchSampler(
                datasets=datasets,
                batch_size=batch_size,
                mixing_ratios=mixing_ratios,
                language_batch_strategy=language_batch_strategy,
                shuffle=shuffle,
                drop_last=drop_last
            )
            
            # Create a combined dataset for the DataLoader
            # We'll use the first dataset as base and override __getitem__
            combined_dataset = CombinedDataset(datasets)
            
            # Create memory-efficient collator
            collate_fn = MemoryEfficientCollator(
                tokenizer=self.tokenizer,
                padding_strategy="longest",
                max_length=self.max_length
            )
            
            data_loader = DataLoader(
                dataset=combined_dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )
            
            logger.info(f"Successfully created mixed-language data loader")
            return data_loader
            
        except Exception as e:
            logger.error(f"Failed to create mixed-language data loader: {e}")
            raise ValueError(f"Mixed-language data loader creation failed: {e}") from e
    
    def create_streaming_loader(
        self,
        data_paths: List[str],
        batch_size: int = 4,
        mixing_ratios: Optional[List[float]] = None,
        buffer_size: int = 10000
    ) -> StreamingBatchLoader:
        """Create streaming batch loader for memory-efficient training.
        
        Args:
            data_paths: List of paths to JSONL files
            batch_size: Batch size
            mixing_ratios: Ratios for mixing datasets
            buffer_size: Size of streaming buffer
            
        Returns:
            StreamingBatchLoader instance
        """
        logger.info(f"Creating streaming batch loader for {len(data_paths)} datasets")
        
        try:
            streaming_loader = StreamingBatchLoader(
                data_paths=data_paths,
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                max_length=self.max_length,
                buffer_size=buffer_size,
                mixing_ratios=mixing_ratios
            )
            
            logger.info("Successfully created streaming batch loader")
            return streaming_loader
            
        except Exception as e:
            logger.error(f"Failed to create streaming batch loader: {e}")
            raise ValueError(f"Streaming batch loader creation failed: {e}") from e
    
    def create_data_loader(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
        use_dynamic_batching: bool = False,
        max_tokens: Optional[int] = None,
        padding_strategy: str = "longest",
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ) -> DataLoader:
        """Create data loader with advanced batching support.
        
        Args:
            dataset: Dataset to create loader for
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            drop_last: Whether to drop last incomplete batch
            use_dynamic_batching: Whether to use dynamic batching by sequence length
            max_tokens: Maximum tokens per batch (for dynamic batching)
            padding_strategy: Padding strategy for collator
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Whether to keep workers alive between epochs
            
        Returns:
            Configured DataLoader
        """
        logger.info(f"Creating data loader with batch_size={batch_size}, num_workers={num_workers}")
        
        try:
            # Create memory-efficient collator
            collate_fn = MemoryEfficientCollator(
                tokenizer=self.tokenizer,
                padding_strategy=padding_strategy,
                max_length=self.max_length
            )
            
            # Create batch sampler if using dynamic batching
            batch_sampler = None
            if use_dynamic_batching:
                batch_sampler = DynamicBatchSampler(
                    dataset=dataset,
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                    shuffle=shuffle,
                    drop_last=drop_last
                )
                # When using batch_sampler, set batch_size=1 and shuffle=False
                loader_kwargs = {
                    'dataset': dataset,
                    'batch_sampler': batch_sampler,
                    'num_workers': num_workers,
                    'pin_memory': pin_memory,
                    'collate_fn': collate_fn
                }
                
                # Add prefetching parameters if workers > 0
                if num_workers > 0:
                    loader_kwargs['prefetch_factor'] = prefetch_factor
                    loader_kwargs['persistent_workers'] = persistent_workers
                
                data_loader = DataLoader(**loader_kwargs)
            else:
                loader_kwargs = {
                    'dataset': dataset,
                    'batch_size': batch_size,
                    'shuffle': shuffle,
                    'num_workers': num_workers,
                    'pin_memory': pin_memory,
                    'drop_last': drop_last,
                    'collate_fn': collate_fn
                }
                
                # Add prefetching parameters if workers > 0
                if num_workers > 0:
                    loader_kwargs['prefetch_factor'] = prefetch_factor
                    loader_kwargs['persistent_workers'] = persistent_workers
                
                data_loader = DataLoader(**loader_kwargs)
            
            logger.info(f"Successfully created data loader for dataset with {len(dataset)} samples")
            return data_loader
            
        except Exception as e:
            logger.error(f"Failed to create data loader: {e}")
            raise ValueError(f"Data loader creation failed: {e}") from e
    
    def _create_collate_fn(self):
        """Create collate function for dynamic batching."""
        
        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            """Collate function for dynamic batching with padding.
            
            Args:
                batch: List of samples from dataset
                
            Returns:
                Batched and padded tensors
            """
            # Extract components
            input_ids = [sample['input_ids'] for sample in batch]
            attention_masks = [sample['attention_mask'] for sample in batch]
            labels = [sample['labels'] for sample in batch]
            
            # Get maximum length in batch
            max_length = max(len(ids) for ids in input_ids)
            
            # Pad sequences
            pad_token_id = self.tokenizer.get_special_tokens().get('pad_id', 0)
            
            padded_input_ids = []
            padded_attention_masks = []
            padded_labels = []
            
            for i in range(len(batch)):
                # Pad input_ids
                padding_length = max_length - len(input_ids[i])
                padded_ids = torch.cat([
                    input_ids[i],
                    torch.full((padding_length,), pad_token_id, dtype=torch.long)
                ])
                padded_input_ids.append(padded_ids)
                
                # Pad attention_mask
                padded_mask = torch.cat([
                    attention_masks[i],
                    torch.zeros(padding_length, dtype=torch.long)
                ])
                padded_attention_masks.append(padded_mask)
                
                # Pad labels (use -100 for ignored tokens in loss calculation)
                padded_label = torch.cat([
                    labels[i],
                    torch.full((padding_length,), -100, dtype=torch.long)
                ])
                padded_labels.append(padded_label)
            
            # Stack into batch tensors
            batch_dict = {
                'input_ids': torch.stack(padded_input_ids),
                'attention_mask': torch.stack(padded_attention_masks),
                'labels': torch.stack(padded_labels)
            }
            
            # Add metadata
            batch_dict['languages'] = [sample['language'] for sample in batch]
            batch_dict['texts'] = [sample['text'] for sample in batch]
            batch_dict['metadata'] = [sample['metadata'] for sample in batch]
            
            return batch_dict
        
        return collate_fn
    
    def _validate_dataset(self, dataset: Dataset, dataset_name: str) -> None:
        """Validate dataset functionality.
        
        Args:
            dataset: Dataset to validate
            dataset_name: Name of dataset for logging
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Check dataset size
            if len(dataset) == 0:
                raise ValueError(f"{dataset_name} dataset is empty")
            
            # Test first sample
            sample = dataset[0]
            required_keys = ['input_ids', 'attention_mask', 'labels', 'text', 'language']
            
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Missing required key '{key}' in {dataset_name} dataset sample")
            
            # Validate tensor shapes and types
            if not isinstance(sample['input_ids'], torch.Tensor):
                raise ValueError(f"input_ids must be torch.Tensor in {dataset_name} dataset")
            
            if not isinstance(sample['attention_mask'], torch.Tensor):
                raise ValueError(f"attention_mask must be torch.Tensor in {dataset_name} dataset")
            
            if not isinstance(sample['labels'], torch.Tensor):
                raise ValueError(f"labels must be torch.Tensor in {dataset_name} dataset")
            
            # Check tensor dimensions
            if sample['input_ids'].dim() != 1:
                raise ValueError(f"input_ids must be 1D tensor in {dataset_name} dataset")
            
            if len(sample['input_ids']) != len(sample['attention_mask']):
                raise ValueError(f"input_ids and attention_mask length mismatch in {dataset_name} dataset")
            
            if len(sample['input_ids']) != len(sample['labels']):
                raise ValueError(f"input_ids and labels length mismatch in {dataset_name} dataset")
            
            # Test a few more samples if available
            test_samples = min(3, len(dataset))
            for i in range(1, test_samples):
                try:
                    test_sample = dataset[i]
                    if not all(key in test_sample for key in required_keys):
                        raise ValueError(f"Sample {i} missing required keys in {dataset_name} dataset")
                except Exception as e:
                    raise ValueError(f"Error accessing sample {i} in {dataset_name} dataset: {e}")
            
            logger.info(f"{dataset_name} dataset validation passed")
            
        except Exception as e:
            logger.error(f"{dataset_name} dataset validation failed: {e}")
            raise ValueError(f"{dataset_name} dataset validation failed: {e}") from e
    
    def _validate_mixed_dataset(self, mixed_dataset: MixedLanguageDataset) -> None:
        """Validate mixed language dataset.
        
        Args:
            mixed_dataset: Mixed dataset to validate
            
        Raises:
            ValueError: If validation fails
        """
        try:
            # Check dataset size
            if len(mixed_dataset) == 0:
                raise ValueError("Mixed dataset is empty")
            
            # Test sample access
            sample = mixed_dataset[0]
            required_keys = ['input_ids', 'attention_mask', 'labels', 'text', 'language']
            
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"Missing required key '{key}' in mixed dataset sample")
            
            # Check language distribution
            distribution = mixed_dataset.get_language_distribution()
            if not distribution:
                raise ValueError("Mixed dataset has no language distribution")
            
            logger.info(f"Mixed dataset validation passed. Language distribution: {distribution}")
            
        except Exception as e:
            logger.error(f"Mixed dataset validation failed: {e}")
            raise ValueError(f"Mixed dataset validation failed: {e}") from e
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded datasets.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "pipeline_config": {
                "max_length": self.max_length,
                "debug_samples": self.debug_samples,
                "tokenizer_vocab_size": self.tokenizer.get_vocab_size()
            },
            "datasets": {}
        }
        
        # Add Tigrinya dataset stats
        if self.tigrinya_dataset:
            stats["datasets"]["tigrinya"] = self.tigrinya_dataset.get_statistics()
        
        # Add English dataset stats
        if self.english_dataset:
            stats["datasets"]["english"] = self.english_dataset.get_statistics()
        
        # Add mixed dataset stats
        if self.mixed_dataset:
            stats["datasets"]["mixed"] = self.mixed_dataset.get_statistics()
        
        return stats
    
    def create_debug_sample(
        self,
        output_dir: str,
        sample_size: int = 100
    ) -> None:
        """Create debug samples from loaded datasets.
        
        Args:
            output_dir: Directory to save debug samples
            sample_size: Number of samples per dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create debug sample for Tigrinya dataset
        if self.tigrinya_dataset:
            self._create_dataset_debug_sample(
                self.tigrinya_dataset,
                output_dir / "tigrinya_debug.jsonl",
                sample_size
            )
        
        # Create debug sample for English dataset
        if self.english_dataset:
            self._create_dataset_debug_sample(
                self.english_dataset,
                output_dir / "english_debug.jsonl",
                sample_size
            )
        
        # Create debug sample for mixed dataset
        if self.mixed_dataset:
            self._create_dataset_debug_sample(
                self.mixed_dataset,
                output_dir / "mixed_debug.jsonl",
                sample_size
            )
        
        logger.info(f"Debug samples created in: {output_dir}")
    
    def _create_dataset_debug_sample(
        self,
        dataset: Dataset,
        output_path: Path,
        sample_size: int
    ) -> None:
        """Create debug sample for a specific dataset.
        
        Args:
            dataset: Dataset to sample from
            output_path: Path to save debug sample
            sample_size: Number of samples
        """
        import json
        
        sample_size = min(sample_size, len(dataset))
        indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))[:sample_size]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx in indices:
                try:
                    sample = dataset[idx]
                    debug_sample = {
                        'text': sample['text'],
                        'language': sample['language'],
                        'token_count': len(sample['input_ids']),
                        'metadata': sample['metadata']
                    }
                    f.write(json.dumps(debug_sample, ensure_ascii=False) + '\n')
                except Exception as e:
                    logger.warning(f"Failed to process sample {idx}: {e}")
        
        logger.info(f"Created debug sample with {sample_size} samples at: {output_path}")
    
    def _create_dataset_with_error_handling(self, dataset_class, **kwargs):
        """Create dataset with comprehensive error handling.
        
        Args:
            dataset_class: Dataset class to instantiate
            **kwargs: Arguments for dataset creation
            
        Returns:
            Created dataset instance
            
        Raises:
            DataLoadingError: If dataset creation fails
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return dataset_class(**kwargs)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Dataset creation attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise DataLoadingError(f"Dataset creation failed after {max_retries} attempts: {e}") from e
    
    def _validate_dataset_with_error_handling(self, dataset: Dataset, dataset_name: str, data_path: str) -> None:
        """Validate dataset with enhanced error handling and recovery.
        
        Args:
            dataset: Dataset to validate
            dataset_name: Name of dataset for logging
            data_path: Path to dataset file
            
        Raises:
            DataLoadingError: If validation fails
        """
        try:
            # Basic validation
            self._validate_dataset(dataset, dataset_name)
            
            # Additional validation with error handling
            self._validate_dataset_samples_with_recovery(dataset, dataset_name, data_path)
            
        except Exception as e:
            raise DataLoadingError(f"{dataset_name} dataset validation failed: {e}") from e
    
    def _validate_dataset_samples_with_recovery(self, dataset: Dataset, dataset_name: str, data_path: str) -> None:
        """Validate dataset samples with error recovery.
        
        Args:
            dataset: Dataset to validate
            dataset_name: Name of dataset
            data_path: Path to dataset file
        """
        # Test multiple samples to detect corruption patterns
        test_indices = [0, len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]
        corrupted_samples = []
        
        for idx in test_indices:
            if idx >= len(dataset):
                continue
                
            try:
                sample = dataset[idx]
                
                # Validate sample structure
                if not self._is_valid_sample(sample):
                    corrupted_samples.append(idx)
                    
            except Exception as e:
                logger.warning(f"Sample {idx} in {dataset_name} dataset is corrupted: {e}")
                corrupted_samples.append(idx)
                
                # Handle corrupted sample
                if not self.error_handler.handle_data_loading_error(
                    e, idx, data_path, skip_corrupted=True
                ):
                    raise DataLoadingError(f"Too many corrupted samples in {dataset_name} dataset")
        
        if corrupted_samples:
            logger.warning(f"Found {len(corrupted_samples)} corrupted samples in {dataset_name} dataset: {corrupted_samples}")
        else:
            logger.info(f"{dataset_name} dataset sample validation passed")
    
    def _is_valid_sample(self, sample: Dict[str, Any]) -> bool:
        """Check if a sample has valid structure and content.
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if sample is valid
        """
        try:
            # Check required keys
            required_keys = ['input_ids', 'attention_mask', 'labels', 'text', 'language']
            for key in required_keys:
                if key not in sample:
                    return False
            
            # Check tensor properties
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            labels = sample['labels']
            
            if not isinstance(input_ids, torch.Tensor):
                return False
            
            if input_ids.dim() != 1 or len(input_ids) == 0:
                return False
            
            if len(input_ids) != len(attention_mask) or len(input_ids) != len(labels):
                return False
            
            # Check for invalid token IDs
            vocab_size = self.tokenizer.get_vocab_size()
            if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
                return False
            
            # Check text content
            if not isinstance(sample['text'], str) or len(sample['text'].strip()) == 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def create_error_resilient_dataloader(self,
                                        dataset: Dataset,
                                        batch_size: int = 4,
                                        **kwargs) -> DataLoader:
        """Create a DataLoader with error resilience for corrupted samples.
        
        Args:
            dataset: Dataset to create loader for
            batch_size: Batch size
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Error-resilient DataLoader
        """
        # Create custom collate function that handles errors
        original_collate_fn = kwargs.get('collate_fn', self._create_collate_fn())
        
        def error_resilient_collate_fn(batch):
            """Collate function that handles corrupted samples."""
            # Filter out None samples (corrupted ones)
            valid_batch = [sample for sample in batch if sample is not None]
            
            if not valid_batch:
                # If all samples are corrupted, create a dummy batch
                logger.warning("All samples in batch are corrupted, creating dummy batch")
                return self._create_dummy_batch(batch_size)
            
            if len(valid_batch) < len(batch):
                logger.warning(f"Filtered out {len(batch) - len(valid_batch)} corrupted samples from batch")
            
            try:
                return original_collate_fn(valid_batch)
            except Exception as e:
                logger.error(f"Collate function failed: {e}")
                # Return dummy batch as fallback
                return self._create_dummy_batch(len(valid_batch))
        
        # Create error-resilient dataset wrapper
        error_resilient_dataset = ErrorResilientDataset(dataset, self.error_handler)
        
        kwargs['collate_fn'] = error_resilient_collate_fn
        
        return DataLoader(error_resilient_dataset, batch_size=batch_size, **kwargs)
    
    def _create_dummy_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Create a dummy batch for error recovery.
        
        Args:
            batch_size: Size of dummy batch
            
        Returns:
            Dummy batch dictionary
        """
        # Create minimal valid batch
        dummy_length = 10
        pad_token_id = self.tokenizer.get_special_tokens().get('pad_id', 0)
        
        dummy_batch = {
            'input_ids': torch.full((batch_size, dummy_length), pad_token_id, dtype=torch.long),
            'attention_mask': torch.zeros((batch_size, dummy_length), dtype=torch.long),
            'labels': torch.full((batch_size, dummy_length), -100, dtype=torch.long),
            'languages': ['unknown'] * batch_size,
            'texts': [''] * batch_size,
            'metadata': [{}] * batch_size
        }
        
        return dummy_batch


class ErrorResilientDataset(Dataset):
    """Dataset wrapper that handles corrupted samples gracefully."""
    
    def __init__(self, dataset: Dataset, error_handler: ErrorHandler):
        """Initialize error-resilient dataset wrapper.
        
        Args:
            dataset: Original dataset
            error_handler: Error handler for managing failures
        """
        self.dataset = dataset
        self.error_handler = error_handler
        self.corrupted_indices = set()
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        """Get item with error handling.
        
        Args:
            index: Sample index
            
        Returns:
            Sample or None if corrupted
        """
        # Skip known corrupted samples
        if index in self.corrupted_indices:
            return None
        
        try:
            return self.dataset[index]
            
        except Exception as e:
            # Mark sample as corrupted
            self.corrupted_indices.add(index)
            
            # Handle the error
            dataset_path = getattr(self.dataset, 'data_path', 'unknown')
            self.error_handler.handle_data_loading_error(
                e, index, dataset_path, skip_corrupted=True
            )
            
            logger.debug(f"Sample {index} marked as corrupted: {e}")
            return None


class CombinedDataset(Dataset):
    """Combined dataset that allows access to multiple datasets through a single interface."""
    
    def __init__(self, datasets: List[Dataset]):
        """Initialize combined dataset.
        
        Args:
            datasets: List of datasets to combine
        """
        self.datasets = datasets
        self.total_length = sum(len(dataset) for dataset in datasets)
    
    def __len__(self) -> int:
        """Get total length across all datasets."""
        return self.total_length
    
    def __getitem__(self, key):
        """Get item using (dataset_idx, sample_idx) tuple or direct index."""
        if isinstance(key, tuple) and len(key) == 2:
            dataset_idx, sample_idx = key
            return self.datasets[dataset_idx][sample_idx]
        else:
            # For compatibility with regular indexing
            raise NotImplementedError("Direct indexing not supported, use (dataset_idx, sample_idx) tuple")


class MixedLanguageDataLoader:
    """Specialized data loader for mixed-language training with streaming support."""
    
    def __init__(
        self,
        data_pipeline: DataPipeline,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2
    ):
        """Initialize mixed-language data loader.
        
        Args:
            data_pipeline: DataPipeline instance
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            prefetch_factor: Number of batches to prefetch
        """
        self.data_pipeline = data_pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        
        logger.info(f"MixedLanguageDataLoader initialized with batch_size={batch_size}")
    
    def create_training_loader(
        self,
        tigrinya_path: str,
        english_path: Optional[str] = None,
        mixing_ratios: Optional[List[float]] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """Create training data loader with mixed languages.
        
        Args:
            tigrinya_path: Path to Tigrinya dataset
            english_path: Path to English dataset (optional)
            mixing_ratios: Mixing ratios for languages
            shuffle: Whether to shuffle data
            
        Returns:
            Training DataLoader
        """
        # Create mixed dataset
        mixed_dataset = self.data_pipeline.create_mixed_dataset(
            tigrinya_path=tigrinya_path,
            english_path=english_path,
            mixing_ratios=mixing_ratios
        )
        
        # Create data loader
        return self.data_pipeline.create_data_loader(
            dataset=mixed_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
    
    def create_validation_loader(
        self,
        validation_path: str,
        language: str = "tigrinya",
        shuffle: bool = False
    ) -> DataLoader:
        """Create validation data loader.
        
        Args:
            validation_path: Path to validation dataset
            language: Language of validation data
            shuffle: Whether to shuffle validation data
            
        Returns:
            Validation DataLoader
        """
        # Load appropriate dataset
        if language.lower() == "tigrinya":
            dataset = self.data_pipeline.load_tigrinya_dataset(validation_path)
        elif language.lower() == "english":
            dataset = self.data_pipeline.load_english_dataset(validation_path)
        else:
            raise ValueError(f"Unsupported validation language: {language}")
        
        # Create data loader
        return self.data_pipeline.create_data_loader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False  # Don't drop last batch for validation
        )
    
    def estimate_memory_usage(self, dataset_size: int) -> Dict[str, float]:
        """Estimate memory usage for data loading.
        
        Args:
            dataset_size: Size of dataset
            
        Returns:
            Memory usage estimates in GB
        """
        # Rough estimates based on typical values
        avg_sequence_length = self.data_pipeline.max_length * 0.7  # Assume 70% of max length
        bytes_per_token = 8  # 4 bytes for input_ids + 4 bytes for attention_mask
        
        # Memory per sample
        memory_per_sample = avg_sequence_length * bytes_per_token
        
        # Memory for one batch
        batch_memory = memory_per_sample * self.batch_size
        
        # Total dataset memory (if loaded entirely)
        total_memory = memory_per_sample * dataset_size
        
        # Worker memory (approximate)
        worker_memory = batch_memory * self.num_workers * self.prefetch_factor
        
        return {
            "memory_per_sample_mb": memory_per_sample / (1024 * 1024),
            "batch_memory_mb": batch_memory / (1024 * 1024),
            "total_dataset_gb": total_memory / (1024 * 1024 * 1024),
            "worker_memory_mb": worker_memory / (1024 * 1024),
            "estimated_peak_gb": (batch_memory + worker_memory) / (1024 * 1024 * 1024)
        }


class DataLoaderFactory:
    """Factory class for creating data loaders with consistent configuration."""
    
    def __init__(self, tokenizer, config):
        """Initialize DataLoaderFactory.
        
        Args:
            tokenizer: Tokenizer instance
            config: Training configuration
        """
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(
            tokenizer=tokenizer,
            max_length=getattr(config.model_config, 'max_length', 2048),
            debug_samples=getattr(config.data_config, 'debug_samples', None),
            preprocessing_config=getattr(config.data_config, 'preprocessing_config', {})
        )
        
        logger.info("DataLoaderFactory initialized")
    
    def create_training_loader(self, 
                             dataset_path: str,
                             batch_size: Optional[int] = None,
                             debug_samples: Optional[int] = None,
                             **kwargs) -> DataLoader:
        """Create training data loader.
        
        Args:
            dataset_path: Path to training dataset
            batch_size: Batch size (uses config default if None)
            debug_samples: Number of debug samples
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Training DataLoader
        """
        batch_size = batch_size or self.config.training_params.batch_size
        
        # Load dataset
        if debug_samples:
            original_debug = self.data_pipeline.debug_samples
            self.data_pipeline.debug_samples = debug_samples
        
        try:
            dataset = self.data_pipeline.load_tigrinya_dataset(dataset_path)
            
            # Create data loader with error resilience
            return self.data_pipeline.create_error_resilient_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=kwargs.get('shuffle', True),
                num_workers=kwargs.get('num_workers', 0),
                pin_memory=kwargs.get('pin_memory', True),
                drop_last=kwargs.get('drop_last', True)
            )
            
        finally:
            if debug_samples:
                self.data_pipeline.debug_samples = original_debug
    
    def create_validation_loader(self,
                               dataset_path: str,
                               batch_size: Optional[int] = None,
                               **kwargs) -> DataLoader:
        """Create validation data loader.
        
        Args:
            dataset_path: Path to validation dataset
            batch_size: Batch size (uses config default if None)
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Validation DataLoader
        """
        batch_size = batch_size or self.config.training_params.batch_size
        
        # Load dataset
        dataset = self.data_pipeline.load_tigrinya_dataset(dataset_path)
        
        # Create data loader
        return self.data_pipeline.create_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=kwargs.get('shuffle', False),
            num_workers=kwargs.get('num_workers', 0),
            pin_memory=kwargs.get('pin_memory', True),
            drop_last=kwargs.get('drop_last', False)
        )
    
    def create_mixed_language_loader(self,
                                   tigrinya_path: str,
                                   english_path: Optional[str] = None,
                                   batch_size: Optional[int] = None,
                                   mixing_ratios: Optional[List[float]] = None,
                                   **kwargs) -> DataLoader:
        """Create mixed-language data loader.
        
        Args:
            tigrinya_path: Path to Tigrinya dataset
            english_path: Path to English dataset (optional)
            batch_size: Batch size (uses config default if None)
            mixing_ratios: Language mixing ratios
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Mixed-language DataLoader
        """
        batch_size = batch_size or self.config.training_params.batch_size
        
        # Create mixed dataset
        mixed_dataset = self.data_pipeline.create_mixed_dataset(
            tigrinya_path=tigrinya_path,
            english_path=english_path,
            mixing_ratios=mixing_ratios
        )
        
        # Create data loader with error resilience
        return self.data_pipeline.create_error_resilient_dataloader(
            dataset=mixed_dataset,
            batch_size=batch_size,
            shuffle=kwargs.get('shuffle', True),
            num_workers=kwargs.get('num_workers', 0),
            pin_memory=kwargs.get('pin_memory', True),
            drop_last=kwargs.get('drop_last', True)
        )
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded datasets.
        
        Returns:
            Dictionary with data statistics
        """
        return self.data_pipeline.get_pipeline_statistics()