"""Configuration management with JSON schema validation."""

import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict, fields

from .base import (
    TrainingConfig, ModelConfig, TrainingParams, DataConfig, 
    HardwareConfig, KnowledgePreservationConfig, LoggingConfig,
    BaseConfigManager
)
from ..utils.helpers import load_json, save_json, merge_configs
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConfigManager(BaseConfigManager):
    """Configuration manager with JSON schema validation."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self._schema = self._create_json_schema()
    
    def _create_json_schema(self) -> Dict[str, Any]:
        """Create JSON schema for configuration validation."""
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {
                        "checkpoint_path": {"type": "string"},
                        "tokenizer_path": {"type": "string"},
                        "vocab_size": {"type": "integer", "minimum": 1000}
                    },
                    "required": ["checkpoint_path", "tokenizer_path"],
                    "additionalProperties": False
                },
                "training": {
                    "type": "object",
                    "properties": {
                        "learning_rate": {"type": "number", "minimum": 1e-8, "maximum": 1.0},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "gradient_accumulation_steps": {"type": "integer", "minimum": 1},
                        "max_steps": {"type": "integer", "minimum": 1},
                        "warmup_steps": {"type": "integer", "minimum": 0},
                        "save_steps": {"type": "integer", "minimum": 1},
                        "eval_steps": {"type": "integer", "minimum": 1},
                        "mixed_precision": {"type": "string", "enum": ["fp16", "bf16", "fp32"]},
                        "gradient_checkpointing": {"type": "boolean"}
                    },
                    "additionalProperties": False
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "tigrinya_dataset": {"type": "string"},
                        "validation_dataset": {"type": "string"},
                        "english_validation": {"type": ["string", "null"]},
                        "max_length": {"type": "integer", "minimum": 1, "maximum": 8192},
                        "debug_samples": {"type": ["integer", "null"], "minimum": 1}
                    },
                    "required": ["tigrinya_dataset", "validation_dataset"],
                    "additionalProperties": False
                },
                "hardware": {
                    "type": "object",
                    "properties": {
                        "device": {"type": "string"},
                        "num_gpus": {"type": "integer", "minimum": 1},
                        "dataloader_workers": {"type": "integer", "minimum": 0},
                        "pin_memory": {"type": "boolean"}
                    },
                    "additionalProperties": False
                },
                "knowledge_preservation": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "english_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "regularization_strength": {"type": "number", "minimum": 0.0},
                        "validation_frequency": {"type": "integer", "minimum": 1}
                    },
                    "additionalProperties": False
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
                        "wandb_project": {"type": ["string", "null"]},
                        "tensorboard_dir": {"type": "string"},
                        "save_metrics": {"type": "boolean"}
                    },
                    "additionalProperties": False
                }
            },
            "required": ["model", "data"],
            "additionalProperties": False
        }
    
    def load_config(self, config_path: str) -> TrainingConfig:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            TrainingConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            jsonschema.ValidationError: If config doesn't match schema
            ValueError: If config contains invalid values
        """
        config_data = load_json(config_path)
        
        # Validate against schema
        self.validate_json_schema(config_data)
        
        # Create configuration objects with defaults
        model_config = ModelConfig(**config_data["model"])
        
        training_data = config_data.get("training", {})
        training_params = TrainingParams(**training_data)
        
        data_config = DataConfig(**config_data["data"])
        
        hardware_data = config_data.get("hardware", {})
        hardware_config = HardwareConfig(**hardware_data)
        
        knowledge_data = config_data.get("knowledge_preservation", {})
        knowledge_preservation = KnowledgePreservationConfig(**knowledge_data)
        
        logging_data = config_data.get("logging", {})
        logging_config = LoggingConfig(**logging_data)
        
        config = TrainingConfig(
            model_config=model_config,
            training_params=training_params,
            data_config=data_config,
            hardware_config=hardware_config,
            knowledge_preservation=knowledge_preservation,
            logging_config=logging_config
        )
        
        # Additional validation
        self.validate_config(config)
        
        return config
    
    def validate_json_schema(self, config_data: Dict[str, Any]) -> None:
        """
        Validate configuration data against JSON schema.
        
        Args:
            config_data: Configuration dictionary
            
        Raises:
            jsonschema.ValidationError: If validation fails
        """
        try:
            jsonschema.validate(config_data, self._schema)
        except jsonschema.ValidationError as e:
            raise jsonschema.ValidationError(
                f"Configuration validation failed: {e.message}\n"
                f"Failed at path: {' -> '.join(str(p) for p in e.absolute_path)}"
            )
    
    def validate_config(self, config: TrainingConfig) -> bool:
        """
        Validate configuration object for logical consistency.
        
        Args:
            config: TrainingConfig object
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate file paths exist (skip for Hugging Face model names)
        model_path = config.model_config.checkpoint_path
        if not model_path.startswith(('TinyLlama/', 'microsoft/', 'meta-llama/', 'huggingface/')):
            # Only validate local paths
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise ValueError(f"Model checkpoint path does not exist: {model_path}")
        
        tokenizer_path = config.model_config.tokenizer_path
        if not tokenizer_path.startswith(('TinyLlama/', 'microsoft/', 'meta-llama/', 'huggingface/')):
            # Only validate local paths
            tokenizer_path_obj = Path(tokenizer_path)
            if not tokenizer_path_obj.exists():
                raise ValueError(f"Tokenizer path does not exist: {tokenizer_path}")
        
        # Validate dataset paths (warn if missing for config-only validation)
        dataset_path = Path(config.data_config.tigrinya_dataset)
        if not dataset_path.exists():
            logger.warning(f"Tigrinya dataset path does not exist: {dataset_path}")
        
        validation_path = Path(config.data_config.validation_dataset)
        if not validation_path.exists():
            logger.warning(f"Validation dataset path does not exist: {validation_path}")
        
        if config.data_config.english_validation:
            english_path = Path(config.data_config.english_validation)
            if not english_path.exists():
                raise ValueError(f"English validation dataset path does not exist: {english_path}")
        
        # Validate training parameters
        if config.training_params.warmup_steps >= config.training_params.max_steps:
            raise ValueError("Warmup steps must be less than max steps")
        
        if config.training_params.save_steps > config.training_params.max_steps:
            raise ValueError("Save steps must be less than or equal to max steps")
        
        if config.training_params.eval_steps > config.training_params.max_steps:
            raise ValueError("Eval steps must be less than or equal to max steps")
        
        # Validate knowledge preservation parameters
        if config.knowledge_preservation.enabled:
            if not (0.0 <= config.knowledge_preservation.english_weight <= 1.0):
                raise ValueError("English weight must be between 0.0 and 1.0")
        
        return True
    
    def save_config(self, config: TrainingConfig, config_path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: TrainingConfig object
            config_path: Output file path
        """
        config_dict = {
            "model": asdict(config.model_config),
            "training": asdict(config.training_params),
            "data": asdict(config.data_config),
            "hardware": asdict(config.hardware_config),
            "knowledge_preservation": asdict(config.knowledge_preservation),
            "logging": asdict(config.logging_config)
        }
        
        save_json(config_dict, config_path)
    
    def merge_configs(self, base_config_path: str, override_config_path: str) -> TrainingConfig:
        """
        Merge two configuration files, with override taking precedence.
        
        Args:
            base_config_path: Path to base configuration
            override_config_path: Path to override configuration
            
        Returns:
            Merged TrainingConfig object
        """
        base_data = load_json(base_config_path)
        override_data = load_json(override_config_path)
        
        merged_data = merge_configs(base_data, override_data)
        
        # Validate merged configuration
        self.validate_json_schema(merged_data)
        
        # Create temporary file to load merged config
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(merged_data, f, indent=2)
            temp_path = f.name
        
        try:
            config = self.load_config(temp_path)
        finally:
            Path(temp_path).unlink()  # Clean up temp file
        
        return config
    
    def create_default_config(self, 
                            model_checkpoint: str,
                            tokenizer_path: str,
                            tigrinya_dataset: str,
                            validation_dataset: str) -> TrainingConfig:
        """
        Create a default configuration with required parameters.
        
        Args:
            model_checkpoint: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            tigrinya_dataset: Path to Tigrinya dataset
            validation_dataset: Path to validation dataset
            
        Returns:
            TrainingConfig with default values
        """
        return TrainingConfig(
            model_config=ModelConfig(
                checkpoint_path=model_checkpoint,
                tokenizer_path=tokenizer_path
            ),
            training_params=TrainingParams(),
            data_config=DataConfig(
                tigrinya_dataset=tigrinya_dataset,
                validation_dataset=validation_dataset
            ),
            hardware_config=HardwareConfig(),
            knowledge_preservation=KnowledgePreservationConfig(),
            logging_config=LoggingConfig()
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for configuration validation.
        
        Returns:
            JSON schema dictionary
        """
        return self._schema.copy()
    
    def validate_partial_config(self, config_data: Dict[str, Any], 
                              required_sections: Optional[list] = None) -> None:
        """
        Validate a partial configuration (useful for testing).
        
        Args:
            config_data: Partial configuration data
            required_sections: List of required sections (default: model, data)
            
        Raises:
            jsonschema.ValidationError: If validation fails
        """
        if required_sections is None:
            required_sections = ["model", "data"]
        
        # Create a modified schema with only required sections
        partial_schema = self._schema.copy()
        partial_schema["required"] = required_sections
        
        jsonschema.validate(config_data, partial_schema)