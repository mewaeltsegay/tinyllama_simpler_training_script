"""Logging utilities and configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the training system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name. If None, uses timestamp-based name
        log_dir: Directory to store log files
        include_timestamp: Whether to include timestamp in log messages
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"tigrinya_training_{timestamp}.log"
    
    log_file_path = log_path / log_file
    
    # Configure logging format
    if include_timestamp:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
    else:
        log_format = "%(name)s - %(levelname)s - %(message)s"
        date_format = None
    
    # Setup logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("tigrinya_training")
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    
    return logger


def log_config(logger: logging.Logger, config: dict, config_name: str = "Configuration") -> None:
    """
    Log configuration parameters in a readable format.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary to log
        config_name: Name of the configuration for logging
    """
    logger.info(f"=== {config_name} ===")
    config_str = json.dumps(config, indent=2, default=str)
    for line in config_str.split('\n'):
        logger.info(line)
    logger.info("=" * (len(config_name) + 8))


def log_hardware_info(logger: logging.Logger, hardware_info: dict) -> None:
    """
    Log hardware information.
    
    Args:
        logger: Logger instance
        hardware_info: Hardware information dictionary
    """
    logger.info("=== Hardware Information ===")
    for key, value in hardware_info.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 31)


def log_training_progress(
    logger: logging.Logger,
    step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    tokens_per_second: float,
    gpu_memory_used: float
) -> None:
    """
    Log training progress in a standardized format.
    
    Args:
        logger: Logger instance
        step: Current training step
        total_steps: Total number of training steps
        loss: Current training loss
        learning_rate: Current learning rate
        tokens_per_second: Training throughput
        gpu_memory_used: GPU memory usage in GB
    """
    progress_pct = (step / total_steps) * 100 if total_steps > 0 else 0
    
    logger.info(
        f"Step {step:6d}/{total_steps} ({progress_pct:5.1f}%) | "
        f"Loss: {loss:.4f} | LR: {learning_rate:.2e} | "
        f"Tokens/s: {tokens_per_second:6.1f} | GPU Mem: {gpu_memory_used:.1f}GB"
    )


def log_validation_results(
    logger: logging.Logger,
    step: int,
    tigrinya_loss: float,
    tigrinya_perplexity: float,
    english_loss: Optional[float] = None,
    english_perplexity: Optional[float] = None
) -> None:
    """
    Log validation results for both languages.
    
    Args:
        logger: Logger instance
        step: Current training step
        tigrinya_loss: Tigrinya validation loss
        tigrinya_perplexity: Tigrinya validation perplexity
        english_loss: Optional English validation loss
        english_perplexity: Optional English validation perplexity
    """
    logger.info(f"=== Validation Results (Step {step}) ===")
    logger.info(f"Tigrinya - Loss: {tigrinya_loss:.4f}, Perplexity: {tigrinya_perplexity:.2f}")
    
    if english_loss is not None and english_perplexity is not None:
        logger.info(f"English  - Loss: {english_loss:.4f}, Perplexity: {english_perplexity:.2f}")
    
    logger.info("=" * 40)


class MetricsLogger:
    """Class for structured metrics logging to JSON files."""
    
    def __init__(self, metrics_file: str = "training_metrics.jsonl"):
        """
        Initialize metrics logger.
        
        Args:
            metrics_file: Path to metrics file (JSONL format)
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_metrics(self, step: int, metrics: dict) -> None:
        """
        Log metrics to JSONL file.
        
        Args:
            step: Training step
            metrics: Dictionary of metrics to log
        """
        metrics_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")
    
    def load_metrics(self) -> list:
        """
        Load all metrics from the file.
        
        Returns:
            List of metrics dictionaries
        """
        if not self.metrics_file.exists():
            return []
        
        metrics = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        
        return metrics