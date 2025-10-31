"""Training monitoring and validation system."""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import psutil
import GPUtil
from collections import defaultdict, deque
from datetime import datetime
import threading
from queue import Queue

from ..config.base import TrainingConfig, TrainingMetrics, ValidationMetrics
from ..utils.logging import get_logger
from .experiment_tracking import ExperimentTracker

logger = get_logger(__name__)


class HardwareMonitor:
    """Monitor hardware usage during training."""
    
    def __init__(self):
        """Initialize hardware monitor."""
        self.gpu_available = torch.cuda.is_available()
        self.cpu_count = psutil.cpu_count()
        
        # History tracking
        self.gpu_memory_history = deque(maxlen=100)
        self.gpu_utilization_history = deque(maxlen=100)
        self.cpu_utilization_history = deque(maxlen=100)
        self.system_memory_history = deque(maxlen=100)
        
        logger.info(f"Hardware monitor initialized - GPU available: {self.gpu_available}")
    
    def get_gpu_stats(self) -> Dict[str, float]:
        """Get current GPU statistics.
        
        Returns:
            Dictionary with GPU statistics
        """
        if not self.gpu_available:
            return {"memory_used_gb": 0.0, "memory_total_gb": 0.0, "utilization_percent": 0.0}
        
        try:
            # PyTorch GPU memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # GPU utilization using GPUtil
            gpu_utilization = 0.0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_utilization = gpus[0].load * 100
            except:
                pass  # GPUtil might not be available
            
            stats = {
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_total_gb": memory_total,
                "memory_used_percent": (memory_reserved / memory_total) * 100,
                "utilization_percent": gpu_utilization
            }
            
            # Update history
            self.gpu_memory_history.append(memory_reserved)
            self.gpu_utilization_history.append(gpu_utilization)
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get GPU stats: {e}")
            return {"memory_used_gb": 0.0, "memory_total_gb": 0.0, "utilization_percent": 0.0}
    
    def get_cpu_stats(self) -> Dict[str, float]:
        """Get current CPU statistics.
        
        Returns:
            Dictionary with CPU statistics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            stats = {
                "cpu_utilization_percent": cpu_percent,
                "system_memory_used_gb": memory.used / 1024**3,
                "system_memory_total_gb": memory.total / 1024**3,
                "system_memory_used_percent": memory.percent
            }
            
            # Update history
            self.cpu_utilization_history.append(cpu_percent)
            self.system_memory_history.append(memory.used / 1024**3)
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get CPU stats: {e}")
            return {"cpu_utilization_percent": 0.0, "system_memory_used_gb": 0.0}
    
    def get_hardware_summary(self) -> Dict[str, Any]:
        """Get comprehensive hardware summary.
        
        Returns:
            Dictionary with hardware summary
        """
        gpu_stats = self.get_gpu_stats()
        cpu_stats = self.get_cpu_stats()
        
        summary = {
            "timestamp": time.time(),
            "gpu": gpu_stats,
            "cpu": cpu_stats,
            "averages": {
                "gpu_memory_avg_gb": sum(self.gpu_memory_history) / len(self.gpu_memory_history) if self.gpu_memory_history else 0.0,
                "gpu_utilization_avg_percent": sum(self.gpu_utilization_history) / len(self.gpu_utilization_history) if self.gpu_utilization_history else 0.0,
                "cpu_utilization_avg_percent": sum(self.cpu_utilization_history) / len(self.cpu_utilization_history) if self.cpu_utilization_history else 0.0,
                "system_memory_avg_gb": sum(self.system_memory_history) / len(self.system_memory_history) if self.system_memory_history else 0.0
            }
        }
        
        return summary


class ValidationRunner:
    """Run validation on both Tigrinya and English datasets."""
    
    def __init__(self, model: nn.Module, device: str):
        """Initialize validation runner.
        
        Args:
            model: Model to validate
            device: Device to run validation on
        """
        self.model = model
        self.device = device
        
        logger.info("Validation runner initialized")
    
    def run_validation(self, dataloader: DataLoader, language: str = "tigrinya") -> ValidationMetrics:
        """Run validation on a dataset.
        
        Args:
            dataloader: DataLoader for validation data
            language: Language being validated ("tigrinya" or "english")
            
        Returns:
            Validation metrics
        """
        logger.info(f"Running {language} validation...")
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Accumulate loss
                batch_size = batch['input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Log progress for large validation sets
                if batch_idx % 100 == 0 and batch_idx > 0:
                    logger.debug(f"Validation batch {batch_idx}, current avg loss: {total_loss / total_samples:.4f}")
        
        # Calculate final metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        if language == "tigrinya":
            metrics = ValidationMetrics(
                step=0,  # Will be set by caller
                tigrinya_loss=avg_loss,
                tigrinya_perplexity=perplexity
            )
        else:
            metrics = ValidationMetrics(
                step=0,  # Will be set by caller
                tigrinya_loss=0.0,
                tigrinya_perplexity=0.0,
                english_loss=avg_loss,
                english_perplexity=perplexity
            )
        
        logger.info(f"{language.capitalize()} validation completed - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        return metrics


class RealTimeMetricsCollector:
    """Real-time metrics collection with structured logging."""
    
    def __init__(self, save_dir: str = "logs", collection_interval: float = 1.0):
        """Initialize real-time metrics collector.
        
        Args:
            save_dir: Directory to save metrics
            collection_interval: Interval in seconds for hardware metrics collection
        """
        self.save_dir = save_dir
        self.collection_interval = collection_interval
        os.makedirs(save_dir, exist_ok=True)
        
        # Metrics storage with thread-safe queues
        self.training_metrics = []
        self.validation_metrics = []
        self.hardware_metrics = []
        self.metrics_queue = Queue()
        
        # Files for continuous logging with timestamps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_log_file = os.path.join(save_dir, f"training_metrics_{timestamp}.jsonl")
        self.validation_log_file = os.path.join(save_dir, f"validation_metrics_{timestamp}.jsonl")
        self.hardware_log_file = os.path.join(save_dir, f"hardware_metrics_{timestamp}.jsonl")
        self.realtime_log_file = os.path.join(save_dir, f"realtime_metrics_{timestamp}.jsonl")
        
        # Real-time collection state
        self.collecting = False
        self.collection_thread = None
        self.hardware_monitor = None
        
        # Performance tracking
        self.step_start_time = None
        self.tokens_processed = 0
        self.last_throughput_calculation = time.time()
        
        logger.info(f"Real-time metrics collector initialized - Save dir: {save_dir}")
    
    def start_collection(self, hardware_monitor: 'HardwareMonitor') -> None:
        """Start real-time metrics collection.
        
        Args:
            hardware_monitor: Hardware monitor instance
        """
        if self.collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.hardware_monitor = hardware_monitor
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Real-time metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop real-time metrics collection."""
        if not self.collecting:
            return
        
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        logger.info("Real-time metrics collection stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop running in separate thread."""
        while self.collecting:
            try:
                # Collect hardware metrics
                if self.hardware_monitor:
                    hardware_stats = self.hardware_monitor.get_hardware_summary()
                    hardware_stats['collection_timestamp'] = datetime.now().isoformat()
                    hardware_stats['collection_type'] = 'realtime'
                    
                    # Log to real-time file
                    self._append_to_jsonl(self.realtime_log_file, hardware_stats)
                    
                    # Add to queue for processing
                    self.metrics_queue.put(('hardware', hardware_stats))
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def _append_to_jsonl(self, filepath: str, data: Dict[str, Any]) -> None:
        """Append data to JSONL file with error handling.
        
        Args:
            filepath: Path to JSONL file
            data: Data to append
        """
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, default=str, ensure_ascii=False) + '\n')
                f.flush()  # Ensure immediate write
        except Exception as e:
            logger.warning(f"Failed to write to {filepath}: {e}")
    
    def mark_step_start(self, step: int, batch_size: int) -> None:
        """Mark the start of a training step.
        
        Args:
            step: Training step number
            batch_size: Current batch size
        """
        self.step_start_time = time.time()
        self.current_step = step
        self.current_batch_size = batch_size
    
    def calculate_throughput(self, sequence_length: int) -> float:
        """Calculate tokens per second throughput.
        
        Args:
            sequence_length: Average sequence length in current batch
            
        Returns:
            Tokens per second
        """
        if self.step_start_time is None:
            return 0.0
        
        step_duration = time.time() - self.step_start_time
        if step_duration <= 0:
            return 0.0
        
        tokens_in_batch = self.current_batch_size * sequence_length
        tokens_per_second = tokens_in_batch / step_duration
        
        # Update running totals
        self.tokens_processed += tokens_in_batch
        
        return tokens_per_second


class MetricsTracker:
    """Enhanced metrics tracker with structured logging and timestamps."""
    
    def __init__(self, save_dir: str = "logs"):
        """Initialize metrics tracker.
        
        Args:
            save_dir: Directory to save metrics
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize real-time collector
        self.realtime_collector = RealTimeMetricsCollector(save_dir)
        
        # Metrics storage
        self.training_metrics = []
        self.validation_metrics = []
        self.hardware_metrics = []
        
        # Files for continuous logging with timestamps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_log_file = os.path.join(save_dir, f"training_metrics_{timestamp}.jsonl")
        self.validation_log_file = os.path.join(save_dir, f"validation_metrics_{timestamp}.jsonl")
        self.hardware_log_file = os.path.join(save_dir, f"hardware_metrics_{timestamp}.jsonl")
        
        # Create summary log file
        self.summary_log_file = os.path.join(save_dir, f"training_summary_{timestamp}.json")
        
        logger.info(f"Enhanced metrics tracker initialized - Save dir: {save_dir}")
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics with structured timestamps.
        
        Args:
            metrics: Training metrics to log
        """
        # Add to memory
        self.training_metrics.append(metrics)
        
        # Create structured metrics with detailed timestamps
        current_time = datetime.now()
        metrics_dict = {
            "timestamp": current_time.isoformat(),
            "unix_timestamp": time.time(),
            "step": metrics.step,
            "loss": metrics.loss,
            "learning_rate": metrics.learning_rate,
            "gpu_memory_used": metrics.gpu_memory_used,
            "tokens_per_second": metrics.tokens_per_second,
            "tigrinya_perplexity": metrics.tigrinya_perplexity,
            "english_perplexity": metrics.english_perplexity,
            "gradient_norm": metrics.gradient_norm,
            "log_type": "training_step",
            "session_id": self._get_session_id()
        }
        
        # Save to file
        self._append_to_jsonl(self.training_log_file, metrics_dict)
        
        # Enhanced console logging with timestamps
        if metrics.step % 10 == 0:
            timestamp_str = current_time.strftime("%H:%M:%S")
            logger.info(f"[{timestamp_str}] Step {metrics.step:6d}: "
                       f"Loss={metrics.loss:.4f}, "
                       f"LR={metrics.learning_rate:.2e}, "
                       f"GPU={metrics.gpu_memory_used:.1f}GB, "
                       f"Tokens/s={metrics.tokens_per_second:.1f}, "
                       f"PPL={metrics.tigrinya_perplexity:.2f}")
        
        # Log detailed metrics every 100 steps
        if metrics.step % 100 == 0:
            self._log_detailed_training_metrics(metrics, current_time)
    
    def log_validation_metrics(self, metrics: ValidationMetrics) -> None:
        """Log validation metrics with structured timestamps.
        
        Args:
            metrics: Validation metrics to log
        """
        # Add to memory
        self.validation_metrics.append(metrics)
        
        # Create structured metrics with detailed timestamps
        current_time = datetime.now()
        metrics_dict = {
            "timestamp": current_time.isoformat(),
            "unix_timestamp": time.time(),
            "step": metrics.step,
            "tigrinya_loss": metrics.tigrinya_loss,
            "tigrinya_perplexity": metrics.tigrinya_perplexity,
            "english_loss": metrics.english_loss,
            "english_perplexity": metrics.english_perplexity,
            "log_type": "validation",
            "session_id": self._get_session_id()
        }
        
        # Save to file
        self._append_to_jsonl(self.validation_log_file, metrics_dict)
        
        # Enhanced console logging with timestamps
        timestamp_str = current_time.strftime("%H:%M:%S")
        english_ppl_str = f"{metrics.english_perplexity:.2f}" if metrics.english_perplexity else "N/A"
        
        logger.info(f"[{timestamp_str}] Validation at step {metrics.step}: "
                   f"Tigrinya PPL={metrics.tigrinya_perplexity:.2f}, "
                   f"English PPL={english_ppl_str}")
        
        # Log detailed validation analysis
        self._log_detailed_validation_metrics(metrics, current_time)
    
    def log_hardware_metrics(self, hardware_stats: Dict[str, Any]) -> None:
        """Log hardware metrics with structured timestamps.
        
        Args:
            hardware_stats: Hardware statistics to log
        """
        # Add structured timestamp information
        current_time = datetime.now()
        enhanced_stats = {
            **hardware_stats,
            "log_timestamp": current_time.isoformat(),
            "unix_timestamp": time.time(),
            "log_type": "hardware_monitoring",
            "session_id": self._get_session_id()
        }
        
        # Add to memory
        self.hardware_metrics.append(enhanced_stats)
        
        # Save to file
        self._append_to_jsonl(self.hardware_log_file, enhanced_stats)
        
        # Log hardware alerts if needed
        self._check_hardware_alerts(enhanced_stats)
    
    def _append_to_jsonl(self, filepath: str, data: Dict[str, Any]) -> None:
        """Append data to JSONL file with error handling.
        
        Args:
            filepath: Path to JSONL file
            data: Data to append
        """
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, default=str, ensure_ascii=False) + '\n')
                f.flush()  # Ensure immediate write
        except Exception as e:
            logger.warning(f"Failed to write to {filepath}: {e}")
    
    def _get_session_id(self) -> str:
        """Get unique session identifier.
        
        Returns:
            Session ID string
        """
        if not hasattr(self, '_session_id'):
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._session_id
    
    def _log_detailed_training_metrics(self, metrics: TrainingMetrics, timestamp: datetime) -> None:
        """Log detailed training metrics analysis.
        
        Args:
            metrics: Training metrics
            timestamp: Current timestamp
        """
        # Calculate moving averages if we have enough data
        if len(self.training_metrics) >= 10:
            recent_losses = [m.loss for m in self.training_metrics[-10:]]
            recent_throughput = [m.tokens_per_second for m in self.training_metrics[-10:]]
            
            avg_loss = sum(recent_losses) / len(recent_losses)
            avg_throughput = sum(recent_throughput) / len(recent_throughput)
            
            logger.info(f"[{timestamp.strftime('%H:%M:%S')}] Training Analysis - "
                       f"Avg Loss (10 steps): {avg_loss:.4f}, "
                       f"Avg Throughput: {avg_throughput:.1f} tokens/s")
    
    def _log_detailed_validation_metrics(self, metrics: ValidationMetrics, timestamp: datetime) -> None:
        """Log detailed validation metrics analysis.
        
        Args:
            metrics: Validation metrics
            timestamp: Current timestamp
        """
        # Analyze validation trends
        if len(self.validation_metrics) >= 2:
            prev_metrics = self.validation_metrics[-2]
            
            tigrinya_trend = "↓" if metrics.tigrinya_perplexity < prev_metrics.tigrinya_perplexity else "↑"
            
            trend_info = f"Tigrinya {tigrinya_trend}"
            
            if metrics.english_perplexity and prev_metrics.english_perplexity:
                english_trend = "↓" if metrics.english_perplexity < prev_metrics.english_perplexity else "↑"
                trend_info += f", English {english_trend}"
            
            logger.info(f"[{timestamp.strftime('%H:%M:%S')}] Validation Trends: {trend_info}")
    
    def _check_hardware_alerts(self, hardware_stats: Dict[str, Any]) -> None:
        """Check for hardware-related alerts.
        
        Args:
            hardware_stats: Hardware statistics
        """
        gpu_stats = hardware_stats.get('gpu', {})
        cpu_stats = hardware_stats.get('cpu', {})
        
        # GPU memory alerts
        gpu_memory_percent = gpu_stats.get('memory_used_percent', 0)
        if gpu_memory_percent > 90:
            logger.warning(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
        
        # GPU utilization alerts
        gpu_utilization = gpu_stats.get('utilization_percent', 0)
        if gpu_utilization < 50:
            logger.warning(f"Low GPU utilization: {gpu_utilization:.1f}%")
        
        # System memory alerts
        system_memory_percent = cpu_stats.get('system_memory_used_percent', 0)
        if system_memory_percent > 85:
            logger.warning(f"High system memory usage: {system_memory_percent:.1f}%")
    
    def start_realtime_collection(self, hardware_monitor: 'HardwareMonitor') -> None:
        """Start real-time metrics collection.
        
        Args:
            hardware_monitor: Hardware monitor instance
        """
        self.realtime_collector.start_collection(hardware_monitor)
    
    def stop_realtime_collection(self) -> None:
        """Stop real-time metrics collection."""
        self.realtime_collector.stop_collection()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary statistics.
        
        Returns:
            Dictionary with training summary
        """
        if not self.training_metrics:
            return {"status": "no_data", "timestamp": datetime.now().isoformat()}
        
        recent_metrics = self.training_metrics[-10:]  # Last 10 steps
        all_metrics = self.training_metrics
        
        # Calculate comprehensive statistics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self._get_session_id(),
            "status": "active",
            "total_steps": len(all_metrics),
            "current_step": all_metrics[-1].step,
            "current_loss": all_metrics[-1].loss,
            "current_learning_rate": all_metrics[-1].learning_rate,
            "current_gpu_memory": all_metrics[-1].gpu_memory_used,
            
            # Recent averages (last 10 steps)
            "recent_metrics": {
                "average_loss": sum(m.loss for m in recent_metrics) / len(recent_metrics),
                "average_gpu_memory": sum(m.gpu_memory_used for m in recent_metrics) / len(recent_metrics),
                "average_tokens_per_second": sum(m.tokens_per_second for m in recent_metrics) / len(recent_metrics),
                "average_gradient_norm": sum(m.gradient_norm for m in recent_metrics) / len(recent_metrics)
            },
            
            # Overall statistics
            "overall_metrics": {
                "min_loss": min(m.loss for m in all_metrics),
                "max_loss": max(m.loss for m in all_metrics),
                "average_loss": sum(m.loss for m in all_metrics) / len(all_metrics),
                "max_gpu_memory": max(m.gpu_memory_used for m in all_metrics),
                "average_throughput": sum(m.tokens_per_second for m in all_metrics) / len(all_metrics)
            },
            
            # Training progress
            "progress": {
                "loss_trend": self._calculate_trend([m.loss for m in all_metrics[-20:]]),
                "throughput_trend": self._calculate_trend([m.tokens_per_second for m in all_metrics[-20:]]),
                "memory_trend": self._calculate_trend([m.gpu_memory_used for m in all_metrics[-20:]])
            }
        }
        
        # Save summary to file
        self._save_training_summary(summary)
        
        return summary
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics.
        
        Returns:
            Dictionary with validation summary
        """
        if not self.validation_metrics:
            return {"status": "no_data"}
        
        latest = self.validation_metrics[-1]
        
        summary = {
            "total_validations": len(self.validation_metrics),
            "latest_step": latest.step,
            "latest_tigrinya_perplexity": latest.tigrinya_perplexity,
            "latest_english_perplexity": latest.english_perplexity,
            "tigrinya_perplexity_trend": self._calculate_trend([m.tigrinya_perplexity for m in self.validation_metrics[-5:]]),
            "english_perplexity_trend": self._calculate_trend([m.english_perplexity for m in self.validation_metrics[-5:] if m.english_perplexity is not None])
        }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values.
        
        Args:
            values: List of values to analyze
            
        Returns:
            Trend direction ("improving", "degrading", "stable", "insufficient_data")
        """
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        if len(values) >= 3:
            recent_avg = sum(values[-2:]) / 2
            earlier_avg = sum(values[:-2]) / (len(values) - 2)
            
            if recent_avg < earlier_avg * 0.95:
                return "improving"
            elif recent_avg > earlier_avg * 1.05:
                return "degrading"
            else:
                return "stable"
        else:
            if values[-1] < values[0]:
                return "improving"
            elif values[-1] > values[0]:
                return "degrading"
            else:
                return "stable"
    
    def _save_training_summary(self, summary: Dict[str, Any]) -> None:
        """Save training summary to file.
        
        Args:
            summary: Training summary to save
        """
        try:
            with open(self.summary_log_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save training summary: {e}")
    
    def create_final_report(self) -> Dict[str, Any]:
        """Create comprehensive final training report.
        
        Returns:
            Final training report
        """
        summary = self.get_training_summary()
        validation_summary = self.get_validation_summary()
        
        # Calculate training duration
        if self.training_metrics:
            first_metric = self.training_metrics[0]
            last_metric = self.training_metrics[-1]
            
            # Estimate duration based on step progression
            total_steps = last_metric.step - first_metric.step + 1
            avg_throughput = summary.get('overall_metrics', {}).get('average_throughput', 0)
            
            report = {
                "final_report": True,
                "timestamp": datetime.now().isoformat(),
                "session_id": self._get_session_id(),
                "training_completed": True,
                "total_training_steps": total_steps,
                "final_loss": last_metric.loss,
                "final_learning_rate": last_metric.learning_rate,
                "final_tigrinya_perplexity": last_metric.tigrinya_perplexity,
                "final_english_perplexity": last_metric.english_perplexity,
                "average_throughput": avg_throughput,
                "peak_gpu_memory": summary.get('overall_metrics', {}).get('max_gpu_memory', 0),
                "training_summary": summary,
                "validation_summary": validation_summary,
                "hardware_summary": self.hardware_metrics[-1] if self.hardware_metrics else None
            }
            
            # Save final report
            final_report_file = os.path.join(self.save_dir, f"final_report_{self._get_session_id()}.json")
            try:
                with open(final_report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str, ensure_ascii=False)
                logger.info(f"Final training report saved to {final_report_file}")
            except Exception as e:
                logger.warning(f"Failed to save final report: {e}")
            
            return report
        
        return {"final_report": True, "status": "no_training_data"}


class TrainingMonitor:
    """Main training monitoring system with experiment tracking."""
    
    def __init__(self, config: TrainingConfig, model: nn.Module, device: str):
        """Initialize training monitor.
        
        Args:
            config: Training configuration
            model: Model being trained
            device: Device being used
        """
        self.config = config
        self.model = model
        self.device = device
        
        # Initialize components
        self.hardware_monitor = HardwareMonitor()
        self.validation_runner = ValidationRunner(model, device)
        self.metrics_tracker = MetricsTracker(config.logging_config.tensorboard_dir)
        self.experiment_tracker = ExperimentTracker(config)
        
        # Validation datasets
        self.tigrinya_val_loader = None
        self.english_val_loader = None
        
        # Timing
        self.last_validation_step = 0
        self.last_checkpoint_step = 0
        
        logger.info("Training monitor with experiment tracking initialized")
    
    def set_validation_loaders(self, tigrinya_loader: Optional[DataLoader], 
                             english_loader: Optional[DataLoader]) -> None:
        """Set validation data loaders.
        
        Args:
            tigrinya_loader: Tigrinya validation DataLoader
            english_loader: English validation DataLoader
        """
        self.tigrinya_val_loader = tigrinya_loader
        self.english_val_loader = english_loader
        
        logger.info(f"Validation loaders set - Tigrinya: {tigrinya_loader is not None}, "
                   f"English: {english_loader is not None}")
    
    def log_training_step(self, metrics: TrainingMetrics) -> None:
        """Log training step metrics with enhanced monitoring and experiment tracking.
        
        Args:
            metrics: Training metrics to log
        """
        # Log training metrics with structured timestamps
        self.metrics_tracker.log_training_metrics(metrics)
        
        # Log to experiment tracking platforms
        self.experiment_tracker.log_training_metrics(metrics)
        
        # Log hardware metrics periodically with real-time collection
        if metrics.step % 5 == 0:
            hardware_stats = self.hardware_monitor.get_hardware_summary()
            self.metrics_tracker.log_hardware_metrics(hardware_stats)
            self.experiment_tracker.log_hardware_metrics(hardware_stats)
        
        # Start real-time collection on first step
        if metrics.step == 1:
            self.metrics_tracker.start_realtime_collection(self.hardware_monitor)
            
            # Log model graph to TensorBoard on first step
            try:
                # Create a dummy input for graph logging
                dummy_input = torch.randint(0, 1000, (1, 512)).to(self.device)
                self.experiment_tracker.log_model_graph(self.model, dummy_input)
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
        
        # Run validation if needed
        if self._should_run_validation(metrics.step):
            self.run_validation(metrics.step)
        
        # Save checkpoint if needed
        if self._should_save_checkpoint(metrics.step):
            self.last_checkpoint_step = metrics.step
            logger.info(f"Checkpoint saved at step {metrics.step}")
        
        # Log progress summary every 100 steps
        if metrics.step % 100 == 0:
            summary = self.metrics_tracker.get_training_summary()
            logger.info(f"Training Progress Summary at step {metrics.step}:")
            logger.info(f"  Recent avg loss: {summary['recent_metrics']['average_loss']:.4f}")
            logger.info(f"  Recent avg throughput: {summary['recent_metrics']['average_tokens_per_second']:.1f} tokens/s")
            logger.info(f"  GPU memory usage: {summary['current_gpu_memory']:.1f}GB")
    
    def run_validation(self, step: int) -> None:
        """Run validation on available datasets with experiment tracking.
        
        Args:
            step: Current training step
        """
        logger.info(f"Running validation at step {step}")
        
        # Tigrinya validation
        if self.tigrinya_val_loader:
            tigrinya_metrics = self.validation_runner.run_validation(
                self.tigrinya_val_loader, "tigrinya"
            )
            tigrinya_metrics.step = step
            self.metrics_tracker.log_validation_metrics(tigrinya_metrics)
            self.experiment_tracker.log_validation_metrics(tigrinya_metrics)
        
        # English validation
        if self.english_val_loader:
            english_metrics = self.validation_runner.run_validation(
                self.english_val_loader, "english"
            )
            english_metrics.step = step
            self.metrics_tracker.log_validation_metrics(english_metrics)
            self.experiment_tracker.log_validation_metrics(english_metrics)
        
        self.last_validation_step = step
    
    def _should_run_validation(self, step: int) -> bool:
        """Check if validation should be run at this step.
        
        Args:
            step: Current training step
            
        Returns:
            True if validation should be run
        """
        eval_steps = self.config.training_params.eval_steps
        return (step - self.last_validation_step) >= eval_steps and (
            self.tigrinya_val_loader is not None or self.english_val_loader is not None
        )
    
    def _should_save_checkpoint(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step.
        
        Args:
            step: Current training step
            
        Returns:
            True if checkpoint should be saved
        """
        save_steps = self.config.training_params.save_steps
        return (step - self.last_checkpoint_step) >= save_steps
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary with enhanced metrics.
        
        Returns:
            Dictionary with monitoring summary
        """
        return {
            "training_summary": self.metrics_tracker.get_training_summary(),
            "validation_summary": self.metrics_tracker.get_validation_summary(),
            "hardware_summary": self.hardware_monitor.get_hardware_summary(),
            "last_validation_step": self.last_validation_step,
            "last_checkpoint_step": self.last_checkpoint_step,
            "realtime_collection_active": self.metrics_tracker.realtime_collector.collecting
        }
    
    def log_checkpoint_saved(self, checkpoint_path: str, step: int, metrics: Dict[str, Any]) -> None:
        """Log checkpoint saving to experiment trackers.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            step: Training step
            metrics: Associated metrics
        """
        self.experiment_tracker.log_model_checkpoint(checkpoint_path, step, metrics)
        logger.info(f"Checkpoint logged to experiment trackers: {checkpoint_path}")
    
    def log_text_generation_samples(self, step: int, samples: Dict[str, str]) -> None:
        """Log text generation samples to experiment trackers.
        
        Args:
            step: Training step
            samples: Dictionary of generated text samples (language -> text)
        """
        self.experiment_tracker.log_text_samples(step, samples)
        logger.info(f"Text samples logged to experiment trackers at step {step}")
    
    def finalize_training(self) -> Dict[str, Any]:
        """Finalize training and create comprehensive report with experiment tracking.
        
        Returns:
            Final training report
        """
        logger.info("Finalizing training monitoring...")
        
        # Stop real-time collection
        self.metrics_tracker.stop_realtime_collection()
        
        # Create final report
        final_report = self.metrics_tracker.create_final_report()
        
        # Create experiment tracking summary
        experiment_summary = self.experiment_tracker.create_training_summary(final_report)
        final_report["experiment_tracking_summary"] = experiment_summary
        
        # Finalize experiment trackers
        self.experiment_tracker.finalize()
        
        logger.info("Training monitoring and experiment tracking finalized")
        return final_report