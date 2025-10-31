"""Knowledge preservation techniques for preventing catastrophic forgetting."""

import logging
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

from ..config.base import KnowledgePreservationConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting."""
    
    def __init__(self, model: nn.Module, config: KnowledgePreservationConfig):
        """Initialize EWC with model and configuration.
        
        Args:
            model: Model to apply EWC to
            config: Knowledge preservation configuration
        """
        self.model = model
        self.config = config
        self.fisher_information = {}
        self.optimal_params = {}
        self.regularization_strength = config.regularization_strength
        
        logger.info(f"EWC initialized with regularization strength: {self.regularization_strength}")
    
    def compute_fisher_information(self, dataloader, num_samples: int = 1000) -> None:
        """Compute Fisher Information Matrix for important parameters.
        
        Args:
            dataloader: DataLoader for computing Fisher information
            num_samples: Number of samples to use for Fisher computation
        """
        logger.info(f"Computing Fisher Information Matrix with {num_samples} samples...")
        
        self.model.eval()
        
        # Initialize Fisher information storage
        self.fisher_information = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param.data)
        
        # Compute Fisher information
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            # Move batch to model device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass to compute gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
            
            sample_count += batch['input_ids'].size(0)
        
        # Normalize Fisher information
        for name in self.fisher_information:
            self.fisher_information[name] /= sample_count
        
        logger.info("Fisher Information Matrix computation completed")
    
    def store_optimal_params(self) -> None:
        """Store current model parameters as optimal parameters."""
        logger.info("Storing optimal parameters for EWC...")
        
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
        
        logger.info("Optimal parameters stored")
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss.
        
        Returns:
            EWC loss tensor
        """
        if not self.fisher_information or not self.optimal_params:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                # EWC loss: λ/2 * F_i * (θ_i - θ*_i)^2
                param_diff = param - self.optimal_params[name]
                ewc_loss += (self.fisher_information[name] * param_diff ** 2).sum()
        
        ewc_loss *= self.regularization_strength / 2.0
        return ewc_loss


class KnowledgeDistillation:
    """Knowledge distillation for preserving model outputs."""
    
    def __init__(self, teacher_model: nn.Module, config: KnowledgePreservationConfig):
        """Initialize knowledge distillation.
        
        Args:
            teacher_model: Original model to preserve knowledge from
            config: Knowledge preservation configuration
        """
        self.teacher_model = teacher_model
        self.config = config
        self.temperature = 4.0  # Temperature for softmax
        self.alpha = 0.7  # Weight for distillation loss
        
        # Set teacher model to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info("Knowledge distillation initialized")
    
    def compute_distillation_loss(self, student_logits: torch.Tensor, 
                                 teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            
        Returns:
            Distillation loss tensor
        """
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Compute KL divergence loss
        distillation_loss = F.kl_div(
            student_soft, teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distillation_loss * self.alpha


class MixedBatchTraining:
    """Mixed batch training for bilingual capability maintenance."""
    
    def __init__(self, config: KnowledgePreservationConfig):
        """Initialize mixed batch training.
        
        Args:
            config: Knowledge preservation configuration
        """
        self.config = config
        self.english_weight = config.english_weight
        self.tigrinya_weight = 1.0 - config.english_weight
        
        logger.info(f"Mixed batch training initialized - English weight: {self.english_weight}, "
                   f"Tigrinya weight: {self.tigrinya_weight}")
    
    def create_mixed_batch(self, tigrinya_batch: Dict[str, torch.Tensor], 
                          english_batch: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Dict[str, torch.Tensor], str]:
        """Create mixed language batch for training.
        
        Args:
            tigrinya_batch: Tigrinya training batch
            english_batch: Optional English training batch
            
        Returns:
            Tuple of (mixed_batch, batch_type)
        """
        if english_batch is None or np.random.random() > self.english_weight:
            # Use Tigrinya batch
            return tigrinya_batch, "tigrinya"
        else:
            # Use English batch
            return english_batch, "english"
    
    def compute_weighted_loss(self, loss: torch.Tensor, batch_type: str) -> torch.Tensor:
        """Apply language-specific weighting to loss.
        
        Args:
            loss: Original loss tensor
            batch_type: Type of batch ("tigrinya" or "english")
            
        Returns:
            Weighted loss tensor
        """
        if batch_type == "english":
            return loss * self.english_weight
        else:
            return loss * self.tigrinya_weight


class KnowledgePreservationManager:
    """Main manager for all knowledge preservation techniques."""
    
    def __init__(self, model: nn.Module, config: KnowledgePreservationConfig, 
                 original_model: Optional[nn.Module] = None):
        """Initialize knowledge preservation manager.
        
        Args:
            model: Model being trained
            config: Knowledge preservation configuration
            original_model: Original model for knowledge distillation
        """
        self.model = model
        self.config = config
        self.enabled = config.enabled
        
        # Initialize components
        self.ewc = ElasticWeightConsolidation(model, config) if self.enabled else None
        self.kd = KnowledgeDistillation(original_model, config) if (self.enabled and original_model) else None
        self.mixed_batch = MixedBatchTraining(config) if self.enabled else None
        
        # Tracking
        self.english_validation_history = []
        self.tigrinya_validation_history = []
        
        logger.info(f"Knowledge preservation manager initialized - Enabled: {self.enabled}")
    
    def setup_preservation(self, english_dataloader, num_fisher_samples: int = 1000) -> None:
        """Setup knowledge preservation components.
        
        Args:
            english_dataloader: DataLoader for English data (for Fisher computation)
            num_fisher_samples: Number of samples for Fisher information
        """
        if not self.enabled:
            logger.info("Knowledge preservation disabled")
            return
        
        logger.info("Setting up knowledge preservation...")
        
        # Compute Fisher information and store optimal parameters
        if self.ewc and english_dataloader:
            self.ewc.compute_fisher_information(english_dataloader, num_fisher_samples)
            self.ewc.store_optimal_params()
        
        logger.info("Knowledge preservation setup completed")
    
    def apply_preservation_loss(self, base_loss: torch.Tensor, batch_type: str,
                               student_logits: Optional[torch.Tensor] = None,
                               teacher_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply knowledge preservation techniques to the loss.
        
        Args:
            base_loss: Original training loss
            batch_type: Type of batch ("tigrinya" or "english")
            student_logits: Student model logits (for distillation)
            teacher_logits: Teacher model logits (for distillation)
            
        Returns:
            Modified loss with preservation techniques applied
        """
        if not self.enabled:
            return base_loss
        
        total_loss = base_loss
        
        # Apply mixed batch weighting
        if self.mixed_batch:
            total_loss = self.mixed_batch.compute_weighted_loss(total_loss, batch_type)
        
        # Add EWC regularization
        if self.ewc:
            ewc_loss = self.ewc.compute_ewc_loss()
            total_loss += ewc_loss
        
        # Add knowledge distillation loss
        if self.kd and student_logits is not None and teacher_logits is not None:
            kd_loss = self.kd.compute_distillation_loss(student_logits, teacher_logits)
            total_loss += kd_loss
        
        return total_loss
    
    def create_mixed_batch(self, tigrinya_batch: Dict[str, torch.Tensor], 
                          english_batch: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Dict[str, torch.Tensor], str]:
        """Create mixed language batch for training.
        
        Args:
            tigrinya_batch: Tigrinya training batch
            english_batch: Optional English training batch
            
        Returns:
            Tuple of (mixed_batch, batch_type)
        """
        if not self.enabled or not self.mixed_batch:
            return tigrinya_batch, "tigrinya"
        
        return self.mixed_batch.create_mixed_batch(tigrinya_batch, english_batch)
    
    def track_validation_performance(self, tigrinya_metrics: Dict[str, float], 
                                   english_metrics: Optional[Dict[str, float]] = None) -> None:
        """Track validation performance for both languages.
        
        Args:
            tigrinya_metrics: Tigrinya validation metrics
            english_metrics: Optional English validation metrics
        """
        self.tigrinya_validation_history.append(tigrinya_metrics)
        
        if english_metrics:
            self.english_validation_history.append(english_metrics)
    
    def check_catastrophic_forgetting(self, threshold: float = 0.1) -> Dict[str, Any]:
        """Check for signs of catastrophic forgetting.
        
        Args:
            threshold: Threshold for performance degradation
            
        Returns:
            Dictionary with forgetting analysis
        """
        if len(self.english_validation_history) < 2:
            return {"status": "insufficient_data", "message": "Not enough validation history"}
        
        # Compare recent performance to initial performance
        initial_perplexity = self.english_validation_history[0].get("perplexity", float('inf'))
        recent_perplexity = self.english_validation_history[-1].get("perplexity", float('inf'))
        
        degradation = (recent_perplexity - initial_perplexity) / initial_perplexity
        
        if degradation > threshold:
            return {
                "status": "forgetting_detected",
                "degradation": degradation,
                "initial_perplexity": initial_perplexity,
                "recent_perplexity": recent_perplexity,
                "message": f"English performance degraded by {degradation:.2%}"
            }
        else:
            return {
                "status": "stable",
                "degradation": degradation,
                "message": f"English performance stable (change: {degradation:.2%})"
            }
    
    def get_preservation_stats(self) -> Dict[str, Any]:
        """Get knowledge preservation statistics.
        
        Returns:
            Dictionary with preservation statistics
        """
        stats = {
            "enabled": self.enabled,
            "english_weight": self.config.english_weight if self.enabled else 0.0,
            "regularization_strength": self.config.regularization_strength if self.enabled else 0.0,
            "validation_history_length": len(self.english_validation_history),
            "components": {
                "ewc": self.ewc is not None,
                "knowledge_distillation": self.kd is not None,
                "mixed_batch_training": self.mixed_batch is not None
            }
        }
        
        # Add forgetting analysis if available
        if len(self.english_validation_history) >= 2:
            stats["forgetting_analysis"] = self.check_catastrophic_forgetting()
        
        return stats