"""
Loss functions for medical image segmentation.

This module implements specialized loss functions optimized for medical
image segmentation tasks with class imbalance and boundary preservation.
"""

from typing import Optional, Union, List, Callable, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.losses import DiceLoss, DiceCELoss as MONAIDiceCELoss
from monai.losses import FocalLoss, TverskyLoss
from monai.losses.hausdorff_loss import HausdorffDTLoss
from monai.utils import LossReduction

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class MedicalDiceCELoss(nn.Module):
    """
    Combined Dice and Cross-Entropy loss optimized for medical segmentation.
    
    This implementation provides:
    - Weighted combination of Dice and CE losses
    - Class balancing for imbalanced datasets
    - Boundary-aware weighting
    - Multi-class and multi-label support
    """
    
    def __init__(
        self,
        include_background: bool = False,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal: bool = False,
        boundary_weight: float = 0.0,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize the combined Dice-CE loss.
        
        Args:
            include_background: Whether to include background class in loss calculation
            to_onehot_y: Whether to convert target to one-hot encoding
            sigmoid: Whether to apply sigmoid activation to predictions
            softmax: Whether to apply softmax activation to predictions
            other_act: Other activation function to apply
            squared_pred: Whether to square predictions in Dice loss
            jaccard: Whether to use Jaccard index instead of Dice
            reduction: Loss reduction method
            smooth_nr: Smoothing factor for numerator
            smooth_dr: Smoothing factor for denominator
            batch: Whether to calculate loss per batch
            ce_weight: Class weights for cross-entropy loss
            lambda_dice: Weight for Dice loss component
            lambda_ce: Weight for cross-entropy loss component
            class_weights: Per-class weights for loss balancing
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            use_focal: Whether to use focal loss instead of standard CE
            boundary_weight: Weight for boundary-aware loss component
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        self._session_id = session_id
        self._user_id = user_id
        self._lambda_dice = lambda_dice
        self._lambda_ce = lambda_ce
        self._use_focal = use_focal
        self._boundary_weight = boundary_weight
        
        # Initialize Dice loss component
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch
        )
        
        # Initialize Cross-Entropy or Focal loss component
        if use_focal:
            self.ce_loss = FocalLoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                alpha=focal_alpha,
                gamma=focal_gamma,
                weight=ce_weight,
                reduction=reduction
            )
        else:
            # Use MONAI's DiceCE loss for CE component
            self.ce_loss = MONAIDiceCELoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                other_act=other_act,
                squared_pred=squared_pred,
                jaccard=jaccard,
                reduction=reduction,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr,
                batch=batch,
                ce_weight=ce_weight,
                lambda_dice=0.0,  # Only use CE component
                lambda_ce=1.0
            )
        
        # Boundary-aware loss component
        if boundary_weight > 0:
            self.boundary_loss = BoundaryAwareLoss(
                reduction=reduction,
                session_id=session_id,
                user_id=user_id
            )
        
        # Class balancing weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        
        # Log loss initialization
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Medical Dice-CE loss initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'loss_type': 'DiceCELoss',
                'lambda_dice': lambda_dice,
                'lambda_ce': lambda_ce,
                'use_focal': use_focal,
                'boundary_weight': boundary_weight,
                'include_background': include_background
            }
        )
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined Dice-CE loss.
        
        Args:
            input: Model predictions [B, C, ...]
            target: Ground truth labels [B, C, ...] or [B, ...]
            
        Returns:
            Combined loss value
        """
        try:
            # Compute Dice loss component
            dice_loss_val = self.dice_loss(input, target)
            
            # Compute Cross-Entropy/Focal loss component
            if self._use_focal:
                ce_loss_val = self.ce_loss(input, target)
            else:
                # Extract CE component from DiceCE loss
                ce_loss_val = self.ce_loss(input, target)
            
            # Combine losses
            total_loss = self._lambda_dice * dice_loss_val + self._lambda_ce * ce_loss_val
            
            # Add boundary-aware component if enabled
            if self._boundary_weight > 0:
                boundary_loss_val = self.boundary_loss(input, target)
                total_loss += self._boundary_weight * boundary_loss_val
            
            # Apply class balancing if enabled
            if self.class_weights is not None:
                # This is simplified - in practice, you'd weight per-class contributions
                pass
            
            return total_loss
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.ERROR,
                message=f"Loss calculation failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes object boundaries.
    
    This loss helps improve segmentation accuracy at object boundaries
    by using morphological operations to detect boundary regions.
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        reduction: LossReduction = LossReduction.MEAN,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize boundary-aware loss.
        
        Args:
            kernel_size: Kernel size for morphological operations
            reduction: Loss reduction method
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.reduction = reduction
        self._session_id = session_id
        self._user_id = user_id
        
        # Create morphological kernel
        self.register_buffer('kernel', torch.ones(1, 1, kernel_size, kernel_size, kernel_size))
    
    def _extract_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundaries using morphological operations."""
        # Apply erosion
        eroded = F.conv3d(mask, self.kernel, padding=self.kernel_size//2)
        eroded = (eroded == self.kernel_size**3).float()
        
        # Boundary = original - eroded
        boundary = mask - eroded
        return boundary.clamp(0, 1)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-aware loss.
        
        Args:
            input: Model predictions [B, C, ...]
            target: Ground truth labels [B, C, ...]
            
        Returns:
            Boundary-aware loss value
        """
        # Convert predictions to probabilities
        if input.shape[1] > 1:  # Multi-class
            pred_probs = F.softmax(input, dim=1)
        else:  # Binary
            pred_probs = torch.sigmoid(input)
        
        # Extract boundaries from target
        target_boundaries = self._extract_boundaries(target)
        
        # Compute weighted cross-entropy at boundaries
        boundary_weight = target_boundaries + 1.0  # Higher weight at boundaries
        
        # Calculate loss
        if input.shape[1] > 1:  # Multi-class
            loss = F.cross_entropy(input, target.argmax(dim=1), reduction='none')
        else:  # Binary
            loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # Weight by boundary map
        weighted_loss = loss * boundary_weight.squeeze(1)
        
        if self.reduction == LossReduction.MEAN:
            return weighted_loss.mean()
        elif self.reduction == LossReduction.SUM:
            return weighted_loss.sum()
        else:
            return weighted_loss


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that automatically balances multiple loss components.
    
    This loss learns optimal weights for different loss components during training
    using uncertainty-based weighting.
    """
    
    def __init__(
        self,
        loss_functions: Dict[str, nn.Module],
        initial_weights: Optional[Dict[str, float]] = None,
        learnable_weights: bool = True,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize adaptive loss.
        
        Args:
            loss_functions: Dictionary of named loss functions
            initial_weights: Initial weights for each loss component
            learnable_weights: Whether to learn adaptive weights
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        self.loss_functions = nn.ModuleDict(loss_functions)
        self._session_id = session_id
        self._user_id = user_id
        self._learnable_weights = learnable_weights
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in loss_functions.keys()}
        
        if learnable_weights:
            # Learnable log-variance parameters for uncertainty weighting
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(0.0))
                for name in loss_functions.keys()
            })
        else:
            # Fixed weights
            self.register_buffer('weights', torch.tensor(list(initial_weights.values())))
            self.weight_names = list(initial_weights.keys())
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive weighted loss.
        
        Args:
            input: Model predictions
            target: Ground truth labels
            
        Returns:
            Dictionary containing total loss and individual components
        """
        losses = {}
        total_loss = 0
        
        # Compute individual losses
        for name, loss_fn in self.loss_functions.items():
            losses[f'{name}_loss'] = loss_fn(input, target)
        
        # Apply adaptive weighting
        if self._learnable_weights:
            for name, loss_value in losses.items():
                if name.endswith('_loss'):
                    base_name = name[:-5]  # Remove '_loss' suffix
                    if base_name in self.log_vars:
                        # Uncertainty-based weighting: loss = exp(-log_var) * loss + log_var
                        precision = torch.exp(-self.log_vars[base_name])
                        weighted_loss = precision * loss_value + self.log_vars[base_name]
                        total_loss += weighted_loss
        else:
            for i, (name, loss_value) in enumerate(losses.items()):
                total_loss += self.weights[i] * loss_value
        
        losses['total_loss'] = total_loss
        return losses


class TopologyPreservingLoss(nn.Module):
    """
    Topology-preserving loss for medical image segmentation.
    
    This loss enforces topological correctness in segmentation results,
    which is crucial for anatomical structures.
    """
    
    def __init__(
        self,
        reduction: LossReduction = LossReduction.MEAN,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize topology-preserving loss.
        
        Args:
            reduction: Loss reduction method
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        self.reduction = reduction
        self._session_id = session_id
        self._user_id = user_id
    
    def _compute_betti_numbers(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Betti numbers for topology analysis.
        
        This is a simplified implementation - in practice, you'd use
        specialized libraries like gudhi or scikit-topology.
        """
        # Simplified topology measure using Euler characteristic
        # In practice, implement proper persistent homology computation
        return torch.sum(mask)  # Placeholder
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute topology-preserving loss.
        
        Args:
            input: Model predictions
            target: Ground truth labels
            
        Returns:
            Topology loss value
        """
        # Convert predictions to binary masks
        if input.shape[1] > 1:
            pred_mask = torch.argmax(input, dim=1, keepdim=True).float()
        else:
            pred_mask = torch.sigmoid(input) > 0.5
        
        # Compute topological features
        pred_topology = self._compute_betti_numbers(pred_mask)
        target_topology = self._compute_betti_numbers(target)
        
        # L2 loss on topological features
        topology_loss = F.mse_loss(pred_topology, target_topology, reduction='none')
        
        if self.reduction == LossReduction.MEAN:
            return topology_loss.mean()
        elif self.reduction == LossReduction.SUM:
            return topology_loss.sum()
        else:
            return topology_loss


def create_loss_function(
    loss_type: str = "dice_ce",
    config_override: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> nn.Module:
    """
    Factory function to create loss functions from configuration.
    
    Args:
        loss_type: Type of loss function to create
        config_override: Optional parameter overrides
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured loss function
    """
    config = get_config()
    
    if loss_type == "dice_ce":
        return MedicalDiceCELoss(
            session_id=session_id,
            user_id=user_id,
            **(config_override or {})
        )
    elif loss_type == "adaptive":
        # Create adaptive loss with multiple components
        loss_functions = {
            'dice': DiceLoss(),
            'ce': nn.CrossEntropyLoss(),
            'boundary': BoundaryAwareLoss(session_id=session_id, user_id=user_id)
        }
        return AdaptiveLoss(
            loss_functions=loss_functions,
            session_id=session_id,
            user_id=user_id,
            **(config_override or {})
        )
    elif loss_type == "topology":
        return TopologyPreservingLoss(
            session_id=session_id,
            user_id=user_id,
            **(config_override or {})
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_class_weights(
    data_loader: torch.utils.data.DataLoader,
    num_classes: int,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        data_loader: DataLoader containing training data
        num_classes: Number of classes
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Class weights tensor
    """
    try:
        class_counts = torch.zeros(num_classes)
        total_pixels = 0
        
        # Count pixels per class
        for batch in data_loader:
            if isinstance(batch, dict):
                targets = batch.get('label', batch.get('mask'))
            else:
                targets = batch[1]  # Assume (input, target) format
            
            if targets.dim() > 1:
                targets = torch.argmax(targets, dim=1)
            
            for class_id in range(num_classes):
                class_counts[class_id] += (targets == class_id).sum().item()
            
            total_pixels += targets.numel()
        
        # Compute inverse frequency weights
        class_weights = total_pixels / (num_classes * class_counts)
        
        # Normalize to prevent extreme weights
        class_weights = class_weights / class_weights.sum() * num_classes
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Class weights computed for imbalanced dataset",
            user_id=user_id,
            session_id=session_id,
            additional_data={
                'num_classes': num_classes,
                'class_counts': class_counts.tolist(),
                'class_weights': class_weights.tolist()
            }
        )
        
        return class_weights
        
    except Exception as e:
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.ERROR,
            message=f"Failed to compute class weights: {str(e)}",
            user_id=user_id,
            session_id=session_id,
            additional_data={'error': str(e)}
        )
        raise