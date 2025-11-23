"""
Evaluation metrics for medical image segmentation and classification.

This module implements comprehensive metrics optimized for medical imaging tasks,
including distance-based metrics, volumetric metrics, and clinical assessment tools.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

from monai.metrics import (
    DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric,
    compute_meandice, compute_hausdorff_distance
)
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class MedicalSegmentationMetrics(nn.Module):
    """
    Comprehensive metrics suite for medical image segmentation evaluation.
    
    Provides volumetric metrics (Dice, IoU, Jaccard), distance metrics
    (Hausdorff, Average Surface Distance), and clinical metrics.
    """
    
    def __init__(
        self,
        include_background: bool = False,
        reduction: MetricReduction = MetricReduction.MEAN,
        get_not_nans: bool = False,
        compute_hausdorff: bool = True,
        compute_surface_distance: bool = True,
        percentile: Optional[float] = None,
        directed: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize medical segmentation metrics.
        
        Args:
            include_background: Whether to include background class
            reduction: Metric reduction method
            get_not_nans: Whether to return only non-NaN values
            compute_hausdorff: Whether to compute Hausdorff distance
            compute_surface_distance: Whether to compute surface distances
            percentile: Percentile for Hausdorff distance (None for max)
            directed: Whether to compute directed Hausdorff distance
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        self._session_id = session_id
        self._user_id = user_id
        self._compute_hausdorff = compute_hausdorff
        self._compute_surface_distance = compute_surface_distance
        
        # Initialize MONAI metrics
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction=reduction,
            get_not_nans=get_not_nans
        )
        
        if compute_hausdorff:
            self.hausdorff_metric = HausdorffDistanceMetric(
                include_background=include_background,
                reduction=reduction,
                get_not_nans=get_not_nans,
                percentile=percentile,
                directed=directed
            )
        
        if compute_surface_distance:
            self.surface_distance_metric = SurfaceDistanceMetric(
                include_background=include_background,
                reduction=reduction,
                get_not_nans=get_not_nans,
                symmetric=True
            )
        
        # Additional metrics
        self.iou_metric = IoUMetric(
            include_background=include_background,
            reduction=reduction
        )
        
        self.sensitivity_metric = SensitivitySpecificityMetric(
            include_background=include_background,
            reduction=reduction
        )
        
        # Clinical assessment metrics
        self.volume_metric = VolumeMetric()
        self.connectivity_metric = ConnectivityMetric()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Medical segmentation metrics initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'metrics': ['dice', 'iou', 'sensitivity', 'specificity', 'volume'],
                'optional_metrics': {
                    'hausdorff': compute_hausdorff,
                    'surface_distance': compute_surface_distance
                }
            }
        )
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive segmentation metrics.
        
        Args:
            y_pred: Predicted segmentation [B, C, ...] or [B, ...]
            y_true: Ground truth segmentation [B, C, ...] or [B, ...]
            
        Returns:
            Dictionary containing all computed metrics
        """
        try:
            metrics = {}
            
            # Ensure proper format for MONAI metrics
            if y_pred.dim() == y_true.dim() and y_pred.dim() > 1:
                if y_pred.shape[1] == 1:  # Binary case
                    y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
                else:  # Multi-class case
                    y_pred_one_hot = F.one_hot(
                        torch.argmax(y_pred, dim=1), 
                        num_classes=y_pred.shape[1]
                    ).permute(0, -1, *range(1, y_pred.dim()-1)).float()
                    y_pred_binary = y_pred_one_hot
            else:
                y_pred_binary = (torch.sigmoid(y_pred) > 0.5).float()
            
            # Core volumetric metrics
            metrics['dice'] = self.dice_metric(y_pred_binary, y_true)
            metrics['iou'] = self.iou_metric(y_pred_binary, y_true)
            
            # Sensitivity and specificity
            sens_spec = self.sensitivity_metric(y_pred_binary, y_true)
            metrics.update(sens_spec)
            
            # Distance-based metrics (if enabled)
            if self._compute_hausdorff:
                try:
                    metrics['hausdorff'] = self.hausdorff_metric(y_pred_binary, y_true)
                except Exception as e:
                    log_audit_event(
                        event_type=AuditEventType.MODEL_INFERENCE,
                        severity=AuditSeverity.WARNING,
                        message=f"Hausdorff distance computation failed: {str(e)}",
                        user_id=self._user_id,
                        session_id=self._session_id
                    )
                    metrics['hausdorff'] = torch.tensor(float('nan'))
            
            if self._compute_surface_distance:
                try:
                    metrics['surface_distance'] = self.surface_distance_metric(
                        y_pred_binary, y_true
                    )
                except Exception as e:
                    log_audit_event(
                        event_type=AuditEventType.MODEL_INFERENCE,
                        severity=AuditSeverity.WARNING,
                        message=f"Surface distance computation failed: {str(e)}",
                        user_id=self._user_id,
                        session_id=self._session_id
                    )
                    metrics['surface_distance'] = torch.tensor(float('nan'))
            
            # Clinical metrics
            metrics['volume_similarity'] = self.volume_metric(y_pred_binary, y_true)
            metrics['connectivity'] = self.connectivity_metric(y_pred_binary, y_true)
            
            return metrics
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.ERROR,
                message=f"Metrics computation failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise


class IoUMetric(nn.Module):
    """Intersection over Union (Jaccard Index) metric."""
    
    def __init__(
        self,
        include_background: bool = False,
        reduction: MetricReduction = MetricReduction.MEAN,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU metric.
        
        Args:
            y_pred: Predicted segmentation
            y_true: Ground truth segmentation
            
        Returns:
            IoU score
        """
        # Flatten tensors
        y_pred_flat = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)
        y_true_flat = y_true.view(y_true.shape[0], y_true.shape[1], -1)
        
        # Compute intersection and union
        intersection = torch.sum(y_pred_flat * y_true_flat, dim=2)
        union = torch.sum(y_pred_flat, dim=2) + torch.sum(y_true_flat, dim=2) - intersection
        
        # Compute IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Handle background exclusion
        if not self.include_background and iou.shape[1] > 1:
            iou = iou[:, 1:]
        
        # Apply reduction
        if self.reduction == MetricReduction.MEAN:
            return torch.mean(iou)
        elif self.reduction == MetricReduction.SUM:
            return torch.sum(iou)
        else:
            return iou


class SensitivitySpecificityMetric(nn.Module):
    """Sensitivity (recall) and specificity metrics for medical evaluation."""
    
    def __init__(
        self,
        include_background: bool = False,
        reduction: MetricReduction = MetricReduction.MEAN,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.smooth = smooth
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute sensitivity and specificity.
        
        Args:
            y_pred: Predicted segmentation
            y_true: Ground truth segmentation
            
        Returns:
            Dictionary with sensitivity and specificity scores
        """
        # Flatten tensors
        y_pred_flat = y_pred.view(y_pred.shape[0], y_pred.shape[1], -1)
        y_true_flat = y_true.view(y_true.shape[0], y_true.shape[1], -1)
        
        # True positives, false positives, false negatives, true negatives
        tp = torch.sum(y_pred_flat * y_true_flat, dim=2)
        fp = torch.sum(y_pred_flat * (1 - y_true_flat), dim=2)
        fn = torch.sum((1 - y_pred_flat) * y_true_flat, dim=2)
        tn = torch.sum((1 - y_pred_flat) * (1 - y_true_flat), dim=2)
        
        # Sensitivity (recall) = TP / (TP + FN)
        sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
        
        # Specificity = TN / (TN + FP)
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)
        
        # Precision = TP / (TP + FP)
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        
        # F1 score = 2 * (precision * recall) / (precision + recall)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + self.smooth)
        
        # Handle background exclusion
        if not self.include_background and sensitivity.shape[1] > 1:
            sensitivity = sensitivity[:, 1:]
            specificity = specificity[:, 1:]
            precision = precision[:, 1:]
            f1_score = f1_score[:, 1:]
        
        # Apply reduction
        metrics = {}
        for name, values in [
            ('sensitivity', sensitivity), 
            ('specificity', specificity),
            ('precision', precision),
            ('f1_score', f1_score)
        ]:
            if self.reduction == MetricReduction.MEAN:
                metrics[name] = torch.mean(values)
            elif self.reduction == MetricReduction.SUM:
                metrics[name] = torch.sum(values)
            else:
                metrics[name] = values
        
        return metrics


class VolumeMetric(nn.Module):
    """Volume similarity metric for clinical assessment."""
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute volume similarity coefficient.
        
        Args:
            y_pred: Predicted segmentation
            y_true: Ground truth segmentation
            
        Returns:
            Volume similarity score
        """
        # Calculate volumes
        pred_volume = torch.sum(y_pred, dim=tuple(range(2, y_pred.dim())))
        true_volume = torch.sum(y_true, dim=tuple(range(2, y_true.dim())))
        
        # Volume similarity = 1 - |V_pred - V_true| / (V_pred + V_true)
        volume_diff = torch.abs(pred_volume - true_volume)
        volume_sum = pred_volume + true_volume + self.smooth
        
        volume_similarity = 1.0 - (volume_diff / volume_sum)
        
        return torch.mean(volume_similarity)


class ConnectivityMetric(nn.Module):
    """Connectivity analysis for anatomical structure assessment."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute connectivity metric based on connected components.
        
        Args:
            y_pred: Predicted segmentation
            y_true: Ground truth segmentation
            
        Returns:
            Connectivity score
        """
        connectivity_scores = []
        
        # Process each sample in batch
        for i in range(y_pred.shape[0]):
            pred_np = y_pred[i].detach().cpu().numpy()
            true_np = y_true[i].detach().cpu().numpy()
            
            # Process each class
            for j in range(y_pred.shape[1]):
                pred_class = pred_np[j] > 0.5
                true_class = true_np[j] > 0.5
                
                if np.sum(true_class) == 0:  # Skip empty ground truth
                    continue
                
                # Count connected components
                pred_components, pred_num = ndimage.label(pred_class)
                true_components, true_num = ndimage.label(true_class)
                
                # Connectivity score based on component count similarity
                if true_num == 0:
                    score = 1.0 if pred_num == 0 else 0.0
                else:
                    score = 1.0 - abs(pred_num - true_num) / max(pred_num, true_num, 1)
                
                connectivity_scores.append(score)
        
        if not connectivity_scores:
            return torch.tensor(1.0)  # Perfect score if no valid classes
        
        return torch.tensor(np.mean(connectivity_scores))


class ClinicalMetrics(nn.Module):
    """Clinical assessment metrics for medical image analysis."""
    
    def __init__(
        self,
        organ_specific_weights: Optional[Dict[str, float]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize clinical metrics.
        
        Args:
            organ_specific_weights: Organ-specific importance weights
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        self.organ_weights = organ_specific_weights or {}
        self._session_id = session_id
        self._user_id = user_id
    
    def compute_clinical_significance(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        organ_labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute clinically relevant metrics.
        
        Args:
            y_pred: Predicted segmentation
            y_true: Ground truth segmentation
            organ_labels: Optional organ label names
            
        Returns:
            Clinical significance scores
        """
        metrics = {}
        
        # Volume error assessment
        volume_errors = self._compute_volume_errors(y_pred, y_true)
        metrics['volume_error_mean'] = float(torch.mean(volume_errors))
        metrics['volume_error_std'] = float(torch.std(volume_errors))
        
        # Boundary accuracy
        boundary_accuracy = self._compute_boundary_accuracy(y_pred, y_true)
        metrics['boundary_accuracy'] = float(boundary_accuracy)
        
        # Critical structure preservation
        critical_preservation = self._compute_critical_preservation(y_pred, y_true)
        metrics['critical_preservation'] = float(critical_preservation)
        
        # Organ-specific weighted score
        if organ_labels and self.organ_weights:
            weighted_score = self._compute_weighted_score(
                y_pred, y_true, organ_labels
            )
            metrics['weighted_clinical_score'] = float(weighted_score)
        
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message="Clinical metrics computed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data=metrics
        )
        
        return metrics
    
    def _compute_volume_errors(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute relative volume errors."""
        pred_volumes = torch.sum(y_pred, dim=tuple(range(2, y_pred.dim())))
        true_volumes = torch.sum(y_true, dim=tuple(range(2, y_true.dim())))
        
        # Relative volume error
        volume_errors = torch.abs(pred_volumes - true_volumes) / (true_volumes + 1e-5)
        return volume_errors
    
    def _compute_boundary_accuracy(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary segmentation accuracy."""
        # Simple gradient-based boundary detection
        pred_boundaries = self._extract_boundaries(y_pred)
        true_boundaries = self._extract_boundaries(y_true)
        
        # Dice score on boundaries
        intersection = torch.sum(pred_boundaries * true_boundaries)
        union = torch.sum(pred_boundaries) + torch.sum(true_boundaries)
        
        boundary_dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return boundary_dice
    
    def _extract_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract object boundaries using gradient magnitude."""
        # Compute gradients
        grad_x = torch.diff(mask, dim=-1, prepend=mask[..., :1])
        grad_y = torch.diff(mask, dim=-2, prepend=mask[..., :1, :])
        
        if mask.dim() > 4:  # 3D case
            grad_z = torch.diff(mask, dim=-3, prepend=mask[..., :1, :, :])
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        else:  # 2D case
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to get boundaries
        boundaries = (gradient_magnitude > 0.1).float()
        return boundaries
    
    def _compute_critical_preservation(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute preservation of critical anatomical structures."""
        # This is a simplified metric - in practice, implement organ-specific logic
        critical_dice = torch.mean(self._compute_dice_per_class(y_pred, y_true))
        return critical_dice
    
    def _compute_dice_per_class(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dice score per class."""
        intersection = torch.sum(y_pred * y_true, dim=tuple(range(2, y_pred.dim())))
        union = torch.sum(y_pred, dim=tuple(range(2, y_pred.dim()))) + \
                torch.sum(y_true, dim=tuple(range(2, y_true.dim())))
        
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        return dice
    
    def _compute_weighted_score(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        organ_labels: List[str]
    ) -> torch.Tensor:
        """Compute organ-weighted clinical score."""
        dice_scores = self._compute_dice_per_class(y_pred, y_true)
        
        weighted_scores = []
        for i, organ in enumerate(organ_labels):
            if i < dice_scores.shape[1]:
                weight = self.organ_weights.get(organ, 1.0)
                weighted_scores.append(weight * dice_scores[:, i])
        
        if weighted_scores:
            return torch.mean(torch.stack(weighted_scores))
        else:
            return torch.tensor(0.0)


def compute_comprehensive_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    include_clinical: bool = True,
    organ_labels: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for medical segmentation.
    
    Args:
        y_pred: Predicted segmentation
        y_true: Ground truth segmentation
        include_clinical: Whether to include clinical metrics
        organ_labels: Optional organ label names
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Comprehensive metrics dictionary
    """
    try:
        # Initialize metrics calculator
        metrics_calculator = MedicalSegmentationMetrics(
            session_id=session_id,
            user_id=user_id
        )
        
        # Compute standard metrics
        standard_metrics = metrics_calculator(y_pred, y_true)
        
        # Compute clinical metrics if requested
        clinical_metrics = {}
        if include_clinical:
            clinical_calculator = ClinicalMetrics(
                session_id=session_id,
                user_id=user_id
            )
            clinical_metrics = clinical_calculator.compute_clinical_significance(
                y_pred, y_true, organ_labels
            )
        
        # Combine all metrics
        all_metrics = {
            'standard': {k: float(v) for k, v in standard_metrics.items()},
            'clinical': clinical_metrics
        }
        
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message="Comprehensive metrics evaluation completed",
            user_id=user_id,
            session_id=session_id,
            additional_data={
                'metrics_count': len(standard_metrics) + len(clinical_metrics),
                'include_clinical': include_clinical
            }
        )
        
        return all_metrics
        
    except Exception as e:
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.ERROR,
            message=f"Comprehensive metrics computation failed: {str(e)}",
            user_id=user_id,
            session_id=session_id,
            additional_data={'error': str(e)}
        )
        raise


def create_metrics_evaluator(
    config_override: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> MedicalSegmentationMetrics:
    """
    Factory function to create metrics evaluator from configuration.
    
    Args:
        config_override: Optional parameter overrides
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured metrics evaluator
    """
    config = get_config()
    
    # Apply configuration overrides
    params = config_override or {}
    
    return MedicalSegmentationMetrics(
        session_id=session_id,
        user_id=user_id,
        **params
    )