"""
Cross-validation framework for clinical model validation.

This module provides comprehensive cross-validation and statistical analysis
tools specifically designed for medical AI validation, including clinical
performance metrics, statistical tests, and regulatory compliance.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, LeaveOneGroupOut,
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    classification_report, roc_curve
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from monai.data import DataLoader as MonaiDataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class ValidationStrategy(Enum):
    """Cross-validation strategies for medical data."""
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    GROUP_K_FOLD = "group_k_fold"
    LEAVE_ONE_GROUP_OUT = "leave_one_group_out"
    LEAVE_ONE_PATIENT_OUT = "leave_one_patient_out"
    TEMPORAL_SPLIT = "temporal_split"
    INSTITUTION_SPLIT = "institution_split"


class ClinicalMetric(Enum):
    """Clinical performance metrics."""
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    POSITIVE_PREDICTIVE_VALUE = "ppv"
    NEGATIVE_PREDICTIVE_VALUE = "npv"
    F1_SCORE = "f1_score"
    BALANCED_ACCURACY = "balanced_accuracy"
    DICE_COEFFICIENT = "dice_coefficient"
    HAUSDORFF_DISTANCE = "hausdorff_distance"
    VOLUME_SIMILARITY = "volume_similarity"
    SURFACE_DISTANCE = "surface_distance"


@dataclass
class ValidationResult:
    """Result of cross-validation."""
    strategy: ValidationStrategy
    fold_results: List[Dict[str, float]]
    aggregated_metrics: Dict[str, Dict[str, float]]  # metric -> {mean, std, ci_lower, ci_upper}
    statistical_tests: Dict[str, Dict[str, Any]]
    clinical_significance: Dict[str, Any]
    regulatory_compliance: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ClinicalValidationReport:
    """Comprehensive clinical validation report."""
    model_name: str
    validation_date: str
    dataset_description: Dict[str, Any]
    validation_results: Dict[str, ValidationResult]
    comparative_analysis: Optional[Dict[str, Any]] = None
    regulatory_assessment: Dict[str, Any] = None
    clinical_recommendations: List[str] = None
    limitations: List[str] = None


class BaseValidator(ABC):
    """Base class for clinical validators."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize validator.
        
        Args:
            model: PyTorch model to validate
            device: Computation device
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.model = model
        self.device = device
        self._session_id = session_id
        self._user_id = user_id
        
        self.model.to(self.device)
        self.model.eval()
    
    @abstractmethod
    def validate(
        self,
        dataset: Any,
        strategy: ValidationStrategy,
        n_folds: int = 5,
        metrics: List[ClinicalMetric] = None,
        **kwargs
    ) -> ValidationResult:
        """Perform cross-validation."""
        pass
    
    def _calculate_confidence_interval(
        self,
        scores: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for metric scores."""
        alpha = 1 - confidence
        n = len(scores)
        
        if n < 30:
            # Use t-distribution for small samples
            t_value = stats.t.ppf(1 - alpha/2, n - 1)
            margin_of_error = t_value * stats.sem(scores)
        else:
            # Use normal distribution for large samples
            z_value = stats.norm.ppf(1 - alpha/2)
            margin_of_error = z_value * stats.sem(scores)
        
        mean_score = np.mean(scores)
        return mean_score - margin_of_error, mean_score + margin_of_error
    
    def _perform_statistical_tests(
        self,
        fold_results: List[Dict[str, float]],
        baseline_results: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests."""
        tests = {}
        
        # Extract metrics across folds
        metrics_data = {}
        for metric in fold_results[0].keys():
            scores = [fold[metric] for fold in fold_results]
            metrics_data[metric] = scores
        
        # Normality tests
        for metric, scores in metrics_data.items():
            if len(scores) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(scores)
                tests[f"{metric}_normality"] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
        
        # One-sample t-tests (against chance performance)
        for metric, scores in metrics_data.items():
            if 'accuracy' in metric or 'dice' in metric:
                # Test against chance performance (0.5 for binary, varies for multiclass)
                null_value = 0.5
                t_stat, t_p = stats.ttest_1samp(scores, null_value)
                tests[f"{metric}_vs_chance"] = {
                    'test': 'One-sample t-test',
                    'null_hypothesis': f'{metric} = {null_value}',
                    'statistic': float(t_stat),
                    'p_value': float(t_p),
                    'significant': t_p < 0.05
                }
        
        # Paired t-tests against baseline if provided
        if baseline_results:
            baseline_metrics = {}
            for metric in baseline_results[0].keys():
                baseline_scores = [fold[metric] for fold in baseline_results]
                baseline_metrics[metric] = baseline_scores
            
            for metric in metrics_data.keys():
                if metric in baseline_metrics:
                    current_scores = metrics_data[metric]
                    baseline_scores = baseline_metrics[metric]
                    
                    if len(current_scores) == len(baseline_scores):
                        t_stat, t_p = stats.ttest_rel(current_scores, baseline_scores)
                        tests[f"{metric}_vs_baseline"] = {
                            'test': 'Paired t-test',
                            'statistic': float(t_stat),
                            'p_value': float(t_p),
                            'significant': t_p < 0.05,
                            'improvement': np.mean(current_scores) > np.mean(baseline_scores)
                        }
        
        return tests
    
    def _assess_clinical_significance(
        self,
        aggregated_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Assess clinical significance of results."""
        significance = {}
        
        # Clinical significance thresholds (can be customized)
        thresholds = {
            'sensitivity': {'excellent': 0.95, 'good': 0.85, 'acceptable': 0.75},
            'specificity': {'excellent': 0.95, 'good': 0.85, 'acceptable': 0.75},
            'dice_coefficient': {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.7},
            'hausdorff_distance': {'excellent': 2.0, 'good': 5.0, 'acceptable': 10.0},  # mm
            'balanced_accuracy': {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.7}
        }
        
        for metric, stats_dict in aggregated_metrics.items():
            mean_value = stats_dict['mean']
            lower_ci = stats_dict['ci_lower']
            
            if metric in thresholds:
                thresh = thresholds[metric]
                
                # For metrics where lower is better (e.g., Hausdorff distance)
                if 'distance' in metric:
                    if mean_value <= thresh['excellent']:
                        level = 'excellent'
                    elif mean_value <= thresh['good']:
                        level = 'good'
                    elif mean_value <= thresh['acceptable']:
                        level = 'acceptable'
                    else:
                        level = 'poor'
                else:
                    # For metrics where higher is better
                    if lower_ci >= thresh['excellent']:
                        level = 'excellent'
                    elif lower_ci >= thresh['good']:
                        level = 'good'
                    elif lower_ci >= thresh['acceptable']:
                        level = 'acceptable'
                    else:
                        level = 'poor'
                
                significance[metric] = {
                    'clinical_level': level,
                    'mean_performance': mean_value,
                    'confidence_interval': (lower_ci, stats_dict['ci_upper']),
                    'meets_clinical_threshold': level in ['excellent', 'good', 'acceptable']
                }
        
        # Overall assessment
        clinical_levels = [info['clinical_level'] for info in significance.values()]
        poor_metrics = [metric for metric, info in significance.items() 
                       if info['clinical_level'] == 'poor']
        
        significance['overall_assessment'] = {
            'predominantly_excellent': clinical_levels.count('excellent') > len(clinical_levels) / 2,
            'meets_minimum_standards': len(poor_metrics) == 0,
            'poor_performing_metrics': poor_metrics,
            'ready_for_clinical_deployment': len(poor_metrics) == 0 and 
                                          clinical_levels.count('poor') == 0
        }
        
        return significance
    
    def _assess_regulatory_compliance(
        self,
        validation_results: ValidationResult,
        regulatory_standard: str = "FDA"
    ) -> Dict[str, Any]:
        """Assess regulatory compliance (FDA, CE mark, etc.)."""
        compliance = {
            'standard': regulatory_standard,
            'requirements_met': {},
            'overall_compliance': False
        }
        
        if regulatory_standard == "FDA":
            # FDA requirements for medical AI (simplified)
            requirements = {
                'minimum_sensitivity': 0.8,
                'minimum_specificity': 0.8,
                'statistical_significance': True,
                'adequate_sample_size': True,
                'cross_validation_performed': True
            }
            
            # Check sensitivity
            if 'sensitivity' in validation_results.aggregated_metrics:
                sens_ci_lower = validation_results.aggregated_metrics['sensitivity']['ci_lower']
                compliance['requirements_met']['minimum_sensitivity'] = sens_ci_lower >= requirements['minimum_sensitivity']
            
            # Check specificity
            if 'specificity' in validation_results.aggregated_metrics:
                spec_ci_lower = validation_results.aggregated_metrics['specificity']['ci_lower']
                compliance['requirements_met']['minimum_specificity'] = spec_ci_lower >= requirements['minimum_specificity']
            
            # Check statistical significance
            significant_tests = [
                test_info['significant'] 
                for test_name, test_info in validation_results.statistical_tests.items()
                if 'vs_chance' in test_name
            ]
            compliance['requirements_met']['statistical_significance'] = any(significant_tests)
            
            # Check cross-validation
            compliance['requirements_met']['cross_validation_performed'] = len(validation_results.fold_results) >= 3
            
            # Check sample size (simplified - actual requirements vary)
            n_samples = validation_results.metadata.get('total_samples', 0)
            compliance['requirements_met']['adequate_sample_size'] = n_samples >= 100
            
            # Overall compliance
            compliance['overall_compliance'] = all(compliance['requirements_met'].values())
        
        return compliance


class ClassificationValidator(BaseValidator):
    """Validator for classification models."""
    
    def validate(
        self,
        dataset: DataLoader,
        strategy: ValidationStrategy = ValidationStrategy.STRATIFIED_K_FOLD,
        n_folds: int = 5,
        metrics: List[ClinicalMetric] = None,
        group_column: Optional[str] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Perform cross-validation for classification model.
        
        Args:
            dataset: DataLoader with dataset
            strategy: Cross-validation strategy
            n_folds: Number of folds
            metrics: List of metrics to calculate
            group_column: Column name for group-based splitting
            
        Returns:
            Validation results
        """
        if metrics is None:
            metrics = [
                ClinicalMetric.SENSITIVITY,
                ClinicalMetric.SPECIFICITY,
                ClinicalMetric.F1_SCORE,
                ClinicalMetric.BALANCED_ACCURACY
            ]
        
        # Extract data and labels from dataset
        all_data, all_labels, groups = self._extract_data_labels(dataset, group_column)
        
        # Create cross-validation splitter
        splitter = self._create_splitter(strategy, n_folds, all_labels, groups)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(all_data, all_labels, groups)):
            # Create fold datasets
            train_subset = Subset(dataset.dataset, train_idx.tolist())
            val_subset = Subset(dataset.dataset, val_idx.tolist())
            
            train_loader = DataLoader(train_subset, batch_size=dataset.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=dataset.batch_size, shuffle=False)
            
            # Train model on fold (if training enabled)
            if kwargs.get('retrain_per_fold', False):
                self._train_fold(train_loader, fold_idx)
            
            # Evaluate on validation set
            fold_metrics = self._evaluate_classification_fold(val_loader, metrics)
            fold_results.append(fold_metrics)
            
            log_audit_event(
                event_type=AuditEventType.MODEL_EVALUATION,
                severity=AuditSeverity.INFO,
                message=f"Completed fold {fold_idx + 1}/{len(list(splitter.split(all_data, all_labels, groups)))}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'fold_metrics': fold_metrics}
            )
        
        # Aggregate results
        aggregated_metrics = self._aggregate_fold_results(fold_results)
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(fold_results)
        
        # Clinical significance
        clinical_significance = self._assess_clinical_significance(aggregated_metrics)
        
        # Regulatory compliance
        result = ValidationResult(
            strategy=strategy,
            fold_results=fold_results,
            aggregated_metrics=aggregated_metrics,
            statistical_tests=statistical_tests,
            clinical_significance=clinical_significance,
            regulatory_compliance={},
            metadata={
                'n_folds': n_folds,
                'total_samples': len(all_data),
                'metrics_used': [m.value for m in metrics],
                'model_name': self.model.__class__.__name__
            }
        )
        
        result.regulatory_compliance = self._assess_regulatory_compliance(result)
        
        log_audit_event(
            event_type=AuditEventType.MODEL_EVALUATION,
            severity=AuditSeverity.INFO,
            message="Cross-validation completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'strategy': strategy.value,
                'n_folds': n_folds,
                'overall_compliance': result.regulatory_compliance['overall_compliance']
            }
        )
        
        return result
    
    def _extract_data_labels(
        self,
        dataset: DataLoader,
        group_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract data indices, labels, and groups from dataset."""
        labels = []
        groups = []
        
        for batch_idx, batch in enumerate(dataset):
            if isinstance(batch, (list, tuple)):
                _, batch_labels = batch[:2]
            else:
                batch_labels = batch['label']
            
            labels.extend(batch_labels.cpu().numpy())
            
            # Extract groups if specified
            if group_column and hasattr(batch, group_column):
                batch_groups = getattr(batch, group_column)
                groups.extend(batch_groups)
            elif group_column and isinstance(batch, dict) and group_column in batch:
                groups.extend(batch[group_column])
        
        data_indices = np.arange(len(labels))
        labels = np.array(labels)
        groups = np.array(groups) if groups else None
        
        return data_indices, labels, groups
    
    def _create_splitter(
        self,
        strategy: ValidationStrategy,
        n_folds: int,
        labels: np.ndarray,
        groups: Optional[np.ndarray]
    ):
        """Create cross-validation splitter."""
        if strategy == ValidationStrategy.K_FOLD:
            return KFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif strategy == ValidationStrategy.STRATIFIED_K_FOLD:
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif strategy == ValidationStrategy.GROUP_K_FOLD:
            if groups is None:
                raise ValueError("Groups required for GroupKFold")
            return GroupKFold(n_splits=n_folds)
        elif strategy == ValidationStrategy.LEAVE_ONE_GROUP_OUT:
            if groups is None:
                raise ValueError("Groups required for LeaveOneGroupOut")
            return LeaveOneGroupOut()
        else:
            raise ValueError(f"Unsupported validation strategy: {strategy}")
    
    def _evaluate_classification_fold(
        self,
        val_loader: DataLoader,
        metrics: List[ClinicalMetric]
    ) -> Dict[str, float]:
        """Evaluate model on validation fold."""
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[:2]
                else:
                    inputs = batch['image']
                    labels = batch['label']
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calculate metrics
        fold_metrics = {}
        
        for metric in metrics:
            if metric == ClinicalMetric.SENSITIVITY:
                # Sensitivity = TP / (TP + FN)
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):  # Binary classification
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    fold_metrics['sensitivity'] = sensitivity
                else:
                    # Multiclass - macro average
                    sensitivities = []
                    for i in range(cm.shape[0]):
                        tp = cm[i, i]
                        fn = np.sum(cm[i, :]) - tp
                        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        sensitivities.append(sens)
                    fold_metrics['sensitivity'] = np.mean(sensitivities)
            
            elif metric == ClinicalMetric.SPECIFICITY:
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):  # Binary classification
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    fold_metrics['specificity'] = specificity
                else:
                    # Multiclass - macro average
                    specificities = []
                    for i in range(cm.shape[0]):
                        tp = cm[i, i]
                        fp = np.sum(cm[:, i]) - tp
                        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + tp
                        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        specificities.append(spec)
                    fold_metrics['specificity'] = np.mean(specificities)
            
            elif metric == ClinicalMetric.F1_SCORE:
                from sklearn.metrics import f1_score
                fold_metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
            
            elif metric == ClinicalMetric.BALANCED_ACCURACY:
                from sklearn.metrics import balanced_accuracy_score
                fold_metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        return fold_metrics
    
    def _train_fold(self, train_loader: DataLoader, fold_idx: int):
        """Train model on fold data (placeholder for actual training)."""
        # This would implement fold-specific training
        # For now, we assume the model is pre-trained
        pass
    
    def _aggregate_fold_results(
        self,
        fold_results: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate results across folds."""
        aggregated = {}
        
        # Get all metrics
        all_metrics = set()
        for fold_result in fold_results:
            all_metrics.update(fold_result.keys())
        
        for metric in all_metrics:
            scores = [fold_result.get(metric, 0.0) for fold_result in fold_results]
            scores = np.array(scores)
            
            ci_lower, ci_upper = self._calculate_confidence_interval(scores)
            
            aggregated[metric] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'median': float(np.median(scores))
            }
        
        return aggregated


class SegmentationValidator(BaseValidator):
    """Validator for segmentation models."""
    
    def validate(
        self,
        dataset: MonaiDataLoader,
        strategy: ValidationStrategy = ValidationStrategy.K_FOLD,
        n_folds: int = 5,
        metrics: List[ClinicalMetric] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Perform cross-validation for segmentation model.
        
        Args:
            dataset: MONAI DataLoader with dataset
            strategy: Cross-validation strategy
            n_folds: Number of folds
            metrics: List of metrics to calculate
            
        Returns:
            Validation results
        """
        if metrics is None:
            metrics = [
                ClinicalMetric.DICE_COEFFICIENT,
                ClinicalMetric.HAUSDORFF_DISTANCE,
                ClinicalMetric.VOLUME_SIMILARITY
            ]
        
        # Initialize MONAI metrics
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        hd_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
        
        # Extract data for cross-validation
        all_data, all_labels, groups = self._extract_segmentation_data(dataset)
        
        # Create splitter
        splitter = self._create_splitter(strategy, n_folds, all_labels, groups)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(all_data, all_labels, groups)):
            # Evaluate on validation set
            fold_metrics = self._evaluate_segmentation_fold(
                dataset, val_idx, metrics, dice_metric, hd_metric
            )
            fold_results.append(fold_metrics)
            
            log_audit_event(
                event_type=AuditEventType.MODEL_EVALUATION,
                severity=AuditSeverity.INFO,
                message=f"Completed segmentation fold {fold_idx + 1}/{n_folds}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'fold_metrics': fold_metrics}
            )
        
        # Aggregate and analyze results
        aggregated_metrics = self._aggregate_fold_results(fold_results)
        statistical_tests = self._perform_statistical_tests(fold_results)
        clinical_significance = self._assess_clinical_significance(aggregated_metrics)
        
        result = ValidationResult(
            strategy=strategy,
            fold_results=fold_results,
            aggregated_metrics=aggregated_metrics,
            statistical_tests=statistical_tests,
            clinical_significance=clinical_significance,
            regulatory_compliance={},
            metadata={
                'n_folds': n_folds,
                'total_samples': len(all_data),
                'metrics_used': [m.value for m in metrics],
                'model_type': 'segmentation'
            }
        )
        
        result.regulatory_compliance = self._assess_regulatory_compliance(result)
        
        return result
    
    def _extract_segmentation_data(
        self,
        dataset: MonaiDataLoader
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract data for segmentation cross-validation."""
        # Simplified extraction - in practice would handle MONAI data properly
        n_samples = len(dataset.dataset)
        data_indices = np.arange(n_samples)
        labels = np.zeros(n_samples)  # Placeholder
        return data_indices, labels, None
    
    def _evaluate_segmentation_fold(
        self,
        dataset: MonaiDataLoader,
        val_indices: np.ndarray,
        metrics: List[ClinicalMetric],
        dice_metric: DiceMetric,
        hd_metric: HausdorffDistanceMetric
    ) -> Dict[str, float]:
        """Evaluate segmentation model on validation fold."""
        self.model.eval()
        
        dice_scores = []
        hd_scores = []
        volume_similarities = []
        
        with torch.no_grad():
            for idx in val_indices:
                try:
                    # Get sample from dataset
                    sample = dataset.dataset[idx]
                    inputs = sample['image'].unsqueeze(0).to(self.device)
                    targets = sample['label'].unsqueeze(0).to(self.device)
                    
                    # Predict
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Convert to binary predictions
                    predictions = torch.sigmoid(outputs) > 0.5
                    
                    # Calculate metrics
                    if ClinicalMetric.DICE_COEFFICIENT in metrics:
                        dice_metric.reset()
                        dice_metric(predictions, targets)
                        dice_score = dice_metric.aggregate().item()
                        dice_scores.append(dice_score)
                    
                    if ClinicalMetric.HAUSDORFF_DISTANCE in metrics:
                        hd_metric.reset()
                        hd_metric(predictions, targets)
                        hd_score = hd_metric.aggregate().item()
                        hd_scores.append(hd_score)
                    
                    if ClinicalMetric.VOLUME_SIMILARITY in metrics:
                        pred_volume = torch.sum(predictions).item()
                        true_volume = torch.sum(targets).item()
                        if true_volume > 0:
                            vol_sim = 1 - abs(pred_volume - true_volume) / true_volume
                        else:
                            vol_sim = 1.0 if pred_volume == 0 else 0.0
                        volume_similarities.append(vol_sim)
                
                except Exception as e:
                    warnings.warn(f"Error processing sample {idx}: {str(e)}")
                    continue
        
        fold_metrics = {}
        if dice_scores:
            fold_metrics['dice_coefficient'] = np.mean(dice_scores)
        if hd_scores:
            fold_metrics['hausdorff_distance'] = np.mean(hd_scores)
        if volume_similarities:
            fold_metrics['volume_similarity'] = np.mean(volume_similarities)
        
        return fold_metrics


def create_clinical_validator(
    model: nn.Module,
    task_type: str = "classification",
    device: str = "cuda",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> BaseValidator:
    """
    Factory function to create appropriate validator.
    
    Args:
        model: PyTorch model to validate
        task_type: Type of task ('classification' or 'segmentation')
        device: Computation device
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured validator
    """
    if task_type == "classification":
        return ClassificationValidator(model, device, session_id, user_id)
    elif task_type == "segmentation":
        return SegmentationValidator(model, device, session_id, user_id)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def generate_clinical_validation_report(
    validation_results: Dict[str, ValidationResult],
    model_name: str,
    dataset_description: Dict[str, Any]
) -> ClinicalValidationReport:
    """
    Generate comprehensive clinical validation report.
    
    Args:
        validation_results: Dictionary of validation results
        model_name: Name of the model
        dataset_description: Description of the dataset used
        
    Returns:
        Comprehensive clinical validation report
    """
    # Generate clinical recommendations
    recommendations = []
    limitations = []
    
    for strategy, result in validation_results.items():
        if result.clinical_significance['overall_assessment']['ready_for_clinical_deployment']:
            recommendations.append(f"Model shows excellent performance with {strategy} validation")
        else:
            poor_metrics = result.clinical_significance['overall_assessment']['poor_performing_metrics']
            limitations.append(f"Poor performance in {poor_metrics} with {strategy} validation")
    
    # Regulatory assessment
    regulatory_assessment = {}
    all_compliant = True
    for strategy, result in validation_results.items():
        regulatory_assessment[strategy] = result.regulatory_compliance
        if not result.regulatory_compliance['overall_compliance']:
            all_compliant = False
    
    regulatory_assessment['overall_regulatory_readiness'] = all_compliant
    
    return ClinicalValidationReport(
        model_name=model_name,
        validation_date=pd.Timestamp.now().isoformat(),
        dataset_description=dataset_description,
        validation_results=validation_results,
        regulatory_assessment=regulatory_assessment,
        clinical_recommendations=recommendations,
        limitations=limitations
    )