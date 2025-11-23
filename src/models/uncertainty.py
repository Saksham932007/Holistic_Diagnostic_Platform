"""
Uncertainty quantification for medical AI models.

This module provides comprehensive uncertainty estimation methods including
Monte Carlo dropout, ensemble methods, and Bayesian inference for reliable
clinical decision making.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import torch.nn.utils.prune as prune

from monai.inferers import sliding_window_inference
from monai.data import MetaTensor

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class UncertaintyType(Enum):
    """Types of uncertainty in medical AI."""
    ALEATORIC = "aleatoric"      # Data uncertainty
    EPISTEMIC = "epistemic"      # Model uncertainty
    TOTAL = "total"              # Combined uncertainty


class UncertaintyMethod(Enum):
    """Uncertainty quantification methods."""
    MONTE_CARLO_DROPOUT = "mc_dropout"
    DEEP_ENSEMBLE = "deep_ensemble"
    BAYESIAN_NN = "bayesian_nn"
    TEMPERATURE_SCALING = "temperature_scaling"
    EVIDENTIAL = "evidential"
    LAPLACE_APPROXIMATION = "laplace"


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimation result."""
    predictions: np.ndarray
    aleatoric_uncertainty: np.ndarray
    epistemic_uncertainty: np.ndarray
    total_uncertainty: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    reliability_score: float
    method: UncertaintyMethod
    metadata: Dict[str, Any]


class BaseUncertaintyEstimator(ABC):
    """Base class for uncertainty estimators."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize uncertainty estimator.
        
        Args:
            model: PyTorch model
            device: Computation device
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.model = model
        self.device = device
        self._session_id = session_id
        self._user_id = user_id
        
        # Move model to device
        self.model.to(self.device)
    
    @abstractmethod
    def estimate_uncertainty(
        self,
        input_data: torch.Tensor,
        n_samples: int = 100
    ) -> UncertaintyEstimate:
        """Estimate uncertainty for input data."""
        pass
    
    def _calculate_confidence_intervals(
        self,
        predictions: np.ndarray,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals from prediction samples."""
        intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(predictions, upper_percentile, axis=0)
            
            intervals[f"{level:.0%}"] = np.stack([lower_bound, upper_bound], axis=0)
        
        return intervals
    
    def _calculate_reliability_score(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray
    ) -> float:
        """Calculate reliability score based on prediction consistency."""
        # Calculate coefficient of variation
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Avoid division by zero
        cv = np.divide(std_pred, mean_pred, out=np.zeros_like(std_pred), where=mean_pred!=0)
        
        # Reliability score (1 - normalized coefficient of variation)
        reliability = 1 - np.mean(cv) / (1 + np.mean(cv))
        
        return max(0.0, min(1.0, reliability))


class MonteCarloDropoutEstimator(BaseUncertaintyEstimator):
    """
    Monte Carlo Dropout uncertainty estimation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dropout_rate: float = 0.1,
        device: str = "cuda",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize MC Dropout estimator.
        
        Args:
            model: PyTorch model
            dropout_rate: Dropout rate for uncertainty estimation
            device: Computation device
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__(model, device, session_id, user_id)
        self.dropout_rate = dropout_rate
        self._enable_dropout_layers()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="MC Dropout estimator initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={'dropout_rate': dropout_rate}
        )
    
    def _enable_dropout_layers(self):
        """Enable dropout layers for uncertainty estimation."""
        def enable_dropout(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout3d):
                module.train()
                module.p = self.dropout_rate
        
        self.model.apply(enable_dropout)
    
    def estimate_uncertainty(
        self,
        input_data: torch.Tensor,
        n_samples: int = 100
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using Monte Carlo Dropout.
        
        Args:
            input_data: Input tensor
            n_samples: Number of forward passes
            
        Returns:
            Uncertainty estimation result
        """
        input_data = input_data.to(self.device)
        predictions = []
        
        # Enable dropout during inference
        self._enable_dropout_layers()
        
        # Perform multiple forward passes
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(input_data)
                if isinstance(output, tuple):
                    output = output[0]  # Take first output if tuple
                
                # Apply softmax for classification
                if output.dim() > 1 and output.shape[1] > 1:
                    output = F.softmax(output, dim=1)
                
                predictions.append(output.cpu().numpy())
        
        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # Shape: [n_samples, batch_size, ...]
        
        # Calculate uncertainties
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = var_pred
        
        # Aleatoric uncertainty (approximate as entropy of mean prediction)
        if mean_pred.ndim > 1 and mean_pred.shape[-1] > 1:
            # Classification case
            entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=-1)
            aleatoric_uncertainty = entropy
        else:
            # Regression case - use constant small value
            aleatoric_uncertainty = np.full_like(mean_pred.squeeze(), 0.01)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(predictions)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(predictions, total_uncertainty)
        
        result = UncertaintyEstimate(
            predictions=mean_pred,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            reliability_score=reliability_score,
            method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            metadata={
                'n_samples': n_samples,
                'dropout_rate': self.dropout_rate,
                'prediction_variance': float(np.mean(var_pred)),
                'mean_confidence': float(np.mean(mean_pred.max(axis=-1) if mean_pred.ndim > 1 else mean_pred))
            }
        )
        
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message="MC Dropout uncertainty estimation completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'n_samples': n_samples,
                'reliability_score': reliability_score,
                'mean_uncertainty': float(np.mean(total_uncertainty))
            }
        )
        
        return result


class DeepEnsembleEstimator(BaseUncertaintyEstimator):
    """
    Deep Ensemble uncertainty estimation.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = "cuda",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize Deep Ensemble estimator.
        
        Args:
            models: List of trained models
            device: Computation device
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        # Use first model as base
        super().__init__(models[0] if models else None, device, session_id, user_id)
        self.models = models
        
        # Move all models to device and set to eval mode
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Deep Ensemble estimator initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={'n_models': len(models)}
        )
    
    def estimate_uncertainty(
        self,
        input_data: torch.Tensor,
        n_samples: int = None
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using Deep Ensemble.
        
        Args:
            input_data: Input tensor
            n_samples: Not used for ensemble (determined by number of models)
            
        Returns:
            Uncertainty estimation result
        """
        if not self.models:
            raise ValueError("No models provided for ensemble")
        
        input_data = input_data.to(self.device)
        predictions = []
        
        # Get predictions from all models
        with torch.no_grad():
            for model in self.models:
                output = model(input_data)
                if isinstance(output, tuple):
                    output = output[0]  # Take first output if tuple
                
                # Apply softmax for classification
                if output.dim() > 1 and output.shape[1] > 1:
                    output = F.softmax(output, dim=1)
                
                predictions.append(output.cpu().numpy())
        
        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # Shape: [n_models, batch_size, ...]
        
        # Calculate uncertainties
        mean_pred = np.mean(predictions, axis=0)
        var_pred = np.var(predictions, axis=0)
        
        # Epistemic uncertainty (model disagreement)
        epistemic_uncertainty = var_pred
        
        # Aleatoric uncertainty (approximate as entropy)
        if mean_pred.ndim > 1 and mean_pred.shape[-1] > 1:
            # Classification case
            entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=-1)
            aleatoric_uncertainty = entropy
        else:
            # Regression case
            aleatoric_uncertainty = np.full_like(mean_pred.squeeze(), 0.01)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(predictions)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(predictions, total_uncertainty)
        
        result = UncertaintyEstimate(
            predictions=mean_pred,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            reliability_score=reliability_score,
            method=UncertaintyMethod.DEEP_ENSEMBLE,
            metadata={
                'n_models': len(self.models),
                'model_agreement': float(1 - np.mean(var_pred)),
                'ensemble_diversity': float(np.std([np.mean(pred) for pred in predictions]))
            }
        )
        
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message="Deep Ensemble uncertainty estimation completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'n_models': len(self.models),
                'reliability_score': reliability_score,
                'model_agreement': result.metadata['model_agreement']
            }
        )
        
        return result


class TemperatureScalingEstimator(BaseUncertaintyEstimator):
    """
    Temperature Scaling for calibrated uncertainty.
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 1.0,
        device: str = "cuda",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize Temperature Scaling estimator.
        
        Args:
            model: PyTorch model
            temperature: Temperature parameter for scaling
            device: Computation device
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__(model, device, session_id, user_id)
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.temperature.to(self.device)
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Temperature Scaling estimator initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={'initial_temperature': temperature}
        )
    
    def calibrate_temperature(
        self,
        validation_loader: torch.utils.data.DataLoader,
        max_iterations: int = 50
    ):
        """
        Calibrate temperature parameter on validation set.
        
        Args:
            validation_loader: Validation data loader
            max_iterations: Maximum optimization iterations
        """
        self.model.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iterations)
        
        def closure():
            optimizer.zero_grad()
            loss = 0
            count = 0
            
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with torch.no_grad():
                    logits = self.model(inputs)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                
                # Apply temperature scaling
                scaled_logits = logits / self.temperature
                loss += F.cross_entropy(scaled_logits, targets)
                count += 1
            
            loss = loss / count
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        log_audit_event(
            event_type=AuditEventType.MODEL_TRAINING,
            severity=AuditSeverity.INFO,
            message="Temperature calibration completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={'calibrated_temperature': float(self.temperature.item())}
        )
    
    def estimate_uncertainty(
        self,
        input_data: torch.Tensor,
        n_samples: int = None
    ) -> UncertaintyEstimate:
        """
        Estimate calibrated uncertainty using temperature scaling.
        
        Args:
            input_data: Input tensor
            n_samples: Not used for temperature scaling
            
        Returns:
            Uncertainty estimation result
        """
        input_data = input_data.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_data)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probabilities = F.softmax(scaled_logits, dim=1)
            
            # Convert to numpy
            predictions = probabilities.cpu().numpy()
        
        # Calculate uncertainties
        # Aleatoric uncertainty (entropy of scaled probabilities)
        entropy = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
        aleatoric_uncertainty = entropy
        
        # Epistemic uncertainty (approximated as temperature-dependent)
        temp_factor = max(0.1, 1.0 / self.temperature.item())
        epistemic_uncertainty = np.full_like(entropy, 0.1 * temp_factor)
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        # Create dummy prediction samples for confidence intervals
        # (Temperature scaling gives single prediction)
        prediction_samples = np.expand_dims(predictions, axis=0)
        confidence_intervals = self._calculate_confidence_intervals(prediction_samples)
        
        # Calculate reliability score
        max_probs = np.max(predictions, axis=1)
        reliability_score = float(np.mean(max_probs))  # Higher max prob = more reliable
        
        result = UncertaintyEstimate(
            predictions=predictions,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            reliability_score=reliability_score,
            method=UncertaintyMethod.TEMPERATURE_SCALING,
            metadata={
                'temperature': float(self.temperature.item()),
                'mean_entropy': float(np.mean(entropy)),
                'mean_max_probability': float(np.mean(max_probs)),
                'calibration_factor': temp_factor
            }
        )
        
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message="Temperature scaling uncertainty estimation completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'temperature': float(self.temperature.item()),
                'reliability_score': reliability_score,
                'mean_uncertainty': float(np.mean(total_uncertainty))
            }
        )
        
        return result


class EvidentialUncertaintyEstimator(BaseUncertaintyEstimator):
    """
    Evidential Deep Learning uncertainty estimation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device: str = "cuda",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize Evidential estimator.
        
        Args:
            model: PyTorch model (should output Dirichlet parameters)
            num_classes: Number of output classes
            device: Computation device
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__(model, device, session_id, user_id)
        self.num_classes = num_classes
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Evidential uncertainty estimator initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={'num_classes': num_classes}
        )
    
    def estimate_uncertainty(
        self,
        input_data: torch.Tensor,
        n_samples: int = None
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using evidential learning.
        
        Args:
            input_data: Input tensor
            n_samples: Not used for evidential learning
            
        Returns:
            Uncertainty estimation result
        """
        input_data = input_data.to(self.device)
        
        with torch.no_grad():
            # Model should output evidence/concentration parameters
            evidence = self.model(input_data)
            if isinstance(evidence, tuple):
                evidence = evidence[0]
            
            # Ensure positive evidence
            evidence = F.relu(evidence) + 1e-8
            
            # Dirichlet parameters (alpha = evidence + 1)
            alpha = evidence + 1
            
            # Calculate predictions (expected probabilities)
            S = torch.sum(alpha, dim=1, keepdim=True)
            predictions = alpha / S
            
            # Convert to numpy
            alpha_np = alpha.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            S_np = S.cpu().numpy().squeeze()
        
        # Calculate uncertainties
        # Aleatoric uncertainty (expected entropy)
        expected_entropy = self._calculate_expected_entropy(alpha_np)
        aleatoric_uncertainty = expected_entropy
        
        # Epistemic uncertainty (mutual information)
        mutual_info = self._calculate_mutual_information(alpha_np)
        epistemic_uncertainty = mutual_info
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        # Create prediction samples for confidence intervals
        # Sample from Dirichlet distribution
        n_mc_samples = 100
        samples = []
        for i in range(len(alpha_np)):
            dirichlet_samples = np.random.dirichlet(alpha_np[i], n_mc_samples)
            samples.append(dirichlet_samples)
        
        samples = np.stack(samples, axis=1)  # Shape: [n_samples, batch_size, n_classes]
        confidence_intervals = self._calculate_confidence_intervals(samples)
        
        # Calculate reliability score based on evidence strength
        reliability_score = float(np.mean(1.0 / (1.0 + self.num_classes / S_np)))
        
        result = UncertaintyEstimate(
            predictions=predictions_np,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            reliability_score=reliability_score,
            method=UncertaintyMethod.EVIDENTIAL,
            metadata={
                'mean_evidence_strength': float(np.mean(S_np - self.num_classes)),
                'evidence_concentration': float(np.mean(np.max(alpha_np, axis=1))),
                'mean_expected_entropy': float(np.mean(expected_entropy)),
                'mean_mutual_information': float(np.mean(mutual_info))
            }
        )
        
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message="Evidential uncertainty estimation completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'reliability_score': reliability_score,
                'mean_evidence_strength': result.metadata['mean_evidence_strength'],
                'mean_uncertainty': float(np.mean(total_uncertainty))
            }
        )
        
        return result
    
    def _calculate_expected_entropy(self, alpha: np.ndarray) -> np.ndarray:
        """Calculate expected entropy of Dirichlet distribution."""
        S = np.sum(alpha, axis=1)
        expected_entropy = np.sum(
            (alpha / S[:, np.newaxis]) * (
                np.log(S[:, np.newaxis] + 1e-8) - 
                np.log(alpha + 1e-8)
            ),
            axis=1
        )
        return expected_entropy
    
    def _calculate_mutual_information(self, alpha: np.ndarray) -> np.ndarray:
        """Calculate mutual information (epistemic uncertainty)."""
        S = np.sum(alpha, axis=1)
        
        # Entropy of mean
        mean_probs = alpha / S[:, np.newaxis]
        entropy_of_mean = -np.sum(
            mean_probs * np.log(mean_probs + 1e-8),
            axis=1
        )
        
        # Expected entropy (calculated above)
        expected_entropy = self._calculate_expected_entropy(alpha)
        
        # Mutual information = entropy of mean - expected entropy
        mutual_info = entropy_of_mean - expected_entropy
        
        return mutual_info


class UncertaintyAggregator:
    """
    Aggregates uncertainty estimates from multiple methods.
    """
    
    def __init__(
        self,
        estimators: Dict[str, BaseUncertaintyEstimator],
        weights: Optional[Dict[str, float]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize uncertainty aggregator.
        
        Args:
            estimators: Dictionary of uncertainty estimators
            weights: Weights for combining estimates (equal if None)
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.estimators = estimators
        self.weights = weights or {name: 1.0 / len(estimators) for name in estimators.keys()}
        self._session_id = session_id
        self._user_id = user_id
        
        # Normalize weights
        weight_sum = sum(self.weights.values())
        self.weights = {name: weight / weight_sum for name, weight in self.weights.items()}
    
    def estimate_uncertainty(
        self,
        input_data: torch.Tensor,
        n_samples: int = 100
    ) -> Dict[str, UncertaintyEstimate]:
        """
        Estimate uncertainty using all methods and return aggregated result.
        
        Args:
            input_data: Input tensor
            n_samples: Number of samples for applicable methods
            
        Returns:
            Dictionary of uncertainty estimates including aggregated result
        """
        estimates = {}
        
        # Get estimates from all methods
        for name, estimator in self.estimators.items():
            try:
                estimate = estimator.estimate_uncertainty(input_data, n_samples)
                estimates[name] = estimate
            except Exception as e:
                warnings.warn(f"Failed to get uncertainty estimate from {name}: {str(e)}")
                continue
        
        # Create aggregated estimate
        if estimates:
            aggregated = self._aggregate_estimates(estimates)
            estimates['aggregated'] = aggregated
        
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message="Aggregated uncertainty estimation completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'methods_used': list(estimates.keys()),
                'weights': self.weights
            }
        )
        
        return estimates
    
    def _aggregate_estimates(
        self,
        estimates: Dict[str, UncertaintyEstimate]
    ) -> UncertaintyEstimate:
        """Aggregate multiple uncertainty estimates."""
        # Weighted average of predictions
        weighted_predictions = np.zeros_like(list(estimates.values())[0].predictions)
        weighted_aleatoric = np.zeros_like(list(estimates.values())[0].aleatoric_uncertainty)
        weighted_epistemic = np.zeros_like(list(estimates.values())[0].epistemic_uncertainty)
        
        total_weight = 0
        
        for name, estimate in estimates.items():
            if name in self.weights:
                weight = self.weights[name]
                weighted_predictions += weight * estimate.predictions
                weighted_aleatoric += weight * estimate.aleatoric_uncertainty
                weighted_epistemic += weight * estimate.epistemic_uncertainty
                total_weight += weight
        
        # Normalize if needed
        if total_weight > 0:
            weighted_predictions /= total_weight
            weighted_aleatoric /= total_weight
            weighted_epistemic /= total_weight
        
        # Total uncertainty
        total_uncertainty = weighted_aleatoric + weighted_epistemic
        
        # Aggregate reliability scores
        reliability_scores = [est.reliability_score for est in estimates.values()]
        aggregated_reliability = float(np.mean(reliability_scores))
        
        # Use first method's confidence intervals (could be improved)
        confidence_intervals = list(estimates.values())[0].confidence_intervals
        
        return UncertaintyEstimate(
            predictions=weighted_predictions,
            aleatoric_uncertainty=weighted_aleatoric,
            epistemic_uncertainty=weighted_epistemic,
            total_uncertainty=total_uncertainty,
            confidence_intervals=confidence_intervals,
            reliability_score=aggregated_reliability,
            method=UncertaintyMethod.DEEP_ENSEMBLE,  # Generic aggregated method
            metadata={
                'aggregated_from': list(estimates.keys()),
                'weights': self.weights,
                'individual_reliability_scores': reliability_scores,
                'method_agreement': self._calculate_method_agreement(estimates)
            }
        )
    
    def _calculate_method_agreement(
        self,
        estimates: Dict[str, UncertaintyEstimate]
    ) -> float:
        """Calculate agreement between different uncertainty methods."""
        if len(estimates) < 2:
            return 1.0
        
        # Calculate pairwise correlation of predictions
        predictions = [est.predictions.flatten() for est in estimates.values()]
        correlations = []
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 1.0


def create_uncertainty_estimator(
    method: UncertaintyMethod,
    model: Union[nn.Module, List[nn.Module]],
    **kwargs
) -> BaseUncertaintyEstimator:
    """
    Factory function to create uncertainty estimator.
    
    Args:
        method: Uncertainty estimation method
        model: Model(s) for uncertainty estimation
        **kwargs: Additional arguments for specific estimators
        
    Returns:
        Configured uncertainty estimator
    """
    if method == UncertaintyMethod.MONTE_CARLO_DROPOUT:
        return MonteCarloDropoutEstimator(model, **kwargs)
    elif method == UncertaintyMethod.DEEP_ENSEMBLE:
        return DeepEnsembleEstimator(model if isinstance(model, list) else [model], **kwargs)
    elif method == UncertaintyMethod.TEMPERATURE_SCALING:
        return TemperatureScalingEstimator(model, **kwargs)
    elif method == UncertaintyMethod.EVIDENTIAL:
        return EvidentialUncertaintyEstimator(model, **kwargs)
    else:
        raise ValueError(f"Unsupported uncertainty method: {method}")


def create_uncertainty_aggregator(
    models: Dict[str, nn.Module],
    methods: List[UncertaintyMethod],
    **kwargs
) -> UncertaintyAggregator:
    """
    Create uncertainty aggregator with multiple methods.
    
    Args:
        models: Dictionary of models for different methods
        methods: List of uncertainty methods to use
        **kwargs: Additional arguments
        
    Returns:
        Configured uncertainty aggregator
    """
    estimators = {}
    
    for method in methods:
        method_name = method.value
        if method_name in models:
            estimators[method_name] = create_uncertainty_estimator(
                method, models[method_name], **kwargs
            )
    
    return UncertaintyAggregator(estimators, **kwargs)