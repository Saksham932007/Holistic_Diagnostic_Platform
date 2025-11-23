"""
NVFlare federated learning integration for medical AI models.

This module provides comprehensive federated learning capabilities using NVIDIA FLARE
for secure, privacy-preserving training across multiple medical institutions.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import nvflare.apis.fl_context as fl_context
    from nvflare.apis.dxo import DXO, from_shareable, DataKind, MetaKey
    from nvflare.apis.executor import Executor
    from nvflare.apis.fl_constant import ReturnCode, ReservedHeaderKey
    from nvflare.apis.shareable import Shareable, make_reply
    from nvflare.apis.signal import Signal
    from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
    from nvflare.app_common.abstract.aggregator import Aggregator
    from nvflare.app_common.np.np_trainer import NPTrainer
    from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
    NVFLARE_AVAILABLE = True
except ImportError:
    NVFLARE_AVAILABLE = False
    # Create mock classes for type hints
    class Executor: pass
    class Aggregator: pass
    class DXO: pass
    class Shareable: pass

from monai.data import DataLoader as MonaiDataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event
from ..models.segmentation import SwinUNetR3D
from ..models.classification import MedicalViT3D
from ..training.trainer import MedicalTrainer


class FederatedStrategy(Enum):
    """Federated learning strategies."""
    FEDERATED_AVERAGING = "fed_avg"
    FEDERATED_PROXIMAL = "fed_prox"
    FEDERATED_LEARNING_WITH_MOMENTUM = "fed_momentum"
    DIFFERENTIAL_PRIVACY = "fed_dp"
    SECURE_AGGREGATION = "secure_agg"


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    strategy: FederatedStrategy = FederatedStrategy.FEDERATED_AVERAGING
    privacy_level: PrivacyLevel = PrivacyLevel.ENHANCED
    num_rounds: int = 100
    min_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    evaluate_fn: Optional[Callable] = None
    on_fit_config_fn: Optional[Callable] = None
    on_evaluate_config_fn: Optional[Callable] = None
    accept_failures: bool = True
    initial_parameters: Optional[torch.Tensor] = None
    fit_metrics_aggregation_fn: Optional[Callable] = None
    evaluate_metrics_aggregation_fn: Optional[Callable] = None
    differential_privacy_config: Dict[str, Any] = field(default_factory=dict)
    secure_aggregation_config: Dict[str, Any] = field(default_factory=dict)


class MedicalFederatedTrainer(Executor):
    """
    Medical AI federated learning trainer using NVFlare.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        local_epochs: int = 5,
        privacy_config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize federated trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Computation device
            learning_rate: Learning rate for optimization
            local_epochs: Number of local training epochs per round
            privacy_config: Privacy preservation configuration
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        if not NVFLARE_AVAILABLE:
            raise ImportError("NVFlare not available. Install with: pip install nvflare")
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.privacy_config = privacy_config or {}
        self._session_id = session_id
        self._user_id = user_id
        
        # Initialize optimizer and loss function
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        
        # Privacy mechanisms
        self._initialize_privacy_mechanisms()
        
        # Training statistics
        self.training_stats = {
            'rounds_completed': 0,
            'local_losses': [],
            'validation_metrics': [],
            'privacy_budget_consumed': 0.0
        }
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Federated trainer initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'model_type': model.__class__.__name__,
                'local_epochs': local_epochs,
                'privacy_enabled': bool(privacy_config)
            }
        )
    
    def _initialize_privacy_mechanisms(self):
        """Initialize differential privacy mechanisms."""
        self.privacy_engine = None
        
        if self.privacy_config.get('enable_dp', False):
            try:
                from opacus import PrivacyEngine
                from opacus.validators import ModuleValidator
                
                # Validate model for differential privacy
                if not ModuleValidator.is_valid(self.model):
                    self.model = ModuleValidator.fix(self.model)
                
                # Initialize privacy engine
                self.privacy_engine = PrivacyEngine()
                
                self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    epochs=self.local_epochs,
                    target_epsilon=self.privacy_config.get('epsilon', 8.0),
                    target_delta=self.privacy_config.get('delta', 1e-5),
                    max_grad_norm=self.privacy_config.get('max_grad_norm', 1.0)
                )
                
                log_audit_event(
                    event_type=AuditEventType.SECURITY_EVENT,
                    severity=AuditSeverity.INFO,
                    message="Differential privacy enabled",
                    user_id=self._user_id,
                    session_id=self._session_id,
                    additional_data={
                        'epsilon': self.privacy_config.get('epsilon', 8.0),
                        'delta': self.privacy_config.get('delta', 1e-5)
                    }
                )
                
            except ImportError:
                warnings.warn("Opacus not available for differential privacy")
    
    def execute(self, task_name: str, shareable: Shareable, fl_ctx) -> Shareable:
        """
        Execute federated learning task.
        
        Args:
            task_name: Name of the federated task
            shareable: Shareable object containing task data
            fl_ctx: Federated learning context
            
        Returns:
            Shareable object with results
        """
        try:
            if task_name == "train":
                return self._train_task(shareable, fl_ctx)
            elif task_name == "validate":
                return self._validate_task(shareable, fl_ctx)
            elif task_name == "submit_model":
                return self._submit_model_task(shareable, fl_ctx)
            else:
                self.log_error(fl_ctx, f"Unknown task: {task_name}")
                return make_reply(ReturnCode.TASK_UNKNOWN)
        
        except Exception as e:
            self.log_error(fl_ctx, f"Error executing task {task_name}: {str(e)}")
            
            log_audit_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                severity=AuditSeverity.ERROR,
                message=f"Federated task execution failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'task_name': task_name,
                    'error': str(e)
                }
            )
            
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
    
    def _train_task(self, shareable: Shareable, fl_ctx) -> Shareable:
        """Execute federated training task."""
        # Extract global model from shareable
        dxo = from_shareable(shareable)
        global_weights = dxo.data
        
        # Update local model with global weights
        self._set_model_weights(global_weights)
        
        # Perform local training
        local_metrics = self._local_training()
        
        # Get updated model weights
        updated_weights = self._get_model_weights()
        
        # Apply privacy mechanisms
        if self.privacy_config.get('add_noise', False):
            updated_weights = self._add_privacy_noise(updated_weights)
        
        # Create DXO for response
        dxo_response = DXO(
            data_kind=DataKind.WEIGHTS,
            data=updated_weights,
            meta={
                MetaKey.NUM_STEPS_CURRENT_ROUND: len(self.train_loader),
                "local_metrics": local_metrics,
                "client_id": fl_ctx.get_identity_name()
            }
        )
        
        self.training_stats['rounds_completed'] += 1
        
        log_audit_event(
            event_type=AuditEventType.MODEL_TRAINING,
            severity=AuditSeverity.INFO,
            message="Local training completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'round': self.training_stats['rounds_completed'],
                'local_metrics': local_metrics,
                'client_id': fl_ctx.get_identity_name()
            }
        )
        
        return dxo_response.to_shareable()
    
    def _validate_task(self, shareable: Shareable, fl_ctx) -> Shareable:
        """Execute federated validation task."""
        # Extract global model from shareable
        dxo = from_shareable(shareable)
        global_weights = dxo.data
        
        # Update local model with global weights
        self._set_model_weights(global_weights)
        
        # Perform validation
        validation_metrics = self._local_validation()
        
        # Create DXO for response
        dxo_response = DXO(
            data_kind=DataKind.METRICS,
            data=validation_metrics,
            meta={"client_id": fl_ctx.get_identity_name()}
        )
        
        log_audit_event(
            event_type=AuditEventType.MODEL_EVALUATION,
            severity=AuditSeverity.INFO,
            message="Local validation completed",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'validation_metrics': validation_metrics,
                'client_id': fl_ctx.get_identity_name()
            }
        )
        
        return dxo_response.to_shareable()
    
    def _submit_model_task(self, shareable: Shareable, fl_ctx) -> Shareable:
        """Submit final model."""
        final_weights = self._get_model_weights()
        
        dxo_response = DXO(
            data_kind=DataKind.WEIGHTS,
            data=final_weights,
            meta={"client_id": fl_ctx.get_identity_name()}
        )
        
        return dxo_response.to_shareable()
    
    def _local_training(self) -> Dict[str, float]:
        """Perform local training for specified epochs."""
        self.model.train()
        
        epoch_losses = []
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in self.train_loader:
                if isinstance(batch, dict):
                    inputs = batch['image'].to(self.device)
                    targets = batch['label'].to(self.device)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                total_samples += inputs.shape[0]
            
            avg_epoch_loss = epoch_loss / batch_count
            epoch_losses.append(avg_epoch_loss)
        
        # Calculate privacy budget if using differential privacy
        if self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(self.privacy_config.get('delta', 1e-5))
            self.training_stats['privacy_budget_consumed'] = epsilon
        
        metrics = {
            'loss': np.mean(epoch_losses),
            'loss_std': np.std(epoch_losses),
            'num_samples': total_samples,
            'num_epochs': self.local_epochs,
            'privacy_epsilon': self.training_stats.get('privacy_budget_consumed', 0.0)
        }
        
        self.training_stats['local_losses'].extend(epoch_losses)
        
        return metrics
    
    def _local_validation(self) -> Dict[str, float]:
        """Perform local validation."""
        self.model.eval()
        
        val_loss = 0.0
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    inputs = batch['image'].to(self.device)
                    targets = batch['label'].to(self.device)
                else:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                val_loss += loss.item()
                
                # Calculate Dice score
                predictions = torch.softmax(outputs, dim=1)
                dice_metric(predictions, targets)
                
                total_samples += inputs.shape[0]
        
        dice_score = dice_metric.aggregate().item()
        dice_metric.reset()
        
        metrics = {
            'val_loss': val_loss / len(self.val_loader),
            'dice_score': dice_score,
            'num_val_samples': total_samples
        }
        
        self.training_stats['validation_metrics'].append(metrics)
        
        return metrics
    
    def _get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current model weights."""
        return {name: param.cpu().detach().clone() for name, param in self.model.named_parameters()}
    
    def _set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from dictionary."""
        model_state = self.model.state_dict()
        
        for name, param in weights.items():
            if name in model_state:
                model_state[name] = param.to(self.device)
        
        self.model.load_state_dict(model_state)
    
    def _add_privacy_noise(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add privacy noise to model weights."""
        if not self.privacy_config.get('add_noise', False):
            return weights
        
        noise_scale = self.privacy_config.get('noise_scale', 0.01)
        
        noisy_weights = {}
        for name, param in weights.items():
            noise = torch.randn_like(param) * noise_scale
            noisy_weights[name] = param + noise
        
        return noisy_weights


class MedicalFederatedAggregator(Aggregator):
    """
    Medical AI federated aggregator using secure aggregation.
    """
    
    def __init__(
        self,
        aggregation_strategy: FederatedStrategy = FederatedStrategy.FEDERATED_AVERAGING,
        min_clients: int = 2,
        privacy_config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize federated aggregator.
        
        Args:
            aggregation_strategy: Strategy for aggregating client updates
            min_clients: Minimum number of clients for aggregation
            privacy_config: Privacy configuration
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        if not NVFLARE_AVAILABLE:
            raise ImportError("NVFlare not available")
        
        self.aggregation_strategy = aggregation_strategy
        self.min_clients = min_clients
        self.privacy_config = privacy_config or {}
        self._session_id = session_id
        self._user_id = user_id
        
        # Aggregation statistics
        self.aggregation_stats = {
            'rounds_completed': 0,
            'clients_participated': [],
            'aggregation_metrics': []
        }
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Federated aggregator initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'strategy': aggregation_strategy.value,
                'min_clients': min_clients
            }
        )
    
    def accept(self, shareable: Shareable, fl_ctx) -> bool:
        """Accept client contribution for aggregation."""
        try:
            dxo = from_shareable(shareable)
            
            # Validate contribution
            if dxo.data_kind != DataKind.WEIGHTS:
                return False
            
            # Additional privacy validation
            if self.privacy_config.get('validate_privacy', False):
                if not self._validate_privacy_compliance(dxo, fl_ctx):
                    return False
            
            return True
            
        except Exception as e:
            self.log_error(fl_ctx, f"Error accepting contribution: {str(e)}")
            return False
    
    def aggregate(self, shareables: List[Shareable], fl_ctx) -> Shareable:
        """
        Aggregate client contributions.
        
        Args:
            shareables: List of client contributions
            fl_ctx: Federated learning context
            
        Returns:
            Aggregated model
        """
        try:
            if len(shareables) < self.min_clients:
                raise ValueError(f"Insufficient clients: {len(shareables)} < {self.min_clients}")
            
            # Extract client data
            client_weights = []
            client_samples = []
            client_metrics = []
            
            for shareable in shareables:
                dxo = from_shareable(shareable)
                weights = dxo.data
                meta = dxo.meta or {}
                
                client_weights.append(weights)
                client_samples.append(meta.get(MetaKey.NUM_STEPS_CURRENT_ROUND, 1))
                client_metrics.append(meta.get('local_metrics', {}))
            
            # Perform aggregation based on strategy
            if self.aggregation_strategy == FederatedStrategy.FEDERATED_AVERAGING:
                aggregated_weights = self._federated_averaging(client_weights, client_samples)
            elif self.aggregation_strategy == FederatedStrategy.FEDERATED_PROXIMAL:
                aggregated_weights = self._federated_proximal(client_weights, client_samples)
            elif self.aggregation_strategy == FederatedStrategy.SECURE_AGGREGATION:
                aggregated_weights = self._secure_aggregation(client_weights, client_samples)
            else:
                aggregated_weights = self._federated_averaging(client_weights, client_samples)
            
            # Create aggregated DXO
            aggregated_dxo = DXO(
                data_kind=DataKind.WEIGHTS,
                data=aggregated_weights,
                meta={
                    "num_clients": len(shareables),
                    "aggregation_strategy": self.aggregation_strategy.value,
                    "round": self.aggregation_stats['rounds_completed'] + 1
                }
            )
            
            # Update statistics
            self.aggregation_stats['rounds_completed'] += 1
            self.aggregation_stats['clients_participated'].append(len(shareables))
            
            # Calculate aggregation metrics
            agg_metrics = self._calculate_aggregation_metrics(client_metrics)
            self.aggregation_stats['aggregation_metrics'].append(agg_metrics)
            
            log_audit_event(
                event_type=AuditEventType.MODEL_TRAINING,
                severity=AuditSeverity.INFO,
                message="Model aggregation completed",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'round': self.aggregation_stats['rounds_completed'],
                    'num_clients': len(shareables),
                    'strategy': self.aggregation_strategy.value,
                    'aggregation_metrics': agg_metrics
                }
            )
            
            return aggregated_dxo.to_shareable()
            
        except Exception as e:
            self.log_error(fl_ctx, f"Error during aggregation: {str(e)}")
            
            log_audit_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                severity=AuditSeverity.ERROR,
                message=f"Model aggregation failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            
            raise
    
    def _federated_averaging(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Perform federated averaging aggregation."""
        # Weighted average based on number of samples
        total_samples = sum(client_samples)
        aggregated_weights = {}
        
        # Initialize with first client weights
        for name in client_weights[0].keys():
            aggregated_weights[name] = torch.zeros_like(client_weights[0][name])
        
        # Weighted sum
        for i, weights in enumerate(client_weights):
            weight_factor = client_samples[i] / total_samples
            
            for name, param in weights.items():
                if name in aggregated_weights:
                    aggregated_weights[name] += weight_factor * param
        
        return aggregated_weights
    
    def _federated_proximal(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Perform federated proximal aggregation."""
        # For now, use standard federated averaging
        # In practice, would incorporate proximal term
        return self._federated_averaging(client_weights, client_samples)
    
    def _secure_aggregation(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation."""
        # Simplified secure aggregation (in practice would use cryptographic protocols)
        aggregated = self._federated_averaging(client_weights, client_samples)
        
        # Add minimal noise for additional privacy
        if self.privacy_config.get('add_aggregation_noise', False):
            noise_scale = self.privacy_config.get('aggregation_noise_scale', 1e-5)
            
            for name, param in aggregated.items():
                noise = torch.randn_like(param) * noise_scale
                aggregated[name] += noise
        
        return aggregated
    
    def _validate_privacy_compliance(self, dxo: DXO, fl_ctx) -> bool:
        """Validate privacy compliance of client contribution."""
        # Implement privacy validation logic
        # Check for gradient leakage, membership inference attacks, etc.
        return True
    
    def _calculate_aggregation_metrics(
        self,
        client_metrics: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate aggregation-level metrics."""
        if not client_metrics:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        for metrics in client_metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Calculate aggregated statistics
        agg_metrics = {}
        for key, values in numeric_metrics.items():
            agg_metrics[f"{key}_mean"] = float(np.mean(values))
            agg_metrics[f"{key}_std"] = float(np.std(values))
            agg_metrics[f"{key}_min"] = float(np.min(values))
            agg_metrics[f"{key}_max"] = float(np.max(values))
        
        # Add aggregation-specific metrics
        agg_metrics['client_diversity'] = self._calculate_client_diversity(client_metrics)
        
        return agg_metrics
    
    def _calculate_client_diversity(self, client_metrics: List[Dict[str, Any]]) -> float:
        """Calculate diversity score among clients."""
        if len(client_metrics) <= 1:
            return 0.0
        
        # Simple diversity measure based on loss variance
        losses = [m.get('loss', 0.0) for m in client_metrics]
        return float(np.std(losses)) if losses else 0.0


class FederatedLearningOrchestrator:
    """
    Orchestrator for federated learning workflow.
    """
    
    def __init__(
        self,
        config: FederatedConfig,
        model_template: nn.Module,
        aggregator: MedicalFederatedAggregator,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize federated learning orchestrator.
        
        Args:
            config: Federated learning configuration
            model_template: Template model for federated learning
            aggregator: Federated aggregator
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.config = config
        self.model_template = model_template
        self.aggregator = aggregator
        self._session_id = session_id
        self._user_id = user_id
        
        # Federated learning state
        self.current_round = 0
        self.global_model_weights = None
        self.federation_metrics = {
            'round_metrics': [],
            'client_participation': [],
            'convergence_metrics': []
        }
        
        # Initialize global model
        self._initialize_global_model()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Federated learning orchestrator initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'strategy': config.strategy.value,
                'num_rounds': config.num_rounds,
                'min_clients': config.min_clients
            }
        )
    
    def _initialize_global_model(self):
        """Initialize global model weights."""
        self.global_model_weights = {
            name: param.cpu().detach().clone() 
            for name, param in self.model_template.named_parameters()
        }
    
    def run_federated_learning(
        self,
        client_trainers: List[MedicalFederatedTrainer]
    ) -> Dict[str, Any]:
        """
        Run complete federated learning workflow.
        
        Args:
            client_trainers: List of client trainers
            
        Returns:
            Federated learning results
        """
        try:
            for round_num in range(self.config.num_rounds):
                self.current_round = round_num + 1
                
                # Select clients for this round
                selected_clients = self._select_clients(client_trainers)
                
                if len(selected_clients) < self.config.min_clients:
                    warnings.warn(f"Insufficient clients for round {self.current_round}")
                    continue
                
                # Distribute global model to selected clients
                client_updates = self._training_round(selected_clients)
                
                # Aggregate client updates
                if client_updates:
                    self.global_model_weights = self._aggregate_round(client_updates)
                
                # Evaluate global model
                round_metrics = self._evaluate_round(selected_clients)
                
                # Check convergence
                if self._check_convergence(round_metrics):
                    log_audit_event(
                        event_type=AuditEventType.MODEL_TRAINING,
                        severity=AuditSeverity.INFO,
                        message=f"Federated learning converged at round {self.current_round}",
                        user_id=self._user_id,
                        session_id=self._session_id,
                        additional_data={'convergence_round': self.current_round}
                    )
                    break
                
                # Update federation metrics
                self._update_federation_metrics(round_metrics, selected_clients)
            
            # Generate final results
            results = self._generate_federation_results()
            
            log_audit_event(
                event_type=AuditEventType.MODEL_TRAINING,
                severity=AuditSeverity.INFO,
                message="Federated learning completed",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'total_rounds': self.current_round,
                    'final_metrics': results.get('final_metrics', {})
                }
            )
            
            return results
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.SYSTEM_ERROR,
                severity=AuditSeverity.ERROR,
                message=f"Federated learning failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise
    
    def _select_clients(
        self,
        client_trainers: List[MedicalFederatedTrainer]
    ) -> List[MedicalFederatedTrainer]:
        """Select clients for current round."""
        num_select = max(
            self.config.min_fit_clients,
            int(len(client_trainers) * self.config.fraction_fit)
        )
        
        # Simple random selection (could be improved with more sophisticated strategies)
        import random
        selected = random.sample(client_trainers, min(num_select, len(client_trainers)))
        
        return selected
    
    def _training_round(
        self,
        selected_clients: List[MedicalFederatedTrainer]
    ) -> List[Dict[str, torch.Tensor]]:
        """Execute training round with selected clients."""
        client_updates = []
        
        for client in selected_clients:
            try:
                # Set global model weights
                client._set_model_weights(self.global_model_weights)
                
                # Perform local training
                local_metrics = client._local_training()
                
                # Get updated weights
                updated_weights = client._get_model_weights()
                client_updates.append(updated_weights)
                
            except Exception as e:
                warnings.warn(f"Client training failed: {str(e)}")
                continue
        
        return client_updates
    
    def _aggregate_round(
        self,
        client_updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates for current round."""
        # Simple federated averaging
        total_clients = len(client_updates)
        aggregated_weights = {}
        
        # Initialize with first client weights
        for name in client_updates[0].keys():
            aggregated_weights[name] = torch.zeros_like(client_updates[0][name])
        
        # Average all client weights
        for weights in client_updates:
            for name, param in weights.items():
                if name in aggregated_weights:
                    aggregated_weights[name] += param / total_clients
        
        return aggregated_weights
    
    def _evaluate_round(
        self,
        selected_clients: List[MedicalFederatedTrainer]
    ) -> Dict[str, float]:
        """Evaluate global model on selected clients."""
        round_metrics = []
        
        for client in selected_clients:
            try:
                # Set global model weights
                client._set_model_weights(self.global_model_weights)
                
                # Perform validation
                val_metrics = client._local_validation()
                round_metrics.append(val_metrics)
                
            except Exception as e:
                warnings.warn(f"Client evaluation failed: {str(e)}")
                continue
        
        # Aggregate validation metrics
        if round_metrics:
            aggregated_metrics = {}
            for key in round_metrics[0].keys():
                values = [m[key] for m in round_metrics if key in m]
                aggregated_metrics[key] = float(np.mean(values))
                aggregated_metrics[f"{key}_std"] = float(np.std(values))
            
            return aggregated_metrics
        
        return {}
    
    def _check_convergence(self, round_metrics: Dict[str, float]) -> bool:
        """Check if federated learning has converged."""
        if len(self.federation_metrics['round_metrics']) < 5:
            return False
        
        # Simple convergence check based on loss stabilization
        recent_losses = [
            m.get('val_loss', float('inf')) 
            for m in self.federation_metrics['round_metrics'][-5:]
        ]
        
        if recent_losses:
            loss_variance = np.var(recent_losses)
            return loss_variance < 1e-6  # Convergence threshold
        
        return False
    
    def _update_federation_metrics(
        self,
        round_metrics: Dict[str, float],
        selected_clients: List[MedicalFederatedTrainer]
    ):
        """Update federation-level metrics."""
        self.federation_metrics['round_metrics'].append({
            'round': self.current_round,
            **round_metrics
        })
        
        self.federation_metrics['client_participation'].append({
            'round': self.current_round,
            'num_clients': len(selected_clients),
            'participation_rate': len(selected_clients) / len(selected_clients)  # Simplified
        })
    
    def _generate_federation_results(self) -> Dict[str, Any]:
        """Generate final federated learning results."""
        return {
            'final_model_weights': self.global_model_weights,
            'total_rounds': self.current_round,
            'federation_metrics': self.federation_metrics,
            'final_metrics': self.federation_metrics['round_metrics'][-1] if self.federation_metrics['round_metrics'] else {},
            'convergence_achieved': len(self.federation_metrics['round_metrics']) < self.config.num_rounds,
            'avg_client_participation': np.mean([
                p['num_clients'] for p in self.federation_metrics['client_participation']
            ]) if self.federation_metrics['client_participation'] else 0
        }


def create_federated_setup(
    model_class: type,
    model_kwargs: Dict[str, Any],
    federated_config: FederatedConfig,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Tuple[MedicalFederatedAggregator, FederatedLearningOrchestrator]:
    """
    Create federated learning setup.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Model initialization arguments
        federated_config: Federated learning configuration
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Tuple of aggregator and orchestrator
    """
    if not NVFLARE_AVAILABLE:
        raise ImportError("NVFlare not available for federated learning")
    
    # Create model template
    model_template = model_class(**model_kwargs)
    
    # Create aggregator
    aggregator = MedicalFederatedAggregator(
        aggregation_strategy=federated_config.strategy,
        min_clients=federated_config.min_clients,
        session_id=session_id,
        user_id=user_id
    )
    
    # Create orchestrator
    orchestrator = FederatedLearningOrchestrator(
        config=federated_config,
        model_template=model_template,
        aggregator=aggregator,
        session_id=session_id,
        user_id=user_id
    )
    
    return aggregator, orchestrator