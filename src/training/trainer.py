"""
Training infrastructure for medical image segmentation models.

This module provides a comprehensive training framework with HIPAA compliance,
automatic mixed precision, checkpointing, early stopping, and clinical evaluation.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable, Union
import os
import time
import warnings
from pathlib import Path
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete
from monai.utils import set_determinism

from ..config import get_config
from ..models.segmentation import MedicalSwinUNETR
from ..models.loss import create_loss_function
from ..models.metrics import create_metrics_evaluator, compute_comprehensive_metrics
from ..utils import AuditEventType, AuditSeverity, log_audit_event
from ..security.encryption import EncryptionManager


class MedicalTrainer:
    """
    Comprehensive trainer for medical image segmentation with HIPAA compliance.
    
    Features:
    - Automatic mixed precision training
    - Comprehensive metrics evaluation
    - Early stopping with patience
    - Model checkpointing with encryption
    - HIPAA-compliant audit logging
    - Clinical evaluation integration
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_function: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        max_epochs: int = 100,
        val_interval: int = 1,
        early_stopping_patience: int = 10,
        save_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        roi_size: Tuple[int, ...] = (96, 96, 96),
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize the medical trainer.
        
        Args:
            model: Neural network model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_function: Loss function (created if None)
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler
            device: Training device
            use_amp: Whether to use automatic mixed precision
            max_epochs: Maximum training epochs
            val_interval: Validation interval in epochs
            early_stopping_patience: Early stopping patience
            save_dir: Directory to save checkpoints
            experiment_name: Name for the experiment
            roi_size: ROI size for sliding window inference
            sw_batch_size: Sliding window batch size
            overlap: Sliding window overlap ratio
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.early_stopping_patience = early_stopping_patience
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self._session_id = session_id
        self._user_id = user_id
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model.to(self.device)
        
        # AMP setup
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Loss function
        if loss_function is None:
            self.loss_function = create_loss_function(
                loss_type="dice_ce",
                session_id=session_id,
                user_id=user_id
            )
        else:
            self.loss_function = loss_function
            
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        
        # Metrics
        self.metrics_evaluator = create_metrics_evaluator(
            session_id=session_id,
            user_id=user_id
        )
        
        # Post-processing transforms
        self.post_pred = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])
        self.post_label = Compose([
            AsDiscrete(to_onehot=None)
        ])
        
        # Experiment tracking
        if save_dir is None:
            save_dir = f"experiments/{experiment_name or 'medical_segmentation'}"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logging
        self.writer = SummaryWriter(
            log_dir=self.save_dir / "tensorboard"
        )
        
        # Encryption for sensitive data
        self.encryption_manager = EncryptionManager()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.early_stopping_counter = 0
        self.training_history = []
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Medical trainer initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'model_type': type(self.model).__name__,
                'device': str(self.device),
                'use_amp': self.use_amp,
                'max_epochs': self.max_epochs,
                'early_stopping_patience': self.early_stopping_patience
            }
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            try:
                # Extract data
                if isinstance(batch_data, dict):
                    inputs = batch_data["image"].to(self.device, non_blocking=True)
                    labels = batch_data["label"].to(self.device, non_blocking=True)
                else:
                    inputs, labels = batch_data
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with AMP
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.loss_function(outputs, labels)
                    
                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                self.global_step += 1
                
                # Log batch metrics
                if batch_idx % 10 == 0:
                    self.writer.add_scalar(
                        "train/batch_loss", 
                        loss.item(), 
                        self.global_step
                    )
                    
                    log_audit_event(
                        event_type=AuditEventType.MODEL_TRAINING,
                        severity=AuditSeverity.DEBUG,
                        message=f"Training batch {batch_idx}/{num_batches}",
                        user_id=self._user_id,
                        session_id=self._session_id,
                        additional_data={
                            'epoch': self.epoch,
                            'batch_idx': batch_idx,
                            'batch_loss': loss.item()
                        }
                    )
                
            except Exception as e:
                log_audit_event(
                    event_type=AuditEventType.MODEL_TRAINING,
                    severity=AuditSeverity.ERROR,
                    message=f"Training batch failed: {str(e)}",
                    user_id=self._user_id,
                    session_id=self._session_id,
                    additional_data={
                        'epoch': self.epoch,
                        'batch_idx': batch_idx,
                        'error': str(e)
                    }
                )
                raise
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / num_batches
        
        # Update learning rate
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Validation metrics for the epoch
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Reset metrics
        self.metrics_evaluator.reset() if hasattr(self.metrics_evaluator, 'reset') else None
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                try:
                    # Extract data
                    if isinstance(batch_data, dict):
                        inputs = batch_data["image"].to(self.device, non_blocking=True)
                        labels = batch_data["label"].to(self.device, non_blocking=True)
                    else:
                        inputs, labels = batch_data
                        inputs = inputs.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                    
                    # Forward pass with sliding window inference
                    if self.use_amp:
                        with autocast():
                            outputs = sliding_window_inference(
                                inputs=inputs,
                                roi_size=self.roi_size,
                                sw_batch_size=self.sw_batch_size,
                                predictor=self.model,
                                overlap=self.overlap,
                                mode="gaussian",
                                sigma_scale=0.125,
                                padding_mode="constant",
                                cval=0.0
                            )
                            loss = self.loss_function(outputs, labels)
                    else:
                        outputs = sliding_window_inference(
                            inputs=inputs,
                            roi_size=self.roi_size,
                            sw_batch_size=self.sw_batch_size,
                            predictor=self.model,
                            overlap=self.overlap,
                            mode="gaussian",
                            sigma_scale=0.125,
                            padding_mode="constant",
                            cval=0.0
                        )
                        loss = self.loss_function(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    # Post-process predictions for metrics
                    outputs_list = decollate_batch(outputs)
                    labels_list = decollate_batch(labels)
                    
                    # Apply post-processing
                    outputs_processed = [
                        self.post_pred(pred) for pred in outputs_list
                    ]
                    labels_processed = [
                        self.post_label(label) for label in labels_list
                    ]
                    
                    all_predictions.extend(outputs_processed)
                    all_labels.extend(labels_processed)
                    
                except Exception as e:
                    log_audit_event(
                        event_type=AuditEventType.MODEL_VALIDATION,
                        severity=AuditSeverity.ERROR,
                        message=f"Validation batch failed: {str(e)}",
                        user_id=self._user_id,
                        session_id=self._session_id,
                        additional_data={
                            'epoch': self.epoch,
                            'batch_idx': batch_idx,
                            'error': str(e)
                        }
                    )
                    continue
        
        # Compute comprehensive metrics
        if all_predictions and all_labels:
            # Stack all predictions and labels
            all_pred_tensor = torch.stack(all_predictions)
            all_label_tensor = torch.stack(all_labels)
            
            # Compute metrics
            try:
                metrics = compute_comprehensive_metrics(
                    y_pred=all_pred_tensor,
                    y_true=all_label_tensor,
                    session_id=self._session_id,
                    user_id=self._user_id
                )
                
                val_metrics = {
                    'loss': val_loss / num_batches,
                    'dice': metrics['standard']['dice'],
                    'iou': metrics['standard']['iou'],
                    'sensitivity': metrics['standard']['sensitivity'],
                    'specificity': metrics['standard']['specificity'],
                    'hausdorff': metrics['standard'].get('hausdorff', float('nan')),
                    'volume_similarity': metrics['standard']['volume_similarity']
                }
                
                # Add clinical metrics if available
                if metrics['clinical']:
                    val_metrics.update({
                        f"clinical_{k}": v for k, v in metrics['clinical'].items()
                    })
                
            except Exception as e:
                log_audit_event(
                    event_type=AuditEventType.MODEL_VALIDATION,
                    severity=AuditSeverity.WARNING,
                    message=f"Metrics computation failed: {str(e)}",
                    user_id=self._user_id,
                    session_id=self._session_id,
                    additional_data={'error': str(e)}
                )
                val_metrics = {'loss': val_loss / num_batches}
        else:
            val_metrics = {'loss': val_loss / num_batches}
        
        return val_metrics
    
    def save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint with encryption.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'global_step': self.global_step,
                'training_history': self.training_history,
                'config': {
                    'model_type': type(self.model).__name__,
                    'roi_size': self.roi_size,
                    'use_amp': self.use_amp
                }
            }
            
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Save regular checkpoint
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if is_best:
                best_path = self.save_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                
                # Encrypt best model for security
                encrypted_path = self.save_dir / "best_model_encrypted.bin"
                with open(best_path, 'rb') as f:
                    model_data = f.read()
                
                encrypted_data = self.encryption_manager.encrypt_data(
                    model_data,
                    context="model_checkpoint"
                )
                
                with open(encrypted_path, 'wb') as f:
                    f.write(encrypted_data)
            
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.INFO,
                message=f"Model checkpoint saved for epoch {epoch}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'epoch': epoch,
                    'is_best': is_best,
                    'metrics': metrics,
                    'checkpoint_path': str(checkpoint_path)
                }
            )
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.ERROR,
                message=f"Failed to save checkpoint: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'epoch': epoch,
                    'error': str(e)
                }
            )
            raise
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training summary with final metrics and history
        """
        log_audit_event(
            event_type=AuditEventType.MODEL_TRAINING,
            severity=AuditSeverity.INFO,
            message="Starting model training",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'max_epochs': self.max_epochs,
                'device': str(self.device),
                'use_amp': self.use_amp
            }
        )
        
        try:
            start_time = time.time()
            
            for epoch in range(self.max_epochs):
                self.epoch = epoch
                epoch_start = time.time()
                
                # Training
                train_metrics = self.train_epoch()
                
                # Validation
                val_metrics = {}
                if epoch % self.val_interval == 0:
                    val_metrics = self.validate_epoch()
                    
                    # Log metrics to TensorBoard
                    for key, value in train_metrics.items():
                        self.writer.add_scalar(f"train/{key}", value, epoch)
                    
                    for key, value in val_metrics.items():
                        if not np.isnan(value):
                            self.writer.add_scalar(f"val/{key}", value, epoch)
                    
                    # Check for best model
                    current_metric = val_metrics.get('dice', 0)
                    is_best = current_metric > self.best_metric
                    
                    if is_best:
                        self.best_metric = current_metric
                        self.best_metric_epoch = epoch
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                    
                    # Save checkpoint
                    self.save_checkpoint(epoch, val_metrics, is_best)
                    
                    # Early stopping
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        log_audit_event(
                            event_type=AuditEventType.MODEL_TRAINING,
                            severity=AuditSeverity.INFO,
                            message=f"Early stopping triggered at epoch {epoch}",
                            user_id=self._user_id,
                            session_id=self._session_id,
                            additional_data={
                                'best_metric': self.best_metric,
                                'best_epoch': self.best_metric_epoch,
                                'patience': self.early_stopping_patience
                            }
                        )
                        break
                
                # Record training history
                epoch_summary = {
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'epoch_time': time.time() - epoch_start,
                    'timestamp': datetime.now().isoformat()
                }
                self.training_history.append(epoch_summary)
                
                # Print progress
                if val_metrics:
                    print(f"Epoch {epoch:03d}: "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Val Dice: {val_metrics.get('dice', 0):.4f}, "
                          f"Val Loss: {val_metrics.get('loss', 0):.4f}")
                else:
                    print(f"Epoch {epoch:03d}: "
                          f"Train Loss: {train_metrics['loss']:.4f}")
            
            total_time = time.time() - start_time
            
            # Training summary
            summary = {
                'total_epochs': epoch + 1,
                'total_time': total_time,
                'best_metric': self.best_metric,
                'best_epoch': self.best_metric_epoch,
                'final_metrics': val_metrics,
                'training_history': self.training_history
            }
            
            log_audit_event(
                event_type=AuditEventType.MODEL_TRAINING,
                severity=AuditSeverity.INFO,
                message="Model training completed",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data=summary
            )
            
            return summary
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_TRAINING,
                severity=AuditSeverity.ERROR,
                message=f"Training failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise
        
        finally:
            self.writer.close()
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.training_history = checkpoint.get('training_history', [])
            
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.INFO,
                message=f"Checkpoint loaded from {checkpoint_path}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'epoch': self.epoch,
                    'checkpoint_path': checkpoint_path
                }
            )
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.ERROR,
                message=f"Failed to load checkpoint: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'checkpoint_path': checkpoint_path,
                    'error': str(e)
                }
            )
            raise


def create_trainer(
    config_override: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> MedicalTrainer:
    """
    Factory function to create trainer from configuration.
    
    Args:
        config_override: Optional parameter overrides
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        **kwargs: Additional trainer parameters
        
    Returns:
        Configured trainer instance
    """
    config = get_config()
    
    # Apply configuration overrides
    params = config_override or {}
    params.update(kwargs)
    
    # Extract required parameters
    model = params.pop('model')
    train_loader = params.pop('train_loader')
    val_loader = params.pop('val_loader')
    
    return MedicalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        session_id=session_id,
        user_id=user_id,
        **params
    )