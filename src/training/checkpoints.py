"""
Enhanced model checkpointing with encryption, versioning, and restoration.

This module provides advanced checkpoint management for medical AI models
with HIPAA compliance, version control, and secure storage capabilities.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import pickle
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event
from ..security.encryption import EncryptionManager


class ModelCheckpointManager:
    """
    Advanced checkpoint management with encryption and versioning.
    
    Features:
    - Encrypted model state storage
    - Version control with metadata
    - Automatic cleanup of old checkpoints
    - Integrity verification
    - Rollback capabilities
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_versions: int = 10,
        encrypt_checkpoints: bool = True,
        compress_checkpoints: bool = True,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_versions: Maximum number of versions to keep
            encrypt_checkpoints: Whether to encrypt checkpoint files
            compress_checkpoints: Whether to compress checkpoint files
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_versions = max_versions
        self.encrypt_checkpoints = encrypt_checkpoints
        self.compress_checkpoints = compress_checkpoints
        self._session_id = session_id
        self._user_id = user_id
        
        # Initialize encryption manager
        if encrypt_checkpoints:
            self.encryption_manager = EncryptionManager()
        
        # Metadata file for version tracking
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Model checkpoint manager initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'checkpoint_dir': str(self.checkpoint_dir),
                'max_versions': max_versions,
                'encrypt_checkpoints': encrypt_checkpoints
            }
        )
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save model checkpoint with encryption and versioning.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Performance metrics
            scheduler: Learning rate scheduler
            additional_data: Additional data to save
            is_best: Whether this is the best checkpoint
            checkpoint_name: Custom checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        try:
            # Generate checkpoint info
            checkpoint_id = self._generate_checkpoint_id()
            timestamp = datetime.now().isoformat()
            
            if checkpoint_name is None:
                checkpoint_name = f"checkpoint_{checkpoint_id}"
            
            # Create checkpoint data
            checkpoint_data = {
                'checkpoint_id': checkpoint_id,
                'timestamp': timestamp,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'model_config': {
                    'model_type': type(model).__name__,
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                }
            }
            
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            if additional_data is not None:
                checkpoint_data['additional_data'] = additional_data
            
            # Calculate model hash for integrity
            model_hash = self._calculate_model_hash(model)
            checkpoint_data['model_hash'] = model_hash
            
            # Save checkpoint
            checkpoint_path = self._save_checkpoint_data(
                checkpoint_data, 
                checkpoint_name
            )
            
            # Update metadata
            self._update_metadata(
                checkpoint_id=checkpoint_id,
                checkpoint_name=checkpoint_name,
                checkpoint_path=str(checkpoint_path),
                epoch=epoch,
                metrics=metrics,
                timestamp=timestamp,
                is_best=is_best,
                model_hash=model_hash
            )
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.INFO,
                message=f"Checkpoint saved: {checkpoint_name}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'checkpoint_id': checkpoint_id,
                    'epoch': epoch,
                    'is_best': is_best,
                    'metrics': metrics,
                    'checkpoint_path': str(checkpoint_path)
                }
            )
            
            return str(checkpoint_path)
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.ERROR,
                message=f"Failed to save checkpoint: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_id: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        load_best: bool = False,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint with verification.
        
        Args:
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_id: Specific checkpoint ID to load
            checkpoint_name: Specific checkpoint name to load
            load_best: Whether to load the best checkpoint
            device: Device to load model to
            
        Returns:
            Checkpoint metadata and additional data
        """
        try:
            # Determine which checkpoint to load
            if load_best:
                checkpoint_info = self._get_best_checkpoint()
            elif checkpoint_id:
                checkpoint_info = self._get_checkpoint_by_id(checkpoint_id)
            elif checkpoint_name:
                checkpoint_info = self._get_checkpoint_by_name(checkpoint_name)
            else:
                checkpoint_info = self._get_latest_checkpoint()
            
            if checkpoint_info is None:
                raise ValueError("No checkpoint found matching criteria")
            
            # Load checkpoint data
            checkpoint_data = self._load_checkpoint_data(
                checkpoint_info['checkpoint_path']
            )
            
            # Verify integrity
            if not self._verify_checkpoint_integrity(checkpoint_data):
                raise ValueError("Checkpoint integrity verification failed")
            
            # Load model state
            if device is None:
                device = next(model.parameters()).device
            
            model.load_state_dict(checkpoint_data['model_state_dict'])
            model.to(device)
            
            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.INFO,
                message=f"Checkpoint loaded: {checkpoint_info['checkpoint_name']}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'checkpoint_id': checkpoint_info['checkpoint_id'],
                    'epoch': checkpoint_info['epoch'],
                    'metrics': checkpoint_info['metrics']
                }
            )
            
            return {
                'checkpoint_info': checkpoint_info,
                'epoch': checkpoint_data['epoch'],
                'metrics': checkpoint_data['metrics'],
                'additional_data': checkpoint_data.get('additional_data', {})
            }
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.ERROR,
                message=f"Failed to load checkpoint: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        return sorted(
            self.metadata.get('checkpoints', []),
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            checkpoint_info = self._get_checkpoint_by_id(checkpoint_id)
            if checkpoint_info is None:
                return False
            
            # Delete checkpoint file
            checkpoint_path = Path(checkpoint_info['checkpoint_path'])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Remove from metadata
            self.metadata['checkpoints'] = [
                cp for cp in self.metadata['checkpoints']
                if cp['checkpoint_id'] != checkpoint_id
            ]
            self._save_metadata()
            
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.INFO,
                message=f"Checkpoint deleted: {checkpoint_id}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'checkpoint_id': checkpoint_id}
            )
            
            return True
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_CHECKPOINT,
                severity=AuditSeverity.ERROR,
                message=f"Failed to delete checkpoint: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'checkpoint_id': checkpoint_id,
                    'error': str(e)
                }
            )
            return False
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID."""
        return hashlib.md5(
            f"{datetime.now().isoformat()}{os.urandom(16).hex()}".encode()
        ).hexdigest()[:16]
    
    def _calculate_model_hash(self, model: nn.Module) -> str:
        """Calculate hash of model parameters for integrity verification."""
        model_bytes = pickle.dumps(model.state_dict())
        return hashlib.sha256(model_bytes).hexdigest()
    
    def _save_checkpoint_data(
        self, 
        checkpoint_data: Dict[str, Any], 
        checkpoint_name: str
    ) -> Path:
        """Save checkpoint data to file with optional encryption."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            torch.save(checkpoint_data, tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            if self.encrypt_checkpoints:
                # Read temporary file and encrypt
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                
                encrypted_data = self.encryption_manager.encrypt_data(
                    data, 
                    context="model_checkpoint"
                )
                
                # Save encrypted data
                with open(checkpoint_path, 'wb') as f:
                    f.write(encrypted_data)
            else:
                # Move temporary file to final location
                shutil.move(tmp_path, checkpoint_path)
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return checkpoint_path
    
    def _load_checkpoint_data(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint data from file with optional decryption."""
        checkpoint_path = Path(checkpoint_path)
        
        if self.encrypt_checkpoints:
            # Read and decrypt
            with open(checkpoint_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.encryption_manager.decrypt_data(
                encrypted_data,
                context="model_checkpoint"
            )
            
            # Load from decrypted bytes
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(decrypted_data)
                tmp_path = tmp_file.name
            
            try:
                checkpoint_data = torch.load(tmp_path, map_location='cpu')
            finally:
                os.unlink(tmp_path)
        else:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        return checkpoint_data
    
    def _verify_checkpoint_integrity(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Verify checkpoint integrity using stored hash."""
        if 'model_hash' not in checkpoint_data:
            return True  # Skip verification for old checkpoints
        
        # Create temporary model to calculate hash
        try:
            stored_hash = checkpoint_data['model_hash']
            current_hash = hashlib.sha256(
                pickle.dumps(checkpoint_data['model_state_dict'])
            ).hexdigest()
            return stored_hash == current_hash
        except:
            return False
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'version': '1.0'}
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _update_metadata(
        self,
        checkpoint_id: str,
        checkpoint_name: str,
        checkpoint_path: str,
        epoch: int,
        metrics: Dict[str, float],
        timestamp: str,
        is_best: bool,
        model_hash: str
    ) -> None:
        """Update checkpoint metadata."""
        checkpoint_info = {
            'checkpoint_id': checkpoint_id,
            'checkpoint_name': checkpoint_name,
            'checkpoint_path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp,
            'is_best': is_best,
            'model_hash': model_hash
        }
        
        # Add to checkpoints list
        self.metadata['checkpoints'].append(checkpoint_info)
        
        # Update best checkpoint if needed
        if is_best:
            self.metadata['best_checkpoint'] = checkpoint_id
        
        self._save_metadata()
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_versions limit."""
        checkpoints = sorted(
            self.metadata['checkpoints'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        if len(checkpoints) > self.max_versions:
            # Keep best checkpoint even if it's old
            best_checkpoint_id = self.metadata.get('best_checkpoint')
            
            to_remove = []
            keep_count = 0
            
            for cp in checkpoints:
                if keep_count < self.max_versions or cp['checkpoint_id'] == best_checkpoint_id:
                    keep_count += 1
                else:
                    to_remove.append(cp)
            
            # Remove old checkpoints
            for cp in to_remove:
                checkpoint_path = Path(cp['checkpoint_path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                # Remove from metadata
                self.metadata['checkpoints'] = [
                    c for c in self.metadata['checkpoints']
                    if c['checkpoint_id'] != cp['checkpoint_id']
                ]
            
            if to_remove:
                self._save_metadata()
                log_audit_event(
                    event_type=AuditEventType.MODEL_CHECKPOINT,
                    severity=AuditSeverity.INFO,
                    message=f"Cleaned up {len(to_remove)} old checkpoints",
                    user_id=self._user_id,
                    session_id=self._session_id,
                    additional_data={'removed_count': len(to_remove)}
                )
    
    def _get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get best checkpoint info."""
        best_id = self.metadata.get('best_checkpoint')
        if best_id:
            return self._get_checkpoint_by_id(best_id)
        return None
    
    def _get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint info by ID."""
        for cp in self.metadata['checkpoints']:
            if cp['checkpoint_id'] == checkpoint_id:
                return cp
        return None
    
    def _get_checkpoint_by_name(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint info by name."""
        for cp in self.metadata['checkpoints']:
            if cp['checkpoint_name'] == checkpoint_name:
                return cp
        return None
    
    def _get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get latest checkpoint info."""
        if not self.metadata['checkpoints']:
            return None
        
        return max(
            self.metadata['checkpoints'],
            key=lambda x: x['timestamp']
        )


def create_checkpoint_manager(
    config_override: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> ModelCheckpointManager:
    """
    Factory function to create checkpoint manager from configuration.
    
    Args:
        config_override: Optional parameter overrides
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured checkpoint manager
    """
    config = get_config()
    
    # Default parameters
    params = {
        'checkpoint_dir': 'checkpoints',
        'max_versions': 10,
        'encrypt_checkpoints': True,
        'compress_checkpoints': True
    }
    
    # Apply configuration overrides
    if config_override:
        params.update(config_override)
    
    return ModelCheckpointManager(
        session_id=session_id,
        user_id=user_id,
        **params
    )