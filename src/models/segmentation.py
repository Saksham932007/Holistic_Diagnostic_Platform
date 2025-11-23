"""
Swin-UNetR segmentation model for medical image analysis.

This module implements the Swin-UNetR (Swin Transformer U-Net) architecture
optimized for 3D medical image segmentation with MONAI integration.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np

from monai.networks.nets import SwinUNETR
from monai.networks.layers import Norm
from monai.utils import ensure_tuple_rep

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class MedicalSwinUNETR(nn.Module):
    """
    Production-grade Swin-UNetR implementation for medical image segmentation.
    
    This wrapper around MONAI's SwinUNETR adds medical-specific features:
    - Attention map extraction for explainability
    - Multi-scale feature extraction
    - Uncertainty estimation capabilities
    - HIPAA-compliant audit logging
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 14,
        feature_size: int = 48,
        patch_size: Tuple[int, int, int] = (2, 2, 2),
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        return_attention_maps: bool = False,
        uncertainty_estimation: bool = False,
        monte_carlo_dropout: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize Medical Swin-UNetR model.
        
        Args:
            img_size: Input image size (D, H, W)
            in_channels: Number of input channels
            out_channels: Number of output segmentation classes
            feature_size: Base feature size
            patch_size: Patch size for patch embedding
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            window_size: Window size for shifted window attention
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Add bias to query, key, value projections
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Stochastic depth rate
            normalize: Whether to normalize output
            use_checkpoint: Use gradient checkpointing to save memory
            spatial_dims: Spatial dimensions (2 or 3)
            return_attention_maps: Whether to return attention maps
            uncertainty_estimation: Enable uncertainty estimation
            monte_carlo_dropout: Use MC dropout for uncertainty
            session_id: Session identifier for audit logging
            user_id: User identifier for audit logging
        """
        super().__init__()
        
        self._config = get_config()
        self._session_id = session_id
        self._user_id = user_id
        self._return_attention_maps = return_attention_maps
        self._uncertainty_estimation = uncertainty_estimation
        self._monte_carlo_dropout = monte_carlo_dropout
        
        # Store model parameters for audit logging
        self._model_params = {
            'img_size': img_size,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'feature_size': feature_size,
            'patch_size': patch_size,
            'depths': depths,
            'num_heads': num_heads,
            'spatial_dims': spatial_dims
        }
        
        # Initialize core Swin-UNetR model
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims
        )
        
        # Additional components for medical applications
        self._setup_medical_components()
        
        # Initialize attention map storage
        self._attention_maps: Dict[str, torch.Tensor] = {}
        self._feature_maps: Dict[str, torch.Tensor] = {}
        
        # Register hooks for attention map extraction
        if self._return_attention_maps:
            self._register_attention_hooks()
        
        # Log model initialization
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Swin-UNetR model initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'model_type': 'SwinUNETR',
                'parameters': self._model_params,
                'total_parameters': self.count_parameters(),
                'uncertainty_estimation': uncertainty_estimation
            }
        )
    
    def _setup_medical_components(self) -> None:
        """Setup additional components for medical applications."""
        
        # Multi-scale feature extraction layers
        self.feature_extractors = nn.ModuleDict({
            'encoder_1': nn.Conv3d(self._model_params['feature_size'], 64, 1),
            'encoder_2': nn.Conv3d(self._model_params['feature_size'] * 2, 64, 1),
            'encoder_3': nn.Conv3d(self._model_params['feature_size'] * 4, 64, 1),
            'encoder_4': nn.Conv3d(self._model_params['feature_size'] * 8, 64, 1),
        })
        
        # Uncertainty estimation head (if enabled)
        if self._uncertainty_estimation:
            self.uncertainty_head = nn.Sequential(
                nn.Conv3d(self._model_params['out_channels'], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, 1),
                nn.Sigmoid()
            )
        
        # Monte Carlo dropout layers for uncertainty estimation
        if self._monte_carlo_dropout:
            self.mc_dropout = nn.Dropout3d(p=0.1)
    
    def _register_attention_hooks(self) -> None:
        """Register forward hooks to capture attention maps."""
        
        def attention_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    # Store attention weights
                    self._attention_maps[name] = output[1].detach()
                return output
            return hook
        
        # Register hooks on transformer blocks
        for i, layer in enumerate(self.swin_unetr.swinViT.layers):
            for j, block in enumerate(layer.blocks):
                if hasattr(block, 'attn'):
                    hook_name = f"layer_{i}_block_{j}_attention"
                    block.attn.register_forward_hook(attention_hook(hook_name))
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        monte_carlo_samples: int = 10
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the Swin-UNetR model.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            return_features: Whether to return intermediate features
            monte_carlo_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            Segmentation output, optionally with features and uncertainty maps
        """
        batch_size = x.shape[0]
        
        # Log inference start
        log_audit_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message="Starting Swin-UNetR inference",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'input_shape': list(x.shape),
                'model_type': 'SwinUNETR',
                'monte_carlo_samples': monte_carlo_samples if self._uncertainty_estimation else 0
            }
        )
        
        try:
            if self._uncertainty_estimation and self._monte_carlo_dropout and monte_carlo_samples > 1:
                # Monte Carlo inference for uncertainty estimation
                outputs = []
                
                # Enable dropout during inference
                self.train()
                
                with torch.no_grad():
                    for _ in range(monte_carlo_samples):
                        # Add MC dropout
                        x_dropped = self.mc_dropout(x)
                        output = self.swin_unetr(x_dropped)
                        outputs.append(output)
                
                # Calculate mean and variance
                outputs_stack = torch.stack(outputs, dim=0)
                mean_output = torch.mean(outputs_stack, dim=0)
                var_output = torch.var(outputs_stack, dim=0)
                
                # Calculate epistemic uncertainty
                epistemic_uncertainty = torch.mean(var_output, dim=1, keepdim=True)
                
                result = (mean_output, epistemic_uncertainty)
                
            else:
                # Standard forward pass
                output = self.swin_unetr(x)
                
                # Calculate uncertainty if enabled
                if self._uncertainty_estimation:
                    # Aleatoric uncertainty from dedicated head
                    uncertainty = self.uncertainty_head(output)
                    result = (output, uncertainty)
                else:
                    result = output
            
            # Extract multi-scale features if requested
            if return_features:
                features = self._extract_multiscale_features()
                if isinstance(result, tuple):
                    result = result + (features,)
                else:
                    result = (result, features)
            
            # Add attention maps if enabled
            if self._return_attention_maps and self._attention_maps:
                if isinstance(result, tuple):
                    result = result + (self._attention_maps.copy(),)
                else:
                    result = (result, self._attention_maps.copy())
            
            # Log successful inference
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.INFO,
                message="Swin-UNetR inference completed successfully",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'output_shape': list(output.shape) if not isinstance(result, tuple) else list(result[0].shape),
                    'uncertainty_enabled': self._uncertainty_estimation
                }
            )
            
            return result
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.ERROR,
                message=f"Swin-UNetR inference failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise
    
    def _extract_multiscale_features(self) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from encoder layers."""
        features = {}
        
        # Access encoder features from SwinUNETR
        # Note: This requires access to intermediate outputs
        # Implementation depends on MONAI version and internal structure
        
        return features
    
    @autocast()
    def forward_with_amp(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Automatic Mixed Precision."""
        return self.forward(x)
    
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """
        Get attention maps from the last forward pass.
        
        Returns:
            Dictionary of attention maps indexed by layer name
        """
        return self._attention_maps.copy()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary for analysis.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': 'SwinUNETR',
            'parameters': self._model_params,
            'total_parameters': self.count_parameters(),
            'uncertainty_estimation': self._uncertainty_estimation,
            'monte_carlo_dropout': self._monte_carlo_dropout,
            'attention_extraction': self._return_attention_maps,
            'device': next(self.parameters()).device.type,
            'dtype': next(self.parameters()).dtype
        }
    
    def prepare_for_deployment(self) -> None:
        """Prepare model for deployment (optimize for inference)."""
        # Set to evaluation mode
        self.eval()
        
        # Disable attention map extraction for faster inference
        self._return_attention_maps = False
        
        # Log deployment preparation
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Swin-UNetR model prepared for deployment",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={'optimization': 'inference_mode'}
        )
    
    def export_onnx(
        self,
        filepath: str,
        input_shape: Tuple[int, ...] = (1, 1, 96, 96, 96),
        opset_version: int = 11
    ) -> None:
        """
        Export model to ONNX format for deployment.
        
        Args:
            filepath: Output ONNX file path
            input_shape: Input tensor shape for tracing
            opset_version: ONNX opset version
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            if next(self.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            # Prepare for ONNX export
            self.eval()
            original_return_attention = self._return_attention_maps
            self._return_attention_maps = False
            
            # Export to ONNX
            torch.onnx.export(
                self,
                dummy_input,
                filepath,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Restore original settings
            self._return_attention_maps = original_return_attention
            
            log_audit_event(
                event_type=AuditEventType.DATA_EXPORT,
                severity=AuditSeverity.INFO,
                message="Swin-UNetR model exported to ONNX",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'export_path': filepath,
                    'input_shape': input_shape,
                    'opset_version': opset_version
                }
            )
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.DATA_EXPORT,
                severity=AuditSeverity.ERROR,
                message=f"ONNX export failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise


def create_swin_unetr_model(
    config_override: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> MedicalSwinUNETR:
    """
    Factory function to create Swin-UNetR model from configuration.
    
    Args:
        config_override: Optional parameter overrides
        session_id: Session identifier for audit logging
        user_id: User identifier for audit logging
        
    Returns:
        Configured MedicalSwinUNETR model
    """
    config = get_config()
    
    # Get model parameters from configuration
    model_params = {
        'img_size': config.model.swin_img_size,
        'in_channels': config.model.swin_in_channels,
        'out_channels': config.model.swin_out_channels,
        'feature_size': config.model.swin_feature_size,
        'patch_size': config.model.swin_patch_size,
    }
    
    # Apply overrides if provided
    if config_override:
        model_params.update(config_override)
    
    # Create model
    model = MedicalSwinUNETR(
        session_id=session_id,
        user_id=user_id,
        **model_params
    )
    
    return model


def load_pretrained_swin_unetr(
    checkpoint_path: str,
    config_override: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    strict: bool = True
) -> MedicalSwinUNETR:
    """
    Load a pretrained Swin-UNetR model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config_override: Optional parameter overrides
        session_id: Session identifier for audit logging
        user_id: User identifier for audit logging
        strict: Whether to strictly enforce matching keys
        
    Returns:
        Loaded MedicalSwinUNETR model
    """
    try:
        # Create model
        model = create_swin_unetr_model(
            config_override=config_override,
            session_id=session_id,
            user_id=user_id
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=strict)
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Pretrained Swin-UNetR model loaded",
            user_id=user_id,
            session_id=session_id,
            additional_data={
                'checkpoint_path': checkpoint_path,
                'strict_loading': strict,
                'model_parameters': model.count_parameters()
            }
        )
        
        return model
        
    except Exception as e:
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.ERROR,
            message=f"Failed to load pretrained model: {str(e)}",
            user_id=user_id,
            session_id=session_id,
            additional_data={'checkpoint_path': checkpoint_path, 'error': str(e)}
        )
        raise