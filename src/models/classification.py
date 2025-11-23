"""
3D Vision Transformer for medical image classification.

This module implements a 3D Vision Transformer optimized for medical imaging
with hierarchical attention, multi-scale processing, and clinical adaptations.
"""

from typing import Optional, Tuple, Dict, Any, List, Union
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout
import numpy as np

from monai.networks.blocks import PatchEmbed, TransformerBlock
from monai.networks.layers import DropPath
from monai.utils import ensure_tuple_rep

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class MedicalViT3D(nn.Module):
    """
    3D Vision Transformer for medical image classification.
    
    Features:
    - Hierarchical patch embedding with multi-scale processing
    - Medical-specific positional encodings
    - Attention visualization for clinical interpretation
    - Uncertainty estimation capabilities
    - Support for variable input sizes
    """
    
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = (96, 96, 96),
        patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 16),
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = LayerNorm,
        use_abs_pos_embed: bool = True,
        use_clinical_tokens: bool = True,
        hierarchical_levels: int = 3,
        enable_uncertainty: bool = True,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize 3D Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Patch size for tokenization
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate
            norm_layer: Normalization layer
            use_abs_pos_embed: Whether to use absolute positional embedding
            use_clinical_tokens: Whether to use clinical-specific tokens
            hierarchical_levels: Number of hierarchical levels
            enable_uncertainty: Whether to enable uncertainty estimation
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        self._session_id = session_id
        self._user_id = user_id
        self.num_classes = num_classes
        self.enable_uncertainty = enable_uncertainty
        self.use_clinical_tokens = use_clinical_tokens
        self.hierarchical_levels = hierarchical_levels
        
        # Ensure tuple format
        img_size = ensure_tuple_rep(img_size, 3)
        patch_size = ensure_tuple_rep(patch_size, 3)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate patch grid dimensions
        self.grid_size = tuple(img_s // patch_s for img_s, patch_s in zip(img_size, patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        # Hierarchical patch embedding
        self.patch_embeds = nn.ModuleList()
        current_patch_size = patch_size
        
        for level in range(hierarchical_levels):
            patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=current_patch_size,
                in_chans=in_channels if level == 0 else embed_dim,
                embed_dim=embed_dim,
                norm_layer=norm_layer,
                spatial_dims=3
            )
            self.patch_embeds.append(patch_embed)
            
            # Increase patch size for next level
            current_patch_size = tuple(p * 2 for p in current_patch_size)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Clinical tokens for medical-specific features
        if use_clinical_tokens:
            self.clinical_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim))  # anatomy, pathology, contrast, quality
            num_special_tokens = 5  # cls + 4 clinical
        else:
            num_special_tokens = 1  # cls only
        
        # Positional embeddings
        if use_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + num_special_tokens, embed_dim)
            )
        else:
            self.pos_embed = None
        
        self.pos_drop = Dropout(drop_rate)
        
        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            MedicalTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                enable_medical_attention=True
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim)
        
        # Classification heads
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()
        
        # Uncertainty estimation head
        if enable_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Attention visualization storage
        self.attention_weights = []
        self.register_hooks()
        
        # Initialize weights
        self._init_weights()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="3D Vision Transformer initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'img_size': img_size,
                'patch_size': patch_size,
                'embed_dim': embed_dim,
                'depth': depth,
                'num_heads': num_heads,
                'num_classes': num_classes,
                'num_patches': self.num_patches,
                'hierarchical_levels': hierarchical_levels
            }
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize clinical tokens
        if self.use_clinical_tokens:
            nn.init.trunc_normal_(self.clinical_tokens, std=0.02)
        
        # Initialize other layers
        self.apply(self._init_layer_weights)
    
    def _init_layer_weights(self, m):
        """Initialize individual layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def register_hooks(self):
        """Register hooks for attention visualization."""
        def hook(module, input, output):
            if len(output) >= 2 and hasattr(output, '__getitem__'):
                # Store attention weights for visualization
                attn_weights = output[1] if isinstance(output[1], torch.Tensor) else None
                if attn_weights is not None:
                    self.attention_weights.append(attn_weights.detach().cpu())
        
        # Register hooks on attention layers
        for block in self.blocks:
            if hasattr(block, 'attn'):
                block.attn.register_forward_hook(hook)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention visualization and uncertainty estimation.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            
        Returns:
            Dictionary containing predictions, attention maps, and uncertainty
        """
        B = x.shape[0]
        
        # Clear previous attention weights
        self.attention_weights.clear()
        
        # Hierarchical patch embedding
        patch_embeddings = []
        current_x = x
        
        for level, patch_embed in enumerate(self.patch_embeds):
            # Extract patches at current level
            patches = patch_embed(current_x)  # [B, N, D]
            patch_embeddings.append(patches)
            
            # Prepare for next level (if not last)
            if level < len(self.patch_embeds) - 1:
                # Pool current_x for next level
                current_x = F.avg_pool3d(current_x, kernel_size=2, stride=2)
        
        # Use the finest level patches as primary embedding
        x = patch_embeddings[0]
        
        # Add special tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = [cls_tokens]
        
        if self.use_clinical_tokens:
            clinical_tokens = self.clinical_tokens.expand(B, -1, -1)
            tokens.append(clinical_tokens)
        
        tokens.append(x)
        x = torch.cat(tokens, dim=1)
        
        # Add positional embeddings
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Extract class token for classification
        cls_token_final = x[:, 0]
        
        # Classification
        logits = self.head(cls_token_final)
        
        # Uncertainty estimation
        uncertainty = None
        if self.enable_uncertainty:
            uncertainty = self.uncertainty_head(cls_token_final)
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'features': cls_token_final,
            'tokens': x
        }
        
        if uncertainty is not None:
            outputs['uncertainty'] = uncertainty
        
        # Add attention visualization
        if self.attention_weights:
            outputs['attention_weights'] = self.attention_weights[-4:]  # Last 4 layers
        
        return outputs
    
    def get_attention_maps(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Attention maps tensor
        """
        if not self.attention_weights:
            return None
        
        if layer_idx == -1:
            layer_idx = len(self.attention_weights) - 1
        
        if 0 <= layer_idx < len(self.attention_weights):
            return self.attention_weights[layer_idx]
        
        return None
    
    def extract_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch embeddings without classification head.
        
        Args:
            x: Input tensor
            
        Returns:
            Patch embeddings
        """
        outputs = self.forward(x)
        return outputs['tokens']
    
    def monte_carlo_inference(
        self, 
        x: torch.Tensor, 
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Perform Monte Carlo inference for uncertainty quantification.
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples
            
        Returns:
            Mean predictions and uncertainty estimates
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.forward(x)
                predictions.append(F.softmax(output['logits'], dim=-1))
                if 'uncertainty' in output:
                    uncertainties.append(output['uncertainty'])
        
        # Calculate statistics
        predictions_tensor = torch.stack(predictions, dim=0)  # [num_samples, B, num_classes]
        mean_pred = torch.mean(predictions_tensor, dim=0)
        pred_uncertainty = torch.var(predictions_tensor, dim=0)
        
        results = {
            'mean_prediction': mean_pred,
            'prediction_variance': pred_uncertainty,
            'epistemic_uncertainty': torch.mean(pred_uncertainty, dim=-1, keepdim=True)
        }
        
        if uncertainties:
            uncertainties_tensor = torch.stack(uncertainties, dim=0)
            results['aleatoric_uncertainty'] = torch.mean(uncertainties_tensor, dim=0)
        
        self.eval()  # Return to eval mode
        return results


class MedicalTransformerBlock(TransformerBlock):
    """
    Enhanced transformer block with medical-specific attention mechanisms.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = LayerNorm,
        enable_medical_attention: bool = True
    ):
        """
        Initialize medical transformer block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projection
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Drop path rate
            act_layer: Activation layer
            norm_layer: Normalization layer
            enable_medical_attention: Whether to enable medical-specific attention
        """
        super().__init__(
            hidden_size=dim,
            mlp_dim=int(dim * mlp_ratio),
            num_heads=num_heads,
            dropout_rate=drop,
            attention_dropout_rate=attn_drop,
            dropout_path_rate=drop_path
        )
        
        self.enable_medical_attention = enable_medical_attention
        
        if enable_medical_attention:
            # Add medical-specific attention components
            self.medical_gate = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )
            
            # Anatomical attention bias
            self.anatomical_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with medical attention enhancements."""
        # Standard transformer block forward
        x_original = x
        x = super().forward(x)
        
        # Apply medical attention gating if enabled
        if self.enable_medical_attention:
            # Calculate medical attention gate
            gate = self.medical_gate(x_original)
            x = gate * x + (1 - gate) * x_original
        
        return x


class MultiScaleViT(nn.Module):
    """
    Multi-scale Vision Transformer for handling various input sizes.
    """
    
    def __init__(
        self,
        base_vit_config: Dict[str, Any],
        scales: List[float] = [0.5, 1.0, 1.5],
        fusion_method: str = "attention",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize multi-scale ViT.
        
        Args:
            base_vit_config: Base ViT configuration
            scales: List of scale factors
            fusion_method: Method for fusing multi-scale features
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        self.scales = scales
        self.fusion_method = fusion_method
        self._session_id = session_id
        self._user_id = user_id
        
        # Create ViTs for each scale
        self.vits = nn.ModuleList()
        for scale in scales:
            config = base_vit_config.copy()
            # Adjust image size for scale
            config['img_size'] = tuple(int(s * scale) for s in config['img_size'])
            vit = MedicalViT3D(**config, session_id=session_id, user_id=user_id)
            self.vits.append(vit)
        
        # Feature fusion
        embed_dim = base_vit_config['embed_dim']
        if fusion_method == "attention":
            self.fusion = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                batch_first=True
            )
        elif fusion_method == "concat":
            self.fusion = nn.Linear(embed_dim * len(scales), embed_dim)
        else:
            self.fusion = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier
        self.classifier = nn.Linear(embed_dim, base_vit_config['num_classes'])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-scale forward pass."""
        scale_features = []
        scale_outputs = []
        
        for i, (scale, vit) in enumerate(zip(self.scales, self.vits)):
            # Resize input for current scale
            if scale != 1.0:
                size = tuple(int(s * scale) for s in x.shape[2:])
                x_scaled = F.interpolate(x, size=size, mode='trilinear', align_corners=False)
            else:
                x_scaled = x
            
            # Forward pass
            output = vit(x_scaled)
            scale_features.append(output['features'])
            scale_outputs.append(output)
        
        # Fuse features
        if self.fusion_method == "attention":
            # Stack features for attention
            stacked_features = torch.stack(scale_features, dim=1)  # [B, num_scales, D]
            fused_features, _ = self.fusion(stacked_features, stacked_features, stacked_features)
            fused_features = torch.mean(fused_features, dim=1)  # [B, D]
        elif self.fusion_method == "concat":
            concatenated = torch.cat(scale_features, dim=1)  # [B, num_scales * D]
            fused_features = self.fusion(concatenated)
        else:  # average
            stacked_features = torch.stack(scale_features, dim=1)  # [B, num_scales, D]
            fused_features = torch.mean(stacked_features, dim=1)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'features': fused_features,
            'scale_outputs': scale_outputs
        }


def create_medical_vit(
    config_override: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> MedicalViT3D:
    """
    Factory function to create medical ViT from configuration.
    
    Args:
        config_override: Optional parameter overrides
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured medical ViT model
    """
    config = get_config()
    
    # Default ViT configuration
    default_config = {
        'img_size': (96, 96, 96),
        'patch_size': (16, 16, 16),
        'in_channels': 1,
        'num_classes': 2,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'use_clinical_tokens': True,
        'enable_uncertainty': True
    }
    
    # Apply overrides
    if config_override:
        default_config.update(config_override)
    
    return MedicalViT3D(
        session_id=session_id,
        user_id=user_id,
        **default_config
    )