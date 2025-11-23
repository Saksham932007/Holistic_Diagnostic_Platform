"""
Advanced model architectures with attention mechanisms and transformer variants.

This module provides state-of-the-art medical AI architectures including
advanced attention mechanisms, transformer variants, and hybrid models
optimized for medical imaging tasks.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.init import trunc_normal_
import numpy as np

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.utils import ensure_tuple_rep

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class AttentionType(Enum):
    """Types of attention mechanisms."""
    STANDARD = "standard"
    MULTI_SCALE = "multi_scale"
    CROSS_ATTENTION = "cross_attention"
    SELF_ATTENTION = "self_attention"
    SPATIAL_ATTENTION = "spatial_attention"
    CHANNEL_ATTENTION = "channel_attention"
    MIXED_ATTENTION = "mixed_attention"
    ROTARY_ATTENTION = "rotary_attention"


class ArchitectureType(Enum):
    """Advanced architecture types."""
    TRANSFORMER = "transformer"
    HYBRID_CNN_TRANSFORMER = "hybrid_cnn_transformer"
    VISION_TRANSFORMER = "vision_transformer"
    SWIN_TRANSFORMER = "swin_transformer"
    UNET_TRANSFORMER = "unet_transformer"
    ATTENTION_UNET = "attention_unet"
    DUAL_ATTENTION_NETWORK = "dual_attention_network"


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding for improved spatial awareness.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 10000):
        """
        Initialize rotary positional encoding.
        
        Args:
            dim: Embedding dimension
            max_position_embeddings: Maximum position for embeddings
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # Create inverse frequency tensor
        inv_freq = 1.0 / (max_position_embeddings ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            seq_len: Sequence length
            
        Returns:
            Tensor with rotary positional encoding applied
        """
        # Generate position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)
        
        # Create rotation matrix
        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)
        
        # Apply rotation
        x_real, x_imag = x[..., 0::2], x[..., 1::2]
        
        # Rotary transformation
        x_rotated_real = x_real * cos_vals - x_imag * sin_vals
        x_rotated_imag = x_real * sin_vals + x_imag * cos_vals
        
        # Combine real and imaginary parts
        x_rotated = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
        x_rotated = x_rotated.flatten(-2)
        
        return x_rotated


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for medical images.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        scales: List[int] = [1, 2, 4],
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """
        Initialize multi-scale attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            scales: List of scale factors for multi-scale attention
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.scales = scales
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-scale QKV projections
        self.qkv_layers = nn.ModuleList([
            nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in scales
        ])
        
        # Scale-specific pooling
        self.scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d((None, None, None)) if scale == 1 
            else nn.AdaptiveAvgPool3d((dim // scale, dim // scale, dim // scale))
            for scale in scales
        ])
        
        # Attention dropouts
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Scale fusion
        self.scale_fusion = nn.Linear(dim * len(scales), dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-scale attention.
        
        Args:
            x: Input tensor of shape [B, N, C]
            
        Returns:
            Output tensor of shape [B, N, C]
        """
        B, N, C = x.shape
        
        scale_outputs = []
        
        for scale_idx, (qkv_layer, scale_pool) in enumerate(zip(self.qkv_layers, self.scale_pools)):
            # Apply scale-specific processing
            x_scaled = x
            
            # Reshape for 3D pooling if needed
            if self.scales[scale_idx] > 1:
                # Assume cubic volume for simplicity
                spatial_dim = int(N ** (1/3))
                x_spatial = x.view(B, spatial_dim, spatial_dim, spatial_dim, C)
                x_spatial = x_spatial.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]
                x_pooled = scale_pool(x_spatial)
                x_scaled = x_pooled.permute(0, 2, 3, 4, 1).flatten(1, 3)  # [B, N', C]
            
            # QKV projection
            qkv = qkv_layer(x_scaled).reshape(x_scaled.shape[0], x_scaled.shape[1], 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
            q, k, v = qkv.unbind(0)
            
            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            x_attn = (attn @ v).transpose(1, 2).reshape(x_scaled.shape[0], x_scaled.shape[1], C)
            
            # Upsample back to original size if needed
            if x_attn.shape[1] != N:
                x_attn = F.interpolate(
                    x_attn.permute(0, 2, 1).unsqueeze(-1), 
                    size=(N,), 
                    mode='linear', 
                    align_corners=False
                ).squeeze(-1).permute(0, 2, 1)
            
            scale_outputs.append(x_attn)
        
        # Fuse multi-scale features
        x_fused = torch.cat(scale_outputs, dim=-1)
        x_fused = self.scale_fusion(x_fused)
        
        # Final projection
        x_out = self.proj(x_fused)
        x_out = self.proj_drop(x_out)
        
        return x_out


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for multi-modal medical imaging.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_modalities: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            num_modalities: Number of input modalities
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Modality-specific projections
        self.q_projs = nn.ModuleList([
            nn.Linear(dim, dim, bias=qkv_bias) for _ in range(num_modalities)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(dim, dim, bias=qkv_bias) for _ in range(num_modalities)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(dim, dim, bias=qkv_bias) for _ in range(num_modalities)
        ])
        
        # Cross-modal fusion weights
        self.modal_weights = nn.Parameter(torch.ones(num_modalities, num_modalities))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of cross-modal attention.
        
        Args:
            modality_features: List of feature tensors for each modality [B, N, C]
            
        Returns:
            Fused cross-modal features [B, N, C]
        """
        B, N, C = modality_features[0].shape
        
        # Project each modality to Q, K, V
        queries = []
        keys = []
        values = []
        
        for i, features in enumerate(modality_features):
            q = self.q_projs[i](features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_projs[i](features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_projs[i](features).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            queries.append(q)
            keys.append(k)
            values.append(v)
        
        # Cross-modal attention computation
        fused_features = []
        
        for i in range(self.num_modalities):
            q_i = queries[i]  # [B, num_heads, N, head_dim]
            
            # Attend to all modalities (including self)
            cross_modal_output = []
            
            for j in range(self.num_modalities):
                k_j = keys[j]
                v_j = values[j]
                
                # Compute attention scores
                attn_scores = (q_i @ k_j.transpose(-2, -1)) * self.scale
                attn_probs = attn_scores.softmax(dim=-1)
                attn_probs = self.attn_drop(attn_probs)
                
                # Apply attention
                cross_output = (attn_probs @ v_j) * self.modal_weights[i, j]
                cross_modal_output.append(cross_output)
            
            # Sum cross-modal outputs
            fused_modality = torch.stack(cross_modal_output).sum(dim=0)
            fused_modality = fused_modality.transpose(1, 2).reshape(B, N, C)
            
            fused_features.append(fused_modality)
        
        # Final fusion
        combined_features = torch.stack(fused_features).mean(dim=0)
        
        # Final projection
        output = self.proj(combined_features)
        output = self.proj_drop(output)
        
        return output


class SpatialChannelAttention(nn.Module):
    """
    Spatial and channel attention mechanism.
    """
    
    def __init__(
        self,
        dim: int,
        spatial_kernel_size: int = 7,
        reduction_ratio: int = 16
    ):
        """
        Initialize spatial-channel attention.
        
        Args:
            dim: Input dimension
            spatial_kernel_size: Kernel size for spatial attention
            reduction_ratio: Reduction ratio for channel attention
        """
        super().__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv3d(
            2, 1, kernel_size=spatial_kernel_size, 
            padding=spatial_kernel_size // 2, bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial-channel attention.
        
        Args:
            x: Input tensor of shape [B, C, H, W, D]
            
        Returns:
            Attention-enhanced tensor
        """
        # Channel attention
        avg_pool_out = self.avg_pool(x).flatten(2)  # [B, C, 1]
        max_pool_out = self.max_pool(x).flatten(2)  # [B, C, 1]
        
        avg_channel_att = self.channel_mlp(avg_pool_out.transpose(1, 2)).transpose(1, 2)
        max_channel_att = self.channel_mlp(max_pool_out.transpose(1, 2)).transpose(1, 2)
        
        channel_att = self.sigmoid(avg_channel_att + max_channel_att).unsqueeze(-1).unsqueeze(-1)
        
        # Apply channel attention
        x_channel = x * channel_att
        
        # Spatial attention
        avg_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        
        spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_input))
        
        # Apply spatial attention
        x_spatial = x_channel * spatial_att
        
        return x_spatial


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer architecture for medical imaging.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        spatial_dims: int = 3,
        conv_block: str = "double_conv",
        res_block: bool = True
    ):
        """
        Initialize hybrid CNN-Transformer.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            feature_size: Base feature size
            hidden_size: Transformer hidden size
            mlp_dim: MLP dimension in transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate
            attention_dropout_rate: Attention dropout rate
            spatial_dims: Number of spatial dimensions
            conv_block: Convolution block type
            res_block: Whether to use residual blocks
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            16 // self.patch_size[0],
            16 // self.patch_size[1], 
            16 // self.patch_size[2]
        )
        
        # CNN feature extractor
        self.cnn_encoder = self._build_cnn_encoder(
            input_channels, feature_size, spatial_dims
        )
        
        # Patch embedding
        self.patch_embedding = nn.Conv3d(
            in_channels=feature_size * 8,
            out_channels=hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, np.prod(self.feat_size), hidden_size)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            HybridTransformerBlock(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _build_cnn_encoder(self, input_channels: int, feature_size: int, spatial_dims: int):
        """Build CNN encoder backbone."""
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d
        
        return nn.Sequential(
            # Block 1
            Conv(input_channels, feature_size, 3, padding=1),
            BatchNorm(feature_size),
            nn.ReLU(inplace=True),
            Conv(feature_size, feature_size, 3, padding=1),
            BatchNorm(feature_size),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2),
            
            # Block 2
            Conv(feature_size, feature_size * 2, 3, padding=1),
            BatchNorm(feature_size * 2),
            nn.ReLU(inplace=True),
            Conv(feature_size * 2, feature_size * 2, 3, padding=1),
            BatchNorm(feature_size * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2),
            
            # Block 3
            Conv(feature_size * 2, feature_size * 4, 3, padding=1),
            BatchNorm(feature_size * 4),
            nn.ReLU(inplace=True),
            Conv(feature_size * 4, feature_size * 4, 3, padding=1),
            BatchNorm(feature_size * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2),
            
            # Block 4
            Conv(feature_size * 4, feature_size * 8, 3, padding=1),
            BatchNorm(feature_size * 8),
            nn.ReLU(inplace=True),
            Conv(feature_size * 8, feature_size * 8, 3, padding=1),
            BatchNorm(feature_size * 8),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of hybrid CNN-Transformer.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            
        Returns:
            Classification logits [B, num_classes]
        """
        # CNN feature extraction
        cnn_features = self.cnn_encoder(x)  # [B, C', H', W', D']
        
        # Patch embedding
        patches = self.patch_embedding(cnn_features)  # [B, hidden_size, H'', W'', D'']
        
        # Flatten patches
        B, C, H, W, D = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]
        
        # Add positional encoding
        patches = patches + self.pos_encoding
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            patches = block(patches)
        
        # Layer normalization
        patches = self.layer_norm(patches)
        
        # Global average pooling
        features = patches.mean(dim=1)  # [B, hidden_size]
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def _init_weights(self):
        """Initialize model weights."""
        def _init_fn(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
        self.apply(_init_fn)


class HybridTransformerBlock(nn.Module):
    """
    Hybrid transformer block with advanced attention mechanisms.
    """
    
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1
    ):
        """
        Initialize hybrid transformer block.
        
        Args:
            hidden_size: Hidden size
            mlp_dim: MLP dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
            attention_dropout_rate: Attention dropout rate
        """
        super().__init__()
        
        # Multi-scale attention
        self.multi_scale_attention = MultiScaleAttention(
            dim=hidden_size,
            num_heads=num_heads,
            attn_drop=attention_dropout_rate,
            proj_drop=dropout_rate
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate)
        )
        
        # Skip connection scaling
        self.skip_scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Multi-scale attention with residual connection
        attn_out = self.multi_scale_attention(self.layer_norm1(x))
        x = x + self.skip_scale * attn_out
        
        # MLP with residual connection
        mlp_out = self.mlp(self.layer_norm2(x))
        x = x + self.skip_scale * mlp_out
        
        return x


class AttentionUNet(nn.Module):
    """
    U-Net with attention gates for medical image segmentation.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 2,
        feature_size: int = 32,
        spatial_dims: int = 3,
        use_checkpoint: bool = False
    ):
        """
        Initialize Attention U-Net.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            feature_size: Base feature size
            spatial_dims: Number of spatial dimensions
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.use_checkpoint = use_checkpoint
        
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        ConvTranspose = nn.ConvTranspose3d if spatial_dims == 3 else nn.ConvTranspose2d
        BatchNorm = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d
        
        # Encoder
        self.encoder1 = self._conv_block(input_channels, feature_size)
        self.pool1 = nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2)
        
        self.encoder2 = self._conv_block(feature_size, feature_size * 2)
        self.pool2 = nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2)
        
        self.encoder3 = self._conv_block(feature_size * 2, feature_size * 4)
        self.pool3 = nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2)
        
        self.encoder4 = self._conv_block(feature_size * 4, feature_size * 8)
        self.pool4 = nn.MaxPool3d(2) if spatial_dims == 3 else nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(feature_size * 8, feature_size * 16)
        
        # Attention gates
        self.attention4 = AttentionGate(feature_size * 16, feature_size * 8, feature_size * 8, spatial_dims)
        self.attention3 = AttentionGate(feature_size * 8, feature_size * 4, feature_size * 4, spatial_dims)
        self.attention2 = AttentionGate(feature_size * 4, feature_size * 2, feature_size * 2, spatial_dims)
        self.attention1 = AttentionGate(feature_size * 2, feature_size, feature_size, spatial_dims)
        
        # Decoder
        self.upconv4 = ConvTranspose(feature_size * 16, feature_size * 8, 2, 2)
        self.decoder4 = self._conv_block(feature_size * 16, feature_size * 8)
        
        self.upconv3 = ConvTranspose(feature_size * 8, feature_size * 4, 2, 2)
        self.decoder3 = self._conv_block(feature_size * 8, feature_size * 4)
        
        self.upconv2 = ConvTranspose(feature_size * 4, feature_size * 2, 2, 2)
        self.decoder2 = self._conv_block(feature_size * 4, feature_size * 2)
        
        self.upconv1 = ConvTranspose(feature_size * 2, feature_size, 2, 2)
        self.decoder1 = self._conv_block(feature_size * 2, feature_size)
        
        # Output layer
        self.output = Conv(feature_size, output_channels, 1)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create convolution block."""
        Conv = nn.Conv3d if self.spatial_dims == 3 else nn.Conv2d
        BatchNorm = nn.BatchNorm3d if self.spatial_dims == 3 else nn.BatchNorm2d
        
        return nn.Sequential(
            Conv(in_channels, out_channels, 3, padding=1),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, 3, padding=1),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Attention U-Net."""
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with attention gates
        up4 = self.upconv4(bottleneck)
        att4 = self.attention4(up4, enc4)
        dec4 = self.decoder4(torch.cat([up4, att4], dim=1))
        
        up3 = self.upconv3(dec4)
        att3 = self.attention3(up3, enc3)
        dec3 = self.decoder3(torch.cat([up3, att3], dim=1))
        
        up2 = self.upconv2(dec3)
        att2 = self.attention2(up2, enc2)
        dec2 = self.decoder2(torch.cat([up2, att2], dim=1))
        
        up1 = self.upconv1(dec2)
        att1 = self.attention1(up1, enc1)
        dec1 = self.decoder1(torch.cat([up1, att1], dim=1))
        
        # Output
        output = self.output(dec1)
        
        return output


class AttentionGate(nn.Module):
    """
    Attention gate for U-Net architectures.
    """
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: int,
        spatial_dims: int
    ):
        """
        Initialize attention gate.
        
        Args:
            gate_channels: Number of gate channels
            skip_channels: Number of skip connection channels
            inter_channels: Number of intermediate channels
            spatial_dims: Number of spatial dimensions
        """
        super().__init__()
        
        Conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        
        self.gate_conv = Conv(gate_channels, inter_channels, 1)
        self.skip_conv = Conv(skip_channels, inter_channels, 1)
        self.psi = Conv(inter_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention gate.
        
        Args:
            gate: Gate input tensor
            skip: Skip connection tensor
            
        Returns:
            Attention-weighted skip connection
        """
        gate_proj = self.gate_conv(gate)
        skip_proj = self.skip_conv(skip)
        
        # Element-wise addition
        combined = self.relu(gate_proj + skip_proj)
        
        # Attention coefficients
        attention = self.sigmoid(self.psi(combined))
        
        # Apply attention to skip connection
        attended_skip = skip * attention
        
        return attended_skip


def create_advanced_architecture(
    architecture_type: ArchitectureType,
    input_channels: int = 1,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create advanced architectures.
    
    Args:
        architecture_type: Type of architecture to create
        input_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional architecture-specific arguments
        
    Returns:
        Instantiated model
    """
    if architecture_type == ArchitectureType.HYBRID_CNN_TRANSFORMER:
        return HybridCNNTransformer(
            input_channels=input_channels,
            num_classes=num_classes,
            **kwargs
        )
    elif architecture_type == ArchitectureType.ATTENTION_UNET:
        return AttentionUNet(
            input_channels=input_channels,
            output_channels=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported architecture type: {architecture_type}")


def get_model_complexity(model: nn.Module) -> Dict[str, Any]:
    """
    Calculate model complexity metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with complexity metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate FLOPs (simplified estimation)
    def count_flops(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d)):
            return np.prod(module.weight.shape) * 2  # Multiply-accumulate operations
        elif isinstance(module, nn.Linear):
            return module.weight.numel() * 2
        return 0
    
    total_flops = sum(count_flops(module) for module in model.modules())
    
    # Model size in MB
    param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': param_size_mb,
        'estimated_flops': total_flops,
        'memory_efficient': total_params < 50e6,  # Less than 50M parameters
        'complexity_level': 'low' if total_params < 10e6 else 'medium' if total_params < 50e6 else 'high'
    }