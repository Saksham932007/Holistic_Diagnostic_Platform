"""
Multi-modal fusion architecture for medical imaging.

This module implements advanced fusion techniques for combining multiple
imaging modalities (CT, MRI, PET) with cross-attention mechanisms and
clinical context integration.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, MultiheadAttention
import numpy as np

from monai.networks.blocks import TransformerBlock
from monai.networks.layers import DropPath

from .segmentation import MedicalSwinUNETR
from .classification import MedicalViT3D
from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing different imaging modalities.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Temperature scaling for attention
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Multi-head cross attention
        self.cross_attention = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Modality-specific gates
        self.modality_gate = nn.Sequential(
            Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        query_modality: torch.Tensor,
        key_value_modality: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform cross-modal attention.
        
        Args:
            query_modality: Query modality features [B, N, D]
            key_value_modality: Key/Value modality features [B, M, D]
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Cross attention
        attended_features, attention_weights = self.cross_attention(
            query=query_modality,
            key=key_value_modality,
            value=key_value_modality,
            attn_mask=mask
        )
        
        # Residual connection and normalization
        attended_features = self.norm1(query_modality + attended_features)
        
        # Apply modality gating
        gate_input = torch.cat([query_modality, attended_features], dim=-1)
        gate = self.modality_gate(gate_input)
        attended_features = gate * attended_features + (1 - gate) * query_modality
        
        # Feed-forward network
        ffn_output = self.ffn(attended_features)
        attended_features = self.norm2(attended_features + ffn_output)
        
        return attended_features, attention_weights


class ModalityEncoder(nn.Module):
    """
    Modality-specific encoder with shared and unique feature extraction.
    """
    
    def __init__(
        self,
        modality: str,
        input_channels: int,
        embed_dim: int = 768,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        img_size: Tuple[int, int, int] = (96, 96, 96),
        depth: int = 6,
        num_heads: int = 12
    ):
        """
        Initialize modality encoder.
        
        Args:
            modality: Modality name (CT, MRI, PET, etc.)
            input_channels: Number of input channels
            embed_dim: Embedding dimension
            patch_size: Patch size for tokenization
            img_size: Input image size
            depth: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.modality = modality
        self.embed_dim = embed_dim
        
        # Modality-specific preprocessing
        self.preprocess = self._create_preprocessing_layers(modality, input_channels)
        
        # Patch embedding
        from monai.networks.blocks import PatchEmbed
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=input_channels,
            embed_dim=embed_dim,
            spatial_dims=3
        )
        
        # Modality token to identify the modality
        self.modality_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Position embeddings
        num_patches = np.prod([s // p for s, p in zip(img_size, patch_size)])
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=embed_dim,
                mlp_dim=embed_dim * 4,
                num_heads=num_heads,
                dropout_rate=0.1,
                attention_dropout_rate=0.0
            )
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = LayerNorm(embed_dim)
    
    def _create_preprocessing_layers(self, modality: str, input_channels: int) -> nn.Module:
        """Create modality-specific preprocessing layers."""
        if modality.upper() == "CT":
            # CT-specific preprocessing (HU windowing, etc.)
            return nn.Sequential(
                nn.Conv3d(input_channels, input_channels, 3, padding=1),
                nn.BatchNorm3d(input_channels),
                nn.ReLU(),
                nn.Conv3d(input_channels, input_channels, 3, padding=1),
                nn.BatchNorm3d(input_channels)
            )
        elif modality.upper() == "MRI":
            # MRI-specific preprocessing
            return nn.Sequential(
                nn.Conv3d(input_channels, input_channels, 3, padding=1),
                nn.InstanceNorm3d(input_channels),
                nn.ReLU(),
                nn.Conv3d(input_channels, input_channels, 3, padding=1),
                nn.InstanceNorm3d(input_channels)
            )
        elif modality.upper() == "PET":
            # PET-specific preprocessing
            return nn.Sequential(
                nn.Conv3d(input_channels, input_channels, 5, padding=2),
                nn.BatchNorm3d(input_channels),
                nn.ReLU(),
                nn.Conv3d(input_channels, input_channels, 3, padding=1),
                nn.BatchNorm3d(input_channels)
            )
        else:
            # Generic preprocessing
            return nn.Sequential(
                nn.Conv3d(input_channels, input_channels, 3, padding=1),
                nn.BatchNorm3d(input_channels),
                nn.ReLU()
            )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through modality encoder.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            
        Returns:
            Tuple of (encoded_features, patch_tokens)
        """
        B = x.shape[0]
        
        # Modality-specific preprocessing
        x = self.preprocess(x)
        
        # Patch embedding
        patch_tokens = self.patch_embed(x)  # [B, N, D]
        
        # Add modality token
        modality_tokens = self.modality_token.expand(B, -1, -1)
        tokens = torch.cat([modality_tokens, patch_tokens], dim=1)
        
        # Add positional embeddings
        tokens = tokens + self.pos_embed
        
        # Transform
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        tokens = self.norm(tokens)
        
        # Separate modality token and patch tokens
        modality_feature = tokens[:, 0]  # [B, D]
        patch_tokens = tokens[:, 1:]     # [B, N, D]
        
        return modality_feature, patch_tokens


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion network with hierarchical attention and clinical context.
    """
    
    def __init__(
        self,
        modalities: List[str],
        input_channels_per_modality: Dict[str, int],
        num_classes: int,
        embed_dim: int = 768,
        fusion_layers: int = 4,
        num_heads: int = 8,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        use_clinical_context: bool = True,
        fusion_strategy: str = "hierarchical",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize multi-modal fusion network.
        
        Args:
            modalities: List of modality names
            input_channels_per_modality: Input channels for each modality
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            fusion_layers: Number of fusion layers
            num_heads: Number of attention heads
            img_size: Input image size
            patch_size: Patch size
            use_clinical_context: Whether to use clinical context
            fusion_strategy: Fusion strategy ('hierarchical', 'parallel', 'sequential')
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.embed_dim = embed_dim
        self.fusion_strategy = fusion_strategy
        self.use_clinical_context = use_clinical_context
        self._session_id = session_id
        self._user_id = user_id
        
        # Modality encoders
        self.modality_encoders = nn.ModuleDict({
            modality: ModalityEncoder(
                modality=modality,
                input_channels=input_channels_per_modality[modality],
                embed_dim=embed_dim,
                patch_size=patch_size,
                img_size=img_size
            )
            for modality in modalities
        })
        
        # Cross-modal attention layers
        if fusion_strategy == "hierarchical":
            self.cross_attention_layers = nn.ModuleList([
                CrossModalAttention(embed_dim, num_heads)
                for _ in range(fusion_layers)
            ])
        elif fusion_strategy == "parallel":
            self.cross_attention_pairs = nn.ModuleDict({
                f"{mod1}_{mod2}": CrossModalAttention(embed_dim, num_heads)
                for i, mod1 in enumerate(modalities)
                for mod2 in modalities[i+1:]
            })
        
        # Clinical context encoder
        if use_clinical_context:
            self.clinical_encoder = ClinicalContextEncoder(embed_dim)
        
        # Global fusion transformer
        self.global_fusion = nn.ModuleList([
            TransformerBlock(
                hidden_size=embed_dim,
                mlp_dim=embed_dim * 4,
                num_heads=num_heads,
                dropout_rate=0.1
            )
            for _ in range(2)
        ])
        
        # Adaptive pooling for variable-length sequences
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-task heads
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Auxiliary heads for individual modalities
        self.modality_heads = nn.ModuleDict({
            modality: nn.Linear(embed_dim, num_classes)
            for modality in modalities
        })
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Attention visualization storage
        self.attention_maps = {}
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Multi-modal fusion network initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'modalities': modalities,
                'fusion_strategy': fusion_strategy,
                'embed_dim': embed_dim,
                'num_classes': num_classes
            }
        )
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        clinical_data: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal fusion network.
        
        Args:
            inputs: Dictionary of input tensors for each modality
            clinical_data: Optional clinical context data
            
        Returns:
            Dictionary containing predictions and auxiliary outputs
        """
        batch_size = next(iter(inputs.values())).shape[0]
        
        # Encode each modality
        modality_features = {}
        modality_tokens = {}
        individual_predictions = {}
        
        for modality in self.modalities:
            if modality in inputs:
                mod_feature, mod_tokens = self.modality_encoders[modality](inputs[modality])
                modality_features[modality] = mod_feature
                modality_tokens[modality] = mod_tokens
                
                # Individual modality predictions
                individual_predictions[modality] = self.modality_heads[modality](mod_feature)
        
        # Fusion strategy
        if self.fusion_strategy == "hierarchical":
            fused_features = self._hierarchical_fusion(modality_features, modality_tokens)
        elif self.fusion_strategy == "parallel":
            fused_features = self._parallel_fusion(modality_features, modality_tokens)
        else:  # sequential
            fused_features = self._sequential_fusion(modality_features, modality_tokens)
        
        # Add clinical context if available
        if self.use_clinical_context and clinical_data is not None:
            clinical_features = self.clinical_encoder(clinical_data)
            fused_features = torch.cat([fused_features, clinical_features], dim=-1)
            # Project back to embed_dim
            fused_features = nn.Linear(fused_features.shape[-1], self.embed_dim).to(fused_features.device)(fused_features)
        
        # Global fusion
        fused_sequence = fused_features.unsqueeze(1)  # [B, 1, D]
        for fusion_block in self.global_fusion:
            fused_sequence = fusion_block(fused_sequence)
        
        # Pool to single feature vector
        final_features = fused_sequence.squeeze(1)  # [B, D]
        
        # Final predictions
        logits = self.classification_head(final_features)
        uncertainty = self.uncertainty_head(final_features)
        
        outputs = {
            'logits': logits,
            'features': final_features,
            'individual_predictions': individual_predictions,
            'uncertainty': uncertainty,
            'modality_features': modality_features
        }
        
        if hasattr(self, 'attention_maps') and self.attention_maps:
            outputs['attention_maps'] = self.attention_maps
        
        return outputs
    
    def _hierarchical_fusion(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_tokens: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Perform hierarchical fusion with cross-modal attention."""
        # Stack modality features
        available_modalities = list(modality_features.keys())
        
        if len(available_modalities) < 2:
            # Single modality case
            return list(modality_features.values())[0]
        
        # Start with first modality as query
        query_features = modality_features[available_modalities[0]]
        
        # Progressively fuse with other modalities
        for i, modality in enumerate(available_modalities[1:]):
            key_value_features = modality_features[modality]
            
            # Apply cross-attention
            for layer in self.cross_attention_layers:
                attended_features, attention_weights = layer(
                    query_features.unsqueeze(1),
                    key_value_features.unsqueeze(1)
                )
                query_features = attended_features.squeeze(1)
                
                # Store attention maps
                self.attention_maps[f"cross_attention_{i}"] = attention_weights
        
        return query_features
    
    def _parallel_fusion(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_tokens: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Perform parallel fusion with pairwise cross-attention."""
        available_modalities = list(modality_features.keys())
        
        if len(available_modalities) < 2:
            return list(modality_features.values())[0]
        
        # Collect all pairwise attention results
        attended_features = []
        
        for i, mod1 in enumerate(available_modalities):
            mod1_attended = modality_features[mod1]
            
            for mod2 in available_modalities[i+1:]:
                pair_key = f"{mod1}_{mod2}"
                if pair_key in self.cross_attention_pairs:
                    # Cross-attend mod1 to mod2
                    attended_mod1, attention_weights = self.cross_attention_pairs[pair_key](
                        modality_features[mod1].unsqueeze(1),
                        modality_features[mod2].unsqueeze(1)
                    )
                    mod1_attended = attended_mod1.squeeze(1)
                    
                    # Store attention maps
                    self.attention_maps[pair_key] = attention_weights
            
            attended_features.append(mod1_attended)
        
        # Average all attended features
        return torch.mean(torch.stack(attended_features), dim=0)
    
    def _sequential_fusion(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_tokens: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Perform sequential fusion by concatenation and projection."""
        feature_list = list(modality_features.values())
        
        if len(feature_list) == 1:
            return feature_list[0]
        
        # Concatenate all modality features
        concatenated = torch.cat(feature_list, dim=-1)
        
        # Project back to embed_dim
        projection = nn.Linear(concatenated.shape[-1], self.embed_dim).to(concatenated.device)
        fused_features = projection(concatenated)
        
        return fused_features


class ClinicalContextEncoder(nn.Module):
    """
    Encoder for clinical context information (demographics, history, etc.).
    """
    
    def __init__(self, embed_dim: int = 768):
        """
        Initialize clinical context encoder.
        
        Args:
            embed_dim: Output embedding dimension
        """
        super().__init__()
        
        # Encoders for different types of clinical data
        self.categorical_encoder = nn.Sequential(
            nn.Linear(100, 256),  # Assume up to 100 categorical features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim // 2)
        )
        
        self.numerical_encoder = nn.Sequential(
            nn.Linear(50, 128),   # Assume up to 50 numerical features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim // 2)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
    
    def forward(self, clinical_data: torch.Tensor) -> torch.Tensor:
        """
        Encode clinical context data.
        
        Args:
            clinical_data: Clinical data tensor
            
        Returns:
            Encoded clinical features
        """
        # Split clinical data into categorical and numerical parts
        # This is a simplified implementation - adjust based on your data format
        categorical_data = clinical_data[:, :100]  # First 100 features as categorical
        numerical_data = clinical_data[:, 100:150]  # Next 50 as numerical
        
        # Encode each type
        cat_features = self.categorical_encoder(categorical_data)
        num_features = self.numerical_encoder(numerical_data)
        
        # Fuse
        combined = torch.cat([cat_features, num_features], dim=-1)
        clinical_features = self.fusion_layer(combined)
        
        return clinical_features


def create_multimodal_fusion(
    config_override: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> MultiModalFusion:
    """
    Factory function to create multi-modal fusion network.
    
    Args:
        config_override: Optional parameter overrides
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured multi-modal fusion network
    """
    config = get_config()
    
    # Default configuration
    default_config = {
        'modalities': ['CT', 'MRI'],
        'input_channels_per_modality': {'CT': 1, 'MRI': 1},
        'num_classes': 2,
        'embed_dim': 768,
        'fusion_layers': 4,
        'num_heads': 8,
        'use_clinical_context': True,
        'fusion_strategy': 'hierarchical'
    }
    
    # Apply overrides
    if config_override:
        default_config.update(config_override)
    
    return MultiModalFusion(
        session_id=session_id,
        user_id=user_id,
        **default_config
    )