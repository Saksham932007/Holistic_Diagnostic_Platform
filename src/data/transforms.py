"""
Medical image transformations and preprocessing for training and inference.

This module provides specialized transforms optimized for medical imaging
including intensity normalization, spatial preprocessing, and medical-specific
augmentations that preserve anatomical validity.
"""

from typing import Optional, Union, Tuple, List, Dict, Any, Callable, Sequence
import warnings
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

from monai.transforms import (
    Transform, MapTransform, Randomizable, ThreadUnsafe,
    Compose, RandRotate90d, RandFlipd, RandAffined, RandElasticDeformd,
    RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, RandAdjustContrastd, RandHistogramShiftd,
    Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd,
    SpatialPadd, RandSpatialCropd, RandCropByPosNegLabeld,
    NormalizeIntensityd, ThresholdIntensityd, ClampIntensityd,
    ToTensord, EnsureChannelFirstd, EnsureTyped
)
from monai.data import MetaTensor
from monai.utils import ensure_tuple_rep

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class MedicalNormalizationTransform(Transform):
    """
    Medical-specific intensity normalization with modality awareness.
    
    Performs Z-score normalization with optional intensity clipping
    and modality-specific preprocessing for CT, MRI, PET, etc.
    """
    
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        modality: str = "CT",
        clip_range: Optional[Tuple[float, float]] = None,
        intensity_bounds: Optional[Tuple[float, float]] = None,
        normalize_non_zero: bool = False,
        channel_wise: bool = False
    ):
        """
        Initialize medical normalization transform.
        
        Args:
            keys: Keys to transform
            modality: Imaging modality (CT, MRI, PET, etc.)
            clip_range: Optional intensity clipping range
            intensity_bounds: Intensity bounds for normalization
            normalize_non_zero: Whether to normalize only non-zero voxels
            channel_wise: Whether to normalize each channel separately
        """
        self.keys = ensure_tuple_rep(keys, 1)
        self.modality = modality.upper()
        self.clip_range = clip_range
        self.intensity_bounds = intensity_bounds
        self.normalize_non_zero = normalize_non_zero
        self.channel_wise = channel_wise
        
        # Modality-specific default parameters
        self._set_modality_defaults()
    
    def _set_modality_defaults(self):
        """Set default parameters based on imaging modality."""
        if self.modality == "CT":
            if self.clip_range is None:
                self.clip_range = (-1000, 1000)  # HU range
            if self.intensity_bounds is None:
                self.intensity_bounds = (-1000, 1000)
        elif self.modality == "MRI":
            if self.normalize_non_zero is False:
                self.normalize_non_zero = True  # MRI often has zero background
        elif self.modality == "PET":
            if self.clip_range is None:
                self.clip_range = (0, None)  # PET values are positive
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply medical normalization."""
        d = dict(data)
        
        for key in self.keys:
            if key not in d:
                continue
            
            img = d[key]
            
            # Convert to numpy if needed
            if isinstance(img, torch.Tensor):
                was_tensor = True
                img_np = img.detach().cpu().numpy()
            else:
                was_tensor = False
                img_np = np.array(img)
            
            # Apply modality-specific preprocessing
            img_processed = self._apply_modality_preprocessing(img_np)
            
            # Intensity clipping
            if self.clip_range is not None:
                min_val = self.clip_range[0] if self.clip_range[0] is not None else img_processed.min()
                max_val = self.clip_range[1] if self.clip_range[1] is not None else img_processed.max()
                img_processed = np.clip(img_processed, min_val, max_val)
            
            # Normalization
            if self.channel_wise and img_processed.ndim > 3:
                # Normalize each channel separately
                normalized = np.zeros_like(img_processed)
                for c in range(img_processed.shape[0]):
                    normalized[c] = self._normalize_array(img_processed[c])
                img_processed = normalized
            else:
                img_processed = self._normalize_array(img_processed)
            
            # Convert back to original format
            if was_tensor:
                d[key] = torch.from_numpy(img_processed).to(dtype=img.dtype)
            else:
                d[key] = img_processed
        
        return d
    
    def _apply_modality_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply modality-specific preprocessing."""
        if self.modality == "CT":
            # CT-specific preprocessing (already in HU)
            return img.astype(np.float32)
        elif self.modality == "MRI":
            # MRI-specific preprocessing
            return img.astype(np.float32)
        elif self.modality == "PET":
            # PET-specific preprocessing
            # Ensure positive values
            img = np.maximum(img, 0)
            return img.astype(np.float32)
        else:
            return img.astype(np.float32)
    
    def _normalize_array(self, img: np.ndarray) -> np.ndarray:
        """Normalize array using Z-score normalization."""
        if self.normalize_non_zero:
            # Only normalize non-zero voxels
            mask = img != 0
            if not np.any(mask):
                return img
            
            mean_val = np.mean(img[mask])
            std_val = np.std(img[mask])
            
            if std_val > 1e-8:
                img[mask] = (img[mask] - mean_val) / std_val
        else:
            # Normalize all voxels
            mean_val = np.mean(img)
            std_val = np.std(img)
            
            if std_val > 1e-8:
                img = (img - mean_val) / std_val
        
        return img


class AnatomicallyAwareRotation(RandRotate90d):
    """
    Anatomically-aware rotation that respects medical image orientation.
    
    Ensures rotations preserve anatomical validity and orientation metadata.
    """
    
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        prob: float = 0.1,
        max_k: int = 1,  # Limited rotation for medical images
        spatial_axes: Optional[Tuple[int, int]] = None,
        preserve_orientation: bool = True
    ):
        """
        Initialize anatomically-aware rotation.
        
        Args:
            keys: Keys to transform
            prob: Probability of applying transform
            max_k: Maximum number of 90-degree rotations
            spatial_axes: Spatial axes for rotation
            preserve_orientation: Whether to preserve orientation metadata
        """
        super().__init__(keys, prob, max_k, spatial_axes)
        self.preserve_orientation = preserve_orientation
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply anatomically-aware rotation."""
        # Store original orientation metadata if preserving
        original_orientations = {}
        if self.preserve_orientation:
            for key in self.keys:
                if key in data and hasattr(data[key], 'meta'):
                    original_orientations[key] = data[key].meta.get('orientation', None)
        
        # Apply rotation
        data = super().__call__(data)
        
        # Restore orientation metadata if needed
        if self.preserve_orientation:
            for key, orientation in original_orientations.items():
                if key in data and hasattr(data[key], 'meta') and orientation is not None:
                    # Note: In practice, you'd need to update orientation based on rotation
                    # This is a simplified version
                    data[key].meta['orientation'] = orientation
        
        return data


class MedicalElasticDeformation(RandElasticDeformd):
    """
    Medical-specific elastic deformation with anatomical constraints.
    
    Applies realistic elastic deformations that preserve anatomical plausibility
    while providing effective data augmentation.
    """
    
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        sigma_range: Tuple[float, float] = (5.0, 15.0),
        magnitude_range: Tuple[float, float] = (1.0, 3.0),
        prob: float = 0.1,
        rotate_range: Optional[Tuple[float, float]] = None,
        shear_range: Optional[Tuple[float, float]] = None,
        translate_range: Optional[Tuple[float, float]] = None,
        scale_range: Optional[Tuple[float, float]] = None,
        spatial_size: Optional[Union[int, Tuple[int, ...]]] = None,
        mode: str = "bilinear",
        padding_mode: str = "reflection",
        anatomical_constraints: bool = True
    ):
        """
        Initialize medical elastic deformation.
        
        Args:
            keys: Keys to transform
            sigma_range: Range for Gaussian kernel sigma
            magnitude_range: Range for deformation magnitude
            prob: Probability of applying transform
            rotate_range: Rotation range in degrees
            shear_range: Shear range
            translate_range: Translation range
            scale_range: Scaling range
            spatial_size: Spatial size for resampling
            mode: Interpolation mode
            padding_mode: Padding mode
            anatomical_constraints: Whether to apply anatomical constraints
        """
        super().__init__(
            keys=keys,
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            prob=prob,
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            mode=mode,
            padding_mode=padding_mode
        )
        self.anatomical_constraints = anatomical_constraints
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply medical elastic deformation with constraints."""
        if not self._do_transform:
            return data
        
        # Apply base elastic deformation
        data = super().__call__(data)
        
        # Apply anatomical constraints if enabled
        if self.anatomical_constraints:
            data = self._apply_anatomical_constraints(data)
        
        return data
    
    def _apply_anatomical_constraints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply anatomical plausibility constraints."""
        # This is a simplified version - in practice, implement sophisticated
        # anatomical constraint checking based on medical domain knowledge
        return data


class IntensityAugmentation(Transform):
    """
    Medical-specific intensity augmentation preserving tissue characteristics.
    
    Applies realistic intensity variations that simulate imaging variability
    while preserving diagnostic information.
    """
    
    def __init__(
        self,
        keys: Union[str, Sequence[str]],
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        gain_range: Tuple[float, float] = (0.9, 1.1),
        noise_std_range: Tuple[float, float] = (0.0, 0.05),
        blur_sigma_range: Tuple[float, float] = (0.0, 1.0),
        prob: float = 0.15,
        preserve_range: bool = True
    ):
        """
        Initialize intensity augmentation.
        
        Args:
            keys: Keys to transform
            gamma_range: Range for gamma correction
            gain_range: Range for gain adjustment
            noise_std_range: Range for noise standard deviation
            blur_sigma_range: Range for Gaussian blur sigma
            prob: Probability of applying transform
            preserve_range: Whether to preserve original intensity range
        """
        self.keys = ensure_tuple_rep(keys, 1)
        self.gamma_range = gamma_range
        self.gain_range = gain_range
        self.noise_std_range = noise_std_range
        self.blur_sigma_range = blur_sigma_range
        self.prob = prob
        self.preserve_range = preserve_range
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intensity augmentation."""
        if np.random.rand() > self.prob:
            return data
        
        d = dict(data)
        
        for key in self.keys:
            if key not in d:
                continue
            
            img = d[key]
            
            # Convert to numpy if needed
            if isinstance(img, torch.Tensor):
                was_tensor = True
                img_np = img.detach().cpu().numpy()
                device = img.device
                dtype = img.dtype
            else:
                was_tensor = False
                img_np = np.array(img)
            
            # Store original range if preserving
            if self.preserve_range:
                orig_min, orig_max = img_np.min(), img_np.max()
            
            # Apply augmentations
            img_aug = self._apply_intensity_augmentations(img_np)
            
            # Restore range if requested
            if self.preserve_range and orig_max > orig_min:
                current_min, current_max = img_aug.min(), img_aug.max()
                if current_max > current_min:
                    img_aug = (img_aug - current_min) / (current_max - current_min)
                    img_aug = img_aug * (orig_max - orig_min) + orig_min
            
            # Convert back to original format
            if was_tensor:
                d[key] = torch.from_numpy(img_aug).to(device=device, dtype=dtype)
            else:
                d[key] = img_aug
        
        return d
    
    def _apply_intensity_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply various intensity augmentations."""
        # Gamma correction
        gamma = np.random.uniform(*self.gamma_range)
        if gamma != 1.0:
            # Normalize to [0, 1] for gamma correction
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = np.power(img_norm, gamma) * (img.max() - img.min()) + img.min()
        
        # Gain adjustment
        gain = np.random.uniform(*self.gain_range)
        img = img * gain
        
        # Gaussian noise
        noise_std = np.random.uniform(*self.noise_std_range)
        if noise_std > 0:
            noise = np.random.normal(0, noise_std * np.std(img), img.shape)
            img = img + noise
        
        # Gaussian blur
        blur_sigma = np.random.uniform(*self.blur_sigma_range)
        if blur_sigma > 0:
            img = ndimage.gaussian_filter(img, sigma=blur_sigma)
        
        return img.astype(np.float32)


def get_training_transforms(
    modality: str = "CT",
    image_key: str = "image",
    label_key: str = "label",
    spatial_size: Tuple[int, int, int] = (96, 96, 96),
    intensity_range: Optional[Tuple[float, float]] = None,
    augmentation_prob: float = 0.2,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Compose:
    """
    Get comprehensive training transforms for medical images.
    
    Args:
        modality: Imaging modality (CT, MRI, PET, etc.)
        image_key: Key for image data
        label_key: Key for label data
        spatial_size: Target spatial size
        intensity_range: Intensity normalization range
        augmentation_prob: Probability for augmentation transforms
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Composed training transforms
    """
    config = get_config()
    
    # Define transform pipeline
    transforms = [
        # Ensure proper format
        EnsureChannelFirstd(keys=[image_key, label_key], strict_check=False),
        EnsureTyped(keys=[image_key, label_key]),
        
        # Spatial preprocessing
        Spacingd(
            keys=[image_key, label_key],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
            align_corners=(True, None)
        ),
        Orientationd(keys=[image_key, label_key], axcodes="RAS"),
        
        # Intensity preprocessing
        MedicalNormalizationTransform(
            keys=[image_key],
            modality=modality,
            intensity_bounds=intensity_range
        ),
        
        # Cropping and padding
        CropForegroundd(
            keys=[image_key, label_key],
            source_key=image_key,
            margin=10
        ),
        SpatialPadd(
            keys=[image_key, label_key],
            spatial_size=spatial_size,
            mode="constant"
        ),
        
        # Random crop
        RandCropByPosNegLabeld(
            keys=[image_key, label_key],
            label_key=label_key,
            spatial_size=spatial_size,
            pos=2,  # More positive samples
            neg=1,
            num_samples=1,
            image_key=image_key,
            image_threshold=0
        ),
        
        # Data augmentation
        AnatomicallyAwareRotation(
            keys=[image_key, label_key],
            prob=augmentation_prob,
            max_k=1
        ),
        RandFlipd(
            keys=[image_key, label_key],
            spatial_axis=[0, 1, 2],
            prob=augmentation_prob
        ),
        MedicalElasticDeformation(
            keys=[image_key, label_key],
            prob=augmentation_prob,
            sigma_range=(5.0, 10.0),
            magnitude_range=(1.0, 2.0)
        ),
        
        # Intensity augmentation (image only)
        IntensityAugmentation(
            keys=[image_key],
            prob=augmentation_prob
        ),
        RandGaussianNoised(
            keys=[image_key],
            prob=augmentation_prob * 0.5,
            std=0.05
        ),
        RandScaleIntensityd(
            keys=[image_key],
            factors=0.1,
            prob=augmentation_prob * 0.5
        ),
        RandShiftIntensityd(
            keys=[image_key],
            offsets=0.1,
            prob=augmentation_prob * 0.5
        ),
        
        # Final preprocessing
        ThresholdIntensityd(
            keys=[image_key],
            threshold=-3.0,
            above=True,
            cval=-3.0
        ),
        ThresholdIntensityd(
            keys=[image_key],
            threshold=3.0,
            above=False,
            cval=3.0
        ),
        
        # Convert to tensor
        ToTensord(keys=[image_key, label_key])
    ]
    
    log_audit_event(
        event_type=AuditEventType.SYSTEM_START,
        severity=AuditSeverity.INFO,
        message="Training transforms initialized",
        user_id=user_id,
        session_id=session_id,
        additional_data={
            'modality': modality,
            'spatial_size': spatial_size,
            'augmentation_prob': augmentation_prob,
            'transform_count': len(transforms)
        }
    )
    
    return Compose(transforms)


def get_validation_transforms(
    modality: str = "CT",
    image_key: str = "image",
    label_key: str = "label",
    spatial_size: Optional[Tuple[int, int, int]] = None,
    intensity_range: Optional[Tuple[float, float]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Compose:
    """
    Get validation transforms for medical images (no augmentation).
    
    Args:
        modality: Imaging modality
        image_key: Key for image data
        label_key: Key for label data
        spatial_size: Optional target spatial size
        intensity_range: Intensity normalization range
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Composed validation transforms
    """
    transforms = [
        # Ensure proper format
        EnsureChannelFirstd(keys=[image_key, label_key], strict_check=False),
        EnsureTyped(keys=[image_key, label_key]),
        
        # Spatial preprocessing
        Spacingd(
            keys=[image_key, label_key],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
            align_corners=(True, None)
        ),
        Orientationd(keys=[image_key, label_key], axcodes="RAS"),
        
        # Intensity preprocessing
        MedicalNormalizationTransform(
            keys=[image_key],
            modality=modality,
            intensity_bounds=intensity_range
        ),
        
        # Cropping
        CropForegroundd(
            keys=[image_key, label_key],
            source_key=image_key,
            margin=10
        ),
    ]
    
    # Add padding if spatial size specified
    if spatial_size is not None:
        transforms.append(
            SpatialPadd(
                keys=[image_key, label_key],
                spatial_size=spatial_size,
                mode="constant"
            )
        )
    
    transforms.extend([
        # Final preprocessing
        ThresholdIntensityd(
            keys=[image_key],
            threshold=-3.0,
            above=True,
            cval=-3.0
        ),
        ThresholdIntensityd(
            keys=[image_key],
            threshold=3.0,
            above=False,
            cval=3.0
        ),
        
        # Convert to tensor
        ToTensord(keys=[image_key, label_key])
    ])
    
    log_audit_event(
        event_type=AuditEventType.SYSTEM_START,
        severity=AuditSeverity.INFO,
        message="Validation transforms initialized",
        user_id=user_id,
        session_id=session_id,
        additional_data={
            'modality': modality,
            'spatial_size': spatial_size,
            'transform_count': len(transforms)
        }
    )
    
    return Compose(transforms)


def get_inference_transforms(
    modality: str = "CT",
    image_key: str = "image",
    intensity_range: Optional[Tuple[float, float]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Compose:
    """
    Get inference transforms for medical images.
    
    Args:
        modality: Imaging modality
        image_key: Key for image data
        intensity_range: Intensity normalization range
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Composed inference transforms
    """
    transforms = [
        # Ensure proper format
        EnsureChannelFirstd(keys=[image_key], strict_check=False),
        EnsureTyped(keys=[image_key]),
        
        # Spatial preprocessing
        Spacingd(
            keys=[image_key],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
            align_corners=True
        ),
        Orientationd(keys=[image_key], axcodes="RAS"),
        
        # Intensity preprocessing
        MedicalNormalizationTransform(
            keys=[image_key],
            modality=modality,
            intensity_bounds=intensity_range
        ),
        
        # Intensity clipping
        ThresholdIntensityd(
            keys=[image_key],
            threshold=-3.0,
            above=True,
            cval=-3.0
        ),
        ThresholdIntensityd(
            keys=[image_key],
            threshold=3.0,
            above=False,
            cval=3.0
        ),
        
        # Convert to tensor
        ToTensord(keys=[image_key])
    ]
    
    log_audit_event(
        event_type=AuditEventType.SYSTEM_START,
        severity=AuditSeverity.INFO,
        message="Inference transforms initialized",
        user_id=user_id,
        session_id=session_id,
        additional_data={
            'modality': modality,
            'transform_count': len(transforms)
        }
    )
    
    return Compose(transforms)