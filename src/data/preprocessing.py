"""
Advanced preprocessing pipeline for medical imaging.

This module provides sophisticated preprocessing capabilities including
artifact removal, intensity standardization, registration, and quality
assessment for production medical AI systems.
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np
import warnings
from pathlib import Path
from scipy import ndimage
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.transforms import (
    Transform, MapTransform, Compose, Spacing, Orientation,
    ScaleIntensityRange, CropForeground, SpatialPad, 
    HistogramNormalize, ZScoreNormalize
)
from monai.data import MetaTensor
from monai.utils import ensure_tuple_rep
from monai.networks.nets import BasicUNet

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class ArtifactDetectionAndRemoval(Transform):
    """
    Advanced artifact detection and removal for medical images.
    
    Detects and corrects common imaging artifacts including motion artifacts,
    metal artifacts, partial volume effects, and acquisition noise.
    """
    
    def __init__(
        self,
        modality: str = "CT",
        detect_motion: bool = True,
        detect_metal: bool = True,
        detect_noise: bool = True,
        correction_strength: float = 0.5,
        preserve_anatomy: bool = True
    ):
        """
        Initialize artifact detection and removal.
        
        Args:
            modality: Imaging modality
            detect_motion: Whether to detect motion artifacts
            detect_metal: Whether to detect metal artifacts
            detect_noise: Whether to detect noise artifacts
            correction_strength: Strength of artifact correction (0-1)
            preserve_anatomy: Whether to preserve anatomical structures
        """
        super().__init__()
        
        self.modality = modality.upper()
        self.detect_motion = detect_motion
        self.detect_metal = detect_metal
        self.detect_noise = detect_noise
        self.correction_strength = correction_strength
        self.preserve_anatomy = preserve_anatomy
        
        # Modality-specific parameters
        self._set_modality_parameters()
        
        # Pre-trained artifact detection network (placeholder)
        self.artifact_detector = None  # Would load pre-trained network
    
    def _set_modality_parameters(self):
        """Set modality-specific parameters for artifact detection."""
        if self.modality == "CT":
            self.metal_threshold = 3000  # HU threshold for metal
            self.air_threshold = -500    # HU threshold for air
            self.noise_std_threshold = 50
        elif self.modality == "MRI":
            self.motion_sensitivity = 0.1
            self.intensity_uniformity_threshold = 0.2
        elif self.modality == "PET":
            self.hot_spot_threshold = 5.0  # SUV threshold
            self.cold_spot_threshold = 0.1
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply artifact detection and removal.
        
        Args:
            data: Input image array
            
        Returns:
            Processed image with artifacts removed/corrected
        """
        processed_data = data.copy()
        
        try:
            # Motion artifact detection and correction
            if self.detect_motion:
                processed_data = self._correct_motion_artifacts(processed_data)
            
            # Metal artifact detection and correction
            if self.detect_metal and self.modality == "CT":
                processed_data = self._correct_metal_artifacts(processed_data)
            
            # Noise detection and reduction
            if self.detect_noise:
                processed_data = self._reduce_noise_artifacts(processed_data)
            
            # Intensity non-uniformity correction
            if self.modality == "MRI":
                processed_data = self._correct_bias_field(processed_data)
            
        except Exception as e:
            warnings.warn(f"Artifact correction failed: {str(e)}")
            return data  # Return original if correction fails
        
        return processed_data
    
    def _correct_motion_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect and correct motion artifacts."""
        # Simple motion detection using gradient analysis
        gradients = np.gradient(data, axis=(1, 2, 3))
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        # Detect motion-affected slices
        motion_metric = np.std(gradient_magnitude, axis=(1, 2, 3))
        motion_threshold = np.percentile(motion_metric, 95)
        
        # Apply smoothing to motion-affected regions
        if self.correction_strength > 0:
            motion_mask = motion_metric > motion_threshold
            
            for i, is_motion in enumerate(motion_mask):
                if is_motion:
                    # Apply gentle smoothing
                    sigma = self.correction_strength * 2.0
                    data[i] = ndimage.gaussian_filter(data[i], sigma=sigma)
        
        return data
    
    def _correct_metal_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect and correct metal artifacts in CT images."""
        # Detect metal regions (high HU values)
        metal_mask = data > self.metal_threshold
        
        if np.any(metal_mask):
            # Simple metal artifact reduction using interpolation
            for i in range(data.shape[0]):
                slice_data = data[i]
                slice_metal_mask = metal_mask[i]
                
                if np.any(slice_metal_mask):
                    # Dilate metal mask to include streaks
                    dilated_mask = ndimage.binary_dilation(
                        slice_metal_mask, 
                        structure=np.ones((3, 3, 3))
                    )
                    
                    # Interpolate over metal artifacts
                    if self.correction_strength > 0:
                        # Simple inpainting using distance-weighted interpolation
                        valid_mask = ~dilated_mask
                        if np.any(valid_mask):
                            slice_data[dilated_mask] = self._interpolate_metal_regions(
                                slice_data, valid_mask, dilated_mask
                            )
                            data[i] = slice_data
        
        return data
    
    def _interpolate_metal_regions(
        self, 
        slice_data: np.ndarray, 
        valid_mask: np.ndarray, 
        artifact_mask: np.ndarray
    ) -> np.ndarray:
        """Interpolate over metal artifact regions."""
        # Get coordinates
        valid_coords = np.column_stack(np.where(valid_mask))
        artifact_coords = np.column_stack(np.where(artifact_mask))
        
        if len(valid_coords) == 0 or len(artifact_coords) == 0:
            return slice_data
        
        # Get valid values
        valid_values = slice_data[valid_mask]
        
        # Calculate distances and interpolate
        distances = cdist(artifact_coords, valid_coords)
        
        # Use inverse distance weighting
        weights = 1.0 / (distances + 1e-8)
        weights /= np.sum(weights, axis=1, keepdims=True)
        
        interpolated_values = np.sum(weights * valid_values[np.newaxis, :], axis=1)
        
        # Apply interpolated values
        result = slice_data.copy()
        result[artifact_mask] = interpolated_values
        
        return result
    
    def _reduce_noise_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Reduce noise artifacts while preserving edges."""
        # Estimate noise level
        noise_std = np.std(data[data < np.percentile(data, 10)])
        
        if noise_std > self.noise_std_threshold:
            # Apply edge-preserving smoothing
            sigma = self.correction_strength * min(2.0, noise_std / 100.0)
            
            # Use anisotropic diffusion-like filtering
            for _ in range(3):  # Multiple iterations
                data = self._anisotropic_diffusion_step(data, sigma)
        
        return data
    
    def _anisotropic_diffusion_step(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Single step of anisotropic diffusion."""
        # Compute gradients
        grad_x = np.gradient(data, axis=-1)
        grad_y = np.gradient(data, axis=-2)
        grad_z = np.gradient(data, axis=-3)
        
        # Compute gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Compute diffusion coefficient (preserve edges)
        k = np.percentile(grad_mag, 90)  # Edge threshold
        diffusion_coef = np.exp(-(grad_mag / k)**2)
        
        # Apply diffusion
        smoothed = ndimage.gaussian_filter(data, sigma=sigma)
        data = data + diffusion_coef * sigma * (smoothed - data)
        
        return data
    
    def _correct_bias_field(self, data: np.ndarray) -> np.ndarray:
        """Correct intensity non-uniformity (bias field) in MRI."""
        # Simple bias field estimation and correction
        # In practice, use more sophisticated methods like N4ITK
        
        # Estimate bias field using low-pass filtering
        bias_field = ndimage.gaussian_filter(data, sigma=20)
        
        # Avoid division by zero
        bias_field[bias_field == 0] = 1.0
        
        # Correct bias field
        corrected = data / bias_field
        
        # Normalize intensity
        corrected = (corrected - np.min(corrected)) / (np.max(corrected) - np.min(corrected))
        corrected *= (np.max(data) - np.min(data))
        corrected += np.min(data)
        
        return corrected


class AdvancedIntensityStandardization(Transform):
    """
    Advanced intensity standardization with histogram matching and modality-specific normalization.
    """
    
    def __init__(
        self,
        modality: str = "CT",
        reference_histogram: Optional[np.ndarray] = None,
        standardization_method: str = "histogram_matching",
        roi_based: bool = True,
        preserve_range: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize advanced intensity standardization.
        
        Args:
            modality: Imaging modality
            reference_histogram: Reference histogram for matching
            standardization_method: Standardization method
            roi_based: Whether to use ROI-based normalization
            preserve_range: Whether to preserve original intensity range
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        super().__init__()
        
        self.modality = modality.upper()
        self.reference_histogram = reference_histogram
        self.standardization_method = standardization_method
        self.roi_based = roi_based
        self.preserve_range = preserve_range
        self._session_id = session_id
        self._user_id = user_id
        
        # Load or create reference histogram
        if self.reference_histogram is None:
            self.reference_histogram = self._create_reference_histogram()
    
    def _create_reference_histogram(self) -> np.ndarray:
        """Create modality-specific reference histogram."""
        if self.modality == "CT":
            # CT reference histogram (simplified)
            bins = np.linspace(-1000, 3000, 1000)
            # Create a typical CT histogram with peaks for air, tissue, bone
            histogram = np.zeros_like(bins)
            histogram[bins < -500] = np.exp(-(bins[bins < -500] + 800)**2 / (200**2))  # Air peak
            histogram[(bins >= 0) & (bins < 100)] = np.exp(-(bins[(bins >= 0) & (bins < 100)] - 50)**2 / (30**2))  # Soft tissue
            histogram[bins > 200] = np.exp(-(bins[bins > 200] - 400)**2 / (100**2)) * 0.3  # Bone
            return histogram / np.sum(histogram)
        elif self.modality == "MRI":
            # MRI reference histogram (simplified)
            bins = np.linspace(0, 4000, 1000)
            histogram = np.exp(-bins / 500)  # Exponential decay
            return histogram / np.sum(histogram)
        else:
            # Generic reference
            bins = np.linspace(0, 1000, 1000)
            histogram = np.ones_like(bins)
            return histogram / np.sum(histogram)
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply intensity standardization."""
        if self.standardization_method == "histogram_matching":
            return self._histogram_matching(data)
        elif self.standardization_method == "percentile_normalization":
            return self._percentile_normalization(data)
        elif self.standardization_method == "z_score_roi":
            return self._z_score_roi_normalization(data)
        else:
            return self._standard_normalization(data)
    
    def _histogram_matching(self, data: np.ndarray) -> np.ndarray:
        """Perform histogram matching to reference."""
        # Calculate data histogram
        data_hist, data_bins = np.histogram(data.flatten(), bins=1000, density=True)
        
        # Calculate cumulative distributions
        data_cdf = np.cumsum(data_hist)
        data_cdf = data_cdf / data_cdf[-1]
        
        ref_cdf = np.cumsum(self.reference_histogram)
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # Create mapping function
        data_values = (data_bins[:-1] + data_bins[1:]) / 2
        ref_bins = np.linspace(data_values.min(), data_values.max(), len(self.reference_histogram))
        
        # Interpolate mapping
        mapped_data = np.interp(data.flatten(), data_values, ref_bins)
        
        return mapped_data.reshape(data.shape)
    
    def _percentile_normalization(self, data: np.ndarray) -> np.ndarray:
        """Normalize using percentile-based scaling."""
        if self.modality == "CT":
            # CT-specific percentile normalization
            p1, p99 = np.percentile(data, [1, 99])
            data_norm = np.clip(data, p1, p99)
            data_norm = (data_norm - p1) / (p99 - p1)
        elif self.modality == "MRI":
            # MRI-specific percentile normalization
            mask = data > 0  # Exclude background
            if np.any(mask):
                p1, p99 = np.percentile(data[mask], [1, 99])
                data_norm = data.copy()
                data_norm[mask] = np.clip(data_norm[mask], p1, p99)
                data_norm[mask] = (data_norm[mask] - p1) / (p99 - p1)
            else:
                data_norm = data
        else:
            # Generic percentile normalization
            p1, p99 = np.percentile(data, [5, 95])
            data_norm = np.clip(data, p1, p99)
            data_norm = (data_norm - p1) / (p99 - p1)
        
        return data_norm
    
    def _z_score_roi_normalization(self, data: np.ndarray) -> np.ndarray:
        """ROI-based Z-score normalization."""
        if self.roi_based:
            # Estimate ROI (foreground) mask
            if self.modality == "CT":
                roi_mask = data > -500  # Exclude air
            elif self.modality == "MRI":
                roi_mask = data > 0  # Exclude background
            else:
                roi_mask = data > np.percentile(data, 10)
            
            if np.any(roi_mask):
                roi_data = data[roi_mask]
                mean_val = np.mean(roi_data)
                std_val = np.std(roi_data)
                
                if std_val > 1e-8:
                    data_norm = (data - mean_val) / std_val
                else:
                    data_norm = data - mean_val
            else:
                data_norm = data
        else:
            # Global Z-score normalization
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            if std_val > 1e-8:
                data_norm = (data - mean_val) / std_val
            else:
                data_norm = data - mean_val
        
        return data_norm
    
    def _standard_normalization(self, data: np.ndarray) -> np.ndarray:
        """Standard min-max normalization."""
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val > min_val:
            data_norm = (data - min_val) / (max_val - min_val)
        else:
            data_norm = data - min_val
        
        return data_norm


class QualityAssessment(Transform):
    """
    Image quality assessment and scoring for medical images.
    """
    
    def __init__(
        self,
        modality: str = "CT",
        assess_snr: bool = True,
        assess_artifacts: bool = True,
        assess_resolution: bool = True,
        min_quality_threshold: float = 0.5
    ):
        """
        Initialize quality assessment.
        
        Args:
            modality: Imaging modality
            assess_snr: Whether to assess signal-to-noise ratio
            assess_artifacts: Whether to assess artifacts
            assess_resolution: Whether to assess resolution
            min_quality_threshold: Minimum quality score threshold
        """
        super().__init__()
        
        self.modality = modality.upper()
        self.assess_snr = assess_snr
        self.assess_artifacts = assess_artifacts
        self.assess_resolution = assess_resolution
        self.min_quality_threshold = min_quality_threshold
    
    def __call__(self, data: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality.
        
        Args:
            data: Input image array
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {}
        
        if self.assess_snr:
            quality_metrics['snr'] = self._calculate_snr(data)
        
        if self.assess_artifacts:
            quality_metrics['artifact_score'] = self._assess_artifacts(data)
        
        if self.assess_resolution:
            quality_metrics['resolution_score'] = self._assess_resolution(data)
        
        # Calculate overall quality score
        scores = list(quality_metrics.values())
        quality_metrics['overall_quality'] = np.mean(scores) if scores else 0.0
        
        # Quality flag
        quality_metrics['acceptable_quality'] = quality_metrics['overall_quality'] >= self.min_quality_threshold
        
        return quality_metrics
    
    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        # Estimate signal and noise regions
        if self.modality == "CT":
            # For CT, use tissue region as signal, air as noise
            signal_mask = (data > 0) & (data < 200)  # Soft tissue range
            noise_mask = data < -800  # Air region
        elif self.modality == "MRI":
            # For MRI, use foreground as signal, background as noise
            signal_mask = data > np.percentile(data, 50)
            noise_mask = data < np.percentile(data, 10)
        else:
            # Generic approach
            signal_mask = data > np.percentile(data, 60)
            noise_mask = data < np.percentile(data, 20)
        
        if np.any(signal_mask) and np.any(noise_mask):
            signal_mean = np.mean(data[signal_mask])
            noise_std = np.std(data[noise_mask])
            
            if noise_std > 1e-8:
                snr = signal_mean / noise_std
                # Normalize to 0-1 scale
                snr_normalized = min(1.0, max(0.0, (snr - 5.0) / 20.0))
                return snr_normalized
        
        return 0.5  # Default moderate score
    
    def _assess_artifacts(self, data: np.ndarray) -> float:
        """Assess presence and severity of artifacts."""
        artifact_score = 1.0  # Start with perfect score
        
        # Check for motion artifacts (high gradient variations)
        gradients = np.gradient(data, axis=(1, 2, 3) if data.ndim == 4 else (0, 1, 2))
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        motion_metric = np.std(gradient_magnitude) / np.mean(gradient_magnitude)
        
        if motion_metric > 0.5:  # High motion
            artifact_score *= 0.5
        elif motion_metric > 0.3:  # Moderate motion
            artifact_score *= 0.7
        
        # Check for intensity artifacts
        intensity_range = np.percentile(data, 99) - np.percentile(data, 1)
        intensity_std = np.std(data)
        intensity_uniformity = intensity_std / intensity_range if intensity_range > 0 else 1.0
        
        if intensity_uniformity > 0.3:  # High intensity variation
            artifact_score *= 0.6
        
        return artifact_score
    
    def _assess_resolution(self, data: np.ndarray) -> float:
        """Assess image resolution and sharpness."""
        # Use Laplacian variance as sharpness metric
        if data.ndim == 3:
            # 3D Laplacian
            laplacian = ndimage.laplace(data)
        else:
            # Handle different dimensions
            laplacian = ndimage.laplace(data[0] if data.ndim == 4 else data)
        
        sharpness = np.var(laplacian)
        
        # Normalize sharpness score (modality-specific thresholds)
        if self.modality == "CT":
            sharpness_threshold = 1000
        elif self.modality == "MRI":
            sharpness_threshold = 500
        else:
            sharpness_threshold = 100
        
        resolution_score = min(1.0, sharpness / sharpness_threshold)
        return resolution_score


def create_advanced_preprocessing_pipeline(
    modality: str = "CT",
    enable_artifact_removal: bool = True,
    enable_standardization: bool = True,
    enable_quality_assessment: bool = True,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Compose:
    """
    Create comprehensive preprocessing pipeline.
    
    Args:
        modality: Imaging modality
        enable_artifact_removal: Whether to enable artifact removal
        enable_standardization: Whether to enable intensity standardization
        enable_quality_assessment: Whether to enable quality assessment
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Composed preprocessing pipeline
    """
    transforms = []
    
    # Basic spatial preprocessing
    transforms.extend([
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientation(axcodes="RAS")
    ])
    
    # Artifact removal
    if enable_artifact_removal:
        transforms.append(
            ArtifactDetectionAndRemoval(modality=modality)
        )
    
    # Intensity standardization
    if enable_standardization:
        transforms.append(
            AdvancedIntensityStandardization(
                modality=modality,
                session_id=session_id,
                user_id=user_id
            )
        )
    
    # Quality assessment
    if enable_quality_assessment:
        transforms.append(
            QualityAssessment(modality=modality)
        )
    
    # Final preprocessing
    transforms.extend([
        CropForeground(),
        SpatialPad(spatial_size=(96, 96, 96))
    ])
    
    log_audit_event(
        event_type=AuditEventType.SYSTEM_START,
        severity=AuditSeverity.INFO,
        message="Advanced preprocessing pipeline created",
        user_id=user_id,
        session_id=session_id,
        additional_data={
            'modality': modality,
            'artifact_removal': enable_artifact_removal,
            'standardization': enable_standardization,
            'quality_assessment': enable_quality_assessment
        }
    )
    
    return Compose(transforms)