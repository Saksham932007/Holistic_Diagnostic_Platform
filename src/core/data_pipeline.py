"""
Advanced Data Pipeline

Comprehensive data processing pipeline for medical imaging
with ETL, validation, augmentation, and quality assurance.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
import nibabel as nib
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
from torchvision import transforms
import monai
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, RandRotated,
    RandZoomd, RandGaussianNoised, ToTensord
)

from src.core.config import settings
from src.core.audit import audit_logger
from src.utils.dicom_utils import DicomProcessor

logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Supported data formats."""
    DICOM = "dicom"
    NIFTI = "nifti"
    PNG = "png"
    JPG = "jpg"
    NUMPY = "npy"
    TIFF = "tiff"

class DataSplit(Enum):
    """Data split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class PipelineStage(Enum):
    """Pipeline processing stages."""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    AUGMENTATION = "augmentation"
    SPLITTING = "splitting"
    CACHING = "caching"
    QUALITY_CHECK = "quality_check"

@dataclass
class DataSample:
    """Represents a single data sample."""
    sample_id: str
    file_path: Path
    format: DataFormat
    modality: Optional[str] = None
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    series_id: Optional[str] = None
    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    is_valid: bool = True
    error_message: Optional[str] = None

@dataclass
class PipelineConfig:
    """Configuration for data pipeline."""
    input_directory: Path
    output_directory: Path
    cache_directory: Optional[Path] = None
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    target_size: Tuple[int, int, int] = (128, 128, 128)
    intensity_range: Tuple[float, float] = (-1000, 1000)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    augmentation_probability: float = 0.5
    cache_rate: float = 1.0
    num_workers: int = 4
    batch_size: int = 4
    enable_validation: bool = True
    quality_threshold: float = 0.7

class DataValidator:
    """Validates medical imaging data quality and integrity."""
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_rules = {
            'dicom_header_completeness': self._check_dicom_headers,
            'image_quality': self._check_image_quality,
            'anatomical_consistency': self._check_anatomical_consistency,
            'contrast_adequacy': self._check_contrast,
            'artifacts_detection': self._detect_artifacts,
            'spatial_resolution': self._check_spatial_resolution
        }
    
    async def validate_sample(self, sample: DataSample) -> DataSample:
        """Validate a single data sample."""
        try:
            if sample.format == DataFormat.DICOM:
                sample = await self._validate_dicom(sample)
            elif sample.format in [DataFormat.NIFTI]:
                sample = await self._validate_nifti(sample)
            elif sample.format in [DataFormat.PNG, DataFormat.JPG]:
                sample = await self._validate_image(sample)
            
            # Run general quality checks
            sample = await self._run_quality_checks(sample)
            
            return sample
            
        except Exception as e:
            logger.error(f"Validation failed for {sample.sample_id}: {e}")
            sample.is_valid = False
            sample.error_message = str(e)
            sample.quality_score = 0.0
            return sample
    
    async def _validate_dicom(self, sample: DataSample) -> DataSample:
        """Validate DICOM file."""
        try:
            ds = pydicom.dcmread(str(sample.file_path))
            
            # Extract metadata
            sample.metadata.update({
                'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'series_description': getattr(ds, 'SeriesDescription', 'Unknown'),
                'slice_thickness': getattr(ds, 'SliceThickness', 'Unknown'),
                'pixel_spacing': getattr(ds, 'PixelSpacing', 'Unknown'),
                'image_orientation': getattr(ds, 'ImageOrientationPatient', 'Unknown')
            })
            
            sample.modality = sample.metadata['modality']
            
            # Check image data
            if hasattr(ds, 'pixel_array'):
                pixel_data = ds.pixel_array
                sample.metadata.update({
                    'image_shape': pixel_data.shape,
                    'image_dtype': str(pixel_data.dtype),
                    'intensity_range': (float(pixel_data.min()), float(pixel_data.max())),
                    'mean_intensity': float(pixel_data.mean()),
                    'std_intensity': float(pixel_data.std())
                })
            else:
                sample.is_valid = False
                sample.error_message = "No pixel data found in DICOM"
                
        except InvalidDicomError as e:
            sample.is_valid = False
            sample.error_message = f"Invalid DICOM file: {e}"
        except Exception as e:
            sample.is_valid = False
            sample.error_message = f"DICOM validation error: {e}"
        
        return sample
    
    async def _validate_nifti(self, sample: DataSample) -> DataSample:
        """Validate NIfTI file."""
        try:
            img = nib.load(str(sample.file_path))
            data = img.get_fdata()
            
            sample.metadata.update({
                'image_shape': data.shape,
                'image_dtype': str(data.dtype),
                'voxel_size': img.header.get_zooms(),
                'intensity_range': (float(data.min()), float(data.max())),
                'mean_intensity': float(data.mean()),
                'std_intensity': float(data.std()),
                'affine_matrix': img.affine.tolist()
            })
            
        except Exception as e:
            sample.is_valid = False
            sample.error_message = f"NIfTI validation error: {e}"
        
        return sample
    
    async def _validate_image(self, sample: DataSample) -> DataSample:
        """Validate standard image file."""
        try:
            img = Image.open(sample.file_path)
            img_array = np.array(img)
            
            sample.metadata.update({
                'image_shape': img_array.shape,
                'image_mode': img.mode,
                'image_format': img.format,
                'intensity_range': (int(img_array.min()), int(img_array.max())),
                'mean_intensity': float(img_array.mean()),
                'std_intensity': float(img_array.std())
            })
            
        except Exception as e:
            sample.is_valid = False
            sample.error_message = f"Image validation error: {e}"
        
        return sample
    
    async def _run_quality_checks(self, sample: DataSample) -> DataSample:
        """Run comprehensive quality checks."""
        if not sample.is_valid:
            return sample
        
        quality_scores = []
        
        for rule_name, rule_func in self.validation_rules.items():
            try:
                score = await rule_func(sample)
                quality_scores.append(score)
                sample.metadata[f"quality_{rule_name}"] = score
            except Exception as e:
                logger.warning(f"Quality check {rule_name} failed for {sample.sample_id}: {e}")
                quality_scores.append(0.5)  # Default score for failed checks
        
        sample.quality_score = np.mean(quality_scores)
        
        return sample
    
    async def _check_dicom_headers(self, sample: DataSample) -> float:
        """Check DICOM header completeness."""
        if sample.format != DataFormat.DICOM:
            return 1.0
        
        required_tags = [
            'patient_id', 'study_date', 'modality', 'series_description'
        ]
        
        missing_count = sum(1 for tag in required_tags if sample.metadata.get(tag) == 'Unknown')
        
        return 1.0 - (missing_count / len(required_tags))
    
    async def _check_image_quality(self, sample: DataSample) -> float:
        """Check overall image quality."""
        if 'intensity_range' not in sample.metadata:
            return 0.5
        
        min_intensity, max_intensity = sample.metadata['intensity_range']
        dynamic_range = max_intensity - min_intensity
        
        # Check for reasonable dynamic range
        if dynamic_range > 100:  # Good dynamic range
            return 1.0
        elif dynamic_range > 50:
            return 0.8
        elif dynamic_range > 10:
            return 0.6
        else:
            return 0.3
    
    async def _check_anatomical_consistency(self, sample: DataSample) -> float:
        """Check anatomical consistency."""
        # Placeholder for anatomical checks
        # In real implementation, would use anatomical landmarks
        return 0.8
    
    async def _check_contrast(self, sample: DataSample) -> float:
        """Check image contrast adequacy."""
        if 'std_intensity' not in sample.metadata:
            return 0.5
        
        std_intensity = sample.metadata['std_intensity']
        
        # Higher standard deviation usually indicates better contrast
        if std_intensity > 100:
            return 1.0
        elif std_intensity > 50:
            return 0.8
        elif std_intensity > 20:
            return 0.6
        else:
            return 0.3
    
    async def _detect_artifacts(self, sample: DataSample) -> float:
        """Detect common imaging artifacts."""
        # Placeholder for artifact detection
        # Would implement specific artifact detection algorithms
        return 0.9
    
    async def _check_spatial_resolution(self, sample: DataSample) -> float:
        """Check spatial resolution adequacy."""
        if 'image_shape' not in sample.metadata:
            return 0.5
        
        shape = sample.metadata['image_shape']
        
        if len(shape) == 3:  # 3D image
            min_dim = min(shape)
            if min_dim >= 128:
                return 1.0
            elif min_dim >= 64:
                return 0.8
            elif min_dim >= 32:
                return 0.6
            else:
                return 0.3
        elif len(shape) == 2:  # 2D image
            min_dim = min(shape[:2])
            if min_dim >= 256:
                return 1.0
            elif min_dim >= 128:
                return 0.8
            elif min_dim >= 64:
                return 0.6
            else:
                return 0.3
        
        return 0.5

class DataPreprocessor:
    """Preprocesses medical imaging data for training."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize data preprocessor."""
        self.config = config
        self.transforms_cache = {}
    
    def get_transforms(self, stage: DataSplit, modality: str = "CT") -> Compose:
        """Get preprocessing transforms for specific stage and modality."""
        cache_key = f"{stage.value}_{modality}"
        
        if cache_key in self.transforms_cache:
            return self.transforms_cache[cache_key]
        
        base_transforms = [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=self.config.target_spacing, mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], 
                a_min=self.config.intensity_range[0], 
                a_max=self.config.intensity_range[1],
                b_min=0.0, 
                b_max=1.0, 
                clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            Resized(keys=["image"], spatial_size=self.config.target_size, mode="trilinear")
        ]
        
        # Add augmentations for training
        if stage == DataSplit.TRAIN:
            augmentations = [
                RandRotated(
                    keys=["image"],
                    range_x=0.2,
                    range_y=0.2,
                    range_z=0.2,
                    prob=self.config.augmentation_probability,
                    mode="bilinear"
                ),
                RandZoomd(
                    keys=["image"],
                    min_zoom=0.9,
                    max_zoom=1.1,
                    prob=self.config.augmentation_probability,
                    mode="trilinear"
                ),
                RandGaussianNoised(
                    keys=["image"],
                    std=0.1,
                    prob=self.config.augmentation_probability
                )
            ]
            base_transforms.extend(augmentations)
        
        base_transforms.append(ToTensord(keys=["image"]))
        
        transform = Compose(base_transforms)
        self.transforms_cache[cache_key] = transform
        
        return transform
    
    async def preprocess_sample(self, sample: DataSample, stage: DataSplit) -> Dict[str, Any]:
        """Preprocess a single sample."""
        try:
            # Create data dictionary for MONAI transforms
            data_dict = {"image": str(sample.file_path)}
            
            # Add labels if available
            if sample.labels:
                data_dict.update(sample.labels)
            
            # Get appropriate transforms
            transforms = self.get_transforms(stage, sample.modality or "CT")
            
            # Apply transforms
            transformed = transforms(data_dict)
            
            # Add metadata
            transformed["sample_id"] = sample.sample_id
            transformed["metadata"] = sample.metadata
            transformed["quality_score"] = sample.quality_score
            
            return transformed
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {sample.sample_id}: {e}")
            raise

class DataSplitter:
    """Handles train/validation/test splitting."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize data splitter."""
        self.config = config
    
    async def split_data(self, samples: List[DataSample]) -> Dict[DataSplit, List[DataSample]]:
        """Split data into train/validation/test sets."""
        # Filter valid samples
        valid_samples = [s for s in samples if s.is_valid and s.quality_score >= self.config.quality_threshold]
        
        logger.info(f"Splitting {len(valid_samples)} valid samples (filtered from {len(samples)} total)")
        
        # Group by patient ID to avoid data leakage
        patient_groups = {}
        for sample in valid_samples:
            patient_id = sample.patient_id or sample.sample_id
            if patient_id not in patient_groups:
                patient_groups[patient_id] = []
            patient_groups[patient_id].append(sample)
        
        patient_ids = list(patient_groups.keys())
        
        # Split patient IDs
        train_ids, temp_ids = train_test_split(
            patient_ids, 
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=42
        )
        
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=self.config.test_ratio / (self.config.val_ratio + self.config.test_ratio),
            random_state=42
        )
        
        # Assign samples to splits
        splits = {
            DataSplit.TRAIN: [],
            DataSplit.VALIDATION: [],
            DataSplit.TEST: []
        }
        
        for patient_id in train_ids:
            splits[DataSplit.TRAIN].extend(patient_groups[patient_id])
        
        for patient_id in val_ids:
            splits[DataSplit.VALIDATION].extend(patient_groups[patient_id])
        
        for patient_id in test_ids:
            splits[DataSplit.TEST].extend(patient_groups[patient_id])
        
        # Log split statistics
        for split_type, split_samples in splits.items():
            logger.info(f"{split_type.value}: {len(split_samples)} samples")
        
        return splits

class DataPipeline:
    """Main data processing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize data pipeline."""
        self.config = config
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(config)
        self.splitter = DataSplitter(config)
        
        # Ensure directories exist
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        if self.config.cache_directory:
            self.config.cache_directory.mkdir(parents=True, exist_ok=True)
    
    async def discover_data(self) -> List[DataSample]:
        """Discover data files in input directory."""
        logger.info(f"Discovering data in {self.config.input_directory}")
        
        samples = []
        supported_extensions = {
            '.dcm': DataFormat.DICOM,
            '.dicom': DataFormat.DICOM,
            '.nii': DataFormat.NIFTI,
            '.nii.gz': DataFormat.NIFTI,
            '.png': DataFormat.PNG,
            '.jpg': DataFormat.JPG,
            '.jpeg': DataFormat.JPG,
            '.npy': DataFormat.NUMPY,
            '.tiff': DataFormat.TIFF,
            '.tif': DataFormat.TIFF
        }
        
        for file_path in self.config.input_directory.rglob("*"):
            if file_path.is_file():
                # Check if file extension is supported
                for ext, format_type in supported_extensions.items():
                    if file_path.name.lower().endswith(ext.lower()):
                        sample_id = str(file_path.relative_to(self.config.input_directory))
                        sample = DataSample(
                            sample_id=sample_id,
                            file_path=file_path,
                            format=format_type
                        )
                        samples.append(sample)
                        break
        
        logger.info(f"Discovered {len(samples)} data files")
        return samples
    
    async def process_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """Process samples through validation and preprocessing."""
        logger.info("Processing samples through pipeline stages")
        
        # Stage 1: Validation
        if self.config.enable_validation:
            logger.info("Stage 1: Validation")
            validated_samples = []
            for i, sample in enumerate(samples):
                if i % 100 == 0:
                    logger.info(f"Validated {i}/{len(samples)} samples")
                validated_sample = await self.validator.validate_sample(sample)
                validated_samples.append(validated_sample)
                
                await audit_logger.log_event(
                    "sample_validated",
                    {
                        "sample_id": sample.sample_id,
                        "is_valid": validated_sample.is_valid,
                        "quality_score": validated_sample.quality_score
                    }
                )
        else:
            validated_samples = samples
        
        # Filter valid samples
        valid_samples = [s for s in validated_samples if s.is_valid]
        logger.info(f"Validation complete: {len(valid_samples)}/{len(samples)} samples valid")
        
        return valid_samples
    
    async def create_datasets(self, samples: List[DataSample]) -> Dict[DataSplit, DataLoader]:
        """Create datasets and dataloaders for each split."""
        # Split data
        splits = await self.splitter.split_data(samples)
        
        dataloaders = {}
        
        for split_type, split_samples in splits.items():
            if not split_samples:
                continue
            
            logger.info(f"Creating dataset for {split_type.value} with {len(split_samples)} samples")
            
            # Prepare data for MONAI Dataset
            data_dicts = []
            for sample in split_samples:
                data_dict = {
                    "image": str(sample.file_path),
                    "sample_id": sample.sample_id,
                    "quality_score": sample.quality_score
                }
                
                # Add labels if available
                if sample.labels:
                    data_dict.update(sample.labels)
                
                data_dicts.append(data_dict)
            
            # Get transforms
            transforms = self.preprocessor.get_transforms(split_type)
            
            # Create dataset
            if self.config.cache_directory and self.config.cache_rate > 0:
                dataset = CacheDataset(
                    data=data_dicts,
                    transform=transforms,
                    cache_rate=self.config.cache_rate,
                    cache_dir=str(self.config.cache_directory)
                )
            else:
                dataset = Dataset(data=data_dicts, transform=transforms)
            
            # Create dataloader
            is_training = split_type == DataSplit.TRAIN
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=is_training,
                num_workers=self.config.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            dataloaders[split_type] = dataloader
        
        return dataloaders
    
    async def run_pipeline(self) -> Dict[DataSplit, DataLoader]:
        """Run the complete data pipeline."""
        start_time = datetime.now()
        
        await audit_logger.log_event(
            "pipeline_started",
            {
                "input_directory": str(self.config.input_directory),
                "output_directory": str(self.config.output_directory),
                "target_size": self.config.target_size,
                "batch_size": self.config.batch_size
            }
        )
        
        try:
            # Discover data
            samples = await self.discover_data()
            
            if not samples:
                raise ValueError("No data files found in input directory")
            
            # Process samples
            valid_samples = await self.process_samples(samples)
            
            if not valid_samples:
                raise ValueError("No valid samples after processing")
            
            # Create datasets
            dataloaders = await self.create_datasets(valid_samples)
            
            # Save pipeline metadata
            await self._save_pipeline_metadata(samples, valid_samples, dataloaders)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            await audit_logger.log_event(
                "pipeline_completed",
                {
                    "total_samples": len(samples),
                    "valid_samples": len(valid_samples),
                    "processing_time_seconds": processing_time,
                    "dataloaders_created": list(dataloaders.keys())
                }
            )
            
            logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
            
            return dataloaders
            
        except Exception as e:
            await audit_logger.log_event(
                "pipeline_failed",
                {"error": str(e)}
            )
            raise
    
    async def _save_pipeline_metadata(
        self, 
        all_samples: List[DataSample], 
        valid_samples: List[DataSample], 
        dataloaders: Dict[DataSplit, DataLoader]
    ):
        """Save pipeline metadata and statistics."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "input_directory": str(self.config.input_directory),
                "output_directory": str(self.config.output_directory),
                "target_spacing": self.config.target_spacing,
                "target_size": self.config.target_size,
                "intensity_range": self.config.intensity_range,
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio
            },
            "statistics": {
                "total_samples_discovered": len(all_samples),
                "valid_samples": len(valid_samples),
                "validation_rate": len(valid_samples) / len(all_samples) if all_samples else 0,
                "splits": {
                    split_type.value: len(dataloader.dataset)
                    for split_type, dataloader in dataloaders.items()
                }
            },
            "quality_stats": {
                "mean_quality_score": np.mean([s.quality_score for s in valid_samples]),
                "min_quality_score": np.min([s.quality_score for s in valid_samples]),
                "max_quality_score": np.max([s.quality_score for s in valid_samples]),
                "quality_threshold": self.config.quality_threshold
            }
        }
        
        metadata_path = self.config.output_directory / "pipeline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Pipeline metadata saved to {metadata_path}")

# Factory function for creating configured pipeline
def create_data_pipeline(
    input_directory: str,
    output_directory: str,
    cache_directory: Optional[str] = None,
    **kwargs
) -> DataPipeline:
    """Create a configured data pipeline."""
    config = PipelineConfig(
        input_directory=Path(input_directory),
        output_directory=Path(output_directory),
        cache_directory=Path(cache_directory) if cache_directory else None,
        **kwargs
    )
    
    return DataPipeline(config)