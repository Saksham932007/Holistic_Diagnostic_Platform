"""
Medical image data loading utilities using MONAI.

This module provides HIPAA-compliant data loading capabilities for medical
imaging with comprehensive validation and audit logging.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from pydicom.errors import InvalidDicomError
import nibabel as nib

from monai.data import Dataset as MONAIDataset
from monai.data import DataLoader as MONAIDataLoader
from monai.data import CacheDataset, PersistentDataset
from monai.data.utils import list_data_collate
from monai.transforms import Compose, Transform

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event
from .deid import DICOMDeidentifier


class MedicalImageDataset(Dataset):
    """
    Base dataset class for medical imaging with HIPAA compliance features.
    
    This class provides standardized loading of medical images with
    comprehensive audit logging and validation.
    """
    
    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        transforms: Optional[Callable] = None,
        cache_rate: float = 0.0,
        num_workers: int = 0,
        validate_data: bool = True,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize the medical image dataset.
        
        Args:
            data_list: List of dictionaries containing file paths and metadata
            transforms: MONAI transforms to apply
            cache_rate: Fraction of dataset to cache in memory (0.0 to 1.0)
            num_workers: Number of worker processes for data loading
            validate_data: Whether to validate data integrity
            session_id: Session identifier for audit logging
            user_id: User identifier for audit logging
        """
        self._config = get_config()
        self._session_id = session_id
        self._user_id = user_id
        self._transforms = transforms
        self._validate_data = validate_data
        
        # Validate and process data list
        self._data_list = self._process_data_list(data_list)
        
        # Initialize MONAI dataset based on cache rate
        if cache_rate > 0.0:
            self._dataset = CacheDataset(
                data=self._data_list,
                transform=transforms,
                cache_rate=cache_rate,
                num_workers=num_workers
            )
        else:
            self._dataset = MONAIDataset(
                data=self._data_list,
                transform=transforms
            )
        
        # Log dataset initialization
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message=f"Medical image dataset initialized with {len(self._data_list)} items",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'dataset_size': len(self._data_list),
                'cache_rate': cache_rate,
                'validation_enabled': validate_data
            }
        )
    
    def _process_data_list(self, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and validate the input data list.
        
        Args:
            data_list: Raw data list from user
            
        Returns:
            Processed and validated data list
        """
        processed_data = []
        invalid_entries = []
        
        for idx, data_dict in enumerate(data_list):
            try:
                # Validate required fields
                if 'image' not in data_dict:
                    raise ValueError(f"Missing 'image' field in data entry {idx}")
                
                # Validate file existence
                image_path = Path(data_dict['image'])
                if not image_path.exists():
                    raise ValueError(f"Image file does not exist: {image_path}")
                
                # Additional validation if enabled
                if self._validate_data:
                    self._validate_medical_image(image_path)
                
                # Add metadata
                processed_entry = data_dict.copy()
                processed_entry['_index'] = idx
                processed_entry['_validated'] = True
                
                processed_data.append(processed_entry)
                
            except Exception as e:
                invalid_entries.append({
                    'index': idx,
                    'data': data_dict,
                    'error': str(e)
                })
                
                log_audit_event(
                    event_type=AuditEventType.IMAGE_PROCESSING,
                    severity=AuditSeverity.WARNING,
                    message=f"Invalid data entry: {str(e)}",
                    user_id=self._user_id,
                    session_id=self._session_id,
                    additional_data={
                        'entry_index': idx,
                        'error': str(e)
                    }
                )
        
        if invalid_entries:
            log_audit_event(
                event_type=AuditEventType.IMAGE_PROCESSING,
                severity=AuditSeverity.WARNING,
                message=f"Found {len(invalid_entries)} invalid data entries",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'invalid_count': len(invalid_entries),
                    'valid_count': len(processed_data)
                }
            )
        
        return processed_data
    
    def _validate_medical_image(self, image_path: Path) -> None:
        """
        Validate a medical image file.
        
        Args:
            image_path: Path to the medical image file
            
        Raises:
            ValueError: If the image is invalid
        """
        file_extension = image_path.suffix.lower()
        
        try:
            if file_extension in ['.dcm', '.dicom']:
                self._validate_dicom(image_path)
            elif file_extension in ['.nii', '.nii.gz']:
                self._validate_nifti(image_path)
            else:
                # Try to load as general image
                import nibabel as nib
                nib.load(str(image_path))
                
        except Exception as e:
            raise ValueError(f"Invalid medical image {image_path}: {str(e)}")
    
    def _validate_dicom(self, dicom_path: Path) -> None:
        """
        Validate a DICOM file.
        
        Args:
            dicom_path: Path to DICOM file
            
        Raises:
            ValueError: If DICOM is invalid
        """
        try:
            dataset = pydicom.dcmread(str(dicom_path))
            
            # Check required DICOM tags
            required_tags = self._config.data.required_tags
            missing_tags = []
            
            for tag_name in required_tags:
                if not hasattr(dataset, tag_name):
                    missing_tags.append(tag_name)
            
            if missing_tags:
                raise ValueError(f"Missing required DICOM tags: {missing_tags}")
            
            # Validate modality if specified
            if (hasattr(dataset, 'Modality') and 
                dataset.Modality not in self._config.data.allowed_modalities):
                raise ValueError(f"Unsupported modality: {dataset.Modality}")
            
            # Validate slice thickness if present
            if hasattr(dataset, 'SliceThickness'):
                thickness = float(dataset.SliceThickness)
                if (thickness < self._config.data.min_slice_thickness or
                    thickness > self._config.data.max_slice_thickness):
                    raise ValueError(f"Invalid slice thickness: {thickness}mm")
            
        except InvalidDicomError as e:
            raise ValueError(f"Invalid DICOM format: {str(e)}")
    
    def _validate_nifti(self, nifti_path: Path) -> None:
        """
        Validate a NIfTI file.
        
        Args:
            nifti_path: Path to NIfTI file
            
        Raises:
            ValueError: If NIfTI is invalid
        """
        try:
            img = nib.load(str(nifti_path))
            
            # Validate header
            header = img.header
            if header is None:
                raise ValueError("Missing NIfTI header")
            
            # Check dimensions
            shape = img.shape
            if len(shape) < 3:
                raise ValueError(f"Invalid dimensions: {shape}")
            
            # Check data type
            if img.get_fdata().dtype not in [np.float32, np.float64, np.int16, np.int32]:
                raise ValueError(f"Unsupported data type: {img.get_fdata().dtype}")
            
        except Exception as e:
            raise ValueError(f"Invalid NIfTI format: {str(e)}")
    
    def __len__(self) -> int:
        """Get the dataset size."""
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a data item by index with audit logging.
        
        Args:
            index: Data index
            
        Returns:
            Data dictionary for the specified index
        """
        try:
            # Log data access
            log_audit_event(
                event_type=AuditEventType.DICOM_READ,
                severity=AuditSeverity.INFO,
                message=f"Accessing data item {index}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'data_index': index}
            )
            
            return self._dataset[index]
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.DICOM_READ,
                severity=AuditSeverity.ERROR,
                message=f"Error accessing data item {index}: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'data_index': index, 'error': str(e)}
            )
            raise
    
    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_items': len(self._data_list),
            'modalities': {},
            'file_types': {},
            'validation_enabled': self._validate_data
        }
        
        for item in self._data_list:
            # Count file types
            image_path = Path(item['image'])
            file_ext = image_path.suffix.lower()
            stats['file_types'][file_ext] = stats['file_types'].get(file_ext, 0) + 1
            
            # Count modalities if available
            if 'modality' in item:
                modality = item['modality']
                stats['modalities'][modality] = stats['modalities'].get(modality, 0) + 1
        
        return stats


class MultimodalMedicalDataset(MedicalImageDataset):
    """
    Dataset for multimodal medical data (imaging + EHR/clinical data).
    
    This dataset handles paired data consisting of medical images and
    corresponding clinical/EHR data in JSON format.
    """
    
    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        transforms: Optional[Callable] = None,
        text_tokenizer: Optional[Callable] = None,
        max_text_length: int = 512,
        **kwargs
    ):
        """
        Initialize multimodal dataset.
        
        Args:
            data_list: List with 'image' and 'text_data' or 'ehr_data' keys
            transforms: Image transforms
            text_tokenizer: Text tokenizer for clinical notes
            max_text_length: Maximum text sequence length
            **kwargs: Additional arguments for parent class
        """
        self._text_tokenizer = text_tokenizer
        self._max_text_length = max_text_length
        
        # Validate multimodal data
        self._validate_multimodal_data(data_list)
        
        super().__init__(data_list, transforms, **kwargs)
    
    def _validate_multimodal_data(self, data_list: List[Dict[str, Any]]) -> None:
        """
        Validate multimodal data list.
        
        Args:
            data_list: Data list to validate
        """
        for idx, item in enumerate(data_list):
            if 'image' not in item:
                raise ValueError(f"Missing 'image' key in item {idx}")
            
            # Check for clinical data
            has_text = 'text_data' in item or 'clinical_notes' in item
            has_ehr = 'ehr_data' in item
            
            if not (has_text or has_ehr):
                raise ValueError(f"Missing clinical data in item {idx}")
    
    def _load_clinical_data(self, clinical_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load clinical data from JSON file.
        
        Args:
            clinical_path: Path to clinical data JSON file
            
        Returns:
            Clinical data dictionary
        """
        try:
            with open(clinical_path, 'r', encoding='utf-8') as f:
                clinical_data = json.load(f)
            
            return clinical_data
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.PATIENT_DATA_QUERY,
                severity=AuditSeverity.ERROR,
                message=f"Error loading clinical data: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'clinical_path': str(clinical_path)}
            )
            raise
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get multimodal data item.
        
        Args:
            index: Data index
            
        Returns:
            Dictionary containing image and clinical data
        """
        # Get base data
        data = super().__getitem__(index)
        
        # Load clinical data if available
        original_item = self._data_list[index]
        
        if 'ehr_data' in original_item:
            ehr_path = original_item['ehr_data']
            clinical_data = self._load_clinical_data(ehr_path)
            data['ehr_data'] = clinical_data
        
        if 'text_data' in original_item or 'clinical_notes' in original_item:
            text_key = 'text_data' if 'text_data' in original_item else 'clinical_notes'
            text_data = original_item[text_key]
            
            # Tokenize text if tokenizer is available
            if self._text_tokenizer is not None:
                tokenized = self._text_tokenizer(
                    text_data,
                    max_length=self._max_text_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                data['text_tokens'] = tokenized
            else:
                data['text_data'] = text_data
        
        return data


def create_medical_dataloader(
    data_list: List[Dict[str, Any]],
    transforms: Optional[Callable] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    cache_rate: float = 0.0,
    multimodal: bool = False,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create a medical image data loader with HIPAA compliance.
    
    Args:
        data_list: List of data dictionaries
        transforms: MONAI transforms to apply
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        cache_rate: Fraction of data to cache
        multimodal: Whether to use multimodal dataset
        session_id: Session identifier for audit logging
        user_id: User identifier for audit logging
        **kwargs: Additional arguments
        
    Returns:
        DataLoader instance
    """
    # Choose dataset class
    if multimodal:
        dataset = MultimodalMedicalDataset(
            data_list=data_list,
            transforms=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
            session_id=session_id,
            user_id=user_id,
            **kwargs
        )
    else:
        dataset = MedicalImageDataset(
            data_list=data_list,
            transforms=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
            session_id=session_id,
            user_id=user_id
        )
    
    # Create data loader
    dataloader = MONAIDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available()
    )
    
    # Log data loader creation
    log_audit_event(
        event_type=AuditEventType.SYSTEM_START,
        severity=AuditSeverity.INFO,
        message="Medical data loader created",
        user_id=user_id,
        session_id=session_id,
        additional_data={
            'batch_size': batch_size,
            'dataset_size': len(dataset),
            'multimodal': multimodal,
            'cache_rate': cache_rate
        }
    )
    
    return dataloader


def load_data_list_from_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load data list from a JSON file.
    
    Args:
        json_path: Path to JSON file containing data list
        
    Returns:
        List of data dictionaries
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        
        if not isinstance(data_list, list):
            raise ValueError("JSON file must contain a list of data dictionaries")
        
        return data_list
        
    except Exception as e:
        log_audit_event(
            event_type=AuditEventType.PATIENT_DATA_QUERY,
            severity=AuditSeverity.ERROR,
            message=f"Error loading data list from JSON: {str(e)}",
            additional_data={'json_path': str(json_path)}
        )
        raise


def create_train_val_split(
    data_list: List[Dict[str, Any]],
    val_fraction: float = 0.2,
    stratify_key: Optional[str] = None,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create train/validation split with optional stratification.
    
    Args:
        data_list: Complete data list
        val_fraction: Fraction of data for validation (0.0 to 1.0)
        stratify_key: Key to use for stratified splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data)
    """
    import random
    from collections import defaultdict
    
    random.seed(random_seed)
    
    if stratify_key is None:
        # Simple random split
        data_copy = data_list.copy()
        random.shuffle(data_copy)
        
        val_size = int(len(data_copy) * val_fraction)
        val_data = data_copy[:val_size]
        train_data = data_copy[val_size:]
        
    else:
        # Stratified split
        strata = defaultdict(list)
        
        # Group by stratification key
        for item in data_list:
            if stratify_key in item:
                strata[item[stratify_key]].append(item)
            else:
                strata['unknown'].append(item)
        
        train_data = []
        val_data = []
        
        # Split each stratum
        for stratum_data in strata.values():
            random.shuffle(stratum_data)
            val_size = int(len(stratum_data) * val_fraction)
            val_data.extend(stratum_data[:val_size])
            train_data.extend(stratum_data[val_size:])
    
    log_audit_event(
        event_type=AuditEventType.SYSTEM_START,
        severity=AuditSeverity.INFO,
        message="Train/validation split created",
        additional_data={
            'total_samples': len(data_list),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'val_fraction': val_fraction,
            'stratify_key': stratify_key
        }
    )
    
    return train_data, val_data