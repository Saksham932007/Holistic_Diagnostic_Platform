"""
Advanced DICOM validation utilities for medical imaging platform.

This module provides comprehensive DICOM validation beyond basic format checks,
including clinical validation, integrity verification, and compliance checks.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path
from datetime import datetime
import re
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag
import numpy as np

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class DICOMValidationError(Exception):
    """Custom exception for DICOM validation errors."""
    pass


class DICOMValidator:
    """
    Comprehensive DICOM validation for medical imaging platform.
    
    This validator performs multi-level validation including:
    - Format validation
    - Clinical validation
    - Integrity checks
    - Compliance verification
    """
    
    # Critical DICOM tags that must be present
    CRITICAL_TAGS = {
        (0x0008, 0x0016): "SOPClassUID",
        (0x0008, 0x0018): "SOPInstanceUID", 
        (0x0008, 0x0060): "Modality",
        (0x0010, 0x0020): "PatientID",
        (0x0020, 0x000D): "StudyInstanceUID",
        (0x0020, 0x000E): "SeriesInstanceUID",
        (0x7FE0, 0x0010): "PixelData"
    }
    
    # Modality-specific required tags
    MODALITY_REQUIRED_TAGS = {
        "CT": [
            (0x0018, 0x0050),  # SliceThickness
            (0x0018, 0x0060),  # KVP
            (0x0018, 0x1151),  # XRayTubeCurrent
            (0x0018, 0x1152),  # Exposure
        ],
        "MR": [
            (0x0018, 0x0050),  # SliceThickness
            (0x0018, 0x0080),  # RepetitionTime
            (0x0018, 0x0081),  # EchoTime
            (0x0018, 0x0087),  # MagneticFieldStrength
        ],
        "US": [
            (0x0018, 0x6011),  # SequenceOfUltrasoundRegions
        ],
        "PT": [
            (0x0054, 0x0016),  # RadiopharmaceuticalInformationSequence
            (0x0018, 0x1074),  # RadionuclideTotalDose
        ]
    }
    
    # Valid SOP Class UIDs for medical imaging
    VALID_SOP_CLASSES = {
        "1.2.840.10008.5.1.4.1.1.2": "CT Image Storage",
        "1.2.840.10008.5.1.4.1.1.4": "MR Image Storage", 
        "1.2.840.10008.5.1.4.1.1.6.1": "Ultrasound Image Storage",
        "1.2.840.10008.5.1.4.1.1.128": "Positron Emission Tomography Image Storage",
        "1.2.840.10008.5.1.4.1.1.481.1": "RT Image Storage",
        "1.2.840.10008.5.1.4.1.1.481.3": "RT Structure Set Storage"
    }
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize DICOM validator.
        
        Args:
            strict_mode: Enable strict validation (recommended for production)
        """
        self._config = get_config()
        self._strict_mode = strict_mode
        self._validation_cache: Dict[str, Dict[str, Any]] = {}
    
    def validate_dicom_file(
        self, 
        file_path: Path,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of a DICOM file.
        
        Args:
            file_path: Path to DICOM file
            session_id: Session identifier for audit logging
            user_id: User identifier for audit logging
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'file_path': str(file_path),
            'is_valid': False,
            'errors': [],
            'warnings': [],
            'info': {},
            'validation_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check if file exists
            if not file_path.exists():
                raise DICOMValidationError(f"File does not exist: {file_path}")
            
            # Load DICOM dataset
            dataset = pydicom.dcmread(str(file_path), force=True)
            
            # Perform validation levels
            self._validate_format(dataset, validation_results)
            self._validate_critical_tags(dataset, validation_results)
            self._validate_modality_specific(dataset, validation_results)
            self._validate_clinical_data(dataset, validation_results)
            self._validate_image_data(dataset, validation_results)
            self._validate_compliance(dataset, validation_results)
            
            # Determine overall validity
            validation_results['is_valid'] = len(validation_results['errors']) == 0
            
            # Log validation event
            log_audit_event(
                event_type=AuditEventType.DICOM_READ,
                severity=AuditSeverity.INFO if validation_results['is_valid'] else AuditSeverity.WARNING,
                message=f"DICOM validation completed: {'PASS' if validation_results['is_valid'] else 'FAIL'}",
                user_id=user_id,
                session_id=session_id,
                study_uid=getattr(dataset, 'StudyInstanceUID', None),
                patient_id=getattr(dataset, 'PatientID', None),
                additional_data={
                    'file_path': str(file_path),
                    'error_count': len(validation_results['errors']),
                    'warning_count': len(validation_results['warnings'])
                }
            )
            
        except Exception as e:
            error_msg = f"DICOM validation failed: {str(e)}"
            validation_results['errors'].append(error_msg)
            
            log_audit_event(
                event_type=AuditEventType.DICOM_READ,
                severity=AuditSeverity.ERROR,
                message=error_msg,
                user_id=user_id,
                session_id=session_id,
                additional_data={
                    'file_path': str(file_path),
                    'error': str(e)
                }
            )
        
        return validation_results
    
    def _validate_format(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """
        Validate DICOM format and structure.
        
        Args:
            dataset: DICOM dataset
            results: Results dictionary to update
        """
        try:
            # Check if it's a valid DICOM file
            if not hasattr(dataset, 'file_meta'):
                results['warnings'].append("Missing DICOM file meta information")
            
            # Validate transfer syntax
            if hasattr(dataset, 'file_meta') and hasattr(dataset.file_meta, 'TransferSyntaxUID'):
                transfer_syntax = dataset.file_meta.TransferSyntaxUID
                results['info']['transfer_syntax'] = str(transfer_syntax)
            else:
                results['warnings'].append("Missing Transfer Syntax UID")
            
            # Check for private tags (may indicate proprietary modifications)
            private_tags = []
            for tag in dataset.keys():
                if tag.group % 2 == 1:  # Odd group numbers are private
                    private_tags.append(str(tag))
            
            if private_tags:
                results['info']['private_tags_count'] = len(private_tags)
                if len(private_tags) > 10:  # Arbitrary threshold
                    results['warnings'].append(f"High number of private tags: {len(private_tags)}")
            
        except Exception as e:
            results['errors'].append(f"Format validation error: {str(e)}")
    
    def _validate_critical_tags(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """
        Validate presence of critical DICOM tags.
        
        Args:
            dataset: DICOM dataset
            results: Results dictionary to update
        """
        missing_critical = []
        
        for tag_tuple, tag_name in self.CRITICAL_TAGS.items():
            tag = Tag(tag_tuple)
            if tag not in dataset:
                missing_critical.append(tag_name)
            else:
                # Store key information
                value = getattr(dataset, tag_name, None)
                if value is not None:
                    results['info'][tag_name] = str(value)
        
        if missing_critical:
            results['errors'].append(f"Missing critical tags: {missing_critical}")
        
        # Validate SOP Class UID
        if hasattr(dataset, 'SOPClassUID'):
            sop_class = str(dataset.SOPClassUID)
            if sop_class not in self.VALID_SOP_CLASSES:
                results['warnings'].append(f"Unknown SOP Class UID: {sop_class}")
            else:
                results['info']['sop_class_name'] = self.VALID_SOP_CLASSES[sop_class]
    
    def _validate_modality_specific(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """
        Validate modality-specific requirements.
        
        Args:
            dataset: DICOM dataset
            results: Results dictionary to update
        """
        if not hasattr(dataset, 'Modality'):
            return
        
        modality = str(dataset.Modality)
        results['info']['modality'] = modality
        
        # Check if modality is allowed
        if modality not in self._config.data.allowed_modalities:
            results['errors'].append(f"Unsupported modality: {modality}")
            return
        
        # Check modality-specific required tags
        if modality in self.MODALITY_REQUIRED_TAGS:
            missing_tags = []
            for tag_tuple in self.MODALITY_REQUIRED_TAGS[modality]:
                tag = Tag(tag_tuple)
                if tag not in dataset:
                    tag_name = f"({tag.group:04X},{tag.element:04X})"
                    missing_tags.append(tag_name)
            
            if missing_tags and self._strict_mode:
                results['warnings'].append(f"Missing {modality}-specific tags: {missing_tags}")
        
        # Modality-specific validation
        if modality == "CT":
            self._validate_ct_specific(dataset, results)
        elif modality == "MR":
            self._validate_mr_specific(dataset, results)
        elif modality == "US":
            self._validate_us_specific(dataset, results)
        elif modality == "PT":
            self._validate_pt_specific(dataset, results)
    
    def _validate_ct_specific(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """Validate CT-specific parameters."""
        # Check slice thickness
        if hasattr(dataset, 'SliceThickness'):
            thickness = float(dataset.SliceThickness)
            results['info']['slice_thickness'] = thickness
            
            if (thickness < self._config.data.min_slice_thickness or
                thickness > self._config.data.max_slice_thickness):
                results['warnings'].append(f"Slice thickness outside valid range: {thickness}mm")
        
        # Check radiation dose parameters
        if hasattr(dataset, 'CTDIvol'):
            ctdi = float(dataset.CTDIvol)
            results['info']['ctdi_vol'] = ctdi
            if ctdi > 100:  # Arbitrary high dose threshold
                results['warnings'].append(f"High radiation dose detected: {ctdi} mGy")
    
    def _validate_mr_specific(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """Validate MR-specific parameters."""
        # Check field strength
        if hasattr(dataset, 'MagneticFieldStrength'):
            field_strength = float(dataset.MagneticFieldStrength)
            results['info']['magnetic_field_strength'] = field_strength
            
            if field_strength > 7.0:  # Tesla
                results['warnings'].append(f"High field strength: {field_strength}T")
        
        # Check sequence parameters
        if hasattr(dataset, 'RepetitionTime'):
            tr = float(dataset.RepetitionTime)
            results['info']['repetition_time'] = tr
        
        if hasattr(dataset, 'EchoTime'):
            te = float(dataset.EchoTime)
            results['info']['echo_time'] = te
    
    def _validate_us_specific(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """Validate Ultrasound-specific parameters."""
        # Check for ultrasound regions
        if hasattr(dataset, 'SequenceOfUltrasoundRegions'):
            regions = dataset.SequenceOfUltrasoundRegions
            results['info']['ultrasound_regions_count'] = len(regions)
    
    def _validate_pt_specific(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """Validate PET-specific parameters."""
        # Check radiopharmaceutical information
        if hasattr(dataset, 'RadiopharmaceuticalInformationSequence'):
            radio_seq = dataset.RadiopharmaceuticalInformationSequence
            if radio_seq:
                radio_info = radio_seq[0]
                if hasattr(radio_info, 'Radiopharmaceutical'):
                    results['info']['radiopharmaceutical'] = str(radio_info.Radiopharmaceutical)
    
    def _validate_clinical_data(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """
        Validate clinical data consistency and reasonableness.
        
        Args:
            dataset: DICOM dataset
            results: Results dictionary to update
        """
        # Validate patient age if present
        if hasattr(dataset, 'PatientAge'):
            age_str = str(dataset.PatientAge)
            if not self._validate_patient_age(age_str):
                results['warnings'].append(f"Invalid patient age format: {age_str}")
            else:
                results['info']['patient_age'] = age_str
        
        # Validate patient sex
        if hasattr(dataset, 'PatientSex'):
            sex = str(dataset.PatientSex).upper()
            if sex not in ['M', 'F', 'O']:  # Male, Female, Other
                results['warnings'].append(f"Invalid patient sex: {sex}")
            else:
                results['info']['patient_sex'] = sex
        
        # Validate study date
        if hasattr(dataset, 'StudyDate'):
            study_date = str(dataset.StudyDate)
            if not self._validate_dicom_date(study_date):
                results['warnings'].append(f"Invalid study date format: {study_date}")
            else:
                results['info']['study_date'] = study_date
                
                # Check if date is reasonable (not in future, not too old)
                try:
                    date_obj = datetime.strptime(study_date, "%Y%m%d")
                    today = datetime.now()
                    if date_obj > today:
                        results['warnings'].append("Study date is in the future")
                    elif (today - date_obj).days > 36500:  # 100 years
                        results['warnings'].append("Study date is very old")
                except ValueError:
                    pass
        
        # Validate UID formats
        uid_tags = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
        for uid_tag in uid_tags:
            if hasattr(dataset, uid_tag):
                uid_value = str(getattr(dataset, uid_tag))
                if not self._validate_uid_format(uid_value):
                    results['warnings'].append(f"Invalid {uid_tag} format: {uid_value}")
    
    def _validate_image_data(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """
        Validate image pixel data.
        
        Args:
            dataset: DICOM dataset
            results: Results dictionary to update
        """
        try:
            if hasattr(dataset, 'PixelData'):
                # Get pixel array
                pixel_array = dataset.pixel_array
                
                # Store image information
                results['info']['image_shape'] = list(pixel_array.shape)
                results['info']['pixel_data_size'] = pixel_array.nbytes
                results['info']['data_type'] = str(pixel_array.dtype)
                
                # Validate image dimensions
                if len(pixel_array.shape) < 2:
                    results['errors'].append("Image must have at least 2 dimensions")
                
                # Check for reasonable image size
                total_pixels = np.prod(pixel_array.shape)
                if total_pixels < 1000:  # Very small image
                    results['warnings'].append(f"Very small image: {total_pixels} pixels")
                elif total_pixels > 500_000_000:  # Very large image
                    results['warnings'].append(f"Very large image: {total_pixels} pixels")
                
                # Check pixel value statistics
                if pixel_array.size > 0:
                    min_val = float(np.min(pixel_array))
                    max_val = float(np.max(pixel_array))
                    mean_val = float(np.mean(pixel_array))
                    
                    results['info']['pixel_stats'] = {
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val
                    }
                    
                    # Check for suspicious pixel values
                    if min_val == max_val:
                        results['warnings'].append("Image has constant pixel values")
                
                # Validate DICOM header consistency with pixel data
                if hasattr(dataset, 'Rows') and hasattr(dataset, 'Columns'):
                    header_rows = int(dataset.Rows)
                    header_cols = int(dataset.Columns)
                    
                    if len(pixel_array.shape) >= 2:
                        actual_rows = pixel_array.shape[0]
                        actual_cols = pixel_array.shape[1]
                        
                        if header_rows != actual_rows or header_cols != actual_cols:
                            results['errors'].append(
                                f"Image dimensions mismatch: header({header_rows}x{header_cols}) "
                                f"vs actual({actual_rows}x{actual_cols})"
                            )
                
        except Exception as e:
            results['warnings'].append(f"Could not validate pixel data: {str(e)}")
    
    def _validate_compliance(self, dataset: Dataset, results: Dict[str, Any]) -> None:
        """
        Validate HIPAA and other compliance requirements.
        
        Args:
            dataset: DICOM dataset
            results: Results dictionary to update
        """
        # Check for potential PHI in text fields
        phi_tags = [
            'PatientName', 'PatientAddress', 'ReferringPhysicianName',
            'InstitutionName', 'StudyDescription', 'SeriesDescription'
        ]
        
        phi_found = []
        for tag_name in phi_tags:
            if hasattr(dataset, tag_name):
                value = str(getattr(dataset, tag_name))
                if self._contains_potential_phi(value):
                    phi_found.append(tag_name)
        
        if phi_found and self._strict_mode:
            results['warnings'].append(f"Potential PHI found in: {phi_found}")
        
        # Check for de-identification markers
        if hasattr(dataset, 'PatientIdentityRemoved'):
            if str(dataset.PatientIdentityRemoved) == "YES":
                results['info']['deidentified'] = True
            else:
                results['info']['deidentified'] = False
                if self._strict_mode:
                    results['warnings'].append("File not marked as de-identified")
        else:
            results['info']['deidentified'] = False
            if self._strict_mode:
                results['warnings'].append("Missing de-identification marker")
    
    def _validate_patient_age(self, age_str: str) -> bool:
        """Validate patient age format (e.g., '025Y', '003M', '120D')."""
        pattern = r'^\d{3}[YMWD]$'
        return bool(re.match(pattern, age_str))
    
    def _validate_dicom_date(self, date_str: str) -> bool:
        """Validate DICOM date format (YYYYMMDD)."""
        pattern = r'^\d{8}$'
        if not re.match(pattern, date_str):
            return False
        
        try:
            datetime.strptime(date_str, "%Y%m%d")
            return True
        except ValueError:
            return False
    
    def _validate_uid_format(self, uid: str) -> bool:
        """Validate DICOM UID format."""
        # UID should contain only digits and dots, start and end with digit
        pattern = r'^[0-9]+(\.[0-9]+)*$'
        if not re.match(pattern, uid):
            return False
        
        # Check length (max 64 characters)
        if len(uid) > 64:
            return False
        
        # Check for reasonable structure
        parts = uid.split('.')
        if len(parts) < 3:  # UIDs should have at least 3 parts
            return False
        
        return True
    
    def _contains_potential_phi(self, text: str) -> bool:
        """Check if text contains potential PHI patterns."""
        if not text:
            return False
        
        # Common PHI patterns
        phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,5}\s+\w+\s+(St|Street|Ave|Avenue|Rd|Road|Dr|Drive)\b',  # Address
        ]
        
        for pattern in phi_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def validate_batch(
        self,
        file_paths: List[Path],
        max_workers: int = 4,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a batch of DICOM files.
        
        Args:
            file_paths: List of DICOM file paths
            max_workers: Maximum number of worker threads
            session_id: Session identifier for audit logging
            user_id: User identifier for audit logging
            
        Returns:
            Batch validation results
        """
        from concurrent.futures import ThreadPoolExecutor
        
        batch_results = {
            'total_files': len(file_paths),
            'valid_files': 0,
            'invalid_files': 0,
            'validation_results': [],
            'summary': {
                'common_errors': {},
                'common_warnings': {},
                'modalities': {}
            }
        }
        
        def validate_single(file_path):
            return self.validate_dicom_file(file_path, session_id, user_id)
        
        # Use thread pool for parallel validation
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(validate_single, file_paths))
        
        # Process results
        for result in results:
            batch_results['validation_results'].append(result)
            
            if result['is_valid']:
                batch_results['valid_files'] += 1
            else:
                batch_results['invalid_files'] += 1
            
            # Collect summary statistics
            for error in result['errors']:
                batch_results['summary']['common_errors'][error] = \
                    batch_results['summary']['common_errors'].get(error, 0) + 1
            
            for warning in result['warnings']:
                batch_results['summary']['common_warnings'][warning] = \
                    batch_results['summary']['common_warnings'].get(warning, 0) + 1
            
            if 'modality' in result['info']:
                modality = result['info']['modality']
                batch_results['summary']['modalities'][modality] = \
                    batch_results['summary']['modalities'].get(modality, 0) + 1
        
        log_audit_event(
            event_type=AuditEventType.IMAGE_PROCESSING,
            severity=AuditSeverity.INFO,
            message=f"Batch validation completed: {batch_results['valid_files']}/{batch_results['total_files']} valid",
            user_id=user_id,
            session_id=session_id,
            additional_data={
                'total_files': batch_results['total_files'],
                'valid_files': batch_results['valid_files'],
                'invalid_files': batch_results['invalid_files']
            }
        )
        
        return batch_results