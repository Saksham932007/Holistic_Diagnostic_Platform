"""
DICOM de-identification utilities for HIPAA compliance.

This module provides comprehensive DICOM de-identification capabilities
to remove or anonymize Protected Health Information (PHI) according to
HIPAA Safe Harbor de-identification standards.
"""

import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.tag import Tag
from pydicom.errors import InvalidDicomError

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class DICOMDeidentifier:
    """
    HIPAA-compliant DICOM de-identification processor.
    
    This class implements the HIPAA Safe Harbor de-identification method
    by removing or anonymizing all 18 types of identifiers specified
    in 45 CFR 164.514(b)(2).
    """
    
    # HIPAA Safe Harbor identifiers (DICOM tags to remove/anonymize)
    PHI_TAGS = {
        # Patient Information
        (0x0010, 0x0010): "PatientName",
        (0x0010, 0x0020): "PatientID", 
        (0x0010, 0x0030): "PatientBirthDate",
        (0x0010, 0x0040): "PatientSex",
        (0x0010, 0x1000): "OtherPatientIDs",
        (0x0010, 0x1001): "OtherPatientNames",
        (0x0010, 0x1010): "PatientAge",
        (0x0010, 0x1020): "PatientSize",
        (0x0010, 0x1030): "PatientWeight",
        (0x0010, 0x1040): "PatientAddress",
        (0x0010, 0x1050): "InsurancePlanIdentification",
        (0x0010, 0x1060): "PatientMotherBirthName",
        (0x0010, 0x2154): "PatientTelephoneNumbers",
        (0x0010, 0x4000): "PatientComments",
        
        # Study Information
        (0x0008, 0x0020): "StudyDate",
        (0x0008, 0x0021): "SeriesDate", 
        (0x0008, 0x0022): "AcquisitionDate",
        (0x0008, 0x0023): "ContentDate",
        (0x0008, 0x0030): "StudyTime",
        (0x0008, 0x0031): "SeriesTime",
        (0x0008, 0x0032): "AcquisitionTime",
        (0x0008, 0x0033): "ContentTime",
        (0x0008, 0x0050): "AccessionNumber",
        (0x0008, 0x0080): "InstitutionName",
        (0x0008, 0x0081): "InstitutionAddress",
        (0x0008, 0x0090): "ReferringPhysicianName",
        (0x0008, 0x0092): "ReferringPhysicianAddress",
        (0x0008, 0x0094): "ReferringPhysicianTelephoneNumbers",
        (0x0008, 0x1010): "StationName",
        (0x0008, 0x1030): "StudyDescription",
        (0x0008, 0x103E): "SeriesDescription",
        (0x0008, 0x1040): "InstitutionalDepartmentName",
        (0x0008, 0x1048): "PhysiciansOfRecord",
        (0x0008, 0x1050): "PerformingPhysicianName",
        (0x0008, 0x1060): "NameOfPhysiciansReadingStudy",
        (0x0008, 0x1070): "OperatorsName",
        (0x0008, 0x1080): "AdmittingDiagnosesDescription",
        (0x0008, 0x1155): "ReferencedSOPInstanceUID",
        (0x0008, 0x2111): "DerivationDescription",
        
        # Equipment Information (contains potential identifiers)
        (0x0008, 0x1010): "StationName",
        (0x0018, 0x1000): "DeviceSerialNumber",
        (0x0018, 0x1020): "SoftwareVersions",
        
        # Other potential identifiers
        (0x0020, 0x4000): "ImageComments",
        (0x4008, 0x0300): "MedicalRecordLocator",
    }
    
    # Tags to keep but with shifted dates
    DATE_SHIFT_TAGS = {
        (0x0008, 0x0020): "StudyDate",
        (0x0008, 0x0021): "SeriesDate",
        (0x0008, 0x0022): "AcquisitionDate", 
        (0x0008, 0x0023): "ContentDate",
    }
    
    # Tags that require special handling
    UID_TAGS = {
        (0x0008, 0x0018): "SOPInstanceUID",
        (0x0020, 0x000D): "StudyInstanceUID", 
        (0x0020, 0x000E): "SeriesInstanceUID",
        (0x0020, 0x0052): "FrameOfReferenceUID",
    }
    
    def __init__(self, date_shift_days: int = None, preserve_age: bool = True):
        """
        Initialize the DICOM de-identifier.
        
        Args:
            date_shift_days: Number of days to shift dates (random if None)
            preserve_age: Whether to preserve patient age ranges
        """
        self._config = get_config()
        self._date_shift_days = date_shift_days or self._generate_date_shift()
        self._preserve_age = preserve_age
        
        # Mapping for consistent anonymization
        self._patient_id_mapping: Dict[str, str] = {}
        self._uid_mapping: Dict[str, str] = {}
        self._study_mapping: Dict[str, str] = {}
        
    def _generate_date_shift(self) -> int:
        """Generate a random date shift between -365 and +365 days."""
        import random
        return random.randint(-365, 365)
    
    def _generate_anonymous_id(self, original_id: str, prefix: str = "ANON") -> str:
        """
        Generate a consistent anonymous ID for the given original ID.
        
        Args:
            original_id: Original identifier
            prefix: Prefix for the anonymous ID
            
        Returns:
            Consistent anonymous identifier
        """
        # Use SHA-256 hash for consistency
        hash_object = hashlib.sha256(original_id.encode())
        hash_hex = hash_object.hexdigest()[:8]  # Use first 8 characters
        return f"{prefix}_{hash_hex.upper()}"
    
    def _generate_anonymous_uid(self, original_uid: str, uid_type: str = "study") -> str:
        """
        Generate a consistent anonymous UID.
        
        Args:
            original_uid: Original UID
            uid_type: Type of UID (study, series, instance)
            
        Returns:
            Anonymous UID with valid DICOM UID format
        """
        if original_uid in self._uid_mapping:
            return self._uid_mapping[original_uid]
        
        # Generate hash-based UID
        hash_object = hashlib.sha256(original_uid.encode())
        hash_hex = hash_object.hexdigest()[:16]
        
        # Convert to valid DICOM UID format
        # Use enterprise root 1.2.826.0.1.3680043.9.7147 (example)
        anonymous_uid = f"1.2.826.0.1.3680043.9.7147.{hash_hex}"
        
        self._uid_mapping[original_uid] = anonymous_uid
        return anonymous_uid
    
    def _shift_date(self, date_str: str) -> str:
        """
        Shift a DICOM date by the specified number of days.
        
        Args:
            date_str: DICOM date string (YYYYMMDD)
            
        Returns:
            Shifted date string in DICOM format
        """
        try:
            # Parse DICOM date format (YYYYMMDD)
            original_date = datetime.strptime(date_str, "%Y%m%d")
            shifted_date = original_date + timedelta(days=self._date_shift_days)
            return shifted_date.strftime("%Y%m%d")
        except ValueError:
            # Invalid date format, return empty string
            return ""
    
    def _anonymize_age(self, age_str: str) -> str:
        """
        Anonymize age while preserving general age range.
        
        Args:
            age_str: Original age string (e.g., "025Y", "003M")
            
        Returns:
            Anonymized age string or empty if over 89
        """
        if not age_str or len(age_str) < 3:
            return ""
        
        try:
            age_num = int(age_str[:3])
            age_unit = age_str[3:]
            
            # HIPAA requires ages 90+ to be reported as "90+"
            if age_unit.upper() == 'Y' and age_num >= 90:
                return ""  # Remove ages 90 and above per HIPAA
            elif age_unit.upper() == 'Y' and age_num >= 85:
                # Group ages 85-89 as "85+"
                return "085Y"
            else:
                return age_str  # Keep other ages as-is
                
        except ValueError:
            return ""
    
    def _clean_text_field(self, text: str) -> str:
        """
        Remove potential identifiers from text fields.
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text with identifiers removed
        """
        if not text:
            return ""
        
        # Remove common patterns that might contain identifiers
        patterns_to_remove = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
            r'\b\d{1,5}\s+[A-Za-z\s]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b',  # Address pattern
        ]
        
        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(pattern, '[REMOVED]', cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def deidentify_dataset(self, dataset: Dataset, patient_id: Optional[str] = None) -> Dataset:
        """
        De-identify a DICOM dataset according to HIPAA Safe Harbor rules.
        
        Args:
            dataset: DICOM dataset to de-identify
            patient_id: Optional patient ID for audit logging
            
        Returns:
            De-identified DICOM dataset
        """
        try:
            # Create a copy to avoid modifying the original
            deidentified = dataset.copy()
            
            # Track what was removed for audit logging
            removed_tags = []
            modified_tags = []
            
            # Remove PHI tags
            for tag_tuple, tag_name in self.PHI_TAGS.items():
                tag = Tag(tag_tuple)
                if tag in deidentified:
                    if tag_tuple in self.DATE_SHIFT_TAGS:
                        # Shift dates instead of removing
                        original_date = str(deidentified[tag].value)
                        shifted_date = self._shift_date(original_date)
                        deidentified[tag].value = shifted_date
                        modified_tags.append(f"{tag_name}: date shifted")
                    else:
                        # Remove the tag completely
                        del deidentified[tag]
                        removed_tags.append(tag_name)
            
            # Handle UIDs specially - replace with consistent anonymous UIDs
            for tag_tuple, tag_name in self.UID_TAGS.items():
                tag = Tag(tag_tuple)
                if tag in deidentified:
                    original_uid = str(deidentified[tag].value)
                    anonymous_uid = self._generate_anonymous_uid(original_uid, tag_name.lower())
                    deidentified[tag].value = anonymous_uid
                    modified_tags.append(f"{tag_name}: anonymized")
            
            # Handle Patient ID specially for consistent mapping
            patient_id_tag = Tag((0x0010, 0x0020))
            if patient_id_tag in dataset:
                original_patient_id = str(dataset[patient_id_tag].value)
                if original_patient_id not in self._patient_id_mapping:
                    self._patient_id_mapping[original_patient_id] = self._generate_anonymous_id(
                        original_patient_id, "PATIENT"
                    )
                deidentified[patient_id_tag].value = self._patient_id_mapping[original_patient_id]
                modified_tags.append("PatientID: anonymized")
            
            # Handle Patient Age if present and preserve_age is True
            if self._preserve_age:
                age_tag = Tag((0x0010, 0x1010))
                if age_tag in deidentified:
                    original_age = str(deidentified[age_tag].value)
                    anonymous_age = self._anonymize_age(original_age)
                    if anonymous_age:
                        deidentified[age_tag].value = anonymous_age
                        modified_tags.append("PatientAge: anonymized")
                    else:
                        del deidentified[age_tag]
                        removed_tags.append("PatientAge")
            
            # Clean text fields that might contain identifiers
            text_tags = [
                (0x0008, 0x1030),  # StudyDescription
                (0x0008, 0x103E),  # SeriesDescription  
                (0x0020, 0x4000),  # ImageComments
            ]
            
            for tag_tuple in text_tags:
                tag = Tag(tag_tuple)
                if tag in deidentified and hasattr(deidentified[tag], 'value'):
                    original_text = str(deidentified[tag].value)
                    cleaned_text = self._clean_text_field(original_text)
                    if cleaned_text != original_text:
                        deidentified[tag].value = cleaned_text
                        modified_tags.append(f"{tag}: text cleaned")
            
            # Add de-identification marker
            deidentified.PatientIdentityRemoved = "YES"
            deidentified.DeidentificationMethod = "HIPAA_SAFE_HARBOR_AUTOMATED"
            deidentified.DeidentificationMethodCodeSequence = []
            
            # Log the de-identification event
            log_audit_event(
                event_type=AuditEventType.DE_IDENTIFICATION,
                severity=AuditSeverity.INFO,
                message="DICOM dataset de-identified",
                patient_id=patient_id or "unknown",
                study_uid=getattr(dataset, 'StudyInstanceUID', 'unknown'),
                additional_data={
                    'removed_tags': removed_tags,
                    'modified_tags': modified_tags,
                    'date_shift_days': self._date_shift_days
                }
            )
            
            return deidentified
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.DE_IDENTIFICATION,
                severity=AuditSeverity.ERROR,
                message=f"De-identification failed: {str(e)}",
                patient_id=patient_id or "unknown",
                additional_data={'error': str(e)}
            )
            raise
    
    def deidentify_file(self, input_path: Path, output_path: Path) -> bool:
        """
        De-identify a DICOM file.
        
        Args:
            input_path: Path to the input DICOM file
            output_path: Path for the de-identified output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the DICOM file
            dataset = pydicom.dcmread(str(input_path))
            
            # Extract patient ID for audit logging
            patient_id = getattr(dataset, 'PatientID', 'unknown')
            
            # De-identify the dataset
            deidentified_dataset = self.deidentify_dataset(dataset, patient_id)
            
            # Write the de-identified file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            deidentified_dataset.save_as(str(output_path))
            
            log_audit_event(
                event_type=AuditEventType.DICOM_WRITE,
                severity=AuditSeverity.INFO,
                message="De-identified DICOM file saved",
                patient_id=self._patient_id_mapping.get(patient_id, patient_id),
                additional_data={
                    'input_file': str(input_path),
                    'output_file': str(output_path)
                }
            )
            
            return True
            
        except InvalidDicomError as e:
            log_audit_event(
                event_type=AuditEventType.DE_IDENTIFICATION,
                severity=AuditSeverity.ERROR,
                message=f"Invalid DICOM file: {str(e)}",
                additional_data={
                    'input_file': str(input_path),
                    'error': str(e)
                }
            )
            return False
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.DE_IDENTIFICATION,
                severity=AuditSeverity.ERROR,
                message=f"De-identification failed: {str(e)}",
                additional_data={
                    'input_file': str(input_path),
                    'error': str(e)
                }
            )
            return False
    
    def batch_deidentify_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        De-identify all DICOM files in a directory.
        
        Args:
            input_dir: Directory containing DICOM files
            output_dir: Directory for de-identified files
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'failed_files': []
        }
        
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Find all DICOM files (*.dcm, *.dicom, or files with DICOM magic number)
        dicom_files = []
        for file_path in input_dir.rglob("*"):
            if file_path.is_file():
                # Check file extension or DICOM magic number
                if (file_path.suffix.lower() in ['.dcm', '.dicom'] or 
                    self._is_dicom_file(file_path)):
                    dicom_files.append(file_path)
        
        results['total_files'] = len(dicom_files)
        
        for file_path in dicom_files:
            # Maintain directory structure in output
            relative_path = file_path.relative_to(input_dir)
            output_file_path = output_dir / relative_path
            
            if self.deidentify_file(file_path, output_file_path):
                results['successful'] += 1
            else:
                results['failed'] += 1
                results['failed_files'].append(str(file_path))
        
        log_audit_event(
            event_type=AuditEventType.DE_IDENTIFICATION,
            severity=AuditSeverity.INFO,
            message="Batch de-identification completed",
            additional_data=results
        )
        
        return results
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """
        Check if a file is a DICOM file by reading its magic number.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file is a DICOM file
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 132 bytes and check for DICOM magic number
                header = f.read(132)
                if len(header) >= 132:
                    # DICOM files have 'DICM' at offset 128
                    return header[128:132] == b'DICM'
                return False
        except (IOError, OSError):
            return False
    
    def verify_deidentification(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Verify that a dataset has been properly de-identified.
        
        Args:
            dataset: DICOM dataset to verify
            
        Returns:
            Verification results with any remaining PHI
        """
        verification_results = {
            'is_deidentified': True,
            'remaining_phi': [],
            'warnings': []
        }
        
        # Check for presence of PHI tags
        for tag_tuple, tag_name in self.PHI_TAGS.items():
            tag = Tag(tag_tuple)
            if tag in dataset:
                verification_results['is_deidentified'] = False
                verification_results['remaining_phi'].append(tag_name)
        
        # Check for de-identification markers
        if not hasattr(dataset, 'PatientIdentityRemoved'):
            verification_results['warnings'].append("Missing PatientIdentityRemoved tag")
        elif dataset.PatientIdentityRemoved != "YES":
            verification_results['warnings'].append("PatientIdentityRemoved not set to YES")
        
        return verification_results
    
    def get_patient_mapping(self) -> Dict[str, str]:
        """
        Get the mapping between original and anonymous patient IDs.
        
        Returns:
            Dictionary mapping original to anonymous patient IDs
        """
        return self._patient_id_mapping.copy()
    
    def export_mapping(self, output_path: Path) -> None:
        """
        Export patient ID mapping to an encrypted file for re-identification.
        
        Args:
            output_path: Path to save the encrypted mapping file
        """
        import json
        from ..security.encryption import encrypt_data
        
        mapping_data = {
            'patient_mapping': self._patient_id_mapping,
            'uid_mapping': self._uid_mapping,
            'date_shift_days': self._date_shift_days,
            'created_timestamp': datetime.now().isoformat()
        }
        
        # Convert to JSON and encrypt
        json_data = json.dumps(mapping_data)
        encrypted_data = encrypt_data(json_data.encode())
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        log_audit_event(
            event_type=AuditEventType.DATA_EXPORT,
            severity=AuditSeverity.INFO,
            message="De-identification mapping exported",
            additional_data={'output_file': str(output_path)}
        )