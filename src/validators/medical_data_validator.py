#!/usr/bin/env python3
"""
Medical Data Validator
Comprehensive validation system for medical imaging data, DICOM compliance,
and clinical data integrity for the holistic diagnostic platform.
"""

import os
import logging
import pydicom
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, date
import hashlib
import json
import re
from pathlib import Path
from PIL import Image
import cv2
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationCategory(Enum):
    """Validation categories for medical data."""
    DICOM_COMPLIANCE = "dicom_compliance"
    IMAGE_QUALITY = "image_quality"
    PATIENT_DATA = "patient_data"
    STUDY_INTEGRITY = "study_integrity"
    HIPAA_COMPLIANCE = "hipaa_compliance"
    CLINICAL_CONTEXT = "clinical_context"

@dataclass
class ValidationIssue:
    """Individual validation issue or finding."""
    category: ValidationCategory
    severity: ValidationSeverity
    code: str
    message: str
    location: Optional[str] = None
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation issue to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "suggested_fix": self.suggested_fix,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }

@dataclass
class ValidationResult:
    """Results of medical data validation."""
    file_path: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_time: Optional[str] = None
    
    def __post_init__(self):
        if self.validation_time is None:
            self.validation_time = datetime.now().isoformat()
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
        # Update overall validity based on critical/error issues
        if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
            self.is_valid = False
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues by category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "file_path": self.file_path,
            "is_valid": self.is_valid,
            "validation_time": self.validation_time,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
            "summary": {
                "total_issues": len(self.issues),
                "critical_issues": len(self.get_issues_by_severity(ValidationSeverity.CRITICAL)),
                "error_issues": len(self.get_issues_by_severity(ValidationSeverity.ERROR)),
                "warning_issues": len(self.get_issues_by_severity(ValidationSeverity.WARNING)),
                "info_issues": len(self.get_issues_by_severity(ValidationSeverity.INFO))
            }
        }

class DICOMValidator:
    """DICOM file format and compliance validator."""
    
    def __init__(self):
        self.required_tags = {
            # Patient Information
            (0x0010, 0x0010): "Patient Name",
            (0x0010, 0x0020): "Patient ID",
            (0x0010, 0x0030): "Patient Birth Date",
            (0x0010, 0x0040): "Patient Sex",
            
            # Study Information
            (0x0020, 0x000D): "Study Instance UID",
            (0x0020, 0x0010): "Study ID",
            (0x0008, 0x0020): "Study Date",
            (0x0008, 0x0030): "Study Time",
            
            # Series Information
            (0x0020, 0x000E): "Series Instance UID",
            (0x0020, 0x0011): "Series Number",
            (0x0008, 0x0060): "Modality",
            
            # Image Information
            (0x0008, 0x0018): "SOP Instance UID",
            (0x0028, 0x0010): "Rows",
            (0x0028, 0x0011): "Columns"
        }
        
        self.critical_tags = {
            (0x0020, 0x000D): "Study Instance UID",
            (0x0020, 0x000E): "Series Instance UID", 
            (0x0008, 0x0018): "SOP Instance UID",
            (0x0008, 0x0060): "Modality"
        }
    
    def validate_dicom_file(self, file_path: str) -> ValidationResult:
        """Validate DICOM file format and compliance."""
        result = ValidationResult(file_path=file_path, is_valid=True)
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.DICOM_COMPLIANCE,
                    severity=ValidationSeverity.CRITICAL,
                    code="FILE_NOT_FOUND",
                    message=f"DICOM file not found: {file_path}"
                ))
                return result
            
            # Try to read DICOM file
            try:
                dcm = pydicom.dcmread(file_path, force=True)
            except Exception as e:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.DICOM_COMPLIANCE,
                    severity=ValidationSeverity.CRITICAL,
                    code="INVALID_DICOM",
                    message=f"Cannot read DICOM file: {str(e)}",
                    suggested_fix="Ensure file is valid DICOM format"
                ))
                return result
            
            # Validate DICOM header
            if not hasattr(dcm, 'file_meta') or not dcm.file_meta:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.DICOM_COMPLIANCE,
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_META_HEADER",
                    message="DICOM file meta header is missing or empty"
                ))
            
            # Validate required tags
            for tag, name in self.required_tags.items():
                if tag not in dcm:
                    severity = ValidationSeverity.CRITICAL if tag in self.critical_tags else ValidationSeverity.WARNING
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.DICOM_COMPLIANCE,
                        severity=severity,
                        code="MISSING_REQUIRED_TAG",
                        message=f"Missing required DICOM tag: {name} {tag}",
                        suggested_fix=f"Add required DICOM tag: {name}"
                    ))
            
            # Validate UIDs
            self._validate_uids(dcm, result)
            
            # Validate patient data
            self._validate_patient_data(dcm, result)
            
            # Validate image data
            self._validate_image_data(dcm, result)
            
            # Validate modality-specific requirements
            self._validate_modality_specific(dcm, result)
            
            # Store metadata
            result.metadata.update({
                "modality": getattr(dcm, 'Modality', 'Unknown'),
                "study_date": getattr(dcm, 'StudyDate', 'Unknown'),
                "patient_id": getattr(dcm, 'PatientID', 'Unknown'),
                "file_size_bytes": os.path.getsize(file_path),
                "transfer_syntax": str(dcm.file_meta.get('TransferSyntaxUID', 'Unknown'))
            })
            
        except Exception as e:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DICOM_COMPLIANCE,
                severity=ValidationSeverity.CRITICAL,
                code="VALIDATION_ERROR",
                message=f"Error during DICOM validation: {str(e)}"
            ))
        
        return result
    
    def _validate_uids(self, dcm: pydicom.Dataset, result: ValidationResult):
        """Validate DICOM UIDs format and uniqueness."""
        uid_tags = [
            ((0x0020, 0x000D), 'StudyInstanceUID'),
            ((0x0020, 0x000E), 'SeriesInstanceUID'),
            ((0x0008, 0x0018), 'SOPInstanceUID')
        ]
        
        uid_pattern = re.compile(r'^[0-9.]+$')
        
        for tag, name in uid_tags:
            if tag in dcm:
                uid_value = str(dcm[tag].value)
                
                # Check UID format
                if not uid_pattern.match(uid_value):
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.DICOM_COMPLIANCE,
                        severity=ValidationSeverity.ERROR,
                        code="INVALID_UID_FORMAT",
                        message=f"Invalid UID format for {name}: {uid_value}",
                        suggested_fix="UID should contain only digits and dots"
                    ))
                
                # Check UID length
                if len(uid_value) > 64:
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.DICOM_COMPLIANCE,
                        severity=ValidationSeverity.ERROR,
                        code="UID_TOO_LONG", 
                        message=f"UID too long for {name}: {len(uid_value)} characters",
                        suggested_fix="UID should be 64 characters or less"
                    ))
    
    def _validate_patient_data(self, dcm: pydicom.Dataset, result: ValidationResult):
        """Validate patient information for HIPAA compliance."""
        # Check for potentially sensitive data in patient name
        if hasattr(dcm, 'PatientName'):
            patient_name = str(dcm.PatientName)
            if len(patient_name) > 64:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.HIPAA_COMPLIANCE,
                    severity=ValidationSeverity.WARNING,
                    code="LONG_PATIENT_NAME",
                    message="Patient name is unusually long, may contain extra identifiers"
                ))
        
        # Check birth date format
        if hasattr(dcm, 'PatientBirthDate'):
            birth_date = str(dcm.PatientBirthDate)
            if not re.match(r'^\d{8}$', birth_date):
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.PATIENT_DATA,
                    severity=ValidationSeverity.ERROR,
                    code="INVALID_BIRTH_DATE",
                    message=f"Invalid birth date format: {birth_date}",
                    suggested_fix="Birth date should be YYYYMMDD format"
                ))
        
        # Check patient sex
        if hasattr(dcm, 'PatientSex'):
            sex = str(dcm.PatientSex).upper()
            if sex not in ['M', 'F', 'O', '']:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.PATIENT_DATA,
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_PATIENT_SEX",
                    message=f"Invalid patient sex value: {sex}",
                    suggested_fix="Patient sex should be M, F, O, or empty"
                ))
    
    def _validate_image_data(self, dcm: pydicom.Dataset, result: ValidationResult):
        """Validate image data and pixel information."""
        try:
            if hasattr(dcm, 'pixel_array'):
                pixel_data = dcm.pixel_array
                
                # Check image dimensions
                if pixel_data.size == 0:
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.IMAGE_QUALITY,
                        severity=ValidationSeverity.ERROR,
                        code="EMPTY_PIXEL_DATA",
                        message="Image contains no pixel data"
                    ))
                
                # Check for reasonable image dimensions
                if hasattr(dcm, 'Rows') and hasattr(dcm, 'Columns'):
                    rows, cols = int(dcm.Rows), int(dcm.Columns)
                    
                    if rows < 10 or cols < 10:
                        result.add_issue(ValidationIssue(
                            category=ValidationCategory.IMAGE_QUALITY,
                            severity=ValidationSeverity.WARNING,
                            code="SMALL_IMAGE_DIMENSIONS",
                            message=f"Unusually small image dimensions: {rows}x{cols}"
                        ))
                    
                    if rows > 10000 or cols > 10000:
                        result.add_issue(ValidationIssue(
                            category=ValidationCategory.IMAGE_QUALITY,
                            severity=ValidationSeverity.WARNING,
                            code="LARGE_IMAGE_DIMENSIONS",
                            message=f"Unusually large image dimensions: {rows}x{cols}"
                        ))
                
                # Check pixel value range
                min_val, max_val = np.min(pixel_data), np.max(pixel_data)
                if min_val == max_val:
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.IMAGE_QUALITY,
                        severity=ValidationSeverity.WARNING,
                        code="UNIFORM_PIXEL_VALUES",
                        message="Image has uniform pixel values (may be blank)"
                    ))
                    
        except Exception as e:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE_QUALITY,
                severity=ValidationSeverity.ERROR,
                code="PIXEL_DATA_ERROR",
                message=f"Error accessing pixel data: {str(e)}"
            ))
    
    def _validate_modality_specific(self, dcm: pydicom.Dataset, result: ValidationResult):
        """Validate modality-specific requirements."""
        if not hasattr(dcm, 'Modality'):
            return
            
        modality = str(dcm.Modality).upper()
        
        # CT-specific validation
        if modality == 'CT':
            if not hasattr(dcm, 'SliceThickness'):
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.CLINICAL_CONTEXT,
                    severity=ValidationSeverity.WARNING,
                    code="MISSING_SLICE_THICKNESS",
                    message="CT image missing slice thickness information"
                ))
            
            if hasattr(dcm, 'KVP'):
                kvp = float(dcm.KVP)
                if kvp < 80 or kvp > 150:
                    result.add_issue(ValidationIssue(
                        category=ValidationCategory.CLINICAL_CONTEXT,
                        severity=ValidationSeverity.INFO,
                        code="UNUSUAL_KVP",
                        message=f"Unusual KVP value for CT: {kvp}"
                    ))
        
        # MR-specific validation
        elif modality == 'MR':
            if not hasattr(dcm, 'MagneticFieldStrength'):
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.CLINICAL_CONTEXT,
                    severity=ValidationSeverity.WARNING,
                    code="MISSING_FIELD_STRENGTH",
                    message="MR image missing magnetic field strength"
                ))
            
            if not hasattr(dcm, 'RepetitionTime'):
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.CLINICAL_CONTEXT,
                    severity=ValidationSeverity.WARNING,
                    code="MISSING_TR",
                    message="MR image missing repetition time (TR)"
                ))

class ImageQualityValidator:
    """Validate medical image quality and characteristics."""
    
    def __init__(self):
        self.min_contrast_threshold = 0.1
        self.max_noise_threshold = 0.3
        self.min_sharpness_threshold = 0.2
    
    def validate_image_quality(self, image_array: np.ndarray, result: ValidationResult):
        """Validate medical image quality metrics."""
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image_array.astype(np.uint8)
            
            # Check contrast
            contrast = self._calculate_contrast(image_gray)
            if contrast < self.min_contrast_threshold:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.IMAGE_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    code="LOW_CONTRAST",
                    message=f"Low image contrast detected: {contrast:.3f}",
                    suggested_fix="Review imaging parameters or post-processing"
                ))
            
            # Check noise levels
            noise_level = self._estimate_noise(image_gray)
            if noise_level > self.max_noise_threshold:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.IMAGE_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    code="HIGH_NOISE",
                    message=f"High noise level detected: {noise_level:.3f}",
                    suggested_fix="Check acquisition parameters or apply denoising"
                ))
            
            # Check sharpness
            sharpness = self._calculate_sharpness(image_gray)
            if sharpness < self.min_sharpness_threshold:
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.IMAGE_QUALITY,
                    severity=ValidationSeverity.WARNING,
                    code="LOW_SHARPNESS",
                    message=f"Low image sharpness detected: {sharpness:.3f}",
                    suggested_fix="Check for motion artifacts or focus issues"
                ))
            
            # Check for artifacts
            self._detect_artifacts(image_gray, result)
            
        except Exception as e:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE_QUALITY,
                severity=ValidationSeverity.ERROR,
                code="QUALITY_ANALYSIS_ERROR",
                message=f"Error analyzing image quality: {str(e)}"
            ))
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast using RMS contrast."""
        return np.std(image) / np.mean(image) if np.mean(image) > 0 else 0.0
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level using Laplacian variance."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian) / (np.mean(image) ** 2) if np.mean(image) > 0 else 0.0
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using gradient magnitude."""
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.mean(gradient_magnitude) / 255.0
    
    def _detect_artifacts(self, image: np.ndarray, result: ValidationResult):
        """Detect common imaging artifacts."""
        # Check for ring artifacts (common in CT)
        rings_detected = self._detect_ring_artifacts(image)
        if rings_detected:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE_QUALITY,
                severity=ValidationSeverity.WARNING,
                code="RING_ARTIFACTS",
                message="Potential ring artifacts detected",
                suggested_fix="Check detector calibration"
            ))
        
        # Check for motion artifacts
        motion_detected = self._detect_motion_artifacts(image)
        if motion_detected:
            result.add_issue(ValidationIssue(
                category=ValidationCategory.IMAGE_QUALITY,
                severity=ValidationSeverity.WARNING,
                code="MOTION_ARTIFACTS",
                message="Potential motion artifacts detected",
                suggested_fix="Consider patient stabilization or faster acquisition"
            ))
    
    def _detect_ring_artifacts(self, image: np.ndarray) -> bool:
        """Simple ring artifact detection."""
        # Convert to polar coordinates and check for periodic patterns
        center = (image.shape[0] // 2, image.shape[1] // 2)
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Sample radial profile
        max_radius = min(center[0], center[1])
        radial_profile = []
        for radius in range(1, max_radius, 5):
            mask = (r >= radius-2) & (r < radius+2)
            if np.any(mask):
                radial_profile.append(np.mean(image[mask]))
        
        # Check for periodic variations
        if len(radial_profile) > 10:
            variations = np.diff(radial_profile)
            return np.std(variations) > 0.1 * np.mean(radial_profile)
        
        return False
    
    def _detect_motion_artifacts(self, image: np.ndarray) -> bool:
        """Simple motion artifact detection."""
        # Check for directional blurring
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_x_std = np.std(sobel_x)
        grad_y_std = np.std(sobel_y)
        
        # Motion artifacts often show as directional blur
        ratio = max(grad_x_std, grad_y_std) / (min(grad_x_std, grad_y_std) + 1e-10)
        return ratio > 3.0

class MedicalDataValidator:
    """Main validator coordinating all validation components."""
    
    def __init__(self):
        self.dicom_validator = DICOMValidator()
        self.quality_validator = ImageQualityValidator()
        self.validation_history: List[ValidationResult] = []
        self.max_history = 1000
    
    def validate_medical_file(self, file_path: str, validate_quality: bool = True) -> ValidationResult:
        """Validate a medical file (DICOM, etc.)."""
        try:
            # Determine file type and validate accordingly
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension in ['.dcm', '.dicom', ''] or self._is_dicom_file(file_path):
                result = self.dicom_validator.validate_dicom_file(file_path)
                
                # Add image quality validation if requested
                if validate_quality and result.is_valid:
                    try:
                        dcm = pydicom.dcmread(file_path, force=True)
                        if hasattr(dcm, 'pixel_array'):
                            self.quality_validator.validate_image_quality(dcm.pixel_array, result)
                    except Exception as e:
                        logger.warning(f"Could not perform quality validation: {str(e)}")
                
            else:
                result = ValidationResult(file_path=file_path, is_valid=False)
                result.add_issue(ValidationIssue(
                    category=ValidationCategory.DICOM_COMPLIANCE,
                    severity=ValidationSeverity.ERROR,
                    code="UNSUPPORTED_FORMAT",
                    message=f"Unsupported file format: {file_extension}",
                    suggested_fix="Convert to DICOM format"
                ))
            
            # Store in history
            self.validation_history.append(result)
            if len(self.validation_history) > self.max_history:
                self.validation_history.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            result = ValidationResult(file_path=file_path, is_valid=False)
            result.add_issue(ValidationIssue(
                category=ValidationCategory.DICOM_COMPLIANCE,
                severity=ValidationSeverity.CRITICAL,
                code="VALIDATION_FAILURE",
                message=f"Validation failed: {str(e)}"
            ))
            return result
    
    def validate_directory(self, directory_path: str, recursive: bool = True) -> List[ValidationResult]:
        """Validate all medical files in a directory."""
        results = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return results
        
        # Find medical files
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                # Check if it might be a medical file
                if self._is_medical_file(file_path):
                    result = self.validate_medical_file(str(file_path))
                    results.append(result)
        
        return results
    
    def _is_dicom_file(self, file_path: str) -> bool:
        """Check if file is likely a DICOM file."""
        try:
            with open(file_path, 'rb') as f:
                # DICOM files have specific preamble and prefix
                f.seek(128)
                prefix = f.read(4)
                return prefix == b'DICM'
        except Exception:
            return False
    
    def _is_medical_file(self, file_path: Path) -> bool:
        """Check if file is likely a medical imaging file."""
        # Check extension
        medical_extensions = {'.dcm', '.dicom', '.img', '.nii', '.nii.gz'}
        if file_path.suffix.lower() in medical_extensions:
            return True
        
        # Check if it's a DICOM file without extension
        return self._is_dicom_file(str(file_path))
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not results:
            return {"error": "No validation results provided"}
        
        total_files = len(results)
        valid_files = sum(1 for r in results if r.is_valid)
        invalid_files = total_files - valid_files
        
        # Categorize issues
        issue_summary = {
            "critical": 0,
            "error": 0, 
            "warning": 0,
            "info": 0
        }
        
        category_summary = {}
        for result in results:
            for issue in result.issues:
                issue_summary[issue.severity.value] += 1
                
                if issue.category.value not in category_summary:
                    category_summary[issue.category.value] = 0
                category_summary[issue.category.value] += 1
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files": total_files,
                "valid_files": valid_files,
                "invalid_files": invalid_files,
                "validation_success_rate": (valid_files / total_files) * 100 if total_files > 0 else 0
            },
            "issue_summary": issue_summary,
            "category_summary": category_summary,
            "detailed_results": [result.to_dict() for result in results]
        }

# CLI Interface
def main():
    """Main CLI interface for medical data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Data Validator")
    parser.add_argument("path", help="File or directory to validate")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursive directory validation")
    parser.add_argument("--no-quality", action="store_true", help="Skip image quality validation")
    parser.add_argument("--output", "-o", help="Output report to file")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    validator = MedicalDataValidator()
    
    try:
        path = Path(args.path)
        
        if path.is_file():
            # Validate single file
            result = validator.validate_medical_file(str(path), not args.no_quality)
            results = [result]
        elif path.is_dir():
            # Validate directory
            results = validator.validate_directory(str(path), args.recursive)
        else:
            print(f"Error: Path not found: {args.path}")
            return 1
        
        # Generate report
        report = validator.generate_validation_report(results)
        
        # Output report
        if args.format == "yaml":
            import yaml
            output = yaml.dump(report, default_flow_style=False, indent=2)
        else:
            output = json.dumps(report, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Validation report saved to: {args.output}")
        else:
            print(output)
        
        # Return non-zero exit code if validation issues found
        if report["summary"]["invalid_files"] > 0:
            return 1
            
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())