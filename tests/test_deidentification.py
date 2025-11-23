"""
Unit tests for DICOM de-identification functionality.

These tests verify that the de-identification process properly removes
or anonymizes PHI according to HIPAA Safe Harbor standards.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import numpy as np

from src.data.deid import DICOMDeidentifier
from src.config import get_config


class TestDICOMDeidentification:
    """Test suite for DICOM de-identification."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_dicom_dataset(self):
        """Create a sample DICOM dataset with PHI for testing."""
        # Create a minimal DICOM dataset
        ds = Dataset()
        
        # Critical DICOM tags
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        ds.SOPInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.Modality = "CT"
        
        # PHI tags that should be removed/anonymized
        ds.PatientName = "Doe^John^Middle^Jr"
        ds.PatientID = "12345678"
        ds.PatientBirthDate = "19800101"
        ds.PatientSex = "M"
        ds.PatientAge = "043Y"
        ds.PatientAddress = "123 Main St, Anytown, ST 12345"
        ds.PatientTelephoneNumbers = "555-123-4567"
        
        # Study information with dates/times
        ds.StudyDate = "20231115"
        ds.StudyTime = "143022"
        ds.SeriesDate = "20231115"
        ds.SeriesTime = "143500"
        ds.AcquisitionDate = "20231115"
        ds.AcquisitionTime = "143530"
        
        # Institution and physician information
        ds.InstitutionName = "Test Hospital"
        ds.InstitutionAddress = "456 Hospital Blvd, Medcity, ST 54321"
        ds.ReferringPhysicianName = "Smith^Dr^Jane"
        ds.PerformingPhysicianName = "Johnson^Dr^Bob"
        ds.OperatorsName = "Tech^John"
        ds.StationName = "CT-SCANNER-01"
        
        # Study and series descriptions
        ds.StudyDescription = "CT Chest with contrast - Patient John Doe"
        ds.SeriesDescription = "Axial chest CT"
        ds.ImageComments = "Patient moved during scan - John Doe"
        
        # Equipment information
        ds.DeviceSerialNumber = "ABC123456"
        ds.SoftwareVersions = "v2.1.3"
        
        # Add pixel data (minimal)
        ds.Rows = 512
        ds.Columns = 512
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        
        # Create minimal pixel data
        pixel_array = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
        ds.PixelData = pixel_array.tobytes()
        
        return ds
    
    @pytest.fixture
    def deidentifier(self):
        """Create a DICOMDeidentifier instance for testing."""
        return DICOMDeidentifier(date_shift_days=100, preserve_age=True)
    
    def test_remove_patient_identifiers(self, deidentifier, sample_dicom_dataset):
        """Test that patient identifiers are properly removed."""
        original_ds = sample_dicom_dataset
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Patient name should be removed
        assert not hasattr(deidentified_ds, 'PatientName')
        
        # Patient address should be removed
        assert not hasattr(deidentified_ds, 'PatientAddress')
        
        # Patient telephone should be removed
        assert not hasattr(deidentified_ds, 'PatientTelephoneNumbers')
        
        # Patient ID should be anonymized (replaced with anonymous ID)
        if hasattr(deidentified_ds, 'PatientID'):
            assert deidentified_ds.PatientID != original_ds.PatientID
            assert deidentified_ds.PatientID.startswith('PATIENT_')
    
    def test_remove_physician_identifiers(self, deidentifier, sample_dicom_dataset):
        """Test that physician identifiers are properly removed."""
        original_ds = sample_dicom_dataset
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Physician names should be removed
        assert not hasattr(deidentified_ds, 'ReferringPhysicianName')
        assert not hasattr(deidentified_ds, 'PerformingPhysicianName')
        assert not hasattr(deidentified_ds, 'OperatorsName')
    
    def test_remove_institution_identifiers(self, deidentifier, sample_dicom_dataset):
        """Test that institution identifiers are properly removed."""
        original_ds = sample_dicom_dataset
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Institution information should be removed
        assert not hasattr(deidentified_ds, 'InstitutionName')
        assert not hasattr(deidentified_ds, 'InstitutionAddress')
        assert not hasattr(deidentified_ds, 'StationName')
    
    def test_date_shifting(self, deidentifier, sample_dicom_dataset):
        """Test that dates are properly shifted."""
        original_ds = sample_dicom_dataset
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Dates should be shifted
        if hasattr(deidentified_ds, 'StudyDate') and hasattr(original_ds, 'StudyDate'):
            original_date = datetime.strptime(original_ds.StudyDate, "%Y%m%d")
            shifted_date = datetime.strptime(deidentified_ds.StudyDate, "%Y%m%d")
            
            # Should be exactly 100 days different (our test shift)
            diff = abs((shifted_date - original_date).days)
            assert diff == 100
    
    def test_uid_anonymization(self, deidentifier, sample_dicom_dataset):
        """Test that UIDs are properly anonymized."""
        original_ds = sample_dicom_dataset
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # UIDs should be replaced with anonymous versions
        assert deidentified_ds.StudyInstanceUID != original_ds.StudyInstanceUID
        assert deidentified_ds.SeriesInstanceUID != original_ds.SeriesInstanceUID
        assert deidentified_ds.SOPInstanceUID != original_ds.SOPInstanceUID
        
        # New UIDs should still be valid DICOM UIDs
        assert "1.2.826.0.1.3680043.9.7147" in deidentified_ds.StudyInstanceUID
        assert "1.2.826.0.1.3680043.9.7147" in deidentified_ds.SeriesInstanceUID
        assert "1.2.826.0.1.3680043.9.7147" in deidentified_ds.SOPInstanceUID
    
    def test_age_preservation_normal(self, deidentifier, sample_dicom_dataset):
        """Test that normal ages are preserved."""
        original_ds = sample_dicom_dataset
        original_ds.PatientAge = "043Y"  # 43 years old
        
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Age under 85 should be preserved
        assert hasattr(deidentified_ds, 'PatientAge')
        assert deidentified_ds.PatientAge == "043Y"
    
    def test_age_anonymization_elderly(self, deidentifier, sample_dicom_dataset):
        """Test that elderly ages are properly anonymized."""
        original_ds = sample_dicom_dataset
        original_ds.PatientAge = "092Y"  # 92 years old
        
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Age 90+ should be removed
        assert not hasattr(deidentified_ds, 'PatientAge')
    
    def test_age_grouping_85_89(self, deidentifier, sample_dicom_dataset):
        """Test that ages 85-89 are grouped."""
        original_ds = sample_dicom_dataset
        original_ds.PatientAge = "087Y"  # 87 years old
        
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Age 85-89 should be grouped as 85+
        assert hasattr(deidentified_ds, 'PatientAge')
        assert deidentified_ds.PatientAge == "085Y"
    
    def test_text_field_cleaning(self, deidentifier, sample_dicom_dataset):
        """Test that text fields are properly cleaned."""
        original_ds = sample_dicom_dataset
        
        # Add text with potential PHI
        original_ds.StudyDescription = "CT scan for patient John Doe, SSN 123-45-6789"
        original_ds.ImageComments = "Contact: john.doe@email.com or 555-123-4567"
        
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # PHI patterns should be replaced with [REMOVED]
        if hasattr(deidentified_ds, 'StudyDescription'):
            assert "123-45-6789" not in deidentified_ds.StudyDescription
            assert "[REMOVED]" in deidentified_ds.StudyDescription
        
        if hasattr(deidentified_ds, 'ImageComments'):
            assert "john.doe@email.com" not in deidentified_ds.ImageComments
            assert "555-123-4567" not in deidentified_ds.ImageComments
            assert "[REMOVED]" in deidentified_ds.ImageComments
    
    def test_deidentification_markers(self, deidentifier, sample_dicom_dataset):
        """Test that proper de-identification markers are added."""
        original_ds = sample_dicom_dataset
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # De-identification markers should be present
        assert hasattr(deidentified_ds, 'PatientIdentityRemoved')
        assert deidentified_ds.PatientIdentityRemoved == "YES"
        
        assert hasattr(deidentified_ds, 'DeidentificationMethod')
        assert deidentified_ds.DeidentificationMethod == "HIPAA_SAFE_HARBOR_AUTOMATED"
    
    def test_consistent_anonymization(self, deidentifier, sample_dicom_dataset):
        """Test that anonymization is consistent across multiple calls."""
        original_ds = sample_dicom_dataset
        
        # De-identify the same dataset twice
        deidentified_ds1 = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        deidentified_ds2 = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Anonymous IDs should be the same
        if hasattr(deidentified_ds1, 'PatientID') and hasattr(deidentified_ds2, 'PatientID'):
            assert deidentified_ds1.PatientID == deidentified_ds2.PatientID
        
        # UIDs should be the same
        assert deidentified_ds1.StudyInstanceUID == deidentified_ds2.StudyInstanceUID
        assert deidentified_ds1.SeriesInstanceUID == deidentified_ds2.SeriesInstanceUID
    
    def test_file_deidentification(self, deidentifier, sample_dicom_dataset, temp_dir):
        """Test de-identification of DICOM files."""
        # Save sample dataset to file
        input_file = temp_dir / "input.dcm"
        output_file = temp_dir / "output.dcm"
        
        sample_dicom_dataset.save_as(str(input_file))
        
        # De-identify file
        success = deidentifier.deidentify_file(input_file, output_file)
        assert success
        assert output_file.exists()
        
        # Read and verify de-identified file
        deidentified_ds = pydicom.dcmread(str(output_file))
        
        # Should have de-identification markers
        assert deidentified_ds.PatientIdentityRemoved == "YES"
        
        # Should not have original patient name
        assert not hasattr(deidentified_ds, 'PatientName')
    
    def test_batch_deidentification(self, deidentifier, sample_dicom_dataset, temp_dir):
        """Test batch de-identification of multiple files."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()
        
        # Create multiple test files
        for i in range(3):
            test_ds = sample_dicom_dataset.copy()
            test_ds.PatientID = f"PATIENT_{i:03d}"
            test_ds.SOPInstanceUID = generate_uid()
            
            input_file = input_dir / f"test_{i}.dcm"
            test_ds.save_as(str(input_file))
        
        # Run batch de-identification
        results = deidentifier.batch_deidentify_directory(input_dir, output_dir)
        
        # Verify results
        assert results['total_files'] == 3
        assert results['successful'] == 3
        assert results['failed'] == 0
        
        # Verify output files exist and are de-identified
        for i in range(3):
            output_file = output_dir / f"test_{i}.dcm"
            assert output_file.exists()
            
            deidentified_ds = pydicom.dcmread(str(output_file))
            assert deidentified_ds.PatientIdentityRemoved == "YES"
    
    def test_verification_function(self, deidentifier, sample_dicom_dataset):
        """Test the de-identification verification function."""
        # Test non-de-identified dataset
        verification = deidentifier.verify_deidentification(sample_dicom_dataset)
        assert not verification['is_deidentified']
        assert len(verification['remaining_phi']) > 0
        
        # Test de-identified dataset
        deidentified_ds = deidentifier.deidentify_dataset(sample_dicom_dataset, "TEST_PATIENT")
        verification = deidentifier.verify_deidentification(deidentified_ds)
        assert verification['is_deidentified']
        assert len(verification['remaining_phi']) == 0
    
    def test_patient_mapping_consistency(self, deidentifier, sample_dicom_dataset):
        """Test that patient ID mapping is consistent and retrievable."""
        original_patient_id = sample_dicom_dataset.PatientID
        
        # De-identify dataset
        deidentified_ds = deidentifier.deidentify_dataset(sample_dicom_dataset, "TEST_PATIENT")
        
        # Get mapping
        mapping = deidentifier.get_patient_mapping()
        
        # Original patient ID should be in mapping
        assert original_patient_id in mapping
        
        # Mapped ID should match what's in the de-identified dataset
        if hasattr(deidentified_ds, 'PatientID'):
            assert mapping[original_patient_id] == deidentified_ds.PatientID
    
    def test_mapping_export_import(self, deidentifier, sample_dicom_dataset, temp_dir):
        """Test exporting and importing de-identification mappings."""
        # De-identify dataset to create mapping
        deidentifier.deidentify_dataset(sample_dicom_dataset, "TEST_PATIENT")
        
        # Export mapping
        mapping_file = temp_dir / "mapping.enc"
        deidentifier.export_mapping(mapping_file)
        
        # File should exist and not be empty
        assert mapping_file.exists()
        assert mapping_file.stat().st_size > 0
        
        # File should be encrypted (not readable as JSON)
        with open(mapping_file, 'rb') as f:
            content = f.read()
        
        # Should not be readable JSON
        try:
            content.decode('utf-8')
            # If we can decode it, it should not be valid JSON
            import json
            json.loads(content.decode('utf-8'))
            pytest.fail("Mapping file should be encrypted, not plain JSON")
        except (UnicodeDecodeError, json.JSONDecodeError):
            # This is expected - file should be encrypted
            pass
    
    def test_invalid_dicom_handling(self, deidentifier, temp_dir):
        """Test handling of invalid DICOM files."""
        # Create a non-DICOM file
        invalid_file = temp_dir / "invalid.dcm"
        with open(invalid_file, 'w') as f:
            f.write("This is not a DICOM file")
        
        output_file = temp_dir / "output.dcm"
        
        # Should return False for invalid file
        success = deidentifier.deidentify_file(invalid_file, output_file)
        assert not success
        assert not output_file.exists()
    
    def test_preserve_critical_medical_data(self, deidentifier, sample_dicom_dataset):
        """Test that critical medical data is preserved."""
        original_ds = sample_dicom_dataset
        deidentified_ds = deidentifier.deidentify_dataset(original_ds, "TEST_PATIENT")
        
        # Critical medical data should be preserved
        assert deidentified_ds.Modality == original_ds.Modality
        assert deidentified_ds.SOPClassUID == original_ds.SOPClassUID
        assert deidentified_ds.Rows == original_ds.Rows
        assert deidentified_ds.Columns == original_ds.Columns
        
        # Pixel data should be preserved
        assert hasattr(deidentified_ds, 'PixelData')
        assert len(deidentified_ds.PixelData) == len(original_ds.PixelData)