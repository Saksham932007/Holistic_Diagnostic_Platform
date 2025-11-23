"""
Advanced Quality Assurance Framework

Comprehensive QA system for medical AI with automated testing,
validation, and compliance verification.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2
from PIL import Image

from src.core.config import settings
from src.core.audit import audit_logger
from src.core.performance_analyzer import performance_analyzer

logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of QA tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    CLINICAL_VALIDATION = "clinical_validation"
    STRESS = "stress"
    REGRESSION = "regression"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class ComplianceStandard(Enum):
    """Medical compliance standards."""
    FDA_510K = "fda_510k"
    FDA_PMA = "fda_pma"
    CE_MDR = "ce_mdr"
    HIPAA = "hipaa"
    ISO_13485 = "iso_13485"
    IEC_62304 = "iec_62304"
    DICOM = "dicom"

@dataclass
class TestCase:
    """Represents a QA test case."""
    test_id: str
    name: str
    description: str
    test_type: TestType
    test_function: Callable
    expected_result: Any
    timeout_seconds: int = 300
    retry_attempts: int = 0
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    criticality: str = "medium"  # low, medium, high, critical

@dataclass
class TestResult:
    """Result of a QA test execution."""
    test_id: str
    name: str
    test_type: TestType
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_seconds: float = 0.0
    actual_result: Any = None
    expected_result: Any = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    compliance_verified: bool = False

@dataclass
class TestSuite:
    """Collection of related test cases."""
    suite_id: str
    name: str
    description: str
    test_cases: List[TestCase]
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    parallel_execution: bool = False

class ModelValidator:
    """Validates AI model quality and performance."""
    
    def __init__(self):
        """Initialize model validator."""
        self.validation_metrics = {}
        self.test_datasets = {}
    
    async def validate_model_accuracy(
        self,
        model: nn.Module,
        test_dataloader: torch.utils.data.DataLoader,
        model_type: str = "classification"
    ) -> Dict[str, float]:
        """Validate model accuracy against test dataset."""
        model.eval()
        
        all_predictions = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                start_time = time.time()
                
                if isinstance(batch, dict):
                    inputs = batch['image']
                    labels = batch['label']
                else:
                    inputs, labels = batch
                
                outputs = model(inputs)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                if model_type == "classification":
                    predictions = torch.argmax(outputs, dim=1)
                elif model_type == "segmentation":
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    predictions = outputs
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        if model_type in ["classification", "segmentation"]:
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
        else:
            # Regression metrics
            mse = np.mean((y_true - y_pred) ** 2)
            mae = np.mean(np.abs(y_true - y_pred))
            accuracy = 1.0 / (1.0 + mse)  # Inverse MSE as accuracy proxy
            precision = recall = f1 = accuracy
        
        avg_inference_time = np.mean(inference_times)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_inference_time': float(avg_inference_time),
            'total_samples': len(y_true)
        }
    
    async def validate_model_robustness(
        self,
        model: nn.Module,
        test_samples: List[torch.Tensor],
        perturbation_types: List[str] = None
    ) -> Dict[str, float]:
        """Validate model robustness against adversarial inputs."""
        if perturbation_types is None:
            perturbation_types = ['noise', 'blur', 'brightness']
        
        model.eval()
        
        robustness_scores = {}
        
        for perturbation in perturbation_types:
            original_predictions = []
            perturbed_predictions = []
            
            with torch.no_grad():
                for sample in test_samples:
                    # Original prediction
                    orig_output = model(sample.unsqueeze(0))
                    orig_pred = torch.argmax(orig_output, dim=1)
                    
                    # Apply perturbation
                    perturbed_sample = self._apply_perturbation(sample, perturbation)
                    
                    # Perturbed prediction
                    pert_output = model(perturbed_sample.unsqueeze(0))
                    pert_pred = torch.argmax(pert_output, dim=1)
                    
                    original_predictions.append(orig_pred.item())
                    perturbed_predictions.append(pert_pred.item())
            
            # Calculate robustness as consistency between predictions
            consistency = np.mean([
                1.0 if orig == pert else 0.0
                for orig, pert in zip(original_predictions, perturbed_predictions)
            ])
            
            robustness_scores[f'robustness_{perturbation}'] = float(consistency)
        
        robustness_scores['overall_robustness'] = float(np.mean(list(robustness_scores.values())))
        
        return robustness_scores
    
    def _apply_perturbation(self, tensor: torch.Tensor, perturbation_type: str) -> torch.Tensor:
        """Apply perturbation to input tensor."""
        perturbed = tensor.clone()
        
        if perturbation_type == 'noise':
            noise = torch.randn_like(perturbed) * 0.05
            perturbed = perturbed + noise
            
        elif perturbation_type == 'blur':
            # Convert to numpy for OpenCV blur
            if len(perturbed.shape) == 3:  # CHW format
                numpy_img = perturbed.permute(1, 2, 0).numpy()
                blurred = cv2.GaussianBlur(numpy_img, (5, 5), 1.0)
                perturbed = torch.from_numpy(blurred).permute(2, 0, 1)
                
        elif perturbation_type == 'brightness':
            brightness_factor = np.random.uniform(0.7, 1.3)
            perturbed = perturbed * brightness_factor
        
        return torch.clamp(perturbed, 0, 1)

class ComplianceValidator:
    """Validates compliance with medical regulations."""
    
    def __init__(self):
        """Initialize compliance validator."""
        self.compliance_checks = {}
        self._load_compliance_requirements()
    
    def _load_compliance_requirements(self):
        """Load compliance requirements for different standards."""
        self.compliance_checks = {
            ComplianceStandard.HIPAA: {
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'access_logging': True,
                'user_authentication': True,
                'data_anonymization': True,
                'audit_trail': True
            },
            ComplianceStandard.FDA_510K: {
                'software_validation': True,
                'risk_management': True,
                'clinical_evaluation': True,
                'quality_management': True,
                'predicate_comparison': True
            },
            ComplianceStandard.ISO_13485: {
                'design_controls': True,
                'risk_management': True,
                'configuration_management': True,
                'validation_verification': True,
                'quality_management': True
            },
            ComplianceStandard.DICOM: {
                'dicom_conformance': True,
                'image_quality': True,
                'metadata_preservation': True,
                'interoperability': True
            }
        }
    
    async def validate_compliance(
        self,
        standard: ComplianceStandard,
        system_config: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Validate compliance with specific standard."""
        requirements = self.compliance_checks.get(standard, {})
        compliance_results = {}
        
        for requirement, required in requirements.items():
            if not required:
                compliance_results[requirement] = True
                continue
            
            # Check specific compliance requirements
            if requirement == 'encryption_at_rest':
                compliance_results[requirement] = await self._check_encryption_at_rest(system_config)
            elif requirement == 'encryption_in_transit':
                compliance_results[requirement] = await self._check_encryption_in_transit(system_config)
            elif requirement == 'access_logging':
                compliance_results[requirement] = await self._check_access_logging(system_config)
            elif requirement == 'user_authentication':
                compliance_results[requirement] = await self._check_user_authentication(system_config)
            elif requirement == 'data_anonymization':
                compliance_results[requirement] = await self._check_data_anonymization(system_config)
            elif requirement == 'audit_trail':
                compliance_results[requirement] = await self._check_audit_trail(system_config)
            elif requirement == 'software_validation':
                compliance_results[requirement] = await self._check_software_validation(system_config)
            elif requirement == 'dicom_conformance':
                compliance_results[requirement] = await self._check_dicom_conformance(system_config)
            else:
                compliance_results[requirement] = False  # Unknown requirement
        
        return compliance_results
    
    async def _check_encryption_at_rest(self, config: Dict) -> bool:
        """Check if data encryption at rest is enabled."""
        database_config = config.get('database', {})
        storage_config = config.get('storage', {})
        
        db_encrypted = database_config.get('encryption_enabled', False)
        storage_encrypted = storage_config.get('encryption_enabled', False)
        
        return db_encrypted and storage_encrypted
    
    async def _check_encryption_in_transit(self, config: Dict) -> bool:
        """Check if data encryption in transit is enabled."""
        api_config = config.get('api', {})
        return api_config.get('https_enabled', False)
    
    async def _check_access_logging(self, config: Dict) -> bool:
        """Check if access logging is enabled."""
        logging_config = config.get('logging', {})
        return logging_config.get('access_logs_enabled', False)
    
    async def _check_user_authentication(self, config: Dict) -> bool:
        """Check if user authentication is properly configured."""
        auth_config = config.get('authentication', {})
        return all([
            auth_config.get('enabled', False),
            auth_config.get('strong_passwords', False),
            auth_config.get('session_management', False)
        ])
    
    async def _check_data_anonymization(self, config: Dict) -> bool:
        """Check if data anonymization is implemented."""
        privacy_config = config.get('privacy', {})
        return privacy_config.get('anonymization_enabled', False)
    
    async def _check_audit_trail(self, config: Dict) -> bool:
        """Check if comprehensive audit trail is maintained."""
        audit_config = config.get('audit', {})
        return audit_config.get('enabled', False)
    
    async def _check_software_validation(self, config: Dict) -> bool:
        """Check if software validation processes are in place."""
        validation_config = config.get('validation', {})
        return all([
            validation_config.get('test_coverage', 0) >= 80,
            validation_config.get('code_review', False),
            validation_config.get('documentation', False)
        ])
    
    async def _check_dicom_conformance(self, config: Dict) -> bool:
        """Check DICOM conformance."""
        dicom_config = config.get('dicom', {})
        return dicom_config.get('conformance_verified', False)

class QualityAssuranceFramework:
    """Main QA framework coordinator."""
    
    def __init__(self):
        """Initialize QA framework."""
        self.model_validator = ModelValidator()
        self.compliance_validator = ComplianceValidator()
        self.test_suites = {}
        self.test_results = {}
        self.test_history = []
    
    def register_test_suite(self, test_suite: TestSuite):
        """Register a test suite."""
        self.test_suites[test_suite.suite_id] = test_suite
    
    def add_test_case(self, suite_id: str, test_case: TestCase):
        """Add test case to existing suite."""
        if suite_id in self.test_suites:
            self.test_suites[suite_id].test_cases.append(test_case)
    
    async def run_test_suite(self, suite_id: str) -> List[TestResult]:
        """Run all tests in a test suite."""
        if suite_id not in self.test_suites:
            raise ValueError(f"Test suite {suite_id} not found")
        
        test_suite = self.test_suites[suite_id]
        
        await audit_logger.log_event(
            "qa_test_suite_started",
            {
                "suite_id": suite_id,
                "test_count": len(test_suite.test_cases)
            }
        )
        
        # Setup
        if test_suite.setup_function:
            try:
                await test_suite.setup_function()
            except Exception as e:
                logger.error(f"Test suite setup failed: {e}")
                return []
        
        # Run tests
        if test_suite.parallel_execution:
            results = await self._run_tests_parallel(test_suite.test_cases)
        else:
            results = await self._run_tests_sequential(test_suite.test_cases)
        
        # Teardown
        if test_suite.teardown_function:
            try:
                await test_suite.teardown_function()
            except Exception as e:
                logger.warning(f"Test suite teardown failed: {e}")
        
        # Store results
        self.test_results[suite_id] = results
        self.test_history.extend(results)
        
        await audit_logger.log_event(
            "qa_test_suite_completed",
            {
                "suite_id": suite_id,
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.status == TestStatus.PASSED),
                "failed": sum(1 for r in results if r.status == TestStatus.FAILED)
            }
        )
        
        return results
    
    async def _run_tests_sequential(self, test_cases: List[TestCase]) -> List[TestResult]:
        """Run test cases sequentially."""
        results = []
        
        for test_case in test_cases:
            result = await self._run_single_test(test_case)
            results.append(result)
        
        return results
    
    async def _run_tests_parallel(self, test_cases: List[TestCase]) -> List[TestResult]:
        """Run test cases in parallel."""
        tasks = [self._run_single_test(test_case) for test_case in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = TestResult(
                    test_id=test_cases[i].test_id,
                    name=test_cases[i].name,
                    test_type=test_cases[i].test_type,
                    status=TestStatus.ERROR,
                    start_time=datetime.now(),
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        start_time = datetime.now()
        
        result = TestResult(
            test_id=test_case.test_id,
            name=test_case.name,
            test_type=test_case.test_type,
            status=TestStatus.RUNNING,
            start_time=start_time,
            expected_result=test_case.expected_result
        )
        
        try:
            # Run test with timeout
            actual_result = await asyncio.wait_for(
                test_case.test_function(),
                timeout=test_case.timeout_seconds
            )
            
            result.actual_result = actual_result
            
            # Compare results
            if self._compare_results(actual_result, test_case.expected_result):
                result.status = TestStatus.PASSED
            else:
                result.status = TestStatus.FAILED
                result.error_message = f"Expected {test_case.expected_result}, got {actual_result}"
            
            # Check compliance if required
            if test_case.compliance_standards:
                compliance_passed = await self._verify_compliance(test_case)
                result.compliance_verified = compliance_passed
                if not compliance_passed and result.status == TestStatus.PASSED:
                    result.status = TestStatus.FAILED
                    result.error_message = "Compliance verification failed"
        
        except asyncio.TimeoutError:
            result.status = TestStatus.FAILED
            result.error_message = f"Test timed out after {test_case.timeout_seconds} seconds"
        
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
        
        finally:
            result.end_time = datetime.now()
            result.execution_time_seconds = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def _compare_results(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected test results."""
        if isinstance(expected, dict) and isinstance(actual, dict):
            # For dictionary results, check if actual meets minimum thresholds
            for key, expected_value in expected.items():
                if key not in actual:
                    return False
                
                if isinstance(expected_value, (int, float)):
                    if actual[key] < expected_value:
                        return False
                else:
                    if actual[key] != expected_value:
                        return False
            return True
        
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            # For numeric results, allow small tolerance
            return abs(actual - expected) < 0.01
        
        else:
            return actual == expected
    
    async def _verify_compliance(self, test_case: TestCase) -> bool:
        """Verify compliance requirements for test case."""
        # Placeholder - would integrate with actual compliance checks
        return True
    
    async def run_full_qa_suite(
        self,
        model: Optional[nn.Module] = None,
        test_data: Optional[Any] = None,
        system_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run comprehensive QA suite."""
        qa_results = {
            'timestamp': datetime.now(),
            'test_suites': {},
            'model_validation': {},
            'compliance_validation': {},
            'overall_status': 'UNKNOWN'
        }
        
        # Run all registered test suites
        for suite_id in self.test_suites:
            results = await self.run_test_suite(suite_id)
            qa_results['test_suites'][suite_id] = {
                'total_tests': len(results),
                'passed': sum(1 for r in results if r.status == TestStatus.PASSED),
                'failed': sum(1 for r in results if r.status == TestStatus.FAILED),
                'errors': sum(1 for r in results if r.status == TestStatus.ERROR),
                'results': results
            }
        
        # Model validation if model provided
        if model and test_data:
            try:
                model_metrics = await self.model_validator.validate_model_accuracy(model, test_data)
                qa_results['model_validation'] = model_metrics
            except Exception as e:
                qa_results['model_validation'] = {'error': str(e)}
        
        # Compliance validation if config provided
        if system_config:
            compliance_results = {}
            for standard in [ComplianceStandard.HIPAA, ComplianceStandard.DICOM]:
                try:
                    compliance_check = await self.compliance_validator.validate_compliance(
                        standard, system_config
                    )
                    compliance_results[standard.value] = compliance_check
                except Exception as e:
                    compliance_results[standard.value] = {'error': str(e)}
            
            qa_results['compliance_validation'] = compliance_results
        
        # Determine overall status
        total_tests = sum(
            suite['total_tests'] for suite in qa_results['test_suites'].values()
        )
        total_passed = sum(
            suite['passed'] for suite in qa_results['test_suites'].values()
        )
        
        if total_tests == 0:
            qa_results['overall_status'] = 'NO_TESTS'
        elif total_passed == total_tests:
            qa_results['overall_status'] = 'PASSED'
        elif total_passed == 0:
            qa_results['overall_status'] = 'FAILED'
        else:
            qa_results['overall_status'] = 'PARTIAL'
        
        return qa_results
    
    def generate_qa_report(self, output_path: Optional[Path] = None) -> str:
        """Generate comprehensive QA report."""
        report = []
        report.append("=" * 60)
        report.append("QUALITY ASSURANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Test suite results
        for suite_id, results in self.test_results.items():
            report.append(f"TEST SUITE: {suite_id}")
            report.append("-" * 30)
            
            passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            failed = sum(1 for r in results if r.status == TestStatus.FAILED)
            errors = sum(1 for r in results if r.status == TestStatus.ERROR)
            
            report.append(f"Total Tests: {len(results)}")
            report.append(f"Passed: {passed}")
            report.append(f"Failed: {failed}")
            report.append(f"Errors: {errors}")
            report.append(f"Success Rate: {(passed/len(results)*100):.1f}%")
            report.append("")
            
            # Failed tests
            failed_tests = [r for r in results if r.status != TestStatus.PASSED]
            if failed_tests:
                report.append("FAILED TESTS:")
                for test in failed_tests:
                    report.append(f"  - {test.name}: {test.error_message}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text

# Global QA framework instance
qa_framework = QualityAssuranceFramework()