"""
Clinical decision support system for medical image analysis.

This module implements AI-driven and rule-based clinical decision support
to assist healthcare professionals in diagnosis and treatment planning.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve

from monai.data import MetaTensor

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class DiagnosisConfidence(Enum):
    """Diagnosis confidence levels."""
    VERY_HIGH = "very_high"  # >90%
    HIGH = "high"           # 80-90%
    MODERATE = "moderate"   # 60-80%
    LOW = "low"            # 40-60%
    VERY_LOW = "very_low"  # <40%


class UrgencyLevel(Enum):
    """Clinical urgency levels."""
    EMERGENCY = "emergency"      # Immediate attention required
    URGENT = "urgent"           # Within hours
    SEMI_URGENT = "semi_urgent" # Within days
    ROUTINE = "routine"         # Standard follow-up


class FindingType(Enum):
    """Types of medical findings."""
    TUMOR = "tumor"
    LESION = "lesion"
    HEMORRHAGE = "hemorrhage"
    INFARCT = "infarct"
    ANEURYSM = "aneurysm"
    FRACTURE = "fracture"
    INFLAMMATION = "inflammation"
    NORMAL = "normal"
    ARTIFACT = "artifact"


@dataclass
class ClinicalFinding:
    """Individual clinical finding."""
    finding_type: FindingType
    location: Tuple[float, float, float]  # 3D coordinates
    size: float  # Volume or diameter
    confidence: float
    characteristics: Dict[str, Any] = field(default_factory=dict)
    severity_score: Optional[float] = None
    description: str = ""


@dataclass
class ClinicalRecommendation:
    """Clinical recommendation with rationale."""
    recommendation: str
    rationale: str
    urgency: UrgencyLevel
    confidence: DiagnosisConfidence
    supporting_findings: List[ClinicalFinding] = field(default_factory=list)
    follow_up_interval: Optional[str] = None
    additional_tests: List[str] = field(default_factory=list)
    specialist_referral: Optional[str] = None


@dataclass
class DiagnosisReport:
    """Comprehensive diagnosis report."""
    patient_id: str
    study_id: str
    timestamp: datetime
    primary_diagnosis: str
    differential_diagnoses: List[str]
    findings: List[ClinicalFinding]
    recommendations: List[ClinicalRecommendation]
    overall_confidence: DiagnosisConfidence
    risk_stratification: str
    clinical_notes: str = ""


class ClinicalRuleEngine:
    """
    Rule-based clinical decision support system.
    """
    
    def __init__(self, rules_config: Optional[Dict[str, Any]] = None):
        """
        Initialize clinical rule engine.
        
        Args:
            rules_config: Configuration for clinical rules
        """
        self.rules_config = rules_config or self._get_default_rules()
        self._initialize_rules()
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default clinical rules configuration."""
        return {
            'tumor_rules': {
                'size_thresholds': {
                    'small': 10,    # mm
                    'medium': 30,   # mm
                    'large': 50     # mm
                },
                'urgency_criteria': {
                    'emergency': {'size': 50, 'hemorrhage': True},
                    'urgent': {'size': 30, 'mass_effect': True},
                    'semi_urgent': {'size': 15, 'enhancement': True},
                    'routine': {'size': 10}
                }
            },
            'hemorrhage_rules': {
                'volume_thresholds': {
                    'small': 5,     # ml
                    'medium': 20,   # ml
                    'large': 50     # ml
                },
                'urgency_always': UrgencyLevel.EMERGENCY
            },
            'stroke_rules': {
                'onset_window': {
                    'acute': 4.5,   # hours
                    'subacute': 24  # hours
                },
                'nihss_thresholds': {
                    'mild': 5,
                    'moderate': 15,
                    'severe': 25
                }
            }
        }
    
    def _initialize_rules(self):
        """Initialize clinical decision rules."""
        self.rules = {
            'tumor_assessment': self._tumor_assessment_rule,
            'hemorrhage_assessment': self._hemorrhage_assessment_rule,
            'stroke_assessment': self._stroke_assessment_rule,
            'mass_effect_assessment': self._mass_effect_rule,
            'enhancement_assessment': self._enhancement_rule
        }
    
    def apply_rules(
        self,
        findings: List[ClinicalFinding],
        clinical_context: Dict[str, Any]
    ) -> List[ClinicalRecommendation]:
        """
        Apply clinical rules to findings.
        
        Args:
            findings: List of clinical findings
            clinical_context: Additional clinical context
            
        Returns:
            List of clinical recommendations
        """
        recommendations = []
        
        for finding in findings:
            # Apply relevant rules based on finding type
            if finding.finding_type == FindingType.TUMOR:
                rec = self.rules['tumor_assessment'](finding, clinical_context)
                if rec:
                    recommendations.append(rec)
            
            elif finding.finding_type == FindingType.HEMORRHAGE:
                rec = self.rules['hemorrhage_assessment'](finding, clinical_context)
                if rec:
                    recommendations.append(rec)
            
            # Apply general assessment rules
            mass_effect_rec = self.rules['mass_effect_assessment'](finding, clinical_context)
            if mass_effect_rec:
                recommendations.append(mass_effect_rec)
            
            enhancement_rec = self.rules['enhancement_assessment'](finding, clinical_context)
            if enhancement_rec:
                recommendations.append(enhancement_rec)
        
        # Remove duplicate recommendations
        unique_recommendations = self._deduplicate_recommendations(recommendations)
        
        return unique_recommendations
    
    def _tumor_assessment_rule(
        self,
        finding: ClinicalFinding,
        context: Dict[str, Any]
    ) -> Optional[ClinicalRecommendation]:
        """Assess tumor findings."""
        if finding.finding_type != FindingType.TUMOR:
            return None
        
        size_mm = finding.size  # Assuming size is in mm
        rules = self.rules_config['tumor_rules']
        
        # Determine urgency based on size and characteristics
        urgency = UrgencyLevel.ROUTINE
        recommendation = "Routine follow-up imaging"
        additional_tests = []
        specialist_referral = None
        
        # Check emergency criteria
        if (size_mm >= rules['urgency_criteria']['emergency']['size'] or
            finding.characteristics.get('hemorrhage', False)):
            urgency = UrgencyLevel.EMERGENCY
            recommendation = "Immediate neurosurgical evaluation"
            specialist_referral = "Neurosurgery"
            additional_tests = ["CT angiography", "MRI with contrast"]
        
        # Check urgent criteria
        elif (size_mm >= rules['urgency_criteria']['urgent']['size'] or
              finding.characteristics.get('mass_effect', False)):
            urgency = UrgencyLevel.URGENT
            recommendation = "Urgent oncology consultation within 24-48 hours"
            specialist_referral = "Oncology"
            additional_tests = ["MRI with contrast", "Perfusion imaging"]
        
        # Check semi-urgent criteria
        elif (size_mm >= rules['urgency_criteria']['semi_urgent']['size'] or
              finding.characteristics.get('enhancement', False)):
            urgency = UrgencyLevel.SEMI_URGENT
            recommendation = "Oncology consultation within 1 week"
            specialist_referral = "Oncology"
            additional_tests = ["MRI with contrast"]
        
        # Determine confidence based on finding characteristics
        confidence = self._determine_confidence(finding)
        
        rationale = f"Tumor size: {size_mm:.1f}mm. "
        if finding.characteristics.get('hemorrhage'):
            rationale += "Associated hemorrhage detected. "
        if finding.characteristics.get('mass_effect'):
            rationale += "Mass effect present. "
        if finding.characteristics.get('enhancement'):
            rationale += "Contrast enhancement observed. "
        
        return ClinicalRecommendation(
            recommendation=recommendation,
            rationale=rationale,
            urgency=urgency,
            confidence=confidence,
            supporting_findings=[finding],
            additional_tests=additional_tests,
            specialist_referral=specialist_referral,
            follow_up_interval=self._get_follow_up_interval(urgency)
        )
    
    def _hemorrhage_assessment_rule(
        self,
        finding: ClinicalFinding,
        context: Dict[str, Any]
    ) -> Optional[ClinicalRecommendation]:
        """Assess hemorrhage findings."""
        if finding.finding_type != FindingType.HEMORRHAGE:
            return None
        
        volume_ml = finding.size  # Assuming size is volume in ml
        rules = self.rules_config['hemorrhage_rules']
        
        # Hemorrhage always requires immediate attention
        urgency = UrgencyLevel.EMERGENCY
        recommendation = "Immediate neurosurgical evaluation and ICU admission"
        specialist_referral = "Neurosurgery"
        additional_tests = ["CT angiography", "Coagulation studies", "Repeat CT in 6 hours"]
        
        confidence = DiagnosisConfidence.VERY_HIGH
        
        rationale = f"Intracranial hemorrhage detected with volume: {volume_ml:.1f}ml. "
        if volume_ml >= rules['volume_thresholds']['large']:
            rationale += "Large hemorrhage requiring urgent intervention. "
        elif volume_ml >= rules['volume_thresholds']['medium']:
            rationale += "Moderate hemorrhage requiring close monitoring. "
        else:
            rationale += "Small hemorrhage requiring evaluation and monitoring. "
        
        return ClinicalRecommendation(
            recommendation=recommendation,
            rationale=rationale,
            urgency=urgency,
            confidence=confidence,
            supporting_findings=[finding],
            additional_tests=additional_tests,
            specialist_referral=specialist_referral,
            follow_up_interval="Immediate"
        )
    
    def _stroke_assessment_rule(
        self,
        finding: ClinicalFinding,
        context: Dict[str, Any]
    ) -> Optional[ClinicalRecommendation]:
        """Assess stroke findings."""
        if finding.finding_type != FindingType.INFARCT:
            return None
        
        # Time from symptom onset
        onset_hours = context.get('symptom_onset_hours', 24)
        rules = self.rules_config['stroke_rules']
        
        if onset_hours <= rules['onset_window']['acute']:
            urgency = UrgencyLevel.EMERGENCY
            recommendation = "Immediate stroke team activation - consider thrombolysis"
            additional_tests = ["CT angiography", "CT perfusion", "Coagulation studies"]
        elif onset_hours <= rules['onset_window']['subacute']:
            urgency = UrgencyLevel.URGENT
            recommendation = "Urgent stroke team evaluation - consider endovascular therapy"
            additional_tests = ["CT angiography", "CT perfusion"]
        else:
            urgency = UrgencyLevel.SEMI_URGENT
            recommendation = "Stroke team evaluation and secondary prevention"
            additional_tests = ["MRI with DWI", "Echocardiogram", "Carotid ultrasound"]
        
        confidence = self._determine_confidence(finding)
        
        rationale = f"Acute ischemic stroke. Time from onset: {onset_hours:.1f} hours. "
        
        return ClinicalRecommendation(
            recommendation=recommendation,
            rationale=rationale,
            urgency=urgency,
            confidence=confidence,
            supporting_findings=[finding],
            additional_tests=additional_tests,
            specialist_referral="Neurology",
            follow_up_interval=self._get_follow_up_interval(urgency)
        )
    
    def _mass_effect_rule(
        self,
        finding: ClinicalFinding,
        context: Dict[str, Any]
    ) -> Optional[ClinicalRecommendation]:
        """Assess mass effect."""
        if not finding.characteristics.get('mass_effect', False):
            return None
        
        return ClinicalRecommendation(
            recommendation="Evaluate for increased intracranial pressure",
            rationale="Mass effect detected suggesting elevated ICP",
            urgency=UrgencyLevel.URGENT,
            confidence=DiagnosisConfidence.HIGH,
            supporting_findings=[finding],
            additional_tests=["Ophthalmology consultation for papilledema"],
            follow_up_interval="24 hours"
        )
    
    def _enhancement_rule(
        self,
        finding: ClinicalFinding,
        context: Dict[str, Any]
    ) -> Optional[ClinicalRecommendation]:
        """Assess contrast enhancement."""
        if not finding.characteristics.get('enhancement', False):
            return None
        
        enhancement_pattern = finding.characteristics.get('enhancement_pattern', 'unknown')
        
        if enhancement_pattern == 'ring_enhancing':
            recommendation = "Consider infectious or neoplastic etiology"
            additional_tests = ["MRI with contrast", "Laboratory workup for infection"]
        else:
            recommendation = "Further characterization with advanced imaging"
            additional_tests = ["MRI with contrast", "Perfusion imaging"]
        
        return ClinicalRecommendation(
            recommendation=recommendation,
            rationale=f"Contrast enhancement pattern: {enhancement_pattern}",
            urgency=UrgencyLevel.SEMI_URGENT,
            confidence=DiagnosisConfidence.MODERATE,
            supporting_findings=[finding],
            additional_tests=additional_tests,
            follow_up_interval="1 week"
        )
    
    def _determine_confidence(self, finding: ClinicalFinding) -> DiagnosisConfidence:
        """Determine confidence level based on finding characteristics."""
        confidence_score = finding.confidence
        
        if confidence_score >= 0.9:
            return DiagnosisConfidence.VERY_HIGH
        elif confidence_score >= 0.8:
            return DiagnosisConfidence.HIGH
        elif confidence_score >= 0.6:
            return DiagnosisConfidence.MODERATE
        elif confidence_score >= 0.4:
            return DiagnosisConfidence.LOW
        else:
            return DiagnosisConfidence.VERY_LOW
    
    def _get_follow_up_interval(self, urgency: UrgencyLevel) -> str:
        """Get follow-up interval based on urgency."""
        intervals = {
            UrgencyLevel.EMERGENCY: "Immediate",
            UrgencyLevel.URGENT: "24-48 hours",
            UrgencyLevel.SEMI_URGENT: "1 week",
            UrgencyLevel.ROUTINE: "3 months"
        }
        return intervals.get(urgency, "As clinically indicated")
    
    def _deduplicate_recommendations(
        self,
        recommendations: List[ClinicalRecommendation]
    ) -> List[ClinicalRecommendation]:
        """Remove duplicate recommendations."""
        seen = set()
        unique = []
        
        for rec in recommendations:
            key = (rec.recommendation, rec.urgency.value)
            if key not in seen:
                seen.add(key)
                unique.append(rec)
        
        return unique


class AIDecisionSupport:
    """
    AI-driven clinical decision support system.
    """
    
    def __init__(
        self,
        model_paths: Dict[str, str],
        confidence_threshold: float = 0.7,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize AI decision support system.
        
        Args:
            model_paths: Dictionary of model paths for different tasks
            confidence_threshold: Minimum confidence for recommendations
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.model_paths = model_paths
        self.confidence_threshold = confidence_threshold
        self._session_id = session_id
        self._user_id = user_id
        
        # Load models
        self.models = self._load_models()
        
        # Initialize rule engine
        self.rule_engine = ClinicalRuleEngine()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="AI decision support system initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={'model_paths': list(model_paths.keys())}
        )
    
    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Load AI models for decision support."""
        models = {}
        
        for task, model_path in self.model_paths.items():
            try:
                model = torch.jit.load(model_path)
                model.eval()
                models[task] = model
            except Exception as e:
                warnings.warn(f"Failed to load model for {task}: {str(e)}")
        
        return models
    
    def generate_diagnosis_report(
        self,
        image_data: torch.Tensor,
        clinical_context: Dict[str, Any],
        patient_id: str,
        study_id: str
    ) -> DiagnosisReport:
        """
        Generate comprehensive diagnosis report.
        
        Args:
            image_data: Medical image tensor
            clinical_context: Clinical context and history
            patient_id: Patient identifier
            study_id: Study identifier
            
        Returns:
            Comprehensive diagnosis report
        """
        try:
            # Extract findings using AI models
            findings = self._extract_findings(image_data, clinical_context)
            
            # Generate AI-based recommendations
            ai_recommendations = self._generate_ai_recommendations(findings, clinical_context)
            
            # Apply rule-based recommendations
            rule_recommendations = self.rule_engine.apply_rules(findings, clinical_context)
            
            # Combine and prioritize recommendations
            all_recommendations = self._combine_recommendations(ai_recommendations, rule_recommendations)
            
            # Generate primary diagnosis
            primary_diagnosis = self._determine_primary_diagnosis(findings, all_recommendations)
            
            # Generate differential diagnoses
            differential_diagnoses = self._generate_differential_diagnoses(findings, clinical_context)
            
            # Determine overall confidence
            overall_confidence = self._calculate_overall_confidence(findings, all_recommendations)
            
            # Risk stratification
            risk_stratification = self._perform_risk_stratification(findings, clinical_context)
            
            # Create report
            report = DiagnosisReport(
                patient_id=patient_id,
                study_id=study_id,
                timestamp=datetime.now(),
                primary_diagnosis=primary_diagnosis,
                differential_diagnoses=differential_diagnoses,
                findings=findings,
                recommendations=all_recommendations,
                overall_confidence=overall_confidence,
                risk_stratification=risk_stratification,
                clinical_notes=self._generate_clinical_notes(findings, all_recommendations)
            )
            
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.INFO,
                message="Diagnosis report generated",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'patient_id': patient_id,
                    'study_id': study_id,
                    'findings_count': len(findings),
                    'recommendations_count': len(all_recommendations),
                    'primary_diagnosis': primary_diagnosis
                }
            )
            
            return report
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.ERROR,
                message=f"Diagnosis report generation failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'patient_id': patient_id,
                    'study_id': study_id,
                    'error': str(e)
                }
            )
            raise
    
    def _extract_findings(
        self,
        image_data: torch.Tensor,
        clinical_context: Dict[str, Any]
    ) -> List[ClinicalFinding]:
        """Extract clinical findings using AI models."""
        findings = []
        
        # Detection and segmentation
        if 'detection' in self.models:
            detections = self._run_detection(image_data)
            findings.extend(self._convert_detections_to_findings(detections))
        
        # Classification
        if 'classification' in self.models:
            classifications = self._run_classification(image_data)
            findings.extend(self._convert_classifications_to_findings(classifications))
        
        # Segmentation for volume analysis
        if 'segmentation' in self.models:
            segmentations = self._run_segmentation(image_data)
            findings.extend(self._analyze_segmentations(segmentations, image_data))
        
        return findings
    
    def _run_detection(self, image_data: torch.Tensor) -> Dict[str, Any]:
        """Run detection model."""
        with torch.no_grad():
            output = self.models['detection'](image_data)
        return self._post_process_detection(output)
    
    def _run_classification(self, image_data: torch.Tensor) -> Dict[str, Any]:
        """Run classification model."""
        with torch.no_grad():
            output = self.models['classification'](image_data)
        return self._post_process_classification(output)
    
    def _run_segmentation(self, image_data: torch.Tensor) -> torch.Tensor:
        """Run segmentation model."""
        with torch.no_grad():
            output = self.models['segmentation'](image_data)
        return torch.softmax(output, dim=1)
    
    def _post_process_detection(self, output: torch.Tensor) -> Dict[str, Any]:
        """Post-process detection model output."""
        # Placeholder for detection post-processing
        # Would include NMS, confidence filtering, etc.
        return {'detections': []}
    
    def _post_process_classification(self, output: torch.Tensor) -> Dict[str, Any]:
        """Post-process classification model output."""
        probabilities = torch.softmax(output, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities, dim=-1)[0]
        
        return {
            'predicted_class': predicted_class.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()
        }
    
    def _convert_detections_to_findings(self, detections: Dict[str, Any]) -> List[ClinicalFinding]:
        """Convert detection results to clinical findings."""
        findings = []
        # Implementation would convert detection bounding boxes to clinical findings
        return findings
    
    def _convert_classifications_to_findings(self, classifications: Dict[str, Any]) -> List[ClinicalFinding]:
        """Convert classification results to clinical findings."""
        findings = []
        
        if classifications['confidence'] >= self.confidence_threshold:
            # Map class index to finding type
            class_to_finding = {
                0: FindingType.NORMAL,
                1: FindingType.TUMOR,
                2: FindingType.HEMORRHAGE,
                3: FindingType.INFARCT
            }
            
            finding_type = class_to_finding.get(
                classifications['predicted_class'],
                FindingType.NORMAL
            )
            
            finding = ClinicalFinding(
                finding_type=finding_type,
                location=(0.0, 0.0, 0.0),  # Would be extracted from model
                size=0.0,  # Would be calculated
                confidence=classifications['confidence'],
                description=f"AI classification: {finding_type.value}"
            )
            
            findings.append(finding)
        
        return findings
    
    def _analyze_segmentations(
        self,
        segmentations: torch.Tensor,
        image_data: torch.Tensor
    ) -> List[ClinicalFinding]:
        """Analyze segmentation masks for clinical findings."""
        findings = []
        
        # Extract connected components and analyze
        for class_idx in range(1, segmentations.shape[1]):  # Skip background
            mask = segmentations[0, class_idx] > 0.5
            
            if mask.sum() > 0:
                # Calculate volume and centroid
                volume = mask.sum().item() * np.prod(image_data.meta.get('pixdim', [1, 1, 1])[1:4])
                
                # Find centroid
                coords = torch.nonzero(mask, as_tuple=False).float()
                centroid = coords.mean(dim=0).tolist()
                
                # Map class to finding type
                class_to_finding = {
                    1: FindingType.TUMOR,
                    2: FindingType.LESION,
                    3: FindingType.HEMORRHAGE
                }
                
                finding_type = class_to_finding.get(class_idx, FindingType.LESION)
                
                finding = ClinicalFinding(
                    finding_type=finding_type,
                    location=tuple(centroid),
                    size=volume,
                    confidence=segmentations[0, class_idx][mask].mean().item(),
                    description=f"Segmented {finding_type.value}, volume: {volume:.2f} mm³"
                )
                
                findings.append(finding)
        
        return findings
    
    def _generate_ai_recommendations(
        self,
        findings: List[ClinicalFinding],
        clinical_context: Dict[str, Any]
    ) -> List[ClinicalRecommendation]:
        """Generate AI-based recommendations."""
        recommendations = []
        
        # Example AI recommendation logic
        for finding in findings:
            if finding.confidence >= self.confidence_threshold:
                if finding.finding_type == FindingType.TUMOR and finding.size > 20:
                    rec = ClinicalRecommendation(
                        recommendation="Consider oncology consultation",
                        rationale=f"AI detected tumor with high confidence ({finding.confidence:.2f})",
                        urgency=UrgencyLevel.SEMI_URGENT,
                        confidence=DiagnosisConfidence.HIGH,
                        supporting_findings=[finding],
                        specialist_referral="Oncology"
                    )
                    recommendations.append(rec)
        
        return recommendations
    
    def _combine_recommendations(
        self,
        ai_recommendations: List[ClinicalRecommendation],
        rule_recommendations: List[ClinicalRecommendation]
    ) -> List[ClinicalRecommendation]:
        """Combine and prioritize recommendations from different sources."""
        all_recommendations = ai_recommendations + rule_recommendations
        
        # Sort by urgency and confidence
        urgency_priority = {
            UrgencyLevel.EMERGENCY: 0,
            UrgencyLevel.URGENT: 1,
            UrgencyLevel.SEMI_URGENT: 2,
            UrgencyLevel.ROUTINE: 3
        }
        
        confidence_priority = {
            DiagnosisConfidence.VERY_HIGH: 0,
            DiagnosisConfidence.HIGH: 1,
            DiagnosisConfidence.MODERATE: 2,
            DiagnosisConfidence.LOW: 3,
            DiagnosisConfidence.VERY_LOW: 4
        }
        
        all_recommendations.sort(
            key=lambda x: (urgency_priority[x.urgency], confidence_priority[x.confidence])
        )
        
        return all_recommendations[:10]  # Limit to top 10 recommendations
    
    def _determine_primary_diagnosis(
        self,
        findings: List[ClinicalFinding],
        recommendations: List[ClinicalRecommendation]
    ) -> str:
        """Determine primary diagnosis from findings and recommendations."""
        if not findings:
            return "No significant abnormalities detected"
        
        # Find the most significant finding
        most_significant = max(findings, key=lambda f: f.confidence * (f.size if f.size > 0 else 1))
        
        diagnosis_mapping = {
            FindingType.NORMAL: "Normal study",
            FindingType.TUMOR: f"Space-occupying lesion, {most_significant.size:.1f} mm",
            FindingType.HEMORRHAGE: f"Intracranial hemorrhage, {most_significant.size:.1f} ml",
            FindingType.INFARCT: "Acute ischemic stroke",
            FindingType.LESION: "Focal brain lesion"
        }
        
        return diagnosis_mapping.get(most_significant.finding_type, "Abnormal findings")
    
    def _generate_differential_diagnoses(
        self,
        findings: List[ClinicalFinding],
        clinical_context: Dict[str, Any]
    ) -> List[str]:
        """Generate differential diagnoses."""
        differentials = []
        
        for finding in findings:
            if finding.finding_type == FindingType.TUMOR:
                differentials.extend([
                    "Primary brain tumor (glioma)",
                    "Metastatic disease",
                    "Meningioma",
                    "Lymphoma"
                ])
            elif finding.finding_type == FindingType.HEMORRHAGE:
                differentials.extend([
                    "Hypertensive hemorrhage",
                    "Arteriovenous malformation",
                    "Aneurysmal rupture",
                    "Hemorrhagic metastasis"
                ])
        
        return list(set(differentials))[:5]  # Limit to 5 unique differentials
    
    def _calculate_overall_confidence(
        self,
        findings: List[ClinicalFinding],
        recommendations: List[ClinicalRecommendation]
    ) -> DiagnosisConfidence:
        """Calculate overall confidence in diagnosis."""
        if not findings:
            return DiagnosisConfidence.LOW
        
        avg_confidence = np.mean([f.confidence for f in findings])
        
        if avg_confidence >= 0.9:
            return DiagnosisConfidence.VERY_HIGH
        elif avg_confidence >= 0.8:
            return DiagnosisConfidence.HIGH
        elif avg_confidence >= 0.6:
            return DiagnosisConfidence.MODERATE
        elif avg_confidence >= 0.4:
            return DiagnosisConfidence.LOW
        else:
            return DiagnosisConfidence.VERY_LOW
    
    def _perform_risk_stratification(
        self,
        findings: List[ClinicalFinding],
        clinical_context: Dict[str, Any]
    ) -> str:
        """Perform clinical risk stratification."""
        high_risk_conditions = [
            FindingType.HEMORRHAGE,
            FindingType.TUMOR,
            FindingType.INFARCT
        ]
        
        emergency_findings = [
            f for f in findings 
            if f.finding_type in high_risk_conditions and f.confidence > 0.7
        ]
        
        if emergency_findings:
            return "High risk - requires immediate evaluation"
        elif findings:
            return "Moderate risk - requires timely follow-up"
        else:
            return "Low risk - routine follow-up"
    
    def _generate_clinical_notes(
        self,
        findings: List[ClinicalFinding],
        recommendations: List[ClinicalRecommendation]
    ) -> str:
        """Generate clinical notes summary."""
        notes = []
        
        if findings:
            notes.append(f"AI analysis identified {len(findings)} significant finding(s).")
            
            for finding in findings[:3]:  # Top 3 findings
                notes.append(
                    f"- {finding.finding_type.value.title()} "
                    f"at location {finding.location} with {finding.confidence:.1%} confidence."
                )
        
        if recommendations:
            urgent_recs = [r for r in recommendations if r.urgency in [UrgencyLevel.EMERGENCY, UrgencyLevel.URGENT]]
            if urgent_recs:
                notes.append(f"{len(urgent_recs)} urgent recommendation(s) generated.")
        
        return " ".join(notes)


def create_clinical_decision_support(
    model_paths: Dict[str, str],
    rules_config: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.7,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> AIDecisionSupport:
    """
    Create clinical decision support system.
    
    Args:
        model_paths: Dictionary of model paths
        rules_config: Clinical rules configuration
        confidence_threshold: Confidence threshold for AI recommendations
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured clinical decision support system
    """
    return AIDecisionSupport(
        model_paths=model_paths,
        confidence_threshold=confidence_threshold,
        session_id=session_id,
        user_id=user_id
    )