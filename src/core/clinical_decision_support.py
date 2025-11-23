"""
Clinical Decision Support System

Advanced clinical decision support with rule engine,
evidence-based recommendations, and integration with
medical knowledge bases.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.audit import audit_logger

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    """Types of clinical recommendations."""
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    FOLLOW_UP = "follow_up"
    CONSULTATION = "consultation"
    LABORATORY = "laboratory"
    IMAGING = "imaging"
    MEDICATION = "medication"
    LIFESTYLE = "lifestyle"

class EvidenceLevel(Enum):
    """Levels of clinical evidence."""
    LEVEL_1A = "1a"  # Systematic review of RCTs
    LEVEL_1B = "1b"  # Individual RCT
    LEVEL_2A = "2a"  # Systematic review of cohort studies
    LEVEL_2B = "2b"  # Individual cohort study
    LEVEL_3A = "3a"  # Systematic review of case-control studies
    LEVEL_3B = "3b"  # Individual case-control study
    LEVEL_4 = "4"    # Case series
    LEVEL_5 = "5"    # Expert opinion

class UrgencyLevel(Enum):
    """Urgency levels for recommendations."""
    CRITICAL = "critical"      # Immediate action required
    URGENT = "urgent"          # Action needed within hours
    MODERATE = "moderate"      # Action needed within days
    ROUTINE = "routine"        # Standard timeline
    INFORMATIONAL = "info"     # For awareness only

@dataclass
class ClinicalFinding:
    """Represents a clinical finding from analysis."""
    finding_type: str
    description: str
    location: Optional[str] = None
    severity: Optional[str] = None
    confidence: float = 0.0
    measurement: Optional[Dict[str, Any]] = None
    reference_ranges: Optional[Dict[str, Any]] = None
    abnormal: bool = False
    
@dataclass
class ClinicalContext:
    """Patient clinical context for decision support."""
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    medical_history: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    study_type: Optional[str] = None
    clinical_indication: Optional[str] = None

@dataclass
class ClinicalRecommendation:
    """Clinical recommendation with evidence."""
    recommendation_id: str
    recommendation_type: RecommendationType
    title: str
    description: str
    rationale: str
    evidence_level: EvidenceLevel
    urgency: UrgencyLevel
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    follow_up_timeline: Optional[str] = None
    estimated_cost: Optional[float] = None
    alternative_options: List[str] = field(default_factory=list)
    
class ClinicalRuleEngine:
    """Rule engine for clinical decision support."""
    
    def __init__(self):
        """Initialize clinical rule engine."""
        self.rules = []
        self.knowledge_base = {}
        self.load_clinical_rules()
        self.load_knowledge_base()
    
    def load_clinical_rules(self):
        """Load clinical decision rules."""
        # Imaging findings rules
        self.rules.extend([
            {
                "rule_id": "mass_lesion_followup",
                "condition": lambda findings, context: any(
                    "mass" in f.finding_type.lower() or "lesion" in f.finding_type.lower()
                    for f in findings
                ),
                "recommendation": {
                    "type": RecommendationType.FOLLOW_UP,
                    "title": "Follow-up for Mass Lesion",
                    "description": "Recommend follow-up imaging or biopsy for detected mass lesion",
                    "evidence_level": EvidenceLevel.LEVEL_2A,
                    "urgency": UrgencyLevel.MODERATE,
                    "rationale": "Mass lesions require further evaluation to rule out malignancy"
                }
            },
            {
                "rule_id": "large_lesion_urgent",
                "condition": lambda findings, context: any(
                    f.measurement and f.measurement.get("diameter_mm", 0) > 30
                    for f in findings if "lesion" in f.finding_type.lower()
                ),
                "recommendation": {
                    "type": RecommendationType.CONSULTATION,
                    "title": "Urgent Specialist Consultation",
                    "description": "Large lesion (>30mm) requires urgent specialist evaluation",
                    "evidence_level": EvidenceLevel.LEVEL_1B,
                    "urgency": UrgencyLevel.URGENT,
                    "rationale": "Large lesions have higher probability of malignancy"
                }
            },
            {
                "rule_id": "multiple_lesions",
                "condition": lambda findings, context: len([
                    f for f in findings if "lesion" in f.finding_type.lower()
                ]) > 3,
                "recommendation": {
                    "type": RecommendationType.IMAGING,
                    "title": "Additional Imaging Studies",
                    "description": "Multiple lesions detected, consider whole-body imaging",
                    "evidence_level": EvidenceLevel.LEVEL_2B,
                    "urgency": UrgencyLevel.MODERATE,
                    "rationale": "Multiple lesions may suggest systemic disease"
                }
            }
        ])
        
        # Age-specific rules
        self.rules.extend([
            {
                "rule_id": "elderly_contrast_caution",
                "condition": lambda findings, context: (
                    context.patient_age and context.patient_age > 70
                ),
                "recommendation": {
                    "type": RecommendationType.LABORATORY,
                    "title": "Renal Function Assessment",
                    "description": "Check renal function before contrast administration in elderly patients",
                    "evidence_level": EvidenceLevel.LEVEL_1A,
                    "urgency": UrgencyLevel.ROUTINE,
                    "rationale": "Elderly patients at higher risk for contrast-induced nephropathy"
                }
            },
            {
                "rule_id": "pediatric_radiation_concern",
                "condition": lambda findings, context: (
                    context.patient_age and context.patient_age < 18
                ),
                "recommendation": {
                    "type": RecommendationType.DIAGNOSTIC,
                    "title": "Consider Alternative Imaging",
                    "description": "Consider MRI or ultrasound to minimize radiation exposure",
                    "evidence_level": EvidenceLevel.LEVEL_1A,
                    "urgency": UrgencyLevel.ROUTINE,
                    "rationale": "Children are more sensitive to radiation effects"
                }
            }
        ])
        
        # Risk factor-based rules
        self.rules.extend([
            {
                "rule_id": "smoking_lung_screening",
                "condition": lambda findings, context: (
                    "smoking" in [rf.lower() for rf in context.risk_factors] and
                    context.study_type and "chest" in context.study_type.lower()
                ),
                "recommendation": {
                    "type": RecommendationType.FOLLOW_UP,
                    "title": "Lung Cancer Screening",
                    "description": "Consider annual lung cancer screening for high-risk smoking history",
                    "evidence_level": EvidenceLevel.LEVEL_1A,
                    "urgency": UrgencyLevel.ROUTINE,
                    "rationale": "Smoking history significantly increases lung cancer risk"
                }
            },
            {
                "rule_id": "diabetes_contrast_prep",
                "condition": lambda findings, context: (
                    "diabetes" in [mh.lower() for mh in context.medical_history]
                ),
                "recommendation": {
                    "type": RecommendationType.MEDICATION,
                    "title": "Contrast Preparation for Diabetes",
                    "description": "Review metformin use and renal function before contrast",
                    "evidence_level": EvidenceLevel.LEVEL_1B,
                    "urgency": UrgencyLevel.MODERATE,
                    "rationale": "Metformin interaction with contrast requires careful management"
                }
            }
        ])
    
    def load_knowledge_base(self):
        """Load medical knowledge base."""
        self.knowledge_base = {
            "differential_diagnosis": {
                "lung_nodule": [
                    {"diagnosis": "Lung cancer", "probability": 0.15, "age_factor": 1.2},
                    {"diagnosis": "Granuloma", "probability": 0.30, "endemic_factor": 1.5},
                    {"diagnosis": "Infection", "probability": 0.25, "immunocompromised_factor": 1.8},
                    {"diagnosis": "Benign tumor", "probability": 0.10, "age_factor": 0.8},
                    {"diagnosis": "Metastasis", "probability": 0.20, "cancer_history_factor": 2.0}
                ]
            },
            "reference_ranges": {
                "lesion_size_risk": {
                    "low_risk": {"min": 0, "max": 8},      # mm
                    "moderate_risk": {"min": 8, "max": 30},
                    "high_risk": {"min": 30, "max": 1000}
                },
                "age_groups": {
                    "pediatric": {"min": 0, "max": 18},
                    "adult": {"min": 18, "max": 65},
                    "elderly": {"min": 65, "max": 120}
                }
            },
            "follow_up_guidelines": {
                "lung_nodule": {
                    "size_mm": {
                        "< 6": "No routine follow-up",
                        "6-8": "12 months CT",
                        "8-20": "3-6 months CT",
                        "> 20": "PET/CT or biopsy"
                    }
                }
            }
        }
    
    def evaluate_rules(
        self, 
        findings: List[ClinicalFinding], 
        context: ClinicalContext
    ) -> List[ClinicalRecommendation]:
        """Evaluate clinical rules and generate recommendations."""
        recommendations = []
        
        for rule in self.rules:
            try:
                if rule["condition"](findings, context):
                    rec_data = rule["recommendation"]
                    
                    # Calculate confidence based on evidence strength and context
                    confidence = self._calculate_recommendation_confidence(
                        rec_data["evidence_level"], findings, context
                    )
                    
                    recommendation = ClinicalRecommendation(
                        recommendation_id=f"{rule['rule_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        recommendation_type=rec_data["type"],
                        title=rec_data["title"],
                        description=rec_data["description"],
                        rationale=rec_data["rationale"],
                        evidence_level=rec_data["evidence_level"],
                        urgency=rec_data["urgency"],
                        confidence=confidence
                    )
                    
                    recommendations.append(recommendation)
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.get('rule_id', 'unknown')}: {e}")
        
        return recommendations
    
    def _calculate_recommendation_confidence(
        self,
        evidence_level: EvidenceLevel,
        findings: List[ClinicalFinding],
        context: ClinicalContext
    ) -> float:
        """Calculate confidence score for recommendation."""
        # Base confidence from evidence level
        evidence_scores = {
            EvidenceLevel.LEVEL_1A: 0.95,
            EvidenceLevel.LEVEL_1B: 0.90,
            EvidenceLevel.LEVEL_2A: 0.85,
            EvidenceLevel.LEVEL_2B: 0.80,
            EvidenceLevel.LEVEL_3A: 0.75,
            EvidenceLevel.LEVEL_3B: 0.70,
            EvidenceLevel.LEVEL_4: 0.60,
            EvidenceLevel.LEVEL_5: 0.50
        }
        
        base_confidence = evidence_scores.get(evidence_level, 0.50)
        
        # Adjust based on finding confidence
        avg_finding_confidence = np.mean([f.confidence for f in findings]) if findings else 0.5
        
        # Adjust based on context completeness
        context_completeness = self._assess_context_completeness(context)
        
        # Combined confidence
        final_confidence = base_confidence * 0.6 + avg_finding_confidence * 0.3 + context_completeness * 0.1
        
        return min(max(final_confidence, 0.0), 1.0)
    
    def _assess_context_completeness(self, context: ClinicalContext) -> float:
        """Assess how complete the clinical context is."""
        total_fields = 9  # Number of context fields
        filled_fields = 0
        
        if context.patient_age is not None:
            filled_fields += 1
        if context.patient_sex:
            filled_fields += 1
        if context.medical_history:
            filled_fields += 1
        if context.current_medications:
            filled_fields += 1
        if context.allergies:
            filled_fields += 1
        if context.symptoms:
            filled_fields += 1
        if context.risk_factors:
            filled_fields += 1
        if context.study_type:
            filled_fields += 1
        if context.clinical_indication:
            filled_fields += 1
        
        return filled_fields / total_fields

class DifferentialDiagnosisEngine:
    """Engine for generating differential diagnoses."""
    
    def __init__(self):
        """Initialize differential diagnosis engine."""
        self.knowledge_base = {}
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_diagnostic_knowledge()
    
    def load_diagnostic_knowledge(self):
        """Load diagnostic knowledge base."""
        self.knowledge_base = {
            "conditions": {
                "lung_cancer": {
                    "typical_findings": ["mass", "nodule", "pleural_effusion", "lymphadenopathy"],
                    "risk_factors": ["smoking", "asbestos_exposure", "family_history"],
                    "age_distribution": {"mean": 65, "std": 15},
                    "sex_ratio": {"male": 1.2, "female": 1.0}
                },
                "pneumonia": {
                    "typical_findings": ["consolidation", "ground_glass", "air_bronchograms"],
                    "risk_factors": ["immunocompromised", "elderly", "chronic_disease"],
                    "age_distribution": {"mean": 50, "std": 25},
                    "sex_ratio": {"male": 1.0, "female": 1.0}
                },
                "pulmonary_embolism": {
                    "typical_findings": ["filling_defect", "wedge_infarct", "pulmonary_hypertension"],
                    "risk_factors": ["immobilization", "surgery", "malignancy", "pregnancy"],
                    "age_distribution": {"mean": 55, "std": 20},
                    "sex_ratio": {"male": 0.8, "female": 1.2}
                }
            }
        }
    
    def generate_differential(
        self,
        findings: List[ClinicalFinding],
        context: ClinicalContext
    ) -> List[Dict[str, Any]]:
        """Generate differential diagnosis list."""
        differentials = []
        
        for condition_name, condition_data in self.knowledge_base["conditions"].items():
            probability = self._calculate_condition_probability(
                condition_name, condition_data, findings, context
            )
            
            if probability > 0.1:  # Only include if probability > 10%
                differentials.append({
                    "condition": condition_name,
                    "probability": probability,
                    "supporting_findings": self._get_supporting_findings(
                        condition_data, findings
                    ),
                    "risk_factors": self._get_matching_risk_factors(
                        condition_data, context
                    )
                })
        
        # Sort by probability
        differentials.sort(key=lambda x: x["probability"], reverse=True)
        
        return differentials[:10]  # Top 10 differentials
    
    def _calculate_condition_probability(
        self,
        condition_name: str,
        condition_data: Dict,
        findings: List[ClinicalFinding],
        context: ClinicalContext
    ) -> float:
        """Calculate probability of a condition given findings and context."""
        base_probability = 0.1  # Base prevalence
        
        # Finding match score
        finding_score = self._calculate_finding_match_score(condition_data, findings)
        
        # Risk factor score
        risk_score = self._calculate_risk_factor_score(condition_data, context)
        
        # Demographic score
        demo_score = self._calculate_demographic_score(condition_data, context)
        
        # Combine scores
        probability = (finding_score * 0.6 + risk_score * 0.3 + demo_score * 0.1) * base_probability
        
        return min(probability, 0.95)  # Cap at 95%
    
    def _calculate_finding_match_score(
        self,
        condition_data: Dict,
        findings: List[ClinicalFinding]
    ) -> float:
        """Calculate how well findings match the condition."""
        if not findings:
            return 0.0
        
        typical_findings = condition_data.get("typical_findings", [])
        if not typical_findings:
            return 0.0
        
        # Semantic similarity between findings and typical findings
        finding_texts = [f.description for f in findings]
        typical_texts = typical_findings
        
        if not finding_texts or not typical_texts:
            return 0.0
        
        # Calculate semantic similarity
        finding_embeddings = self.semantic_model.encode(finding_texts)
        typical_embeddings = self.semantic_model.encode(typical_texts)
        
        # Find best matches
        similarities = []
        for finding_emb in finding_embeddings:
            max_sim = max([
                np.dot(finding_emb, typ_emb) / (np.linalg.norm(finding_emb) * np.linalg.norm(typ_emb))
                for typ_emb in typical_embeddings
            ])
            similarities.append(max_sim)
        
        return np.mean(similarities)
    
    def _calculate_risk_factor_score(
        self,
        condition_data: Dict,
        context: ClinicalContext
    ) -> float:
        """Calculate risk factor match score."""
        condition_risk_factors = condition_data.get("risk_factors", [])
        patient_risk_factors = context.risk_factors
        
        if not condition_risk_factors or not patient_risk_factors:
            return 0.0
        
        matches = 0
        for patient_rf in patient_risk_factors:
            for condition_rf in condition_risk_factors:
                if patient_rf.lower() in condition_rf.lower() or condition_rf.lower() in patient_rf.lower():
                    matches += 1
                    break
        
        return matches / len(condition_risk_factors)
    
    def _calculate_demographic_score(
        self,
        condition_data: Dict,
        context: ClinicalContext
    ) -> float:
        """Calculate demographic likelihood score."""
        score = 1.0
        
        # Age factor
        if context.patient_age and "age_distribution" in condition_data:
            age_dist = condition_data["age_distribution"]
            age_mean = age_dist["mean"]
            age_std = age_dist["std"]
            
            # Gaussian probability
            age_prob = np.exp(-0.5 * ((context.patient_age - age_mean) / age_std) ** 2)
            score *= age_prob
        
        # Sex factor
        if context.patient_sex and "sex_ratio" in condition_data:
            sex_ratio = condition_data["sex_ratio"]
            if context.patient_sex.lower() in sex_ratio:
                score *= sex_ratio[context.patient_sex.lower()]
        
        return score
    
    def _get_supporting_findings(
        self,
        condition_data: Dict,
        findings: List[ClinicalFinding]
    ) -> List[str]:
        """Get findings that support this condition."""
        typical_findings = condition_data.get("typical_findings", [])
        supporting = []
        
        for finding in findings:
            for typical in typical_findings:
                if typical.lower() in finding.description.lower():
                    supporting.append(finding.description)
                    break
        
        return supporting
    
    def _get_matching_risk_factors(
        self,
        condition_data: Dict,
        context: ClinicalContext
    ) -> List[str]:
        """Get risk factors that match this condition."""
        condition_risk_factors = condition_data.get("risk_factors", [])
        patient_risk_factors = context.risk_factors
        
        matching = []
        for patient_rf in patient_risk_factors:
            for condition_rf in condition_risk_factors:
                if patient_rf.lower() in condition_rf.lower() or condition_rf.lower() in patient_rf.lower():
                    matching.append(patient_rf)
                    break
        
        return matching

class ClinicalDecisionSupport:
    """Main clinical decision support system."""
    
    def __init__(self):
        """Initialize clinical decision support system."""
        self.rule_engine = ClinicalRuleEngine()
        self.diagnosis_engine = DifferentialDiagnosisEngine()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
    
    async def analyze_case(
        self,
        findings: List[ClinicalFinding],
        context: ClinicalContext,
        analysis_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive clinical decision support analysis."""
        
        await audit_logger.log_event(
            "cds_analysis_started",
            {
                "findings_count": len(findings),
                "patient_age": context.patient_age,
                "study_type": context.study_type
            }
        )
        
        # Generate recommendations
        recommendations = self.rule_engine.evaluate_rules(findings, context)
        
        # Generate differential diagnoses
        differentials = self.diagnosis_engine.generate_differential(findings, context)
        
        # Detect anomalies in findings
        anomalies = await self._detect_anomalous_findings(findings)
        
        # Calculate overall risk assessment
        risk_assessment = self._calculate_risk_assessment(findings, context, differentials)
        
        # Generate follow-up suggestions
        follow_up = self._generate_follow_up_plan(recommendations, differentials, risk_assessment)
        
        result = {
            "timestamp": datetime.now(),
            "recommendations": [self._recommendation_to_dict(rec) for rec in recommendations],
            "differential_diagnoses": differentials,
            "risk_assessment": risk_assessment,
            "anomalous_findings": anomalies,
            "follow_up_plan": follow_up,
            "confidence_score": self._calculate_overall_confidence(recommendations, differentials),
            "clinical_context": self._context_to_dict(context)
        }
        
        await audit_logger.log_event(
            "cds_analysis_completed",
            {
                "recommendations_count": len(recommendations),
                "differentials_count": len(differentials),
                "overall_confidence": result["confidence_score"]
            }
        )
        
        return result
    
    async def _detect_anomalous_findings(self, findings: List[ClinicalFinding]) -> List[Dict]:
        """Detect anomalous findings using ML."""
        if len(findings) < 3:  # Need minimum findings for anomaly detection
            return []
        
        # Extract numerical features from findings
        features = []
        finding_descriptions = []
        
        for finding in findings:
            feature_vector = [
                finding.confidence,
                len(finding.description),
                1.0 if finding.abnormal else 0.0,
                finding.measurement.get("diameter_mm", 0) if finding.measurement else 0,
                finding.measurement.get("volume_mm3", 0) if finding.measurement else 0
            ]
            
            features.append(feature_vector)
            finding_descriptions.append(finding.description)
        
        features_array = np.array(features)
        
        # Detect anomalies
        anomaly_scores = self.anomaly_detector.fit_predict(features_array)
        
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score == -1:  # Anomaly detected
                anomalies.append({
                    "finding": finding_descriptions[i],
                    "anomaly_reason": "Statistical outlier in finding characteristics",
                    "recommendation": "Consider additional validation or expert review"
                })
        
        return anomalies
    
    def _calculate_risk_assessment(
        self,
        findings: List[ClinicalFinding],
        context: ClinicalContext,
        differentials: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate overall risk assessment."""
        # High-risk findings
        high_risk_keywords = ["mass", "malignancy", "metastasis", "large", "aggressive"]
        high_risk_count = sum(
            1 for finding in findings
            if any(keyword in finding.description.lower() for keyword in high_risk_keywords)
        )
        
        # Risk from differentials
        high_risk_conditions = ["lung_cancer", "malignancy", "aggressive"]
        high_risk_differential_prob = sum(
            diff["probability"] for diff in differentials
            if any(condition in diff["condition"].lower() for condition in high_risk_conditions)
        )
        
        # Age risk factor
        age_risk = 0.0
        if context.patient_age:
            if context.patient_age > 70:
                age_risk = 0.3
            elif context.patient_age > 50:
                age_risk = 0.15
        
        # Overall risk score
        risk_score = min(
            (high_risk_count * 0.2 + high_risk_differential_prob + age_risk) / 3.0,
            1.0
        )
        
        risk_level = "low"
        if risk_score > 0.7:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "moderate"
        
        return {
            "overall_risk_score": risk_score,
            "risk_level": risk_level,
            "high_risk_findings_count": high_risk_count,
            "high_risk_differential_probability": high_risk_differential_prob,
            "age_risk_factor": age_risk,
            "risk_explanation": self._generate_risk_explanation(risk_level, risk_score)
        }
    
    def _generate_risk_explanation(self, risk_level: str, risk_score: float) -> str:
        """Generate human-readable risk explanation."""
        if risk_level == "high":
            return f"High risk ({risk_score:.2f}) due to concerning findings requiring immediate attention"
        elif risk_level == "moderate":
            return f"Moderate risk ({risk_score:.2f}) requiring timely follow-up and monitoring"
        else:
            return f"Low risk ({risk_score:.2f}) with routine follow-up recommended"
    
    def _generate_follow_up_plan(
        self,
        recommendations: List[ClinicalRecommendation],
        differentials: List[Dict],
        risk_assessment: Dict
    ) -> Dict[str, Any]:
        """Generate comprehensive follow-up plan."""
        # Extract timelines from recommendations
        urgent_actions = [
            rec for rec in recommendations
            if rec.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.URGENT]
        ]
        
        moderate_actions = [
            rec for rec in recommendations
            if rec.urgency == UrgencyLevel.MODERATE
        ]
        
        routine_actions = [
            rec for rec in recommendations
            if rec.urgency == UrgencyLevel.ROUTINE
        ]
        
        plan = {
            "immediate_actions": [
                {
                    "action": rec.title,
                    "description": rec.description,
                    "timeline": "Immediate"
                }
                for rec in urgent_actions
            ],
            "short_term_actions": [
                {
                    "action": rec.title,
                    "description": rec.description,
                    "timeline": "Within 1-7 days"
                }
                for rec in moderate_actions
            ],
            "long_term_actions": [
                {
                    "action": rec.title,
                    "description": rec.description,
                    "timeline": "Within 1-3 months"
                }
                for rec in routine_actions
            ],
            "monitoring_plan": self._generate_monitoring_plan(differentials, risk_assessment)
        }
        
        return plan
    
    def _generate_monitoring_plan(self, differentials: List[Dict], risk_assessment: Dict) -> List[Dict]:
        """Generate monitoring plan based on differentials and risk."""
        monitoring = []
        
        # High-probability differentials need monitoring
        for diff in differentials[:3]:  # Top 3 differentials
            if diff["probability"] > 0.3:
                monitoring.append({
                    "condition": diff["condition"],
                    "monitoring_frequency": "3-6 months" if risk_assessment["risk_level"] == "high" else "6-12 months",
                    "monitoring_type": "Imaging follow-up",
                    "parameters_to_monitor": ["Size changes", "New findings", "Progression"]
                })
        
        return monitoring
    
    def _calculate_overall_confidence(
        self,
        recommendations: List[ClinicalRecommendation],
        differentials: List[Dict]
    ) -> float:
        """Calculate overall confidence in the analysis."""
        if not recommendations and not differentials:
            return 0.0
        
        # Average recommendation confidence
        rec_confidence = np.mean([rec.confidence for rec in recommendations]) if recommendations else 0.5
        
        # Differential diagnosis confidence (based on top differential probability)
        diff_confidence = differentials[0]["probability"] if differentials else 0.5
        
        # Combined confidence
        return (rec_confidence * 0.6 + diff_confidence * 0.4)
    
    def _recommendation_to_dict(self, rec: ClinicalRecommendation) -> Dict:
        """Convert recommendation to dictionary."""
        return {
            "id": rec.recommendation_id,
            "type": rec.recommendation_type.value,
            "title": rec.title,
            "description": rec.description,
            "rationale": rec.rationale,
            "evidence_level": rec.evidence_level.value,
            "urgency": rec.urgency.value,
            "confidence": rec.confidence,
            "supporting_evidence": rec.supporting_evidence,
            "contraindications": rec.contraindications,
            "follow_up_timeline": rec.follow_up_timeline,
            "estimated_cost": rec.estimated_cost,
            "alternative_options": rec.alternative_options
        }
    
    def _context_to_dict(self, context: ClinicalContext) -> Dict:
        """Convert clinical context to dictionary."""
        return {
            "patient_age": context.patient_age,
            "patient_sex": context.patient_sex,
            "medical_history": context.medical_history,
            "current_medications": context.current_medications,
            "allergies": context.allergies,
            "symptoms": context.symptoms,
            "risk_factors": context.risk_factors,
            "study_type": context.study_type,
            "clinical_indication": context.clinical_indication
        }

# Global clinical decision support instance
clinical_decision_support = ClinicalDecisionSupport()