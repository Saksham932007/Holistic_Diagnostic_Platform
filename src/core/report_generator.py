"""
Medical Report Generation System

Advanced automated report generation for medical imaging
analysis with clinical templates and natural language processing.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from jinja2 import Environment, FileSystemLoader, Template
import pandas as pd
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch

from src.core.config import settings
from src.core.audit import audit_logger
from src.core.clinical_decision_support import (
    ClinicalFinding, ClinicalContext, ClinicalRecommendation,
    clinical_decision_support
)

logger = logging.getLogger(__name__)

class ReportType(Enum):
    """Types of medical reports."""
    DIAGNOSTIC = "diagnostic"
    SCREENING = "screening"
    FOLLOW_UP = "follow_up"
    COMPARISON = "comparison"
    COMPREHENSIVE = "comprehensive"

class ReportSection(Enum):
    """Standard report sections."""
    HEADER = "header"
    CLINICAL_HISTORY = "clinical_history"
    TECHNIQUE = "technique"
    FINDINGS = "findings"
    IMPRESSION = "impression"
    RECOMMENDATIONS = "recommendations"
    COMPARISON = "comparison"
    LIMITATIONS = "limitations"
    SIGNATURE = "signature"

class ConfidenceLevel(Enum):
    """Confidence levels for findings."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class ReportTemplate:
    """Template for generating reports."""
    template_id: str
    name: str
    report_type: ReportType
    modality: str
    body_part: str
    template_content: str
    required_sections: List[ReportSection]
    optional_sections: List[ReportSection]
    clinical_guidelines: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)

@dataclass
class ReportData:
    """Data container for report generation."""
    patient_info: Dict[str, Any]
    study_info: Dict[str, Any]
    clinical_context: ClinicalContext
    findings: List[ClinicalFinding]
    recommendations: List[ClinicalRecommendation]
    analysis_results: Dict[str, Any]
    comparison_studies: List[Dict[str, Any]] = field(default_factory=list)
    technical_parameters: Dict[str, Any] = field(default_factory=dict)
    radiologist_notes: Optional[str] = None

@dataclass
class GeneratedReport:
    """Generated medical report."""
    report_id: str
    report_type: ReportType
    template_id: str
    timestamp: datetime
    patient_id: str
    study_id: str
    content: str
    sections: Dict[ReportSection, str]
    confidence_score: float
    validation_status: str
    findings_count: int
    recommendations_count: int
    word_count: int
    radiologist_reviewed: bool = False
    approved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class ClinicalLanguageProcessor:
    """Processes and generates clinical language."""
    
    def __init__(self):
        """Initialize clinical language processor."""
        self.clinical_terms = self._load_clinical_terminology()
        self.sentence_patterns = self._load_sentence_patterns()
        
        # Initialize language model for text generation
        try:
            self.text_generator = pipeline(
                "text-generation",
                model="distilgpt2",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not initialize text generator: {e}")
            self.text_generator = None
    
    def _load_clinical_terminology(self) -> Dict[str, List[str]]:
        """Load clinical terminology and synonyms."""
        return {
            "positive_findings": [
                "demonstrates", "shows", "reveals", "exhibits", "displays",
                "is notable for", "is significant for", "is consistent with"
            ],
            "negative_findings": [
                "no evidence of", "no signs of", "unremarkable for",
                "within normal limits", "no abnormality detected"
            ],
            "uncertainty_terms": [
                "possible", "probable", "likely", "suspicious for",
                "cannot exclude", "differential includes", "consider"
            ],
            "comparison_terms": [
                "compared to prior", "in comparison with", "relative to previous",
                "when compared to", "as compared to the prior study"
            ],
            "anatomical_descriptors": [
                "bilateral", "unilateral", "symmetric", "asymmetric",
                "diffuse", "focal", "multifocal", "patchy", "confluent"
            ]
        }
    
    def _load_sentence_patterns(self) -> Dict[str, List[str]]:
        """Load sentence patterns for different types of statements."""
        return {
            "finding_description": [
                "There is {finding} {location}.",
                "The {anatomy} {demonstrates} {finding}.",
                "{Finding} is {observed} {location}.",
                "Notable for {finding} {location}."
            ],
            "measurement": [
                "The {structure} measures approximately {measurement}.",
                "Size of the {structure} is {measurement}.",
                "{Structure} demonstrates {measurement} in {dimension}."
            ],
            "comparison": [
                "{Finding} is {comparison} compared to {prior_date}.",
                "Compared to prior study, {finding} is {status}.",
                "The {structure} has {changed} since {prior_date}."
            ],
            "recommendation": [
                "Recommend {action} for {indication}.",
                "Suggest {action} to {purpose}.",
                "Consider {action} given {rationale}."
            ]
        }
    
    def generate_finding_description(
        self,
        finding: ClinicalFinding,
        confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    ) -> str:
        """Generate natural language description for a finding."""
        
        # Select appropriate language based on confidence
        if confidence == ConfidenceLevel.HIGH:
            verbs = ["demonstrates", "shows", "reveals"]
        elif confidence == ConfidenceLevel.MODERATE:
            verbs = ["appears to show", "suggests", "is consistent with"]
        else:
            verbs = ["possible", "may represent", "cannot exclude"]
        
        # Build description
        description_parts = []
        
        # Location
        if finding.location:
            location_desc = f"in the {finding.location}"
        else:
            location_desc = ""
        
        # Main finding
        verb = verbs[0]  # Select appropriate verb
        main_desc = f"The study {verb} {finding.description} {location_desc}."
        description_parts.append(main_desc)
        
        # Add measurements if available
        if finding.measurement:
            measurement_desc = self._format_measurements(finding.measurement)
            if measurement_desc:
                description_parts.append(measurement_desc)
        
        # Add severity if specified
        if finding.severity:
            severity_desc = f"The finding is characterized as {finding.severity}."
            description_parts.append(severity_desc)
        
        return " ".join(description_parts)
    
    def _format_measurements(self, measurements: Dict[str, Any]) -> str:
        """Format measurement data into natural language."""
        measurement_parts = []
        
        if "diameter_mm" in measurements:
            measurement_parts.append(f"measuring approximately {measurements['diameter_mm']:.1f} mm in diameter")
        
        if "volume_mm3" in measurements:
            volume_cm3 = measurements["volume_mm3"] / 1000
            measurement_parts.append(f"with an estimated volume of {volume_cm3:.1f} cm³")
        
        if "dimensions" in measurements:
            dims = measurements["dimensions"]
            if isinstance(dims, dict) and all(k in dims for k in ["length", "width", "height"]):
                measurement_parts.append(
                    f"measuring {dims['length']:.1f} x {dims['width']:.1f} x {dims['height']:.1f} mm"
                )
        
        return ". ".join(measurement_parts) + "." if measurement_parts else ""
    
    def generate_comparison_text(
        self,
        current_findings: List[ClinicalFinding],
        prior_findings: List[ClinicalFinding],
        time_interval: str
    ) -> str:
        """Generate comparison text with prior studies."""
        
        if not prior_findings:
            return f"No prior studies available for comparison."
        
        comparison_parts = []
        comparison_parts.append(f"Compared to prior study from {time_interval}:")
        
        # Find corresponding findings
        for current_finding in current_findings:
            # Simple matching by finding type and location
            prior_match = None
            for prior_finding in prior_findings:
                if (current_finding.finding_type == prior_finding.finding_type and
                    current_finding.location == prior_finding.location):
                    prior_match = prior_finding
                    break
            
            if prior_match:
                # Generate comparison
                comparison = self._compare_findings(current_finding, prior_match)
                comparison_parts.append(f"- {comparison}")
            else:
                comparison_parts.append(f"- {current_finding.description} is newly identified.")
        
        # Check for resolved findings
        for prior_finding in prior_findings:
            current_match = None
            for current_finding in current_findings:
                if (prior_finding.finding_type == current_finding.finding_type and
                    prior_finding.location == current_finding.location):
                    current_match = current_finding
                    break
            
            if not current_match:
                comparison_parts.append(f"- Previously noted {prior_finding.description} is no longer evident.")
        
        return "\n".join(comparison_parts)
    
    def _compare_findings(
        self,
        current: ClinicalFinding,
        prior: ClinicalFinding
    ) -> str:
        """Compare two findings and generate descriptive text."""
        
        # Compare measurements if available
        if (current.measurement and prior.measurement and 
            "diameter_mm" in both_measurements := (current.measurement, prior.measurement)):
            
            current_size = current.measurement["diameter_mm"]
            prior_size = prior.measurement["diameter_mm"]
            
            size_change = current_size - prior_size
            percent_change = (size_change / prior_size) * 100 if prior_size > 0 else 0
            
            if abs(percent_change) < 10:
                return f"{current.description} is stable in size"
            elif size_change > 0:
                return f"{current.description} has increased in size by {size_change:.1f} mm ({percent_change:.1f}%)"
            else:
                return f"{current.description} has decreased in size by {abs(size_change):.1f} mm ({abs(percent_change):.1f}%)"
        
        # Compare confidence scores
        if current.confidence > prior.confidence:
            return f"{current.description} appears more prominent than previously"
        elif current.confidence < prior.confidence:
            return f"{current.description} appears less prominent than previously"
        else:
            return f"{current.description} appears stable"
    
    def generate_impression_summary(
        self,
        findings: List[ClinicalFinding],
        recommendations: List[ClinicalRecommendation]
    ) -> str:
        """Generate impression summary."""
        
        impression_parts = []
        
        # Categorize findings by severity
        significant_findings = [f for f in findings if f.confidence > 0.7 and f.abnormal]
        incidental_findings = [f for f in findings if f.confidence <= 0.7 or not f.abnormal]
        
        # Main findings
        if significant_findings:
            if len(significant_findings) == 1:
                impression_parts.append(f"Primary finding: {significant_findings[0].description}.")
            else:
                impression_parts.append("Key findings include:")
                for i, finding in enumerate(significant_findings, 1):
                    impression_parts.append(f"{i}. {finding.description}")
        else:
            impression_parts.append("No significant abnormalities identified.")
        
        # Incidental findings
        if incidental_findings:
            impression_parts.append(f"Incidental findings include {len(incidental_findings)} minor observations.")
        
        # Clinical significance
        if any(rec.urgency.value in ["critical", "urgent"] for rec in recommendations):
            impression_parts.append("Findings require urgent clinical attention.")
        elif any(rec.urgency.value == "moderate" for rec in recommendations):
            impression_parts.append("Findings warrant timely follow-up.")
        
        return " ".join(impression_parts)

class ReportTemplateEngine:
    """Manages report templates and generation."""
    
    def __init__(self):
        """Initialize template engine."""
        self.templates_dir = Path(__file__).parent.parent.parent / "templates" / "reports"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default report templates."""
        
        # Chest CT diagnostic template
        chest_ct_template = ReportTemplate(
            template_id="chest_ct_diagnostic",
            name="Chest CT Diagnostic Report",
            report_type=ReportType.DIAGNOSTIC,
            modality="CT",
            body_part="Chest",
            template_content=self._get_chest_ct_template(),
            required_sections=[
                ReportSection.HEADER,
                ReportSection.TECHNIQUE,
                ReportSection.FINDINGS,
                ReportSection.IMPRESSION
            ],
            optional_sections=[
                ReportSection.CLINICAL_HISTORY,
                ReportSection.COMPARISON,
                ReportSection.RECOMMENDATIONS
            ]
        )
        
        self.templates["chest_ct_diagnostic"] = chest_ct_template
        
        # Brain MRI template
        brain_mri_template = ReportTemplate(
            template_id="brain_mri_diagnostic",
            name="Brain MRI Diagnostic Report",
            report_type=ReportType.DIAGNOSTIC,
            modality="MRI",
            body_part="Brain",
            template_content=self._get_brain_mri_template(),
            required_sections=[
                ReportSection.HEADER,
                ReportSection.TECHNIQUE,
                ReportSection.FINDINGS,
                ReportSection.IMPRESSION
            ],
            optional_sections=[
                ReportSection.CLINICAL_HISTORY,
                ReportSection.COMPARISON,
                ReportSection.RECOMMENDATIONS
            ]
        )
        
        self.templates["brain_mri_diagnostic"] = brain_mri_template
    
    def _get_chest_ct_template(self) -> str:
        """Get chest CT template content."""
        return """
CHEST CT REPORT

{% if clinical_history %}
CLINICAL HISTORY:
{{ clinical_history }}
{% endif %}

TECHNIQUE:
{{ technique }}

{% if comparison %}
COMPARISON:
{{ comparison }}
{% endif %}

FINDINGS:
{{ findings }}

IMPRESSION:
{{ impression }}

{% if recommendations %}
RECOMMENDATIONS:
{{ recommendations }}
{% endif %}

{% if limitations %}
LIMITATIONS:
{{ limitations }}
{% endif %}

Electronically signed by: {{ radiologist_name }}
Date: {{ report_date }}
"""
    
    def _get_brain_mri_template(self) -> str:
        """Get brain MRI template content."""
        return """
BRAIN MRI REPORT

{% if clinical_history %}
CLINICAL HISTORY:
{{ clinical_history }}
{% endif %}

TECHNIQUE:
{{ technique }}

{% if comparison %}
COMPARISON:
{{ comparison }}
{% endif %}

FINDINGS:
{{ findings }}

IMPRESSION:
{{ impression }}

{% if recommendations %}
RECOMMENDATIONS:
{{ recommendations }}
{% endif %}

Electronically signed by: {{ radiologist_name }}
Date: {{ report_date }}
"""
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def render_template(
        self,
        template_id: str,
        template_data: Dict[str, Any]
    ) -> str:
        """Render template with data."""
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        jinja_template = self.jinja_env.from_string(template.template_content)
        
        return jinja_template.render(**template_data)

class MedicalReportGenerator:
    """Main medical report generation system."""
    
    def __init__(self):
        """Initialize report generator."""
        self.language_processor = ClinicalLanguageProcessor()
        self.template_engine = ReportTemplateEngine()
        self.generated_reports = {}
    
    async def generate_report(
        self,
        report_data: ReportData,
        template_id: str,
        report_type: ReportType = ReportType.DIAGNOSTIC
    ) -> GeneratedReport:
        """Generate a medical report."""
        
        await audit_logger.log_event(
            "report_generation_started",
            {
                "template_id": template_id,
                "patient_id": report_data.patient_info.get("patient_id"),
                "study_id": report_data.study_info.get("study_id")
            }
        )
        
        try:
            # Get template
            template = self.template_engine.get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Generate report sections
            sections = await self._generate_report_sections(report_data, template)
            
            # Prepare template data
            template_data = {
                "clinical_history": sections.get(ReportSection.CLINICAL_HISTORY, ""),
                "technique": sections.get(ReportSection.TECHNIQUE, ""),
                "findings": sections.get(ReportSection.FINDINGS, ""),
                "impression": sections.get(ReportSection.IMPRESSION, ""),
                "recommendations": sections.get(ReportSection.RECOMMENDATIONS, ""),
                "comparison": sections.get(ReportSection.COMPARISON, ""),
                "limitations": sections.get(ReportSection.LIMITATIONS, ""),
                "radiologist_name": "AI Assistant",
                "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Render final report
            report_content = self.template_engine.render_template(template_id, template_data)
            
            # Calculate confidence score
            confidence_score = self._calculate_report_confidence(report_data, sections)
            
            # Create report object
            report_id = f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{report_data.patient_info.get('patient_id', 'UNKNOWN')}"
            
            generated_report = GeneratedReport(
                report_id=report_id,
                report_type=report_type,
                template_id=template_id,
                timestamp=datetime.now(),
                patient_id=report_data.patient_info.get("patient_id", "Unknown"),
                study_id=report_data.study_info.get("study_id", "Unknown"),
                content=report_content,
                sections=sections,
                confidence_score=confidence_score,
                validation_status="Generated",
                findings_count=len(report_data.findings),
                recommendations_count=len(report_data.recommendations),
                word_count=len(report_content.split())
            )
            
            # Store report
            self.generated_reports[report_id] = generated_report
            
            await audit_logger.log_event(
                "report_generation_completed",
                {
                    "report_id": report_id,
                    "word_count": generated_report.word_count,
                    "confidence_score": confidence_score,
                    "findings_count": len(report_data.findings)
                }
            )
            
            return generated_report
            
        except Exception as e:
            await audit_logger.log_event(
                "report_generation_failed",
                {
                    "template_id": template_id,
                    "error": str(e)
                }
            )
            raise
    
    async def _generate_report_sections(
        self,
        report_data: ReportData,
        template: ReportTemplate
    ) -> Dict[ReportSection, str]:
        """Generate individual report sections."""
        
        sections = {}
        
        # Clinical History
        if ReportSection.CLINICAL_HISTORY in template.required_sections or template.optional_sections:
            sections[ReportSection.CLINICAL_HISTORY] = self._generate_clinical_history(report_data)
        
        # Technique
        if ReportSection.TECHNIQUE in template.required_sections or template.optional_sections:
            sections[ReportSection.TECHNIQUE] = self._generate_technique_section(report_data)
        
        # Findings
        sections[ReportSection.FINDINGS] = await self._generate_findings_section(report_data)
        
        # Impression
        sections[ReportSection.IMPRESSION] = self._generate_impression_section(report_data)
        
        # Recommendations
        if report_data.recommendations:
            sections[ReportSection.RECOMMENDATIONS] = self._generate_recommendations_section(report_data)
        
        # Comparison
        if report_data.comparison_studies:
            sections[ReportSection.COMPARISON] = self._generate_comparison_section(report_data)
        
        return sections
    
    def _generate_clinical_history(self, report_data: ReportData) -> str:
        """Generate clinical history section."""
        history_parts = []
        
        context = report_data.clinical_context
        
        # Age and sex
        if context.patient_age and context.patient_sex:
            history_parts.append(f"{context.patient_age}-year-old {context.patient_sex}")
        
        # Clinical indication
        if context.clinical_indication:
            history_parts.append(f"Clinical indication: {context.clinical_indication}")
        
        # Symptoms
        if context.symptoms:
            symptoms_text = ", ".join(context.symptoms)
            history_parts.append(f"Presenting symptoms: {symptoms_text}")
        
        # Medical history
        if context.medical_history:
            history_text = ", ".join(context.medical_history)
            history_parts.append(f"Medical history: {history_text}")
        
        # Risk factors
        if context.risk_factors:
            risk_text = ", ".join(context.risk_factors)
            history_parts.append(f"Risk factors: {risk_text}")
        
        return ". ".join(history_parts) + "." if history_parts else "Clinical history not provided."
    
    def _generate_technique_section(self, report_data: ReportData) -> str:
        """Generate technique section."""
        technique_parts = []
        
        study_info = report_data.study_info
        tech_params = report_data.technical_parameters
        
        # Study type and modality
        if study_info.get("modality"):
            modality = study_info["modality"]
            study_desc = study_info.get("study_description", "")
            technique_parts.append(f"{modality} examination of the {study_desc}")
        
        # Technical parameters
        if tech_params:
            if "slice_thickness" in tech_params:
                technique_parts.append(f"Slice thickness: {tech_params['slice_thickness']} mm")
            
            if "contrast" in tech_params:
                contrast_info = tech_params["contrast"]
                if contrast_info:
                    technique_parts.append("Intravenous contrast material administered")
                else:
                    technique_parts.append("No intravenous contrast administered")
        
        return ". ".join(technique_parts) + "." if technique_parts else "Standard imaging technique employed."
    
    async def _generate_findings_section(self, report_data: ReportData) -> str:
        """Generate findings section."""
        findings_parts = []
        
        if not report_data.findings:
            return "No significant abnormalities identified."
        
        # Group findings by anatomical region or type
        findings_by_region = {}
        for finding in report_data.findings:
            region = finding.location or "General"
            if region not in findings_by_region:
                findings_by_region[region] = []
            findings_by_region[region].append(finding)
        
        # Generate descriptions for each region
        for region, findings in findings_by_region.items():
            if len(findings_by_region) > 1:
                findings_parts.append(f"{region}:")
            
            for finding in findings:
                confidence = self._determine_confidence_level(finding.confidence)
                description = self.language_processor.generate_finding_description(finding, confidence)
                findings_parts.append(description)
        
        return "\n\n".join(findings_parts)
    
    def _generate_impression_section(self, report_data: ReportData) -> str:
        """Generate impression section."""
        return self.language_processor.generate_impression_summary(
            report_data.findings,
            report_data.recommendations
        )
    
    def _generate_recommendations_section(self, report_data: ReportData) -> str:
        """Generate recommendations section."""
        rec_parts = []
        
        # Group recommendations by urgency
        urgent_recs = [r for r in report_data.recommendations if r.urgency.value in ["critical", "urgent"]]
        moderate_recs = [r for r in report_data.recommendations if r.urgency.value == "moderate"]
        routine_recs = [r for r in report_data.recommendations if r.urgency.value == "routine"]
        
        # Urgent recommendations first
        if urgent_recs:
            rec_parts.append("URGENT:")
            for rec in urgent_recs:
                rec_parts.append(f"- {rec.description}")
        
        # Moderate recommendations
        if moderate_recs:
            if urgent_recs:
                rec_parts.append("\nModerate priority:")
            for rec in moderate_recs:
                rec_parts.append(f"- {rec.description}")
        
        # Routine recommendations
        if routine_recs:
            if urgent_recs or moderate_recs:
                rec_parts.append("\nRoutine follow-up:")
            for rec in routine_recs:
                rec_parts.append(f"- {rec.description}")
        
        return "\n".join(rec_parts)
    
    def _generate_comparison_section(self, report_data: ReportData) -> str:
        """Generate comparison section."""
        if not report_data.comparison_studies:
            return "No prior studies available for comparison."
        
        # Use most recent prior study
        prior_study = report_data.comparison_studies[0]
        prior_date = prior_study.get("study_date", "previous study")
        
        # Extract prior findings if available
        prior_findings = prior_study.get("findings", [])
        
        return self.language_processor.generate_comparison_text(
            report_data.findings,
            prior_findings,
            prior_date
        )
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from numeric score."""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            return ConfidenceLevel.MODERATE
        else:
            return ConfidenceLevel.LOW
    
    def _calculate_report_confidence(
        self,
        report_data: ReportData,
        sections: Dict[ReportSection, str]
    ) -> float:
        """Calculate overall report confidence."""
        confidence_factors = []
        
        # Findings confidence
        if report_data.findings:
            avg_finding_confidence = np.mean([f.confidence for f in report_data.findings])
            confidence_factors.append(avg_finding_confidence)
        
        # Recommendations confidence
        if report_data.recommendations:
            avg_rec_confidence = np.mean([r.confidence for r in report_data.recommendations])
            confidence_factors.append(avg_rec_confidence)
        
        # Completeness factor
        required_sections = [ReportSection.FINDINGS, ReportSection.IMPRESSION]
        completeness = sum(1 for section in required_sections if sections.get(section)) / len(required_sections)
        confidence_factors.append(completeness)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def get_report(self, report_id: str) -> Optional[GeneratedReport]:
        """Get generated report by ID."""
        return self.generated_reports.get(report_id)
    
    def export_report(self, report_id: str, format_type: str = "text") -> str:
        """Export report in specified format."""
        report = self.get_report(report_id)
        if not report:
            raise ValueError(f"Report {report_id} not found")
        
        if format_type.lower() == "json":
            return json.dumps({
                "report_id": report.report_id,
                "timestamp": report.timestamp.isoformat(),
                "patient_id": report.patient_id,
                "study_id": report.study_id,
                "content": report.content,
                "sections": {k.value: v for k, v in report.sections.items()},
                "confidence_score": report.confidence_score,
                "findings_count": report.findings_count,
                "recommendations_count": report.recommendations_count
            }, indent=2)
        
        return report.content

# Global report generator instance
medical_report_generator = MedicalReportGenerator()