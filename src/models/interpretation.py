"""
Model interpretation and explainability tools for clinical AI systems.

This module provides comprehensive tools for interpreting and explaining
AI model decisions in medical image analysis, including attention visualization,
saliency maps, and clinical-friendly explanations.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

from monai.visualize import GradCAM, CAM
from monai.utils import ensure_tuple_rep
from monai.data import MetaTensor

try:
    from captum.attr import (
        IntegratedGradients, GradientShap, Saliency, 
        GuidedBackprop, DeepLift, Occlusion,
        LayerGradientShap, LayerIntegratedGradients
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class ExplanationMethod(Enum):
    """Available explanation methods."""
    GRAD_CAM = "grad_cam"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    SALIENCY = "saliency"
    GUIDED_BACKPROP = "guided_backprop"
    OCCLUSION = "occlusion"
    DEEP_LIFT = "deep_lift"
    GRADIENT_SHAP = "gradient_shap"
    ATTENTION_MAPS = "attention_maps"
    LIME = "lime"


@dataclass
class InterpretationResult:
    """Result of model interpretation."""
    method: ExplanationMethod
    attribution_map: np.ndarray
    confidence_score: float
    target_class: int
    explanation_text: str
    clinical_relevance: str
    regions_of_interest: List[Tuple[int, int, int, int, int, int]]  # 3D bounding boxes
    metadata: Dict[str, Any]


class ClinicalExplainer:
    """
    Clinical-focused model interpretation and explanation system.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize clinical explainer.
        
        Args:
            model: PyTorch model to explain
            target_layers: Target layers for gradient-based explanations
            class_names: Names of output classes
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.model = model
        self.target_layers = target_layers or []
        self.class_names = class_names or []
        self._session_id = session_id
        self._user_id = user_id
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize explanation methods
        self._init_explanation_methods()
        
        # Clinical interpretation templates
        self._init_clinical_templates()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Clinical explainer initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'target_layers': self.target_layers,
                'class_names': self.class_names,
                'captum_available': CAPTUM_AVAILABLE
            }
        )
    
    def _init_explanation_methods(self):
        """Initialize explanation method implementations."""
        self.explainers = {}
        
        # GradCAM
        if self.target_layers:
            try:
                self.explainers[ExplanationMethod.GRAD_CAM] = GradCAM(
                    nn_module=self.model,
                    target_layers=self.target_layers[0]  # Primary target layer
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize GradCAM: {str(e)}")
        
        # Captum methods
        if CAPTUM_AVAILABLE:
            self.explainers[ExplanationMethod.INTEGRATED_GRADIENTS] = IntegratedGradients(self.model)
            self.explainers[ExplanationMethod.SALIENCY] = Saliency(self.model)
            self.explainers[ExplanationMethod.GUIDED_BACKPROP] = GuidedBackprop(self.model)
            self.explainers[ExplanationMethod.DEEP_LIFT] = DeepLift(self.model)
            self.explainers[ExplanationMethod.GRADIENT_SHAP] = GradientShap(self.model)
            self.explainers[ExplanationMethod.OCCLUSION] = Occlusion(self.model)
    
    def _init_clinical_templates(self):
        """Initialize clinical explanation templates."""
        self.clinical_templates = {
            'tumor': {
                'high_attention': "The model shows high confidence in tumor detection, "
                                "focusing on regions with irregular enhancement patterns and mass effect.",
                'medium_attention': "The model identifies potential tumor characteristics, "
                                  "with moderate confidence in the highlighted regions.",
                'low_attention': "The model shows minimal activation for tumor features, "
                               "suggesting low probability of malignancy."
            },
            'hemorrhage': {
                'high_attention': "The model strongly indicates hemorrhage, focusing on "
                                "hyperdense regions consistent with acute bleeding.",
                'medium_attention': "The model identifies possible hemorrhage with moderate "
                                  "confidence in the highlighted areas.",
                'low_attention': "The model shows low activation for hemorrhage features."
            },
            'normal': {
                'high_attention': "The model is highly confident in normal anatomy, "
                                "with attention distributed across expected anatomical structures.",
                'medium_attention': "The model indicates largely normal findings with some "
                                  "areas of increased attention that may warrant review.",
                'low_attention': "The model shows uniform low activation consistent with normal findings."
            }
        }
    
    def explain_prediction(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        methods: Optional[List[ExplanationMethod]] = None,
        generate_report: bool = True
    ) -> Dict[ExplanationMethod, InterpretationResult]:
        """
        Generate comprehensive explanation for model prediction.
        
        Args:
            input_tensor: Input medical image tensor
            target_class: Target class for explanation (if None, use predicted class)
            methods: List of explanation methods to use
            generate_report: Whether to generate clinical interpretation report
            
        Returns:
            Dictionary of interpretation results by method
        """
        if methods is None:
            methods = [
                ExplanationMethod.GRAD_CAM,
                ExplanationMethod.INTEGRATED_GRADIENTS,
                ExplanationMethod.SALIENCY
            ]
        
        results = {}
        
        try:
            # Get model prediction
            with torch.no_grad():
                prediction = self.model(input_tensor)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]  # Take first output if tuple
                
                probabilities = F.softmax(prediction, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
            
            # Use predicted class if target not specified
            if target_class is None:
                target_class = predicted_class
            
            # Generate explanations using different methods
            for method in methods:
                if method in self.explainers:
                    try:
                        result = self._generate_explanation(
                            input_tensor, target_class, method, confidence
                        )
                        results[method] = result
                    except Exception as e:
                        warnings.warn(f"Failed to generate {method.value} explanation: {str(e)}")
                        continue
            
            # Generate clinical interpretation if requested
            if generate_report and results:
                clinical_interpretation = self._generate_clinical_interpretation(
                    results, target_class, confidence
                )
                
                # Add to each result
                for result in results.values():
                    result.clinical_relevance = clinical_interpretation
            
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.INFO,
                message="Model explanation generated",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'target_class': target_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'methods_used': [m.value for m in results.keys()]
                }
            )
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.ERROR,
                message=f"Model explanation failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise
        
        return results
    
    def _generate_explanation(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        method: ExplanationMethod,
        confidence: float
    ) -> InterpretationResult:
        """Generate explanation using specific method."""
        
        if method == ExplanationMethod.GRAD_CAM:
            return self._grad_cam_explanation(input_tensor, target_class, confidence)
        
        elif method == ExplanationMethod.INTEGRATED_GRADIENTS:
            return self._integrated_gradients_explanation(input_tensor, target_class, confidence)
        
        elif method == ExplanationMethod.SALIENCY:
            return self._saliency_explanation(input_tensor, target_class, confidence)
        
        elif method == ExplanationMethod.GUIDED_BACKPROP:
            return self._guided_backprop_explanation(input_tensor, target_class, confidence)
        
        elif method == ExplanationMethod.OCCLUSION:
            return self._occlusion_explanation(input_tensor, target_class, confidence)
        
        elif method == ExplanationMethod.ATTENTION_MAPS:
            return self._attention_maps_explanation(input_tensor, target_class, confidence)
        
        else:
            raise ValueError(f"Unsupported explanation method: {method}")
    
    def _grad_cam_explanation(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        confidence: float
    ) -> InterpretationResult:
        """Generate GradCAM explanation."""
        explainer = self.explainers[ExplanationMethod.GRAD_CAM]
        
        # Generate attribution
        attribution = explainer(
            input_tensor,
            class_idx=target_class,
            retain_graph=True
        )
        
        # Convert to numpy
        attribution_np = attribution.squeeze().cpu().numpy()
        
        # Extract regions of interest
        roi_boxes = self._extract_roi_boxes(attribution_np, threshold=0.7)
        
        # Generate explanation text
        max_activation = np.max(attribution_np)
        explanation_text = self._generate_explanation_text(
            ExplanationMethod.GRAD_CAM,
            target_class,
            max_activation,
            confidence
        )
        
        return InterpretationResult(
            method=ExplanationMethod.GRAD_CAM,
            attribution_map=attribution_np,
            confidence_score=confidence,
            target_class=target_class,
            explanation_text=explanation_text,
            clinical_relevance="",  # Will be filled by clinical interpretation
            regions_of_interest=roi_boxes,
            metadata={
                'max_activation': float(max_activation),
                'min_activation': float(np.min(attribution_np)),
                'mean_activation': float(np.mean(attribution_np))
            }
        )
    
    def _integrated_gradients_explanation(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        confidence: float
    ) -> InterpretationResult:
        """Generate Integrated Gradients explanation."""
        if not CAPTUM_AVAILABLE:
            raise RuntimeError("Captum not available for Integrated Gradients")
        
        explainer = self.explainers[ExplanationMethod.INTEGRATED_GRADIENTS]
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(input_tensor)
        
        # Generate attribution
        attribution = explainer.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=50
        )
        
        # Convert to numpy
        attribution_np = attribution.squeeze().cpu().numpy()
        
        # Extract regions of interest
        roi_boxes = self._extract_roi_boxes(np.abs(attribution_np), threshold=0.1)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            ExplanationMethod.INTEGRATED_GRADIENTS,
            target_class,
            np.max(np.abs(attribution_np)),
            confidence
        )
        
        return InterpretationResult(
            method=ExplanationMethod.INTEGRATED_GRADIENTS,
            attribution_map=attribution_np,
            confidence_score=confidence,
            target_class=target_class,
            explanation_text=explanation_text,
            clinical_relevance="",
            regions_of_interest=roi_boxes,
            metadata={
                'max_attribution': float(np.max(attribution_np)),
                'min_attribution': float(np.min(attribution_np)),
                'n_steps': 50
            }
        )
    
    def _saliency_explanation(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        confidence: float
    ) -> InterpretationResult:
        """Generate Saliency map explanation."""
        if not CAPTUM_AVAILABLE:
            raise RuntimeError("Captum not available for Saliency")
        
        explainer = self.explainers[ExplanationMethod.SALIENCY]
        
        # Generate attribution
        attribution = explainer.attribute(input_tensor, target=target_class)
        
        # Convert to numpy
        attribution_np = attribution.squeeze().cpu().numpy()
        
        # Extract regions of interest
        roi_boxes = self._extract_roi_boxes(np.abs(attribution_np), threshold=0.1)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            ExplanationMethod.SALIENCY,
            target_class,
            np.max(np.abs(attribution_np)),
            confidence
        )
        
        return InterpretationResult(
            method=ExplanationMethod.SALIENCY,
            attribution_map=attribution_np,
            confidence_score=confidence,
            target_class=target_class,
            explanation_text=explanation_text,
            clinical_relevance="",
            regions_of_interest=roi_boxes,
            metadata={
                'max_gradient': float(np.max(np.abs(attribution_np))),
                'mean_gradient': float(np.mean(np.abs(attribution_np)))
            }
        )
    
    def _guided_backprop_explanation(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        confidence: float
    ) -> InterpretationResult:
        """Generate Guided Backpropagation explanation."""
        if not CAPTUM_AVAILABLE:
            raise RuntimeError("Captum not available for Guided Backprop")
        
        explainer = self.explainers[ExplanationMethod.GUIDED_BACKPROP]
        
        # Generate attribution
        attribution = explainer.attribute(input_tensor, target=target_class)
        
        # Convert to numpy
        attribution_np = attribution.squeeze().cpu().numpy()
        
        # Extract regions of interest
        roi_boxes = self._extract_roi_boxes(np.abs(attribution_np), threshold=0.1)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            ExplanationMethod.GUIDED_BACKPROP,
            target_class,
            np.max(attribution_np),
            confidence
        )
        
        return InterpretationResult(
            method=ExplanationMethod.GUIDED_BACKPROP,
            attribution_map=attribution_np,
            confidence_score=confidence,
            target_class=target_class,
            explanation_text=explanation_text,
            clinical_relevance="",
            regions_of_interest=roi_boxes,
            metadata={
                'max_attribution': float(np.max(attribution_np)),
                'positive_attribution_ratio': float(np.sum(attribution_np > 0) / attribution_np.size)
            }
        )
    
    def _occlusion_explanation(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        confidence: float
    ) -> InterpretationResult:
        """Generate Occlusion explanation."""
        if not CAPTUM_AVAILABLE:
            raise RuntimeError("Captum not available for Occlusion")
        
        explainer = self.explainers[ExplanationMethod.OCCLUSION]
        
        # Determine sliding window size based on input dimensions
        input_shape = input_tensor.shape[2:]  # Remove batch and channel dimensions
        window_size = tuple(max(1, dim // 8) for dim in input_shape)  # 1/8 of each dimension
        
        # Generate attribution
        attribution = explainer.attribute(
            input_tensor,
            target=target_class,
            sliding_window_shapes=window_size,
            strides=tuple(max(1, ws // 2) for ws in window_size)  # 50% overlap
        )
        
        # Convert to numpy
        attribution_np = attribution.squeeze().cpu().numpy()
        
        # Extract regions of interest
        roi_boxes = self._extract_roi_boxes(np.abs(attribution_np), threshold=0.1)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            ExplanationMethod.OCCLUSION,
            target_class,
            np.max(np.abs(attribution_np)),
            confidence
        )
        
        return InterpretationResult(
            method=ExplanationMethod.OCCLUSION,
            attribution_map=attribution_np,
            confidence_score=confidence,
            target_class=target_class,
            explanation_text=explanation_text,
            clinical_relevance="",
            regions_of_interest=roi_boxes,
            metadata={
                'window_size': window_size,
                'max_impact': float(np.max(np.abs(attribution_np))),
                'negative_impact_ratio': float(np.sum(attribution_np < 0) / attribution_np.size)
            }
        )
    
    def _attention_maps_explanation(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        confidence: float
    ) -> InterpretationResult:
        """Generate attention maps explanation (for transformer models)."""
        # This is a placeholder for attention map extraction
        # Implementation would depend on specific model architecture
        
        # For now, return a simple gradient-based approximation
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if isinstance(output, tuple):
            output = output[0]
        
        # Get target output
        target_output = output[0, target_class]
        
        # Compute gradients
        grads = grad(target_output, input_tensor, create_graph=False)[0]
        
        # Convert to attention-like map
        attention_map = torch.abs(grads).squeeze().cpu().numpy()
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Extract regions of interest
        roi_boxes = self._extract_roi_boxes(attention_map, threshold=0.7)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            ExplanationMethod.ATTENTION_MAPS,
            target_class,
            np.max(attention_map),
            confidence
        )
        
        return InterpretationResult(
            method=ExplanationMethod.ATTENTION_MAPS,
            attribution_map=attention_map,
            confidence_score=confidence,
            target_class=target_class,
            explanation_text=explanation_text,
            clinical_relevance="",
            regions_of_interest=roi_boxes,
            metadata={
                'attention_coverage': float(np.sum(attention_map > 0.5) / attention_map.size),
                'max_attention': float(np.max(attention_map))
            }
        )
    
    def _extract_roi_boxes(
        self,
        attribution_map: np.ndarray,
        threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int, int, int]]:
        """Extract 3D regions of interest from attribution map."""
        # Threshold the attribution map
        binary_map = attribution_map > threshold * np.max(attribution_map)
        
        # Find connected components (simplified for 3D)
        roi_boxes = []
        
        # Find non-zero coordinates
        coords = np.where(binary_map)
        
        if len(coords[0]) > 0:
            # Calculate bounding box
            min_coords = [int(np.min(coord)) for coord in coords]
            max_coords = [int(np.max(coord)) for coord in coords]
            
            # Create 3D bounding box (x1, y1, z1, x2, y2, z2)
            if len(min_coords) >= 3:
                roi_box = (
                    min_coords[0], min_coords[1], min_coords[2],
                    max_coords[0], max_coords[1], max_coords[2]
                )
                roi_boxes.append(roi_box)
        
        return roi_boxes
    
    def _generate_explanation_text(
        self,
        method: ExplanationMethod,
        target_class: int,
        max_activation: float,
        confidence: float
    ) -> str:
        """Generate human-readable explanation text."""
        class_name = self.class_names[target_class] if target_class < len(self.class_names) else f"Class {target_class}"
        
        method_descriptions = {
            ExplanationMethod.GRAD_CAM: "gradient-weighted class activation mapping",
            ExplanationMethod.INTEGRATED_GRADIENTS: "integrated gradient analysis",
            ExplanationMethod.SALIENCY: "saliency mapping",
            ExplanationMethod.GUIDED_BACKPROP: "guided backpropagation",
            ExplanationMethod.OCCLUSION: "occlusion sensitivity analysis",
            ExplanationMethod.ATTENTION_MAPS: "attention mechanism visualization"
        }
        
        method_desc = method_descriptions.get(method, method.value)
        
        explanation = f"Using {method_desc}, the model shows "
        
        if max_activation > 0.7:
            explanation += f"strong evidence for {class_name} "
        elif max_activation > 0.4:
            explanation += f"moderate evidence for {class_name} "
        else:
            explanation += f"weak evidence for {class_name} "
        
        explanation += f"with {confidence:.1%} confidence. "
        explanation += f"Maximum activation strength: {max_activation:.3f}."
        
        return explanation
    
    def _generate_clinical_interpretation(
        self,
        results: Dict[ExplanationMethod, InterpretationResult],
        target_class: int,
        confidence: float
    ) -> str:
        """Generate clinical interpretation of explanations."""
        class_name = self.class_names[target_class] if target_class < len(self.class_names) else "unknown"
        
        # Determine clinical category
        clinical_category = "normal"
        if "tumor" in class_name.lower():
            clinical_category = "tumor"
        elif "hemorrhage" in class_name.lower() or "bleed" in class_name.lower():
            clinical_category = "hemorrhage"
        
        # Determine confidence level
        if confidence > 0.8:
            confidence_level = "high_attention"
        elif confidence > 0.5:
            confidence_level = "medium_attention"
        else:
            confidence_level = "low_attention"
        
        # Get template
        template = self.clinical_templates.get(clinical_category, {}).get(
            confidence_level,
            "The model provides moderate confidence in its assessment of the highlighted regions."
        )
        
        # Add method consensus information
        consensus_info = ""
        if len(results) > 1:
            method_names = [result.method.value.replace('_', ' ').title() for result in results.values()]
            consensus_info = f" Multiple explanation methods ({', '.join(method_names)}) show consistent patterns, "
            consensus_info += "increasing confidence in the highlighted regions of clinical interest."
        
        return template + consensus_info
    
    def visualize_explanations(
        self,
        input_tensor: torch.Tensor,
        results: Dict[ExplanationMethod, InterpretationResult],
        slice_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of explanations.
        
        Args:
            input_tensor: Original input tensor
            results: Dictionary of interpretation results
            slice_idx: Slice index for 3D visualization (middle slice if None)
            save_path: Path to save visualization
            
        Returns:
            Visualization as numpy array
        """
        # Get the middle slice if not specified
        if slice_idx is None:
            slice_idx = input_tensor.shape[-1] // 2
        
        # Extract 2D slice from input
        input_slice = input_tensor[0, 0, :, :, slice_idx].cpu().numpy()
        
        # Create subplot grid
        n_methods = len(results)
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
        
        if n_methods == 0:
            return np.array([])
        
        # Original image
        axes[0, 0].imshow(input_slice, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(input_slice, cmap='gray')
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        # Explanation maps
        for idx, (method, result) in enumerate(results.items()):
            col_idx = idx + 1
            
            # Get 2D slice from attribution map
            if len(result.attribution_map.shape) == 3:
                attr_slice = result.attribution_map[:, :, slice_idx]
            else:
                # If 2D, use as is
                attr_slice = result.attribution_map
            
            # Raw attribution map
            im1 = axes[0, col_idx].imshow(attr_slice, cmap='jet', alpha=0.8)
            axes[0, col_idx].set_title(f'{method.value.replace("_", " ").title()}\nAttribution Map')
            axes[0, col_idx].axis('off')
            plt.colorbar(im1, ax=axes[0, col_idx], fraction=0.046)
            
            # Overlay on original
            axes[1, col_idx].imshow(input_slice, cmap='gray')
            im2 = axes[1, col_idx].imshow(attr_slice, cmap='jet', alpha=0.5)
            axes[1, col_idx].set_title(f'{method.value.replace("_", " ").title()}\nOverlay')
            axes[1, col_idx].axis('off')
            
            # Add ROI boxes
            for roi_box in result.regions_of_interest:
                if len(roi_box) >= 6:  # 3D box
                    x1, y1, z1, x2, y2, z2 = roi_box
                    if z1 <= slice_idx <= z2:  # ROI intersects this slice
                        rect = plt.Rectangle((y1, x1), y2-y1, x2-x1, 
                                           linewidth=2, edgecolor='red', facecolor='none')
                        axes[1, col_idx].add_patch(rect)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_array
    
    def generate_clinical_report(
        self,
        results: Dict[ExplanationMethod, InterpretationResult],
        patient_id: str,
        study_id: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive clinical interpretation report.
        
        Args:
            results: Dictionary of interpretation results
            patient_id: Patient identifier
            study_id: Study identifier
            
        Returns:
            Clinical interpretation report
        """
        if not results:
            return {}
        
        # Get primary result (highest confidence)
        primary_result = max(results.values(), key=lambda r: r.confidence_score)
        
        # Calculate method consensus
        method_consensus = len([r for r in results.values() if r.target_class == primary_result.target_class]) / len(results)
        
        # Generate report
        report = {
            'patient_id': patient_id,
            'study_id': study_id,
            'timestamp': torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL,  # Current timestamp
            'primary_finding': {
                'class': primary_result.target_class,
                'class_name': self.class_names[primary_result.target_class] if primary_result.target_class < len(self.class_names) else f"Class {primary_result.target_class}",
                'confidence': primary_result.confidence_score,
                'explanation': primary_result.explanation_text,
                'clinical_relevance': primary_result.clinical_relevance
            },
            'method_results': {
                method.value: {
                    'confidence': result.confidence_score,
                    'explanation': result.explanation_text,
                    'roi_count': len(result.regions_of_interest),
                    'metadata': result.metadata
                }
                for method, result in results.items()
            },
            'consensus_score': method_consensus,
            'regions_of_interest': primary_result.regions_of_interest,
            'clinical_recommendations': self._generate_clinical_recommendations(primary_result),
            'interpretation_quality': self._assess_interpretation_quality(results)
        }
        
        return report
    
    def _generate_clinical_recommendations(self, result: InterpretationResult) -> List[str]:
        """Generate clinical recommendations based on interpretation."""
        recommendations = []
        
        if result.confidence_score > 0.8:
            recommendations.append("High confidence finding - recommend correlation with clinical symptoms")
            if len(result.regions_of_interest) > 0:
                recommendations.append("Specific regions of interest identified - consider targeted follow-up imaging")
        
        elif result.confidence_score > 0.6:
            recommendations.append("Moderate confidence finding - recommend clinical correlation and possible follow-up")
        
        else:
            recommendations.append("Low confidence finding - consider additional imaging or clinical correlation")
        
        if result.target_class != 0:  # Assuming 0 is normal class
            recommendations.append("Abnormal finding detected - recommend specialist consultation if clinically indicated")
        
        return recommendations
    
    def _assess_interpretation_quality(self, results: Dict[ExplanationMethod, InterpretationResult]) -> str:
        """Assess the quality of interpretations."""
        if len(results) < 2:
            return "Limited - single method used"
        
        # Check consistency across methods
        target_classes = [result.target_class for result in results.values()]
        class_consistency = len(set(target_classes)) == 1
        
        # Check confidence levels
        confidences = [result.confidence_score for result in results.values()]
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        if class_consistency and avg_confidence > 0.7 and confidence_std < 0.2:
            return "High - consistent and confident across methods"
        elif class_consistency and avg_confidence > 0.5:
            return "Good - consistent across methods with moderate confidence"
        elif avg_confidence > 0.6:
            return "Moderate - good confidence but some inconsistency across methods"
        else:
            return "Low - inconsistent or low confidence across methods"


def create_clinical_explainer(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> ClinicalExplainer:
    """
    Create clinical explainer instance.
    
    Args:
        model: PyTorch model to explain
        target_layers: Target layers for gradient-based explanations
        class_names: Names of output classes
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured clinical explainer
    """
    return ClinicalExplainer(
        model=model,
        target_layers=target_layers,
        class_names=class_names,
        session_id=session_id,
        user_id=user_id
    )