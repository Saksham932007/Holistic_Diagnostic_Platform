"""
Model Performance Analyzer

Comprehensive model performance analysis, benchmarking,
and optimization recommendations for medical AI models.
"""

import time
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import torch.profiler

from src.core.config import settings
from src.core.audit import audit_logger

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision" 
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    DICE_SCORE = "dice_score"
    IOU = "iou"
    HAUSDORFF_DISTANCE = "hausdorff_distance"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    NPV = "npv"  # Negative Predictive Value
    PPV = "ppv"  # Positive Predictive Value

class ModelType(Enum):
    """Types of models."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"
    DETECTION = "detection"
    MULTI_TASK = "multi_task"

class BenchmarkType(Enum):
    """Types of benchmarks."""
    ACCURACY = "accuracy"
    SPEED = "speed"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    FLOPS = "flops"

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    model_name: str
    model_type: ModelType
    timestamp: datetime
    dataset_name: str
    sample_count: int
    
    # Core metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    
    # Segmentation metrics
    dice_score: Optional[float] = None
    iou: Optional[float] = None
    hausdorff_distance: Optional[float] = None
    
    # Clinical metrics
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    npv: Optional[float] = None
    ppv: Optional[float] = None
    
    # Performance metrics
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    throughput_samples_per_second: Optional[float] = None
    model_size_mb: Optional[float] = None
    flops: Optional[int] = None
    
    # Additional data
    confusion_matrix: Optional[np.ndarray] = None
    class_weights: Optional[Dict[str, float]] = None
    calibration_data: Optional[Dict] = None
    per_class_metrics: Optional[Dict] = None
    confidence_intervals: Optional[Dict] = None

@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    benchmark_type: BenchmarkType
    model_name: str
    result_value: float
    unit: str
    test_conditions: Dict[str, Any]
    timestamp: datetime
    hardware_info: Dict[str, Any]
    
class MetricsCalculator:
    """Calculates various performance metrics."""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Per-class metrics
        if class_names:
            report = classification_report(
                y_true, y_pred, 
                target_names=class_names, 
                output_dict=True,
                zero_division=0
            )
            metrics['per_class_metrics'] = report
        
        # AUC-ROC for binary classification or multi-class with probabilities
        if y_prob is not None:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:  # Multi-class
                try:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                except ValueError:
                    metrics['auc_roc'] = None
        
        # Clinical metrics for binary classification
        if len(np.unique(y_true)) == 2 and cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    @staticmethod
    def calculate_segmentation_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate segmentation metrics."""
        metrics = {}
        
        # Flatten arrays for metric calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true_flat, y_pred_flat)
        
        # Dice score
        dice_scores = []
        iou_scores = []
        
        unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
        if num_classes:
            unique_classes = range(num_classes)
        
        per_class_metrics = {}
        
        for class_id in unique_classes:
            # Binary masks for current class
            true_mask = (y_true_flat == class_id)
            pred_mask = (y_pred_flat == class_id)
            
            # Dice coefficient
            intersection = np.sum(true_mask & pred_mask)
            dice = (2.0 * intersection) / (np.sum(true_mask) + np.sum(pred_mask) + 1e-7)
            dice_scores.append(dice)
            
            # IoU (Jaccard index)
            union = np.sum(true_mask | pred_mask)
            iou = intersection / (union + 1e-7)
            iou_scores.append(iou)
            
            per_class_metrics[f'class_{class_id}'] = {
                'dice': dice,
                'iou': iou
            }
        
        metrics['dice_score'] = np.mean(dice_scores)
        metrics['iou'] = np.mean(iou_scores)
        metrics['per_class_dice'] = dice_scores
        metrics['per_class_iou'] = iou_scores
        metrics['per_class_metrics'] = per_class_metrics
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-7))
        
        # Mean Absolute Percentage Error
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-7))) * 100
        
        return metrics

class ModelProfiler:
    """Profiles model performance characteristics."""
    
    def __init__(self):
        """Initialize model profiler."""
        self.profiler_data = {}
    
    def profile_model_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        num_warmup: int = 10,
        num_iterations: int = 100,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Profile model inference performance."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = model.to(device)
        sample_input = sample_input.to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(sample_input)
        
        # Synchronize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(sample_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_inference_time = (total_time / num_iterations) * 1000  # Convert to ms
        throughput = num_iterations / total_time
        
        # Memory usage
        memory_usage = 0
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            torch.cuda.reset_peak_memory_stats()
        
        # Model size
        param_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # MB (assuming float32)
        
        return {
            'avg_inference_time_ms': avg_inference_time,
            'throughput_samples_per_second': throughput,
            'memory_usage_mb': memory_usage,
            'model_size_mb': param_size,
            'device': str(device),
            'batch_size': sample_input.shape[0] if len(sample_input.shape) > 0 else 1
        }
    
    def profile_with_pytorch_profiler(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """Profile using PyTorch profiler."""
        model.eval()
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(sample_input)
        
        # Extract profiling data
        events = prof.events()
        
        # CPU time
        cpu_time = sum([event.cpu_time_total for event in events]) / num_iterations
        
        # GPU time
        cuda_time = sum([event.cuda_time_total for event in events]) / num_iterations
        
        # Memory usage
        memory_usage = max([event.cpu_memory_usage for event in events if event.cpu_memory_usage])
        
        return {
            'cpu_time_us': cpu_time,
            'cuda_time_us': cuda_time,
            'memory_usage_bytes': memory_usage,
            'profiler_trace': prof.key_averages().table(sort_by="cpu_time_total")
        }

class ModelBenchmark:
    """Benchmarks models against standard datasets and metrics."""
    
    def __init__(self):
        """Initialize model benchmark."""
        self.benchmark_results = {}
        self.profiler = ModelProfiler()
        self.metrics_calculator = MetricsCalculator()
    
    async def run_accuracy_benchmark(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        model_type: ModelType,
        device: Optional[torch.device] = None
    ) -> PerformanceMetrics:
        """Run accuracy benchmark on a dataset."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = model.to(device)
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, dict):
                    inputs = batch['image'].to(device)
                    labels = batch.get('label', batch.get('mask')).to(device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                # Process outputs based on model type
                if model_type == ModelType.CLASSIFICATION:
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('classification'))
                    else:
                        logits = outputs
                    
                    probabilities = torch.softmax(logits, dim=1)
                    predictions = torch.argmax(probabilities, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                
                elif model_type == ModelType.SEGMENTATION:
                    if isinstance(outputs, dict):
                        logits = outputs.get('segmentation', outputs.get('logits'))
                    else:
                        logits = outputs
                    
                    predictions = torch.argmax(logits, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities) if all_probabilities else None
        
        if model_type == ModelType.CLASSIFICATION:
            metrics_dict = self.metrics_calculator.calculate_classification_metrics(
                y_true, y_pred, y_prob
            )
        elif model_type == ModelType.SEGMENTATION:
            metrics_dict = self.metrics_calculator.calculate_segmentation_metrics(
                y_true.reshape(-1), y_pred.reshape(-1)
            )
        else:
            metrics_dict = {}
        
        # Performance metrics
        avg_inference_time = np.mean(inference_times)
        throughput = len(dataloader.dataset) / (sum(inference_times) / 1000)  # samples per second
        
        # Create performance metrics object
        performance_metrics = PerformanceMetrics(
            model_name=model.__class__.__name__,
            model_type=model_type,
            timestamp=datetime.now(),
            dataset_name="benchmark_dataset",
            sample_count=len(dataloader.dataset),
            accuracy=metrics_dict.get('accuracy'),
            precision=metrics_dict.get('precision'),
            recall=metrics_dict.get('recall'),
            f1_score=metrics_dict.get('f1_score'),
            auc_roc=metrics_dict.get('auc_roc'),
            dice_score=metrics_dict.get('dice_score'),
            iou=metrics_dict.get('iou'),
            sensitivity=metrics_dict.get('sensitivity'),
            specificity=metrics_dict.get('specificity'),
            npv=metrics_dict.get('npv'),
            ppv=metrics_dict.get('ppv'),
            inference_time_ms=avg_inference_time,
            throughput_samples_per_second=throughput,
            confusion_matrix=metrics_dict.get('confusion_matrix'),
            per_class_metrics=metrics_dict.get('per_class_metrics')
        )
        
        await audit_logger.log_event(
            "model_benchmark_completed",
            {
                "model_name": model.__class__.__name__,
                "model_type": model_type.value,
                "accuracy": performance_metrics.accuracy,
                "inference_time_ms": avg_inference_time
            }
        )
        
        return performance_metrics
    
    async def run_speed_benchmark(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> BenchmarkResult:
        """Run speed benchmark."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        profile_results = self.profiler.profile_model_inference(
            model, sample_input, device=device
        )
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.SPEED,
            model_name=model.__class__.__name__,
            result_value=profile_results['avg_inference_time_ms'],
            unit="ms",
            test_conditions={
                "batch_size": sample_input.shape[0],
                "input_shape": list(sample_input.shape),
                "device": str(device)
            },
            timestamp=datetime.now(),
            hardware_info=self._get_hardware_info()
        )
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {}
        
        # GPU info
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name()
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info['cuda_version'] = torch.version.cuda
        
        # CPU info (simplified)
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['memory_gb'] = psutil.virtual_memory().total / 1024**3
        
        return info

class PerformanceAnalyzer:
    """Analyzes model performance and provides optimization recommendations."""
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.benchmark = ModelBenchmark()
        self.analysis_history = []
    
    async def analyze_model_performance(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        model_type: ModelType,
        baseline_metrics: Optional[PerformanceMetrics] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        
        # Run benchmarks
        accuracy_metrics = await self.benchmark.run_accuracy_benchmark(
            model, dataloader, model_type
        )
        
        # Sample input for speed benchmark
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, dict):
            sample_input = sample_batch['image'][:1]  # Single sample
        else:
            sample_input = sample_batch[0][:1]
        
        speed_benchmark = await self.benchmark.run_speed_benchmark(model, sample_input)
        
        # Analyze results
        analysis = {
            "accuracy_metrics": accuracy_metrics,
            "speed_benchmark": speed_benchmark,
            "optimization_recommendations": self._generate_recommendations(accuracy_metrics, speed_benchmark),
            "comparison_with_baseline": self._compare_with_baseline(accuracy_metrics, baseline_metrics) if baseline_metrics else None,
            "performance_score": self._calculate_performance_score(accuracy_metrics, speed_benchmark)
        }
        
        self.analysis_history.append(analysis)
        
        return analysis
    
    def _generate_recommendations(
        self,
        accuracy_metrics: PerformanceMetrics,
        speed_benchmark: BenchmarkResult
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Accuracy-based recommendations
        if accuracy_metrics.accuracy and accuracy_metrics.accuracy < 0.85:
            recommendations.append("Consider data augmentation or additional training data")
            recommendations.append("Experiment with different architectures or pre-trained models")
        
        if accuracy_metrics.precision and accuracy_metrics.recall:
            if accuracy_metrics.precision < 0.8:
                recommendations.append("Consider adjusting decision threshold to improve precision")
            if accuracy_metrics.recall < 0.8:
                recommendations.append("Consider class balancing techniques to improve recall")
        
        # Speed-based recommendations
        if speed_benchmark.result_value > 1000:  # > 1 second
            recommendations.append("Consider model quantization for faster inference")
            recommendations.append("Explore model pruning to reduce computational overhead")
            recommendations.append("Consider using TensorRT or similar optimization frameworks")
        
        # Memory recommendations
        if accuracy_metrics.memory_usage_mb and accuracy_metrics.memory_usage_mb > 4000:  # > 4GB
            recommendations.append("Consider reducing batch size to lower memory usage")
            recommendations.append("Implement gradient checkpointing if training")
        
        # Model size recommendations
        if accuracy_metrics.model_size_mb and accuracy_metrics.model_size_mb > 500:  # > 500MB
            recommendations.append("Consider knowledge distillation to create smaller models")
            recommendations.append("Evaluate model compression techniques")
        
        return recommendations
    
    def _compare_with_baseline(
        self,
        current_metrics: PerformanceMetrics,
        baseline_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        comparison = {}
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'dice_score']
        
        for metric_name in metrics_to_compare:
            current_value = getattr(current_metrics, metric_name)
            baseline_value = getattr(baseline_metrics, metric_name)
            
            if current_value is not None and baseline_value is not None:
                improvement = current_value - baseline_value
                improvement_percent = (improvement / baseline_value) * 100 if baseline_value != 0 else 0
                
                comparison[metric_name] = {
                    'current': current_value,
                    'baseline': baseline_value,
                    'improvement': improvement,
                    'improvement_percent': improvement_percent,
                    'better': improvement > 0
                }
        
        return comparison
    
    def _calculate_performance_score(
        self,
        accuracy_metrics: PerformanceMetrics,
        speed_benchmark: BenchmarkResult
    ) -> float:
        """Calculate overall performance score (0-100)."""
        accuracy_score = 0
        speed_score = 0
        
        # Accuracy component (0-50 points)
        if accuracy_metrics.accuracy:
            accuracy_score = min(accuracy_metrics.accuracy * 50, 50)
        
        # Speed component (0-50 points)
        # Normalize inference time (lower is better)
        inference_time = speed_benchmark.result_value
        if inference_time <= 10:  # ≤ 10ms = full points
            speed_score = 50
        elif inference_time <= 100:  # 10-100ms = scaled
            speed_score = 50 * (1 - (inference_time - 10) / 90)
        else:  # > 100ms = minimal points
            speed_score = max(10, 50 * (1 - (inference_time - 100) / 1000))
        
        return accuracy_score + speed_score
    
    def generate_performance_report(self, output_path: Optional[Path] = None) -> str:
        """Generate comprehensive performance report."""
        if not self.analysis_history:
            return "No analysis data available."
        
        latest_analysis = self.analysis_history[-1]
        
        report = []
        report.append("=" * 60)
        report.append("MODEL PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Accuracy metrics
        accuracy_metrics = latest_analysis["accuracy_metrics"]
        report.append("ACCURACY METRICS")
        report.append("-" * 20)
        if accuracy_metrics.accuracy:
            report.append(f"Accuracy: {accuracy_metrics.accuracy:.4f}")
        if accuracy_metrics.precision:
            report.append(f"Precision: {accuracy_metrics.precision:.4f}")
        if accuracy_metrics.recall:
            report.append(f"Recall: {accuracy_metrics.recall:.4f}")
        if accuracy_metrics.f1_score:
            report.append(f"F1 Score: {accuracy_metrics.f1_score:.4f}")
        if accuracy_metrics.dice_score:
            report.append(f"Dice Score: {accuracy_metrics.dice_score:.4f}")
        report.append("")
        
        # Speed metrics
        speed_benchmark = latest_analysis["speed_benchmark"]
        report.append("PERFORMANCE METRICS")
        report.append("-" * 20)
        report.append(f"Inference Time: {speed_benchmark.result_value:.2f} ms")
        if accuracy_metrics.throughput_samples_per_second:
            report.append(f"Throughput: {accuracy_metrics.throughput_samples_per_second:.2f} samples/sec")
        if accuracy_metrics.memory_usage_mb:
            report.append(f"Memory Usage: {accuracy_metrics.memory_usage_mb:.1f} MB")
        report.append("")
        
        # Performance score
        performance_score = latest_analysis["performance_score"]
        report.append(f"OVERALL PERFORMANCE SCORE: {performance_score:.1f}/100")
        report.append("")
        
        # Recommendations
        recommendations = latest_analysis["optimization_recommendations"]
        if recommendations:
            report.append("OPTIMIZATION RECOMMENDATIONS")
            report.append("-" * 30)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text

# Global performance analyzer instance
performance_analyzer = PerformanceAnalyzer()