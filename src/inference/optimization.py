"""
Inference optimization for medical AI models.

This module provides comprehensive inference optimization including ONNX export,
TensorRT integration, model quantization, and performance profiling for
production deployment of medical AI systems.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import os
import warnings
import tempfile
from pathlib import Path
import time
import numpy as np

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.jit import trace, script
import onnx
import onnxruntime as ort

# Optional TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch, MetaTensor

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class ModelOptimizer:
    """
    Comprehensive model optimization for inference acceleration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 1, 96, 96, 96),
        optimization_level: str = "medium",
        target_device: str = "cuda",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize model optimizer.
        
        Args:
            model: PyTorch model to optimize
            input_shape: Input tensor shape for optimization
            optimization_level: Optimization level ('low', 'medium', 'high')
            target_device: Target device ('cuda', 'cpu', 'tensorrt')
            session_id: Session ID for audit logging
            user_id: User ID for audit logging
        """
        self.model = model
        self.input_shape = input_shape
        self.optimization_level = optimization_level
        self.target_device = target_device
        self._session_id = session_id
        self._user_id = user_id
        
        # Optimization configurations
        self.config = self._get_optimization_config()
        
        # Results storage
        self.optimization_results = {}
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Model optimizer initialized",
            user_id=self._user_id,
            session_id=self._session_id,
            additional_data={
                'input_shape': input_shape,
                'optimization_level': optimization_level,
                'target_device': target_device
            }
        )
    
    def _get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration based on level."""
        configs = {
            'low': {
                'enable_quantization': False,
                'enable_onnx': True,
                'enable_tensorrt': False,
                'enable_torchscript': True,
                'quantization_backend': None
            },
            'medium': {
                'enable_quantization': True,
                'enable_onnx': True,
                'enable_tensorrt': TRT_AVAILABLE and self.target_device == 'cuda',
                'enable_torchscript': True,
                'quantization_backend': 'fbgemm'
            },
            'high': {
                'enable_quantization': True,
                'enable_onnx': True,
                'enable_tensorrt': TRT_AVAILABLE and self.target_device == 'cuda',
                'enable_torchscript': True,
                'quantization_backend': 'qnnpack' if self.target_device == 'cpu' else 'fbgemm',
                'enable_pruning': True,
                'enable_fusion': True
            }
        }
        return configs.get(self.optimization_level, configs['medium'])
    
    def optimize(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model optimization.
        
        Args:
            save_dir: Directory to save optimized models
            
        Returns:
            Dictionary containing optimization results and performance metrics
        """
        if save_dir is None:
            save_dir = tempfile.mkdtemp()
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'original_model': self._benchmark_model(self.model, "Original PyTorch"),
            'optimized_models': {}
        }
        
        try:
            # TorchScript optimization
            if self.config['enable_torchscript']:
                scripted_model = self._optimize_torchscript()
                if scripted_model is not None:
                    results['optimized_models']['torchscript'] = {
                        'model': scripted_model,
                        'metrics': self._benchmark_model(scripted_model, "TorchScript"),
                        'path': str(save_path / "model_torchscript.pt")
                    }
                    torch.jit.save(scripted_model, save_path / "model_torchscript.pt")
            
            # ONNX optimization
            if self.config['enable_onnx']:
                onnx_path = self._optimize_onnx(save_path / "model.onnx")
                if onnx_path is not None:
                    ort_session = self._create_onnx_session(onnx_path)
                    results['optimized_models']['onnx'] = {
                        'session': ort_session,
                        'metrics': self._benchmark_onnx(ort_session),
                        'path': str(onnx_path)
                    }
            
            # Quantization optimization
            if self.config['enable_quantization']:
                quantized_model = self._optimize_quantization()
                if quantized_model is not None:
                    results['optimized_models']['quantized'] = {
                        'model': quantized_model,
                        'metrics': self._benchmark_model(quantized_model, "Quantized"),
                        'path': str(save_path / "model_quantized.pt")
                    }
                    torch.save(quantized_model.state_dict(), save_path / "model_quantized.pt")
            
            # TensorRT optimization
            if self.config['enable_tensorrt'] and TRT_AVAILABLE:
                trt_engine = self._optimize_tensorrt(save_path / "model.onnx")
                if trt_engine is not None:
                    results['optimized_models']['tensorrt'] = {
                        'engine': trt_engine,
                        'metrics': self._benchmark_tensorrt(trt_engine),
                        'path': str(save_path / "model_tensorrt.engine")
                    }
            
            # Pruning optimization (if enabled)
            if self.config.get('enable_pruning', False):
                pruned_model = self._optimize_pruning()
                if pruned_model is not None:
                    results['optimized_models']['pruned'] = {
                        'model': pruned_model,
                        'metrics': self._benchmark_model(pruned_model, "Pruned"),
                        'path': str(save_path / "model_pruned.pt")
                    }
                    torch.save(pruned_model.state_dict(), save_path / "model_pruned.pt")
            
            # Generate optimization report
            results['optimization_report'] = self._generate_optimization_report(results)
            
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.INFO,
                message="Model optimization completed",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={
                    'optimized_models': list(results['optimized_models'].keys()),
                    'best_model': results['optimization_report'].get('best_model', 'unknown')
                }
            )
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.MODEL_INFERENCE,
                severity=AuditSeverity.ERROR,
                message=f"Model optimization failed: {str(e)}",
                user_id=self._user_id,
                session_id=self._session_id,
                additional_data={'error': str(e)}
            )
            raise
        
        self.optimization_results = results
        return results
    
    def _optimize_torchscript(self) -> Optional[torch.jit.ScriptModule]:
        """Optimize model using TorchScript."""
        try:
            self.model.eval()
            
            # Create example input
            example_input = torch.randn(self.input_shape)
            if next(self.model.parameters()).is_cuda:
                example_input = example_input.cuda()
            
            # Try tracing first
            try:
                scripted_model = trace(self.model, example_input)
                return scripted_model
            except:
                # Fallback to scripting
                scripted_model = script(self.model)
                return scripted_model
                
        except Exception as e:
            warnings.warn(f"TorchScript optimization failed: {str(e)}")
            return None
    
    def _optimize_onnx(self, output_path: Path) -> Optional[Path]:
        """Optimize model using ONNX export."""
        try:
            self.model.eval()
            
            # Create example input
            example_input = torch.randn(self.input_shape)
            if next(self.model.parameters()).is_cuda:
                example_input = example_input.cuda()
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                opset_version=11
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            return output_path
            
        except Exception as e:
            warnings.warn(f"ONNX optimization failed: {str(e)}")
            return None
    
    def _create_onnx_session(self, onnx_path: Path) -> Optional[ort.InferenceSession]:
        """Create ONNX Runtime inference session."""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.target_device == 'cpu':
                providers = ['CPUExecutionProvider']
            
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            return session
            
        except Exception as e:
            warnings.warn(f"ONNX session creation failed: {str(e)}")
            return None
    
    def _optimize_quantization(self) -> Optional[nn.Module]:
        """Optimize model using quantization."""
        try:
            # Prepare model for quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv3d},
                dtype=torch.qint8
            )
            
            return quantized_model
            
        except Exception as e:
            warnings.warn(f"Quantization optimization failed: {str(e)}")
            return None
    
    def _optimize_tensorrt(self, onnx_path: Path) -> Optional[Any]:
        """Optimize model using TensorRT."""
        if not TRT_AVAILABLE:
            warnings.warn("TensorRT not available")
            return None
        
        try:
            # Create TensorRT logger
            logger = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            
            # Set memory and precision
            config.max_workspace_size = 1 << 30  # 1GB
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            # Parse ONNX model
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    warnings.warn("ONNX parsing failed for TensorRT")
                    return None
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                warnings.warn("TensorRT engine building failed")
                return None
            
            return engine
            
        except Exception as e:
            warnings.warn(f"TensorRT optimization failed: {str(e)}")
            return None
    
    def _optimize_pruning(self) -> Optional[nn.Module]:
        """Optimize model using structured pruning."""
        try:
            import torch.nn.utils.prune as prune
            
            # Create a copy of the model for pruning
            pruned_model = torch.nn.DataParallel(self.model) if torch.cuda.device_count() > 1 else self.model
            
            # Apply pruning to linear and conv layers
            for module in pruned_model.modules():
                if isinstance(module, (nn.Linear, nn.Conv3d)):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
            
            # Remove pruning re-parameterization
            for module in pruned_model.modules():
                if isinstance(module, (nn.Linear, nn.Conv3d)):
                    prune.remove(module, 'weight')
            
            return pruned_model
            
        except Exception as e:
            warnings.warn(f"Pruning optimization failed: {str(e)}")
            return None
    
    def _benchmark_model(self, model: nn.Module, name: str) -> Dict[str, float]:
        """Benchmark PyTorch model performance."""
        model.eval()
        
        # Create test input
        test_input = torch.randn(self.input_shape)
        if next(model.parameters()).is_cuda:
            test_input = test_input.cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(20):
                _ = model(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        inference_time = (end_time - start_time) / 20.0
        throughput = 1.0 / inference_time
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_cached = torch.cuda.memory_reserved() / (1024**3)  # GB
        else:
            memory_allocated = 0.0
            memory_cached = 0.0
        
        return {
            'inference_time_ms': inference_time * 1000,
            'throughput_fps': throughput,
            'memory_allocated_gb': memory_allocated,
            'memory_cached_gb': memory_cached,
            'model_name': name
        }
    
    def _benchmark_onnx(self, session: ort.InferenceSession) -> Dict[str, float]:
        """Benchmark ONNX model performance."""
        # Create test input
        input_name = session.get_inputs()[0].name
        test_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(5):
            _ = session.run(None, {input_name: test_input})
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(20):
            _ = session.run(None, {input_name: test_input})
        
        end_time = time.time()
        
        inference_time = (end_time - start_time) / 20.0
        throughput = 1.0 / inference_time
        
        return {
            'inference_time_ms': inference_time * 1000,
            'throughput_fps': throughput,
            'memory_allocated_gb': 0.0,  # ONNX memory tracking not straightforward
            'memory_cached_gb': 0.0,
            'model_name': 'ONNX'
        }
    
    def _benchmark_tensorrt(self, engine: Any) -> Dict[str, float]:
        """Benchmark TensorRT model performance."""
        if not TRT_AVAILABLE:
            return {}
        
        try:
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate memory
            h_input = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
            h_output = cuda.pagelocked_empty((self.input_shape[0], 2), dtype=np.float32)  # Assuming binary classification
            
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            
            stream = cuda.Stream()
            
            # Warmup
            for _ in range(5):
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
            
            # Benchmark
            start_time = time.time()
            
            for _ in range(20):
                cuda.memcpy_htod_async(d_input, h_input, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
            
            end_time = time.time()
            
            inference_time = (end_time - start_time) / 20.0
            throughput = 1.0 / inference_time
            
            return {
                'inference_time_ms': inference_time * 1000,
                'throughput_fps': throughput,
                'memory_allocated_gb': 0.0,  # TensorRT memory tracking
                'memory_cached_gb': 0.0,
                'model_name': 'TensorRT'
            }
            
        except Exception as e:
            warnings.warn(f"TensorRT benchmarking failed: {str(e)}")
            return {}
    
    def _generate_optimization_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            'baseline_performance': results['original_model'],
            'optimized_performance': {},
            'speedup_factors': {},
            'memory_savings': {},
            'best_model': 'original'
        }
        
        baseline_time = results['original_model']['inference_time_ms']
        baseline_memory = results['original_model']['memory_allocated_gb']
        best_time = baseline_time
        
        for model_name, model_data in results['optimized_models'].items():
            metrics = model_data['metrics']
            report['optimized_performance'][model_name] = metrics
            
            # Calculate speedup
            speedup = baseline_time / metrics['inference_time_ms']
            report['speedup_factors'][model_name] = speedup
            
            # Calculate memory savings
            memory_saving = (baseline_memory - metrics['memory_allocated_gb']) / baseline_memory if baseline_memory > 0 else 0
            report['memory_savings'][model_name] = memory_saving
            
            # Track best model
            if metrics['inference_time_ms'] < best_time:
                best_time = metrics['inference_time_ms']
                report['best_model'] = model_name
        
        return report


class InferenceEngine:
    """
    High-performance inference engine with multiple backend support.
    """
    
    def __init__(
        self,
        model_path: str,
        backend: str = "pytorch",
        device: str = "cuda",
        batch_size: int = 1,
        use_sliding_window: bool = True,
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        sw_batch_size: int = 4,
        overlap: float = 0.25
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to optimized model
            backend: Inference backend ('pytorch', 'onnx', 'tensorrt')
            device: Target device
            batch_size: Inference batch size
            use_sliding_window: Whether to use sliding window inference
            roi_size: ROI size for sliding window
            sw_batch_size: Sliding window batch size
            overlap: Sliding window overlap
        """
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.use_sliding_window = use_sliding_window
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        
        # Load model based on backend
        self.model = self._load_model()
    
    def _load_model(self):
        """Load model based on specified backend."""
        if self.backend == "pytorch":
            return torch.jit.load(self.model_path)
        elif self.backend == "onnx":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if self.device == 'cpu':
                providers = ['CPUExecutionProvider']
            return ort.InferenceSession(self.model_path, providers=providers)
        elif self.backend == "tensorrt" and TRT_AVAILABLE:
            # Load TensorRT engine
            with open(self.model_path, 'rb') as f:
                engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
            return engine
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def predict(self, input_data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Perform inference on input data.
        
        Args:
            input_data: Input tensor or array
            
        Returns:
            Prediction results
        """
        if self.backend == "pytorch":
            return self._pytorch_inference(input_data)
        elif self.backend == "onnx":
            return self._onnx_inference(input_data)
        elif self.backend == "tensorrt":
            return self._tensorrt_inference(input_data)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _pytorch_inference(self, input_data: torch.Tensor) -> np.ndarray:
        """PyTorch inference."""
        self.model.eval()
        
        if self.device == "cuda" and torch.cuda.is_available():
            input_data = input_data.cuda()
            self.model = self.model.cuda()
        
        with torch.no_grad():
            if self.use_sliding_window and input_data.shape[2:] != self.roi_size:
                predictions = sliding_window_inference(
                    inputs=input_data,
                    roi_size=self.roi_size,
                    sw_batch_size=self.sw_batch_size,
                    predictor=self.model,
                    overlap=self.overlap
                )
            else:
                predictions = self.model(input_data)
        
        return predictions.cpu().numpy()
    
    def _onnx_inference(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX inference."""
        input_name = self.model.get_inputs()[0].name
        
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        
        results = self.model.run(None, {input_name: input_data.astype(np.float32)})
        return results[0]
    
    def _tensorrt_inference(self, input_data: np.ndarray) -> np.ndarray:
        """TensorRT inference."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        # Implementation would depend on specific TensorRT setup
        # This is a placeholder for the actual implementation
        raise NotImplementedError("TensorRT inference implementation needed")


def create_optimized_inference_engine(
    model_path: str,
    optimization_config: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> InferenceEngine:
    """
    Create optimized inference engine from configuration.
    
    Args:
        model_path: Path to model file
        optimization_config: Optimization configuration
        session_id: Session ID for audit logging
        user_id: User ID for audit logging
        
    Returns:
        Configured inference engine
    """
    config = optimization_config or {}
    
    return InferenceEngine(
        model_path=model_path,
        backend=config.get('backend', 'pytorch'),
        device=config.get('device', 'cuda'),
        batch_size=config.get('batch_size', 1),
        use_sliding_window=config.get('use_sliding_window', True)
    )