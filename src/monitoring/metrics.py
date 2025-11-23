"""
Medical Platform Monitoring Module

Comprehensive monitoring and metrics collection for the Medical Diagnostic
Platform with Prometheus integration, health checks, and performance tracking.

Author: Holistic Diagnostic Platform Team
Version: 1.0.0
"""

import asyncio
import time
import psutil
import torch
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import redis.asyncio as redis

# Configure logging
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    INFO = "info"

class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Health check result."""
    service: str
    status: ServiceStatus
    timestamp: datetime
    response_time: float
    details: Dict[str, Any]
    error: Optional[str] = None

class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        
        # API metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float('inf')],
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'active_sessions_total',
            'Number of active user sessions',
            registry=self.registry
        )
        
        # Model metrics
        self.model_inference_duration_seconds = Histogram(
            'model_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_type', 'task'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')],
            registry=self.registry
        )
        
        self.model_inference_total = Counter(
            'model_inference_total',
            'Total model inferences',
            ['model_type', 'task', 'status'],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy score',
            ['model_type', 'task'],
            registry=self.registry
        )
        
        self.models_loaded = Gauge(
            'models_loaded_total',
            'Number of loaded models',
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.disk_usage_bytes = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            ['mountpoint', 'type'],
            registry=self.registry
        )
        
        self.gpu_utilization_percent = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory_bytes = Gauge(
            'gpu_memory_bytes',
            'GPU memory usage in bytes',
            ['gpu_id', 'type'],
            registry=self.registry
        )
        
        # Medical specific metrics
        self.studies_processed_total = Counter(
            'studies_processed_total',
            'Total studies processed',
            ['modality', 'status'],
            registry=self.registry
        )
        
        self.image_processing_duration_seconds = Histogram(
            'image_processing_duration_seconds',
            'Image processing duration in seconds',
            ['modality', 'processing_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float('inf')],
            registry=self.registry
        )
        
        self.segmentation_dice_score = Gauge(
            'segmentation_dice_score',
            'Segmentation Dice score',
            ['model_type', 'organ'],
            registry=self.registry
        )
        
        self.classification_confidence = Gauge(
            'classification_confidence',
            'Classification confidence score',
            ['model_type', 'class'],
            registry=self.registry
        )
        
        # Security metrics
        self.authentication_attempts_total = Counter(
            'authentication_attempts_total',
            'Total authentication attempts',
            ['result'],
            registry=self.registry
        )
        
        self.rate_limit_violations_total = Counter(
            'rate_limit_violations_total',
            'Total rate limit violations',
            ['endpoint'],
            registry=self.registry
        )
        
        # Application info
        self.application_info = Info(
            'application_info',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        self.application_info.info({
            'version': '1.0.0',
            'name': 'Medical Diagnostic Platform',
            'python_version': f'{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}',
            'pytorch_version': torch.__version__ if torch else 'N/A'
        })
        
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
    def record_model_inference(self, model_type: str, task: str, duration: float, status: str):
        """Record model inference metrics."""
        self.model_inference_total.labels(
            model_type=model_type,
            task=task,
            status=status
        ).inc()
        
        if status == "success":
            self.model_inference_duration_seconds.labels(
                model_type=model_type,
                task=task
            ).observe(duration)
            
    def update_system_metrics(self):
        """Update system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage_percent.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage_bytes.labels(type='used').set(memory.used)
        self.memory_usage_bytes.labels(type='available').set(memory.available)
        self.memory_usage_bytes.labels(type='total').set(memory.total)
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                disk_usage = psutil.disk_usage(partition.mountpoint)
                self.disk_usage_bytes.labels(
                    mountpoint=partition.mountpoint,
                    type='used'
                ).set(disk_usage.used)
                self.disk_usage_bytes.labels(
                    mountpoint=partition.mountpoint,
                    type='free'
                ).set(disk_usage.free)
                self.disk_usage_bytes.labels(
                    mountpoint=partition.mountpoint,
                    type='total'
                ).set(disk_usage.total)
            except PermissionError:
                pass
                
        # GPU metrics
        if torch and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # GPU utilization (requires nvidia-ml-py)
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_utilization_percent.labels(gpu_id=str(i)).set(utilization.gpu)
                    
                    # Memory
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_memory_bytes.labels(gpu_id=str(i), type='used').set(memory_info.used)
                    self.gpu_memory_bytes.labels(gpu_id=str(i), type='free').set(memory_info.free)
                    self.gpu_memory_bytes.labels(gpu_id=str(i), type='total').set(memory_info.total)
                    
                except ImportError:
                    # Fallback to torch memory info
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        memory_allocated = torch.cuda.memory_allocated(i)
                        memory_reserved = torch.cuda.memory_reserved(i)
                        
                        self.gpu_memory_bytes.labels(gpu_id=str(i), type='allocated').set(memory_allocated)
                        self.gpu_memory_bytes.labels(gpu_id=str(i), type='reserved').set(memory_reserved)
                except Exception as e:
                    logger.warning(f"Failed to collect GPU metrics: {e}")
                    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')

class HealthChecker:
    """Performs health checks on various services."""
    
    def __init__(self):
        self.redis_client = None
        self.health_history = {}
        
    async def initialize(self, redis_client=None):
        """Initialize health checker."""
        self.redis_client = redis_client
        
    async def check_api_health(self) -> HealthCheck:
        """Check API health."""
        start_time = time.time()
        try:
            # Basic health check
            import asyncio
            await asyncio.sleep(0.01)  # Simulate check
            
            response_time = time.time() - start_time
            
            return HealthCheck(
                service="api",
                status=ServiceStatus.HEALTHY,
                timestamp=datetime.utcnow(),
                response_time=response_time,
                details={
                    "status": "API is responding",
                    "endpoints": ["health", "analyze", "auth"]
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheck(
                service="api",
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                response_time=response_time,
                details={},
                error=str(e)
            )
            
    async def check_redis_health(self) -> HealthCheck:
        """Check Redis health."""
        start_time = time.time()
        try:
            if self.redis_client:
                await self.redis_client.ping()
                info = await self.redis_client.info()
                
                response_time = time.time() - start_time
                
                return HealthCheck(
                    service="redis",
                    status=ServiceStatus.HEALTHY,
                    timestamp=datetime.utcnow(),
                    response_time=response_time,
                    details={
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory": info.get("used_memory", 0),
                        "uptime_seconds": info.get("uptime_in_seconds", 0)
                    }
                )
            else:
                raise Exception("Redis client not available")
                
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheck(
                service="redis",
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                response_time=response_time,
                details={},
                error=str(e)
            )
            
    async def check_model_health(self, model_manager) -> HealthCheck:
        """Check model health."""
        start_time = time.time()
        try:
            if model_manager and hasattr(model_manager, 'models'):
                loaded_models = len(model_manager.models)
                
                # Test inference if models are loaded
                inference_status = "ready" if loaded_models > 0 else "no_models"
                
                response_time = time.time() - start_time
                
                status = ServiceStatus.HEALTHY if loaded_models > 0 else ServiceStatus.DEGRADED
                
                return HealthCheck(
                    service="models",
                    status=status,
                    timestamp=datetime.utcnow(),
                    response_time=response_time,
                    details={
                        "loaded_models": loaded_models,
                        "inference_status": inference_status,
                        "models": list(model_manager.models.keys())
                    }
                )
            else:
                raise Exception("Model manager not available")
                
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheck(
                service="models",
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                response_time=response_time,
                details={},
                error=str(e)
            )
            
    async def check_gpu_health(self) -> HealthCheck:
        """Check GPU health."""
        start_time = time.time()
        try:
            if torch and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_details = {}
                
                for i in range(gpu_count):
                    device = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    gpu_details[f"gpu_{i}"] = {
                        "name": device.name,
                        "memory_allocated": memory_allocated,
                        "memory_reserved": memory_reserved,
                        "memory_total": device.total_memory
                    }
                
                response_time = time.time() - start_time
                
                return HealthCheck(
                    service="gpu",
                    status=ServiceStatus.HEALTHY,
                    timestamp=datetime.utcnow(),
                    response_time=response_time,
                    details={
                        "gpu_count": gpu_count,
                        "gpus": gpu_details
                    }
                )
            else:
                response_time = time.time() - start_time
                return HealthCheck(
                    service="gpu",
                    status=ServiceStatus.DEGRADED,
                    timestamp=datetime.utcnow(),
                    response_time=response_time,
                    details={"message": "No GPU available, using CPU"},
                    error="CUDA not available"
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheck(
                service="gpu",
                status=ServiceStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                response_time=response_time,
                details={},
                error=str(e)
            )
            
    async def check_all_services(self, model_manager=None) -> Dict[str, HealthCheck]:
        """Check health of all services."""
        health_checks = {}
        
        # Run health checks concurrently
        tasks = [
            ("api", self.check_api_health()),
            ("redis", self.check_redis_health()),
            ("models", self.check_model_health(model_manager)),
            ("gpu", self.check_gpu_health())
        ]
        
        for service, task in tasks:
            try:
                health_checks[service] = await task
            except Exception as e:
                health_checks[service] = HealthCheck(
                    service=service,
                    status=ServiceStatus.UNKNOWN,
                    timestamp=datetime.utcnow(),
                    response_time=0.0,
                    details={},
                    error=f"Health check failed: {str(e)}"
                )
                
        return health_checks
        
    def get_overall_status(self, health_checks: Dict[str, HealthCheck]) -> ServiceStatus:
        """Get overall system status based on individual service checks."""
        if not health_checks:
            return ServiceStatus.UNKNOWN
            
        statuses = [check.status for check in health_checks.values()]
        
        if all(status == ServiceStatus.HEALTHY for status in statuses):
            return ServiceStatus.HEALTHY
        elif any(status == ServiceStatus.UNHEALTHY for status in statuses):
            return ServiceStatus.UNHEALTHY
        elif any(status == ServiceStatus.DEGRADED for status in statuses):
            return ServiceStatus.DEGRADED
        else:
            return ServiceStatus.UNKNOWN

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alert_rules = {}
        self.active_alerts = {}
        self.notification_handlers = []
        
    def add_alert_rule(self, name: str, condition: callable, severity: str, message: str):
        """Add an alert rule."""
        self.alert_rules[name] = {
            "condition": condition,
            "severity": severity,
            "message": message,
            "last_triggered": None
        }
        
    async def check_alerts(self, metrics: Dict[str, Any], health_checks: Dict[str, HealthCheck]):
        """Check alert conditions."""
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule["condition"](metrics, health_checks):
                    await self._trigger_alert(rule_name, rule)
                else:
                    await self._resolve_alert(rule_name)
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
                
    async def _trigger_alert(self, rule_name: str, rule: Dict[str, Any]):
        """Trigger an alert."""
        if rule_name not in self.active_alerts:
            alert = {
                "name": rule_name,
                "severity": rule["severity"],
                "message": rule["message"],
                "triggered_at": datetime.utcnow(),
                "count": 1
            }
            self.active_alerts[rule_name] = alert
            
            # Send notifications
            for handler in self.notification_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Notification handler failed: {e}")
        else:
            self.active_alerts[rule_name]["count"] += 1
            
    async def _resolve_alert(self, rule_name: str):
        """Resolve an alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts.pop(rule_name)
            alert["resolved_at"] = datetime.utcnow()
            
            # Send resolution notifications
            for handler in self.notification_handlers:
                try:
                    await handler(alert, resolved=True)
                except Exception as e:
                    logger.error(f"Resolution notification failed: {e}")

# Pre-defined alert rules
def create_default_alert_rules(alert_manager: AlertManager):
    """Create default alert rules."""
    
    # High CPU usage
    alert_manager.add_alert_rule(
        "high_cpu_usage",
        lambda m, h: m.get("cpu_usage", 0) > 90,
        "warning",
        "High CPU usage detected (>90%)"
    )
    
    # High memory usage
    alert_manager.add_alert_rule(
        "high_memory_usage",
        lambda m, h: m.get("memory_usage_percent", 0) > 85,
        "warning",
        "High memory usage detected (>85%)"
    )
    
    # GPU memory exhaustion
    alert_manager.add_alert_rule(
        "gpu_memory_high",
        lambda m, h: m.get("gpu_memory_usage_percent", 0) > 95,
        "critical",
        "GPU memory usage critical (>95%)"
    )
    
    # Service unhealthy
    alert_manager.add_alert_rule(
        "service_unhealthy",
        lambda m, h: any(
            check.status == ServiceStatus.UNHEALTHY 
            for check in h.values()
        ),
        "critical",
        "One or more services are unhealthy"
    )
    
    # High error rate
    alert_manager.add_alert_rule(
        "high_error_rate",
        lambda m, h: m.get("error_rate", 0) > 0.05,
        "warning",
        "High error rate detected (>5%)"
    )

# Global monitoring instance
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
alert_manager = AlertManager()

# Initialize default alert rules
create_default_alert_rules(alert_manager)