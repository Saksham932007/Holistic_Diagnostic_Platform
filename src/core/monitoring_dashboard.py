"""
Real-time Monitoring Dashboard

Advanced real-time monitoring system for platform health,
performance metrics, and clinical operations.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

import websocket
import redis
import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest

from src.core.config import settings
from src.core.audit import audit_logger

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricValue:
    """Represents a metric value."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None
    labels: Dict[str, str] = None

@dataclass
class Alert:
    """Represents an alert."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    metric_name: str
    threshold: float
    current_value: float
    tags: Dict[str, str] = None
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None

class MetricsCollector:
    """Collects system and application metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.redis_client = None
        self._init_redis()
        self._init_prometheus_metrics()
    
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # System metrics
        self.metrics['cpu_usage'] = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.metrics['memory_usage'] = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.metrics['disk_usage'] = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Application metrics
        self.metrics['active_sessions'] = Gauge(
            'app_active_sessions_total',
            'Number of active user sessions',
            registry=self.registry
        )
        
        self.metrics['api_requests'] = Counter(
            'app_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['inference_time'] = Histogram(
            'model_inference_duration_seconds',
            'Model inference duration',
            ['model_name', 'model_type'],
            registry=self.registry
        )
        
        self.metrics['inference_count'] = Counter(
            'model_inference_total',
            'Total model inferences',
            ['model_name', 'model_type', 'status'],
            registry=self.registry
        )
        
        # Medical platform metrics
        self.metrics['studies_processed'] = Counter(
            'medical_studies_processed_total',
            'Total medical studies processed',
            ['modality', 'status'],
            registry=self.registry
        )
        
        self.metrics['findings_detected'] = Counter(
            'medical_findings_detected_total',
            'Total medical findings detected',
            ['finding_type', 'confidence_level'],
            registry=self.registry
        )
        
        self.metrics['reports_generated'] = Counter(
            'medical_reports_generated_total',
            'Total medical reports generated',
            ['report_type', 'template'],
            registry=self.registry
        )
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        timestamp = datetime.now()
        
        # Store in Redis if available
        if self.redis_client:
            try:
                metric_key = f"metrics:{name}:{timestamp.isoformat()}"
                metric_data = {
                    'value': value,
                    'timestamp': timestamp.isoformat(),
                    'labels': json.dumps(labels or {})
                }
                self.redis_client.hset(metric_key, mapping=metric_data)
                self.redis_client.expire(metric_key, 86400)  # 24 hours TTL
                
                # Add to time series
                ts_key = f"timeseries:{name}"
                self.redis_client.zadd(ts_key, {timestamp.timestamp(): value})
                self.redis_client.expire(ts_key, 86400)
                
            except Exception as e:
                logger.error(f"Failed to store metric in Redis: {e}")
        
        # Update Prometheus metric if it exists
        if name in self.metrics:
            prometheus_metric = self.metrics[name]
            if labels:
                if hasattr(prometheus_metric, 'labels'):
                    prometheus_metric.labels(**labels).set(value)
                else:
                    prometheus_metric.inc(value)
            else:
                if hasattr(prometheus_metric, 'set'):
                    prometheus_metric.set(value)
                else:
                    prometheus_metric.inc(value)
    
    def get_metric_history(self, name: str, duration_minutes: int = 60) -> List[MetricValue]:
        """Get metric history from Redis."""
        if not self.redis_client:
            return []
        
        try:
            ts_key = f"timeseries:{name}"
            end_time = time.time()
            start_time = end_time - (duration_minutes * 60)
            
            results = self.redis_client.zrangebyscore(
                ts_key, start_time, end_time, withscores=True
            )
            
            return [
                MetricValue(
                    name=name,
                    value=float(value),
                    timestamp=datetime.fromtimestamp(timestamp)
                )
                for value, timestamp in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return []
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager."""
        self.metrics_collector = metrics_collector
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = []
        self.subscribers = set()
        self._load_default_alert_rules()
    
    def _load_default_alert_rules(self):
        """Load default alert rules."""
        self.alert_rules = {
            'high_cpu_usage': {
                'metric': 'cpu_usage',
                'threshold': 90.0,
                'comparison': 'greater_than',
                'severity': AlertSeverity.WARNING,
                'description': 'High CPU usage detected'
            },
            'high_memory_usage': {
                'metric': 'memory_usage',
                'threshold': 85.0,
                'comparison': 'greater_than',
                'severity': AlertSeverity.WARNING,
                'description': 'High memory usage detected'
            },
            'high_disk_usage': {
                'metric': 'disk_usage',
                'threshold': 95.0,
                'comparison': 'greater_than',
                'severity': AlertSeverity.CRITICAL,
                'description': 'Critical disk usage detected'
            },
            'slow_inference': {
                'metric': 'inference_time',
                'threshold': 10.0,
                'comparison': 'greater_than',
                'severity': AlertSeverity.WARNING,
                'description': 'Slow model inference detected'
            }
        }
    
    def add_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "greater_than",
        severity: AlertSeverity = AlertSeverity.WARNING,
        description: str = ""
    ):
        """Add a new alert rule."""
        self.alert_rules[rule_name] = {
            'metric': metric_name,
            'threshold': threshold,
            'comparison': comparison,
            'severity': severity,
            'description': description
        }
    
    def evaluate_alerts(self):
        """Evaluate all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule['metric']
            threshold = rule['threshold']
            comparison = rule['comparison']
            
            # Get recent metric values
            recent_metrics = self.metrics_collector.get_metric_history(metric_name, 5)
            
            if not recent_metrics:
                continue
            
            # Use the most recent value
            current_value = recent_metrics[-1].value
            
            # Evaluate condition
            triggered = False
            if comparison == 'greater_than':
                triggered = current_value > threshold
            elif comparison == 'less_than':
                triggered = current_value < threshold
            elif comparison == 'equal':
                triggered = abs(current_value - threshold) < 0.01
            
            if triggered and rule_name not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    alert_id=f"{rule_name}_{int(time.time())}",
                    name=rule_name,
                    description=rule['description'],
                    severity=rule['severity'],
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=current_value
                )
                
                self.active_alerts[rule_name] = alert
                self.alert_history.append(alert)
                self._notify_subscribers(alert)
                
            elif not triggered and rule_name in self.active_alerts:
                # Resolve existing alert
                alert = self.active_alerts[rule_name]
                alert.resolved = True
                alert.resolved_timestamp = datetime.now()
                del self.active_alerts[rule_name]
                self._notify_subscribers(alert)
    
    def _notify_subscribers(self, alert: Alert):
        """Notify alert subscribers."""
        alert_data = asdict(alert)
        alert_data['timestamp'] = alert.timestamp.isoformat()
        if alert.resolved_timestamp:
            alert_data['resolved_timestamp'] = alert.resolved_timestamp.isoformat()
        
        # Notify WebSocket subscribers
        for subscriber in list(self.subscribers):
            try:
                if hasattr(subscriber, 'send'):
                    subscriber.send(json.dumps({
                        'type': 'alert',
                        'data': alert_data
                    }))
            except Exception as e:
                logger.error(f"Failed to notify subscriber: {e}")
                self.subscribers.discard(subscriber)
    
    def subscribe(self, subscriber):
        """Subscribe to alerts."""
        self.subscribers.add(subscriber)
    
    def unsubscribe(self, subscriber):
        """Unsubscribe from alerts."""
        self.subscribers.discard(subscriber)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

class SystemMonitor:
    """Monitors system resources."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize system monitor."""
        self.metrics_collector = metrics_collector
        self.monitoring_active = False
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start system monitoring."""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU usage
            cpu_percent = await self._get_cpu_usage()
            self.metrics_collector.record_metric('cpu_usage', cpu_percent)
            
            # Memory usage
            memory_percent = await self._get_memory_usage()
            self.metrics_collector.record_metric('memory_usage', memory_percent)
            
            # Disk usage
            disk_percent = await self._get_disk_usage()
            self.metrics_collector.record_metric('disk_usage', disk_percent)
            
            # GPU metrics if available
            gpu_metrics = await self._get_gpu_metrics()
            if gpu_metrics:
                for i, (usage, memory) in enumerate(gpu_metrics):
                    self.metrics_collector.record_metric(
                        'gpu_usage', usage, {'gpu_id': str(i)}
                    )
                    self.metrics_collector.record_metric(
                        'gpu_memory_usage', memory, {'gpu_id': str(i)}
                    )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    async def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent
        except ImportError:
            return 0.0
    
    async def _get_disk_usage(self) -> float:
        """Get disk usage percentage."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return (disk.used / disk.total) * 100
        except ImportError:
            return 0.0
    
    async def _get_gpu_metrics(self) -> Optional[List[Tuple[float, float]]]:
        """Get GPU metrics if available."""
        try:
            import torch
            if torch.cuda.is_available():
                metrics = []
                for i in range(torch.cuda.device_count()):
                    # GPU utilization (simplified)
                    usage = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0.0
                    
                    # Memory usage
                    memory_used = torch.cuda.memory_allocated(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    memory_percent = (memory_used / memory_total) * 100
                    
                    metrics.append((usage, memory_percent))
                
                return metrics
        except ImportError:
            pass
        
        return None

class DashboardServer:
    """WebSocket server for real-time dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        """Initialize dashboard server."""
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.clients = set()
        self.server = None
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start WebSocket server."""
        try:
            import websockets
            
            self.server = await websockets.serve(
                self.handle_client,
                host,
                port
            )
            
            logger.info(f"Dashboard server started on ws://{host}:{port}")
            
            # Start periodic updates
            asyncio.create_task(self._periodic_updates())
            
        except ImportError:
            logger.error("websockets package not available")
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        self.clients.add(websocket)
        self.alert_manager.subscribe(websocket)
        
        try:
            # Send initial data
            await self._send_initial_data(websocket)
            
            # Listen for client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {message}")
                
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            self.clients.discard(websocket)
            self.alert_manager.unsubscribe(websocket)
    
    async def _send_initial_data(self, websocket):
        """Send initial dashboard data to client."""
        try:
            # Get recent metrics
            metric_names = ['cpu_usage', 'memory_usage', 'disk_usage']
            metrics_data = {}
            
            for metric_name in metric_names:
                history = self.metrics_collector.get_metric_history(metric_name, 60)
                metrics_data[metric_name] = [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'value': m.value
                    }
                    for m in history
                ]
            
            # Get active alerts
            active_alerts = [
                {
                    'id': alert.alert_id,
                    'name': alert.name,
                    'description': alert.description,
                    'severity': alert.severity.value,
                    'timestamp': alert.timestamp.isoformat(),
                    'current_value': alert.current_value,
                    'threshold': alert.threshold
                }
                for alert in self.alert_manager.get_active_alerts()
            ]
            
            initial_data = {
                'type': 'initial_data',
                'data': {
                    'metrics': metrics_data,
                    'alerts': active_alerts
                }
            }
            
            await websocket.send(json.dumps(initial_data))
            
        except Exception as e:
            logger.error(f"Failed to send initial data: {e}")
    
    async def _handle_client_message(self, websocket, data):
        """Handle message from client."""
        message_type = data.get('type')
        
        if message_type == 'get_metrics':
            metric_names = data.get('metrics', [])
            duration = data.get('duration_minutes', 60)
            
            metrics_data = {}
            for metric_name in metric_names:
                history = self.metrics_collector.get_metric_history(metric_name, duration)
                metrics_data[metric_name] = [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'value': m.value
                    }
                    for m in history
                ]
            
            response = {
                'type': 'metrics_data',
                'data': metrics_data
            }
            
            await websocket.send(json.dumps(response))
    
    async def _periodic_updates(self):
        """Send periodic updates to all clients."""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                if not self.clients:
                    continue
                
                # Get latest metrics
                current_metrics = {}
                for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
                    recent = self.metrics_collector.get_metric_history(metric_name, 1)
                    if recent:
                        current_metrics[metric_name] = {
                            'timestamp': recent[-1].timestamp.isoformat(),
                            'value': recent[-1].value
                        }
                
                if current_metrics:
                    update_data = {
                        'type': 'metrics_update',
                        'data': current_metrics
                    }
                    
                    # Send to all connected clients
                    message = json.dumps(update_data)
                    disconnected_clients = set()
                    
                    for client in self.clients:
                        try:
                            await client.send(message)
                        except Exception:
                            disconnected_clients.add(client)
                    
                    # Remove disconnected clients
                    self.clients -= disconnected_clients
                    
            except Exception as e:
                logger.error(f"Periodic update error: {e}")

class MonitoringDashboard:
    """Main monitoring dashboard coordinator."""
    
    def __init__(self):
        """Initialize monitoring dashboard."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.dashboard_server = DashboardServer(self.metrics_collector, self.alert_manager)
        
        self.running = False
        self.tasks = []
    
    async def start(self):
        """Start the monitoring dashboard."""
        if self.running:
            return
        
        self.running = True
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self.system_monitor.start_monitoring())
        self.tasks.append(monitor_task)
        
        # Start alert evaluation
        alert_task = asyncio.create_task(self._alert_evaluation_loop())
        self.tasks.append(alert_task)
        
        # Start dashboard server
        await self.dashboard_server.start_server()
        
        await audit_logger.log_event(
            "monitoring_dashboard_started",
            {"timestamp": datetime.now().isoformat()}
        )
        
        logger.info("Monitoring dashboard started")
    
    async def stop(self):
        """Stop the monitoring dashboard."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop system monitoring
        self.system_monitor.stop_monitoring()
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        await audit_logger.log_event(
            "monitoring_dashboard_stopped",
            {"timestamp": datetime.now().isoformat()}
        )
        
        logger.info("Monitoring dashboard stopped")
    
    async def _alert_evaluation_loop(self):
        """Continuous alert evaluation loop."""
        while self.running:
            try:
                self.alert_manager.evaluate_alerts()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        self.metrics_collector.record_metric(name, value, labels)
    
    def add_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "greater_than",
        severity: AlertSeverity = AlertSeverity.WARNING,
        description: str = ""
    ):
        """Add an alert rule."""
        self.alert_manager.add_alert_rule(
            rule_name, metric_name, threshold, comparison, severity, description
        )
    
    def get_metrics_export(self) -> str:
        """Get Prometheus metrics export."""
        return self.metrics_collector.export_prometheus_metrics()

# Global monitoring dashboard instance
monitoring_dashboard = MonitoringDashboard()