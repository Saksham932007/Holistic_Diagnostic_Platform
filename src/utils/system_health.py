#!/usr/bin/env python3
"""
System Health Monitor
Comprehensive system health monitoring and diagnostic utilities for the medical imaging platform.
Monitors system resources, service health, and provides real-time status reporting.
"""

import asyncio
import logging
import psutil
import time
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import subprocess
import socket
import signal
import gc
import resource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance and health metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    available_memory_gb: float
    total_memory_gb: float
    network_io_sent: int
    network_io_recv: int
    process_count: int
    load_average: Tuple[float, float, float]
    uptime_seconds: float
    temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    def is_healthy(self) -> bool:
        """Check if system metrics indicate healthy state."""
        return (
            self.cpu_percent < 80.0 and
            self.memory_percent < 85.0 and
            self.disk_usage_percent < 90.0 and
            self.available_memory_gb > 1.0
        )

@dataclass
class ServiceHealth:
    """Service health status information."""
    service_name: str
    status: str  # 'healthy', 'unhealthy', 'unknown'
    response_time_ms: Optional[float]
    last_check: str
    error_message: Optional[str] = None
    endpoint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert service health to dictionary."""
        return asdict(self)

class SystemResourceMonitor:
    """Monitor system resources and performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.monitoring_active = False
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1000
        self._lock = threading.Lock()
        
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            available_memory_gb = memory.available / (1024**3)
            total_memory_gb = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            net_io = psutil.net_io_counters()
            network_io_sent = net_io.bytes_sent
            network_io_recv = net_io.bytes_recv
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average
            load_average = psutil.getloadavg()
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature if available
                    for name, entries in temps.items():
                        if entries:
                            temperature = entries[0].current
                            break
            except (AttributeError, OSError):
                pass
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                available_memory_gb=available_memory_gb,
                total_memory_gb=total_memory_gb,
                network_io_sent=network_io_sent,
                network_io_recv=network_io_recv,
                process_count=process_count,
                load_average=load_average,
                uptime_seconds=uptime_seconds,
                temperature=temperature
            )
            
            # Store in history
            with self._lock:
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            raise
    
    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get aggregated metrics for the specified time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time period"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
        
        # Find peaks
        max_cpu = max(m.cpu_percent for m in recent_metrics)
        max_memory = max(m.memory_percent for m in recent_metrics)
        
        return {
            "time_period_minutes": minutes,
            "sample_count": len(recent_metrics),
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "disk_usage_percent": round(avg_disk, 2)
            },
            "peaks": {
                "max_cpu_percent": round(max_cpu, 2),
                "max_memory_percent": round(max_memory, 2)
            },
            "current_status": recent_metrics[-1].is_healthy()
        }

class ServiceHealthChecker:
    """Monitor health of various services and endpoints."""
    
    def __init__(self):
        self.service_configs = {
            "database": {
                "type": "tcp",
                "host": "localhost",
                "port": 5432,
                "timeout": 5
            },
            "redis": {
                "type": "tcp", 
                "host": "localhost",
                "port": 6379,
                "timeout": 5
            },
            "api_health": {
                "type": "http",
                "url": "http://localhost:8000/health",
                "timeout": 10
            },
            "ai_service": {
                "type": "http",
                "url": "http://localhost:8001/health",
                "timeout": 15
            }
        }
        
    async def check_tcp_service(self, host: str, port: int, timeout: int = 5) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check if TCP service is responding."""
        start_time = time.time()
        
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            
            response_time = (time.time() - start_time) * 1000
            return True, None, response_time
            
        except asyncio.TimeoutError:
            return False, f"Connection timeout after {timeout}s", None
        except Exception as e:
            return False, str(e), None
    
    async def check_http_service(self, url: str, timeout: int = 10) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check if HTTP service is responding."""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        return True, None, response_time
                    else:
                        return False, f"HTTP {response.status}", response_time
                        
        except asyncio.TimeoutError:
            return False, f"HTTP timeout after {timeout}s", None
        except Exception as e:
            return False, str(e), None
    
    async def check_service_health(self, service_name: str, config: Dict[str, Any]) -> ServiceHealth:
        """Check health of a single service."""
        start_time = datetime.now().isoformat()
        
        try:
            if config["type"] == "tcp":
                is_healthy, error, response_time = await self.check_tcp_service(
                    config["host"], config["port"], config["timeout"]
                )
            elif config["type"] == "http":
                is_healthy, error, response_time = await self.check_http_service(
                    config["url"], config["timeout"]
                )
            else:
                return ServiceHealth(
                    service_name=service_name,
                    status="unknown",
                    response_time_ms=None,
                    last_check=start_time,
                    error_message=f"Unknown service type: {config['type']}",
                    endpoint=config.get("url") or f"{config.get('host')}:{config.get('port')}"
                )
            
            return ServiceHealth(
                service_name=service_name,
                status="healthy" if is_healthy else "unhealthy",
                response_time_ms=response_time,
                last_check=start_time,
                error_message=error,
                endpoint=config.get("url") or f"{config.get('host')}:{config.get('port')}"
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                status="unhealthy",
                response_time_ms=None,
                last_check=start_time,
                error_message=str(e),
                endpoint=config.get("url") or f"{config.get('host')}:{config.get('port')}"
            )
    
    async def check_all_services(self) -> List[ServiceHealth]:
        """Check health of all configured services."""
        tasks = [
            self.check_service_health(name, config)
            for name, config in self.service_configs.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_checks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_name = list(self.service_configs.keys())[i]
                health_checks.append(ServiceHealth(
                    service_name=service_name,
                    status="unhealthy",
                    response_time_ms=None,
                    last_check=datetime.now().isoformat(),
                    error_message=str(result)
                ))
            else:
                health_checks.append(result)
        
        return health_checks

class SystemHealthMonitor:
    """Main system health monitoring coordinator."""
    
    def __init__(self):
        self.resource_monitor = SystemResourceMonitor()
        self.service_checker = ServiceHealthChecker()
        self.monitoring_active = False
        self.monitor_task = None
        self.health_history: List[Dict[str, Any]] = []
        self.max_history = 500
        self._shutdown_event = asyncio.Event()
        
    async def get_full_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status including system metrics and service health."""
        try:
            # Get system metrics
            system_metrics = self.resource_monitor.get_current_metrics()
            
            # Get service health
            service_health = await self.service_checker.check_all_services()
            
            # Calculate overall health
            system_healthy = system_metrics.is_healthy()
            services_healthy = all(s.status == "healthy" for s in service_health)
            overall_healthy = system_healthy and services_healthy
            
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy" if overall_healthy else "unhealthy",
                "system_metrics": system_metrics.to_dict(),
                "service_health": [s.to_dict() for s in service_health],
                "summary": {
                    "system_healthy": system_healthy,
                    "services_healthy": services_healthy,
                    "total_services": len(service_health),
                    "healthy_services": sum(1 for s in service_health if s.status == "healthy"),
                    "unhealthy_services": sum(1 for s in service_health if s.status == "unhealthy")
                }
            }
            
            # Store in history
            self.health_history.append(health_status)
            if len(self.health_history) > self.max_history:
                self.health_history.pop(0)
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    async def start_continuous_monitoring(self, interval_seconds: int = 30):
        """Start continuous health monitoring."""
        self.monitoring_active = True
        
        try:
            while self.monitoring_active and not self._shutdown_event.is_set():
                try:
                    health_status = await self.get_full_health_status()
                    
                    # Log critical issues
                    if health_status.get("overall_status") == "unhealthy":
                        logger.warning(f"System health check failed: {health_status.get('summary', {})}")
                        
                        # Log specific unhealthy services
                        for service in health_status.get("service_health", []):
                            if service.get("status") == "unhealthy":
                                logger.error(
                                    f"Service {service.get('service_name')} is unhealthy: "
                                    f"{service.get('error_message', 'Unknown error')}"
                                )
                    
                    # Wait for next check
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in continuous monitoring: {str(e)}")
                    await asyncio.sleep(interval_seconds)
                    
        except Exception as e:
            logger.error(f"Critical error in monitoring loop: {str(e)}")
        finally:
            self.monitoring_active = False
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        self._shutdown_event.set()
    
    def get_health_trend(self, hours: int = 1) -> Dict[str, Any]:
        """Get health trend analysis for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_health = [
            h for h in self.health_history
            if datetime.fromisoformat(h["timestamp"]) > cutoff_time
        ]
        
        if not recent_health:
            return {"error": "No health data available for the specified time period"}
        
        # Calculate health statistics
        total_checks = len(recent_health)
        healthy_checks = sum(1 for h in recent_health if h.get("overall_status") == "healthy")
        health_percentage = (healthy_checks / total_checks) * 100
        
        # Service availability
        service_stats = {}
        for health_check in recent_health:
            for service in health_check.get("service_health", []):
                service_name = service.get("service_name")
                if service_name not in service_stats:
                    service_stats[service_name] = {"total": 0, "healthy": 0}
                
                service_stats[service_name]["total"] += 1
                if service.get("status") == "healthy":
                    service_stats[service_name]["healthy"] += 1
        
        service_availability = {
            name: (stats["healthy"] / stats["total"]) * 100
            for name, stats in service_stats.items()
        }
        
        return {
            "time_period_hours": hours,
            "total_health_checks": total_checks,
            "overall_health_percentage": round(health_percentage, 2),
            "service_availability": {
                name: round(availability, 2)
                for name, availability in service_availability.items()
            },
            "current_status": recent_health[-1].get("overall_status") if recent_health else "unknown"
        }
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        try:
            current_health = await self.get_full_health_status()
            trend_1h = self.get_health_trend(hours=1)
            trend_24h = self.get_health_trend(hours=24)
            system_summary = self.resource_monitor.get_metrics_summary(minutes=15)
            
            return {
                "report_timestamp": datetime.now().isoformat(),
                "current_health": current_health,
                "trends": {
                    "last_hour": trend_1h,
                    "last_24_hours": trend_24h
                },
                "system_performance": system_summary,
                "recommendations": self._generate_recommendations(current_health, system_summary)
            }
            
        except Exception as e:
            logger.error(f"Error generating health report: {str(e)}")
            return {
                "report_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _generate_recommendations(self, current_health: Dict[str, Any], system_summary: Dict[str, Any]) -> List[str]:
        """Generate health and performance recommendations."""
        recommendations = []
        
        try:
            # Check system metrics
            system_metrics = current_health.get("system_metrics", {})
            
            if system_metrics.get("cpu_percent", 0) > 80:
                recommendations.append("High CPU usage detected. Consider scaling or optimizing workloads.")
            
            if system_metrics.get("memory_percent", 0) > 85:
                recommendations.append("High memory usage detected. Check for memory leaks or increase available RAM.")
            
            if system_metrics.get("disk_usage_percent", 0) > 90:
                recommendations.append("High disk usage detected. Clean up logs or increase disk capacity.")
            
            if system_metrics.get("available_memory_gb", 10) < 1:
                recommendations.append("Low available memory. Restart services or add more RAM.")
            
            # Check service health
            unhealthy_services = [
                s for s in current_health.get("service_health", [])
                if s.get("status") == "unhealthy"
            ]
            
            if unhealthy_services:
                service_names = [s.get("service_name") for s in unhealthy_services]
                recommendations.append(f"Unhealthy services detected: {', '.join(service_names)}. Check service logs and configuration.")
            
            # Check response times
            slow_services = [
                s for s in current_health.get("service_health", [])
                if s.get("response_time_ms", 0) > 5000
            ]
            
            if slow_services:
                service_names = [s.get("service_name") for s in slow_services]
                recommendations.append(f"Slow response times detected for: {', '.join(service_names)}. Investigate performance issues.")
            
            if not recommendations:
                recommendations.append("All systems are operating normally.")
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations

# CLI Interface
async def main():
    """Main CLI interface for system health monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="System Health Monitor")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--report", action="store_true", help="Generate health report")
    parser.add_argument("--check", action="store_true", help="Single health check")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    monitor = SystemHealthMonitor()
    
    try:
        if args.monitor:
            print("Starting continuous health monitoring...")
            print("Press Ctrl+C to stop")
            
            def signal_handler(sig, frame):
                print("\nStopping health monitoring...")
                monitor.stop_monitoring()
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            await monitor.start_continuous_monitoring(args.interval)
            
        elif args.report:
            print("Generating health report...")
            report = await monitor.generate_health_report()
            print(json.dumps(report, indent=2))
            
        elif args.check:
            print("Performing health check...")
            health = await monitor.get_full_health_status()
            print(json.dumps(health, indent=2))
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nHealth monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())