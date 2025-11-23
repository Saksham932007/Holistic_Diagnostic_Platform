#!/usr/bin/env python3
"""
Production Deployment Orchestrator

Advanced deployment orchestration script for medical platform
with blue-green deployment, health checks, and rollback capabilities.
"""

import os
import sys
import time
import json
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str
    namespace: str
    image_tag: str
    replicas: int
    health_check_timeout: int = 300
    blue_green_enabled: bool = True
    auto_rollback: bool = True
    resource_limits: Dict[str, str] = None

class DeploymentOrchestrator:
    """Orchestrates production deployment."""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize deployment orchestrator."""
        self.config = config
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.k8s_dir = self.project_root / "k8s"
        
        # Deployment state
        self.current_deployment = None
        self.backup_deployment = None
        self.rollback_available = False
        
        # Validate prerequisites
        self._validate_prerequisites()
    
    def _validate_prerequisites(self):
        """Validate deployment prerequisites."""
        logger.info("Validating deployment prerequisites...")
        
        # Check kubectl
        try:
            subprocess.run(['kubectl', 'version', '--client'], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("kubectl not available or not configured")
        
        # Check Docker
        try:
            subprocess.run(['docker', '--version'], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("Docker not available")
        
        # Check Kubernetes manifests
        if not self.k8s_dir.exists():
            raise RuntimeError(f"Kubernetes manifests directory not found: {self.k8s_dir}")
        
        logger.info("Prerequisites validation passed")
    
    def deploy(self) -> bool:
        """Execute deployment."""
        logger.info(f"Starting deployment to {self.config.environment}")
        
        try:
            # Pre-deployment checks
            self._pre_deployment_checks()
            
            # Build and push Docker image
            if not self._build_and_push_image():
                return False
            
            # Deploy based on strategy
            if self.config.blue_green_enabled:
                return self._blue_green_deploy()
            else:
                return self._rolling_deploy()
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            if self.config.auto_rollback:
                self._rollback()
            return False
    
    def _pre_deployment_checks(self):
        """Execute pre-deployment checks."""
        logger.info("Running pre-deployment checks...")
        
        # Check cluster connectivity
        try:
            result = subprocess.run([
                'kubectl', 'get', 'nodes'
            ], check=True, capture_output=True, text=True)
            logger.info("Kubernetes cluster accessible")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Cannot access Kubernetes cluster: {e}")
        
        # Check namespace
        try:
            subprocess.run([
                'kubectl', 'get', 'namespace', self.config.namespace
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.info(f"Creating namespace: {self.config.namespace}")
            subprocess.run([
                'kubectl', 'create', 'namespace', self.config.namespace
            ], check=True)
        
        # Run tests
        self._run_tests()
        
        # Check resource requirements
        self._check_resource_requirements()
        
        logger.info("Pre-deployment checks passed")
    
    def _build_and_push_image(self) -> bool:
        """Build and push Docker image."""
        logger.info("Building Docker image...")
        
        image_name = f"medical-platform:{self.config.image_tag}"
        
        try:
            # Build image
            build_cmd = [
                'docker', 'build',
                '-t', image_name,
                '-f', str(self.project_root / 'Dockerfile'),
                str(self.project_root)
            ]
            
            subprocess.run(build_cmd, check=True)
            logger.info(f"Image built successfully: {image_name}")
            
            # Push to registry (if configured)
            registry = os.getenv('DOCKER_REGISTRY')
            if registry:
                full_image_name = f"{registry}/{image_name}"
                
                # Tag for registry
                subprocess.run([
                    'docker', 'tag', image_name, full_image_name
                ], check=True)
                
                # Push to registry
                subprocess.run([
                    'docker', 'push', full_image_name
                ], check=True)
                
                logger.info(f"Image pushed to registry: {full_image_name}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Image build/push failed: {e}")
            return False
    
    def _blue_green_deploy(self) -> bool:
        """Execute blue-green deployment."""
        logger.info("Executing blue-green deployment...")
        
        try:
            # Determine current and new environments
            current_env = self._get_current_environment()
            new_env = 'blue' if current_env == 'green' else 'green'
            
            logger.info(f"Current environment: {current_env}, deploying to: {new_env}")
            
            # Deploy to new environment
            if not self._deploy_to_environment(new_env):
                return False
            
            # Health check new environment
            if not self._health_check(new_env):
                logger.error("Health check failed for new environment")
                self._cleanup_environment(new_env)
                return False
            
            # Switch traffic to new environment
            if not self._switch_traffic(new_env):
                return False
            
            # Cleanup old environment
            if current_env != 'unknown':
                self._cleanup_environment(current_env)
            
            logger.info("Blue-green deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return False
    
    def _rolling_deploy(self) -> bool:
        """Execute rolling deployment."""
        logger.info("Executing rolling deployment...")
        
        try:
            # Update deployment manifests
            self._update_manifests()
            
            # Apply manifests
            manifests = [
                'configmap.yaml',
                'secret.yaml', 
                'deployment.yaml',
                'service.yaml',
                'ingress.yaml'
            ]
            
            for manifest in manifests:
                manifest_path = self.k8s_dir / manifest
                if manifest_path.exists():
                    subprocess.run([
                        'kubectl', 'apply',
                        '-f', str(manifest_path),
                        '-n', self.config.namespace
                    ], check=True)
            
            # Wait for rollout
            subprocess.run([
                'kubectl', 'rollout', 'status',
                f'deployment/medical-platform-api',
                '-n', self.config.namespace,
                f'--timeout={self.config.health_check_timeout}s'
            ], check=True)
            
            # Health check
            if not self._health_check('current'):
                return False
            
            logger.info("Rolling deployment completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Rolling deployment failed: {e}")
            return False
    
    def _get_current_environment(self) -> str:
        """Get currently active environment."""
        try:
            result = subprocess.run([
                'kubectl', 'get', 'service', 'medical-platform-service',
                '-n', self.config.namespace,
                '-o', 'jsonpath={.spec.selector.environment}'
            ], capture_output=True, text=True, check=True)
            
            return result.stdout.strip() or 'unknown'
            
        except subprocess.CalledProcessError:
            return 'unknown'
    
    def _deploy_to_environment(self, environment: str) -> bool:
        """Deploy to specific environment (blue/green)."""
        logger.info(f"Deploying to {environment} environment...")
        
        try:
            # Create environment-specific manifests
            env_manifests = self._create_environment_manifests(environment)
            
            # Apply manifests
            for manifest_content in env_manifests:
                # Write to temporary file
                temp_file = f"/tmp/{environment}-manifest.yaml"
                with open(temp_file, 'w') as f:
                    f.write(manifest_content)
                
                # Apply manifest
                subprocess.run([
                    'kubectl', 'apply', '-f', temp_file,
                    '-n', self.config.namespace
                ], check=True)
                
                # Clean up temp file
                os.unlink(temp_file)
            
            # Wait for deployment
            subprocess.run([
                'kubectl', 'rollout', 'status',
                f'deployment/medical-platform-api-{environment}',
                '-n', self.config.namespace,
                f'--timeout={self.config.health_check_timeout}s'
            ], check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment to {environment} failed: {e}")
            return False
    
    def _create_environment_manifests(self, environment: str) -> List[str]:
        """Create environment-specific Kubernetes manifests."""
        manifests = []
        
        # Deployment manifest
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-platform-api-{environment}
  labels:
    app: medical-platform-api
    environment: {environment}
spec:
  replicas: {self.config.replicas}
  selector:
    matchLabels:
      app: medical-platform-api
      environment: {environment}
  template:
    metadata:
      labels:
        app: medical-platform-api
        environment: {environment}
    spec:
      containers:
      - name: api
        image: medical-platform:{self.config.image_tag}
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "{self.config.environment}"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: medical-platform-secrets
              key: database-url
        resources:
          limits:
            cpu: {self.config.resource_limits.get('cpu', '1000m')}
            memory: {self.config.resource_limits.get('memory', '2Gi')}
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
"""
        
        manifests.append(deployment_yaml)
        
        # Service manifest (for testing)
        service_yaml = f"""
apiVersion: v1
kind: Service
metadata:
  name: medical-platform-service-{environment}
  labels:
    app: medical-platform-api
    environment: {environment}
spec:
  selector:
    app: medical-platform-api
    environment: {environment}
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
"""
        
        manifests.append(service_yaml)
        
        return manifests
    
    def _health_check(self, environment: str) -> bool:
        """Perform health check on deployment."""
        logger.info(f"Performing health check on {environment} environment...")
        
        service_name = f"medical-platform-service-{environment}" if environment != 'current' else "medical-platform-service"
        
        # Wait for pods to be ready
        try:
            for attempt in range(self.config.health_check_timeout // 10):
                result = subprocess.run([
                    'kubectl', 'get', 'pods',
                    '-l', f'app=medical-platform-api',
                    '-n', self.config.namespace,
                    '-o', 'jsonpath={.items[*].status.phase}'
                ], capture_output=True, text=True, check=True)
                
                pod_statuses = result.stdout.strip().split()
                if all(status == 'Running' for status in pod_statuses):
                    logger.info("All pods are running")
                    break
                
                logger.info(f"Waiting for pods to be ready... (attempt {attempt + 1})")
                time.sleep(10)
            else:
                logger.error("Pods failed to start within timeout")
                return False
            
            # Test health endpoint
            for attempt in range(30):  # 5 minutes total
                try:
                    result = subprocess.run([
                        'kubectl', 'run', 'health-check-pod',
                        '--rm', '-i', '--restart=Never',
                        '--image=curlimages/curl',
                        '-n', self.config.namespace,
                        '--', 'curl', '-f',
                        f'http://{service_name}/health'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        logger.info("Health check passed")
                        return True
                        
                except subprocess.TimeoutExpired:
                    pass
                
                logger.info(f"Health check attempt {attempt + 1}/30...")
                time.sleep(10)
            
            logger.error("Health check failed")
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _switch_traffic(self, new_environment: str) -> bool:
        """Switch traffic to new environment."""
        logger.info(f"Switching traffic to {new_environment} environment...")
        
        try:
            # Update main service selector
            subprocess.run([
                'kubectl', 'patch', 'service', 'medical-platform-service',
                '-n', self.config.namespace,
                '--type=merge',
                '-p', f'{{"spec":{{"selector":{{"environment":"{new_environment}"}}}}}}'
            ], check=True)
            
            logger.info("Traffic switched successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to switch traffic: {e}")
            return False
    
    def _cleanup_environment(self, environment: str):
        """Clean up old environment."""
        logger.info(f"Cleaning up {environment} environment...")
        
        try:
            # Delete deployment
            subprocess.run([
                'kubectl', 'delete', 'deployment',
                f'medical-platform-api-{environment}',
                '-n', self.config.namespace,
                '--ignore-not-found'
            ], check=True)
            
            # Delete service
            subprocess.run([
                'kubectl', 'delete', 'service',
                f'medical-platform-service-{environment}',
                '-n', self.config.namespace,
                '--ignore-not-found'
            ], check=True)
            
            logger.info(f"Cleanup of {environment} environment completed")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Cleanup warnings: {e}")
    
    def _update_manifests(self):
        """Update Kubernetes manifests with new image tag."""
        deployment_file = self.k8s_dir / 'deployment.yaml'
        
        if deployment_file.exists():
            with open(deployment_file, 'r') as f:
                deployment = yaml.safe_load(f)
            
            # Update image tag
            containers = deployment['spec']['template']['spec']['containers']
            for container in containers:
                if 'medical-platform' in container.get('image', ''):
                    container['image'] = f"medical-platform:{self.config.image_tag}"
            
            # Update replicas
            deployment['spec']['replicas'] = self.config.replicas
            
            with open(deployment_file, 'w') as f:
                yaml.dump(deployment, f, default_flow_style=False)
    
    def _run_tests(self):
        """Run tests before deployment."""
        logger.info("Running tests...")
        
        try:
            # Run unit tests
            subprocess.run([
                sys.executable, '-m', 'pytest',
                str(self.project_root / 'tests'),
                '-v', '--tb=short'
            ], check=True, cwd=str(self.project_root))
            
            logger.info("Tests passed")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Tests failed: {e}")
    
    def _check_resource_requirements(self):
        """Check if cluster has sufficient resources."""
        logger.info("Checking resource requirements...")
        
        try:
            # Get node resources
            result = subprocess.run([
                'kubectl', 'top', 'nodes', '--no-headers'
            ], capture_output=True, text=True, check=True)
            
            # Simple check - ensure we have at least one node
            if not result.stdout.strip():
                logger.warning("Could not get node resource information")
            else:
                logger.info("Resource requirements check passed")
                
        except subprocess.CalledProcessError:
            logger.warning("Could not check resource requirements")
    
    def _rollback(self) -> bool:
        """Rollback to previous deployment."""
        logger.info("Initiating rollback...")
        
        try:
            subprocess.run([
                'kubectl', 'rollout', 'undo',
                'deployment/medical-platform-api',
                '-n', self.config.namespace
            ], check=True)
            
            # Wait for rollback
            subprocess.run([
                'kubectl', 'rollout', 'status',
                'deployment/medical-platform-api',
                '-n', self.config.namespace,
                '--timeout=300s'
            ], check=True)
            
            logger.info("Rollback completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            return False

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy medical platform to production')
    parser.add_argument('--environment', default='staging', help='Deployment environment')
    parser.add_argument('--namespace', default='medical-platform', help='Kubernetes namespace')
    parser.add_argument('--image-tag', default='latest', help='Docker image tag')
    parser.add_argument('--replicas', type=int, default=3, help='Number of replicas')
    parser.add_argument('--no-blue-green', action='store_true', help='Disable blue-green deployment')
    parser.add_argument('--no-auto-rollback', action='store_true', help='Disable auto rollback')
    
    args = parser.parse_args()
    
    # Create deployment configuration
    config = DeploymentConfig(
        environment=args.environment,
        namespace=args.namespace,
        image_tag=args.image_tag,
        replicas=args.replicas,
        blue_green_enabled=not args.no_blue_green,
        auto_rollback=not args.no_auto_rollback,
        resource_limits={
            'cpu': '1000m',
            'memory': '2Gi'
        }
    )
    
    # Execute deployment
    orchestrator = DeploymentOrchestrator(config)
    success = orchestrator.deploy()
    
    if success:
        logger.info("Deployment completed successfully")
        sys.exit(0)
    else:
        logger.error("Deployment failed")
        sys.exit(1)

if __name__ == '__main__':
    main()