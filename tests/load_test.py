"""
Medical Platform Load Testing Framework

Comprehensive load testing suite using Locust for performance validation
of the medical image analysis platform under realistic load conditions.

Author: Holistic Diagnostic Platform Team
Version: 1.0.0
"""

import os
import time
import random
import json
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional
import requests
from locust import HttpUser, task, between, events
import numpy as np
from PIL import Image

class MedicalPlatformUser(HttpUser):
    """Simulate a medical platform user performing typical workflows."""
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    def on_start(self):
        """Initialize user session."""
        self.token = None
        self.analysis_results = []
        
        # Authenticate user
        self.authenticate()
        
        # Generate test medical images
        self.test_images = self._generate_test_images()
        
    def authenticate(self):
        """Authenticate and get access token."""
        response = self.client.post("/auth/token", data={
            "username": "load_test_user",
            "password": "load_test_password"
        }, catch_response=True)
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
            response.success()
        else:
            response.failure(f"Authentication failed: {response.status_code}")
    
    def _generate_test_images(self) -> List[Dict[str, Any]]:
        """Generate synthetic test images for load testing."""
        test_images = []
        
        # Generate different types of test images
        image_configs = [
            {"size": (128, 128, 64), "modality": "ct", "name": "chest_ct.nii.gz"},
            {"size": (256, 256, 32), "modality": "mri", "name": "brain_mri.nii.gz"},
            {"size": (512, 512, 16), "modality": "pet", "name": "whole_body_pet.nii.gz"},
            {"size": (224, 224, 1), "modality": "xray", "name": "chest_xray.jpg"},
        ]
        
        for config in image_configs:
            # Create synthetic medical image data
            if len(config["size"]) == 3:  # 3D image
                image_data = np.random.randint(0, 256, config["size"], dtype=np.uint8)
                # Add some structure to make it more realistic
                image_data = self._add_medical_structure(image_data, config["modality"])
                
                # Convert to bytes (simulate NIfTI file)
                image_bytes = image_data.tobytes()
            else:  # 2D image
                image_data = np.random.randint(0, 256, config["size"] + (1,), dtype=np.uint8)
                # Convert to PIL Image and then to bytes
                pil_image = Image.fromarray(image_data.squeeze(), mode='L')
                buffer = BytesIO()
                pil_image.save(buffer, format='JPEG')
                image_bytes = buffer.getvalue()
            
            test_images.append({
                "name": config["name"],
                "modality": config["modality"],
                "data": image_bytes,
                "size": len(image_bytes)
            })
            
        return test_images
    
    def _add_medical_structure(self, image_data: np.ndarray, modality: str) -> np.ndarray:
        """Add realistic medical structures to synthetic images."""
        if modality == "ct":
            # Add lung-like structures for CT
            center = np.array(image_data.shape) // 2
            for i in range(2):  # Two lungs
                lung_center = center + np.array([(-1)**i * 30, 0, 0])
                distances = np.linalg.norm(
                    np.indices(image_data.shape).T - lung_center, axis=-1
                )
                lung_mask = distances < 25
                image_data[lung_mask.T] = np.clip(image_data[lung_mask.T] + 50, 0, 255)
                
        elif modality == "mri":
            # Add brain-like structures for MRI
            center = np.array(image_data.shape) // 2
            distances = np.linalg.norm(
                np.indices(image_data.shape).T - center, axis=-1
            )
            brain_mask = distances < min(image_data.shape) // 3
            image_data[brain_mask.T] = np.clip(image_data[brain_mask.T] + 100, 0, 255)
            
        elif modality == "pet":
            # Add hotspots for PET
            for _ in range(3):
                hotspot = np.random.randint(0, min(image_data.shape), 3)
                distances = np.linalg.norm(
                    np.indices(image_data.shape).T - hotspot, axis=-1
                )
                hotspot_mask = distances < 10
                image_data[hotspot_mask.T] = 255
        
        return image_data
    
    @task(3)
    def health_check(self):
        """Perform health check (high frequency)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def get_models_info(self):
        """Get information about available models."""
        with self.client.get("/models/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Models info failed: {response.status_code}")
    
    @task(2)
    def segmentation_analysis(self):
        """Perform medical image segmentation analysis."""
        if not self.token:
            self.authenticate()
            
        # Select random test image
        test_image = random.choice(self.test_images)
        
        # Prepare request data
        files = {"image_file": (test_image["name"], test_image["data"], "application/octet-stream")}
        data = {
            "model_type": random.choice(["swin_unetr", "attention_unet"]),
            "confidence_threshold": random.uniform(0.5, 0.8)
        }
        
        with self.client.post(
            "/analyze/segmentation",
            files=files,
            data=data,
            catch_response=True,
            timeout=60  # Extended timeout for analysis
        ) as response:
            if response.status_code == 200:
                result = response.json()
                self.analysis_results.append(result["analysis_id"])
                response.success()
                
                # Record custom metrics
                events.request.fire(
                    request_type="ANALYSIS",
                    name="segmentation",
                    response_time=result.get("processing_time", 0) * 1000,  # Convert to ms
                    response_length=test_image["size"],
                    exception=None
                )
            else:
                response.failure(f"Segmentation analysis failed: {response.status_code}")
    
    @task(1)
    def classification_analysis(self):
        """Perform medical image classification analysis."""
        if not self.token:
            self.authenticate()
            
        # Select random test image
        test_image = random.choice(self.test_images)
        
        # Prepare request data
        files = {"image_file": (test_image["name"], test_image["data"], "application/octet-stream")}
        data = {
            "model_type": random.choice(["vit", "hybrid_cnn_transformer"]),
            "num_classes": random.choice([2, 3, 4]),
            "return_attention": random.choice([True, False])
        }
        
        with self.client.post(
            "/analyze/classification",
            files=files,
            data=data,
            catch_response=True,
            timeout=60
        ) as response:
            if response.status_code == 200:
                result = response.json()
                self.analysis_results.append(result["analysis_id"])
                response.success()
                
                # Record custom metrics
                events.request.fire(
                    request_type="ANALYSIS", 
                    name="classification",
                    response_time=result.get("processing_time", 0) * 1000,
                    response_length=test_image["size"],
                    exception=None
                )
            else:
                response.failure(f"Classification analysis failed: {response.status_code}")
    
    @task(1)
    def get_analysis_results(self):
        """Retrieve cached analysis results."""
        if not self.analysis_results:
            return
            
        analysis_id = random.choice(self.analysis_results)
        
        with self.client.get(
            f"/analysis/{analysis_id}/results",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # Results might have expired, remove from list
                if analysis_id in self.analysis_results:
                    self.analysis_results.remove(analysis_id)
                response.success()  # This is expected behavior
            else:
                response.failure(f"Get results failed: {response.status_code}")

class AdminUser(HttpUser):
    """Simulate admin user performing monitoring and management tasks."""
    
    wait_time = between(5, 15)  # Longer intervals for admin tasks
    weight = 1  # Lower weight than regular users
    
    def on_start(self):
        """Initialize admin session."""
        self.token = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate as admin user."""
        response = self.client.post("/auth/token", data={
            "username": "admin",
            "password": "admin_password"
        }, catch_response=True)
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
            response.success()
        else:
            response.failure(f"Admin authentication failed: {response.status_code}")
    
    @task(1)
    def check_system_status(self):
        """Check comprehensive system status."""
        endpoints = ["/health", "/models/info"]
        
        for endpoint in endpoints:
            with self.client.get(endpoint, catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Admin check failed for {endpoint}: {response.status_code}")

# Performance monitoring and custom events
@events.request.on(request_type="ANALYSIS", name="segmentation")
def segmentation_metrics(name, response_time, response_length, **kwargs):
    """Record segmentation-specific metrics."""
    print(f"Segmentation analysis completed in {response_time:.2f}ms, image size: {response_length} bytes")

@events.request.on(request_type="ANALYSIS", name="classification") 
def classification_metrics(name, response_time, response_length, **kwargs):
    """Record classification-specific metrics."""
    print(f"Classification analysis completed in {response_time:.2f}ms, image size: {response_length} bytes")

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize load testing environment."""
    print("Medical Platform Load Testing Started")
    print(f"Target host: {environment.host}")
    print(f"Users: {getattr(environment, 'user_count', 'N/A')}")

# Load testing scenarios
class QuickLoadTest(HttpUser):
    """Quick load test for CI/CD pipelines."""
    
    wait_time = between(1, 2)
    
    def on_start(self):
        """Quick authentication."""
        response = self.client.post("/auth/token", data={
            "username": "test_user",
            "password": "test_password"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
    
    @task
    def health_check_only(self):
        """Only perform health checks for quick testing."""
        self.client.get("/health")

class StressTest(MedicalPlatformUser):
    """Stress test with higher load and more concurrent operations."""
    
    wait_time = between(0.1, 1)  # Much shorter wait times
    
    @task(5)  # Increased frequency
    def rapid_segmentation(self):
        """Perform rapid segmentation requests."""
        self.segmentation_analysis()
    
    @task(5)  # Increased frequency
    def rapid_classification(self):
        """Perform rapid classification requests."""
        self.classification_analysis()

# Utility functions for custom test scenarios
def generate_load_test_data(num_images: int = 10) -> List[Dict[str, Any]]:
    """Generate a set of test medical images for load testing."""
    images = []
    
    for i in range(num_images):
        # Create various image sizes and types
        size = random.choice([(128, 128, 32), (256, 256, 64), (512, 512, 16)])
        modality = random.choice(["ct", "mri", "pet"])
        
        image_data = np.random.randint(0, 256, size, dtype=np.uint8)
        images.append({
            "name": f"test_image_{i}_{modality}.nii.gz",
            "data": image_data.tobytes(),
            "modality": modality,
            "size": image_data.nbytes
        })
    
    return images

def run_performance_benchmark(host: str, duration: int = 300, users: int = 10):
    """Run a performance benchmark test."""
    import subprocess
    import tempfile
    
    # Create temporary locustfile
    locustfile_content = f"""
from locust import HttpUser, task, between
import sys
sys.path.append('.')
from tests.load_test import MedicalPlatformUser

class BenchmarkUser(MedicalPlatformUser):
    wait_time = between(1, 3)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(locustfile_content)
        locustfile = f.name
    
    try:
        # Run Locust test
        cmd = [
            "locust",
            "-f", locustfile,
            "--headless",
            "-u", str(users),
            "-r", str(users // 5),  # Spawn rate
            "-t", f"{duration}s",
            "--host", host,
            "--csv", "benchmark_results"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Performance benchmark completed successfully")
            print("Results saved to benchmark_results_*.csv files")
        else:
            print(f"Benchmark failed: {result.stderr}")
            
    finally:
        os.unlink(locustfile)

if __name__ == "__main__":
    # Example usage for standalone testing
    print("Medical Platform Load Testing Framework")
    print("Use with Locust: locust -f load_test.py --host=http://localhost:8000")
    
    # Generate test data
    test_data = generate_load_test_data(5)
    print(f"Generated {len(test_data)} test images for load testing")