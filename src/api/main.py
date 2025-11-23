"""
Medical Image Analysis API

Production-grade FastAPI service for medical image segmentation,
classification, and analysis with HIPAA compliance and comprehensive
monitoring capabilities.

Author: Holistic Diagnostic Platform Team
Version: 1.0.0
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging
from contextlib import asynccontextmanager

import jwt
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import numpy as np
import torch
import nibabel as nib
from PIL import Image
import io
from pathlib import Path
import json
import tempfile
import hashlib
import aiofiles
import redis.asyncio as redis

# Import our medical imaging components
from ..models.swin_unetr import SwinUNetR3D, SwinUNetRConfig
from ..models.vision_transformer import MedicalViT3D, MedicalViTConfig
from ..models.multi_modal_fusion import MultiModalFusionModel, MultiModalConfig
from ..models.advanced_architectures import (
    HybridCNNTransformer, AttentionUNet, create_hybrid_cnn_transformer,
    create_attention_unet, analyze_model_complexity
)
from ..core.security import SecurityManager, AuditLogger
from ..core.config import Config
from ..preprocessing.data_preprocessing import MedicalImagePreprocessor
from ..inference.inference_engine import InferenceEngine
from ..utils.metrics import calculate_dice_coefficient, calculate_hausdorff_distance
from ..federated.nvflare_integration import FederatedLearningOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
config = Config()
security_manager = SecurityManager()
audit_logger = AuditLogger()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security scheme
security = HTTPBearer()

# Global model cache
MODEL_CACHE = {}
REDIS_CLIENT = None

class ModelManager:
    """Manages model lifecycle, loading, and caching."""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = MedicalImagePreprocessor()
        self.inference_engine = None
        
    async def initialize(self):
        """Initialize models and inference engine."""
        try:
            # Initialize inference engine
            self.inference_engine = InferenceEngine()
            await self.inference_engine.initialize()
            
            logger.info("Model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise
            
    async def get_model(self, model_type: str, task: str = None):
        """Get or load a model by type."""
        model_key = f"{model_type}_{task}" if task else model_type
        
        if model_key not in self.models:
            await self._load_model(model_type, task)
            
        return self.models[model_key]
        
    async def _load_model(self, model_type: str, task: str = None):
        """Load a specific model."""
        model_key = f"{model_type}_{task}" if task else model_type
        
        try:
            if model_type == "swin_unetr":
                config_obj = SwinUNetRConfig()
                model = SwinUNetR3D(config_obj)
                
            elif model_type == "vit":
                config_obj = MedicalViTConfig()
                model = MedicalViT3D(config_obj)
                
            elif model_type == "multi_modal":
                config_obj = MultiModalConfig()
                model = MultiModalFusionModel(config_obj)
                
            elif model_type == "hybrid_cnn_transformer":
                model = create_hybrid_cnn_transformer()
                
            elif model_type == "attention_unet":
                model = create_attention_unet()
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            # Load pretrained weights if available
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f"{model_key}_best.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded pretrained weights for {model_key}")
                
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                
            self.models[model_key] = model
            logger.info(f"Model {model_key} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            raise

class AuthenticationManager:
    """Manages JWT authentication and authorization."""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        
    def create_access_token(self, data: dict):
        """Create a new access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
        
    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Security(security)):
        """Verify JWT token."""
        token = credentials.credentials
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
                
            return {"username": username, "permissions": payload.get("permissions", [])}
            
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Global managers
model_manager = ModelManager()
auth_manager = AuthenticationManager()

# Pydantic models for request/response
from pydantic import BaseModel, Field
from typing import Any

class ImageUpload(BaseModel):
    """Image upload metadata."""
    patient_id: Optional[str] = Field(None, description="De-identified patient ID")
    study_id: str = Field(..., description="Study identifier")
    modality: str = Field(..., description="Imaging modality (CT, MRI, etc.)")
    task: str = Field(..., description="Analysis task (segmentation, classification)")
    
class SegmentationRequest(BaseModel):
    """Segmentation analysis request."""
    model_type: str = Field("swin_unetr", description="Model type to use")
    roi_names: Optional[List[str]] = Field(None, description="ROI names for segmentation")
    confidence_threshold: float = Field(0.5, description="Confidence threshold")
    
class ClassificationRequest(BaseModel):
    """Classification analysis request."""
    model_type: str = Field("vit", description="Model type to use")
    num_classes: int = Field(2, description="Number of classes")
    return_attention: bool = Field(False, description="Return attention maps")
    
class AnalysisResponse(BaseModel):
    """Analysis result response."""
    analysis_id: str
    status: str
    results: Dict[str, Any]
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time: float
    model_info: Dict[str, str]
    
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    models_loaded: List[str]
    gpu_available: bool
    
# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Medical Diagnostic Platform API...")
    
    try:
        # Initialize Redis
        global REDIS_CLIENT
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        REDIS_CLIENT = redis.from_url(redis_url, decode_responses=True)
        await REDIS_CLIENT.ping()
        logger.info("Redis connection established")
        
        # Initialize model manager
        await model_manager.initialize()
        
        logger.info("Medical Diagnostic Platform API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
        
    yield
    
    # Shutdown
    logger.info("Shutting down Medical Diagnostic Platform API...")
    
    if REDIS_CLIENT:
        await REDIS_CLIENT.close()

# Create FastAPI application
app = FastAPI(
    title="Medical Diagnostic Platform API",
    description="Production-grade API for medical image analysis with HIPAA compliance",
    version="1.0.0",
    docs_url=None,  # Disable default docs for security
    redoc_url=None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.middleware("http")
async def audit_middleware(request, call_next):
    """Audit logging middleware."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Log request
    audit_logger.log_event(
        "api_request",
        {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    processing_time = time.time() - start_time
    audit_logger.log_event(
        "api_response",
        {
            "request_id": request_id,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return response

# Custom OpenAPI documentation endpoint with authentication
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(user_info = Depends(auth_manager.verify_token)):
    """Protected Swagger UI documentation."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Medical Diagnostic Platform API - Documentation"
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_endpoint(user_info = Depends(auth_manager.verify_token)):
    """Protected OpenAPI schema."""
    return get_openapi(
        title="Medical Diagnostic Platform API",
        version="1.0.0",
        description="Production-grade API for medical image analysis",
        routes=app.routes,
    )

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check."""
    try:
        # Check model availability
        models_loaded = list(model_manager.models.keys())
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            models_loaded=models_loaded,
            gpu_available=gpu_available
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Authentication endpoints
@app.post("/auth/token", tags=["Authentication"])
async def login(username: str, password: str):
    """Authenticate and get access token."""
    # In production, validate against secure user store
    if username == "demo" and password == "demo123":
        token_data = {
            "sub": username,
            "permissions": ["read", "write", "analyze"]
        }
        access_token = auth_manager.create_access_token(token_data)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": auth_manager.access_token_expire_minutes * 60
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Medical image analysis endpoints
@app.post("/analyze/segmentation", response_model=AnalysisResponse, tags=["Medical Analysis"])
@limiter.limit("10/minute")
async def analyze_segmentation(
    request_info: SegmentationRequest,
    image_file: UploadFile = File(...),
    user_info = Depends(auth_manager.verify_token)
):
    """Perform medical image segmentation analysis."""
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if not image_file.filename.lower().endswith(('.nii', '.nii.gz', '.dcm')):
            raise HTTPException(status_code=400, detail="Invalid file format")
            
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file:
            content = await image_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            
        try:
            # Load and preprocess image
            if temp_path.endswith('.nii.gz') or temp_path.endswith('.nii'):
                image_data = nib.load(temp_path).get_fdata()
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
                
            # Preprocess image
            preprocessed = model_manager.preprocessor.preprocess_image(
                image_data, normalize=True, resize_shape=(128, 128, 128)
            )
            
            # Get model
            model = await model_manager.get_model(request_info.model_type, "segmentation")
            
            # Run inference
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs = torch.from_numpy(preprocessed).float().cuda().unsqueeze(0).unsqueeze(0)
                else:
                    inputs = torch.from_numpy(preprocessed).float().unsqueeze(0).unsqueeze(0)
                    
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                    
                # Apply softmax for multi-class
                probabilities = torch.softmax(predictions, dim=1)
                segmentation = torch.argmax(probabilities, dim=1)
                
                # Convert to numpy
                seg_numpy = segmentation.squeeze().cpu().numpy()
                prob_numpy = probabilities.squeeze().cpu().numpy()
                
            # Calculate confidence scores
            confidence_scores = {}
            for i in range(prob_numpy.shape[0]):
                mask = seg_numpy == i
                if mask.any():
                    confidence_scores[f"class_{i}"] = float(prob_numpy[i][mask].mean())
                    
            # Calculate metrics if ground truth is available
            metrics = {}
            # In production, you might have ground truth for validation
            
            processing_time = time.time() - start_time
            
            # Cache results
            results = {
                "segmentation_shape": seg_numpy.shape,
                "num_classes": prob_numpy.shape[0],
                "volume_stats": {
                    "total_voxels": int(seg_numpy.size),
                    "segmented_voxels": int((seg_numpy > 0).sum())
                },
                "metrics": metrics
            }
            
            if REDIS_CLIENT:
                await REDIS_CLIENT.setex(
                    f"analysis:{analysis_id}",
                    3600,  # 1 hour TTL
                    json.dumps({
                        "results": results,
                        "confidence_scores": confidence_scores,
                        "segmentation": seg_numpy.tolist()  # Store for download
                    })
                )
                
            return AnalysisResponse(
                analysis_id=analysis_id,
                status="completed",
                results=results,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                model_info={
                    "model_type": request_info.model_type,
                    "task": "segmentation",
                    "version": "1.0.0"
                }
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Segmentation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/classification", response_model=AnalysisResponse, tags=["Medical Analysis"])
@limiter.limit("10/minute")
async def analyze_classification(
    request_info: ClassificationRequest,
    image_file: UploadFile = File(...),
    user_info = Depends(auth_manager.verify_token)
):
    """Perform medical image classification analysis."""
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if not image_file.filename.lower().endswith(('.nii', '.nii.gz', '.dcm')):
            raise HTTPException(status_code=400, detail="Invalid file format")
            
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file:
            content = await image_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            
        try:
            # Load and preprocess image
            if temp_path.endswith('.nii.gz') or temp_path.endswith('.nii'):
                image_data = nib.load(temp_path).get_fdata()
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
                
            # Preprocess image
            preprocessed = model_manager.preprocessor.preprocess_image(
                image_data, normalize=True, resize_shape=(128, 128, 128)
            )
            
            # Get model
            model = await model_manager.get_model(request_info.model_type, "classification")
            
            # Run inference
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs = torch.from_numpy(preprocessed).float().cuda().unsqueeze(0).unsqueeze(0)
                else:
                    inputs = torch.from_numpy(preprocessed).float().unsqueeze(0).unsqueeze(0)
                    
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    predictions, attention_maps = outputs
                else:
                    predictions = outputs
                    attention_maps = None
                    
                # Apply softmax
                probabilities = torch.softmax(predictions, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                
                # Convert to numpy
                prob_numpy = probabilities.squeeze().cpu().numpy()
                pred_class = predicted_class.item()
                
            # Calculate confidence scores
            confidence_scores = {f"class_{i}": float(prob_numpy[i]) for i in range(len(prob_numpy))}
            
            processing_time = time.time() - start_time
            
            # Prepare results
            results = {
                "predicted_class": pred_class,
                "probabilities": prob_numpy.tolist(),
                "confidence": float(prob_numpy.max()),
                "num_classes": len(prob_numpy)
            }
            
            # Add attention maps if requested
            if request_info.return_attention and attention_maps is not None:
                attention_numpy = attention_maps.squeeze().cpu().numpy()
                results["attention_shape"] = attention_numpy.shape
                
            # Cache results
            if REDIS_CLIENT:
                cache_data = {
                    "results": results,
                    "confidence_scores": confidence_scores
                }
                if request_info.return_attention and attention_maps is not None:
                    cache_data["attention_maps"] = attention_numpy.tolist()
                    
                await REDIS_CLIENT.setex(
                    f"analysis:{analysis_id}",
                    3600,  # 1 hour TTL
                    json.dumps(cache_data)
                )
                
            return AnalysisResponse(
                analysis_id=analysis_id,
                status="completed",
                results=results,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                model_info={
                    "model_type": request_info.model_type,
                    "task": "classification",
                    "version": "1.0.0"
                }
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Classification analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analysis/{analysis_id}/results", tags=["Medical Analysis"])
async def get_analysis_results(
    analysis_id: str,
    user_info = Depends(auth_manager.verify_token)
):
    """Retrieve cached analysis results."""
    try:
        if not REDIS_CLIENT:
            raise HTTPException(status_code=503, detail="Results cache unavailable")
            
        cached_data = await REDIS_CLIENT.get(f"analysis:{analysis_id}")
        
        if not cached_data:
            raise HTTPException(status_code=404, detail="Analysis results not found")
            
        return json.loads(cached_data)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid cached data")
    except Exception as e:
        logger.error(f"Failed to retrieve results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve results")

@app.get("/models/info", tags=["Models"])
async def get_models_info(user_info = Depends(auth_manager.verify_token)):
    """Get information about available models."""
    try:
        models_info = {}
        
        for model_key, model in model_manager.models.items():
            complexity = analyze_model_complexity(model)
            
            models_info[model_key] = {
                "loaded": True,
                "parameters": complexity["total_params"],
                "model_size_mb": complexity["model_size_mb"],
                "device": "cuda" if next(model.parameters()).is_cuda else "cpu"
            }
            
        return {
            "available_models": models_info,
            "total_models": len(models_info)
        }
        
    except Exception as e:
        logger.error(f"Failed to get models info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get models info")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    audit_logger.log_event(
        "api_error",
        {
            "url": str(request.url),
            "method": request.method,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    audit_logger.log_event(
        "api_internal_error",
        {
            "url": str(request.url),
            "method": request.method,
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )

# Main application entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )