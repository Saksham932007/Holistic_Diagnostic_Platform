"""
Medical Diagnostic Platform API Documentation Generator

Generates comprehensive OpenAPI documentation with medical workflow examples,
security specifications, and integration guides.

Author: Holistic Diagnostic Platform Team
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class APIEndpoint:
    """API endpoint documentation."""
    path: str
    method: str
    summary: str
    description: str
    tags: List[str]
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    security: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]

def get_openapi_spec() -> Dict[str, Any]:
    """Generate complete OpenAPI 3.0 specification."""
    
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Medical Diagnostic Platform API",
            "description": """
# Medical Diagnostic Platform API

A comprehensive REST API for medical image analysis with HIPAA compliance and production-grade security.

## Features

- **Medical Image Analysis**: Segmentation and classification of medical images
- **Multi-Modal Support**: CT, MRI, PET, and other medical imaging modalities
- **Advanced AI Models**: Swin-UNetR, Vision Transformers, and hybrid architectures
- **HIPAA Compliance**: Full compliance with healthcare data protection regulations
- **Security**: JWT authentication, rate limiting, and comprehensive audit logging
- **Monitoring**: Prometheus metrics and health checks for production deployment

## Authentication

All API endpoints require authentication using JWT Bearer tokens. To obtain a token:

1. Call the `/auth/token` endpoint with valid credentials
2. Include the token in the `Authorization` header as `Bearer <token>`
3. Tokens expire after 30 minutes and must be refreshed

## Rate Limiting

API calls are rate-limited to prevent abuse:

- Authentication: 5 requests per 5 minutes
- Analysis endpoints: 10 requests per minute
- General endpoints: 60 requests per minute

## Medical Workflow Example

```python
import requests

# 1. Authenticate
auth_response = requests.post("https://api.medical-platform.com/auth/token", {
    "username": "doctor@hospital.com",
    "password": "secure_password"
})
token = auth_response.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# 2. Upload and analyze medical image
with open("brain_mri.nii.gz", "rb") as f:
    files = {"image_file": f}
    data = {
        "model_type": "swin_unetr",
        "roi_names": ["tumor", "edema"],
        "confidence_threshold": 0.7
    }
    
    response = requests.post(
        "https://api.medical-platform.com/analyze/segmentation",
        files=files,
        data=data,
        headers=headers
    )

# 3. Get results
analysis_id = response.json()["analysis_id"]
results = requests.get(
    f"https://api.medical-platform.com/analysis/{analysis_id}/results",
    headers=headers
).json()
```
            """,
            "version": "1.0.0",
            "contact": {
                "name": "Medical Platform Team",
                "email": "support@medical-platform.com",
                "url": "https://medical-platform.com/support"
            },
            "license": {
                "name": "Medical Platform License",
                "url": "https://medical-platform.com/license"
            }
        },
        "servers": [
            {
                "url": "https://api.medical-platform.com",
                "description": "Production server"
            },
            {
                "url": "https://staging.api.medical-platform.com",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ],
        "tags": [
            {
                "name": "System",
                "description": "System health and monitoring endpoints"
            },
            {
                "name": "Authentication",
                "description": "User authentication and authorization"
            },
            {
                "name": "Medical Analysis",
                "description": "Medical image analysis endpoints"
            },
            {
                "name": "Models",
                "description": "AI model information and management"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "tags": ["System"],
                    "summary": "System health check",
                    "description": "Check the health status of all system components",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "System is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"},
                                    "examples": {
                                        "healthy": {
                                            "value": {
                                                "status": "healthy",
                                                "timestamp": "2025-01-15T10:30:00Z",
                                                "version": "1.0.0",
                                                "models_loaded": ["swin_unetr", "vit", "multi_modal"],
                                                "gpu_available": True
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "503": {
                            "description": "Service unavailable",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/auth/token": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "Authenticate user and get access token",
                    "description": "Authenticate with username and password to receive JWT access token",
                    "operationId": "authenticate",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/x-www-form-urlencoded": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "username": {
                                            "type": "string",
                                            "description": "User's username or email"
                                        },
                                        "password": {
                                            "type": "string",
                                            "description": "User's password"
                                        }
                                    },
                                    "required": ["username", "password"]
                                },
                                "examples": {
                                    "doctor": {
                                        "value": {
                                            "username": "doctor@hospital.com",
                                            "password": "secure_password123"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Authentication successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TokenResponse"},
                                    "examples": {
                                        "success": {
                                            "value": {
                                                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                                                "token_type": "bearer",
                                                "expires_in": 1800
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "401": {
                            "description": "Invalid credentials",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/analyze/segmentation": {
                "post": {
                    "tags": ["Medical Analysis"],
                    "summary": "Analyze medical image for segmentation",
                    "description": """
Perform medical image segmentation analysis using advanced AI models.

Supports multiple medical imaging modalities including:
- CT scans
- MRI (T1, T2, FLAIR, etc.)
- PET scans
- Ultrasound
- X-ray

The analysis returns detailed segmentation masks with confidence scores
for each anatomical structure or pathological region.
                    """,
                    "operationId": "analyzeSegmentation",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "image_file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Medical image file (NIfTI, DICOM)"
                                        },
                                        "model_type": {
                                            "type": "string",
                                            "enum": ["swin_unetr", "attention_unet", "hybrid_cnn_transformer"],
                                            "default": "swin_unetr",
                                            "description": "AI model type for segmentation"
                                        },
                                        "roi_names": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Regions of interest to segment"
                                        },
                                        "confidence_threshold": {
                                            "type": "number",
                                            "minimum": 0.1,
                                            "maximum": 0.95,
                                            "default": 0.5,
                                            "description": "Minimum confidence threshold"
                                        }
                                    },
                                    "required": ["image_file"]
                                },
                                "examples": {
                                    "brain_tumor_mri": {
                                        "summary": "Brain tumor segmentation in MRI",
                                        "value": {
                                            "model_type": "swin_unetr",
                                            "roi_names": ["tumor", "edema", "necrosis"],
                                            "confidence_threshold": 0.7
                                        }
                                    },
                                    "liver_ct": {
                                        "summary": "Liver segmentation in CT",
                                        "value": {
                                            "model_type": "attention_unet",
                                            "roi_names": ["liver", "lesion"],
                                            "confidence_threshold": 0.6
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Analysis completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AnalysisResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid input",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        },
                        "429": {
                            "description": "Rate limit exceeded",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/analyze/classification": {
                "post": {
                    "tags": ["Medical Analysis"],
                    "summary": "Analyze medical image for classification",
                    "description": """
Perform medical image classification analysis using Vision Transformers and other advanced models.

Common classification tasks:
- Disease detection (cancer, fractures, etc.)
- Anatomical structure identification
- Image quality assessment
- Modality classification

Returns classification probabilities and optional attention visualizations.
                    """,
                    "operationId": "analyzeClassification",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "image_file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "Medical image file"
                                        },
                                        "model_type": {
                                            "type": "string",
                                            "enum": ["vit", "hybrid_cnn_transformer"],
                                            "default": "vit",
                                            "description": "AI model type for classification"
                                        },
                                        "num_classes": {
                                            "type": "integer",
                                            "minimum": 2,
                                            "maximum": 100,
                                            "default": 2,
                                            "description": "Number of classification classes"
                                        },
                                        "return_attention": {
                                            "type": "boolean",
                                            "default": False,
                                            "description": "Return attention maps for visualization"
                                        }
                                    },
                                    "required": ["image_file"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Classification completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AnalysisResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/models/info": {
                "get": {
                    "tags": ["Models"],
                    "summary": "Get information about available AI models",
                    "description": "Retrieve details about loaded AI models including parameters and capabilities",
                    "operationId": "getModelsInfo",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Model information retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "available_models": {
                                                "type": "object",
                                                "additionalProperties": {
                                                    "type": "object",
                                                    "properties": {
                                                        "loaded": {"type": "boolean"},
                                                        "parameters": {"type": "integer"},
                                                        "model_size_mb": {"type": "number"},
                                                        "device": {"type": "string"}
                                                    }
                                                }
                                            },
                                            "total_models": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "version": {"type": "string"},
                        "models_loaded": {"type": "array", "items": {"type": "string"}},
                        "gpu_available": {"type": "boolean"}
                    },
                    "required": ["status", "timestamp", "version"]
                },
                "TokenResponse": {
                    "type": "object",
                    "properties": {
                        "access_token": {"type": "string"},
                        "token_type": {"type": "string"},
                        "expires_in": {"type": "integer"}
                    },
                    "required": ["access_token", "token_type", "expires_in"]
                },
                "AnalysisResponse": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {"type": "string", "format": "uuid"},
                        "status": {"type": "string", "enum": ["completed", "failed", "processing"]},
                        "results": {"type": "object"},
                        "confidence_scores": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        },
                        "processing_time": {"type": "number"},
                        "model_info": {
                            "type": "object",
                            "properties": {
                                "model_type": {"type": "string"},
                                "task": {"type": "string"},
                                "version": {"type": "string"}
                            }
                        }
                    },
                    "required": ["analysis_id", "status", "results", "processing_time", "model_info"]
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "path": {"type": "string"}
                    },
                    "required": ["error", "timestamp"]
                }
            },
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT access token obtained from /auth/token endpoint"
                }
            }
        },
        "security": [
            {"BearerAuth": []}
        ]
    }

def generate_medical_examples() -> Dict[str, Any]:
    """Generate medical workflow examples."""
    
    return {
        "brain_tumor_analysis": {
            "title": "Brain Tumor Analysis Workflow",
            "description": "Complete workflow for brain tumor segmentation and classification",
            "steps": [
                {
                    "step": 1,
                    "action": "authenticate",
                    "endpoint": "/auth/token",
                    "example": {
                        "username": "neuroradiologist@hospital.com",
                        "password": "secure_pass123"
                    }
                },
                {
                    "step": 2,
                    "action": "upload_and_segment",
                    "endpoint": "/analyze/segmentation",
                    "example": {
                        "model_type": "swin_unetr",
                        "roi_names": ["tumor", "edema", "necrosis"],
                        "confidence_threshold": 0.75
                    }
                },
                {
                    "step": 3,
                    "action": "classify_tumor_type",
                    "endpoint": "/analyze/classification",
                    "example": {
                        "model_type": "vit",
                        "num_classes": 4,
                        "return_attention": True
                    }
                }
            ]
        },
        "lung_screening": {
            "title": "Lung Cancer Screening",
            "description": "Automated lung nodule detection and classification",
            "steps": [
                {
                    "step": 1,
                    "action": "authenticate",
                    "endpoint": "/auth/token"
                },
                {
                    "step": 2,
                    "action": "detect_nodules",
                    "endpoint": "/analyze/segmentation",
                    "example": {
                        "model_type": "attention_unet",
                        "roi_names": ["nodule", "mass"],
                        "confidence_threshold": 0.8
                    }
                },
                {
                    "step": 3,
                    "action": "classify_malignancy",
                    "endpoint": "/analyze/classification",
                    "example": {
                        "model_type": "hybrid_cnn_transformer",
                        "num_classes": 2,
                        "return_attention": False
                    }
                }
            ]
        }
    }

def generate_postman_collection() -> Dict[str, Any]:
    """Generate Postman collection for API testing."""
    
    return {
        "info": {
            "name": "Medical Diagnostic Platform API",
            "description": "Complete API collection for medical image analysis platform",
            "version": "1.0.0",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {
                    "key": "token",
                    "value": "{{access_token}}",
                    "type": "string"
                }
            ]
        },
        "variable": [
            {
                "key": "base_url",
                "value": "https://api.medical-platform.com",
                "type": "string"
            },
            {
                "key": "access_token",
                "value": "",
                "type": "string"
            }
        ],
        "item": [
            {
                "name": "Authentication",
                "item": [
                    {
                        "name": "Get Access Token",
                        "request": {
                            "method": "POST",
                            "header": [
                                {
                                    "key": "Content-Type",
                                    "value": "application/x-www-form-urlencoded"
                                }
                            ],
                            "body": {
                                "mode": "urlencoded",
                                "urlencoded": [
                                    {
                                        "key": "username",
                                        "value": "demo",
                                        "type": "text"
                                    },
                                    {
                                        "key": "password",
                                        "value": "demo123",
                                        "type": "text"
                                    }
                                ]
                            },
                            "url": {
                                "raw": "{{base_url}}/auth/token",
                                "host": ["{{base_url}}"],
                                "path": ["auth", "token"]
                            }
                        },
                        "event": [
                            {
                                "listen": "test",
                                "script": {
                                    "exec": [
                                        "if (pm.response.code === 200) {",
                                        "    var jsonData = pm.response.json();",
                                        "    pm.collectionVariables.set('access_token', jsonData.access_token);",
                                        "}"
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
            {
                "name": "System",
                "item": [
                    {
                        "name": "Health Check",
                        "request": {
                            "method": "GET",
                            "url": {
                                "raw": "{{base_url}}/health",
                                "host": ["{{base_url}}"],
                                "path": ["health"]
                            }
                        }
                    }
                ]
            },
            {
                "name": "Medical Analysis",
                "item": [
                    {
                        "name": "Segmentation Analysis",
                        "request": {
                            "method": "POST",
                            "header": [
                                {
                                    "key": "Authorization",
                                    "value": "Bearer {{access_token}}"
                                }
                            ],
                            "body": {
                                "mode": "formdata",
                                "formdata": [
                                    {
                                        "key": "image_file",
                                        "type": "file",
                                        "src": []
                                    },
                                    {
                                        "key": "model_type",
                                        "value": "swin_unetr",
                                        "type": "text"
                                    },
                                    {
                                        "key": "confidence_threshold",
                                        "value": "0.7",
                                        "type": "text"
                                    }
                                ]
                            },
                            "url": {
                                "raw": "{{base_url}}/analyze/segmentation",
                                "host": ["{{base_url}}"],
                                "path": ["analyze", "segmentation"]
                            }
                        }
                    }
                ]
            }
        ]
    }