"""
Configuration management for the Holistic Diagnostic Platform.

This module provides HIPAA/FDA-compliant configuration management using
Pydantic BaseSettings for type-safe configuration handling.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic.types import SecretStr
import os


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="medical_imaging_db", description="Database name")
    username: str = Field(description="Database username")
    password: SecretStr = Field(description="Database password")
    ssl_mode: str = Field(default="require", description="SSL mode for database connection")
    
    class Config:
        env_prefix = "DB_"


class SecurityConfig(BaseSettings):
    """Security and encryption configuration."""
    
    encryption_key: SecretStr = Field(description="AES-256 encryption key for data at rest")
    jwt_secret: SecretStr = Field(description="JWT signing secret")
    password_salt: SecretStr = Field(description="Password hashing salt")
    audit_retention_days: int = Field(default=2555, description="Audit log retention (7 years)")
    
    class Config:
        env_prefix = "SECURITY_"


class ModelConfig(BaseSettings):
    """Model training and inference configuration."""
    
    # Swin-UNetR Configuration
    swin_img_size: tuple = Field(default=(96, 96, 96), description="Input image size for Swin-UNetR")
    swin_in_channels: int = Field(default=1, description="Input channels for medical imaging")
    swin_out_channels: int = Field(default=14, description="Output segmentation classes")
    swin_feature_size: int = Field(default=48, description="Feature size for Swin-UNetR")
    swin_patch_size: tuple = Field(default=(2, 2, 2), description="Patch size for Swin transformer")
    
    # ViT Configuration
    vit_img_size: int = Field(default=224, description="Input image size for ViT")
    vit_patch_size: int = Field(default=16, description="Patch size for ViT")
    vit_num_classes: int = Field(default=1000, description="Number of classification classes")
    vit_embed_dim: int = Field(default=768, description="Embedding dimension")
    vit_num_heads: int = Field(default=12, description="Number of attention heads")
    vit_num_layers: int = Field(default=12, description="Number of transformer layers")
    
    # Training Configuration
    batch_size: int = Field(default=2, description="Training batch size")
    learning_rate: float = Field(default=1e-4, description="Initial learning rate")
    max_epochs: int = Field(default=1000, description="Maximum training epochs")
    val_interval: int = Field(default=2, description="Validation interval")
    
    # Mixed Precision Training
    use_amp: bool = Field(default=True, description="Use Automatic Mixed Precision")
    
    class Config:
        env_prefix = "MODEL_"


class DataConfig(BaseSettings):
    """Data processing and storage configuration."""
    
    raw_data_path: Path = Field(description="Path to raw DICOM data")
    processed_data_path: Path = Field(description="Path to processed data")
    cache_path: Path = Field(description="Path for caching processed data")
    
    # DICOM Processing
    allowed_modalities: List[str] = Field(
        default=["CT", "MR", "PT", "US"],
        description="Allowed DICOM modalities"
    )
    min_slice_thickness: float = Field(default=1.0, description="Minimum slice thickness (mm)")
    max_slice_thickness: float = Field(default=10.0, description="Maximum slice thickness (mm)")
    
    # Data Validation
    validate_dicom_tags: bool = Field(default=True, description="Validate critical DICOM tags")
    required_tags: List[str] = Field(
        default=["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "Modality"],
        description="Required DICOM tags for processing"
    )
    
    @validator('raw_data_path', 'processed_data_path', 'cache_path')
    def validate_paths(cls, v: Path) -> Path:
        """Validate that paths exist or can be created."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "DATA_"


class FederatedConfig(BaseSettings):
    """Federated learning configuration."""
    
    server_host: str = Field(default="localhost", description="NVFlare server host")
    server_port: int = Field(default=8002, description="NVFlare server port")
    client_id: str = Field(description="Unique client identifier")
    rounds: int = Field(default=100, description="Number of federated rounds")
    local_epochs: int = Field(default=1, description="Local training epochs per round")
    
    class Config:
        env_prefix = "FL_"


class APIConfig(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    workers: int = Field(default=1, description="Number of worker processes")
    max_request_size: int = Field(default=100 * 1024 * 1024, description="Max request size (100MB)")
    timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # CORS settings for medical applications
    allowed_origins: List[str] = Field(
        default=["https://hospital-domain.com"],
        description="Allowed CORS origins"
    )
    
    class Config:
        env_prefix = "API_"


class ComplianceConfig(BaseSettings):
    """HIPAA and FDA compliance configuration."""
    
    # HIPAA Requirements
    enable_audit_logging: bool = Field(default=True, description="Enable comprehensive audit logging")
    log_phi_access: bool = Field(default=True, description="Log PHI access events")
    data_retention_days: int = Field(default=2555, description="Data retention period (7 years)")
    
    # FDA 21 CFR Part 11 Requirements
    enable_digital_signatures: bool = Field(default=True, description="Enable digital signatures")
    audit_trail_integrity: bool = Field(default=True, description="Ensure audit trail integrity")
    user_authentication: bool = Field(default=True, description="Require user authentication")
    
    # De-identification
    remove_phi_tags: List[str] = Field(
        default=[
            "PatientName", "PatientID", "PatientBirthDate", "InstitutionName",
            "InstitutionAddress", "ReferringPhysicianName", "StationName",
            "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
            "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime"
        ],
        description="DICOM tags to remove for de-identification"
    )
    
    class Config:
        env_prefix = "COMPLIANCE_"


class PlatformConfig(BaseSettings):
    """Main platform configuration combining all sub-configurations."""
    
    # Environment
    environment: str = Field(default="development", description="Deployment environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    src_path: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    
    @validator('environment')
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v: str) -> str:
        """Validate log level setting."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
config = PlatformConfig()


def get_config() -> PlatformConfig:
    """
    Get the global configuration instance.
    
    Returns:
        PlatformConfig: The configuration instance.
    """
    return config


def reload_config() -> PlatformConfig:
    """
    Reload configuration from environment variables.
    
    Returns:
        PlatformConfig: The reloaded configuration instance.
    """
    global config
    config = PlatformConfig()
    return config