"""
Medical Platform Database Models

SQLAlchemy ORM models for persistent data storage including
users, studies, analyses, and audit logs.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class User(Base):
    """User account model with RBAC support."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), nullable=False, default="viewer")
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    
    # Relationships
    studies = relationship("Study", back_populates="user")
    analyses = relationship("Analysis", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

class Study(Base):
    """Medical study/case model."""
    
    __tablename__ = "studies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_instance_uid = Column(String(255), unique=True, nullable=False, index=True)
    patient_id = Column(String(255), nullable=False, index=True)  # De-identified
    study_date = Column(DateTime)
    study_time = Column(DateTime)
    modality = Column(String(50), nullable=False)
    study_description = Column(Text)
    institution_name = Column(String(255))
    referring_physician = Column(String(255))
    series_count = Column(Integer, default=0)
    instance_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="studies")
    analyses = relationship("Analysis", back_populates="study")
    series = relationship("Series", back_populates="study")

class Series(Base):
    """DICOM series model."""
    
    __tablename__ = "series"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    series_instance_uid = Column(String(255), unique=True, nullable=False, index=True)
    series_number = Column(Integer)
    series_description = Column(Text)
    modality = Column(String(50), nullable=False)
    body_part_examined = Column(String(255))
    series_date = Column(DateTime)
    series_time = Column(DateTime)
    protocol_name = Column(String(255))
    instance_count = Column(Integer, default=0)
    file_path = Column(Text)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Foreign keys
    study_id = Column(UUID(as_uuid=True), ForeignKey("studies.id"), nullable=False)
    
    # Relationships
    study = relationship("Study", back_populates="series")
    analyses = relationship("Analysis", back_populates="series")

class Analysis(Base):
    """Medical image analysis results model."""
    
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_type = Column(String(50), nullable=False)  # segmentation, classification, etc.
    model_type = Column(String(50), nullable=False)
    model_version = Column(String(50))
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    progress = Column(Float, default=0.0)
    
    # Analysis parameters
    parameters = Column(JSON)
    
    # Results
    results = Column(JSON)
    confidence_scores = Column(JSON)
    processing_time = Column(Float)
    
    # File paths
    input_file_path = Column(Text)
    output_file_path = Column(Text)
    visualization_path = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Error handling
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    study_id = Column(UUID(as_uuid=True), ForeignKey("studies.id"))
    series_id = Column(UUID(as_uuid=True), ForeignKey("series.id"))
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    study = relationship("Study", back_populates="analyses")
    series = relationship("Series", back_populates="analyses")

class AuditLog(Base):
    """Audit log for compliance and security tracking."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)  # security, access, analysis, etc.
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # User and session info
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    username = Column(String(50))
    session_id = Column(String(255))
    
    # Request info
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    endpoint = Column(String(255))
    http_method = Column(String(10))
    request_id = Column(String(255))
    
    # Event details
    resource_type = Column(String(50))
    resource_id = Column(String(255))
    action = Column(String(100))
    status = Column(String(20))
    
    # Additional data
    details = Column(JSON)
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")

class ModelMetadata(Base):
    """AI model metadata and versioning."""
    
    __tablename__ = "model_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    version = Column(String(50), nullable=False)
    framework = Column(String(50))  # pytorch, tensorflow, etc.
    
    # Model files
    model_path = Column(Text)
    config_path = Column(Text)
    weights_path = Column(Text)
    
    # Model info
    description = Column(Text)
    architecture = Column(JSON)
    hyperparameters = Column(JSON)
    training_data = Column(JSON)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    dice_score = Column(Float)
    
    # Model status
    is_active = Column(Boolean, default=False)
    is_approved = Column(Boolean, default=False)
    approval_date = Column(DateTime)
    approved_by = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SystemMetrics(Base):
    """System performance metrics for monitoring."""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metric_type = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    
    # Metric values
    value = Column(Float)
    unit = Column(String(20))
    
    # Context
    service = Column(String(50))
    instance = Column(String(100))
    environment = Column(String(20))
    
    # Additional data
    tags = Column(JSON)
    metadata = Column(JSON)

class HealthCheck(Base):
    """Health check results for system monitoring."""
    
    __tablename__ = "health_checks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    service_name = Column(String(100), nullable=False, index=True)
    status = Column(String(20), nullable=False, index=True)  # healthy, degraded, unhealthy
    response_time = Column(Float)
    
    # Details
    details = Column(JSON)
    error_message = Column(Text)
    
    # Environment
    environment = Column(String(20))
    instance = Column(String(100))