"""
Configuration Management System

Advanced configuration system for environment-specific
settings, secrets management, and runtime configuration.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import secrets
import base64

from cryptography.fernet import Fernet
from pydantic import BaseSettings, Field, validator
import yaml

logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SecurityLevel(Enum):
    """Security configuration levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "medical_platform"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"
    connection_params: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class CacheConfig:
    """Cache configuration."""
    backend: str = "redis"
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl: bool = False
    max_connections: int = 10
    default_timeout: int = 300
    key_prefix: str = "medical_platform"

@dataclass
class StorageConfig:
    """Storage configuration."""
    provider: str = "local"  # local, s3, azure, gcp
    local_path: str = "./data"
    bucket_name: Optional[str] = None
    region: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    encryption_enabled: bool = True
    backup_enabled: bool = True
    retention_days: int = 2555  # 7 years for HIPAA

@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption_key: Optional[str] = None
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_hours: int = 24
    password_min_length: int = 12
    password_require_special: bool = True
    password_require_uppercase: bool = True
    password_require_numbers: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    rate_limit_requests: int = 1000
    rate_limit_window_minutes: int = 15
    cors_origins: List[str] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    """Model configuration."""
    model_storage_path: str = "./models"
    checkpoint_storage_path: str = "./checkpoints"
    cache_enabled: bool = True
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    compile_models: bool = False
    inference_batch_size: int = 1
    max_concurrent_inferences: int = 4
    model_warmup: bool = True
    quantization_enabled: bool = False

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    logging_level: LogLevel = LogLevel.INFO
    metrics_port: int = 9090
    health_check_interval: int = 30
    alert_webhook_url: Optional[str] = None
    log_file_path: str = "./logs/application.log"
    log_rotation_size: str = "100MB"
    log_retention_days: int = 30

class ConfigurationManager:
    """Manages application configuration across environments."""
    
    def __init__(self, environment: Optional[str] = None):
        """Initialize configuration manager."""
        self.environment = DeploymentEnvironment(environment or os.getenv("ENVIRONMENT", "development"))
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        self.secrets_dir = Path(__file__).parent.parent.parent / "secrets"
        
        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.secrets_dir.mkdir(exist_ok=True, mode=0o700)
        
        self._config_cache: Dict[str, Any] = {}
        self._secrets_cache: Dict[str, Any] = {}
        
        # Initialize encryption for secrets
        self._init_encryption()
        
        # Load configuration
        self._load_configuration()
    
    def _init_encryption(self):
        """Initialize encryption for secrets."""
        key_file = self.secrets_dir / "encryption.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)
        
        self.cipher = Fernet(key)
    
    def _load_configuration(self):
        """Load configuration from files."""
        # Load base configuration
        base_config_file = self.config_dir / "base.yaml"
        if base_config_file.exists():
            with open(base_config_file) as f:
                self._config_cache.update(yaml.safe_load(f) or {})
        
        # Load environment-specific configuration
        env_config_file = self.config_dir / f"{self.environment.value}.yaml"
        if env_config_file.exists():
            with open(env_config_file) as f:
                env_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config_cache, env_config)
        
        # Load secrets
        self._load_secrets()
        
        # Override with environment variables
        self._load_from_environment()
    
    def _load_secrets(self):
        """Load encrypted secrets."""
        secrets_file = self.secrets_dir / f"{self.environment.value}_secrets.enc"
        
        if secrets_file.exists():
            try:
                with open(secrets_file, 'rb') as f:
                    encrypted_data = f.read()
                
                decrypted_data = self.cipher.decrypt(encrypted_data)
                self._secrets_cache = json.loads(decrypted_data.decode())
                
            except Exception as e:
                logger.warning(f"Failed to load secrets: {e}")
                self._secrets_cache = {}
        else:
            self._secrets_cache = {}
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "DATABASE_HOST": ["database", "host"],
            "DATABASE_PORT": ["database", "port"],
            "DATABASE_NAME": ["database", "database"],
            "DATABASE_USER": ["database", "username"],
            "DATABASE_PASSWORD": ["database", "password"],
            "REDIS_HOST": ["cache", "host"],
            "REDIS_PORT": ["cache", "port"],
            "REDIS_PASSWORD": ["cache", "password"],
            "JWT_SECRET_KEY": ["security", "jwt_secret_key"],
            "ENCRYPTION_KEY": ["security", "encryption_key"],
            "LOG_LEVEL": ["monitoring", "logging_level"],
            "METRICS_PORT": ["monitoring", "metrics_port"]
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(self._config_cache, config_path, value)
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _set_nested_value(self, config: Dict, path: List[str], value: Any):
        """Set nested configuration value."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config = self._config_cache.get("database", {})
        
        # Add password from secrets if available
        if "database_password" in self._secrets_cache:
            db_config["password"] = self._secrets_cache["database_password"]
        
        return DatabaseConfig(**db_config)
    
    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration."""
        cache_config = self._config_cache.get("cache", {})
        
        # Add password from secrets if available
        if "redis_password" in self._secrets_cache:
            cache_config["password"] = self._secrets_cache["redis_password"]
        
        return CacheConfig(**cache_config)
    
    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        storage_config = self._config_cache.get("storage", {})
        
        # Add secrets
        secret_mappings = {
            "access_key_id": "aws_access_key_id",
            "secret_access_key": "aws_secret_access_key"
        }
        
        for config_key, secret_key in secret_mappings.items():
            if secret_key in self._secrets_cache:
                storage_config[config_key] = self._secrets_cache[secret_key]
        
        return StorageConfig(**storage_config)
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        security_config = self._config_cache.get("security", {})
        
        # Add secrets
        secret_mappings = {
            "encryption_key": "encryption_key",
            "jwt_secret_key": "jwt_secret_key"
        }
        
        for config_key, secret_key in secret_mappings.items():
            if secret_key in self._secrets_cache:
                security_config[config_key] = self._secrets_cache[secret_key]
        
        # Generate missing secrets
        if "jwt_secret_key" not in security_config or not security_config["jwt_secret_key"]:
            security_config["jwt_secret_key"] = self._generate_secret_key()
        
        if "encryption_key" not in security_config or not security_config["encryption_key"]:
            security_config["encryption_key"] = self._generate_secret_key()
        
        return SecurityConfig(**security_config)
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        model_config = self._config_cache.get("model", {})
        return ModelConfig(**model_config)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        monitoring_config = self._config_cache.get("monitoring", {})
        
        # Convert log level string to enum if needed
        if "logging_level" in monitoring_config and isinstance(monitoring_config["logging_level"], str):
            monitoring_config["logging_level"] = LogLevel(monitoring_config["logging_level"])
        
        return MonitoringConfig(**monitoring_config)
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path."""
        keys = path.split(".")
        current = self._config_cache
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set_config_value(self, path: str, value: Any):
        """Set configuration value by dot-separated path."""
        keys = path.split(".")
        self._set_nested_value(self._config_cache, keys, value)
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get secret value."""
        return self._secrets_cache.get(key)
    
    def set_secret(self, key: str, value: str):
        """Set secret value."""
        self._secrets_cache[key] = value
        self._save_secrets()
    
    def _save_secrets(self):
        """Save encrypted secrets to file."""
        secrets_file = self.secrets_dir / f"{self.environment.value}_secrets.enc"
        
        try:
            json_data = json.dumps(self._secrets_cache).encode()
            encrypted_data = self.cipher.encrypt(json_data)
            
            with open(secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            os.chmod(secrets_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
    
    def _generate_secret_key(self, length: int = 32) -> str:
        """Generate a random secret key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode()
    
    def save_configuration(self):
        """Save current configuration to file."""
        env_config_file = self.config_dir / f"{self.environment.value}.yaml"
        
        try:
            with open(env_config_file, 'w') as f:
                yaml.dump(self._config_cache, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {env_config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def reload_configuration(self):
        """Reload configuration from files."""
        self._config_cache.clear()
        self._secrets_cache.clear()
        self._load_configuration()
        logger.info("Configuration reloaded")
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any issues."""
        issues = []
        
        # Validate database configuration
        db_config = self.get_database_config()
        if not db_config.password and self.environment == DeploymentEnvironment.PRODUCTION:
            issues.append("Database password not set in production")
        
        # Validate security configuration
        security_config = self.get_security_config()
        if not security_config.jwt_secret_key:
            issues.append("JWT secret key not configured")
        
        if self.environment == DeploymentEnvironment.PRODUCTION:
            if len(security_config.cors_origins) == 0:
                issues.append("CORS origins not configured for production")
            
            if security_config.jwt_access_token_expire_minutes > 60:
                issues.append("JWT access token expiry too long for production")
        
        # Validate storage configuration
        storage_config = self.get_storage_config()
        if storage_config.provider != "local" and not storage_config.bucket_name:
            issues.append("Bucket name required for cloud storage")
        
        return issues
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about current environment."""
        return {
            "environment": self.environment.value,
            "config_dir": str(self.config_dir),
            "secrets_dir": str(self.secrets_dir),
            "config_keys": list(self._config_cache.keys()),
            "secrets_count": len(self._secrets_cache),
            "validation_issues": self.validate_configuration()
        }

# Global configuration manager instance
config_manager = ConfigurationManager()

# Convenience functions for common configurations
def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return config_manager.get_database_config()

def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return config_manager.get_security_config()

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return config_manager.get_monitoring_config()

def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return config_manager.get_model_config()

def get_storage_config() -> StorageConfig:
    """Get storage configuration."""
    return config_manager.get_storage_config()

def get_cache_config() -> CacheConfig:
    """Get cache configuration."""
    return config_manager.get_cache_config()