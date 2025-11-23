"""
Medical Platform Environment Configuration

Environment-specific configuration for different deployment scenarios
including development, staging, and production environments.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional

class Environment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class EnvironmentConfig:
    """Environment-specific configuration manager."""
    
    def __init__(self, env: Environment = None):
        self.env = env or self._detect_environment()
        self._config = self._load_environment_config()
    
    def _detect_environment(self) -> Environment:
        """Auto-detect current environment."""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        
        if env_name in ["prod", "production"]:
            return Environment.PRODUCTION
        elif env_name in ["stage", "staging"]:
            return Environment.STAGING
        else:
            return Environment.DEVELOPMENT
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        base_config = self._get_base_config()
        env_config = self._get_environment_specific_config()
        
        # Merge configurations with environment-specific overrides
        base_config.update(env_config)
        return base_config
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration shared across environments."""
        return {
            # Application
            "app_name": "Medical Diagnostic Platform",
            "app_version": "1.0.0",
            "api_prefix": "/api/v1",
            
            # Security
            "jwt_algorithm": "HS256",
            "password_min_length": 8,
            "max_login_attempts": 5,
            "session_timeout": 3600,
            
            # Model
            "model_cache_size": 1000,
            "max_image_size": 512,
            "inference_timeout": 300,
            
            # Monitoring
            "metrics_enabled": True,
            "health_check_interval": 30,
            "log_level": "INFO",
        }
    
    def _get_environment_specific_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides."""
        if self.env == Environment.DEVELOPMENT:
            return self._get_development_config()
        elif self.env == Environment.STAGING:
            return self._get_staging_config()
        elif self.env == Environment.PRODUCTION:
            return self._get_production_config()
        else:
            return {}
    
    def _get_development_config(self) -> Dict[str, Any]:
        """Development environment configuration."""
        return {
            # Database
            "database_url": os.getenv("DATABASE_URL", "sqlite:///./medical_dev.db"),
            "database_pool_size": 5,
            "database_echo": True,
            
            # Redis
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
            "redis_max_connections": 10,
            
            # Security
            "jwt_secret_key": os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-me"),
            "jwt_expire_minutes": 60,  # Longer for development
            "cors_origins": ["http://localhost:3000", "http://localhost:8080"],
            
            # API
            "api_host": "localhost",
            "api_port": 8000,
            "workers": 1,
            "reload": True,
            
            # Rate Limiting (more permissive)
            "rate_limit_requests": 1000,
            "rate_limit_window": 60,
            
            # Logging
            "log_level": "DEBUG",
            "log_file": "logs/medical_dev.log",
            
            # Features
            "enable_swagger": True,
            "enable_metrics": True,
            "enable_profiling": True,
        }
    
    def _get_staging_config(self) -> Dict[str, Any]:
        """Staging environment configuration."""
        return {
            # Database
            "database_url": os.getenv("DATABASE_URL", "postgresql://staging_user:staging_pass@postgres:5432/medical_staging"),
            "database_pool_size": 10,
            "database_echo": False,
            
            # Redis
            "redis_url": os.getenv("REDIS_URL", "redis://redis:6379"),
            "redis_max_connections": 20,
            
            # Security
            "jwt_secret_key": os.getenv("JWT_SECRET_KEY"),
            "jwt_expire_minutes": 30,
            "cors_origins": [
                "https://staging.medical-platform.com",
                "https://staging-app.medical-platform.com"
            ],
            
            # API
            "api_host": "0.0.0.0",
            "api_port": 8000,
            "workers": 2,
            "reload": False,
            
            # Rate Limiting
            "rate_limit_requests": 100,
            "rate_limit_window": 60,
            
            # Logging
            "log_level": "INFO",
            "log_file": "logs/medical_staging.log",
            
            # Features
            "enable_swagger": True,
            "enable_metrics": True,
            "enable_profiling": False,
            
            # Monitoring
            "prometheus_enabled": True,
            "grafana_enabled": True,
        }
    
    def _get_production_config(self) -> Dict[str, Any]:
        """Production environment configuration."""
        return {
            # Database
            "database_url": os.getenv("DATABASE_URL"),
            "database_pool_size": 20,
            "database_echo": False,
            "database_ssl_mode": "require",
            
            # Redis
            "redis_url": os.getenv("REDIS_URL"),
            "redis_max_connections": 50,
            "redis_ssl": True,
            
            # Security
            "jwt_secret_key": os.getenv("JWT_SECRET_KEY"),
            "jwt_expire_minutes": 15,  # Shorter for security
            "cors_origins": [
                "https://medical-platform.com",
                "https://app.medical-platform.com"
            ],
            
            # API
            "api_host": "0.0.0.0",
            "api_port": 8000,
            "workers": 4,
            "reload": False,
            
            # Rate Limiting (strict)
            "rate_limit_requests": 60,
            "rate_limit_window": 60,
            
            # Logging
            "log_level": "WARNING",
            "log_file": "logs/medical_production.log",
            "log_format": "json",
            
            # Features
            "enable_swagger": False,  # Disabled in production
            "enable_metrics": True,
            "enable_profiling": False,
            
            # Monitoring
            "prometheus_enabled": True,
            "grafana_enabled": True,
            "alerting_enabled": True,
            
            # Security
            "enable_security_headers": True,
            "enable_csrf_protection": True,
            "force_https": True,
            
            # Performance
            "enable_compression": True,
            "enable_caching": True,
            "cache_ttl": 300,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": self.get("database_url"),
            "pool_size": self.get("database_pool_size", 10),
            "echo": self.get("database_echo", False),
            "ssl_mode": self.get("database_ssl_mode", "prefer")
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "url": self.get("redis_url"),
            "max_connections": self.get("redis_max_connections", 20),
            "ssl": self.get("redis_ssl", False)
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "jwt_secret_key": self.get("jwt_secret_key"),
            "jwt_algorithm": self.get("jwt_algorithm", "HS256"),
            "jwt_expire_minutes": self.get("jwt_expire_minutes", 30),
            "cors_origins": self.get("cors_origins", []),
            "max_login_attempts": self.get("max_login_attempts", 5)
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return {
            "host": self.get("api_host", "0.0.0.0"),
            "port": self.get("api_port", 8000),
            "workers": self.get("workers", 2),
            "reload": self.get("reload", False)
        }
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env == Environment.DEVELOPMENT
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.env == Environment.STAGING
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == Environment.PRODUCTION

# Global environment configuration instance
env_config = EnvironmentConfig()