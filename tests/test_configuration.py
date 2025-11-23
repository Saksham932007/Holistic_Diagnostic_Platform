"""
Unit tests for configuration management.

These tests verify that the Pydantic-based configuration system properly
loads, validates, and manages settings for the medical imaging platform.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.config import (
    PlatformConfig,
    DatabaseConfig,
    SecurityConfig,
    ModelConfig,
    DataConfig,
    FederatedConfig,
    APIConfig,
    ComplianceConfig,
    get_config,
    reload_config
)


class TestDatabaseConfig:
    """Test suite for database configuration."""
    
    def test_default_values(self):
        """Test that default values are properly set."""
        config = DatabaseConfig()
        
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "medical_imaging_db"
        assert config.ssl_mode == "require"
    
    def test_environment_variables(self):
        """Test loading from environment variables."""
        with patch.dict(os.environ, {
            'DB_HOST': 'test-db.example.com',
            'DB_PORT': '3306',
            'DB_NAME': 'test_db',
            'DB_USERNAME': 'test_user',
            'DB_PASSWORD': 'test_password',
            'DB_SSL_MODE': 'prefer'
        }):
            config = DatabaseConfig()
            
            assert config.host == "test-db.example.com"
            assert config.port == 3306
            assert config.name == "test_db"
            assert config.username == "test_user"
            assert config.password.get_secret_value() == "test_password"
            assert config.ssl_mode == "prefer"
    
    def test_secret_handling(self):
        """Test that sensitive fields are handled as secrets."""
        with patch.dict(os.environ, {
            'DB_USERNAME': 'sensitive_user',
            'DB_PASSWORD': 'sensitive_password'
        }):
            config = DatabaseConfig()
            
            # Password should be a SecretStr
            assert str(config.password) == "**********"
            assert config.password.get_secret_value() == "sensitive_password"


class TestSecurityConfig:
    """Test suite for security configuration."""
    
    def test_required_fields(self):
        """Test that required security fields must be provided."""
        # Should raise validation error if required fields are missing
        with pytest.raises(Exception):  # Pydantic ValidationError
            SecurityConfig()
    
    def test_environment_loading(self):
        """Test loading security config from environment."""
        with patch.dict(os.environ, {
            'SECURITY_ENCRYPTION_KEY': 'test_encryption_key_32_bytes_long',
            'SECURITY_JWT_SECRET': 'test_jwt_secret',
            'SECURITY_PASSWORD_SALT': 'test_password_salt',
            'SECURITY_AUDIT_RETENTION_DAYS': '1000'
        }):
            config = SecurityConfig()
            
            assert config.encryption_key.get_secret_value() == "test_encryption_key_32_bytes_long"
            assert config.jwt_secret.get_secret_value() == "test_jwt_secret"
            assert config.password_salt.get_secret_value() == "test_password_salt"
            assert config.audit_retention_days == 1000
    
    def test_default_audit_retention(self):
        """Test default audit retention period."""
        with patch.dict(os.environ, {
            'SECURITY_ENCRYPTION_KEY': 'test_key',
            'SECURITY_JWT_SECRET': 'test_secret',
            'SECURITY_PASSWORD_SALT': 'test_salt'
        }):
            config = SecurityConfig()
            assert config.audit_retention_days == 2555  # 7 years


class TestModelConfig:
    """Test suite for model configuration."""
    
    def test_default_model_params(self):
        """Test default model parameters."""
        config = ModelConfig()
        
        # Swin-UNetR defaults
        assert config.swin_img_size == (96, 96, 96)
        assert config.swin_in_channels == 1
        assert config.swin_out_channels == 14
        assert config.swin_feature_size == 48
        assert config.swin_patch_size == (2, 2, 2)
        
        # ViT defaults
        assert config.vit_img_size == 224
        assert config.vit_patch_size == 16
        assert config.vit_num_classes == 1000
        assert config.vit_embed_dim == 768
        
        # Training defaults
        assert config.batch_size == 2
        assert config.learning_rate == 1e-4
        assert config.max_epochs == 1000
        assert config.use_amp == True
    
    def test_environment_override(self):
        """Test overriding model config via environment."""
        with patch.dict(os.environ, {
            'MODEL_BATCH_SIZE': '8',
            'MODEL_LEARNING_RATE': '0.001',
            'MODEL_MAX_EPOCHS': '500',
            'MODEL_USE_AMP': 'false'
        }):
            config = ModelConfig()
            
            assert config.batch_size == 8
            assert config.learning_rate == 0.001
            assert config.max_epochs == 500
            assert config.use_amp == False


class TestDataConfig:
    """Test suite for data configuration."""
    
    def test_path_validation_and_creation(self):
        """Test that paths are validated and created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with patch.dict(os.environ, {
                'DATA_RAW_DATA_PATH': str(temp_path / 'raw'),
                'DATA_PROCESSED_DATA_PATH': str(temp_path / 'processed'),
                'DATA_CACHE_PATH': str(temp_path / 'cache')
            }):
                config = DataConfig()
                
                # Paths should exist after validation
                assert config.raw_data_path.exists()
                assert config.processed_data_path.exists()
                assert config.cache_path.exists()
    
    def test_default_modalities(self):
        """Test default allowed modalities."""
        with patch.dict(os.environ, {
            'DATA_RAW_DATA_PATH': '/tmp/raw',
            'DATA_PROCESSED_DATA_PATH': '/tmp/processed',
            'DATA_CACHE_PATH': '/tmp/cache'
        }):
            config = DataConfig()
            
            expected_modalities = ["CT", "MR", "PT", "US"]
            assert config.allowed_modalities == expected_modalities
    
    def test_slice_thickness_validation(self):
        """Test slice thickness validation ranges."""
        with patch.dict(os.environ, {
            'DATA_RAW_DATA_PATH': '/tmp/raw',
            'DATA_PROCESSED_DATA_PATH': '/tmp/processed', 
            'DATA_CACHE_PATH': '/tmp/cache',
            'DATA_MIN_SLICE_THICKNESS': '0.5',
            'DATA_MAX_SLICE_THICKNESS': '15.0'
        }):
            config = DataConfig()
            
            assert config.min_slice_thickness == 0.5
            assert config.max_slice_thickness == 15.0
    
    def test_required_dicom_tags(self):
        """Test default required DICOM tags."""
        with patch.dict(os.environ, {
            'DATA_RAW_DATA_PATH': '/tmp/raw',
            'DATA_PROCESSED_DATA_PATH': '/tmp/processed',
            'DATA_CACHE_PATH': '/tmp/cache'
        }):
            config = DataConfig()
            
            required_tags = [
                "StudyInstanceUID", "SeriesInstanceUID", 
                "SOPInstanceUID", "Modality"
            ]
            assert config.required_tags == required_tags


class TestComplianceConfig:
    """Test suite for compliance configuration."""
    
    def test_hipaa_defaults(self):
        """Test HIPAA compliance defaults."""
        config = ComplianceConfig()
        
        assert config.enable_audit_logging == True
        assert config.log_phi_access == True
        assert config.data_retention_days == 2555  # 7 years
    
    def test_fda_defaults(self):
        """Test FDA 21 CFR Part 11 defaults."""
        config = ComplianceConfig()
        
        assert config.enable_digital_signatures == True
        assert config.audit_trail_integrity == True
        assert config.user_authentication == True
    
    def test_phi_tags_list(self):
        """Test PHI tags for de-identification."""
        config = ComplianceConfig()
        
        # Should contain key PHI tags
        assert "PatientName" in config.remove_phi_tags
        assert "PatientID" in config.remove_phi_tags
        assert "PatientBirthDate" in config.remove_phi_tags
        assert "InstitutionName" in config.remove_phi_tags
        assert "StudyDate" in config.remove_phi_tags


class TestAPIConfig:
    """Test suite for API configuration."""
    
    def test_default_api_settings(self):
        """Test default API settings."""
        config = APIConfig()
        
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert config.max_request_size == 100 * 1024 * 1024  # 100MB
        assert config.timeout == 300
    
    def test_cors_settings(self):
        """Test CORS settings for medical applications."""
        config = APIConfig()
        
        # Should have secure default CORS settings
        assert "https://hospital-domain.com" in config.allowed_origins


class TestPlatformConfig:
    """Test suite for main platform configuration."""
    
    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments should work
        for env in ["development", "staging", "production"]:
            with patch.dict(os.environ, {'ENVIRONMENT': env}):
                config = PlatformConfig()
                assert config.environment == env
    
    def test_invalid_environment(self):
        """Test invalid environment rejection."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'invalid_env'}):
            with pytest.raises(Exception):  # Pydantic ValidationError
                PlatformConfig()
    
    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            with patch.dict(os.environ, {'LOG_LEVEL': level}):
                config = PlatformConfig()
                assert config.log_level == level
        
        # Case insensitive
        with patch.dict(os.environ, {'LOG_LEVEL': 'info'}):
            config = PlatformConfig()
            assert config.log_level == "INFO"
    
    def test_invalid_log_level(self):
        """Test invalid log level rejection."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'INVALID'}):
            with pytest.raises(Exception):  # Pydantic ValidationError
                PlatformConfig()
    
    def test_project_paths(self):
        """Test project path configuration."""
        config = PlatformConfig()
        
        # Paths should be Path objects
        assert isinstance(config.project_root, Path)
        assert isinstance(config.src_path, Path)
        
        # Paths should be reasonable
        assert config.project_root.name == "Holistic_Diagnostic_Platform"
        assert config.src_path.name == "src"
    
    def test_nested_configurations(self):
        """Test that all sub-configurations are properly initialized."""
        # Mock all required environment variables for sub-configs
        with patch.dict(os.environ, {
            # Security (required)
            'SECURITY_ENCRYPTION_KEY': 'test_encryption_key_32_bytes_long',
            'SECURITY_JWT_SECRET': 'test_jwt_secret',
            'SECURITY_PASSWORD_SALT': 'test_password_salt',
            # Database (required)
            'DB_USERNAME': 'test_user',
            'DB_PASSWORD': 'test_password',
            # Data paths (required) 
            'DATA_RAW_DATA_PATH': '/tmp/raw',
            'DATA_PROCESSED_DATA_PATH': '/tmp/processed',
            'DATA_CACHE_PATH': '/tmp/cache'
        }):
            config = PlatformConfig()
            
            # All sub-configs should be initialized
            assert isinstance(config.database, DatabaseConfig)
            assert isinstance(config.security, SecurityConfig)
            assert isinstance(config.model, ModelConfig)
            assert isinstance(config.data, DataConfig)
            assert isinstance(config.federated, FederatedConfig)
            assert isinstance(config.api, APIConfig)
            assert isinstance(config.compliance, ComplianceConfig)
    
    def test_env_file_loading(self):
        """Test loading configuration from .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("ENVIRONMENT=production\n")
            f.write("DEBUG=false\n")
            f.write("LOG_LEVEL=ERROR\n")
            env_file = f.name
        
        try:
            # Mock the env_file path
            with patch.object(PlatformConfig.Config, 'env_file', env_file):
                config = PlatformConfig()
                
                # Values from .env file should be loaded
                assert config.environment == "production"
                assert config.debug == False
                assert config.log_level == "ERROR"
        finally:
            os.unlink(env_file)


class TestConfigurationGlobalFunctions:
    """Test suite for global configuration functions."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        
        # Should be the same instance
        assert config1 is config2
    
    def test_reload_config(self):
        """Test configuration reloading."""
        # Get initial config
        config1 = get_config()
        
        # Reload config (this creates a new instance)
        config2 = reload_config()
        
        # Should be a new instance
        assert config1 is not config2
        
        # But get_config should now return the new instance
        config3 = get_config()
        assert config2 is config3
    
    @patch.dict(os.environ, {
        'SECURITY_ENCRYPTION_KEY': 'test_key',
        'SECURITY_JWT_SECRET': 'test_secret', 
        'SECURITY_PASSWORD_SALT': 'test_salt',
        'DB_USERNAME': 'test_user',
        'DB_PASSWORD': 'test_password',
        'DATA_RAW_DATA_PATH': '/tmp/raw',
        'DATA_PROCESSED_DATA_PATH': '/tmp/processed',
        'DATA_CACHE_PATH': '/tmp/cache'
    })
    def test_config_with_all_required_fields(self):
        """Test configuration with all required fields provided."""
        # Should not raise any exceptions
        config = PlatformConfig()
        
        # All sub-configs should be properly initialized
        assert config.database.username == "test_user"
        assert config.security.encryption_key.get_secret_value() == "test_key"
        assert str(config.data.raw_data_path) == "/tmp/raw"


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'production',
        'LOG_LEVEL': 'WARNING',
        'SECURITY_ENCRYPTION_KEY': 'prod_encryption_key_32_bytes_long',
        'SECURITY_JWT_SECRET': 'prod_jwt_secret',
        'SECURITY_PASSWORD_SALT': 'prod_password_salt',
        'DB_HOST': 'prod-db.hospital.com',
        'DB_USERNAME': 'prod_user',
        'DB_PASSWORD': 'prod_password',
        'MODEL_BATCH_SIZE': '16',
        'MODEL_LEARNING_RATE': '0.0001',
        'DATA_RAW_DATA_PATH': '/data/medical/raw',
        'DATA_PROCESSED_DATA_PATH': '/data/medical/processed',
        'DATA_CACHE_PATH': '/cache/medical'
    })
    def test_production_like_configuration(self):
        """Test a production-like configuration scenario."""
        config = PlatformConfig()
        
        # Environment settings
        assert config.environment == "production"
        assert config.log_level == "WARNING"
        
        # Database settings
        assert config.database.host == "prod-db.hospital.com"
        assert config.database.username == "prod_user"
        
        # Model settings
        assert config.model.batch_size == 16
        assert config.model.learning_rate == 0.0001
        
        # Security settings
        assert config.security.encryption_key.get_secret_value() == "prod_encryption_key_32_bytes_long"
        
        # All required components should be present
        assert config.compliance.enable_audit_logging == True
        assert config.compliance.enable_digital_signatures == True
    
    def test_configuration_serialization(self):
        """Test that configuration can be properly serialized (for debugging)."""
        with patch.dict(os.environ, {
            'SECURITY_ENCRYPTION_KEY': 'test_key',
            'SECURITY_JWT_SECRET': 'test_secret',
            'SECURITY_PASSWORD_SALT': 'test_salt',
            'DB_USERNAME': 'test_user',
            'DB_PASSWORD': 'test_password',
            'DATA_RAW_DATA_PATH': '/tmp/raw',
            'DATA_PROCESSED_DATA_PATH': '/tmp/processed',
            'DATA_CACHE_PATH': '/tmp/cache'
        }):
            config = PlatformConfig()
            
            # Should be able to convert to dict
            config_dict = config.dict()
            assert isinstance(config_dict, dict)
            
            # Sensitive fields should be masked
            assert "**********" in str(config_dict['database']['password'])
            assert "**********" in str(config_dict['security']['encryption_key'])
    
    def test_missing_required_environment_variables(self):
        """Test behavior when required environment variables are missing."""
        # Clear relevant environment variables
        env_vars_to_clear = [
            'SECURITY_ENCRYPTION_KEY', 'SECURITY_JWT_SECRET', 
            'SECURITY_PASSWORD_SALT', 'DB_USERNAME', 'DB_PASSWORD'
        ]
        
        with patch.dict(os.environ, {}, clear=True):
            # Should raise validation error for missing required fields
            with pytest.raises(Exception):  # Pydantic ValidationError
                PlatformConfig()