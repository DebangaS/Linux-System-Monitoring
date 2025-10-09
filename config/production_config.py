"""
Production Configuration Management
Author: Member 1 
"""
import os
import json
import secrets
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import timedelta
from dataclasses import dataclass, asdict, fields

# NOTE: Using the logging module is better practice than print() for libraries.
logger = logging.getLogger(__name__)

# NOTE: No changes to dataclasses needed, they were well-defined.
@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str
    jwt_secret: str
    password_salt_rounds: int = 12
    session_timeout_hours: int = 24
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    require_https: bool = True
    secure_cookies: bool = True

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = 'sqlite:///data/production.db'
    pool_size: int = 20
    pool_timeout: int = 30
    pool_recycle_hours: int = 1
    backup_enabled: bool = True
    backup_interval_hours: int = 6
    cleanup_enabled: bool = True
    cleanup_retention_days: int = 30

@dataclass
class MonitoringConfig:
    """Monitoring configuration settings"""
    enabled: bool = True
    interval_seconds: int = 2
    collect_gpu: bool = True
    collect_network_advanced: bool = True
    collect_processes: bool = True
    anomaly_detection_enabled: bool = True
    predictive_monitoring_enabled: bool = True

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    max_workers: int = 8
    max_connections: int = 100
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    compression_enabled: bool = True
    rate_limiting_enabled: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    file_path: str = "data/logs/production.log"
    max_file_size_mb: int = 100
    backup_count: int = 10
    # FIX: Corrected the incomplete format string.
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _recursive_update(d: dict, u: dict) -> dict:
    """Recursively update a dictionary."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class ProductionConfigManager:
    """
    Manages application configuration with a clear precedence:
    1. Default values from dataclasses.
    2. Values from the JSON config file.
    3. Values from environment variables (highest precedence).
    """
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file or "config/production.json")
        self.config_dir = self.config_file.parent
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._config: Dict[str, Any] = {}
        # FIX: Store typed config objects for cleaner access.
        self.security: SecurityConfig
        self.database: DatabaseConfig
        self.monitoring: MonitoringConfig
        self.performance: PerformanceConfig
        self.logging: LoggingConfig
        
        self.load_configuration()

    def load_configuration(self):
        """Loads configuration with the correct precedence: Defaults < File < Env."""
        try:
            # FIX: Restructured loading logic to be more robust and logical.
            # 1. Start with defaults derived directly from the dataclasses.
            config = self._get_defaults_from_dataclasses()

            # 2. Update with settings from the JSON file if it exists.
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                config = _recursive_update(config, file_config)

            # 3. Override with environment variables.
            self._load_from_environment(config)

            # FIX: Critical fix for secret key persistence.
            # Generate and save secrets ONLY if they don't exist after loading.
            should_save = self._ensure_secrets_exist(config)

            self._config = config
            
            # Create typed dataclass instances from the final config dictionary.
            self._initialize_dataclass_instances()

            if should_save:
                self.save_configuration()
                logger.info("New secret keys generated and saved to configuration file.")

        except Exception as e:
            logger.error(f"FATAL: Error loading configuration: {e}. Falling back to complete defaults.", exc_info=True)
            self._config = self._get_defaults_from_dataclasses()
            self._ensure_secrets_exist(self._config)
            self._initialize_dataclass_instances()

    def _get_defaults_from_dataclasses(self) -> dict:
        """Dynamically generate default config dict from the dataclasses."""
        # NOTE: This removes the need for the repetitive `_validate_and_set_defaults` method.
        return {
            "security": asdict(SecurityConfig(secret_key="", jwt_secret="")), # Secrets handled separately
            "database": asdict(DatabaseConfig()),
            "monitoring": asdict(MonitoringConfig()),
            "performance": asdict(PerformanceConfig()),
            "logging": asdict(LoggingConfig()),
        }
    
    def _initialize_dataclass_instances(self):
        """Populates the typed config instances like self.security, self.database, etc."""
        self.security = SecurityConfig(**self._config['security'])
        self.database = DatabaseConfig(**self._config['database'])
        self.monitoring = MonitoringConfig(**self._config['monitoring'])
        self.performance = PerformanceConfig(**self._config['performance'])
        self.logging = LoggingConfig(**self._config['logging'])

    def _ensure_secrets_exist(self, config: dict) -> bool:
        """Generates secrets if they are missing. Returns True if changes were made."""
        made_changes = False
        if not config.get('security', {}).get('secret_key'):
            config['security']['secret_key'] = secrets.token_urlsafe(32)
            made_changes = True
        if not config.get('security', {}).get('jwt_secret'):
            config['security']['jwt_secret'] = secrets.token_urlsafe(32)
            made_changes = True
        return made_changes

    def _load_from_environment(self, config: dict):
        """Loads configuration from environment variables, overriding existing values."""
        # NOTE: Mappings can be easily extended here.
        env_mappings = {
            'SECRET_KEY': 'security.secret_key', 'JWT_SECRET': 'security.jwt_secret',
            'DATABASE_URL': 'database.url', 'LOG_LEVEL': 'logging.level',
            'MONITORING_ENABLED': 'monitoring.enabled', 'CACHE_ENABLED': 'performance.cache_enabled'
        }
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested_config(config, config_path, value)

    def _set_nested_config(self, config: dict, path: str, value: Any):
        """Sets a value in a nested dictionary based on a dot-separated path."""
        keys = path.split('.')
        current = config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        
        final_key = keys[-1]
        
        # FIX: More robust type conversion for environment variables.
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                current[final_key] = value.lower() == 'true'
                return
            try:
                current[final_key] = int(value)
                return
            except ValueError:
                pass # Not an int
            try:
                current[final_key] = float(value)
                return
            except ValueError:
                pass # Not a float
        
        current[final_key] = value

    def save_configuration(self):
        """Saves the current configuration dictionary to the JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_file}: {e}")

    # NOTE: These accessors are now simpler, returning the pre-initialized dataclass instances.
    def get_security_config(self) -> SecurityConfig:
        return self.security

    def get_database_config(self) -> DatabaseConfig:
        return self.database

    def get_monitoring_config(self) -> MonitoringConfig:
        return self.monitoring

    def get_performance_config(self) -> PerformanceConfig:
        return self.performance

    def get_logging_config(self) -> LoggingConfig:
        return self.logging
    
    def get_config(self, key: str, default=None) -> Any:
        """Gets a raw configuration value by dot-separated key."""
        keys = key.split('.')
        current = self._config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def get_flask_config(self) -> Dict[str, Any]:
        """Returns a dictionary formatted for Flask app configuration."""
        return {
            'SECRET_KEY': self.security.secret_key,
            'JWT_SECRET_KEY': self.security.jwt_secret,
            'PERMANENT_SESSION_LIFETIME': timedelta(hours=self.security.session_timeout_hours),
            'SESSION_COOKIE_SECURE': self.security.secure_cookies,
            'SESSION_COOKIE_HTTPONLY': True,
            'SQLALCHEMY_DATABASE_URI': self.database.url, # NOTE: Common Flask-SQLAlchemy key
            'SQLALCHEMY_ENGINE_OPTIONS': {
                'pool_size': self.database.pool_size,
                'pool_timeout': self.database.pool_timeout,
            },
            'MAX_CONTENT_LENGTH': 50 * 1024 * 1024,  # 50MB example
            # FIX: Corrected incomplete ternary operator.
            'COMPRESS_ALGORITHM': 'gzip' if self.performance.compression_enabled else None,
        }

# Global singleton instance
production_config = ProductionConfigManager()

def get_production_config() -> ProductionConfigManager:
    """Returns the global production configuration manager instance."""
    return production_config
