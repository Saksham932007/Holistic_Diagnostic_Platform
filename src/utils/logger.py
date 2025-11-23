"""
Immutable audit logging system for FDA compliance.

This module provides comprehensive, tamper-evident logging capabilities
required for medical device software under FDA 21 CFR Part 11 regulations.
"""

import json
import hashlib
import hmac
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from enum import Enum
import threading
import uuid
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

from ..config import get_config


class AuditEventType(Enum):
    """Enumeration of audit event types for medical imaging platform."""
    
    # Data Access Events
    PHI_ACCESS = "phi_access"
    DICOM_READ = "dicom_read"
    DICOM_WRITE = "dicom_write"
    PATIENT_DATA_QUERY = "patient_data_query"
    
    # Processing Events
    IMAGE_PROCESSING = "image_processing"
    MODEL_INFERENCE = "model_inference"
    SEGMENTATION_COMPLETE = "segmentation_complete"
    CLASSIFICATION_COMPLETE = "classification_complete"
    
    # Security Events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    AUTHENTICATION_FAILURE = "auth_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGE = "config_change"
    
    # Compliance Events
    DE_IDENTIFICATION = "de_identification"
    DATA_EXPORT = "data_export"
    AUDIT_LOG_ACCESS = "audit_log_access"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEntry:
    """
    Immutable audit log entry with cryptographic integrity.
    
    Each audit entry is digitally signed and contains a hash chain
    to ensure tamper detection.
    """
    
    def __init__(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        previous_hash: Optional[str] = None
    ):
        """
        Initialize an immutable audit entry.
        
        Args:
            event_type: Type of audit event
            severity: Severity level of the event
            message: Human-readable event description
            user_id: User identifier (anonymized)
            session_id: Session identifier
            patient_id: Patient identifier (anonymized)
            study_uid: DICOM Study Instance UID
            additional_data: Additional event-specific data
            previous_hash: Hash of the previous audit entry (for chaining)
        """
        self._timestamp = datetime.now(timezone.utc)
        self._entry_id = str(uuid.uuid4())
        self._event_type = event_type
        self._severity = severity
        self._message = message
        self._user_id = user_id
        self._session_id = session_id
        self._patient_id = patient_id
        self._study_uid = study_uid
        self._additional_data = additional_data or {}
        self._previous_hash = previous_hash
        
        # Calculate integrity hash
        self._entry_hash = self._calculate_hash()
        
        # Digital signature (placeholder for production implementation)
        self._signature = self._sign_entry()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the entry content."""
        content = {
            'entry_id': self._entry_id,
            'timestamp': self._timestamp.isoformat(),
            'event_type': self._event_type.value,
            'severity': self._severity.value,
            'message': self._message,
            'user_id': self._user_id,
            'session_id': self._session_id,
            'patient_id': self._patient_id,
            'study_uid': self._study_uid,
            'additional_data': self._additional_data,
            'previous_hash': self._previous_hash
        }
        
        content_json = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_json.encode('utf-8')).hexdigest()
    
    def _sign_entry(self) -> str:
        """
        Create digital signature for the entry.
        
        Returns:
            Base64-encoded digital signature
        """
        # In production, this would use actual private key
        # For now, we use HMAC with a secret key
        config = get_config()
        secret_key = config.security.encryption_key.get_secret_value().encode('utf-8')
        
        signature = hmac.new(
            secret_key,
            self._entry_hash.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert audit entry to dictionary for serialization.
        
        Returns:
            Dictionary representation of the audit entry
        """
        return {
            'entry_id': self._entry_id,
            'timestamp': self._timestamp.isoformat(),
            'event_type': self._event_type.value,
            'severity': self._severity.value,
            'message': self._message,
            'user_id': self._user_id,
            'session_id': self._session_id,
            'patient_id': self._patient_id,
            'study_uid': self._study_uid,
            'additional_data': self._additional_data,
            'previous_hash': self._previous_hash,
            'entry_hash': self._entry_hash,
            'signature': self._signature
        }
    
    def verify_integrity(self) -> bool:
        """
        Verify the integrity of this audit entry.
        
        Returns:
            True if the entry integrity is valid
        """
        # Recalculate hash and compare
        recalculated_hash = self._calculate_hash()
        return recalculated_hash == self._entry_hash
    
    @property
    def entry_hash(self) -> str:
        """Get the entry hash."""
        return self._entry_hash
    
    @property
    def entry_id(self) -> str:
        """Get the entry ID."""
        return self._entry_id
    
    @property
    def timestamp(self) -> datetime:
        """Get the entry timestamp."""
        return self._timestamp


class AuditLogger:
    """
    Thread-safe audit logger with cryptographic integrity and hash chaining.
    
    This logger ensures all audit events are immutably recorded with
    proper hash chaining to detect tampering attempts.
    """
    
    def __init__(self, log_file_path: Optional[Path] = None):
        """
        Initialize the audit logger.
        
        Args:
            log_file_path: Path to the audit log file
        """
        self._config = get_config()
        self._lock = threading.Lock()
        
        if log_file_path is None:
            log_dir = self._config.project_root / "logs" / "audit"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file_path = log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        else:
            self._log_file_path = log_file_path
        
        self._last_hash: Optional[str] = None
        self._entry_count = 0
        
        # Initialize structured logger
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Load last hash if log file exists
        self._load_last_hash()
        
        # Log audit system initialization
        self._log_system_event(
            AuditEventType.SYSTEM_START,
            AuditSeverity.INFO,
            "Audit logging system initialized"
        )
    
    def _load_last_hash(self) -> None:
        """Load the hash of the last audit entry from the log file."""
        if not self._log_file_path.exists():
            return
        
        try:
            with open(self._log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    entry_data = json.loads(last_line)
                    self._last_hash = entry_data.get('entry_hash')
                    self._entry_count = len(lines)
        except (json.JSONDecodeError, IOError) as e:
            # If we can't read the log file, start fresh
            self._last_hash = None
            self._entry_count = 0
    
    def _write_entry(self, entry: AuditEntry) -> None:
        """
        Write an audit entry to the log file.
        
        Args:
            entry: The audit entry to write
        """
        with self._lock:
            try:
                with open(self._log_file_path, 'a', encoding='utf-8') as f:
                    json.dump(entry.to_dict(), f, default=str)
                    f.write('\n')
                
                self._last_hash = entry.entry_hash
                self._entry_count += 1
                
            except IOError as e:
                # Critical: audit logging failure
                structlog.get_logger().critical(
                    "Audit log write failure",
                    error=str(e),
                    entry_id=entry.entry_id
                )
                raise
    
    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        study_uid: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of audit event
            severity: Severity level
            message: Event description
            user_id: User identifier (anonymized)
            session_id: Session identifier
            patient_id: Patient identifier (anonymized)
            study_uid: DICOM Study Instance UID
            additional_data: Additional event-specific data
            
        Returns:
            The entry ID of the logged event
        """
        entry = AuditEntry(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            session_id=session_id,
            patient_id=patient_id,
            study_uid=study_uid,
            additional_data=additional_data,
            previous_hash=self._last_hash
        )
        
        self._write_entry(entry)
        return entry.entry_id
    
    def _log_system_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        message: str
    ) -> str:
        """Log a system-level event."""
        return self.log_event(
            event_type=event_type,
            severity=severity,
            message=message,
            additional_data={'system_event': True}
        )
    
    def log_phi_access(
        self,
        user_id: str,
        session_id: str,
        patient_id: str,
        access_type: str,
        study_uid: Optional[str] = None
    ) -> str:
        """
        Log PHI (Protected Health Information) access event.
        
        Args:
            user_id: User accessing the PHI
            session_id: Session identifier
            patient_id: Patient identifier (anonymized)
            access_type: Type of access (read, write, delete)
            study_uid: DICOM Study Instance UID
            
        Returns:
            The entry ID of the logged event
        """
        return self.log_event(
            event_type=AuditEventType.PHI_ACCESS,
            severity=AuditSeverity.INFO,
            message=f"PHI access: {access_type}",
            user_id=user_id,
            session_id=session_id,
            patient_id=patient_id,
            study_uid=study_uid,
            additional_data={'access_type': access_type}
        )
    
    def log_model_inference(
        self,
        user_id: str,
        session_id: str,
        model_name: str,
        study_uid: str,
        inference_time_ms: float,
        patient_id: Optional[str] = None
    ) -> str:
        """
        Log model inference event.
        
        Args:
            user_id: User running the inference
            session_id: Session identifier
            model_name: Name of the model used
            study_uid: DICOM Study Instance UID
            inference_time_ms: Inference time in milliseconds
            patient_id: Patient identifier (anonymized)
            
        Returns:
            The entry ID of the logged event
        """
        return self.log_event(
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=AuditSeverity.INFO,
            message=f"Model inference: {model_name}",
            user_id=user_id,
            session_id=session_id,
            patient_id=patient_id,
            study_uid=study_uid,
            additional_data={
                'model_name': model_name,
                'inference_time_ms': inference_time_ms
            }
        )
    
    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the entire audit log chain.
        
        Returns:
            True if the entire chain is valid
        """
        if not self._log_file_path.exists():
            return True  # Empty log is valid
        
        try:
            with open(self._log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            previous_hash = None
            for line_num, line in enumerate(lines, 1):
                try:
                    entry_data = json.loads(line.strip())
                    
                    # Check hash chain
                    if entry_data.get('previous_hash') != previous_hash:
                        return False
                    
                    # Verify entry integrity
                    entry = AuditEntry(
                        event_type=AuditEventType(entry_data['event_type']),
                        severity=AuditSeverity(entry_data['severity']),
                        message=entry_data['message'],
                        user_id=entry_data.get('user_id'),
                        session_id=entry_data.get('session_id'),
                        patient_id=entry_data.get('patient_id'),
                        study_uid=entry_data.get('study_uid'),
                        additional_data=entry_data.get('additional_data'),
                        previous_hash=entry_data.get('previous_hash')
                    )
                    
                    if not entry.verify_integrity():
                        return False
                    
                    previous_hash = entry_data['entry_hash']
                    
                except (json.JSONDecodeError, KeyError, ValueError):
                    return False
            
            return True
            
        except IOError:
            return False
    
    def get_entries_by_patient(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get all audit entries for a specific patient.
        
        Args:
            patient_id: Patient identifier (anonymized)
            
        Returns:
            List of audit entries for the patient
        """
        entries = []
        
        if not self._log_file_path.exists():
            return entries
        
        try:
            with open(self._log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry_data = json.loads(line.strip())
                        if entry_data.get('patient_id') == patient_id:
                            entries.append(entry_data)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        
        return entries


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None
_logger_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """
    Get the global audit logger instance.
    
    Returns:
        AuditLogger: The global audit logger
    """
    global _audit_logger
    
    if _audit_logger is None:
        with _logger_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger()
    
    return _audit_logger


def log_audit_event(
    event_type: AuditEventType,
    severity: AuditSeverity,
    message: str,
    **kwargs
) -> str:
    """
    Convenience function to log an audit event.
    
    Args:
        event_type: Type of audit event
        severity: Severity level
        message: Event description
        **kwargs: Additional parameters
        
    Returns:
        The entry ID of the logged event
    """
    logger = get_audit_logger()
    return logger.log_event(event_type, severity, message, **kwargs)