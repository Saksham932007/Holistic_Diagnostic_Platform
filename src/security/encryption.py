"""
AES-256 encryption utilities for medical data at rest.

This module provides HIPAA-compliant encryption/decryption capabilities
for protecting sensitive medical data with strong cryptographic standards.
"""

import os
import base64
import hashlib
import secrets
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import json

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, padding, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import hmac as crypto_hmac

from ..config import get_config
from ..utils import AuditEventType, AuditSeverity, log_audit_event


class EncryptionError(Exception):
    """Custom exception for encryption operations."""
    pass


class MedicalDataEncryptor:
    """
    HIPAA-compliant AES-256 encryption for medical data.
    
    This class provides secure encryption/decryption capabilities with
    proper key management, integrity verification, and audit logging.
    """
    
    # Encryption parameters
    AES_KEY_SIZE = 32  # 256 bits
    IV_SIZE = 16  # 128 bits for AES-CBC
    SALT_SIZE = 32  # 256 bits
    TAG_SIZE = 16  # 128 bits for HMAC
    PBKDF2_ITERATIONS = 100000  # NIST recommended minimum
    
    def __init__(
        self,
        master_key: Optional[bytes] = None,
        key_derivation_method: str = "pbkdf2"
    ):
        """
        Initialize the medical data encryptor.
        
        Args:
            master_key: Master encryption key (32 bytes). If None, uses config.
            key_derivation_method: Key derivation method ("pbkdf2" or "scrypt")
        """
        self._config = get_config()
        self._key_derivation_method = key_derivation_method
        
        # Get or generate master key
        if master_key is None:
            master_key_str = self._config.security.encryption_key.get_secret_value()
            self._master_key = self._normalize_key(master_key_str)
        else:
            if len(master_key) != self.AES_KEY_SIZE:
                raise EncryptionError(f"Master key must be {self.AES_KEY_SIZE} bytes")
            self._master_key = master_key
        
        # Initialize backend
        self._backend = default_backend()
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="Medical data encryptor initialized",
            additional_data={
                'key_derivation_method': key_derivation_method,
                'encryption_algorithm': 'AES-256-CBC'
            }
        )
    
    def _normalize_key(self, key_input: Union[str, bytes]) -> bytes:
        """
        Normalize input to 32-byte key using SHA-256.
        
        Args:
            key_input: Key material as string or bytes
            
        Returns:
            32-byte key suitable for AES-256
        """
        if isinstance(key_input, str):
            key_input = key_input.encode('utf-8')
        
        # Use SHA-256 to normalize to 32 bytes
        digest = hashes.Hash(hashes.SHA256(), backend=self._backend)
        digest.update(key_input)
        return digest.finalize()
    
    def _derive_key(self, password: bytes, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2 or Scrypt.
        
        Args:
            password: Password bytes
            salt: Salt bytes
            
        Returns:
            Derived encryption key
        """
        if self._key_derivation_method == "pbkdf2":
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.AES_KEY_SIZE,
                salt=salt,
                iterations=self.PBKDF2_ITERATIONS,
                backend=self._backend
            )
        elif self._key_derivation_method == "scrypt":
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=self.AES_KEY_SIZE,
                salt=salt,
                n=2**14,  # CPU/memory cost parameter
                r=8,      # Block size parameter
                p=1,      # Parallelization parameter
                backend=self._backend
            )
        else:
            raise EncryptionError(f"Unsupported key derivation method: {self._key_derivation_method}")
        
        return kdf.derive(password)
    
    def _generate_iv(self) -> bytes:
        """Generate a random initialization vector."""
        return secrets.token_bytes(self.IV_SIZE)
    
    def _generate_salt(self) -> bytes:
        """Generate a random salt."""
        return secrets.token_bytes(self.SALT_SIZE)
    
    def _calculate_hmac(self, key: bytes, data: bytes) -> bytes:
        """
        Calculate HMAC-SHA256 for integrity verification.
        
        Args:
            key: HMAC key
            data: Data to authenticate
            
        Returns:
            HMAC tag
        """
        h = crypto_hmac.HMAC(key, hashes.SHA256(), backend=self._backend)
        h.update(data)
        return h.finalize()
    
    def _verify_hmac(self, key: bytes, data: bytes, tag: bytes) -> bool:
        """
        Verify HMAC-SHA256 tag.
        
        Args:
            key: HMAC key
            data: Original data
            tag: HMAC tag to verify
            
        Returns:
            True if verification succeeds
        """
        try:
            h = crypto_hmac.HMAC(key, hashes.SHA256(), backend=self._backend)
            h.update(data)
            h.verify(tag)
            return True
        except InvalidSignature:
            return False
    
    def encrypt_data(
        self,
        plaintext: Union[str, bytes],
        additional_context: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None
    ) -> bytes:
        """
        Encrypt data using AES-256-CBC with HMAC authentication.
        
        Args:
            plaintext: Data to encrypt
            additional_context: Additional context for audit logging
            patient_id: Patient ID for audit logging
            
        Returns:
            Encrypted data with embedded metadata
        """
        try:
            # Convert to bytes if string
            if isinstance(plaintext, str):
                plaintext_bytes = plaintext.encode('utf-8')
            else:
                plaintext_bytes = plaintext
            
            # Generate random salt and IV
            salt = self._generate_salt()
            iv = self._generate_iv()
            
            # Derive encryption key
            encryption_key = self._derive_key(self._master_key, salt)
            
            # PKCS7 padding for AES-CBC
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(plaintext_bytes)
            padded_data += padder.finalize()
            
            # Encrypt
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.CBC(iv),
                backend=self._backend
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Create metadata
            metadata = {
                'algorithm': 'AES-256-CBC',
                'kdf': self._key_derivation_method,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '1.0'
            }
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            
            # Combine components: salt + iv + metadata_length + metadata + ciphertext
            metadata_length = len(metadata_bytes).to_bytes(4, 'big')
            combined_data = salt + iv + metadata_length + metadata_bytes + ciphertext
            
            # Calculate HMAC for integrity
            hmac_key = self._derive_key(self._master_key + b'hmac', salt)
            hmac_tag = self._calculate_hmac(hmac_key, combined_data)
            
            # Final encrypted blob: combined_data + hmac_tag
            encrypted_blob = combined_data + hmac_tag
            
            # Audit log
            log_audit_event(
                event_type=AuditEventType.DATA_EXPORT,
                severity=AuditSeverity.INFO,
                message="Data encrypted successfully",
                patient_id=patient_id,
                additional_data={
                    'data_size': len(plaintext_bytes),
                    'encrypted_size': len(encrypted_blob),
                    'algorithm': 'AES-256-CBC',
                    'context': additional_context or {}
                }
            )
            
            return encrypted_blob
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.DATA_EXPORT,
                severity=AuditSeverity.ERROR,
                message=f"Encryption failed: {str(e)}",
                patient_id=patient_id,
                additional_data={'error': str(e)}
            )
            raise EncryptionError(f"Encryption failed: {str(e)}")
    
    def decrypt_data(
        self,
        encrypted_blob: bytes,
        additional_context: Optional[Dict[str, Any]] = None,
        patient_id: Optional[str] = None
    ) -> bytes:
        """
        Decrypt data encrypted with encrypt_data.
        
        Args:
            encrypted_blob: Encrypted data blob
            additional_context: Additional context for audit logging
            patient_id: Patient ID for audit logging
            
        Returns:
            Decrypted plaintext data
        """
        try:
            if len(encrypted_blob) < (self.SALT_SIZE + self.IV_SIZE + 4 + self.TAG_SIZE):
                raise EncryptionError("Invalid encrypted blob format")
            
            # Extract components
            offset = 0
            salt = encrypted_blob[offset:offset + self.SALT_SIZE]
            offset += self.SALT_SIZE
            
            iv = encrypted_blob[offset:offset + self.IV_SIZE]
            offset += self.IV_SIZE
            
            metadata_length = int.from_bytes(encrypted_blob[offset:offset + 4], 'big')
            offset += 4
            
            metadata_bytes = encrypted_blob[offset:offset + metadata_length]
            offset += metadata_length
            
            # Split ciphertext and HMAC
            hmac_tag = encrypted_blob[-self.TAG_SIZE:]
            ciphertext = encrypted_blob[offset:-self.TAG_SIZE]
            combined_data = encrypted_blob[:-self.TAG_SIZE]
            
            # Verify HMAC integrity
            hmac_key = self._derive_key(self._master_key + b'hmac', salt)
            if not self._verify_hmac(hmac_key, combined_data, hmac_tag):
                raise EncryptionError("HMAC verification failed - data may be corrupted")
            
            # Parse metadata
            try:
                metadata = json.loads(metadata_bytes.decode('utf-8'))
            except json.JSONDecodeError:
                raise EncryptionError("Invalid metadata format")
            
            # Verify algorithm compatibility
            if metadata.get('algorithm') != 'AES-256-CBC':
                raise EncryptionError(f"Unsupported algorithm: {metadata.get('algorithm')}")
            
            # Derive decryption key
            encryption_key = self._derive_key(self._master_key, salt)
            
            # Decrypt
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.CBC(iv),
                backend=self._backend
            )
            decryptor = cipher.decryptor()
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove PKCS7 padding
            unpadder = padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_plaintext)
            plaintext += unpadder.finalize()
            
            # Audit log
            log_audit_event(
                event_type=AuditEventType.PHI_ACCESS,
                severity=AuditSeverity.INFO,
                message="Data decrypted successfully",
                patient_id=patient_id,
                additional_data={
                    'decrypted_size': len(plaintext),
                    'encrypted_size': len(encrypted_blob),
                    'algorithm': metadata.get('algorithm'),
                    'context': additional_context or {}
                }
            )
            
            return plaintext
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.PHI_ACCESS,
                severity=AuditSeverity.ERROR,
                message=f"Decryption failed: {str(e)}",
                patient_id=patient_id,
                additional_data={'error': str(e)}
            )
            raise EncryptionError(f"Decryption failed: {str(e)}")
    
    def encrypt_file(
        self,
        input_path: Path,
        output_path: Path,
        patient_id: Optional[str] = None
    ) -> bool:
        """
        Encrypt a file to disk.
        
        Args:
            input_path: Path to input file
            output_path: Path for encrypted output file
            patient_id: Patient ID for audit logging
            
        Returns:
            True if successful
        """
        try:
            # Read input file
            with open(input_path, 'rb') as f:
                plaintext = f.read()
            
            # Encrypt data
            encrypted_data = self.encrypt_data(
                plaintext,
                additional_context={'input_file': str(input_path)},
                patient_id=patient_id
            )
            
            # Write encrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            log_audit_event(
                event_type=AuditEventType.DATA_EXPORT,
                severity=AuditSeverity.INFO,
                message="File encrypted successfully",
                patient_id=patient_id,
                additional_data={
                    'input_file': str(input_path),
                    'output_file': str(output_path),
                    'original_size': len(plaintext),
                    'encrypted_size': len(encrypted_data)
                }
            )
            
            return True
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.DATA_EXPORT,
                severity=AuditSeverity.ERROR,
                message=f"File encryption failed: {str(e)}",
                patient_id=patient_id,
                additional_data={
                    'input_file': str(input_path),
                    'error': str(e)
                }
            )
            return False
    
    def decrypt_file(
        self,
        input_path: Path,
        output_path: Path,
        patient_id: Optional[str] = None
    ) -> bool:
        """
        Decrypt a file from disk.
        
        Args:
            input_path: Path to encrypted input file
            output_path: Path for decrypted output file
            patient_id: Patient ID for audit logging
            
        Returns:
            True if successful
        """
        try:
            # Read encrypted file
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt data
            plaintext = self.decrypt_data(
                encrypted_data,
                additional_context={'encrypted_file': str(input_path)},
                patient_id=patient_id
            )
            
            # Write decrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(plaintext)
            
            log_audit_event(
                event_type=AuditEventType.PHI_ACCESS,
                severity=AuditSeverity.INFO,
                message="File decrypted successfully",
                patient_id=patient_id,
                additional_data={
                    'input_file': str(input_path),
                    'output_file': str(output_path),
                    'encrypted_size': len(encrypted_data),
                    'decrypted_size': len(plaintext)
                }
            )
            
            return True
            
        except Exception as e:
            log_audit_event(
                event_type=AuditEventType.PHI_ACCESS,
                severity=AuditSeverity.ERROR,
                message=f"File decryption failed: {str(e)}",
                patient_id=patient_id,
                additional_data={
                    'input_file': str(input_path),
                    'error': str(e)
                }
            )
            return False
    
    def generate_master_key(self) -> bytes:
        """
        Generate a cryptographically secure master key.
        
        Returns:
            32-byte master key suitable for AES-256
        """
        return secrets.token_bytes(self.AES_KEY_SIZE)
    
    def export_key_safely(self, key: bytes, password: str) -> str:
        """
        Export a key in password-protected format.
        
        Args:
            key: Key to export
            password: Password for protection
            
        Returns:
            Base64-encoded encrypted key
        """
        # Use the key as plaintext and encrypt with password
        temp_encryptor = MedicalDataEncryptor(self._normalize_key(password))
        encrypted_key = temp_encryptor.encrypt_data(key)
        return base64.b64encode(encrypted_key).decode('ascii')
    
    def import_key_safely(self, encrypted_key_b64: str, password: str) -> bytes:
        """
        Import a password-protected key.
        
        Args:
            encrypted_key_b64: Base64-encoded encrypted key
            password: Password for decryption
            
        Returns:
            Decrypted key bytes
        """
        encrypted_key = base64.b64decode(encrypted_key_b64.encode('ascii'))
        temp_encryptor = MedicalDataEncryptor(self._normalize_key(password))
        return temp_encryptor.decrypt_data(encrypted_key)


class RSAKeyManager:
    """
    RSA key management for digital signatures and key exchange.
    
    This class provides RSA key generation, storage, and operations
    for digital signatures required by FDA 21 CFR Part 11.
    """
    
    KEY_SIZE = 2048  # RSA key size in bits
    
    def __init__(self):
        """Initialize RSA key manager."""
        self._config = get_config()
        self._backend = default_backend()
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate an RSA key pair.
        
        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.KEY_SIZE,
            backend=self._backend
        )
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        log_audit_event(
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
            message="RSA key pair generated",
            additional_data={'key_size': self.KEY_SIZE}
        )
        
        return private_pem, public_pem
    
    def sign_data(self, private_key_pem: bytes, data: bytes) -> bytes:
        """
        Create RSA digital signature.
        
        Args:
            private_key_pem: Private key in PEM format
            data: Data to sign
            
        Returns:
            Digital signature
        """
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self._backend
        )
        
        signature = private_key.sign(
            data,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(
        self,
        public_key_pem: bytes,
        data: bytes,
        signature: bytes
    ) -> bool:
        """
        Verify RSA digital signature.
        
        Args:
            public_key_pem: Public key in PEM format
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if signature is valid
        """
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=self._backend
            )
            
            public_key.verify(
                signature,
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False


# Convenience functions for global use
_global_encryptor: Optional[MedicalDataEncryptor] = None


def get_encryptor() -> MedicalDataEncryptor:
    """Get global encryptor instance."""
    global _global_encryptor
    if _global_encryptor is None:
        _global_encryptor = MedicalDataEncryptor()
    return _global_encryptor


def encrypt_data(data: Union[str, bytes], **kwargs) -> bytes:
    """Convenience function to encrypt data."""
    return get_encryptor().encrypt_data(data, **kwargs)


def decrypt_data(encrypted_data: bytes, **kwargs) -> bytes:
    """Convenience function to decrypt data."""
    return get_encryptor().decrypt_data(encrypted_data, **kwargs)


def encrypt_file(input_path: Path, output_path: Path, **kwargs) -> bool:
    """Convenience function to encrypt a file."""
    return get_encryptor().encrypt_file(input_path, output_path, **kwargs)


def decrypt_file(input_path: Path, output_path: Path, **kwargs) -> bool:
    """Convenience function to decrypt a file."""
    return get_encryptor().decrypt_file(input_path, output_path, **kwargs)