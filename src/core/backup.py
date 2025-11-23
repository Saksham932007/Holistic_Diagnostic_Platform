"""
Automated Backup System

Comprehensive backup system for medical data with encryption,
versioning, and compliance with HIPAA retention requirements.
"""

import os
import asyncio
import logging
import shutil
import hashlib
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import boto3
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import settings
from src.core.audit import audit_logger

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Types of backups supported."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"

class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

@dataclass
class BackupMetadata:
    """Metadata for backup operations."""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    size_bytes: int
    file_count: int
    checksum: str
    encryption_key_id: str
    retention_date: datetime
    status: BackupStatus
    error_message: Optional[str] = None
    restore_tested: bool = False
    compliance_verified: bool = False

class EncryptionManager:
    """Manages encryption for backup data."""
    
    def __init__(self):
        """Initialize encryption manager."""
        self.encryption_key = self._get_or_create_key()
        self.fernet = Fernet(self.encryption_key)
    
    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key."""
        key_path = Path(settings.backup_encryption_key_path)
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            with open(key_path, 'wb') as f:
                f.write(key)
            os.chmod(key_path, 0o600)  # Restrict permissions
            return key
    
    def encrypt_file(self, input_path: Path, output_path: Path) -> str:
        """Encrypt file and return checksum."""
        with open(input_path, 'rb') as infile:
            data = infile.read()
        
        encrypted_data = self.fernet.encrypt(data)
        
        with open(output_path, 'wb') as outfile:
            outfile.write(encrypted_data)
        
        # Calculate checksum of encrypted data
        return hashlib.sha256(encrypted_data).hexdigest()
    
    def decrypt_file(self, input_path: Path, output_path: Path) -> bool:
        """Decrypt file."""
        try:
            with open(input_path, 'rb') as infile:
                encrypted_data = infile.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(decrypted_data)
            
            return True
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return False

class S3BackupStorage:
    """Manages S3 storage for backups."""
    
    def __init__(self):
        """Initialize S3 client."""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket_name = settings.backup_s3_bucket
    
    async def upload_backup(self, local_path: Path, s3_key: str) -> bool:
        """Upload backup file to S3."""
        try:
            # Use multipart upload for large files
            file_size = local_path.stat().st_size
            
            if file_size > 100 * 1024 * 1024:  # 100MB
                await self._multipart_upload(local_path, s3_key)
            else:
                self.s3_client.upload_file(
                    str(local_path), 
                    self.bucket_name, 
                    s3_key,
                    ExtraArgs={
                        'ServerSideEncryption': 'AES256',
                        'StorageClass': 'STANDARD_IA'
                    }
                )
            
            logger.info(f"Backup uploaded to S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    async def _multipart_upload(self, local_path: Path, s3_key: str):
        """Perform multipart upload for large files."""
        chunk_size = 100 * 1024 * 1024  # 100MB chunks
        
        multipart_upload = self.s3_client.create_multipart_upload(
            Bucket=self.bucket_name,
            Key=s3_key,
            ServerSideEncryption='AES256',
            StorageClass='STANDARD_IA'
        )
        
        upload_id = multipart_upload['UploadId']
        parts = []
        
        try:
            with open(local_path, 'rb') as f:
                part_number = 1
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    
                    response = self.s3_client.upload_part(
                        Bucket=self.bucket_name,
                        Key=s3_key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=data
                    )
                    
                    parts.append({
                        'ETag': response['ETag'],
                        'PartNumber': part_number
                    })
                    
                    part_number += 1
            
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
        except Exception as e:
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id
            )
            raise e
    
    async def download_backup(self, s3_key: str, local_path: Path) -> bool:
        """Download backup from S3."""
        try:
            self.s3_client.download_file(
                self.bucket_name, 
                s3_key, 
                str(local_path)
            )
            logger.info(f"Backup downloaded from S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            return False
    
    def list_backups(self, prefix: str = "") -> List[Dict]:
        """List available backups in S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            backups = []
            for obj in response.get('Contents', []):
                backups.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'modified': obj['LastModified'],
                    'storage_class': obj.get('StorageClass', 'STANDARD')
                })
            
            return backups
            
        except ClientError as e:
            logger.error(f"Failed to list S3 backups: {e}")
            return []

class DatabaseBackupManager:
    """Manages database backups."""
    
    def __init__(self):
        """Initialize database backup manager."""
        self.engine = create_engine(settings.database_url)
    
    async def backup_database(self, backup_path: Path) -> bool:
        """Create database backup."""
        try:
            # For PostgreSQL
            if "postgresql" in settings.database_url:
                return await self._backup_postgresql(backup_path)
            # For SQLite
            elif "sqlite" in settings.database_url:
                return await self._backup_sqlite(backup_path)
            else:
                logger.error("Unsupported database type for backup")
                return False
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    async def _backup_postgresql(self, backup_path: Path) -> bool:
        """Backup PostgreSQL database."""
        import subprocess
        
        # Extract connection parameters
        db_url_parts = settings.database_url.replace("postgresql://", "").split("@")
        user_pass = db_url_parts[0].split(":")
        host_db = db_url_parts[1].split("/")
        
        env = os.environ.copy()
        env['PGPASSWORD'] = user_pass[1]
        
        cmd = [
            "pg_dump",
            "-h", host_db[0].split(":")[0],
            "-p", host_db[0].split(":")[1] if ":" in host_db[0] else "5432",
            "-U", user_pass[0],
            "-d", host_db[1],
            "--no-password",
            "--format=custom",
            "--compress=9",
            "--file", str(backup_path)
        ]
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"PostgreSQL backup created: {backup_path}")
            return True
        else:
            logger.error(f"pg_dump failed: {result.stderr}")
            return False
    
    async def _backup_sqlite(self, backup_path: Path) -> bool:
        """Backup SQLite database."""
        db_path = settings.database_url.replace("sqlite:///", "")
        
        try:
            shutil.copy2(db_path, backup_path)
            logger.info(f"SQLite backup created: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            return False

class BackupScheduler:
    """Manages backup scheduling and execution."""
    
    def __init__(self):
        """Initialize backup scheduler."""
        self.encryption_manager = EncryptionManager()
        self.s3_storage = S3BackupStorage()
        self.db_backup_manager = DatabaseBackupManager()
        self.backup_metadata: Dict[str, BackupMetadata] = {}
        self.backup_root = Path(settings.backup_local_path)
        self.backup_root.mkdir(parents=True, exist_ok=True)
    
    async def create_backup(
        self, 
        backup_type: BackupType = BackupType.FULL,
        include_database: bool = True,
        include_files: bool = True
    ) -> Optional[BackupMetadata]:
        """Create a new backup."""
        backup_id = f"{backup_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(exist_ok=True)
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=datetime.now(),
            size_bytes=0,
            file_count=0,
            checksum="",
            encryption_key_id="default",
            retention_date=self._calculate_retention_date(backup_type),
            status=BackupStatus.RUNNING
        )
        
        try:
            await audit_logger.log_event(
                "backup_started",
                {"backup_id": backup_id, "type": backup_type.value}
            )
            
            # Backup database
            if include_database:
                db_backup_path = backup_dir / "database.dump"
                if not await self.db_backup_manager.backup_database(db_backup_path):
                    metadata.status = BackupStatus.FAILED
                    metadata.error_message = "Database backup failed"
                    return metadata
                
                # Encrypt database backup
                encrypted_db_path = backup_dir / "database.dump.enc"
                self.encryption_manager.encrypt_file(db_backup_path, encrypted_db_path)
                db_backup_path.unlink()  # Remove unencrypted version
            
            # Backup files
            if include_files:
                await self._backup_files(backup_dir, backup_type)
            
            # Create compressed archive
            archive_path = self.backup_root / f"{backup_id}.tar.gz"
            await self._create_compressed_archive(backup_dir, archive_path)
            
            # Calculate metadata
            metadata.size_bytes = archive_path.stat().st_size
            metadata.file_count = sum(1 for _ in backup_dir.rglob("*") if _.is_file())
            metadata.checksum = self._calculate_file_checksum(archive_path)
            
            # Upload to S3
            s3_key = f"backups/{backup_id}.tar.gz"
            if await self.s3_storage.upload_backup(archive_path, s3_key):
                metadata.status = BackupStatus.COMPLETED
                
                # Verify backup integrity
                if await self._verify_backup_integrity(archive_path, metadata):
                    metadata.compliance_verified = True
                
                # Clean up local files
                shutil.rmtree(backup_dir)
                archive_path.unlink()
                
                await audit_logger.log_event(
                    "backup_completed",
                    {
                        "backup_id": backup_id,
                        "size_bytes": metadata.size_bytes,
                        "file_count": metadata.file_count
                    }
                )
            else:
                metadata.status = BackupStatus.FAILED
                metadata.error_message = "S3 upload failed"
            
            self.backup_metadata[backup_id] = metadata
            return metadata
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            
            await audit_logger.log_event(
                "backup_failed",
                {
                    "backup_id": backup_id,
                    "error": str(e)
                }
            )
            
            return metadata
    
    async def _backup_files(self, backup_dir: Path, backup_type: BackupType):
        """Backup file system data."""
        source_dirs = [
            Path(settings.data_storage_path),
            Path(settings.model_storage_path),
            Path(settings.logs_path)
        ]
        
        for source_dir in source_dirs:
            if source_dir.exists():
                dest_dir = backup_dir / source_dir.name
                dest_dir.mkdir(exist_ok=True)
                
                if backup_type == BackupType.FULL:
                    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
                elif backup_type == BackupType.INCREMENTAL:
                    await self._incremental_copy(source_dir, dest_dir)
    
    async def _incremental_copy(self, source_dir: Path, dest_dir: Path):
        """Copy only files modified since last backup."""
        last_backup_time = self._get_last_backup_time()
        
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime > last_backup_time:
                    rel_path = file_path.relative_to(source_dir)
                    dest_file = dest_dir / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_file)
    
    def _get_last_backup_time(self) -> datetime:
        """Get timestamp of last successful backup."""
        completed_backups = [
            meta for meta in self.backup_metadata.values()
            if meta.status == BackupStatus.COMPLETED
        ]
        
        if completed_backups:
            return max(backup.timestamp for backup in completed_backups)
        else:
            return datetime.min
    
    async def _create_compressed_archive(self, source_dir: Path, archive_path: Path):
        """Create compressed archive."""
        import tarfile
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source_dir, arcname=source_dir.name)
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _verify_backup_integrity(self, archive_path: Path, metadata: BackupMetadata) -> bool:
        """Verify backup integrity."""
        # Verify checksum
        calculated_checksum = self._calculate_file_checksum(archive_path)
        if calculated_checksum != metadata.checksum:
            logger.error("Backup checksum mismatch")
            return False
        
        # Test archive extraction
        try:
            import tarfile
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path="/tmp/backup_test", members=tar.getmembers()[:5])  # Test first 5 files
            shutil.rmtree("/tmp/backup_test", ignore_errors=True)
            return True
        except Exception as e:
            logger.error(f"Backup integrity test failed: {e}")
            return False
    
    def _calculate_retention_date(self, backup_type: BackupType) -> datetime:
        """Calculate retention date based on backup type and compliance requirements."""
        # HIPAA requires 6 years minimum retention for medical records
        base_retention_years = 7  # Extra year for safety
        
        if backup_type == BackupType.FULL:
            return datetime.now() + timedelta(days=365 * base_retention_years)
        elif backup_type == BackupType.INCREMENTAL:
            return datetime.now() + timedelta(days=365 * 2)  # 2 years for incremental
        else:
            return datetime.now() + timedelta(days=365 * base_retention_years)
    
    async def restore_backup(self, backup_id: str, restore_path: Path) -> bool:
        """Restore backup from storage."""
        try:
            # Download from S3
            s3_key = f"backups/{backup_id}.tar.gz"
            local_archive = self.backup_root / f"{backup_id}.tar.gz"
            
            if not await self.s3_storage.download_backup(s3_key, local_archive):
                return False
            
            # Extract archive
            import tarfile
            with tarfile.open(local_archive, "r:gz") as tar:
                tar.extractall(path=restore_path)
            
            # Decrypt files
            for encrypted_file in restore_path.rglob("*.enc"):
                decrypted_file = encrypted_file.with_suffix("")
                if self.encryption_manager.decrypt_file(encrypted_file, decrypted_file):
                    encrypted_file.unlink()
                else:
                    logger.error(f"Failed to decrypt: {encrypted_file}")
                    return False
            
            # Clean up
            local_archive.unlink()
            
            await audit_logger.log_event(
                "backup_restored",
                {"backup_id": backup_id, "restore_path": str(restore_path)}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Backup restore failed: {e}")
            return False
    
    async def cleanup_expired_backups(self):
        """Clean up expired backups based on retention policy."""
        current_time = datetime.now()
        
        for backup_id, metadata in list(self.backup_metadata.items()):
            if current_time > metadata.retention_date:
                # Archive to glacier or delete
                s3_key = f"backups/{backup_id}.tar.gz"
                
                try:
                    # Move to glacier storage class
                    self.s3_storage.s3_client.copy_object(
                        Bucket=self.s3_storage.bucket_name,
                        Key=f"archived/{backup_id}.tar.gz",
                        CopySource={'Bucket': self.s3_storage.bucket_name, 'Key': s3_key},
                        StorageClass='GLACIER'
                    )
                    
                    # Delete original
                    self.s3_storage.s3_client.delete_object(
                        Bucket=self.s3_storage.bucket_name,
                        Key=s3_key
                    )
                    
                    metadata.status = BackupStatus.ARCHIVED
                    
                    await audit_logger.log_event(
                        "backup_archived",
                        {"backup_id": backup_id}
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to archive backup {backup_id}: {e}")
    
    def get_backup_status(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get status of a specific backup."""
        return self.backup_metadata.get(backup_id)
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all backup metadata."""
        return list(self.backup_metadata.values())

# Global backup scheduler instance
backup_scheduler = BackupScheduler()