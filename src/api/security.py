"""
API Security and Authentication Module

Provides comprehensive security features for the medical diagnostic API
including JWT authentication, rate limiting, CORS protection, and
security middleware for HIPAA compliance.

Author: Holistic Diagnostic Platform Team
Version: 1.0.0
"""

import os
import jwt
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from slowapi import Limiter
from slowapi.util import get_remote_address
import redis.asyncio as redis
import json

# Configure logging
logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles with different permission levels."""
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    SYSTEM = "system"

class Permission(Enum):
    """System permissions."""
    READ_STUDIES = "read_studies"
    WRITE_STUDIES = "write_studies"
    DELETE_STUDIES = "delete_studies"
    ANALYZE_IMAGES = "analyze_images"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    CONFIGURE_SYSTEM = "configure_system"

@dataclass
class User:
    """User data model."""
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    session_timeout: int = 3600  # 1 hour
    
class SecurityConfig:
    """Security configuration settings."""
    
    def __init__(self):
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", self._generate_secret_key())
        self.jwt_algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        self.password_min_length = 8
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 15
        self.session_timeout_minutes = 60
        
    @staticmethod
    def _generate_secret_key() -> str:
        """Generate a secure random secret key."""
        return secrets.token_urlsafe(32)

class PasswordManager:
    """Handles password hashing and verification."""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

class JWTManager:
    """Manages JWT token creation and validation."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        self.security = HTTPBearer()
        
    def create_access_token(self, user: User) -> str:
        """Create an access token for a user."""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.config.access_token_expire_minutes)
        
        payload = {
            "sub": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": [perm.value for perm in user.permissions],
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "jti": secrets.token_urlsafe(16)  # JWT ID for revocation
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        
    def create_refresh_token(self, user: User) -> str:
        """Create a refresh token for a user."""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.config.refresh_token_expire_days)
        
        payload = {
            "sub": user.username,
            "type": "refresh",
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
            "jti": secrets.token_urlsafe(16)
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        
    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        token = credentials.credentials
        
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check if token is expired
            if datetime.utcnow().timestamp() > payload.get("exp", 0):
                raise HTTPException(status_code=401, detail="Token has expired")
                
            # Check if token is revoked (implement token blacklist)
            jti = payload.get("jti")
            if jti and await self._is_token_revoked(jti):
                raise HTTPException(status_code=401, detail="Token has been revoked")
                
            return {
                "username": payload.get("sub"),
                "email": payload.get("email"),
                "role": payload.get("role"),
                "permissions": payload.get("permissions", []),
                "jti": jti
            }
            
        except jwt.PyJWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")
            
    async def _is_token_revoked(self, jti: str) -> bool:
        """Check if a token is in the revocation list."""
        # In production, check against Redis or database
        # For now, return False (no revocation)
        return False
        
    async def revoke_token(self, jti: str):
        """Add a token to the revocation list."""
        # In production, add to Redis or database
        pass

class UserManager:
    """Manages user data and authentication."""
    
    def __init__(self):
        self.password_manager = PasswordManager()
        self.redis_client = None
        
        # Default users (in production, load from secure database)
        self.users = {
            "admin": User(
                username="admin",
                email="admin@hospital.com",
                role=UserRole.ADMIN,
                permissions=list(Permission),
                created_at=datetime.utcnow()
            ),
            "analyst": User(
                username="analyst",
                email="analyst@hospital.com",
                role=UserRole.ANALYST,
                permissions=[
                    Permission.READ_STUDIES,
                    Permission.WRITE_STUDIES,
                    Permission.ANALYZE_IMAGES
                ],
                created_at=datetime.utcnow()
            ),
            "viewer": User(
                username="viewer",
                email="viewer@hospital.com",
                role=UserRole.VIEWER,
                permissions=[Permission.READ_STUDIES],
                created_at=datetime.utcnow()
            )
        }
        
        # Default passwords (in production, users set their own)
        self.passwords = {
            "admin": self.password_manager.hash_password("Admin123!"),
            "analyst": self.password_manager.hash_password("Analyst123!"),
            "viewer": self.password_manager.hash_password("Viewer123!")
        }
        
    async def initialize(self, redis_client):
        """Initialize user manager with Redis client."""
        self.redis_client = redis_client
        
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        try:
            # Check if account is locked
            if await self._is_account_locked(username):
                raise HTTPException(
                    status_code=423, 
                    detail="Account is temporarily locked due to too many failed attempts"
                )
                
            # Get user
            user = self.users.get(username)
            if not user or not user.is_active:
                await self._record_failed_attempt(username)
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
            # Verify password
            stored_password = self.passwords.get(username)
            if not stored_password or not self.password_manager.verify_password(password, stored_password):
                await self._record_failed_attempt(username)
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
            # Reset failed attempts on successful login
            await self._reset_failed_attempts(username)
            
            # Update last login
            user.last_login = datetime.utcnow()
            
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}")
            raise HTTPException(status_code=500, detail="Authentication failed")
            
    async def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)
        
    async def _is_account_locked(self, username: str) -> bool:
        """Check if an account is locked due to failed attempts."""
        if not self.redis_client:
            return False
            
        lock_key = f"account_lock:{username}"
        return await self.redis_client.exists(lock_key)
        
    async def _record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        if not self.redis_client:
            return
            
        attempts_key = f"login_attempts:{username}"
        lock_key = f"account_lock:{username}"
        
        # Increment attempt counter
        attempts = await self.redis_client.incr(attempts_key)
        await self.redis_client.expire(attempts_key, 900)  # 15 minutes
        
        # Lock account if too many attempts
        if attempts >= 5:  # MAX_LOGIN_ATTEMPTS
            await self.redis_client.setex(lock_key, 900, "locked")  # 15 minutes lockout
            logger.warning(f"Account {username} locked due to too many failed attempts")
            
    async def _reset_failed_attempts(self, username: str):
        """Reset failed login attempts for a user."""
        if not self.redis_client:
            return
            
        attempts_key = f"login_attempts:{username}"
        await self.redis_client.delete(attempts_key)

class SessionManager:
    """Manages user sessions and access control."""
    
    def __init__(self):
        self.redis_client = None
        self.session_timeout = 3600  # 1 hour
        
    async def initialize(self, redis_client):
        """Initialize session manager with Redis client."""
        self.redis_client = redis_client
        
    async def create_session(self, user: User, jti: str) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            "username": user.username,
            "role": user.role.value,
            "permissions": [perm.value for perm in user.permissions],
            "jti": jti,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        
        if self.redis_client:
            await self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session_data)
            )
            
        return session_id
        
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by session ID."""
        if not self.redis_client:
            return None
            
        session_data = await self.redis_client.get(f"session:{session_id}")
        if session_data:
            return json.loads(session_data)
            
        return None
        
    async def update_session_activity(self, session_id: str):
        """Update last activity timestamp for a session."""
        if not self.redis_client:
            return
            
        session_data = await self.get_session(session_id)
        if session_data:
            session_data["last_activity"] = datetime.utcnow().isoformat()
            await self.redis_client.setex(
                f"session:{session_id}",
                self.session_timeout,
                json.dumps(session_data)
            )
            
    async def revoke_session(self, session_id: str):
        """Revoke a user session."""
        if self.redis_client:
            await self.redis_client.delete(f"session:{session_id}")

class SecurityMiddleware:
    """Security middleware for request processing."""
    
    def __init__(self, security_config: SecurityConfig):
        self.config = security_config
        
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (when behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        return request.client.host if request.client else "unknown"
        
    def validate_request_headers(self, request: Request) -> bool:
        """Validate security-related request headers."""
        # Check for required security headers
        user_agent = request.headers.get("User-Agent")
        if not user_agent or len(user_agent) > 1000:
            return False
            
        # Validate content type for POST requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("Content-Type")
            allowed_types = [
                "application/json",
                "multipart/form-data",
                "application/x-www-form-urlencoded"
            ]
            if content_type and not any(ct in content_type for ct in allowed_types):
                return False
                
        return True
        
    async def check_rate_limit(self, request: Request, redis_client) -> bool:
        """Check rate limiting for the request."""
        if not redis_client:
            return True
            
        client_ip = self.get_client_ip(request)
        endpoint = f"{request.method}:{request.url.path}"
        
        # Different rate limits for different endpoints
        rate_limits = {
            "POST:/auth/token": {"limit": 5, "window": 300},  # 5 per 5 minutes
            "POST:/analyze/": {"limit": 10, "window": 60},    # 10 per minute
            "GET:/": {"limit": 100, "window": 60},            # 100 per minute
        }
        
        # Find matching rate limit
        rate_limit = None
        for pattern, limits in rate_limits.items():
            if endpoint.startswith(pattern):
                rate_limit = limits
                break
                
        if not rate_limit:
            rate_limit = {"limit": 60, "window": 60}  # Default: 60 per minute
            
        # Check rate limit
        key = f"rate_limit:{client_ip}:{endpoint}"
        current = await redis_client.get(key)
        
        if current is None:
            await redis_client.setex(key, rate_limit["window"], 1)
            return True
        elif int(current) < rate_limit["limit"]:
            await redis_client.incr(key)
            return True
        else:
            return False

class PermissionChecker:
    """Handles permission-based access control."""
    
    @staticmethod
    def has_permission(user_permissions: List[str], required_permission: Permission) -> bool:
        """Check if user has required permission."""
        return required_permission.value in user_permissions
        
    @staticmethod
    def has_any_permission(user_permissions: List[str], required_permissions: List[Permission]) -> bool:
        """Check if user has any of the required permissions."""
        return any(perm.value in user_permissions for perm in required_permissions)
        
    @staticmethod
    def has_all_permissions(user_permissions: List[str], required_permissions: List[Permission]) -> bool:
        """Check if user has all required permissions."""
        return all(perm.value in user_permissions for perm in required_permissions)

def require_permission(permission: Permission):
    """Decorator to require specific permission for endpoint access."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user info from kwargs (injected by auth dependency)
            user_info = kwargs.get('user_info')
            if not user_info:
                raise HTTPException(status_code=401, detail="Authentication required")
                
            user_permissions = user_info.get('permissions', [])
            if not PermissionChecker.has_permission(user_permissions, permission):
                raise HTTPException(
                    status_code=403, 
                    detail=f"Permission required: {permission.value}"
                )
                
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(role: UserRole):
    """Decorator to require specific role for endpoint access."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user_info = kwargs.get('user_info')
            if not user_info:
                raise HTTPException(status_code=401, detail="Authentication required")
                
            user_role = user_info.get('role')
            if user_role != role.value:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Role required: {role.value}"
                )
                
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Rate limiting utility
def create_limiter() -> Limiter:
    """Create a rate limiter instance."""
    return Limiter(key_func=get_remote_address)

# Security utility functions
def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def validate_api_key(api_key: str, hashed_key: str) -> bool:
    """Validate an API key against its hash."""
    return hashlib.sha256(api_key.encode()).hexdigest() == hashed_key

# CORS configuration
def get_cors_config():
    """Get CORS configuration for production."""
    return {
        "allow_origins": ["https://yourdomain.com"],  # Configure for production
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["*"],
        "expose_headers": ["X-Request-ID"]
    }

# Security headers
def get_security_headers():
    """Get security headers for responses."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    }