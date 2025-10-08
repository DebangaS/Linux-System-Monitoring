"""
User authentication models
Author: Member 1
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from database.base import DatabaseConnection


class User:
    """User model for authentication"""

    def __init__(self, user_id=None, username=None, email=None,
                 password_hash=None, created_at=None, last_login=None,
                 is_active=True, role='user'):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at or datetime.utcnow()
        self.last_login = last_login
        self.is_active = is_active
        self.role = role  # 'admin', 'user', 'viewer'

    @staticmethod
    def hash_password(password):
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return salt + password_hash.hex()

    def verify_password(self, password):
        """Verify password against stored hash"""
        if not self.password_hash:
            return False
        salt = self.password_hash[:64]
        stored_hash = self.password_hash[64:]
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return password_hash.hex() == stored_hash

    @classmethod
    def create_user(cls, username, email, password, role='user'):
        """Create new user"""
        password_hash = cls.hash_password(password)
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO users (username, email, password_hash, role, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (username, email, password_hash, role, datetime.utcnow(), True)
            )
            user_id = cursor.lastrowid
        return cls(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role
        )

    @classmethod
    def get_by_username(cls, username):
        """Get user by username"""
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM users WHERE username = ? AND is_active = 1',
                (username,)
            )
            row = cursor.fetchone()
        if row:
            return cls(*row)
        return None

    @classmethod
    def get_by_id(cls, user_id):
        """Get user by ID"""
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM users WHERE user_id = ? AND is_active = 1',
                (user_id,)
            )
            row = cursor.fetchone()
        if row:
            return cls(*row)
        return None

    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET last_login = ? WHERE user_id = ?',
                (self.last_login, self.user_id)
            )

    def has_permission(self, permission):
        """Check if user has specific permission"""
        role_permissions = {
            'admin': ['read', 'write', 'delete', 'manage_users', 'system_config'],
            'user': ['read', 'write'],
            'viewer': ['read']
        }
        return permission in role_permissions.get(self.role, [])

    def to_dict(self):
        """Convert user to dictionary (without sensitive data)"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
        }


class Session:
    """User session model"""

    def __init__(self, session_id=None, user_id=None,
                 created_at=None, expires_at=None, is_active=True):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = created_at or datetime.utcnow()
        self.expires_at = expires_at or (datetime.utcnow() + timedelta(hours=24))
        self.is_active = is_active

    @classmethod
    def create_session(cls, user_id):
        """Create new session"""
        session_id = secrets.token_urlsafe(32)
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO sessions (session_id, user_id, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (session_id, user_id, datetime.utcnow(),
                 datetime.utcnow() + timedelta(hours=24), True)
            )
        return cls(session_id=session_id, user_id=user_id)

    @classmethod
    def get_session(cls, session_id):
        """Get session by ID"""
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT * FROM sessions
                WHERE session_id = ? AND is_active = 1 AND expires_at > ?
                ''',
                (session_id, datetime.utcnow())
            )
            row = cursor.fetchone()
        if row:
            return cls(*row)
        return None

    def invalidate(self):
        """Invalidate session"""
        self.is_active = False
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE sessions SET is_active = 0 WHERE session_id = ?',
                (self.session_id,)
            )

    @classmethod
    def cleanup_expired_sessions(cls):
        """Clean up expired sessions"""
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM sessions WHERE expires_at < ?',
                (datetime.utcnow(),)
            )
            return cursor.rowcount
