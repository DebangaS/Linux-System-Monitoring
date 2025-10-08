"""
Authentication routes
Author: Member 1
"""

from flask import Blueprint, request, jsonify, session, current_app
from functools import wraps
import re
from .models import User, Session
from database.base import DatabaseConnection  # Needed for list_users()

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    return True, "Password is valid"


def login_required(f):
    """Decorator to require authentication"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'Authentication required'}), 401

        user_session = Session.get_session(session_id)
        if not user_session:
            session.clear()
            return jsonify({'error': 'Invalid or expired session'}), 401

        # Get user and add to request context
        user = User.get_by_id(user_session.user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 401

        request.current_user = user
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    """Decorator to require admin role"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(request, 'current_user'):
            return jsonify({'error': 'Authentication required'}), 401
        if request.current_user.role != 'admin':
            return jsonify({'error': 'Admin privileges required'}), 403
        return f(*args, **kwargs)

    return decorated_function


@auth_bp.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400

        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']

        # Validate input
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        is_valid, message = validate_password(password)
        if not is_valid:
            return jsonify({'error': message}), 400

        # Check if user exists
        existing_user = User.get_by_username(username)
        if existing_user:
            return jsonify({'error': 'Username already exists'}), 409

        # Create user
        allow_admin = current_app.config.get('ALLOW_ADMIN_REGISTRATION', False)
        role = 'admin' if data.get('is_admin') and allow_admin else 'user'

        user = User.create_user(username, email, password, role)
        current_app.logger.info(f"New user registered: {username}")

        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict()
        }), 201

    except Exception as e:
        current_app.logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400

        # Get user
        user = User.get_by_username(username)
        if not user or not user.verify_password(password):
            return jsonify({'error': 'Invalid username or password'}), 401

        # Create session
        user_session = Session.create_session(user.user_id)
        session['session_id'] = user_session.session_id
        session.permanent = True

        # Update last login
        user.update_last_login()
        current_app.logger.info(f"User logged in: {username}")

        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'session_id': user_session.session_id
        })

    except Exception as e:
        current_app.logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """User logout endpoint"""
    try:
        session_id = session.get('session_id')
        if session_id:
            user_session = Session.get_session(session_id)
            if user_session:
                user_session.invalidate()
        session.clear()
        current_app.logger.info("User logged out")
        return jsonify({'message': 'Logout successful'})

    except Exception as e:
        current_app.logger.error(f"Logout error: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500


@auth_bp.route('/profile', methods=['GET'])
@login_required
def profile():
    """Get user profile"""
    return jsonify({
        'user': request.current_user.to_dict()
    })


@auth_bp.route('/users', methods=['GET'])
@login_required
@admin_required
def list_users():
    """List all users (admin only)"""
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE is_active = 1 ORDER BY created_at DESC')
            users = []
            for row in cursor.fetchall():
                user = User(*row)
                users.append(user.to_dict())
        return jsonify({'users': users})

    except Exception as e:
        current_app.logger.error(f"List users error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve users'}), 500


@auth_bp.route('/cleanup-sessions', methods=['POST'])
@login_required
@admin_required
def cleanup_sessions():
    """Clean up expired sessions (admin only)"""
    try:
        count = Session.cleanup_expired_sessions()
        current_app.logger.info(f"Cleaned up {count} expired sessions")
        return jsonify({
            'message': f'Cleaned up {count} expired sessions'
        })
    except Exception as e:
        current_app.logger.error(f"Session cleanup error: {str(e)}")
        return jsonify({'error': 'Session cleanup failed'}), 500
