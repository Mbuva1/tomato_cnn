"""
=============================================================
DATABASE.PY
Project: Tomato Leaf Disease Detection
Database: MySQL
=============================================================
"""

import os
import mysql.connector
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DATABASE CONFIGURATION
# ─────────────────────────────────────────────

def get_db_config():
    """
    Get database configuration from environment variables.
    Priority:
    1. MYSQL_PUBLIC_URL (Railway - works from anywhere)
    2. MYSQL_URL (Railway - internal, only works in same project)
    3. Individual MYSQL_* variables (Railway)
    4. Individual DB_* variables (local development)
    """
    # Try MYSQL_PUBLIC_URL FIRST (Railway - works from anywhere)
    mysql_url = os.environ.get('MYSQL_PUBLIC_URL')
    
    if mysql_url:
        logger.info("[Database] Using MYSQL_PUBLIC_URL (public/anywhere)")
    else:
        # Fallback to MYSQL_URL (internal - only works in same project)
        mysql_url = os.environ.get('MYSQL_URL')
        if mysql_url:
            logger.info("[Database] Using MYSQL_URL (internal)")
    
    if mysql_url:
        try:
            parsed = urlparse(mysql_url)
            config = {
                'host': parsed.hostname,
                'user': parsed.username,
                'password': parsed.password,
                'database': parsed.path.lstrip('/'),
                'port': parsed.port or 3306,
            }
            logger.info(f"[Database] Connection config: {config['host']}:{config['port']}")
            return config
        except Exception as e:
            logger.error(f"[Database] Error parsing URL: {e}")
    
    # Fallback to individual variables
    config = {
        'host': os.environ.get('MYSQLHOST') or 'localhost',
        'user': os.environ.get('MYSQLUSER') or 'root',
        'password': os.environ.get('MYSQLPASSWORD') or 'root123',
        'database': os.environ.get('MYSQL_DATABASE') or 'tomato_cnn',
        'port': int(os.environ.get('MYSQLPORT') or 3306),
    }
    
    logger.info(f"[Database] Using individual variables: {config['host']}:{config['port']}")
    return config


def get_connection():
    """Get a database connection."""
    try:
        config = get_db_config()
        logger.info(f"[Database] Connecting to {config['host']}:{config['port']} "
                   f"as {config['user']} to database {config['database']}")
        
        connection = mysql.connector.connect(**config)
        logger.info("[Database] ✅ Connection successful!")
        return connection
        
    except mysql.connector.Error as e:
        error_code = e.errno if hasattr(e, 'errno') else 'Unknown'
        logger.error(f"[Database] ❌ MySQL Error {error_code}: {str(e)}")
        return None
        
    except Exception as e:
        logger.error(f"[Database] Unexpected error: {str(e)}")
        return None


# ─────────────────────────────────────────────
# TABLE SETUP - FIXED: Creates tables in correct order
# ─────────────────────────────────────────────

def setup_all_tables():
    """Setup all database tables in correct order (parents first, then children)."""
    conn = get_connection()
    if not conn:
        logger.error("[Database] Could not connect to setup tables")
        return False
    
    try:
        cursor = conn.cursor()
        
        # ============================================
        # 1. Create farmers table (NO foreign keys - MUST BE FIRST)
        # ============================================
        logger.info("[Database] Creating farmers table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farmers (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(100) NOT NULL,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        logger.info("[Database] ✅ farmers table ready")
        
        # ============================================
        # 2. Create predictions table (references farmers)
        # ============================================
        logger.info("[Database] Creating predictions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                farmer_id INT NOT NULL,
                image_name VARCHAR(255),
                predicted_disease VARCHAR(100),
                confidence FLOAT,
                treatment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE
            )
        """)
        logger.info("[Database] ✅ predictions table ready")
        
        # ============================================
        # 3. Create feedback table (references predictions & farmers)
        # ============================================
        logger.info("[Database] Creating feedback table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                prediction_id INT NOT NULL,
                farmer_id INT NOT NULL,
                is_correct TINYINT(1) DEFAULT NULL,
                rating INT DEFAULT NULL,
                comment TEXT DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE,
                UNIQUE KEY unique_feedback (prediction_id, farmer_id)
            )
        """)
        logger.info("[Database] ✅ feedback table ready")
        
        # ============================================
        # 4. Create transactions table (references farmers)
        # ============================================
        logger.info("[Database] Creating transactions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                farmer_id INT NOT NULL,
                plan_id VARCHAR(20) NOT NULL,
                amount INT NOT NULL,
                phone VARCHAR(20) NOT NULL,
                checkout_request_id VARCHAR(50) UNIQUE,
                merchant_request_id VARCHAR(50),
                status ENUM('pending', 'completed', 'failed') DEFAULT 'pending',
                result_code VARCHAR(10),
                result_desc VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE,
                INDEX idx_transactions_farmer (farmer_id),
                INDEX idx_transactions_checkout (checkout_request_id)
            )
        """)
        logger.info("[Database] ✅ transactions table ready")
        
        # ============================================
        # 5. Create subscriptions table (references farmers)
        # ============================================
        logger.info("[Database] Creating subscriptions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                farmer_id INT NOT NULL UNIQUE,
                plan_id VARCHAR(20) NOT NULL,
                is_active TINYINT(1) DEFAULT 1,
                expires_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE,
                INDEX idx_subscriptions_active (is_active),
                INDEX idx_subscriptions_expiry (expires_at)
            )
        """)
        logger.info("[Database] ✅ subscriptions table ready")
        
        # ============================================
        # 6. Create notification_settings table (references farmers)
        # ============================================
        logger.info("[Database] Creating notification_settings table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notification_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                farmer_id INT NOT NULL UNIQUE,
                phone1 VARCHAR(20),
                phone2 VARCHAR(20),
                email VARCHAR(100) DEFAULT NULL,
                lang VARCHAR(5) DEFAULT 'en',
                notify_disease TINYINT(1) DEFAULT 1,
                notify_healthy TINYINT(1) DEFAULT 0,
                notify_weekly TINYINT(1) DEFAULT 0,
                min_severity VARCHAR(20) DEFAULT 'mild',
                send_time VARCHAR(20) DEFAULT 'instant',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE
            )
        """)
        logger.info("[Database] ✅ notification_settings table ready")
        
        # ============================================
        # 7. Create free_trials table (references farmers)
        # ============================================
        logger.info("[Database] Creating free_trials table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS free_trials (
                id INT AUTO_INCREMENT PRIMARY KEY,
                farmer_id INT NOT NULL UNIQUE,
                scans_used INT DEFAULT 0,
                max_scans INT DEFAULT 10,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE,
                INDEX idx_free_trials_farmer (farmer_id)
            )
        """)
        logger.info("[Database] ✅ free_trials table ready")
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("[Database] ✅ All tables setup complete!")
        return True
        
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error setting up tables: {e}")
        if conn:
            conn.close()
        return False


# ─────────────────────────────────────────────
# DATABASE OPERATIONS
# ─────────────────────────────────────────────

def save_prediction(farmer_id, image_name, predicted_disease, confidence, treatment):
    """Save a prediction to the database."""
    conn = get_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO predictions
               (farmer_id, image_name, predicted_disease, confidence, treatment)
               VALUES (%s, %s, %s, %s, %s)""",
            (farmer_id, image_name, predicted_disease, confidence, treatment)
        )
        conn.commit()
        pid = cursor.lastrowid
        cursor.close()
        conn.close()
        logger.info(f"[Database] Saved prediction ID={pid} for farmer_id={farmer_id}")
        return pid
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error saving prediction: {e}")
        if conn:
            conn.close()
        return None


def get_recent_predictions(farmer_id, limit=5):
    """Get recent predictions for a farmer."""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT * FROM predictions WHERE farmer_id=%s
               ORDER BY created_at DESC LIMIT %s""",
            (farmer_id, limit)
        )
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error getting recent predictions: {e}")
        if conn:
            conn.close()
        return []


def get_all_predictions(farmer_id):
    """Get all predictions for a farmer."""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM predictions WHERE farmer_id=%s ORDER BY created_at DESC",
            (farmer_id,)
        )
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error getting all predictions: {e}")
        if conn:
            conn.close()
        return []


def save_feedback(prediction_id, farmer_id, is_correct, rating=None, comment=None):
    """Save or update feedback for a prediction."""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM feedback WHERE prediction_id = %s AND farmer_id = %s",
            (prediction_id, farmer_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute(
                """UPDATE feedback 
                   SET is_correct = %s, rating = %s, comment = %s
                   WHERE prediction_id = %s AND farmer_id = %s""",
                (is_correct, rating, comment, prediction_id, farmer_id)
            )
        else:
            cursor.execute(
                """INSERT INTO feedback (prediction_id, farmer_id, is_correct, rating, comment)
                   VALUES (%s, %s, %s, %s, %s)""",
                (prediction_id, farmer_id, is_correct, rating, comment)
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"[Database] Saved feedback for prediction_id={prediction_id}")
        return True
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error saving feedback: {e}")
        if conn:
            conn.close()
        return False


def get_feedback(prediction_id, farmer_id):
    """Get feedback for a specific prediction."""
    conn = get_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM feedback WHERE prediction_id = %s AND farmer_id = %s",
            (prediction_id, farmer_id)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error getting feedback: {e}")
        if conn:
            conn.close()
        return None


def get_feedback_summary(farmer_id):
    """Get aggregated feedback statistics for a farmer."""
    conn = get_connection()
    if not conn:
        return {
            'total': 0, 'correct': 0, 'incorrect': 0,
            'correct_pct': 0, 'incorrect_pct': 0, 'avg_rating': 0
        }
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT COUNT(*) AS total FROM feedback WHERE farmer_id = %s",
            (farmer_id,)
        )
        total = cursor.fetchone()['total']
        
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM feedback WHERE farmer_id = %s AND is_correct = 1",
            (farmer_id,)
        )
        correct = cursor.fetchone()['cnt']
        
        cursor.execute(
            "SELECT COUNT(*) AS cnt FROM feedback WHERE farmer_id = %s AND is_correct = 0",
            (farmer_id,)
        )
        incorrect = cursor.fetchone()['cnt']
        
        cursor.execute(
            "SELECT AVG(rating) AS avg_rating FROM feedback WHERE farmer_id = %s AND rating IS NOT NULL",
            (farmer_id,)
        )
        avg_rating = cursor.fetchone()['avg_rating'] or 0
        
        cursor.close()
        conn.close()
        
        correct_pct = round((correct / total * 100), 1) if total > 0 else 0
        incorrect_pct = round((incorrect / total * 100), 1) if total > 0 else 0
        
        return {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'correct_pct': correct_pct,
            'incorrect_pct': incorrect_pct,
            'avg_rating': round(float(avg_rating), 1)
        }
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error getting feedback summary: {e}")
        if conn:
            conn.close()
        return {
            'total': 0, 'correct': 0, 'incorrect': 0,
            'correct_pct': 0, 'incorrect_pct': 0, 'avg_rating': 0
        }


def get_predictions_with_feedback(farmer_id, limit=None):
    """Get predictions joined with feedback data."""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT p.*, f.is_correct, f.rating, f.comment as feedback_comment, f.created_at as feedback_date
            FROM predictions p
            LEFT JOIN feedback f ON p.id = f.prediction_id AND f.farmer_id = p.farmer_id
            WHERE p.farmer_id = %s
            ORDER BY p.created_at DESC
        """
        
        if limit:
            query += " LIMIT %s"
            cursor.execute(query, (farmer_id, limit))
        else:
            cursor.execute(query, (farmer_id,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error getting predictions with feedback: {e}")
        if conn:
            conn.close()
        return []


def get_farmer_email(farmer_id):
    """Get farmer's email from database."""
    conn = get_connection()
    if not conn:
        return None
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT full_name, email FROM farmers WHERE id=%s", (farmer_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error getting farmer email: {e}")
        if conn:
            conn.close()
        return None


def update_notification_settings_with_email(farmer_id, phone1, phone2, email, lang, 
                                           notify_disease, notify_healthy, notify_weekly, 
                                           min_severity, send_time):
    """Update notification settings including email."""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO notification_settings
                (farmer_id, phone1, phone2, email, lang, notify_disease, notify_healthy,
                 notify_weekly, min_severity, send_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                phone1=VALUES(phone1), phone2=VALUES(phone2),
                email=VALUES(email),
                lang=VALUES(lang),
                notify_disease=VALUES(notify_disease),
                notify_healthy=VALUES(notify_healthy),
                notify_weekly=VALUES(notify_weekly),
                min_severity=VALUES(min_severity),
                send_time=VALUES(send_time)
        """, (farmer_id, phone1, phone2, email, lang, notify_disease, notify_healthy,
              notify_weekly, min_severity, send_time))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"[Database] Updated notification settings for farmer_id={farmer_id}")
        return True
    except mysql.connector.Error as e:
        logger.error(f"[Database] Error updating notification settings: {e}")
        if conn:
            conn.close()
        return False


# ─────────────────────────────────────────────
# TEST CONNECTION
# ─────────────────────────────────────────────

def test_connection():
    """Test database connection and print results."""
    logger.info("[Database] Testing connection...")
    conn = get_connection()
    if conn:
        logger.info("✅ Database connection successful!")
        conn.close()
        return True
    else:
        logger.error("❌ Database connection failed!")
        return False


if __name__ == "__main__":
    test_connection()