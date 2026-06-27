"""
=============================================================
DATABASE.PY
Project: Tomato Leaf Disease Detection
Database: MySQL
=============================================================

SQL to create tables (run once in MySQL):

CREATE DATABASE tomato_cnn;
USE tomato_cnn;

CREATE TABLE farmers (
    id        INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    username  VARCHAR(50)  NOT NULL UNIQUE,
    password  VARCHAR(64)  NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    id               INT AUTO_INCREMENT PRIMARY KEY,
    farmer_id        INT NOT NULL,
    image_name       VARCHAR(255),
    predicted_disease VARCHAR(100),
    confidence       FLOAT,
    treatment        TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE
);

CREATE TABLE feedback (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT NOT NULL,
    farmer_id     INT NOT NULL,
    is_correct    TINYINT(1) DEFAULT NULL,
    rating        INT DEFAULT NULL,
    comment       TEXT DEFAULT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
    FOREIGN KEY (farmer_id)     REFERENCES farmers(id) ON DELETE CASCADE,
    UNIQUE KEY unique_feedback (prediction_id, farmer_id)
);

CREATE TABLE transactions (
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
);

CREATE TABLE subscriptions (
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
);

CREATE TABLE notification_settings (
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
);

CREATE TABLE free_trials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    farmer_id INT NOT NULL UNIQUE,
    scans_used INT DEFAULT 0,
    max_scans INT DEFAULT 10,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (farmer_id) REFERENCES farmers(id) ON DELETE CASCADE,
    INDEX idx_free_trials_farmer (farmer_id)
);
=============================================================
"""

import mysql.connector
import os


# ─────────────────────────────────────────────
# DATABASE CONFIGURATION
# ─────────────────────────────────────────────

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'root123'),
    'database': os.environ.get('DB_NAME', 'tomato_cnn'),
    'port': int(os.environ.get('DB_PORT', 3306)),  # ← FIXED: Convert to int
}


def get_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"[Database] Connection error: {e}")
        return None


def save_prediction(farmer_id, image_name, predicted_disease, confidence, treatment):
    conn = get_connection()
    if not conn:
        return None
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
    print(f"[Database] Saved prediction ID={pid} for farmer_id={farmer_id}")
    return pid


def get_recent_predictions(farmer_id, limit=5):
    conn = get_connection()
    if not conn:
        return []
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


def get_all_predictions(farmer_id):
    conn = get_connection()
    if not conn:
        return []
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM predictions WHERE farmer_id=%s ORDER BY created_at DESC",
        (farmer_id,)
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def save_feedback(prediction_id, farmer_id, is_correct, rating=None, comment=None):
    """Save or update feedback for a prediction."""
    conn = get_connection()
    if not conn:
        return False
    
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
    return True


def get_feedback(prediction_id, farmer_id):
    """Get feedback for a specific prediction by a specific farmer."""
    conn = get_connection()
    if not conn:
        return None
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM feedback WHERE prediction_id = %s AND farmer_id = %s",
        (prediction_id, farmer_id)
    )
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result


def get_feedback_summary(farmer_id):
    """Get aggregated feedback statistics for a farmer."""
    conn = get_connection()
    if not conn:
        return {
            'total': 0, 'correct': 0, 'incorrect': 0,
            'correct_pct': 0, 'incorrect_pct': 0, 'avg_rating': 0
        }
    
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


def get_predictions_with_feedback(farmer_id, limit=None):
    """Get predictions joined with feedback data."""
    conn = get_connection()
    if not conn:
        return []
    
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


def create_free_trials_table():
    """Create the free_trials table if it doesn't exist."""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
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
        conn.commit()
        cursor.close()
        conn.close()
        print("[Database] Free trials table ready")
        return True
    except Exception as e:
        print(f"[Database] Error creating free_trials table: {e}")
        return False


def setup_all_tables():
    """Setup all database tables."""
    create_free_trials_table()


# ── Email Settings Functions ──

def get_farmer_email(farmer_id):
    """Get farmer's email from database."""
    conn = get_connection()
    if not conn:
        return None
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT full_name, email FROM farmers WHERE id=%s", (farmer_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return result


def update_notification_settings_with_email(farmer_id, phone1, phone2, email, lang, notify_disease, notify_healthy, notify_weekly, min_severity, send_time):
    """Update notification settings including email."""
    conn = get_connection()
    if not conn:
        return False
    
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
    return True