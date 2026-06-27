#!/usr/bin/env python3
"""
Setup all database tables for TomatoGuard
Run: python setup_all_tables.py
"""

import mysql.connector
import os

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'root123'),
    'database': os.environ.get('DB_NAME', 'tomato_cnn'),
}

def setup_all_tables():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("📦 Creating all tables...")
        
        # 1. Farmers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farmers (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(100) NOT NULL,
                username VARCHAR(50) NOT NULL UNIQUE,
                password VARCHAR(64) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ farmers table ready")
        
        # 2. Predictions table
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
        print("✅ predictions table ready")
        
        # 3. Feedback table
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
        print("✅ feedback table ready")
        
        # 4. Notification settings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notification_settings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                farmer_id INT NOT NULL UNIQUE,
                phone1 VARCHAR(20),
                phone2 VARCHAR(20),
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
        print("✅ notification_settings table ready")
        
        # 5. Transactions table
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
        print("✅ transactions table ready")
        
        # 6. Subscriptions table
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
        print("✅ subscriptions table ready")
        
        # 7. Free trials table
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
        print("✅ free_trials table ready")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n" + "="*50)
        print("✅ ALL TABLES CREATED SUCCESSFULLY!")
        print("="*50)
        print("\n📋 Tables created:")
        print("   - farmers")
        print("   - predictions")
        print("   - feedback")
        print("   - notification_settings")
        print("   - transactions")
        print("   - subscriptions")
        print("   - free_trials")
        print("\n🍅 TomatoGuard is ready to use!")
        
    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    setup_all_tables()