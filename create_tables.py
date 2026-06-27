#!/usr/bin/env python3
"""
Quick script to create database tables
Run: python create_tables.py
"""

import mysql.connector

# Update these with your credentials
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root123',
    'database': 'tomato_cnn',
}

def create_tables():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Run all SQL commands
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
        print("✅ transactions table created")

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
        print("✅ subscriptions table created")

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
        print("✅ notification_settings table created")

        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n✅ All tables created successfully!")
        print("\n📋 Tables created:")
        print("   - transactions")
        print("   - subscriptions")
        print("   - notification_settings")
        
    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_tables()