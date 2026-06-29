#!/usr/bin/env python3
"""
Add missing columns to Railway MySQL database
Run: python add_columns.py
"""

import mysql.connector
import os

# ── Railway MySQL Config ──
config = {
    'host': 'reseau.proxy.rlwy.net',
    'user': 'root',
    'password': '',  # ← REPLACE THIS
    'database': 'railway',
    'port': 20812,
}

print("🔌 Connecting to Railway MySQL...")

try:
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    print("✅ Connected!\n")

    # ── 1. Add status column ──
    print("📌 Adding status column...")
    try:
        cursor.execute("ALTER TABLE feedback ADD COLUMN status ENUM('pending', 'reviewed', 'resolved', 'closed') DEFAULT 'pending'")
        conn.commit()
        print("✅ Added 'status' column")
    except mysql.connector.Error as e:
        if e.errno == 1060:
            print("ℹ️ 'status' column already exists")
        else:
            print(f"❌ Error: {e}")

    # ── 2. Add response column ──
    print("📌 Adding response column...")
    try:
        cursor.execute("ALTER TABLE feedback ADD COLUMN response TEXT DEFAULT NULL")
        conn.commit()
        print("✅ Added 'response' column")
    except mysql.connector.Error as e:
        if e.errno == 1060:
            print("ℹ️ 'response' column already exists")
        else:
            print(f"❌ Error: {e}")

    # ── 3. Add is_admin column ──
    print("📌 Adding is_admin column...")
    try:
        cursor.execute("ALTER TABLE farmers ADD COLUMN is_admin TINYINT(1) DEFAULT 0")
        conn.commit()
        print("✅ Added 'is_admin' column")
    except mysql.connector.Error as e:
        if e.errno == 1060:
            print("ℹ️ 'is_admin' column already exists")
        else:
            print(f"❌ Error: {e}")

    # ── 4. Add role column ──
    print("📌 Adding role column...")
    try:
        cursor.execute("ALTER TABLE farmers ADD COLUMN role VARCHAR(20) DEFAULT 'farmer'")
        conn.commit()
        print("✅ Added 'role' column")
    except mysql.connector.Error as e:
        if e.errno == 1060:
            print("ℹ️ 'role' column already exists")
        else:
            print(f"❌ Error: {e}")

    # ── 5. Create admin user ──
    print("📌 Creating admin user...")
    try:
        cursor.execute("""
            INSERT INTO farmers (full_name, username, password, email, phone, is_admin, role) 
            VALUES ('Admin', 'admin', 'e00cf25ad42683b3df678c61f42c6bda', 'admin@tomatoguard.com', '+254700000000', 1, 'admin')
            ON DUPLICATE KEY UPDATE is_admin=1, role='admin'
        """)
        conn.commit()
        print("✅ Admin user created/updated (password: admin123)")
    except mysql.connector.Error as e:
        print(f"❌ Error creating admin: {e}")

    # ── 6. Verify ──
    print("\n📌 Verifying...")
    cursor.execute("SELECT id, full_name, username, is_admin, role FROM farmers WHERE username='admin'")
    admin = cursor.fetchone()
    if admin:
        print(f"\n👤 Admin user found:")
        print(f"   ID: {admin[0]}")
        print(f"   Name: {admin[1]}")
        print(f"   Username: {admin[2]}")
        print(f"   is_admin: {admin[3]}")
        print(f"   Role: {admin[4]}")
    else:
        print("⚠️ Admin user not found!")

    cursor.close()
    conn.close()

    print("\n" + "="*50)
    print("✅ ALL DONE!")
    print("="*50)
    print("\n🔑 Login at: https://tomatocnn-production.up.railway.app/admin/login")
    print("   Username: admin")
    print("   Password: admin123")

except mysql.connector.Error as e:
    print(f"❌ Database Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")