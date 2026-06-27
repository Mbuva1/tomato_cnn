"""
=============================================================
PAYMENT ROUTES - M-Pesa Integration
Project: Tomato Leaf Disease Detection
=============================================================
"""

import os
import json
import base64
import requests
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from database import get_connection

# ── Load Environment Variables ──
from dotenv import load_dotenv
load_dotenv()

payment_bp = Blueprint('payment', __name__, url_prefix='/payment')

# ── Configuration from Environment ──
MPESA_CONSUMER_KEY = os.environ.get('MPESA_CONSUMER_KEY', '')
MPESA_CONSUMER_SECRET = os.environ.get('MPESA_CONSUMER_SECRET', '')
MPESA_PASSKEY = os.environ.get('MPESA_PASSKEY', '')
MPESA_SHORTCODE = os.environ.get('MPESA_SHORTCODE', '174379')
MPESA_CALLBACK_URL = os.environ.get('MPESA_CALLBACK_URL', '')

# Check if M-Pesa is configured
MPESA_CONFIGURED = all([MPESA_CONSUMER_KEY, MPESA_CONSUMER_SECRET, MPESA_PASSKEY])

print(f"[Payment] M-Pesa Configured: {MPESA_CONFIGURED}")
if MPESA_CONFIGURED:
    print(f"[Payment] ✅ M-Pesa is configured and ready")
    print(f"[Payment] Shortcode: {MPESA_SHORTCODE}")
    print(f"[Payment] Callback URL: {MPESA_CALLBACK_URL}")
else:
    print("[Payment] ⚠️ M-Pesa is NOT configured. Please set API keys in .env file")

# ── Plans ──
PLANS = {
    'starter': {
        'name': 'Starter',
        'price': 10,
        'price_kes': 'KSh 10',
        'scans': 50,
        'features': ['All diseases', 'Mobile access', 'Analytics', 'SMS alerts']
    },
    'pro': {
        'name': 'Pro',
        'price': 10,
        'price_kes': 'KSh 10',
        'scans': 'Unlimited',
        'features': ['All diseases', 'Multi-user', 'Advanced analytics', 'Priority support']
    }
}


# ── M-Pesa Helpers ──

def get_mpesa_token():
    """Get OAuth token from M-Pesa API."""
    if not MPESA_CONFIGURED:
        print("[M-Pesa] API keys not configured")
        return None
    
    # CORRECT SANDBOX URL FOR TOKEN
    url = 'https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials'
    auth = base64.b64encode(f"{MPESA_CONSUMER_KEY}:{MPESA_CONSUMER_SECRET}".encode()).decode()
    
    try:
        print(f"[M-Pesa] Getting token...")
        response = requests.get(
            url,
            headers={'Authorization': f'Basic {auth}'},
            timeout=30,
            verify=False  # For testing only
        )
        
        print(f"[M-Pesa] Token Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            if token:
                print("[M-Pesa] ✅ Token obtained successfully")
                return token
            else:
                print(f"[M-Pesa] ❌ No token in response: {data}")
                return None
        else:
            print(f"[M-Pesa] ❌ Token request failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"[M-Pesa] ❌ Token error: {e}")
        return None


def initiate_stk_push(phone_number, amount, account_reference, transaction_desc):
    """
    Initiate M-Pesa STK Push (Lipa Na M-Pesa Online).
    """
    if not MPESA_CONFIGURED:
        return False, {'error': 'M-Pesa is not configured. Please set API keys.'}
    
    token = get_mpesa_token()
    if not token:
        return False, {'error': 'Could not get M-Pesa token. Check your API keys.'}
    
    # Format phone number (254XXXXXXXXX format)
    phone = ''.join(filter(str.isdigit, phone_number))
    if phone.startswith('0'):
        phone = '254' + phone[1:]
    elif not phone.startswith('254'):
        phone = '254' + phone
    
    # Ensure phone is exactly 12 digits (254XXXXXXXXX)
    if len(phone) != 12:
        return False, {'error': f'Invalid phone number format. Should be 12 digits (254XXXXXXXXX)'}
    
    print(f"[M-Pesa] Sending STK Push to: {phone}")
    print(f"[M-Pesa] Amount: {amount}")
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    password = base64.b64encode(
        f"{MPESA_SHORTCODE}{MPESA_PASSKEY}{timestamp}".encode()
    ).decode()
    
    # CORRECT SANDBOX URL FOR STK PUSH
    url = 'https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest'
    
    payload = {
        'BusinessShortCode': MPESA_SHORTCODE,
        'Password': password,
        'Timestamp': timestamp,
        'TransactionType': 'CustomerPayBillOnline',
        'Amount': str(amount),
        'PartyA': phone,
        'PartyB': MPESA_SHORTCODE,
        'PhoneNumber': phone,
        'CallBackURL': MPESA_CALLBACK_URL,
        'AccountReference': account_reference[:12],
        'TransactionDesc': transaction_desc[:20] or 'TomatoGuard Payment',
    }
    
    print(f"[M-Pesa] Sending to URL: {url}")
    print(f"[M-Pesa] Payload: {json.dumps(payload, indent=2)}")
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30,
            verify=False  # For testing only
        )
        
        print(f"[M-Pesa] Response Status: {response.status_code}")
        print(f"[M-Pesa] Response Body: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"[M-Pesa] STK Push Response: {json.dumps(data, indent=2)}")
            
            if data.get('ResponseCode') == '0':
                return True, data
            else:
                error_msg = data.get('errorMessage') or data.get('ResponseDescription') or 'Unknown error'
                return False, {'error': error_msg, 'response': data}
        else:
            return False, {'error': f'HTTP {response.status_code}: {response.text}'}
            
    except requests.exceptions.RequestException as e:
        print(f"[M-Pesa] Request error: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"[M-Pesa] Error Response: {e.response.text}")
        return False, {'error': str(e)}
    except Exception as e:
        print(f"[M-Pesa] STK Push error: {e}")
        return False, {'error': str(e)}


# ── Routes ──

@payment_bp.route('/<plan_id>')
def checkout(plan_id):
    """Show payment checkout page for a plan."""
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    
    plan = PLANS.get(plan_id)
    if not plan:
        return redirect(url_for('landing'))
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute(
        "SELECT * FROM subscriptions WHERE farmer_id=%s AND is_active=1",
        (session['farmer_id'],)
    )
    existing = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return render_template(
        'checkout.html',
        plan=plan,
        plan_id=plan_id,
        existing=existing,
        farmer_name=session.get('farmer_name'),
        mpesa_configured=MPESA_CONFIGURED
    )


@payment_bp.route('/api/initiate', methods=['POST'])
def initiate_payment():
    """API endpoint to initiate M-Pesa payment."""
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    if not MPESA_CONFIGURED:
        return jsonify({
            'error': 'M-Pesa is not configured. Please contact support.',
            'demo_mode': True
        }), 400
    
    data = request.get_json()
    plan_id = data.get('plan_id')
    phone = data.get('phone', '').strip()
    
    if not plan_id or not phone:
        return jsonify({'error': 'Plan and phone number required'}), 400
    
    plan = PLANS.get(plan_id)
    if not plan:
        return jsonify({'error': 'Invalid plan'}), 400
    
    amount = plan['price']
    ref = f"TOMATO-{session['farmer_id']}-{datetime.now().strftime('%y%m%d%H%M')}"
    
    success, response = initiate_stk_push(
        phone_number=phone,
        amount=amount,
        account_reference=ref,
        transaction_desc=f"TomatoGuard {plan['name']}"
    )
    
    if success:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions 
            (farmer_id, plan_id, amount, phone, checkout_request_id, merchant_request_id, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            session['farmer_id'],
            plan_id,
            amount,
            phone,
            response.get('CheckoutRequestID'),
            response.get('MerchantRequestID'),
            'pending'
        ))
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'checkout_request_id': response.get('CheckoutRequestID'),
            'message': '✅ Checkout initiated! Please check your phone for the M-Pesa prompt.'
        })
    
    error_msg = response.get('error') or response.get('errorMessage') or 'Payment initiation failed'
    return jsonify({
        'success': False,
        'error': error_msg
    }), 400


@payment_bp.route('/callback', methods=['POST'])
def callback():
    """M-Pesa callback endpoint — called when payment completes."""
    try:
        data = request.get_json()
        print(f"[M-Pesa Callback] Received: {json.dumps(data, indent=2)}")
        
        body = data.get('Body', {})
        stk_callback = body.get('stkCallback', {})
        checkout_id = stk_callback.get('CheckoutRequestID')
        result_code = stk_callback.get('ResultCode')
        result_desc = stk_callback.get('ResultDesc')
        
        if not checkout_id:
            return jsonify({'success': False, 'error': 'No CheckoutRequestID'}), 400
        
        conn = get_connection()
        cursor = conn.cursor()
        
        if result_code == '0':
            # Success
            cursor.execute("""
                UPDATE transactions 
                SET status = 'completed', result_code = %s, result_desc = %s
                WHERE checkout_request_id = %s
            """, (result_code, result_desc, checkout_id))
            conn.commit()
            
            # Activate subscription
            cursor.execute("""
                SELECT farmer_id, plan_id FROM transactions 
                WHERE checkout_request_id = %s AND status = 'completed'
            """, (checkout_id,))
            trans = cursor.fetchone()
            
            if trans:
                farmer_id, plan_id = trans
                
                expiry = datetime.now() + timedelta(days=30)
                
                cursor.execute("""
                    INSERT INTO subscriptions (farmer_id, plan_id, is_active, expires_at)
                    VALUES (%s, %s, 1, %s)
                    ON DUPLICATE KEY UPDATE
                        plan_id = VALUES(plan_id),
                        is_active = 1,
                        expires_at = VALUES(expires_at),
                        updated_at = CURRENT_TIMESTAMP
                """, (farmer_id, plan_id, expiry))
                conn.commit()
                
                print(f"[Payment] ✅ Subscription activated for farmer {farmer_id} ({plan_id})")
            
        else:
            # Failed
            cursor.execute("""
                UPDATE transactions 
                SET status = 'failed', result_code = %s, result_desc = %s
                WHERE checkout_request_id = %s
            """, (result_code, result_desc, checkout_id))
            conn.commit()
            print(f"[Payment] ❌ Payment failed: {result_desc}")
        
        cursor.close()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"[M-Pesa Callback] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@payment_bp.route('/status/<checkout_id>')
def payment_status(checkout_id):
    """Check payment status."""
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT status, result_desc, created_at
        FROM transactions
        WHERE checkout_request_id = %s AND farmer_id = %s
    """, (checkout_id, session['farmer_id']))
    trans = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not trans:
        return jsonify({'status': 'not_found'})
    
    return jsonify({
        'status': trans['status'],
        'message': trans['result_desc'],
        'created_at': str(trans['created_at'])
    })


@payment_bp.route('/history')
def payment_history():
    """View payment/subscription history."""
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM transactions 
        WHERE farmer_id=%s 
        ORDER BY created_at DESC
        LIMIT 20
    """, (session['farmer_id'],))
    transactions = cursor.fetchall()
    
    cursor.execute("""
        SELECT * FROM subscriptions 
        WHERE farmer_id=%s 
        ORDER BY created_at DESC
    """, (session['farmer_id'],))
    subscriptions = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template(
        'payment_history.html',
        transactions=transactions,
        subscriptions=subscriptions,
        farmer_name=session.get('farmer_name')
    )


@payment_bp.route('/api/test', methods=['POST'])
def test_payment():
    """Test endpoint to send a test STK push."""
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    if not MPESA_CONFIGURED:
        return jsonify({
            'error': 'M-Pesa is not configured. Please set API keys in .env file'
        }), 400
    
    data = request.get_json()
    phone = data.get('phone', '').strip()
    amount = data.get('amount', 1)
    
    if not phone:
        return jsonify({'error': 'Phone number required'}), 400
    
    ref = f"TEST-{datetime.now().strftime('%H%M%S')}"
    
    success, response = initiate_stk_push(
        phone_number=phone,
        amount=amount,
        account_reference=ref,
        transaction_desc="Test Payment"
    )
    
    if success:
        return jsonify({
            'success': True,
            'message': f'✅ Test STK push sent to {phone}',
            'checkout_request_id': response.get('CheckoutRequestID')
        })
    
    return jsonify({
        'success': False,
        'error': response.get('error') or response.get('errorMessage') or 'Test failed'
    }), 400


@payment_bp.route('/api/demo', methods=['POST'])
def demo_subscription():
    """Activate a free demo subscription for testing."""
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    plan_id = data.get('plan_id', 'starter')
    
    expiry = datetime.now() + timedelta(days=7)
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO subscriptions (farmer_id, plan_id, is_active, expires_at)
        VALUES (%s, %s, 1, %s)
        ON DUPLICATE KEY UPDATE
            plan_id = VALUES(plan_id),
            is_active = 1,
            expires_at = VALUES(expires_at),
            updated_at = CURRENT_TIMESTAMP
    """, (session['farmer_id'], plan_id, expiry))
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({
        'success': True, 
        'message': '✅ Demo subscription activated for 7 days!'
    })


# ── Database setup ──

def setup_payment_tables():
    """Create tables needed for payment system."""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
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
        
        conn.commit()
        cursor.close()
        conn.close()
        print("[Database] Payment tables created/verified")
        return True
    except Exception as e:
        print(f"[Database] Error creating payment tables: {e}")
        return False


def register_payment_routes(app):
    """Register payment blueprint with the Flask app."""
    app.register_blueprint(payment_bp)
    setup_payment_tables()
    
    if MPESA_CONFIGURED:
        print("[Payment] ✅ M-Pesa is configured and ready")
    else:
        print("[Payment] ⚠️ M-Pesa is NOT configured. Set API keys in .env file")
    
    print("[Payment] Routes registered")


if __name__ == "__main__":
    setup_payment_tables()
    print("Payment module ready ✓")