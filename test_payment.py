"""
Test M-Pesa Payment - With Your Number
Run: python test_payment.py
"""

import os
import base64
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("TESTING M-PESA PAYMENT")
print("=" * 60)

# Get credentials
CONSUMER_KEY = os.environ.get('MPESA_CONSUMER_KEY', '')
CONSUMER_SECRET = os.environ.get('MPESA_CONSUMER_SECRET', '')
PASSKEY = os.environ.get('MPESA_PASSKEY', '')
SHORTCODE = os.environ.get('MPESA_SHORTCODE', '174379')
CALLBACK_URL = os.environ.get('MPESA_CALLBACK_URL', '')

print(f"✓ Consumer Key: {CONSUMER_KEY[:10]}... (length: {len(CONSUMER_KEY)})")
print(f"✓ Consumer Secret: {CONSUMER_SECRET[:10]}... (length: {len(CONSUMER_SECRET)})")
print(f"✓ Passkey: {PASSKEY[:15]}... (length: {len(PASSKEY)})")
print(f"✓ Shortcode: {SHORTCODE}")
print(f"✓ Callback URL: {CALLBACK_URL}")

print("\n" + "=" * 60)
print("STEP 1: Getting Token...")
print("=" * 60)

# Get token
url = 'https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials'
auth = base64.b64encode(f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode()).decode()

try:
    response = requests.get(
        url,
        headers={'Authorization': f'Basic {auth}'},
        timeout=30
    )
    
    print(f"Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        token = data.get('access_token')
        if token:
            print("✅ TOKEN OBTAINED SUCCESSFULLY!")
            
            print("\n" + "=" * 60)
            print("STEP 2: Sending STK Push to YOUR Phone...")
            print("=" * 60)
            
            # Send STK Push - USE YOUR NUMBER HERE
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            password = base64.b64encode(
                f"{SHORTCODE}{PASSKEY}{timestamp}".encode()
            ).decode()
            
            # --- CHANGE THIS TO YOUR NUMBER ---
            phone = '0706857508'  # Your number
            
            print(f"📱 Phone: {phone}")
            print(f"💰 Amount: 1 KES (Test)")
            print(f"⏰ Timestamp: {timestamp}")
            
            payload = {
                'BusinessShortCode': SHORTCODE,
                'Password': password,
                'Timestamp': timestamp,
                'TransactionType': 'CustomerPayBillOnline',
                'Amount': '1',
                'PartyA': phone,
                'PartyB': SHORTCODE,
                'PhoneNumber': phone,
                'CallBackURL': CALLBACK_URL,
                'AccountReference': 'TEST001',
                'TransactionDesc': 'Test Payment',
            }
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            stk_url = 'https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest'
            
            print(f"\n📤 Sending request to M-Pesa...")
            
            response = requests.post(
                stk_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
                
                if data.get('ResponseCode') == '0':
                    print("\n" + "=" * 60)
                    print("🎉 SUCCESS! STK Push sent to your phone!")
                    print("=" * 60)
                    print(f"📋 CheckoutRequestID: {data.get('CheckoutRequestID')}")
                    print("\n📱 CHECK YOUR PHONE NOW!")
                    print("   You should receive an M-Pesa prompt.")
                    print("   Enter PIN to complete the test.")
                else:
                    print(f"\n❌ STK Push failed: {data.get('errorMessage', data)}")
            else:
                print(f"\n❌ HTTP Error: {response.text}")
                
        else:
            print("❌ No token in response")
    else:
        print(f"❌ Token request failed: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")