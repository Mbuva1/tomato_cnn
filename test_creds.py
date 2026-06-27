"""
Test M-Pesa Credentials - Step by Step
Run: python test_creds.py
"""

import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# Get credentials
CONSUMER_KEY = os.environ.get('MPESA_CONSUMER_KEY', '')
CONSUMER_SECRET = os.environ.get('MPESA_CONSUMER_SECRET', '')
PASSKEY = os.environ.get('MPESA_PASSKEY', '')
SHORTCODE = os.environ.get('MPESA_SHORTCODE', '174379')

print("=" * 60)
print("M-PESA CREDENTIALS TEST")
print("=" * 60)

# Step 1: Check lengths
print("\n[Step 1] Checking credential lengths...")
print(f"  Consumer Key length: {len(CONSUMER_KEY)} characters")
print(f"  Consumer Secret length: {len(CONSUMER_SECRET)} characters")
print(f"  Passkey length: {len(PASSKEY)} characters")

if len(CONSUMER_KEY) < 20:
    print("  ❌ Consumer Key is too short! Should be ~40 characters.")
    print("  You likely copied the truncated version (MeU5***).")
    print("  Go back to the portal and click the eye icon to reveal the full key.")
    exit(1)

if len(CONSUMER_SECRET) < 20:
    print("  ❌ Consumer Secret is too short! Should be ~40 characters.")
    print("  You likely copied the truncated version (o3pK***).")
    print("  Go back to the portal and click the eye icon to reveal the full secret.")
    exit(1)

print("  ✅ Credential lengths look good!")

# Step 2: Test token
print("\n[Step 2] Testing token generation...")
url = 'https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials'
auth = base64.b64encode(f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode()).decode()

try:
    response = requests.get(
        url,
        headers={'Authorization': f'Basic {auth}'},
        timeout=30
    )
    
    print(f"  Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        token = data.get('access_token')
        if token:
            print(f"  ✅ Token obtained successfully!")
            print(f"  Token: {token[:20]}...{token[-10:]}")
            
            # Step 3: Test STK Push
            print("\n[Step 3] Testing STK Push...")
            from datetime import datetime
            import json
            
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            password = base64.b64encode(
                f"{SHORTCODE}{PASSKEY}{timestamp}".encode()
            ).decode()
            
            stk_url = 'https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest'
            
            payload = {
                'BusinessShortCode': SHORTCODE,
                'Password': password,
                'Timestamp': timestamp,
                'TransactionType': 'CustomerPayBillOnline',
                'Amount': '1',
                'PartyA': '254708374149',
                'PartyB': SHORTCODE,
                'PhoneNumber': '254708374149',
                'CallBackURL': 'https://unknown-energetic-gaming.ngrok-free.dev/payment/callback',
                'AccountReference': 'TEST-001',
                'TransactionDesc': 'Test Payment',
            }
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            stk_response = requests.post(
                stk_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            print(f"  Response Status: {stk_response.status_code}")
            
            if stk_response.status_code == 200:
                data = stk_response.json()
                if data.get('ResponseCode') == '0':
                    print("  ✅ STK Push successful!")
                    print(f"  CheckoutRequestID: {data.get('CheckoutRequestID')}")
                else:
                    print(f"  ❌ STK Push failed: {data}")
            else:
                print(f"  ❌ STK Push HTTP Error: {stk_response.text}")
                
        else:
            print(f"  ❌ No token in response: {data}")
    else:
        print(f"  ❌ Token request failed: {response.text}")
        
except Exception as e:
    print(f"  ❌ Error: {e}")