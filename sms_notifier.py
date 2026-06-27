"""
SMS Notification Module for TomatoGuard
Africa's Talking — simplified, farmer-friendly messages
"""

import os
import requests
from dotenv import load_dotenv
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

USERNAME  = os.getenv("AFRICASTALKING_USERNAME", "sandbox")
API_KEY   = os.getenv("AFRICASTALKING_API_KEY")
SENDER_ID = os.getenv("AFRICASTALKING_SENDER_ID")

BASE_URL = (
    "https://api.sandbox.africastalking.com/version1/messaging"
    if USERNAME == "sandbox"
    else "https://api.africastalking.com/version1/messaging"
)

if not API_KEY:
    raise ValueError("AFRICASTALKING_API_KEY is not set in your .env file.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def format_phone(phone: str) -> str:
    phone = ''.join(filter(str.isdigit, phone))
    if phone.startswith('254') and len(phone) == 12:
        pass
    elif phone.startswith('0'):
        phone = '254' + phone[1:]
    elif phone.startswith('7') or phone.startswith('1'):
        phone = '254' + phone
    return '+' + phone


def _send(message: str, phone: str):
    """Send SMS via Africa's Talking REST API."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "apiKey": API_KEY
    }
    payload = {"username": USERNAME, "to": phone, "message": message}
    if USERNAME != "sandbox" and SENDER_ID and SENDER_ID.strip():
        payload["from"] = SENDER_ID.strip()

    try:
        response = requests.post(BASE_URL, headers=headers, data=payload,
                                 timeout=30, verify=False)
        response.raise_for_status()
        data = response.json()

        recipients = data.get("SMSMessageData", {}).get("Recipients", [])
        if recipients:
            status = recipients[0].get("status", "")
            if status == "UserInBlacklist":
                return False, "Blocked by Safaricom DND. Ask recipient to disable marketing blocks via *100#."
            if status == "InvalidSenderId":
                return False, "Invalid Sender ID."

        return True, data

    except requests.exceptions.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# ── Public API ─────────────────────────────────────────────────────────────────

def send_disease_alert(phone_number, disease, severity_level, treatment, lang='en'):
    """
    Short, clean disease alert SMS.
    Example (EN):
        🍅 TomatoGuard
        Disease: Early Blight (🟡 Moderate)
        Action: Remove infected leaves, apply fungicide.

    Example (SW):
        🍅 TomatoGuard
        Ugonjwa: Uozu wa Mapema (🟡 Wastani)
        Tibu: Ondoa majani yaliyoathiriwa, tumia dawa ya kuua kuvu.
    """
    sev_icons = {'Mild': '🟢', 'Moderate': '🟡', 'Severe': '🔴',
                 'Kidogo': '🟢', 'Ya Wastani': '🟡', 'Kali Sana': '🔴'}
    icon = sev_icons.get(severity_level, '⚠️')

    # Trim treatment to one short sentence
    short_treatment = treatment.split('.')[0].strip() + '.'

    if lang == 'sw':
        message = (
            f"🍅 TomatoGuard\n"
            f"Ugonjwa: {disease} ({icon} {severity_level})\n"
            f"Tibu: {short_treatment}"
        )
    else:
        message = (
            f"🍅 TomatoGuard\n"
            f"Disease: {disease} ({icon} {severity_level})\n"
            f"Action: {short_treatment}"
        )

    return _send(message, format_phone(phone_number))


def send_healthy_alert(phone_number, lang='en'):
    """Short confirmation that the scan showed a healthy plant."""
    if lang == 'sw':
        message = "🍅 TomatoGuard\n✅ Mmea wako wa nyanya una afya. Endelea kufuatilia."
    else:
        message = "🍅 TomatoGuard\n✅ Your tomato plant is healthy. Keep monitoring."
    return _send(message, format_phone(phone_number))


def send_weekly_summary(phone_number, total, healthy, diseased, top_disease, lang='en'):
    """Brief weekly digest SMS."""
    if lang == 'sw':
        message = (
            f"🍅 TomatoGuard — Wiki hii\n"
            f"Uchunguzi: {total} | Afya: {healthy} | Ugonjwa: {diseased}\n"
            f"Ugonjwa wa kawaida: {top_disease}"
        )
    else:
        message = (
            f"🍅 TomatoGuard — This week\n"
            f"Scans: {total} | Healthy: {healthy} | Diseased: {diseased}\n"
            f"Top: {top_disease}"
        )
    return _send(message, format_phone(phone_number))


def test_sms_connection(phone_number):
    """Send a minimal test SMS."""
    message = "🍅 TomatoGuard — SMS is working correctly."
    return _send(message, format_phone(phone_number))


# ── Test Runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("TomatoGuard SMS Test")
    print("-" * 30)
    number = input("Enter phone number (e.g. 0712345678): ").strip()
    success, response = test_sms_connection(number)
    if success:
        print("\n✅ Sent successfully!")
    else:
        print("\n❌ Failed:", response)