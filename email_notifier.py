"""
=============================================================
EMAIL NOTIFIER MODULE - Resend API (Railway Compatible)
Project: Tomato Leaf Disease Detection
=============================================================
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Resend Configuration ──
RESEND_API_KEY = os.environ.get('RESEND_API_KEY', '')
MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'onboarding@resend.dev')

# Check if email is configured
EMAIL_CONFIGURED = bool(RESEND_API_KEY)

if EMAIL_CONFIGURED:
    print("[Email] ✅ Resend API configured and ready")
else:
    print("[Email] ⚠️ Email is NOT configured. Set RESEND_API_KEY in .env")


def send_email(to_email, subject, html_content, plain_text=None):
    """Send email using Resend API."""
    if not EMAIL_CONFIGURED:
        return False, "Email is not configured"
    
    if not to_email:
        return False, "No email address provided"
    
    try:
        response = requests.post(
            'https://api.resend.com/emails',
            headers={
                'Authorization': f'Bearer {RESEND_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'from': MAIL_DEFAULT_SENDER,
                'to': [to_email],
                'subject': subject,
                'html': html_content,
                'text': plain_text or ''
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"[Email] ✅ Sent to: {to_email}")
            return True, "Email sent successfully"
        else:
            error = response.json().get('error', response.text)
            print(f"[Email] ❌ Error: {error}")
            return False, f"API error: {error}"
            
    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except Exception as e:
        return False, f"Error: {str(e)}"


# ── Disease Alert Email ──

def send_disease_alert_email(to_email, farmer_name, disease, severity, confidence, treatment, lang='en'):
    """Send disease detection alert via email."""
    if not to_email:
        return False, "No email address provided"
    
    severity_info = {
        'Severe': {'color': '#ef5350', 'icon': '🔴'},
        'Moderate': {'color': '#ffca28', 'icon': '🟡'},
        'Mild': {'color': '#66bb6a', 'icon': '🟢'},
    }
    sev_info = severity_info.get(severity, {'color': '#4caf50', 'icon': 'ℹ️'})
    
    current_year = datetime.now().year
    
    if lang == 'sw':
        subject = f"🚨 Ugonjwa Umegunduliwa kwenye Nyanya Yako - TomatoGuard"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family:Arial,sans-serif;background:#f5faf5;padding:20px;">
            <div style="max-width:600px;margin:0 auto;background:white;border-radius:12px;padding:30px;border:1px solid #c8e6c9;">
                <div style="text-align:center;padding-bottom:20px;border-bottom:2px solid #2e7d32;">
                    <div style="font-size:24px;font-weight:bold;color:#2e7d32;">🍅 TomatoGuard</div>
                    <p style="color:#666;">Mfumo wa AI wa Kugundua Magonjwa ya Nyanya</p>
                </div>
                <h2>Habari {farmer_name},</h2>
                <p>Ugonjwa umegunduliwa kwenye mmea wako wa nyanya. Hapa kuna maelezo:</p>
                <div style="background:{sev_info['color']}15;border-left:4px solid {sev_info['color']};padding:15px;margin:20px 0;border-radius:4px;">
                    <div style="font-size:20px;font-weight:bold;color:{sev_info['color']};">{sev_info['icon']} {disease}</div>
                    <p><strong>Ukali:</strong> <span style="display:inline-block;background:{sev_info['color']};color:white;padding:4px 16px;border-radius:20px;font-size:12px;font-weight:bold;">{severity}</span></p>
                    <p><strong>Uhakika:</strong> {confidence}%</p>
                </div>
                <h3>💊 Matibabu Yanayopendekezwa</h3>
                <p style="background:#f8fdf8;padding:12px;border-radius:8px;border-left:3px solid #2e7d32;">{treatment}</p>
                <div style="text-align:center;margin:20px 0;">
                    <a href="https://your-domain.com/dashboard" style="display:inline-block;background:#2e7d32;color:white;padding:10px 24px;text-decoration:none;border-radius:8px;">🔍 Angalia Dashboard</a>
                </div>
                <div style="margin-top:30px;padding-top:20px;border-top:1px solid #e8f5e9;text-align:center;font-size:12px;color:#888;">
                    <p>© {current_year} TomatoGuard. Imeundwa kwa ajili ya wakulima wa nyanya.</p>
                    <p><a href="mailto:support@tomatoguard.com" style="color:#2e7d32;text-decoration:none;">support@tomatoguard.com</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        plain_text = f"""
🚨 Ugonjwa Umegunduliwa!

Habari {farmer_name},

Ugonjwa: {disease}
Ukali: {severity}
Uhakika: {confidence}%

Matibabu:
{treatment}

Angalia dashboard yako kwa maelezo zaidi.

© {current_year} TomatoGuard
        """
    else:
        subject = f"🚨 Disease Detected on Your Tomato Plant - TomatoGuard"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family:Arial,sans-serif;background:#f5faf5;padding:20px;">
            <div style="max-width:600px;margin:0 auto;background:white;border-radius:12px;padding:30px;border:1px solid #c8e6c9;">
                <div style="text-align:center;padding-bottom:20px;border-bottom:2px solid #2e7d32;">
                    <div style="font-size:24px;font-weight:bold;color:#2e7d32;">🍅 TomatoGuard</div>
                    <p style="color:#666;">AI-Powered Tomato Disease Detection System</p>
                </div>
                <h2>Hello {farmer_name},</h2>
                <p>A disease has been detected on your tomato plant. Here are the details:</p>
                <div style="background:{sev_info['color']}15;border-left:4px solid {sev_info['color']};padding:15px;margin:20px 0;border-radius:4px;">
                    <div style="font-size:20px;font-weight:bold;color:{sev_info['color']};">{sev_info['icon']} {disease}</div>
                    <p><strong>Severity:</strong> <span style="display:inline-block;background:{sev_info['color']};color:white;padding:4px 16px;border-radius:20px;font-size:12px;font-weight:bold;">{severity}</span></p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                </div>
                <h3>💊 Recommended Treatment</h3>
                <p style="background:#f8fdf8;padding:12px;border-radius:8px;border-left:3px solid #2e7d32;">{treatment}</p>
                <div style="text-align:center;margin:20px 0;">
                    <a href="https://your-domain.com/dashboard" style="display:inline-block;background:#2e7d32;color:white;padding:10px 24px;text-decoration:none;border-radius:8px;">🔍 View Dashboard</a>
                </div>
                <div style="margin-top:30px;padding-top:20px;border-top:1px solid #e8f5e9;text-align:center;font-size:12px;color:#888;">
                    <p>© {current_year} TomatoGuard. Built for tomato farmers.</p>
                    <p><a href="mailto:support@tomatoguard.com" style="color:#2e7d32;text-decoration:none;">support@tomatoguard.com</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        plain_text = f"""
🚨 Disease Detected!

Hello {farmer_name},

Disease: {disease}
Severity: {severity}
Confidence: {confidence}%

Treatment:
{treatment}

View your dashboard for more details.

© {current_year} TomatoGuard
        """
    
    return send_email(to_email, subject, html_content, plain_text)


# ── Healthy Alert Email ──

def send_healthy_alert_email(to_email, farmer_name, lang='en'):
    """Send healthy plant confirmation via email."""
    if not to_email:
        return False, "No email address provided"
    
    current_year = datetime.now().year
    
    if lang == 'sw':
        subject = "✅ Nyanya Yako Ina Afya - TomatoGuard"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family:Arial,sans-serif;background:#f5faf5;padding:20px;">
            <div style="max-width:600px;margin:0 auto;background:white;border-radius:12px;padding:30px;border:1px solid #c8e6c9;">
                <div style="text-align:center;padding-bottom:20px;border-bottom:2px solid #2e7d32;">
                    <div style="font-size:24px;font-weight:bold;color:#2e7d32;">🍅 TomatoGuard</div>
                </div>
                <div style="background:#e8f5e9;border-left:4px solid #2e7d32;padding:20px;margin:20px 0;border-radius:4px;text-align:center;">
                    <div style="font-size:48px;">✅</div>
                    <h2 style="color:#2e7d32;">Mmea Wako Una Afya!</h2>
                    <p>Hakuna dalili za ugonjwa zilizogunduliwa.</p>
                    <p>Endelea kufuatilia mimea yako mara kwa mara.</p>
                </div>
                <div style="margin-top:30px;padding-top:20px;border-top:1px solid #e8f5e9;text-align:center;font-size:12px;color:#888;">
                    <p>© {current_year} TomatoGuard</p>
                </div>
            </div>
        </body>
        </html>
        """
    else:
        subject = "✅ Your Tomato Plant is Healthy - TomatoGuard"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="UTF-8"></head>
        <body style="font-family:Arial,sans-serif;background:#f5faf5;padding:20px;">
            <div style="max-width:600px;margin:0 auto;background:white;border-radius:12px;padding:30px;border:1px solid #c8e6c9;">
                <div style="text-align:center;padding-bottom:20px;border-bottom:2px solid #2e7d32;">
                    <div style="font-size:24px;font-weight:bold;color:#2e7d32;">🍅 TomatoGuard</div>
                </div>
                <div style="background:#e8f5e9;border-left:4px solid #2e7d32;padding:20px;margin:20px 0;border-radius:4px;text-align:center;">
                    <div style="font-size:48px;">✅</div>
                    <h2 style="color:#2e7d32;">Your Plant is Healthy!</h2>
                    <p>No signs of disease were detected.</p>
                    <p>Continue monitoring your plants regularly.</p>
                </div>
                <div style="margin-top:30px;padding-top:20px;border-top:1px solid #e8f5e9;text-align:center;font-size:12px;color:#888;">
                    <p>© {current_year} TomatoGuard. Built for tomato farmers.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    return send_email(to_email, subject, html_content)


# ── Test Email ──

def test_email(to_email):
    """Send a test email to verify configuration."""
    current_year = datetime.now().year
    subject = "🧪 TomatoGuard - Email Test"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f5faf5; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; padding: 30px; border: 1px solid #c8e6c9; }}
            .header {{ text-align: center; padding-bottom: 20px; border-bottom: 2px solid #2e7d32; }}
            .logo {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
            .success-box {{ background: #e8f5e9; padding: 20px; border-radius: 8px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">🍅 TomatoGuard</div>
            </div>
            <div class="success-box">
                <div style="font-size: 48px;">✅</div>
                <h2 style="color: #2e7d32;">Email Test Successful!</h2>
                <p>Your email notifications are configured correctly.</p>
                <p style="color: #666; font-size: 12px;">Sent at: {datetime.now().strftime('%d %B %Y, %H:%M')}</p>
            </div>
            <p style="text-align: center; color: #888; font-size: 12px; margin-top: 20px;">
                © {current_year} TomatoGuard
            </p>
        </div>
    </body>
    </html>
    """
    
    return send_email(to_email, subject, html_content)