"""
=============================================================
APP.PY - Flask Backend
Project: Tomato Leaf Disease Detection
=============================================================
"""

import os
import hashlib
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, make_response, send_from_directory

# ── Load Environment Variables ──
from dotenv import load_dotenv
load_dotenv()

from module1_image_loader import load_image, resize_image, TOMATO_CLASSES
from module4_forward_pass import TomatoCNN
from module8_rejection import TomatoRejector
from database import save_prediction, get_recent_predictions, get_connection, setup_all_tables

# ── Import Payment Module ──
from payment_routes import register_payment_routes, setup_payment_tables

# ── Import Email Notifier ──
from email_notifier import (
    send_disease_alert_email,
    send_healthy_alert_email,
    test_email,
    EMAIL_CONFIGURED
)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'tomato_cnn_secret_key_2024')

# ── Register Payment Routes ──
register_payment_routes(app)

UPLOAD_FOLDER      = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMAGE_SIZE = (64, 64)


# ─────────────────────────────────────────────
# FREE TRIAL CONFIGURATION
# ─────────────────────────────────────────────

FREE_TRIAL_SCANS = 10
FREE_TRIAL_DAYS = 7


def get_free_trial_usage(farmer_id):
    conn = get_connection()
    if not conn:
        return 0, FREE_TRIAL_SCANS, False, 0, FREE_TRIAL_SCANS
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id FROM subscriptions WHERE farmer_id=%s AND is_active=1", (farmer_id,))
        has_paid_sub = cursor.fetchone() is not None
        if has_paid_sub:
            cursor.close()
            conn.close()
            return 0, 0, False, 0, 0
        cursor.execute("SELECT * FROM free_trials WHERE farmer_id=%s", (farmer_id,))
        trial = cursor.fetchone()
        if not trial:
            cursor.execute("INSERT INTO free_trials (farmer_id, scans_used, max_scans, created_at) VALUES (%s, 0, %s, NOW())", (farmer_id, FREE_TRIAL_SCANS))
            conn.commit()
            cursor.execute("SELECT * FROM free_trials WHERE farmer_id=%s", (farmer_id,))
            trial = cursor.fetchone()
        scans_used = trial['scans_used'] if trial else 0
        max_scans = trial['max_scans'] if trial else FREE_TRIAL_SCANS
        created_at = trial['created_at'] if trial else None
        trial_active = True
        days_left = FREE_TRIAL_DAYS
        if created_at:
            trial_days = (datetime.now() - created_at).days
            days_left = max(0, FREE_TRIAL_DAYS - trial_days)
            if trial_days >= FREE_TRIAL_DAYS:
                trial_active = False
        if scans_used >= max_scans:
            trial_active = False
        scans_remaining = max(0, max_scans - scans_used)
        cursor.close()
        conn.close()
        return scans_used, scans_remaining, trial_active, days_left, max_scans
    except Exception as e:
        print(f"[Free Trial] Error: {e}")
        return 0, FREE_TRIAL_SCANS, True, FREE_TRIAL_DAYS, FREE_TRIAL_SCANS


def increment_free_trial_usage(farmer_id):
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE free_trials SET scans_used = scans_used + 1 WHERE farmer_id=%s AND scans_used < max_scans", (farmer_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[Free Trial] Increment error: {e}")
        return False


def has_subscription_or_trial(farmer_id):
    has_sub, subscription = has_active_subscription(farmer_id)
    if has_sub:
        return True, 'subscription', {
            'plan': subscription.get('plan_id') if subscription else 'starter',
            'expires_at': subscription.get('expires_at') if subscription else None,
            'scan_count': 0,
            'scan_limit': 'Unlimited' if subscription and subscription.get('plan_id') == 'pro' else 50
        }
    scans_used, scans_remaining, trial_active, days_left, scan_limit = get_free_trial_usage(farmer_id)
    if trial_active:
        return True, 'trial', {
            'scans_used': scans_used,
            'scans_remaining': scans_remaining,
            'scan_limit': scan_limit,
            'days_left': days_left,
            'max_days': FREE_TRIAL_DAYS
        }
    return False, 'none', {
        'scans_used': scans_used,
        'scan_limit': scan_limit
    }


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

print("[App] Loading model...")
model = TomatoCNN(num_classes=11)
model.load_weights("saved_weights/best_model.npz")
model.set_training(False)
rejector = TomatoRejector(confidence_threshold=0.75)
print("[App] Model ready!")


# ─────────────────────────────────────────────
# SUBSCRIPTION CHECK HELPER
# ─────────────────────────────────────────────

def has_active_subscription(farmer_id):
    conn = get_connection()
    if not conn:
        return False, None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM subscriptions WHERE farmer_id=%s AND is_active=1 AND (expires_at IS NULL OR expires_at > NOW())", (farmer_id,))
        sub = cursor.fetchone()
        cursor.close()
        conn.close()
        return sub is not None, sub
    except Exception as e:
        print(f"[Subscription Check] Error: {e}")
        return False, None


def get_scan_count(farmer_id, days=30):
    conn = get_connection()
    if not conn:
        return 0
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE farmer_id=%s AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)", (farmer_id, days))
        count = cursor.fetchone()['cnt']
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"[Scan Count] Error: {e}")
        return 0


# ─────────────────────────────────────────────
# SEVERITY
# ─────────────────────────────────────────────

def get_severity(confidence, predicted_class, lang='en'):
    if "healthy" in predicted_class.lower():
        return None
    if confidence >= 0.92:
        return {
            'level_en': 'Severe',   'level_sw': 'Kali Sana',
            'emoji': '🔴', 'color': '#ef5350',
            'desc_en': 'Strong signs of disease detected. Act immediately.',
            'desc_sw': 'Dalili kali za ugonjwa zimegunduliwa. Chukua hatua mara moja.',
        }
    elif confidence >= 0.83:
        return {
            'level_en': 'Moderate', 'level_sw': 'Ya Wastani',
            'emoji': '🟡', 'color': '#ffca28',
            'desc_en': 'Moderate signs of disease detected. Treat as soon as possible.',
            'desc_sw': 'Dalili za wastani za ugonjwa zimegunduliwa. Tibu haraka iwezekanavyo.',
        }
    else:
        return {
            'level_en': 'Mild',     'level_sw': 'Kidogo',
            'emoji': '🟢', 'color': '#66bb6a',
            'desc_en': 'Early signs of disease detected. Monitor closely and treat early.',
            'desc_sw': 'Dalili za mapema za ugonjwa zimegunduliwa. Angalia kwa makini na tibu mapema.',
        }


# ─────────────────────────────────────────────
# TREATMENTS
# ─────────────────────────────────────────────

TREATMENTS_EN = {
    "Tomato_Bacterial_spot": "Remove and destroy infected plant parts immediately. Apply copper-based bactericides every 7-10 days. Avoid overhead irrigation. Rotate crops next season.",
    "Tomato_Early_blight": "Remove infected leaves immediately. Apply fungicides containing chlorothalonil or mancozeb. Mulch around plants to prevent soil splash. Water in the morning so leaves dry during the day.",
    "Tomato_Late_blight": "Remove and destroy ALL infected plant material immediately. Apply fungicides with metalaxyl or copper compounds. Avoid wetting foliage. Consider removing the entire plant if severely infected.",
    "Tomato_Leaf_Mold": "Improve air circulation by pruning and spacing plants. Apply fungicides containing chlorothalonil or mancozeb. Reduce humidity. Avoid overhead watering.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves as soon as symptoms appear. Apply fungicides with chlorothalonil or copper. Mulch soil to prevent spore splashing. Rotate crops annually.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray plants with strong water jets to dislodge mites. Apply miticides or insecticidal soap sprays. Keep plants well watered — mites thrive in dry conditions.",
    "Tomato__Target_Spot": "Apply fungicides containing azoxystrobin or chlorothalonil. Remove and destroy infected plant material. Improve air circulation. Practice crop rotation.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "No cure — remove and destroy infected plants immediately. Control whitefly populations using insecticides. Plant virus-resistant tomato varieties.",
    "Tomato__Tomato_mosaic_virus": "No cure — remove and destroy infected plants. Wash hands before handling plants. Disinfect all tools. Control aphid populations. Plant resistant varieties.",
    "Tomato_healthy": "Your tomato plant looks healthy! Continue regular watering at the base of plants. Monitor weekly for early signs of disease.",
}

TREATMENTS_SW = {
    "Tomato_Bacterial_spot": "Ondoa na uharibu sehemu zote zilizoathiriwa mara moja. Tumia dawa ya shaba kila siku 7-10. Epuka kumwagilia kutoka juu. Zungushia mazao msimu ujao.",
    "Tomato_Early_blight": "Ondoa majani yaliyoathiriwa mara moja. Tumia dawa ya kuua kuvu kama vile chlorothalonil au mancozeb. Weka matandazo karibu na mimea kuzuia udongo kuruka. Mwagilia asubuhi ili majani yakauke mchana.",
    "Tomato_Late_blight": "Ondoa na uharibu NYENZO ZOTE zilizoathiriwa mara moja. Tumia dawa yenye metalaxyl au misombo ya shaba. Epuka kulowanisha majani. Fikiria kuondoa mmea wote ikiwa umeathiriwa sana.",
    "Tomato_Leaf_Mold": "Boresha mzunguko wa hewa kwa kupogoa na kupanga nafasi kati ya mimea. Tumia dawa ya kuua kuvu yenye chlorothalonil au mancozeb. Punguza unyevu. Epuka kumwagilia kutoka juu.",
    "Tomato_Septoria_leaf_spot": "Ondoa majani yaliyoathiriwa mara dalili zinapoonekana. Tumia dawa ya kuua kuvu yenye chlorothalonil au shaba. Weka matandazo kuzuia mbegu za ugonjwa kuruka. Zungushia mazao kila mwaka.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Nyunyizia mimea maji kwa nguvu kuondoa sarafu. Tumia dawa ya kuua wadudu au sabuni ya kuua wadudu. Mwagilia mimea vizuri — sarafu hustawi katika hali kavu.",
    "Tomato__Target_Spot": "Tumia dawa ya kuua kuvu yenye azoxystrobin au chlorothalonil. Ondoa na uharibu nyenzo zilizoathiriwa. Boresha mzunguko wa hewa. Zungushia mazao.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Hakuna tiba — ondoa na uharibu mimea iliyoathiriwa mara moja. Dhibiti idadi ya nzi weupe kwa kutumia dawa. Panda aina za nyanya zinazostahimili virusi.",
    "Tomato__Tomato_mosaic_virus": "Hakuna tiba — ondoa na uharibu mimea iliyoathiriwa. Osha mikono kabla ya kushughulikia mimea. Safisha zana zote. Dhibiti idadi ya aphid. Panda aina zinazostahimili.",
    "Tomato_healthy": "Mmea wako wa nyanya unaonekana kuwa na afya! Endelea kumwagilia mara kwa mara kwenye msingi wa mimea. Angalia kila wiki kwa ishara za mapema za ugonjwa.",
}

DISEASE_NAMES_SW = {
    "Tomato_Bacterial_spot"                       : "Madoa ya Bakteria",
    "Tomato_Early_blight"                         : "Uozo wa Mapema",
    "Tomato_Late_blight"                          : "Uozo wa Marehemu",
    "Tomato_Leaf_Mold"                            : "Ukungu wa Majani",
    "Tomato_Septoria_leaf_spot"                   : "Madoa ya Septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite" : "Sarafu za Buibui",
    "Tomato__Target_Spot"                         : "Madoa ya Lengo",
    "Tomato__Tomato_YellowLeaf__Curl_Virus"       : "Virusi vya Majani Kujikunja",
    "Tomato__Tomato_mosaic_virus"                 : "Virusi vya Mosaic",
    "Tomato_healthy"                              : "Mmea Wenye Afya",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_display_name(predicted_class, lang='en'):
    if lang == 'sw':
        return DISEASE_NAMES_SW.get(predicted_class,
            predicted_class.replace("Tomato__","").replace("Tomato_","").replace("_"," "))
    return predicted_class.replace("Tomato__","").replace("Tomato_","").replace("_"," ")

def get_treatment(predicted_class, lang='en'):
    if lang == 'sw':
        return TREATMENTS_SW.get(predicted_class, "Hakuna maelezo ya matibabu.")
    return TREATMENTS_EN.get(predicted_class, "No treatment information available.")

def severity_label(confidence, predicted_class, lang='en'):
    if "healthy" in predicted_class.lower():
        return "—"
    if confidence >= 0.92:
        return "Kali Sana" if lang == 'sw' else "Severe"
    elif confidence >= 0.83:
        return "Ya Wastani" if lang == 'sw' else "Moderate"
    else:
        return "Kidogo" if lang == 'sw' else "Mild"


# ─────────────────────────────────────────────
# SMS HELPER
# ─────────────────────────────────────────────

def maybe_send_sms(farmer_id, predicted_class, display_name, confidence, severity, lang='en'):
    try:
        from sms_notifier import send_disease_alert, send_healthy_alert
        conn = get_connection(); cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM notification_settings WHERE farmer_id=%s", (farmer_id,))
        notif = cursor.fetchone()
        cursor.close(); conn.close()
        if not notif:
            return
        sms_lang = notif.get('lang', lang)
        phone    = notif.get('phone1', '').strip()
        if not phone:
            return
        is_healthy = "healthy" in predicted_class.lower()
        if is_healthy and notif.get('notify_healthy'):
            send_healthy_alert(phone, lang=sms_lang)
            return
        if not is_healthy and notif.get('notify_disease'):
            sev_level = severity['level_en'] if severity else 'Mild'
            min_sev   = notif.get('min_severity', 'mild')
            sev_order = {'mild': 1, 'moderate': 2, 'severe': 3}
            if sev_order.get(sev_level.lower(), 1) >= sev_order.get(min_sev, 1):
                treatment = TREATMENTS_EN.get(predicted_class, "")
                send_disease_alert(
                    phone_number   = phone,
                    disease        = display_name,
                    severity_level = sev_level,
                    treatment      = treatment,
                    lang           = sms_lang
                )
    except Exception as e:
        print(f"[SMS] Notification skipped: {e}")


# ─────────────────────────────────────────────
# EMAIL HELPER
# ─────────────────────────────────────────────

def maybe_send_email(farmer_id, predicted_class, display_name, confidence, severity, treatment, lang='en'):
    """Send email notification if farmer has email configured."""
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT ns.email, ns.notify_disease, ns.notify_healthy, ns.lang,
                   f.full_name as farmer_name
            FROM notification_settings ns
            JOIN farmers f ON f.id = ns.farmer_id
            WHERE ns.farmer_id = %s
        """, (farmer_id,))
        notif = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not notif or not notif.get('email'):
            return
        
        email = notif['email'].strip()
        if not email:
            return
        
        farmer_name = notif.get('farmer_name', 'Farmer')
        email_lang = notif.get('lang', lang)
        is_healthy = "healthy" in predicted_class.lower()
        
        if is_healthy and notif.get('notify_healthy'):
            send_healthy_alert_email(email, farmer_name, lang=email_lang)
            print(f"[Email] ✅ Healthy alert sent to {email}")
            return
        
        if not is_healthy and notif.get('notify_disease'):
            sev_level = severity['level_en'] if severity else 'Mild'
            min_sev = 'mild'
            sev_order = {'mild': 1, 'moderate': 2, 'severe': 3}
            if sev_order.get(sev_level.lower(), 1) >= sev_order.get(min_sev, 1):
                send_disease_alert_email(
                    to_email=email,
                    farmer_name=farmer_name,
                    disease=display_name,
                    severity=sev_level,
                    confidence=round(confidence, 1),
                    treatment=treatment,
                    lang=email_lang
                )
                print(f"[Email] ✅ Disease alert sent to {email}")
                
    except Exception as e:
        print(f"[Email] Notification skipped: {e}")


# ─────────────────────────────────────────────
# DATASET IMAGES ROUTE
# ─────────────────────────────────────────────

@app.route('/dataset/<path:filename>')
def dataset_images(filename):
    """Serve dataset images for the landing page carousel."""
    return send_from_directory('dataset', filename)


@app.route('/api/dataset-images')
def get_dataset_images():
    """Return list of all images in dataset organized by class."""
    import os
    dataset_path = 'dataset'
    result = {}
    
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset folder not found'})
    
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            images = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                    images.append(file)
            if images:
                result[folder] = images[:10]
    
    return jsonify(result)


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def home():
    if 'farmer_id' not in session:
        return render_template('landing.html')
    return redirect(url_for('index'))

@app.route('/landing')
def landing():
    if 'farmer_id' in session:
        return redirect(url_for('index'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed   = hash_password(password)
        conn = get_connection(); cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM farmers WHERE username=%s AND password=%s", (username, hashed))
        farmer = cursor.fetchone(); cursor.close(); conn.close()
        if farmer:
            session['farmer_id']   = farmer['id']
            session['farmer_name'] = farmer['full_name']
            return redirect(url_for('index'))
        else:
            error = "Invalid username or password."
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None; success = None
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        username  = request.form.get('username')
        password  = request.form.get('password')
        confirm   = request.form.get('confirm_password')
        if password != confirm:
            error = "Passwords do not match."
        else:
            hashed = hash_password(password)
            try:
                conn = get_connection(); cursor = conn.cursor()
                cursor.execute("INSERT INTO farmers (full_name, username, password) VALUES (%s,%s,%s)",
                               (full_name, username, hashed))
                conn.commit(); cursor.close(); conn.close()
                success = "Account created! You can now log in."
            except Exception:
                error = "Username already exists."
    return render_template('login.html', error=error, success=success, show_register=True)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

@app.route('/check_username', methods=['POST'])
def check_username():
    data     = request.get_json()
    username = data.get('username', '').strip()
    conn = get_connection(); cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM farmers WHERE username=%s", (username,))
    found = cursor.fetchone() is not None
    cursor.close(); conn.close()
    return jsonify({'found': found})


# ─────────────────────────────────────────────
# MAIN ROUTES
# ─────────────────────────────────────────────

@app.route('/dashboard')
def index():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    has_access, access_type, access_details = has_subscription_or_trial(session['farmer_id'])
    recent = get_recent_predictions(farmer_id=session['farmer_id'], limit=10)
    scan_count = get_scan_count(session['farmer_id'])
    trial_info = {}
    if access_type == 'trial':
        trial_info = {
            'scans_used': access_details.get('scans_used', 0),
            'scans_remaining': access_details.get('scans_remaining', 0),
            'scan_limit': access_details.get('scan_limit', FREE_TRIAL_SCANS),
            'days_left': access_details.get('days_left', 0),
            'max_days': access_details.get('max_days', FREE_TRIAL_DAYS),
            'is_trial': True
        }
    return render_template('index.html', 
        recent=recent,
        farmer_name=session.get('farmer_name'),
        has_subscription=has_access,
        access_type=access_type,
        subscription=access_details if access_type == 'subscription' else None,
        trial_info=trial_info,
        scan_count=scan_count,
        scan_limit='Unlimited' if access_type == 'subscription' and access_details.get('plan') == 'pro' else 50
    )

@app.route('/stats')
def stats():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    return render_template('stats.html', farmer_name=session.get('farmer_name'))

@app.route('/api/stats')
def api_stats():
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    farmer_id = session['farmer_id']
    conn = get_connection(); cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT COUNT(*) AS total FROM predictions WHERE farmer_id=%s", (farmer_id,))
    total = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) AS cnt FROM predictions WHERE farmer_id=%s AND predicted_disease='Tomato_healthy'", (farmer_id,))
    healthy_count = cursor.fetchone()['cnt']
    diseased_count = total - healthy_count
    
    cursor.execute("SELECT predicted_disease, COUNT(*) AS cnt FROM predictions WHERE farmer_id=%s GROUP BY predicted_disease ORDER BY cnt DESC", (farmer_id,))
    disease_rows = cursor.fetchall()
    
    cursor.execute("SELECT confidence FROM predictions WHERE farmer_id=%s AND predicted_disease != 'Tomato_healthy'", (farmer_id,))
    conf_rows = cursor.fetchall()
    mild = moderate = severe = 0
    for row in conf_rows:
        c = row['confidence']
        if c >= 0.92:
            severe += 1
        elif c >= 0.83:
            moderate += 1
        else:
            mild += 1
    
    cursor.execute("SELECT AVG(confidence) AS avg_conf FROM predictions WHERE farmer_id=%s", (farmer_id,))
    avg_conf = cursor.fetchone()['avg_conf'] or 0
    
    cursor.execute("""SELECT DATE(created_at) AS day, COUNT(*) AS cnt
                      FROM predictions WHERE farmer_id=%s
                      AND created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                      GROUP BY DATE(created_at) ORDER BY day ASC""", (farmer_id,))
    timeline_rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    disease_labels = []
    disease_counts = []
    for row in disease_rows:
        name = row['predicted_disease'].replace('Tomato__','').replace('Tomato_','').replace('_',' ')
        disease_labels.append(name)
        disease_counts.append(row['cnt'])
    
    return jsonify({
        'total': total,
        'healthy_count': healthy_count,
        'diseased_count': diseased_count,
        'avg_confidence': round(float(avg_conf)*100, 1),
        'severity': {'mild': mild, 'moderate': moderate, 'severe': severe},
        'disease_labels': disease_labels,
        'disease_counts': disease_counts,
        'timeline_labels': [str(r['day']) for r in timeline_rows],
        'timeline_counts': [r['cnt'] for r in timeline_rows],
        'farmer_name': session.get('farmer_name'),
    })

@app.route('/tips_settings')
def tips_settings():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    return render_template('tips_settings.html', farmer_name=session.get('farmer_name'))


# ─────────────────────────────────────────────
# NOTIFICATION ROUTES
# ─────────────────────────────────────────────

@app.route('/notifications')
def notifications():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM notification_settings WHERE farmer_id=%s", (session['farmer_id'],))
    settings = cursor.fetchone() or {}
    cursor.close()
    conn.close()
    return render_template('notifications.html', settings=settings, farmer_name=session.get('farmer_name'))


@app.route('/api/notifications/get', methods=['GET'])
def get_notifications():
    """Get current notification settings for the logged-in farmer."""
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM notification_settings WHERE farmer_id=%s", (session['farmer_id'],))
    settings = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return jsonify({'settings': settings or {}})


@app.route('/api/notifications/save', methods=['POST'])
def save_notifications():
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    conn = get_connection()
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
    """, (
        session['farmer_id'],
        data.get('phone1', '').strip(),
        data.get('phone2', '').strip(),
        data.get('email', '').strip(),
        data.get('lang', 'en'),
        1 if data.get('disease') else 0,
        1 if data.get('healthy') else 0,
        1 if data.get('weekly') else 0,
        data.get('severity', 'mild'),
        data.get('send_time', 'instant'),
    ))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/notifications/test', methods=['POST'])
def test_notification():
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    phone = data.get('phone1', '').strip()
    if not phone:
        return jsonify({'error': 'No phone number provided'}), 400
    try:
        from sms_notifier import test_sms_connection
        success, response = test_sms_connection(phone)
        return jsonify({'success': success, 'response': str(response)})
    except Exception as e:
        return jsonify({'success': False, 'response': str(e)}), 500


# ─────────────────────────────────────────────
# EMAIL TEST ROUTE
# ─────────────────────────────────────────────

@app.route('/api/email/test', methods=['POST'])
def test_email_endpoint():
    """Send a test email to verify email configuration."""
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    email = data.get('email', '').strip()
    
    if not email:
        return jsonify({'error': 'No email provided'}), 400
    
    try:
        success, message = test_email(email)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─────────────────────────────────────────────
# FEEDBACK ROUTES
# ─────────────────────────────────────────────

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    prediction_id = data.get('prediction_id')
    is_correct = data.get('is_correct')
    rating = data.get('rating')
    comment = data.get('comment', '').strip()
    
    if not prediction_id:
        return jsonify({'error': 'Missing prediction_id'}), 400
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM predictions WHERE id=%s AND farmer_id=%s", (prediction_id, session['farmer_id']))
    pred = cursor.fetchone()
    if not pred:
        cursor.close()
        conn.close()
        return jsonify({'error': 'Prediction not found'}), 404
    
    cursor.execute("SELECT id FROM feedback WHERE prediction_id=%s AND farmer_id=%s", (prediction_id, session['farmer_id']))
    existing = cursor.fetchone()
    if existing:
        cursor.execute("UPDATE feedback SET is_correct=%s, rating=%s, comment=%s WHERE prediction_id=%s AND farmer_id=%s",
                       (is_correct, rating, comment, prediction_id, session['farmer_id']))
    else:
        cursor.execute("INSERT INTO feedback (prediction_id, farmer_id, is_correct, rating, comment) VALUES (%s, %s, %s, %s, %s)",
                       (prediction_id, session['farmer_id'], is_correct, rating, comment))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'success': True, 'message': 'Feedback submitted successfully'})

@app.route('/feedback/<int:prediction_id>', methods=['GET'])
def get_feedback(prediction_id):
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM feedback WHERE prediction_id=%s AND farmer_id=%s", (prediction_id, session['farmer_id']))
    fb = cursor.fetchone()
    cursor.close()
    conn.close()
    if fb:
        return jsonify({
            'exists': True,
            'is_correct': fb['is_correct'],
            'rating': fb['rating'],
            'comment': fb['comment'],
        })
    return jsonify({'exists': False})

@app.route('/api/feedback/summary', methods=['GET'])
def feedback_summary():
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    farmer_id = session['farmer_id']
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT COUNT(*) AS total FROM feedback WHERE farmer_id=%s", (farmer_id,))
    total = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) AS cnt FROM feedback WHERE farmer_id=%s AND is_correct=1", (farmer_id,))
    correct = cursor.fetchone()['cnt']
    
    cursor.execute("SELECT COUNT(*) AS cnt FROM feedback WHERE farmer_id=%s AND is_correct=0", (farmer_id,))
    incorrect = cursor.fetchone()['cnt']
    
    cursor.execute("SELECT AVG(rating) AS avg_rating FROM feedback WHERE farmer_id=%s AND rating IS NOT NULL", (farmer_id,))
    avg_rating = cursor.fetchone()['avg_rating'] or 0
    
    cursor.close()
    conn.close()
    
    correct_pct = round((correct / total * 100), 1) if total > 0 else 0
    incorrect_pct = round((incorrect / total * 100), 1) if total > 0 else 0
    
    return jsonify({
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'correct_pct': correct_pct,
        'incorrect_pct': incorrect_pct,
        'avg_rating': round(float(avg_rating), 1),
    })


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────

@app.route('/report')
def download_report():
    if 'farmer_id' not in session:
        return redirect(url_for('login'))

    from datetime import date, timedelta

    lang = request.args.get('lang', 'en')
    period = request.args.get('period', 'all')
    disease_f = request.args.get('disease', 'all')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    farmer_id = session['farmer_id']

    query = "SELECT * FROM predictions WHERE farmer_id=%s"
    params = [farmer_id]
    today = date.today()

    if period == 'today':
        query += " AND DATE(created_at)=%s"
        params.append(str(today))
        period_label = str(today)
    elif period == 'week':
        week_start = today - timedelta(days=today.weekday())
        query += " AND DATE(created_at) >= %s"
        params.append(str(week_start))
        period_label = f"{week_start} → {today}"
    elif period == 'month':
        query += " AND MONTH(created_at)=MONTH(CURDATE()) AND YEAR(created_at)=YEAR(CURDATE())"
        period_label = today.strftime("%B %Y")
    elif period == 'custom' and date_from and date_to:
        query += " AND DATE(created_at) BETWEEN %s AND %s"
        params += [date_from, date_to]
        period_label = f"{date_from} → {date_to}"
    else:
        period_label = "All time" if lang == 'en' else "Wakati Wote"

    if disease_f != 'all':
        query += " AND predicted_disease=%s"
        params.append(disease_f)

    query += " ORDER BY created_at DESC"

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params)
    predictions = cursor.fetchall()

    cursor.execute(
        "SELECT DISTINCT predicted_disease FROM predictions WHERE farmer_id=%s ORDER BY predicted_disease",
        (farmer_id,)
    )
    all_diseases = [r['predicted_disease'] for r in cursor.fetchall()]

    cursor.execute("SELECT full_name FROM farmers WHERE id=%s", (farmer_id,))
    farmer = cursor.fetchone()
    cursor.close()
    conn.close()

    farmer_name = farmer['full_name'] if farmer else session.get('farmer_name', '')
    now = datetime.now().strftime("%d %B %Y, %H:%M")
    total = len(predictions)
    healthy = sum(1 for p in predictions if 'healthy' in p['predicted_disease'].lower())
    diseased = total - healthy

    if lang == 'sw':
        title = "Ripoti ya Utambuzi wa Magonjwa ya Nyanya"
        subtitle = "TomatoGuard — Mfumo wa AI wa Kugundua Magonjwa ya Nyanya"
        farmer_lbl = "Mkulima"
        date_lbl = "Tarehe ya Ripoti"
        summary_lbl = "Muhtasari"
        total_lbl = "Jumla ya Uchunguzi"
        healthy_lbl = "Mimea Yenye Afya"
        diseased_lbl = "Mimea Yenye Ugonjwa"
        history_lbl = "Historia ya Utambuzi"
        no_lbl = "Na."
        image_lbl = "Picha"
        disease_lbl = "Ugonjwa"
        conf_lbl = "Uhakika"
        sev_lbl = "Ukali"
        date_col = "Tarehe"
        no_data = "Hakuna rekodi zinazolingana na chujio lililochaguliwa."
        footer_txt = "Imetolewa na TomatoGuard | Mfumo wa CNN kutoka Mwanzo"
        print_btn = "🖨️ Chapisha / Hifadhi PDF"
        close_btn = "✕ Funga"
        filter_title = "🔍 Chuja Ripoti"
        period_lbl = "Kipindi"
        disease_lbl2 = "Aina ya Ugonjwa"
        custom_lbl = "Tarehe Maalum"
        apply_btn = "Tumia Chujio"
        all_lbl = "Zote"
        today_lbl = "Leo"
        week_lbl = "Wiki Hii"
        month_lbl = "Mwezi Huu"
        custom_lbl2 = "Maalum"
        period_row = "Kipindi:"
    else:
        title = "Tomato Disease Detection Report"
        subtitle = "TomatoGuard — AI-Powered Tomato Disease Detection System"
        farmer_lbl = "Farmer"
        date_lbl = "Report Date"
        summary_lbl = "Summary"
        total_lbl = "Total Scans"
        healthy_lbl = "Healthy Plants"
        diseased_lbl = "Diseased Plants"
        history_lbl = "Detection History"
        no_lbl = "No."
        image_lbl = "Image"
        disease_lbl = "Disease"
        conf_lbl = "Confidence"
        sev_lbl = "Severity"
        date_col = "Date"
        no_data = "No records match the selected filter."
        footer_txt = "Generated by TomatoGuard | CNN from Scratch System"
        print_btn = "🖨️ Print / Save as PDF"
        close_btn = "✕ Close"
        filter_title = "🔍 Filter Report"
        period_lbl = "Period"
        disease_lbl2 = "Disease Type"
        custom_lbl = "Custom Date Range"
        apply_btn = "Apply Filter"
        all_lbl = "All"
        today_lbl = "Today"
        week_lbl = "This Week"
        month_lbl = "This Month"
        custom_lbl2 = "Custom"
        period_row = "Period:"

    disease_options_html = f'<option value="all" {"selected" if disease_f == "all" else ""}>{all_lbl}</option>'
    for d in all_diseases:
        disp = d.replace("Tomato__","").replace("Tomato_","").replace("_"," ")
        if lang == "sw":
            disp = DISEASE_NAMES_SW.get(d, disp)
        sel = "selected" if disease_f == d else ""
        disease_options_html += f'<option value="{d}" {sel}>{disp}</option>'

    rows_html = ""
    for i, p in enumerate(predictions, 1):
        disease_name = (p['predicted_disease']
                        .replace('Tomato__','').replace('Tomato_','').replace('_',' '))
        if lang == 'sw':
            disease_name = DISEASE_NAMES_SW.get(p['predicted_disease'], disease_name)
        conf = round(p['confidence'] * 100, 1)
        sev = severity_label(p['confidence'], p['predicted_disease'], lang)
        date_str = str(p['created_at'])[:16]
        is_healthy = 'healthy' in p['predicted_disease'].lower()
        sev_color = '#888'
        if sev in ('Severe', 'Kali Sana'):
            sev_color = '#ef5350'
        elif sev in ('Moderate', 'Ya Wastani'):
            sev_color = '#e6a817'
        elif sev in ('Mild', 'Kidogo'):
            sev_color = '#4caf50'
        row_bg = '#f9fff9' if is_healthy else '#fffaf6'
        rows_html += f"""
        <tr style="background:{row_bg};">
            <td style="text-align:center;">{i}</td>
            <td style="font-size:0.82em; color:#555;">{p['image_name'][:28]}...</td>
            <td><strong>{'✅ ' if is_healthy else '⚠️ '}{disease_name}</strong></td>
            <td style="text-align:center;">{conf}%</td>
            <td style="text-align:center; color:{sev_color}; font-weight:bold;">{sev}</td>
            <td style="font-size:0.85em; color:#666;">{date_str}</td>
        </tr>"""

    if not rows_html:
        rows_html = f'<tr><td colspan="6" style="text-align:center;padding:20px;color:#888;">{no_data}</td></tr>'

    active_disease_label = ""
    if disease_f != "all":
        dn = disease_f.replace("Tomato__","").replace("Tomato_","").replace("_"," ")
        if lang == "sw":
            dn = DISEASE_NAMES_SW.get(disease_f, dn)
        active_disease_label = f" &nbsp;|&nbsp; {disease_lbl}: <strong>{dn}</strong>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',Arial,sans-serif; color:#222; background:white; padding:30px; font-size:13px; }}
.header {{ display:flex; justify-content:space-between; align-items:center; border-bottom:3px solid #2e7d32; padding-bottom:16px; margin-bottom:20px; }}
.header-left h1 {{ font-size:1.4em; color:#1b5e20; margin-bottom:4px; }}
.header-left p {{ color:#555; font-size:0.88em; }}
.header-right {{ text-align:right; color:#555; font-size:0.85em; line-height:1.7; }}
.logo-box {{ font-size:2em; background:#2e7d32; color:white; width:55px; height:55px; border-radius:12px; display:flex; align-items:center; justify-content:center; }}
.summary {{ display:flex; gap:15px; margin-bottom:25px; }}
.summary-card {{ flex:1; padding:15px; border-radius:10px; text-align:center; border:1px solid #ddd; }}
.summary-card .val {{ font-size:2em; font-weight:bold; }}
.summary-card .lbl {{ font-size:0.8em; color:#666; margin-top:3px; }}
.card-total {{ border-color:#4caf50; background:#f1f8e9; }}
.card-healthy .val {{ color:#2e7d32; }}
.card-diseased {{ border-color:#ff6f00; background:#fff8f0; }}
.card-diseased .val {{ color:#e65100; }}
.section-title {{ font-size:1em; font-weight:bold; color:#1b5e20; border-left:4px solid #4caf50; padding-left:10px; margin-bottom:12px; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:20px; }}
th {{ background:#2e7d32; color:white; padding:9px 10px; text-align:left; font-size:0.85em; font-weight:600; }}
td {{ padding:8px 10px; border-bottom:1px solid #eee; vertical-align:middle; }}
tr:hover td {{ background:#f5f5f5 !important; }}
.toolbar {{ display:flex; gap:10px; justify-content:flex-end; margin-bottom:16px; flex-wrap:wrap; }}
.btn-print {{ padding:10px 24px; background:#2e7d32; color:white; border:none; border-radius:8px; font-size:0.95em; cursor:pointer; }}
.btn-print:hover {{ background:#4caf50; }}
.btn-close {{ padding:10px 18px; background:#eee; color:#333; border:none; border-radius:8px; font-size:0.95em; cursor:pointer; }}
.btn-close:hover {{ background:#ddd; }}
.btn-download {{ padding:10px 24px; background:#1565c0; color:white; border:none; border-radius:8px; font-size:0.95em; cursor:pointer; }}
.btn-download:hover {{ background:#1976d2; }}
.filter-bar {{ background:#f8fdf8; border:1px solid #c8e6c9; border-radius:10px; padding:16px 18px; margin-bottom:20px; }}
.filter-bar h3 {{ font-size:0.9em; color:#2e7d32; margin-bottom:12px; }}
.filter-row {{ display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end; }}
.filter-group {{ display:flex; flex-direction:column; gap:4px; }}
.filter-group label {{ font-size:0.78em; color:#555; font-weight:600; }}
.filter-group select, .filter-group input {{ padding:7px 10px; border:1px solid #c8e6c9; border-radius:7px; font-size:0.85em; color:#222; background:white; }}
.period-btns {{ display:flex; gap:6px; flex-wrap:wrap; }}
.period-btn {{ padding:6px 14px; border:1px solid #c8e6c9; border-radius:20px; font-size:0.82em; cursor:pointer; background:white; color:#2e7d32; transition:all 0.2s; }}
.period-btn:hover, .period-btn.active {{ background:#2e7d32; color:white; border-color:#2e7d32; }}
.custom-dates {{ display:none; gap:8px; align-items:center; margin-top:8px; flex-wrap:wrap; }}
.custom-dates input {{ padding:6px 10px; border:1px solid #c8e6c9; border-radius:7px; font-size:0.82em; }}
.btn-apply {{ padding:8px 20px; background:#2e7d32; color:white; border:none; border-radius:8px; font-size:0.85em; cursor:pointer; margin-top:4px; }}
.btn-apply:hover {{ background:#4caf50; }}
.active-filter {{ font-size:0.82em; color:#555; margin-bottom:14px; padding:7px 12px; background:#f1f8e9; border-radius:7px; border-left:3px solid #4caf50; }}
.footer {{ margin-top:30px; padding-top:12px; border-top:1px solid #ddd; text-align:center; color:#888; font-size:0.8em; }}
@media print {{ body {{ padding:15px; }} .toolbar {{ display:none; }} .filter-bar {{ display:none; }} @page {{ margin:1.5cm; }} }}
</style></head><body>

<div class="toolbar">
    <button class="btn-download" onclick="downloadPDF()">⬇️ Download PDF</button>
    <button class="btn-print" onclick="window.print()">{print_btn}</button>
    <button class="btn-close" onclick="window.close()">{close_btn}</button>
</div>

<div class="filter-bar">
    <h3>{filter_title}</h3>
    <div class="filter-row">
        <div class="filter-group">
            <label>{period_lbl}</label>
            <div class="period-btns">
                <button class="period-btn {"active" if period == "all" else ""}"   onclick="setPeriod('all')">{all_lbl}</button>
                <button class="period-btn {"active" if period == "today" else ""}" onclick="setPeriod('today')">{today_lbl}</button>
                <button class="period-btn {"active" if period == "week" else ""}"  onclick="setPeriod('week')">{week_lbl}</button>
                <button class="period-btn {"active" if period == "month" else ""}" onclick="setPeriod('month')">{month_lbl}</button>
                <button class="period-btn {"active" if period == "custom" else ""}" onclick="setPeriod('custom')">{custom_lbl2}</button>
            </div>
            <div class="custom-dates" id="customDates" style="{"display:flex;" if period == "custom" else "display:none;"}">
                <input type="date" id="dateFrom" value="{date_from}">
                <span>→</span>
                <input type="date" id="dateTo" value="{date_to}">
            </div>
        </div>
        <div class="filter-group">
            <label>{disease_lbl2}</label>
            <select id="diseaseFilter" onchange="applyFilters()">
                {disease_options_html}
            </select>
        </div>
        <div class="filter-group">
            <button class="btn-apply" onclick="applyFilters()">{apply_btn}</button>
        </div>
    </div>
</div>

<div class="active-filter">
    {period_row} <strong>{period_label}</strong>{active_disease_label}
</div>

<div class="header">
    <div style="display:flex; align-items:center; gap:14px;">
        <div class="logo-box">🍅</div>
        <div class="header-left"><h1>{title}</h1><p>{subtitle}</p></div>
    </div>
    <div class="header-right"><strong>{farmer_lbl}:</strong> {farmer_name}<br><strong>{date_lbl}:</strong> {now}</div>
</div>

<div class="section-title">📊 {summary_lbl}</div>
<div class="summary">
    <div class="summary-card card-total"><div class="val">{total}</div><div class="lbl">🔍 {total_lbl}</div></div>
    <div class="summary-card card-healthy"><div class="val" style="color:#2e7d32;">{healthy}</div><div class="lbl">✅ {healthy_lbl}</div></div>
    <div class="summary-card card-diseased"><div class="val">{diseased}</div><div class="lbl">⚠️ {diseased_lbl}</div></div>
</div>

<div class="section-title">📋 {history_lbl}</div>
<table>
    <thead>
        <tr>
            <th style="width:40px;">{no_lbl}</th>
            <th>{image_lbl}</th>
            <th>{disease_lbl}</th>
            <th style="width:80px;">{conf_lbl}</th>
            <th style="width:90px;">{sev_lbl}</th>
            <th style="width:130px;">{date_col}</th>
        </tr>
    </thead>
    <tbody>{rows_html}</tbody>
</table>
<div class="footer">{footer_txt} &nbsp;|&nbsp; {period_label} &nbsp;|&nbsp; {now}</div>

<script>
    let currentPeriod = '{period}';

    function setPeriod(p) {{
        currentPeriod = p;
        document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');
        document.getElementById('customDates').style.display = p === 'custom' ? 'flex' : 'none';
        if (p !== 'custom') applyFilters();
    }}

    function applyFilters() {{
        const disease  = document.getElementById('diseaseFilter').value;
        const dateFrom = document.getElementById('dateFrom') ? document.getElementById('dateFrom').value : '';
        const dateTo   = document.getElementById('dateTo')   ? document.getElementById('dateTo').value   : '';
        let url = '/report?lang={lang}&period=' + currentPeriod + '&disease=' + disease;
        if (currentPeriod === 'custom' && dateFrom && dateTo) {{
            url += '&date_from=' + dateFrom + '&date_to=' + dateTo;
        }}
        window.location.href = url;
    }}
    function downloadPDF() {{
        const url = new URL(window.location.href);
        url.searchParams.set('download', '1');
        window.location.href = url.toString();
    }}
</script>
</body></html>"""

    if request.args.get('download') == '1':
        try:
            from xhtml2pdf import pisa
            import io
            pdf_buffer = io.BytesIO()
            pisa.CreatePDF(html, dest=pdf_buffer, encoding='utf-8')
            pdf_buffer.seek(0)
            filename = f"TomatoGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            response = make_response(pdf_buffer.read())
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response
        except Exception as e:
            print(f"[PDF Error] {e}")
            # Fallback: show HTML version
            response = make_response(html)
            response.headers['Content-Type'] = 'text/html; charset=utf-8'
            return response
    response = make_response(html)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    farmer_id = session['farmer_id']
    lang = request.form.get('lang', 'en')
    
    has_access, access_type, access_details = has_subscription_or_trial(farmer_id)
    
    if not has_access:
        return jsonify({
            'error': 'no_access',
            'message': 'You have used all your free scans. Please subscribe to continue.',
            'redirect': '/payment/starter',
            'scans_used': access_details.get('scans_used', 0),
            'scans_limit': access_details.get('scan_limit', FREE_TRIAL_SCANS)
        }), 403
    
    if access_type == 'trial':
        scans_used = access_details.get('scans_used', 0)
        scan_limit = access_details.get('scan_limit', FREE_TRIAL_SCANS)
        days_left = access_details.get('days_left', 0)
        
        if scans_used >= scan_limit:
            return jsonify({
                'error': 'trial_expired',
                'message': f'You have used all your {FREE_TRIAL_SCANS} free scans. Subscribe to continue!',
                'redirect': '/payment/starter',
                'scans_used': scans_used,
                'scans_limit': scan_limit
            }), 403
        
        if days_left <= 0:
            return jsonify({
                'error': 'trial_expired',
                'message': f'Your {FREE_TRIAL_DAYS}-day free trial has expired. Subscribe to continue!',
                'redirect': '/payment/starter'
            }), 403
    
    if access_type == 'subscription':
        plan = access_details.get('plan', 'starter')
        if plan == 'starter':
            scan_count = get_scan_count(farmer_id, days=30)
            if scan_count >= 50:
                return jsonify({
                    'error': 'scan_limit_reached',
                    'message': 'You have used all 50 scans for this month. Upgrade to Pro for unlimited scans.',
                    'redirect': '/payment/pro'
                }), 403
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'invalid_file',
            'disease': 'Invalid File Type' if lang == 'en' else 'Aina ya Faili si Sahihi',
            'confidence': 0,
            'treatment': 'Please upload a JPG or PNG image.' if lang == 'en'
                         else 'Tafadhali pakia picha ya JPG au PNG.'
        }), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        img = load_image(filepath)
        img = resize_image(img, IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, :]

        is_plausible, prefilter_reason, green_ratio = rejector.prefilter(img)

        if not is_plausible:
            msg = (
                f"\n  Picha hii si jani la nyanya.\n  Sababu: {prefilter_reason}\n\n"
                "  Vidokezo:\n    - Pakia picha wazi ya jani moja la nyanya\n"
                "    - Hakikisha mwanga mzuri\n    - Jaza fremu na jani\n    - Epuka picha zisizo wazi\n"
            ) if lang == 'sw' else rejector.get_rejection_message(prefilter_reason, green_ratio)
            return jsonify({
                'success': False,
                'error': 'rejected',
                'disease': 'Jani Zuri la Nyanya Halijulikani' if lang == 'sw' else 'Not a Valid Tomato Leaf',
                'confidence': 0,
                'treatment': msg
            })

        probs = model.forward(img)
        class_id = int(np.argmax(probs, axis=1)[0])
        confidence = float(np.max(probs, axis=1)[0])
        predicted_class = TOMATO_CLASSES[class_id]

        if predicted_class == "non_tomato":
            msg = 'Mfano ulibainisha picha hii si jani la nyanya.' if lang == 'sw' \
                  else 'The model identified this as not a tomato leaf.'
            return jsonify({
                'success': False,
                'error': 'rejected',
                'disease': 'Si Jani la Nyanya' if lang == 'sw' else 'Not a Tomato Leaf',
                'confidence': round(confidence * 100, 2),
                'treatment': rejector.get_rejection_message(msg)
            })

        should_reject, reject_reason = rejector.should_reject_by_confidence(confidence, predicted_class)
        if should_reject:
            msg = f"Uhakika ni mdogo sana ({confidence*100:.1f}% < 75%)" if lang == 'sw' else reject_reason
            return jsonify({
                'success': False,
                'error': 'low_confidence',
                'disease': 'Utambuzi Hauko Wazi' if lang == 'sw' else 'Uncertain Detection',
                'confidence': round(confidence * 100, 2),
                'treatment': rejector.get_rejection_message(msg, confidence=confidence)
            })

        display_name = get_display_name(predicted_class, lang)
        treatment = get_treatment(predicted_class, lang)
        severity = get_severity(confidence, predicted_class, lang)

        prediction_id = save_prediction(
            farmer_id=farmer_id,
            image_name=file.filename,
            predicted_disease=predicted_class,
            confidence=confidence,
            treatment=TREATMENTS_EN.get(predicted_class, "")
        )

        scans_remaining = None
        if access_type == 'trial':
            increment_free_trial_usage(farmer_id)
            scans_remaining = access_details.get('scans_remaining', 0) - 1

        # ── Send SMS Notification ──
        maybe_send_sms(
            farmer_id=farmer_id,
            predicted_class=predicted_class,
            display_name=display_name,
            confidence=round(confidence * 100, 2),
            severity=severity,
            lang=lang
        )

        # ── Send Email Notification ──
        if EMAIL_CONFIGURED:
            maybe_send_email(
                farmer_id=farmer_id,
                predicted_class=predicted_class,
                display_name=display_name,
                confidence=round(confidence * 100, 2),
                severity=severity,
                treatment=treatment,
                lang=lang
            )

        return jsonify({
            'success': True,
            'disease': display_name,
            'confidence': round(confidence * 100, 2),
            'treatment': treatment.strip(),
            'severity': severity,
            'prediction_id': prediction_id,
            'access_type': access_type,
            'trial_remaining': scans_remaining if access_type == 'trial' else None
        })

    except Exception as e:
        print(f"[Error] {e}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# OTHER ROUTES
# ─────────────────────────────────────────────

@app.route('/history')
def history():
    if 'farmer_id' not in session:
        return jsonify([])
    recent = get_recent_predictions(farmer_id=session['farmer_id'], limit=10)
    for r in recent:
        if 'created_at' in r:
            r['created_at'] = str(r['created_at'])
    return jsonify(recent)

@app.route('/prediction/<int:prediction_id>', methods=['GET'])
def view_prediction(prediction_id):
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM predictions WHERE id=%s AND farmer_id=%s", (prediction_id, session['farmer_id']))
    prediction = cursor.fetchone()
    cursor.close()
    conn.close()
    if not prediction:
        return jsonify({'error': 'Prediction not found'}), 404
    prediction['created_at'] = str(prediction['created_at'])
    return jsonify(prediction)

@app.route('/prediction/<int:prediction_id>/delete', methods=['POST', 'DELETE'])
def delete_prediction(prediction_id):
    if 'farmer_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM predictions WHERE id=%s AND farmer_id=%s", (prediction_id, session['farmer_id']))
    prediction = cursor.fetchone()
    if not prediction:
        cursor.close()
        conn.close()
        return jsonify({'error': 'Prediction not found or not yours'}), 404
    cursor.execute("DELETE FROM predictions WHERE id=%s AND farmer_id=%s", (prediction_id, session['farmer_id']))
    conn.commit()
    cursor.close()
    conn.close()
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], prediction['image_name'])
    if os.path.exists(image_path):
        os.remove(image_path)
    return jsonify({'success': True, 'message': 'Prediction deleted successfully'})


# ─────────────────────────────────────────────
# CSV REPORT FALLBACK
# ─────────────────────────────────────────────

@app.route('/report_csv')
def download_report_csv():
    """Download CSV report instead of PDF (fallback if PDF fails)"""
    if 'farmer_id' not in session:
        return redirect(url_for('login'))
    
    import csv
    from io import StringIO
    
    farmer_id = session['farmer_id']
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM predictions WHERE farmer_id=%s ORDER BY created_at DESC", (farmer_id,))
    predictions = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not predictions:
        return "No detections found", 404
    
    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['#', 'Image', 'Disease', 'Confidence', 'Severity', 'Date'])
    
    for i, p in enumerate(predictions, 1):
        disease = p['predicted_disease'].replace('Tomato_', '').replace('_', ' ')
        sev = severity_label(p['confidence'], p['predicted_disease'], 'en')
        writer.writerow([
            i,
            p['image_name'],
            disease,
            f"{p['confidence']*100:.1f}%",
            sev,
            p['created_at']
        ])
    
    output = si.getvalue()
    si.close()
    
    response = make_response(output)
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename=TomatoGuard_Report_{datetime.now().strftime("%Y%m%d")}.csv'
    return response


# ─────────────────────────────────────────────
# INITIALIZE DATABASE TABLES (FIXED - CORRECT ORDER)
# ─────────────────────────────────────────────

print("[App] Setting up database tables...")

# ✅ FIXED: Call setup_all_tables() which creates tables in correct order
# This creates: farmers → predictions → feedback → transactions → 
# subscriptions → notification_settings → free_trials
setup_all_tables()

# Setup payment tables (they reference farmers too)
setup_payment_tables()

print("[App] ✅ Database tables ready!")

# Print email status
if EMAIL_CONFIGURED:
    print("[App] ✅ Email notifications configured (Resend API)")
else:
    print("[App] ⚠️ Email notifications NOT configured. Set RESEND_API_KEY in .env")


# ─────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    if debug_mode:
        import webbrowser
        webbrowser.open(f'http://localhost:{port}')

    app.run(host='0.0.0.0', port=port, debug=debug_mode)