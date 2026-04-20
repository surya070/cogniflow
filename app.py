import json
import os
import threading
from datetime import datetime, timedelta
from functools import wraps

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, url_for)
from flask_login import (LoginManager, current_user, login_required,
                         login_user, logout_user)
from sqlalchemy import text

from models import (HeartRateReading, OxygenSaturationReading,
                    StepReading, User, WatchReading, db)
from rule_engine import score_cognitive_state, generate_insights
from services.processor import (extract_heart_rate_readings,
                                 extract_oxygen_readings,
                                 extract_step_readings, process_payload,
                                 split_payload_by_sleep_sessions)

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "cogniflow-dev-change-in-prod")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///cogniflow.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# ── Flask-Login ───────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please sign in to access Cogniflow."
login_manager.login_message_category = "info"


@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))


# ── Google OAuth (optional) ───────────────────────────────────────────────────
_google_oauth_enabled = bool(
    os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")
)

if _google_oauth_enabled:
    from authlib.integrations.flask_client import OAuth as _OAuth
    _oauth = _OAuth(app)
    _google = _oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


# ── Load ML model ─────────────────────────────────────────────────────────────
_model_bundle = None
_model_path = os.path.join(os.path.dirname(__file__), "models", "energy_score_model.joblib")
if os.path.exists(_model_path):
    try:
        _model_bundle = joblib.load(_model_path)
    except Exception:
        pass

# Fallback: derived formula (used only if model fails to load)
_formula = None
_formula_path = os.path.join(os.path.dirname(__file__), "dataset", "derived_formula.json")
if not _model_bundle and os.path.exists(_formula_path):
    try:
        with open(_formula_path, "r", encoding="utf-8") as fh:
            _formula = json.load(fh)
    except Exception:
        pass


def _compute_energy_score(reading, user: User = None) -> float:
    """
    Compute energy score 0–100 using the trained RandomForest model.
    Falls back to derived formula, then rule engine pct.
    """
    if _model_bundle:
        try:
            feat_names = _model_bundle["feature_names"]
            model      = _model_bundle["model"]

            # Demographics — prefer user record, fall back to neutral defaults
            age       = (user.age       if user and user.age       else 25)
            height_cm = (user.height_cm if user and user.height_cm else 170.0)
            weight_kg = (user.weight_kg if user and user.weight_kg else 70.0)
            sex       = (user.sex       if user and user.sex       else "M")

            gender_m = 1 if sex == "M" else 0
            gender_f = 1 if sex == "F" else 0

            # Sleep efficiency
            sleep_eff = 0.0
            if reading.sleep_efficiency_pct:
                sleep_eff = reading.sleep_efficiency_pct
            elif reading.sleep_start and reading.sleep_end and reading.total_sleep_minutes:
                tib = (reading.sleep_end - reading.sleep_start).total_seconds() / 60
                if tib > 0:
                    sleep_eff = (reading.total_sleep_minutes / tib) * 100

            tsm  = reading.total_sleep_minutes or 0
            deep_pct = ((reading.deep_sleep_minutes  or 0) / tsm * 100) if tsm else 0
            rem_pct  = ((reading.rem_sleep_minutes   or 0) / tsm * 100) if tsm else 0

            val_map = {
                "age":              age,
                "height_cm":        height_cm,
                "weight_kg":        weight_kg,
                "total_sleep_min":  tsm,
                "deep_min":         reading.deep_sleep_minutes  or 0,
                "rem_min":          reading.rem_sleep_minutes   or 0,
                "light_min":        reading.light_sleep_minutes or 0,
                "awake_min":        reading.awake_minutes       or 0,
                "deep_pct":         deep_pct,
                "rem_pct":          rem_pct,
                "sleep_efficiency_pct": sleep_eff,
                "resting_hr":       reading.resting_heart_rate  or 0,
                "avg_hr_overnight": reading.avg_heart_rate      or 0,
                "avg_hr_day":       reading.avg_hr_day          or 0,
                "hr_std":           reading.hr_std              or 0,
                "hr_min":           reading.hr_min              or 0,
                "hr_max":           reading.hr_max              or 0,
                "spo2_avg":         reading.spo2                or 0,
                "spo2_min":         reading.spo2_min            or 0,
                "steps":            reading.steps               or 0,
                "gender_F":         gender_f,
                "gender_M":         gender_m,
            }
            x = np.array([[val_map.get(f, 0) for f in feat_names]])
            score = float(model.predict(x)[0])
            return round(max(0.0, min(100.0, score)), 1)
        except Exception as exc:
            app.logger.warning("ML model prediction failed: %s", exc)

    # Fallback to derived formula
    if _formula:
        try:
            features  = _formula["features"]
            coefs     = _formula["coefficients"]
            intercept = _formula["intercept"]
            sleep_eff = 0.0
            if reading.sleep_start and reading.sleep_end and reading.total_sleep_minutes:
                tib = (reading.sleep_end - reading.sleep_start).total_seconds() / 60
                if tib > 0:
                    sleep_eff = (reading.total_sleep_minutes / tib) * 100
            val_map = {
                "total_sleep_min":      reading.total_sleep_minutes or 0,
                "sleep_efficiency_pct": sleep_eff,
                "resting_hr":           reading.resting_heart_rate or 0,
                "spo2_avg":             reading.spo2 or 0,
                "steps":                reading.steps or 0,
            }
            raw = intercept + sum(coefs[i] * val_map.get(f, 0) for i, f in enumerate(features))
            return round(max(0.0, min(100.0, raw)), 1)
        except Exception:
            pass

    return round(score_cognitive_state(reading)["pct"], 1)


# ── Jinja helpers ─────────────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    return {"google_login_enabled": _google_oauth_enabled}


@app.template_filter("to_ist")
def to_ist(dt):
    if not dt:
        return None
    return dt + timedelta(hours=5, minutes=30)


@app.template_filter("ist_time")
def ist_time(dt):
    if not dt:
        return "—"
    return to_ist(dt).strftime("%H:%M")


@app.template_filter("format_duration")
def format_duration(minutes):
    if not minutes:
        return "—"
    mins = int(minutes)
    hrs, mins = divmod(mins, 60)
    if hrs and mins:
        return f"{hrs}hr {mins}m"
    return f"{hrs}hr" if hrs else f"{mins}m"


# ── Role helpers ──────────────────────────────────────────────────────────────
def manager_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if current_user.role not in ("manager", "admin"):
            flash("You don't have permission to view that page.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if current_user.role != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


# ── Webhook log dir ───────────────────────────────────────────────────────────
WEBHOOK_LOGS_DIR = "webhook_logs"
os.makedirs(WEBHOOK_LOGS_DIR, exist_ok=True)


def _save_webhook_payload(payload: dict, folder_name: str) -> str:
    emp_dir = os.path.join(WEBHOOK_LOGS_DIR, folder_name)
    os.makedirs(emp_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}.json"
    with open(os.path.join(emp_dir, filename), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return filename


# ── Processing pipeline ───────────────────────────────────────────────────────
def _run_pipeline(raw: dict, employee_id: str) -> dict:
    """Processor → rule engine → ML energy score → DB. No LLM."""
    fields = process_payload(raw, employee_id)

    # Sleep efficiency
    start_dt = fields.get("sleep_start")
    end_dt   = fields.get("sleep_end")
    tsm      = fields.get("total_sleep_minutes")
    if start_dt and end_dt and tsm:
        tib = (end_dt - start_dt).total_seconds() / 60
        fields["sleep_efficiency_pct"] = round((tsm / tib * 100), 1) if tib > 0 else None

    # Deduplication
    existing = None
    if fields.get("sleep_end"):
        existing = WatchReading.query.filter_by(
            employee_id=employee_id, sleep_end=fields["sleep_end"]
        ).first()
    if not existing and fields.get("sleep_date"):
        candidates = WatchReading.query.filter_by(
            employee_id=employee_id, sleep_date=fields["sleep_date"]
        ).all()
        if len(candidates) == 1:
            existing = candidates[0]

    if existing:
        reading = existing
        for k, v in fields.items():
            setattr(reading, k, v)
    else:
        reading = WatchReading(**fields)
        db.session.add(reading)

    db.session.flush()

    # Rule engine labels
    rule_result = score_cognitive_state(reading)
    reading.cognitive_score = rule_result["cognitive_score"]
    reading.fatigue_level   = rule_result["fatigue_level"]
    reading.recovery_state  = rule_result["recovery_state"]

    # ML energy score — look up user demographics
    user_obj = User.query.filter_by(employee_id=employee_id).first()
    reading.energy_score = _compute_energy_score(reading, user_obj)

    db.session.commit()

    # Detailed time-series readings
    if existing:
        HeartRateReading.query.filter_by(reading_id=reading.id).delete()
        OxygenSaturationReading.query.filter_by(reading_id=reading.id).delete()
        StepReading.query.filter_by(reading_id=reading.id).delete()

    for hr in extract_heart_rate_readings(raw):
        db.session.add(HeartRateReading(
            reading_id=reading.id, employee_id=employee_id,
            timestamp=hr["timestamp"], bpm=hr["bpm"]
        ))
    for s in extract_oxygen_readings(raw):
        db.session.add(OxygenSaturationReading(
            reading_id=reading.id, employee_id=employee_id,
            timestamp=s["timestamp"], percentage=s["percentage"]
        ))
    for st in extract_step_readings(raw):
        db.session.add(StepReading(
            reading_id=reading.id, employee_id=employee_id,
            start_time=st["start_time"], end_time=st["end_time"], count=st["count"]
        ))

    db.session.commit()

    return {
        "status":       "ok",
        "employee_id":  employee_id,
        "sleep_date":   fields["sleep_date"],
        "energy_score": reading.energy_score,
        "message":      "Data received and processed.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    email = ""
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid email or password.", "error")

    return render_template("auth/login.html", email=email)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("auth/register.html", name=name, email=email)
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("auth/register.html", name=name, email=email)
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("auth/register.html", name=name, email=email)
        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
            return render_template("auth/register.html", name=name, email=email)

        user = User(
            name=name, email=email, employee_id=email,
            webhook_token=User.make_webhook_token(),
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash(f"Account created! Welcome, {user.name}. Fill in your profile details for accurate energy scores.", "success")
        return redirect(url_for("profile"))

    return render_template("auth/register.html", name="", email="")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You've been signed out.", "info")
    return redirect(url_for("login"))


# ── Google OAuth ──────────────────────────────────────────────────────────────

@app.route("/login/google")
def google_login():
    if not _google_oauth_enabled:
        flash("Google login is not configured on this server.", "error")
        return redirect(url_for("login"))
    return _google.authorize_redirect(url_for("google_callback", _external=True))


@app.route("/login/google/callback")
def google_callback():
    if not _google_oauth_enabled:
        return redirect(url_for("login"))
    try:
        token     = _google.authorize_access_token()
        user_info = token.get("userinfo") or {}
    except Exception:
        flash("Google authentication failed.", "error")
        return redirect(url_for("login"))

    google_id  = user_info.get("sub")
    email      = user_info.get("email", "").lower()
    name       = user_info.get("name") or email
    avatar_url = user_info.get("picture")

    if not google_id or not email:
        flash("Google did not return required information.", "error")
        return redirect(url_for("login"))

    user = User.query.filter_by(google_id=google_id).first()
    if not user:
        user = User.query.filter_by(email=email).first()
        if user:
            user.google_id  = google_id
            user.avatar_url = avatar_url
        else:
            user = User(email=email, name=name, google_id=google_id,
                        avatar_url=avatar_url, employee_id=email,
                        webhook_token=User.make_webhook_token())
            db.session.add(user)
    db.session.commit()

    login_user(user)
    flash(f"Welcome, {user.name}!", "success")
    return redirect(url_for("dashboard"))


# ── Profile ───────────────────────────────────────────────────────────────────

@app.route("/profile")
@login_required
def profile():
    webhook_url = (
        request.host_url.rstrip("/")
        + url_for("receive_watch_data")
        + f"?token={current_user.webhook_token}"
    )
    return render_template("profile.html", webhook_url=webhook_url)


@app.route("/profile/details", methods=["POST"])
@login_required
def update_profile_details():
    """Save user demographics used by the ML model."""
    try:
        age = int(request.form.get("age", "0") or 0)
        current_user.age = age if 1 <= age <= 120 else None
    except ValueError:
        current_user.age = None

    sex = request.form.get("sex", "").strip().upper()
    current_user.sex = sex if sex in ("M", "F") else None

    try:
        h = float(request.form.get("height_cm", "0") or 0)
        current_user.height_cm = h if h > 0 else None
    except ValueError:
        current_user.height_cm = None

    try:
        w = float(request.form.get("weight_kg", "0") or 0)
        current_user.weight_kg = w if w > 0 else None
    except ValueError:
        current_user.weight_kg = None

    db.session.commit()

    # Re-score all readings for this user with updated demographics
    readings = WatchReading.query.filter_by(employee_id=current_user.employee_id).all()
    for r in readings:
        r.energy_score = _compute_energy_score(r, current_user)
    db.session.commit()

    flash("Profile updated. Energy scores recalculated.", "success")
    return redirect(url_for("profile"))


@app.route("/profile/regenerate-token", methods=["POST"])
@login_required
def regenerate_token():
    current_user.regenerate_token()
    db.session.commit()
    flash("Webhook token regenerated. Update your Health Connect automation.", "success")
    return redirect(url_for("profile"))


@app.route("/profile/sharing", methods=["POST"])
@login_required
def toggle_sharing():
    current_user.share_readiness = not current_user.share_readiness
    db.session.commit()
    state = "enabled" if current_user.share_readiness else "disabled"
    flash(f"Team sharing {state}.", "success")
    return redirect(url_for("profile"))


# ═══════════════════════════════════════════════════════════════════════════════
# WEBHOOK
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/webhook/watch", methods=["POST"])
def receive_watch_data():
    raw_body = request.get_data(as_text=True)
    payload  = request.get_json(silent=True)

    if payload is None and raw_body:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON payload"}), 400
    if not payload:
        return jsonify({"error": "Empty or missing payload"}), 400

    token = request.args.get("token")
    folder_name = "unknown"
    if token:
        user = User.query.filter_by(webhook_token=token).first()
        if not user:
            return jsonify({"error": "Invalid webhook token"}), 401
        employee_id = user.employee_id
        folder_name = user.email
    else:
        employee_id = (
            request.args.get("employee_id")
            or (payload.get("employee_id") if isinstance(payload, dict) else None)
            or "unknown"
        )
        folder_name = employee_id

    saved_filename = _save_webhook_payload(payload, folder_name)

    def _process_bg(p, eid):
        with app.app_context():
            sessions = split_payload_by_sleep_sessions(p) if isinstance(p, dict) else (p if isinstance(p, list) else [])
            for sub in sessions:
                if isinstance(sub, dict):
                    try:
                        _run_pipeline(sub, eid)
                    except Exception as exc:
                        app.logger.error("Pipeline error for %s: %s", eid, exc)

    threading.Thread(target=_process_bg, args=(payload, employee_id), daemon=True).start()
    return jsonify({"status": "received", "webhook_log_file": saved_filename}), 200


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def dashboard():
    cutoff = datetime.utcnow() - timedelta(days=7)

    if current_user.role == "employee":
        readings = (WatchReading.query
                    .filter_by(employee_id=current_user.employee_id)
                    .filter(WatchReading.received_at >= cutoff)
                    .order_by(WatchReading.received_at.desc())
                    .all())
        return render_template("dashboard.html", readings=readings[:1],
                               all_readings=readings)

    # Managers and admins see their own data on the main dashboard
    # (team view is at /manager)
    readings = (WatchReading.query
                .filter_by(employee_id=current_user.employee_id)
                .filter(WatchReading.received_at >= cutoff)
                .order_by(WatchReading.received_at.desc())
                .all())
    return render_template("dashboard.html", readings=readings[:1],
                           all_readings=readings)


@app.route("/last-received")
@login_required
def last_received():
    raw_payload = "{}"
    filename    = "No files received yet"
    if os.path.exists(WEBHOOK_LOGS_DIR):
        all_files = []
        for root, _, files in os.walk(WEBHOOK_LOGS_DIR):
            for f in files:
                if f.endswith(".json"):
                    all_files.append(os.path.join(root, f))
        if all_files:
            all_files.sort(reverse=True)
            latest_path = all_files[0]
            filename    = os.path.relpath(latest_path, WEBHOOK_LOGS_DIR)
            try:
                with open(latest_path, encoding="utf-8") as f:
                    raw_payload = json.dumps(json.load(f), indent=2)
            except Exception as e:
                raw_payload = f"Error reading file: {e}"
    return render_template("last_received.html",
                           filename=filename, raw_payload=raw_payload)


# ═══════════════════════════════════════════════════════════════════════════════
# EMPLOYEE DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/employee/<path:employee_id>")
@login_required
def employee_detail(employee_id):
    if current_user.role == "employee" and current_user.employee_id != employee_id:
        flash("You don't have permission to view that profile.", "error")
        return redirect(url_for("dashboard"))

    readings = (WatchReading.query
                .filter_by(employee_id=employee_id)
                .order_by(WatchReading.sleep_date.desc())
                .limit(30).all())
    latest   = readings[0] if readings else None

    energy_history = [
        {"date": r.sleep_date, "energy": round(r.energy_score, 1) if r.energy_score is not None else None}
        for r in reversed(readings)
        if r.sleep_date
    ]

    user_obj = User.query.filter_by(employee_id=employee_id).first()
    insights = generate_insights(latest) if latest else []

    return render_template(
        "employee.html",
        employee_id=employee_id,
        readings=readings[:14],
        latest=latest,
        energy_history=energy_history,
        user_obj=user_obj,
        insights=insights,
    )


@app.route("/employee/<path:employee_id>/heart-rate")
@login_required
def heart_rate_analysis(employee_id):
    if current_user.role == "employee" and current_user.employee_id != employee_id:
        flash("You don't have permission to view that profile.", "error")
        return redirect(url_for("dashboard"))
    readings = (WatchReading.query
                .filter_by(employee_id=employee_id)
                .order_by(WatchReading.sleep_date.desc())
                .limit(30).all())
    return render_template("hr_analysis.html",
                           employee_id=employee_id, readings=readings)


# ═══════════════════════════════════════════════════════════════════════════════
# MANAGER PORTAL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/manager")
@manager_required
def manager_portal():
    sharing_users = User.query.filter_by(share_readiness=True).all()
    sharing_ids   = {u.employee_id for u in sharing_users}

    if not sharing_ids:
        return render_template("manager.html", team_stats=None,
                               dept_groups={}, sharing_count=0)

    cutoff   = datetime.utcnow() - timedelta(days=7)
    readings = (WatchReading.query
                .filter(WatchReading.employee_id.in_(sharing_ids))
                .filter(WatchReading.received_at >= cutoff)
                .order_by(WatchReading.received_at.desc())
                .all())

    seen, latest_by_emp = set(), {}
    for r in readings:
        if r.employee_id not in seen:
            seen.add(r.employee_id)
            latest_by_emp[r.employee_id] = r

    green  = sum(1 for r in latest_by_emp.values() if r.readiness_color == "high")
    amber  = sum(1 for r in latest_by_emp.values() if r.readiness_color == "moderate")
    red    = sum(1 for r in latest_by_emp.values() if r.readiness_color == "low")
    team_stats = {"green": green, "amber": amber, "red": red, "total": len(latest_by_emp)}

    dept_groups: dict = {}
    for u in sharing_users:
        dept = u.department or "Unassigned"
        r    = latest_by_emp.get(u.employee_id)
        if r:
            dept_groups.setdefault(dept, []).append(r)

    return render_template("manager.html", team_stats=team_stats,
                           dept_groups=dept_groups, sharing_count=len(sharing_ids))


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PORTAL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/admin")
@admin_required
def admin_portal():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin.html", users=users)


@app.route("/admin/user/<int:user_id>/role", methods=["POST"])
@admin_required
def admin_set_role(user_id):
    user     = User.query.get_or_404(user_id)
    new_role = request.form.get("role", "").strip()
    if new_role not in ("employee", "manager", "admin"):
        flash("Invalid role.", "error")
        return redirect(url_for("admin_portal"))
    if user.id == current_user.id:
        flash("You cannot change your own role.", "error")
        return redirect(url_for("admin_portal"))
    user.role = new_role
    db.session.commit()
    flash(f"Role updated for {user.email}.", "success")
    return redirect(url_for("admin_portal"))


@app.route("/admin/user/<int:user_id>/department", methods=["POST"])
@admin_required
def admin_set_department(user_id):
    user = User.query.get_or_404(user_id)
    dept = request.form.get("department", "").strip()
    user.department = dept if dept else None
    db.session.commit()
    flash(f"Department updated for {user.email}.", "success")
    return redirect(url_for("admin_portal"))


# ═══════════════════════════════════════════════════════════════════════════════
# JSON API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/me/today-score")
@login_required
def api_today_score():
    """Return the current user's latest energy score for calendar flagging."""
    latest = (WatchReading.query
              .filter_by(employee_id=current_user.employee_id)
              .order_by(WatchReading.received_at.desc())
              .first())
    if not latest or latest.energy_score is None:
        return jsonify({"score": None, "color": "neutral", "date": None})
    return jsonify({
        "score": round(latest.energy_score, 1),
        "color": latest.readiness_color,
        "date":  latest.sleep_date,
    })


@app.route("/api/me/google-client-id")
@login_required
def api_google_client_id():
    return jsonify({"client_id": os.getenv("GOOGLE_CLIENT_ID", "")})


@app.route("/api/employee/<path:employee_id>/energy-history")
@login_required
def api_energy_history(employee_id):
    if current_user.role == "employee" and current_user.employee_id != employee_id:
        return jsonify({"error": "Forbidden"}), 403

    days   = request.args.get("days", 7, type=int)
    cutoff = datetime.utcnow() - timedelta(days=days)

    readings = (WatchReading.query
                .filter_by(employee_id=employee_id)
                .filter(WatchReading.received_at >= cutoff)
                .order_by(WatchReading.sleep_date.asc())
                .all())

    return jsonify([
        {"date": r.sleep_date,
         "energy": round(r.energy_score, 1) if r.energy_score is not None else None}
        for r in readings if r.sleep_date
    ])


@app.route("/api/employee/<path:employee_id>/heart-rate")
@login_required
def api_heart_rate(employee_id):
    if current_user.role == "employee" and current_user.employee_id != employee_id:
        return jsonify({"error": "Forbidden"}), 403

    reading_id = request.args.get("reading_id", type=int)
    date_str   = request.args.get("date")

    if date_str and not reading_id:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format"}), 400
        reading = WatchReading.query.filter_by(
            employee_id=employee_id, sleep_date=str(target_date)).first()
        if not reading:
            return jsonify({"error": "No reading found for that date"}), 404
        reading_id = reading.id

    if not reading_id:
        reading = (WatchReading.query.filter_by(employee_id=employee_id)
                   .order_by(WatchReading.sleep_date.desc()).first())
        if not reading:
            return jsonify({"error": "No readings found"}), 404
        reading_id = reading.id

    reading_data = WatchReading.query.get(reading_id)
    if not reading_data:
        return jsonify({"error": "Reading not found"}), 404

    hr_query = (HeartRateReading.query
                .filter_by(reading_id=reading_id, employee_id=employee_id)
                .order_by(HeartRateReading.timestamp))
    if reading_data.sleep_start and reading_data.sleep_end:
        hr_query = hr_query.filter(
            HeartRateReading.timestamp >= reading_data.sleep_start,
            HeartRateReading.timestamp <= reading_data.sleep_end)

    hr_readings = hr_query.all()
    if not hr_readings:
        return jsonify({"reading_id": reading_id, "date": reading_data.sleep_date,
                        "message": "No heart rate data", "hourly": []})

    hourly_data: dict = {}
    for hr in hr_readings:
        key = hr.timestamp.strftime("%H:00")
        hourly_data.setdefault(key, []).append(hr.bpm)

    hourly_stats = [
        {"hour": h, "avg": round(sum(b)/len(b), 1), "min": min(b), "max": max(b), "count": len(b)}
        for h, b in sorted(hourly_data.items())
    ]
    return jsonify({
        "reading_id":     reading_id,
        "date":           reading_data.sleep_date,
        "sleep_start":    reading_data.sleep_start.isoformat() if reading_data.sleep_start else None,
        "sleep_end":      reading_data.sleep_end.isoformat()   if reading_data.sleep_end   else None,
        "total_readings": len(hr_readings),
        "resting_hr":     reading_data.resting_heart_rate,
        "hourly":         hourly_stats,
    })


# ── Test inject ───────────────────────────────────────────────────────────────

@app.route("/test/inject")
@login_required
def test_inject():
    employee_id = request.args.get("employee_id", current_user.employee_id)
    if current_user.role == "employee" and employee_id != current_user.employee_id:
        employee_id = current_user.employee_id

    sample = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT08:00:00Z"),
        "sleep": [{
            "session_end_time": datetime.utcnow().strftime("%Y-%m-%dT08:00:00Z"),
            "stages": [
                {"stage": "4", "start_time": "2026-01-01T23:00:00Z",
                 "end_time": "2026-01-02T00:30:00Z", "duration_seconds": 5400},
                {"stage": "5", "start_time": "2026-01-02T00:30:00Z",
                 "end_time": "2026-01-02T02:00:00Z", "duration_seconds": 5400},
                {"stage": "6", "start_time": "2026-01-02T02:00:00Z",
                 "end_time": "2026-01-02T06:00:00Z", "duration_seconds": 14400},
                {"stage": "1", "start_time": "2026-01-02T06:00:00Z",
                 "end_time": "2026-01-02T07:00:00Z", "duration_seconds": 3600},
            ]
        }],
        "heart_rate": [
            {"bpm": 58, "time": "2026-01-02T00:00:00Z"},
            {"bpm": 56, "time": "2026-01-02T02:00:00Z"},
            {"bpm": 55, "time": "2026-01-02T04:00:00Z"},
            {"bpm": 57, "time": "2026-01-02T06:00:00Z"},
            {"bpm": 75, "time": "2026-01-02T10:00:00Z"},
            {"bpm": 80, "time": "2026-01-02T14:00:00Z"},
        ],
        "oxygen_saturation": [
            {"percentage": 97, "time": "2026-01-02T02:00:00Z"},
            {"percentage": 96, "time": "2026-01-02T05:00:00Z"},
        ],
        "steps": [
            {"count": 8200, "start_time": "2026-01-01T00:00:00Z",
             "end_time": "2026-01-02T00:00:00Z"}
        ],
    }
    result = _run_pipeline(sample, employee_id)
    return jsonify(result), 200


# ═══════════════════════════════════════════════════════════════════════════════
# DB INIT + MIGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def _migrate_db():
    new_cols = [
        ("watch_readings", "energy_score",          "REAL"),
        ("watch_readings", "sleep_efficiency_pct",   "REAL"),
        ("watch_readings", "avg_hr_day",             "REAL"),
        ("watch_readings", "hr_std",                 "REAL"),
        ("watch_readings", "hr_min",                 "INTEGER"),
        ("watch_readings", "hr_max",                 "INTEGER"),
        ("watch_readings", "spo2_min",               "REAL"),
        ("users",          "share_readiness",        "INTEGER DEFAULT 0"),
        ("users",          "department",             "VARCHAR(128)"),
        ("users",          "age",                    "INTEGER"),
        ("users",          "sex",                    "VARCHAR(1)"),
        ("users",          "height_cm",              "REAL"),
        ("users",          "weight_kg",              "REAL"),
    ]
    with db.engine.connect() as conn:
        for table, col, typedef in new_cols:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}"))
                conn.commit()
            except Exception:
                pass

        # Migrate employee_id → email (idempotent)
        try:
            rows = conn.execute(text("SELECT id, email, employee_id FROM users")).fetchall()
            for uid, email, eid in rows:
                if eid != email:
                    conn.execute(text("UPDATE users SET employee_id = :e WHERE id = :i"),
                                 {"e": email, "i": uid})
                    for tbl in ["watch_readings", "heart_rate_readings",
                                "oxygen_saturation_readings", "step_readings",
                                "daily_analyses"]:
                        conn.execute(
                            text(f"UPDATE {tbl} SET employee_id = :e WHERE employee_id = :o"),
                            {"e": email, "o": eid})
            conn.commit()
        except Exception:
            pass


with app.app_context():
    db.create_all()
    _migrate_db()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
