import json
import os
import threading
from datetime import datetime, timedelta
from functools import wraps

from dotenv import load_dotenv
from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, url_for)
from flask_login import (LoginManager, current_user, login_required,
                         login_user, logout_user)

from models import (DailyAnalysis, HeartRateReading, OxygenSaturationReading,
                    StepReading, User, WatchReading, db)
from rule_engine import score_cognitive_state
from services.llm_service import analyze_reading
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


# ── Google OAuth (optional — only enabled when env vars are present) ──────────
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


# ── Jinja helpers ─────────────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    return {
        "google_login_enabled": _google_oauth_enabled,
    }


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
    """Decorator: requires manager or admin role."""
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if current_user.role not in ("manager", "admin"):
            flash("You don't have permission to view that page.", "error")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated


# ── Webhook log dir ───────────────────────────────────────────────────────────
WEBHOOK_LOGS_DIR = "webhook_logs"
os.makedirs(WEBHOOK_LOGS_DIR, exist_ok=True)


def _save_webhook_payload(payload: dict, employee_id: str = "unknown") -> str:
    """Save raw webhook JSON to webhook_logs/<employee_id>/<timestamp>.json."""
    emp_dir = os.path.join(WEBHOOK_LOGS_DIR, employee_id)
    os.makedirs(emp_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}.json"
    with open(os.path.join(emp_dir, filename), "w") as f:
        json.dump(payload, f, indent=2)
    return filename


# ── Processing pipeline ───────────────────────────────────────────────────────
def _run_pipeline(raw: dict, employee_id: str) -> dict:
    """Full pipeline: processor → rule engine → LLM → DB."""
    fields = process_payload(raw, employee_id)

    # Deduplication on (employee_id, sleep_end)
    existing = None
    if fields.get("sleep_end"):
        existing = WatchReading.query.filter_by(
            employee_id=employee_id,
            sleep_end=fields["sleep_end"]
        ).first()

    if not existing and fields.get("sleep_date"):
        candidates = WatchReading.query.filter_by(
            employee_id=employee_id,
            sleep_date=fields["sleep_date"]
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

    score = score_cognitive_state(reading)
    reading.cognitive_score = score["cognitive_score"]
    reading.fatigue_level   = score["fatigue_level"]
    reading.recovery_state  = score["recovery_state"]
    db.session.commit()

    llm_result = analyze_reading(reading, score)

    if existing and existing.analysis:
        analysis = existing.analysis
    else:
        analysis = DailyAnalysis(reading_id=reading.id, employee_id=employee_id)
        db.session.add(analysis)

    analysis.summary         = llm_result.get("summary", "")
    analysis.insights        = json.dumps(llm_result.get("insights", []))
    analysis.task_allocation = json.dumps(llm_result.get("task_allocation", {}))
    analysis.warnings        = json.dumps(llm_result.get("warnings", []))
    analysis.full_response   = json.dumps(llm_result)
    db.session.commit()

    # Detailed readings
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
        "status":          "ok",
        "employee_id":     employee_id,
        "sleep_date":      fields["sleep_date"],
        "cognitive_score": score["cognitive_score"],
        "readiness_pct":   score["pct"],
        "readiness_label": llm_result.get("readiness_label"),
        "message":         "Data received, processed and analysed.",
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
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))

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

        # Validation
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

        # Create user
        user = User(
            name=name,
            email=email,
            webhook_token=User.make_webhook_token(),
            employee_id=User.make_employee_id(),
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash(f"Account created! Welcome, {user.name}.", "success")
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
    redirect_uri = url_for("google_callback", _external=True)
    return _google.authorize_redirect(redirect_uri)


@app.route("/login/google/callback")
def google_callback():
    if not _google_oauth_enabled:
        return redirect(url_for("login"))

    try:
        token     = _google.authorize_access_token()
        user_info = token.get("userinfo") or {}
    except Exception:
        flash("Google authentication failed. Please try again.", "error")
        return redirect(url_for("login"))

    google_id  = user_info.get("sub")
    email      = user_info.get("email", "").lower()
    name       = user_info.get("name") or email
    avatar_url = user_info.get("picture")

    if not google_id or not email:
        flash("Google did not return required account information.", "error")
        return redirect(url_for("login"))

    # Find or create user
    user = User.query.filter_by(google_id=google_id).first()
    if not user:
        user = User.query.filter_by(email=email).first()
        if user:
            # Link Google to existing email/password account
            user.google_id  = google_id
            user.avatar_url = avatar_url
        else:
            user = User(
                email=email,
                name=name,
                google_id=google_id,
                avatar_url=avatar_url,
                webhook_token=User.make_webhook_token(),
                employee_id=User.make_employee_id(),
            )
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


@app.route("/profile/regenerate-token", methods=["POST"])
@login_required
def regenerate_token():
    current_user.regenerate_token()
    db.session.commit()
    flash("Webhook token regenerated. Update your watch app with the new URL.", "success")
    return redirect(url_for("profile"))


# ═══════════════════════════════════════════════════════════════════════════════
# WEBHOOK  (token-based, no login session needed — called by watch app)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/webhook/watch", methods=["POST"])
def receive_watch_data():
    """
    Accepts POST from Health Auto Export.
    Authentication: ?token=<webhook_token>   (preferred)
    Legacy fallback: ?employee_id=<id>       (for existing setups)
    """
    raw_body = request.get_data(as_text=True)
    payload  = request.get_json(silent=True)

    if payload is None and raw_body:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON payload"}), 400

    if not payload:
        return jsonify({"error": "Empty or missing payload"}), 400

    # ── Resolve employee_id from token or legacy param ──
    token = request.args.get("token")
    if token:
        user = User.query.filter_by(webhook_token=token).first()
        if not user:
            return jsonify({"error": "Invalid webhook token"}), 401
        employee_id = user.employee_id
    else:
        # Legacy: accept employee_id query param (existing setups)
        employee_id = (
            request.args.get("employee_id")
            or (payload.get("employee_id") if isinstance(payload, dict) else None)
            or "unknown"
        )

    saved_filename = _save_webhook_payload(payload, employee_id)

    # Process in background so the watch app gets an immediate 200
    # (Gemini analysis can take 10–20 s; the watch app times out otherwise)
    def _process_bg(p, eid):
        with app.app_context():
            if isinstance(p, dict):
                for sub in split_payload_by_sleep_sessions(p):
                    try:
                        _run_pipeline(sub, eid)
                    except Exception as exc:
                        app.logger.error("Pipeline error for %s: %s", eid, exc)
            elif isinstance(p, list):
                for item in p:
                    if isinstance(item, dict):
                        try:
                            _run_pipeline(item, eid)
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
    query  = WatchReading.query.filter(WatchReading.received_at >= cutoff)

    # Employees only see their own data
    if current_user.role == "employee":
        query = query.filter_by(employee_id=current_user.employee_id)

    readings = query.order_by(WatchReading.received_at.desc()).all()

    # Deduplicate to one card per employee (latest reading)
    seen, latest = set(), []
    for r in readings:
        if r.employee_id not in seen:
            seen.add(r.employee_id)
            latest.append(r)

    return render_template("dashboard.html", readings=latest)


# ── Last received (debug) ─────────────────────────────────────────────────────

@app.route("/last-received")
@login_required
def last_received():
    raw_payload = "{}"
    filename    = "No files received yet"

    if os.path.exists(WEBHOOK_LOGS_DIR):
        # Search recursively through employee subdirectories
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
                with open(latest_path) as f:
                    raw_payload = json.dumps(json.load(f), indent=2)
            except Exception as e:
                raw_payload = f"Error reading file: {e}"

    return render_template("last_received.html",
                           filename=filename, raw_payload=raw_payload)


# ═══════════════════════════════════════════════════════════════════════════════
# EMPLOYEE DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/employee/<employee_id>")
@login_required
def employee_detail(employee_id):
    # Employees can only view their own data
    if (current_user.role == "employee"
            and current_user.employee_id != employee_id):
        flash("You don't have permission to view that profile.", "error")
        return redirect(url_for("dashboard"))

    readings = (
        WatchReading.query
        .filter_by(employee_id=employee_id)
        .order_by(WatchReading.sleep_date.desc())
        .limit(14)
        .all()
    )
    latest   = readings[0] if readings else None
    analysis = latest.analysis if latest else None

    insights = task_alloc = warnings = {}
    insights, task_alloc, warnings, readiness_label = [], {}, [], "-"

    if analysis:
        try:
            insights        = json.loads(analysis.insights or "[]")
            task_alloc      = json.loads(analysis.task_allocation or "{}")
            warnings        = json.loads(analysis.warnings or "[]")
            full            = json.loads(analysis.full_response or "{}")
            readiness_label = full.get("readiness_label", "-")
        except Exception:
            pass

    return render_template(
        "employee.html",
        employee_id=employee_id,
        readings=readings,
        latest=latest,
        analysis=analysis,
        insights=insights,
        task_alloc=task_alloc,
        warnings=warnings,
        readiness_label=readiness_label,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# JSON API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/employees")
@login_required
def api_employees():
    if current_user.role == "employee":
        return jsonify([current_user.employee_id])
    rows = db.session.query(WatchReading.employee_id).distinct().all()
    return jsonify([r[0] for r in rows])


@app.route("/api/reading/<int:reading_id>")
@login_required
def api_reading(reading_id):
    r = WatchReading.query.get_or_404(reading_id)
    if current_user.role == "employee" and r.employee_id != current_user.employee_id:
        return jsonify({"error": "Forbidden"}), 403
    return jsonify({
        "id":              r.id,
        "employee_id":     r.employee_id,
        "sleep_date":      r.sleep_date,
        "total_sleep_hrs": round((r.total_sleep_minutes or 0) / 60, 2),
        "deep_sleep_min":  r.deep_sleep_minutes,
        "resting_hr":      r.resting_heart_rate,
        "hrv":             r.hrv,
        "spo2":            r.spo2,
        "cognitive_score": r.cognitive_score,
        "fatigue_level":   r.fatigue_level,
        "recovery_state":  r.recovery_state,
    })


# ── Heart rate ────────────────────────────────────────────────────────────────

@app.route("/employee/<employee_id>/heart-rate")
@login_required
def heart_rate_analysis(employee_id):
    if (current_user.role == "employee"
            and current_user.employee_id != employee_id):
        flash("You don't have permission to view that profile.", "error")
        return redirect(url_for("dashboard"))

    readings = (
        WatchReading.query
        .filter_by(employee_id=employee_id)
        .order_by(WatchReading.sleep_date.desc())
        .limit(30)
        .all()
    )
    return render_template("hr_analysis.html",
                           employee_id=employee_id, readings=readings)


@app.route("/api/employee/<employee_id>/heart-rate")
@login_required
def api_heart_rate(employee_id):
    if (current_user.role == "employee"
            and current_user.employee_id != employee_id):
        return jsonify({"error": "Forbidden"}), 403

    reading_id = request.args.get("reading_id", type=int)
    date_str   = request.args.get("date")

    if date_str and not reading_id:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        reading = WatchReading.query.filter_by(
            employee_id=employee_id, sleep_date=target_date).first()
        if not reading:
            return jsonify({"error": "No reading found for that date"}), 404
        reading_id = reading.id

    if not reading_id:
        reading = (WatchReading.query
                   .filter_by(employee_id=employee_id)
                   .order_by(WatchReading.sleep_date.desc())
                   .first())
        if not reading:
            return jsonify({"error": "No readings found for this employee"}), 404
        reading_id = reading.id

    reading_data = WatchReading.query.get(reading_id)
    if not reading_data:
        return jsonify({"error": "Reading not found"}), 404

    hr_readings = (
        HeartRateReading.query
        .filter_by(reading_id=reading_id, employee_id=employee_id)
        .filter(
            HeartRateReading.timestamp >= reading_data.sleep_start,
            HeartRateReading.timestamp <= reading_data.sleep_end
        )
        .order_by(HeartRateReading.timestamp)
        .all()
    )

    if not hr_readings:
        return jsonify({
            "reading_id": reading_id,
            "date": reading_data.sleep_date,
            "message": "No heart rate data",
            "hourly": []
        })

    hourly_data: dict = {}
    for hr in hr_readings:
        key = hr.timestamp.strftime("%H:00")
        hourly_data.setdefault(key, []).append(hr.bpm)

    hourly_stats = [
        {
            "hour":  h,
            "avg":   round(sum(bpms) / len(bpms), 1),
            "min":   min(bpms),
            "max":   max(bpms),
            "count": len(bpms),
        }
        for h, bpms in sorted(hourly_data.items())
    ]

    return jsonify({
        "reading_id":    reading_id,
        "employee_id":   employee_id,
        "date":          reading_data.sleep_date,
        "sleep_start":   reading_data.sleep_start.isoformat() if reading_data.sleep_start else None,
        "sleep_end":     reading_data.sleep_end.isoformat() if reading_data.sleep_end else None,
        "total_readings": len(hr_readings),
        "resting_hr":    reading_data.resting_heart_rate,
        "avg_hr":        reading_data.avg_heart_rate,
        "hourly":        hourly_stats,
    })


# ── Test inject ───────────────────────────────────────────────────────────────

@app.route("/test/inject")
@login_required
def test_inject():
    """GET /test/inject — runs the full pipeline with sample data."""
    employee_id = request.args.get("employee_id", current_user.employee_id)

    # Employees can only inject data for themselves
    if current_user.role == "employee" and employee_id != current_user.employee_id:
        employee_id = current_user.employee_id

    sample = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT08:00:00Z"),
        "sleep": [{
            "stages": [
                {"stage": 4, "duration_seconds": 5700},
                {"stage": 3, "duration_seconds": 6000},
                {"stage": 2, "duration_seconds": 12600},
                {"stage": 1, "duration_seconds": 1800},
            ]
        }],
        "heart_rate": [
            {"bpm": 56}, {"bpm": 58}, {"bpm": 57},
            {"bpm": 60}, {"bpm": 62}, {"bpm": 59},
        ],
        "oxygen_saturation": [{"percentage": 97}],
        "steps": [{"count": 8200}],
    }
    result = _run_pipeline(sample, employee_id)
    return jsonify(result), 200


# ── DB init & run ─────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
