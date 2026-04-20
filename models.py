import secrets
from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(db.Model, UserMixin):
    """User account — linked to watch readings via employee_id (= email)."""
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name          = db.Column(db.String(128), nullable=False)

    # Auth — password_hash is NULL for Google-only accounts
    password_hash = db.Column(db.String(256))
    google_id     = db.Column(db.String(128), unique=True, index=True)
    avatar_url    = db.Column(db.String(512))

    # employee_id equals email — stable across DB resets because it's derived
    # from the immutable email address.  Webhook token is separate (rotatable).
    webhook_token = db.Column(db.String(64), unique=True, nullable=False, index=True)
    employee_id   = db.Column(db.String(255), unique=True, nullable=False, index=True)

    # role: "employee" / "manager" / "admin"
    role          = db.Column(db.String(32), default="employee")
    is_active     = db.Column(db.Boolean, default=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    # Privacy — when True this user's aggregate readiness state flows to manager view
    share_readiness = db.Column(db.Boolean, default=False)

    # Optional department assignment (set by admin)
    department    = db.Column(db.String(128))

    # Demographics — used by ML model for energy score prediction
    age           = db.Column(db.Integer)
    sex           = db.Column(db.String(1))   # "M" or "F"
    height_cm     = db.Column(db.Float)
    weight_kg     = db.Column(db.Float)

    # ── helpers ───────────────────────────────────────────────────────────────

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def regenerate_token(self) -> str:
        self.webhook_token = User.make_webhook_token()
        return self.webhook_token

    @staticmethod
    def make_webhook_token() -> str:
        return secrets.token_urlsafe(32)

    def __repr__(self):
        return f"<User {self.email}>"


class WatchReading(db.Model):
    """Raw processed watch data — one row per sleep session."""
    __tablename__ = "watch_readings"

    id              = db.Column(db.Integer, primary_key=True)
    employee_id     = db.Column(db.String(255), nullable=False, index=True)
    received_at     = db.Column(db.DateTime, default=datetime.utcnow)

    # Sleep metrics
    sleep_date          = db.Column(db.String(20))
    sleep_start         = db.Column(db.DateTime)
    sleep_end           = db.Column(db.DateTime)
    total_sleep_minutes = db.Column(db.Float)
    deep_sleep_minutes  = db.Column(db.Float)
    rem_sleep_minutes   = db.Column(db.Float)
    light_sleep_minutes = db.Column(db.Float)
    awake_minutes       = db.Column(db.Float)
    sleep_efficiency_pct = db.Column(db.Float)   # total_sleep / time_in_bed * 100
    sleep_score         = db.Column(db.Float)

    # Heart metrics
    resting_heart_rate  = db.Column(db.Float)
    avg_heart_rate      = db.Column(db.Float)
    hrv                 = db.Column(db.Float)
    spo2                = db.Column(db.Float)

    # Extra HR stats (needed by ML model)
    avg_hr_day          = db.Column(db.Float)
    hr_std              = db.Column(db.Float)
    hr_min              = db.Column(db.Integer)
    hr_max              = db.Column(db.Integer)

    # Extra SpO2
    spo2_min            = db.Column(db.Float)

    # Activity
    steps               = db.Column(db.Integer)
    active_minutes      = db.Column(db.Float)
    calories_burned     = db.Column(db.Float)

    # Scores
    energy_score        = db.Column(db.Float)    # 0–100, primary readiness metric

    # Rule engine categorical labels
    cognitive_score     = db.Column(db.String(16))
    fatigue_level       = db.Column(db.String(16))
    recovery_state      = db.Column(db.String(16))

    # Raw payload for debugging / re-processing
    raw_payload         = db.Column(db.Text)

    @property
    def readiness_color(self):
        s = self.energy_score
        if s is None:
            return "neutral"
        if s >= 70:
            return "high"
        if s >= 40:
            return "moderate"
        return "low"

    def __repr__(self):
        return f"<WatchReading {self.employee_id} {self.sleep_date}>"


class DailyAnalysis(db.Model):
    """LLM-generated analysis and task allocation for a daily reading."""
    __tablename__ = "daily_analyses"

    id              = db.Column(db.Integer, primary_key=True)
    reading_id      = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), unique=True)
    employee_id     = db.Column(db.String(255), nullable=False, index=True)
    generated_at    = db.Column(db.DateTime, default=datetime.utcnow)

    summary         = db.Column(db.Text)
    insights        = db.Column(db.Text)
    task_allocation = db.Column(db.Text)
    warnings        = db.Column(db.Text)
    full_response   = db.Column(db.Text)

    def __repr__(self):
        return f"<DailyAnalysis {self.employee_id} reading={self.reading_id}>"


class HeartRateReading(db.Model):
    __tablename__ = "heart_rate_readings"

    id          = db.Column(db.Integer, primary_key=True)
    reading_id  = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id = db.Column(db.String(255), nullable=False, index=True)
    timestamp   = db.Column(db.DateTime, nullable=False, index=True)
    bpm         = db.Column(db.Integer, nullable=False)


class OxygenSaturationReading(db.Model):
    __tablename__ = "oxygen_saturation_readings"

    id          = db.Column(db.Integer, primary_key=True)
    reading_id  = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id = db.Column(db.String(255), nullable=False, index=True)
    timestamp   = db.Column(db.DateTime, nullable=False, index=True)
    percentage  = db.Column(db.Float, nullable=False)


class StepReading(db.Model):
    __tablename__ = "step_readings"

    id          = db.Column(db.Integer, primary_key=True)
    reading_id  = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id = db.Column(db.String(255), nullable=False, index=True)
    start_time  = db.Column(db.DateTime, nullable=False, index=True)
    end_time    = db.Column(db.DateTime, nullable=False)
    count       = db.Column(db.Integer, nullable=False)
