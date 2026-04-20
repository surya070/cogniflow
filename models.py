import secrets
from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(db.Model, UserMixin):
    """User account — linked to watch readings via employee_id."""
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name          = db.Column(db.String(128), nullable=False)

    # Auth — password_hash is NULL for Google-only accounts
    password_hash = db.Column(db.String(256))
    google_id     = db.Column(db.String(128), unique=True, index=True)
    avatar_url    = db.Column(db.String(512))

    # Each user gets a personal webhook token (used in the watch-app URL)
    # and a stable employee_id that links to all their WatchReading rows.
    webhook_token = db.Column(db.String(64), unique=True, nullable=False, index=True)
    employee_id   = db.Column(db.String(64), unique=True, nullable=False, index=True)

    # role: "employee" (own data only) / "manager" (all data) / "admin" (all + admin UI)
    role          = db.Column(db.String(32), default="employee")
    is_active     = db.Column(db.Boolean, default=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

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
    def make_employee_id() -> str:
        return "emp_" + secrets.token_hex(3)

    @staticmethod
    def make_webhook_token() -> str:
        return secrets.token_urlsafe(32)

    def __repr__(self):
        return f"<User {self.email} ({self.employee_id})>"


class WatchReading(db.Model):
    """Raw processed watch data — one row per sleep session."""
    __tablename__ = "watch_readings"

    id              = db.Column(db.Integer, primary_key=True)
    employee_id     = db.Column(db.String(64), nullable=False, index=True)
    received_at     = db.Column(db.DateTime, default=datetime.utcnow)

    # Sleep metrics
    sleep_date          = db.Column(db.String(20))   # YYYY-MM-DD of the night
    sleep_start         = db.Column(db.DateTime)
    sleep_end           = db.Column(db.DateTime)
    total_sleep_minutes = db.Column(db.Float)
    deep_sleep_minutes  = db.Column(db.Float)
    rem_sleep_minutes   = db.Column(db.Float)
    light_sleep_minutes = db.Column(db.Float)
    awake_minutes       = db.Column(db.Float)
    sleep_score         = db.Column(db.Float)        # 0–100 if watch provides it

    # Heart metrics
    resting_heart_rate  = db.Column(db.Float)
    avg_heart_rate      = db.Column(db.Float)
    hrv                 = db.Column(db.Float)        # Heart Rate Variability (ms)
    spo2                = db.Column(db.Float)        # Blood oxygen %

    # Activity
    steps               = db.Column(db.Integer)
    active_minutes      = db.Column(db.Float)
    calories_burned     = db.Column(db.Float)

    # Stress
    stress_score        = db.Column(db.Float)        # 0–100 if watch provides

    # Raw payload stored for debugging / re-processing
    raw_payload         = db.Column(db.Text)

    # Computed by our rule engine
    cognitive_score     = db.Column(db.String(16))   # high / moderate / low
    fatigue_level       = db.Column(db.String(16))   # low / moderate / high
    recovery_state      = db.Column(db.String(16))   # recovered / partial / poor

    analysis            = db.relationship("DailyAnalysis", backref="reading", uselist=False)

    def __repr__(self):
        return f"<WatchReading {self.employee_id} {self.sleep_date}>"


class DailyAnalysis(db.Model):
    """LLM-generated analysis and task allocation for a daily reading."""
    __tablename__ = "daily_analyses"

    id              = db.Column(db.Integer, primary_key=True)
    reading_id      = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), unique=True)
    employee_id     = db.Column(db.String(64), nullable=False, index=True)
    generated_at    = db.Column(db.DateTime, default=datetime.utcnow)

    summary         = db.Column(db.Text)       # Short LLM narrative
    insights        = db.Column(db.Text)       # JSON array of bullet-point insights
    task_allocation = db.Column(db.Text)       # JSON with recommended/avoid tasks
    warnings        = db.Column(db.Text)       # JSON array of burnout / risk alerts
    full_response   = db.Column(db.Text)       # Full LLM JSON response

    def __repr__(self):
        return f"<DailyAnalysis {self.employee_id} reading={self.reading_id}>"


class HeartRateReading(db.Model):
    """Individual heart rate readings with timestamps."""
    __tablename__ = "heart_rate_readings"

    id          = db.Column(db.Integer, primary_key=True)
    reading_id  = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id = db.Column(db.String(64), nullable=False, index=True)
    timestamp   = db.Column(db.DateTime, nullable=False, index=True)
    bpm         = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<HeartRateReading {self.bpm}bpm at {self.timestamp}>"


class OxygenSaturationReading(db.Model):
    """Individual SpO2 readings with timestamps."""
    __tablename__ = "oxygen_saturation_readings"

    id          = db.Column(db.Integer, primary_key=True)
    reading_id  = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id = db.Column(db.String(64), nullable=False, index=True)
    timestamp   = db.Column(db.DateTime, nullable=False, index=True)
    percentage  = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"<OxygenSaturationReading {self.percentage}% at {self.timestamp}>"


class StepReading(db.Model):
    """Step count readings with time windows."""
    __tablename__ = "step_readings"

    id          = db.Column(db.Integer, primary_key=True)
    reading_id  = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id = db.Column(db.String(64), nullable=False, index=True)
    start_time  = db.Column(db.DateTime, nullable=False, index=True)
    end_time    = db.Column(db.DateTime, nullable=False)
    count       = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<StepReading {self.count} steps {self.start_time}–{self.end_time}>"
