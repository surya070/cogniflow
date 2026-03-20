from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class WatchReading(db.Model):
    """Raw processed watch data — one row per daily sync"""
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

    # Raw payload stored for debugging
    raw_payload         = db.Column(db.Text)

    # Computed by our rule engine
    cognitive_score     = db.Column(db.String(16))   # high / moderate / low
    fatigue_level       = db.Column(db.String(16))   # low / moderate / high
    recovery_state      = db.Column(db.String(16))   # recovered / partial / poor

    analysis            = db.relationship("DailyAnalysis", backref="reading", uselist=False)

    def __repr__(self):
        return f"<WatchReading {self.employee_id} {self.sleep_date}>"


class DailyAnalysis(db.Model):
    """LLM-generated analysis and task allocation for a daily reading"""
    __tablename__ = "daily_analyses"

    id              = db.Column(db.Integer, primary_key=True)
    reading_id      = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), unique=True)
    employee_id     = db.Column(db.String(64), nullable=False, index=True)
    generated_at    = db.Column(db.DateTime, default=datetime.utcnow)

    summary         = db.Column(db.Text)       # Short LLM narrative
    insights        = db.Column(db.Text)       # Bullet-point health insights
    task_allocation = db.Column(db.Text)       # Recommended task types
    warnings        = db.Column(db.Text)       # Burnout / risk alerts
    full_response   = db.Column(db.Text)       # Full LLM JSON response

    def __repr__(self):
        return f"<DailyAnalysis {self.employee_id} reading={self.reading_id}>"


class HeartRateReading(db.Model):
    """Individual heart rate readings with timestamps"""
    __tablename__ = "heart_rate_readings"

    id              = db.Column(db.Integer, primary_key=True)
    reading_id      = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id     = db.Column(db.String(64), nullable=False, index=True)
    timestamp       = db.Column(db.DateTime, nullable=False, index=True)
    bpm             = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<HeartRateReading {self.bpm}bpm at {self.timestamp}>"


class OxygenSaturationReading(db.Model):
    """Individual SpO2 readings with timestamps"""
    __tablename__ = "oxygen_saturation_readings"

    id              = db.Column(db.Integer, primary_key=True)
    reading_id      = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id     = db.Column(db.String(64), nullable=False, index=True)
    timestamp       = db.Column(db.DateTime, nullable=False, index=True)
    percentage      = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"<OxygenSaturationReading {self.percentage}% at {self.timestamp}>"


class StepReading(db.Model):
    """Step count readings with time windows"""
    __tablename__ = "step_readings"

    id              = db.Column(db.Integer, primary_key=True)
    reading_id      = db.Column(db.Integer, db.ForeignKey("watch_readings.id"), nullable=False, index=True)
    employee_id     = db.Column(db.String(64), nullable=False, index=True)
    start_time      = db.Column(db.DateTime, nullable=False, index=True)
    end_time        = db.Column(db.DateTime, nullable=False)
    count           = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"<StepReading {self.count} steps from {self.start_time} to {self.end_time}>"
