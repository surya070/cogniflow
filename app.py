import json
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from models import DailyAnalysis, WatchReading, HeartRateReading, OxygenSaturationReading, StepReading, db
from services.processor import process_payload, split_payload_by_sleep_sessions, extract_heart_rate_readings, extract_oxygen_readings, extract_step_readings
from rule_engine import score_cognitive_state
from services.llm_service import analyze_reading

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "cogniflow-dev")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///cogniflow.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()

# Custom Jinja filters
@app.template_filter('to_ist')
def to_ist(dt):
    """Convert UTC datetime to IST (UTC+5:30)"""
    if not dt:
        return None
    from datetime import timedelta
    ist_time = dt + timedelta(hours=5, minutes=30)
    return ist_time

@app.template_filter('ist_time')
def ist_time(dt):
    """Convert UTC datetime to IST and format as HH:MM"""
    if not dt:
        return '—'
    ist = to_ist(dt)
    return ist.strftime('%H:%M')

@app.template_filter('format_duration')
def format_duration(minutes):
    """Format minutes as 'Xhr Ym' format"""
    if not minutes:
        return '—'
    mins = int(minutes)
    hrs = mins // 60
    mins = mins % 60
    if hrs > 0 and mins > 0:
        return f'{hrs}hr {mins}m'
    elif hrs > 0:
        return f'{hrs}hr'
    else:
        return f'{mins}m'

# Create webhook logs directory if it doesn't exist
WEBHOOK_LOGS_DIR = "webhook_logs"
if not os.path.exists(WEBHOOK_LOGS_DIR):
    os.makedirs(WEBHOOK_LOGS_DIR)


def _save_webhook_payload(payload: dict) -> str:
    """
    Save incoming webhook payload to a JSON file with timestamp as filename.
    Returns the filename.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Includes milliseconds
    filename = f"{timestamp}.json"
    filepath = os.path.join(WEBHOOK_LOGS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(payload, f, indent=2)
    
    return filename


def _run_pipeline(raw: dict, employee_id: str) -> dict:
    """
    Full pipeline: processor -> rule engine -> LLM -> DB.
    Called directly (no HTTP self-call) to avoid timeout issues.
    """
    fields = process_payload(raw, employee_id)

    # Deduplication: use sleep_end as the unique identifier
    # (along with employee_id) since multiple sessions can occur on same date
    existing = None
    if fields.get("sleep_end"):
        existing = WatchReading.query.filter_by(
            employee_id=employee_id,
            sleep_end=fields["sleep_end"]
        ).first()
    
    # Fallback: if no sleep_end, try sleep_date (for backward compatibility)
    if not existing and fields.get("sleep_date"):
        # Only fallback if there's just one record for that date
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
    
    # ── Store detailed readings ──────────────────────────────
    # Delete old detailed readings for this sleep session (if updating)
    if existing:
        HeartRateReading.query.filter_by(reading_id=reading.id).delete()
        OxygenSaturationReading.query.filter_by(reading_id=reading.id).delete()
        StepReading.query.filter_by(reading_id=reading.id).delete()
    
    # Extract and store heart rate readings
    hr_readings = extract_heart_rate_readings(raw)
    for hr_data in hr_readings:
        hr_reading = HeartRateReading(
            reading_id=reading.id,
            employee_id=employee_id,
            timestamp=hr_data["timestamp"],
            bpm=hr_data["bpm"]
        )
        db.session.add(hr_reading)
    
    # Extract and store oxygen saturation readings
    spo2_readings = extract_oxygen_readings(raw)
    for spo2_data in spo2_readings:
        spo2_reading = OxygenSaturationReading(
            reading_id=reading.id,
            employee_id=employee_id,
            timestamp=spo2_data["timestamp"],
            percentage=spo2_data["percentage"]
        )
        db.session.add(spo2_reading)
    
    # Extract and store step readings
    step_readings = extract_step_readings(raw)
    for step_data in step_readings:
        step_reading = StepReading(
            reading_id=reading.id,
            employee_id=employee_id,
            start_time=step_data["start_time"],
            end_time=step_data["end_time"],
            count=step_data["count"]
        )
        db.session.add(step_reading)
    
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


# -------------------------------------------------------------------
# WEBHOOK
# -------------------------------------------------------------------

@app.route("/webhook/watch",  methods=["POST"])
def receive_watch_data():
    """
    Accepts single JSON object or array (same as your old code).
    Falls back to manual JSON parse if Content-Type header is wrong
    (common issue with watch apps sending data).
    Saves all incoming payloads to webhook_logs directory with timestamp filenames.
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
    
    # Save the webhook payload with timestamp filename
    saved_filename = _save_webhook_payload(payload)

    employee_id = (
        request.args.get("employee_id")
        or (payload.get("employee_id") if isinstance(payload, dict) else None)
        or "unknown"
    )

    if isinstance(payload, dict):
        # Split payload by sleep sessions if it contains multiple
        split_payloads = split_payload_by_sleep_sessions(payload)
        results = []
        for split_payload in split_payloads:
            results.append(_run_pipeline(split_payload, employee_id))
        # Return the last result but include all results
        response = results[-1] if results else {"status": "error"}
        response["webhook_log_file"] = saved_filename
        if len(results) > 1:
            response["note"] = f"Split and processed {len(results)} sleep sessions"
        return jsonify(response), 200

    elif isinstance(payload, list):
        results = []
        for item in payload:
            if isinstance(item, dict):
                results.append(_run_pipeline(item, employee_id))
        if not results:
            return jsonify({"error": "Array contained no valid objects"}), 400
        results[-1]["webhook_log_file"] = saved_filename
        return jsonify(results[-1]), 200

    return jsonify({"error": "Payload must be a JSON object or array"}), 400


# -------------------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------------------

@app.route("/")
def dashboard():
    cutoff = datetime.utcnow() - timedelta(days=7)
    readings = (
        WatchReading.query
        .filter(WatchReading.received_at >= cutoff)
        .order_by(WatchReading.received_at.desc())
        .all()
    )
    seen = set()
    latest = []
    for r in readings:
        if r.employee_id not in seen:
            seen.add(r.employee_id)
            latest.append(r)
    return render_template("dashboard.html", readings=latest)


# -------------------------------------------------------------------
# LAST RECEIVED
# -------------------------------------------------------------------

@app.route("/last-received")
def last_received():
    raw_payload = "{}"
    filename = "No files received yet"
    
    # Find the most recent JSON file in webhook_logs directory
    if os.path.exists(WEBHOOK_LOGS_DIR):
        json_files = [f for f in os.listdir(WEBHOOK_LOGS_DIR) if f.endswith('.json')]
        if json_files:
            # Sort by filename (timestamp-based) and get the latest
            json_files.sort(reverse=True)
            latest_file = json_files[0]
            filename = latest_file
            
            try:
                filepath = os.path.join(WEBHOOK_LOGS_DIR, latest_file)
                with open(filepath, 'r') as f:
                    payload = json.load(f)
                    raw_payload = json.dumps(payload, indent=2)
            except Exception as e:
                raw_payload = f"Error reading file: {str(e)}"
    
    return render_template("last_received.html", filename=filename, raw_payload=raw_payload)


@app.route("/employee/<employee_id>")
def employee_detail(employee_id):
    readings = (
        WatchReading.query
        .filter_by(employee_id=employee_id)
        .order_by(WatchReading.sleep_date.desc())
        .limit(14)
        .all()
    )
    latest   = readings[0] if readings else None
    analysis = latest.analysis if latest else None

    insights = []
    task_alloc = {}
    warnings = []
    readiness_label = "-"

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


# -------------------------------------------------------------------
# JSON API
# -------------------------------------------------------------------

@app.route("/api/employees")
def api_employees():
    rows = db.session.query(WatchReading.employee_id).distinct().all()
    return jsonify([r[0] for r in rows])


@app.route("/api/reading/<int:reading_id>")
def api_reading(reading_id):
    r = WatchReading.query.get_or_404(reading_id)
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


# -------------------------------------------------------------------
# HEART RATE ANALYSIS
# -------------------------------------------------------------------

@app.route("/employee/<employee_id>/heart-rate")
def heart_rate_analysis(employee_id):
    """Display heart rate trend analysis page"""
    readings = (
        WatchReading.query
        .filter_by(employee_id=employee_id)
        .order_by(WatchReading.sleep_date.desc())
        .limit(30)
        .all()
    )
    
    if not readings:
        return render_template("hr_analysis.html", employee_id=employee_id, readings=[], data=None)
    
    return render_template("hr_analysis.html", employee_id=employee_id, readings=readings)


@app.route("/api/employee/<employee_id>/heart-rate")
def api_heart_rate(employee_id):
    """
    Get hourly aggregated heart rate data.
    
    Query params:
    - date: YYYY-MM-DD (optional, defaults to most recent reading)
    - reading_id: specific reading ID (optional)
    
    Returns hourly data with avg, min, max BPM
    """
    reading_id = request.args.get('reading_id', type=int)
    date_str = request.args.get('date')
    
    # If date provided, find reading for that date
    if date_str and not reading_id:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        reading = WatchReading.query.filter_by(
            employee_id=employee_id,
            sleep_date=target_date
        ).first()
        
        if not reading:
            return jsonify({"error": "No reading found for that date"}), 404
        reading_id = reading.id
    
    # If no date/reading specified, use most recent
    if not reading_id:
        reading = WatchReading.query.filter_by(
            employee_id=employee_id
        ).order_by(WatchReading.sleep_date.desc()).first()
        
        if not reading:
            return jsonify({"error": "No readings found for this employee"}), 404
        reading_id = reading.id
    
    # Get the reading object to get sleep time bounds
    reading_data = WatchReading.query.get(reading_id)
    if not reading_data:
        return jsonify({"error": "Reading not found"}), 404
    
    # Get all heart rate readings for this sleep session
    # Filter by reading_id AND by the actual sleep time window (sleep_start to sleep_end)
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
    
    # Aggregate by hour (within the sleep session only)
    hourly_data = {}
    for hr in hr_readings:
        hour_key = hr.timestamp.strftime("%H:00")
        if hour_key not in hourly_data:
            hourly_data[hour_key] = []
        hourly_data[hour_key].append(hr.bpm)
    
    # Calculate stats for each hour
    hourly_stats = []
    for hour in sorted(hourly_data.keys()):
        bpms = hourly_data[hour]
        hourly_stats.append({
            "hour": hour,
            "avg": round(sum(bpms) / len(bpms), 1),
            "min": min(bpms),
            "max": max(bpms),
            "count": len(bpms)
        })
    
    return jsonify({
        "reading_id": reading_id,
        "employee_id": employee_id,
        "date": reading_data.sleep_date,
        "sleep_start": reading_data.sleep_start.isoformat() if reading_data.sleep_start else None,
        "sleep_end": reading_data.sleep_end.isoformat() if reading_data.sleep_end else None,
        "total_readings": len(hr_readings),
        "resting_hr": reading_data.resting_heart_rate if reading_data else None,
        "avg_hr": reading_data.avg_heart_rate if reading_data else None,
        "hourly": hourly_stats
    })


# -------------------------------------------------------------------
# TEST — in-process injection, no HTTP self-call (fixes the timeout)
# -------------------------------------------------------------------

@app.route("/test/inject")
def test_inject():
    """
    GET /test/inject?employee_id=emp_001
    Runs full pipeline with sample data. No external HTTP call.
    """
    employee_id = request.args.get("employee_id", "emp_001")
    sample = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT08:00:00Z"),
        "sleep": [
            {
                "stages": [
                    {"stage": 4, "duration_seconds": 5700},
                    {"stage": 3, "duration_seconds": 6000},
                    {"stage": 2, "duration_seconds": 12600},
                    {"stage": 1, "duration_seconds": 1800},
                ]
            }
        ],
        "heart_rate": [
            {"bpm": 56}, {"bpm": 58}, {"bpm": 57},
            {"bpm": 60}, {"bpm": 62}, {"bpm": 59},
        ],
        "oxygen_saturation": [{"percentage": 97}],
        "steps": [{"count": 8200}],
    }
    result = _run_pipeline(sample, employee_id)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)