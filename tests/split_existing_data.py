"""
Script to split existing WatchReading records that contain multiple sleep sessions.
Processes raw_payload JSON to identify separate sleep sessions and creates individual records.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, db
from models import WatchReading, DailyAnalysis
from services.processor import process_payload, parse_datetime

STAGE_DEEP  = {"6"}
STAGE_REM   = {"5"}
STAGE_LIGHT = {"4"}
STAGE_AWAKE = {"1"}


def split_record_by_sleep_sessions(reading: WatchReading) -> list:
    """
    Take a WatchReading with multiple sleep sessions and create new records.
    Returns list of new WatchReading objects (not yet committed).
    """
    try:
        raw = json.loads(reading.raw_payload)
    except (json.JSONDecodeError, TypeError):
        print(f"  ✗ Cannot parse raw_payload for reading {reading.id}")
        return []
    
    sleep_sessions = raw.get("sleep") or []
    
    # If only 0 or 1 session, no split needed
    if len(sleep_sessions) <= 1:
        return []
    
    print(f"  → Found {len(sleep_sessions)} sleep sessions in reading {reading.id}")
    
    new_readings = []
    for idx, session in enumerate(sleep_sessions):
        # Create a new payload with just this session
        split_payload = raw.copy()
        split_payload["sleep"] = [session]
        
        # Process the split payload
        fields = process_payload(split_payload, reading.employee_id)
        
        # Create new WatchReading
        new_reading = WatchReading(
            employee_id=reading.employee_id,
            received_at=reading.received_at,
            sleep_date=fields["sleep_date"],
            sleep_start=fields["sleep_start"],
            sleep_end=fields["sleep_end"],
            total_sleep_minutes=fields["total_sleep_minutes"],
            deep_sleep_minutes=fields["deep_sleep_minutes"],
            rem_sleep_minutes=fields["rem_sleep_minutes"],
            light_sleep_minutes=fields["light_sleep_minutes"],
            awake_minutes=fields["awake_minutes"],
            sleep_score=fields["sleep_score"],
            resting_heart_rate=fields["resting_heart_rate"],
            avg_heart_rate=fields["avg_heart_rate"],
            hrv=fields["hrv"],
            spo2=fields["spo2"],
            steps=fields["steps"],
            active_minutes=fields["active_minutes"],
            calories_burned=fields["calories_burned"],
            stress_score=fields["stress_score"],
            raw_payload=fields["raw_payload"],
            # Copy cognitive assessment if set
            cognitive_score=reading.cognitive_score,
            fatigue_level=reading.fatigue_level,
            recovery_state=reading.recovery_state,
        )
        new_readings.append(new_reading)
        print(f"    → Session {idx+1}: {fields['sleep_date']}")
    
    return new_readings


def main():
    with app.app_context():
        # Get all readings
        readings = WatchReading.query.all()
        print(f"\nProcessing {len(readings)} WatchReading records...\n")
        
        total_split = 0
        records_to_delete = []
        
        for reading in readings:
            try:
                raw = json.loads(reading.raw_payload)
            except (json.JSONDecodeError, TypeError):
                continue
            
            sleep_sessions = raw.get("sleep") or []
            
            # Only process records with multiple sessions
            if len(sleep_sessions) > 1:
                print(f"Reading {reading.id} ({reading.employee_id} - {reading.sleep_date}):")
                
                # Split the record
                new_readings = split_record_by_sleep_sessions(reading)
                
                if new_readings:
                    # Add new records to session
                    for new_reading in new_readings:
                        db.session.add(new_reading)
                    
                    # Mark original for deletion
                    records_to_delete.append(reading.id)
                    total_split += 1
        
        # Commit new records first
        if records_to_delete:
            print(f"\n✓ Created {len(records_to_delete)} split records")
            try:
                db.session.commit()
                print("✓ Committed new records to database")
            except Exception as e:
                db.session.rollback()
                print(f"✗ Error committing new records: {e}")
                return
            
            # Now delete the original combined records
            print(f"\nDeleting {len(records_to_delete)} original combined records...")
            for reading_id in records_to_delete:
                reading = WatchReading.query.get(reading_id)
                if reading:
                    # Delete associated analysis first
                    analysis = DailyAnalysis.query.filter_by(reading_id=reading_id).first()
                    if analysis:
                        db.session.delete(analysis)
                        print(f"  → Deleted analysis for reading {reading_id}")
                    
                    # Delete the reading
                    db.session.delete(reading)
            
            try:
                db.session.commit()
                print("✓ Deleted original combined records")
            except Exception as e:
                db.session.rollback()
                print(f"✗ Error deleting records: {e}")
                return
            
            print(f"\n✓ Successfully split {total_split} record(s)")
        else:
            print("\n✓ No records with multiple sleep sessions found")


if __name__ == "__main__":
    main()
