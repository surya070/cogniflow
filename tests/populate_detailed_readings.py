"""
Populate the new detailed reading tables from existing raw payloads.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app, db
from models import WatchReading, HeartRateReading, OxygenSaturationReading, StepReading
from services.processor import extract_heart_rate_readings, extract_oxygen_readings, extract_step_readings

def populate_detailed_readings():
    with app.app_context():
        readings = WatchReading.query.all()
        print(f"Processing {len(readings)} existing WatchReading records...\n")
        
        total_hr = 0
        total_spo2 = 0
        total_steps = 0
        
        for reading in readings:
            try:
                raw = json.loads(reading.raw_payload)
            except (json.JSONDecodeError, TypeError):
                print(f"⚠ Cannot parse raw_payload for reading {reading.id}")
                continue
            
            # Extract readings
            hr_readings = extract_heart_rate_readings(raw)
            spo2_readings = extract_oxygen_readings(raw)
            step_readings = extract_step_readings(raw)
            
            print(f"Reading {reading.id} ({reading.employee_id} - {reading.sleep_date}):")
            print(f"  Heart Rate: {len(hr_readings)} readings")
            print(f"  SpO2: {len(spo2_readings)} readings")
            print(f"  Steps: {len(step_readings)} readings")
            
            # Store heart rate readings
            for hr_data in hr_readings:
                hr_reading = HeartRateReading(
                    reading_id=reading.id,
                    employee_id=reading.employee_id,
                    timestamp=hr_data["timestamp"],
                    bpm=hr_data["bpm"]
                )
                db.session.add(hr_reading)
                total_hr += 1
            
            # Store SpO2 readings
            for spo2_data in spo2_readings:
                spo2_reading = OxygenSaturationReading(
                    reading_id=reading.id,
                    employee_id=reading.employee_id,
                    timestamp=spo2_data["timestamp"],
                    percentage=spo2_data["percentage"]
                )
                db.session.add(spo2_reading)
                total_spo2 += 1
            
            # Store step readings
            for step_data in step_readings:
                step_reading = StepReading(
                    reading_id=reading.id,
                    employee_id=reading.employee_id,
                    start_time=step_data["start_time"],
                    end_time=step_data["end_time"],
                    count=step_data["count"]
                )
                db.session.add(step_reading)
                total_steps += 1
        
        db.session.commit()
        
        print(f"\n✓ Successfully stored:")
        print(f"  {total_hr} heart rate readings")
        print(f"  {total_spo2} SpO2 readings")
        print(f"  {total_steps} step readings")

if __name__ == "__main__":
    populate_detailed_readings()
