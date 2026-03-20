"""
Script to reprocess existing WatchReading records with corrected stage mapping.
"""
import json
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app, db
from models import WatchReading
from services.processor import process_payload

def main():
    with app.app_context():
        readings = WatchReading.query.all()
        print(f"\nReprocessing {len(readings)} WatchReading records with corrected stage mapping...\n")
        
        updated_count = 0
        
        for reading in readings:
            try:
                raw = json.loads(reading.raw_payload)
            except (json.JSONDecodeError, TypeError):
                continue
            
            # Reprocess with corrected stage mapping
            fields = process_payload(raw, reading.employee_id)
            
            # Update the reading with new values
            reading.total_sleep_minutes = fields["total_sleep_minutes"]
            reading.deep_sleep_minutes = fields["deep_sleep_minutes"]
            reading.rem_sleep_minutes = fields["rem_sleep_minutes"]
            reading.light_sleep_minutes = fields["light_sleep_minutes"]
            reading.awake_minutes = fields["awake_minutes"]
            
            updated_count += 1
            
            sleep_date = reading.sleep_date
            old_total = (reading.deep_sleep_minutes or 0) + (reading.rem_sleep_minutes or 0) + (reading.light_sleep_minutes or 0)
            print(f"Reading {reading.id} ({sleep_date}): Updated sleep stages")
        
        if updated_count > 0:
            try:
                db.session.commit()
                print(f"\n✓ Successfully reprocessed {updated_count} records with corrected stage mapping")
            except Exception as e:
                db.session.rollback()
                print(f"✗ Error saving changes: {e}")
        else:
            print("\n✓ No records to reprocess")

if __name__ == "__main__":
    main()
