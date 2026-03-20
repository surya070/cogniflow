import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, db
from models import HeartRateReading, WatchReading

with app.app_context():
    # Check readings
    reading = WatchReading.query.get(3)
    print(f"Reading ID 3: {reading}")
    if reading:
        print(f" - Employee: {reading.employee_id}")
        print(f" - Date: {reading.sleep_date}")
        print(f" - Avg HR: {reading.avg_heart_rate}")
    
    # Check heart rate readings
    hr_count = HeartRateReading.query.filter_by(reading_id=3).count()
    print(f"\nHeart rate readings for reading_id=3: {hr_count}")
    
    hr_sample = HeartRateReading.query.filter_by(reading_id=3).first()
    if hr_sample:
        print(f"Sample HR: {hr_sample.timestamp} = {hr_sample.bpm} bpm")
