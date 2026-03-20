"""Check what raw data comes in vs what gets stored"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, db
from models import WatchReading

with app.app_context():
    reading = WatchReading.query.first()
    if reading and reading.raw_payload:
        raw = json.loads(reading.raw_payload)
        
        print('=== RAW DATA STRUCTURE ===\n')
        
        print('What COMES IN webhook:')
        print('- heart_rate: ', end='')
        if raw.get('heart_rate'):
            print(f'array with {len(raw["heart_rate"])} readings')
            print(f'  First 2: {raw["heart_rate"][:2]}')
        else:
            print('N/A')
        
        print('\n- oxygen_saturation: ', end='')
        if raw.get('oxygen_saturation'):
            print(f'array with {len(raw["oxygen_saturation"])} readings')
            print(f'  First 2: {raw["oxygen_saturation"][:2]}')
        else:
            print('N/A')
        
        print('\n- steps: ', end='')
        if raw.get('steps'):
            print(f'array with {len(raw["steps"])} readings')
            print(f'  Example: {raw["steps"][:2]}')
        else:
            print('N/A')
        
        print('\n=== WHAT GETS STORED IN WatchReading TABLE ===\n')
        print(f'resting_heart_rate: {reading.resting_heart_rate} (10th percentile of overnight readings)')
        print(f'avg_heart_rate: {reading.avg_heart_rate} (average of overnight readings)')
        print(f'spo2: {reading.spo2} (average of all readings)')
        print(f'steps: {reading.steps} (total sum)')
        
        print('\n=== WHAT GETS DISCARDED (LOST DATA) ===')
        print('✗ Individual heart rate readings with timestamps')
        print('✗ Individual SpO2 readings with timestamps')
        print('✗ Individual step count readings with timestamps')
        print('✗ Time-series data for trends/analysis')
        print('✗ Detailed activity breakdown by time')
        print('\nRaw payload is stored, but individual readings are NOT parsed into DB')
