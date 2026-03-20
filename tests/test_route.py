import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app import app
    print("App imported successfully!")
    
    # Test if route exists
    with app.test_client() as client:
        resp = client.get('/api/employee/test_emp/heart-rate?reading_id=3')
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.data.decode()[:200]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
