import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

# List all routes
print("Available routes:")
for rule in app.url_map.iter_rules():
    print(f'{rule.rule} | Methods: {rule.methods}')
