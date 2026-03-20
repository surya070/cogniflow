# Cogniflow 🧠

AI-powered workforce intelligence system. Uses wearable physiological data to assess employee cognitive readiness and dynamically allocate tasks.

## Setup

```bash
cd cogniflow
pip install -r requirements.txt

# Copy env file and add your Gemini API key
cp .env.example .env
# Edit .env → set GEMINI_API_KEY

python app.py
```

Server runs at `http://localhost:5000`

---

## Architecture

```
Watch (Health Auto Export)
        ↓ POST /webhook/watch
processor.py  →  structures raw payload
        ↓
rule_engine.py  →  scores cognitive state (rules based on sleep science)
        ↓
llm_service.py  →  Gemini 2.0 Flash analysis + task allocation
        ↓
SQLite DB  →  stores reading + analysis
        ↓
Dashboard  →  http://localhost:5000
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/webhook/watch?employee_id=emp_001` | Receive watch data |
| GET | `/` | Dashboard |
| GET | `/employee/<id>` | Employee detail + LLM analysis |
| GET | `/test/inject?employee_id=emp_001` | Inject sample data (no watch needed) |
| GET | `/api/employees` | List all employees (JSON) |
| GET | `/api/reading/<id>` | Single reading (JSON) |

---

## Adapting to Your Watch Payload

When you have your actual watch payload, open `processor.py` and update `_extract_fields()`.

The field paths follow dot notation:
```python
# If your payload is:
# { "sleep": { "totalSleepDuration": 460 } }
# Then:
total_sleep = get("sleep.totalSleepDuration")
```

---

## Watch Setup (Health Auto Export)

1. Install **Health Auto Export** on iPhone
2. Set automation: every day at 8am, POST to `http://<your-ip>:5000/webhook/watch`
3. Add `employee_id` as a query param or in the payload body

---

## Get Gemini API Key

1. Go to https://aistudio.google.com/
2. Create API key (free tier is sufficient)
3. Paste in `.env` as `GEMINI_API_KEY`
