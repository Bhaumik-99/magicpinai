# magicpin AI Challenge — Local Bot

This repository includes a minimal **candidate bot** implementation (`bot.py`) that exposes the required HTTP endpoints and composes **deterministic** messages from the pushed contexts.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

In another terminal:

```powershell
# judge_simulator.py requires an LLM API key to score messages,
# but the warmup + behavioral scenarios will still hit your endpoints.
python judge_simulator.py
```

## What’s implemented

- `GET /v1/healthz`
- `GET /v1/metadata`
- `POST /v1/context` (idempotent by version, in-memory store)
- `POST /v1/tick` (dispatch by `trigger.kind`, suppression-key dedup)
- `POST /v1/reply` (auto-reply detection, hostile stop, intent commitment handling)

