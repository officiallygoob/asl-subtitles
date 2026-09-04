# ASL Recognition Server

FastAPI service that accepts **landmark geometry only** (hands / body / face) and returns gloss + English.

> **Honesty:** open-domain conversational ASL→English is an unsolved research problem. This server ships a **limited-domain continuous recognition** pipeline plus a clean plug-in path for Uni-Sign / PoseLSTM weights. It is **not** Google DeepMind SL2T.

## Privacy

- Endpoints never accept video frames.
- Clients should stream `LandmarkFrame` JSON (joints + timestamps).
- Run on localhost / LAN for friend conversations.

## Quick start

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional ML stack for PoseLSTM / Uni-Sign inspect:
# pip install -r requirements-ml.txt
uvicorn main:app --host 0.0.0.0 --port 8765
```

Or:

```bash
cd server
docker compose up --build
```

Health check: `curl http://127.0.0.1:8765/health`

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness + loaded model name |
| `POST /v1/translate` | Buffered window → `{gloss, english, confidence}` |
| `WS /v1/stream` | Continuous landmark stream → partial/final JSON |

### WebSocket client messages

- `{ "type": "hello", "protocolVersion": 1, "sampleRateHz": 15, "windowFrames": 36 }`
- `{ "type": "frame", "frame": { ...LandmarkFrame } }`
- `{ "type": "utterance_end", "frames": [ ... ] }`

### Server messages

- `{ "type": "welcome"|"status", "model": "...", "connected": true }`
- `{ "type": "partial"|"final", "gloss": [...], "english": "...", "confidence": 0.5, "isFinal": bool }`

## Models

See root [`MODELS.md`](../MODELS.md).

**Shipping default:** `models/sign_classifier.pt` (PoseLSTM, ~73 conversational glosses, synthetic pretrain).

Also supported:

- `poselstm.pt` / `model.onnx` / optional `labels.json`
- Uni-Sign `*.pth` — detected on `/health` only (architecture mismatch; not inferred)

Rebuild:

```bash
pip install -r requirements-ml.txt
python scripts/synthesize_pose_dataset.py
python scripts/train_poselstm.py
```

Without weights, a **demo continuous decoder** runs so Conversation Mode works end-to-end.

## Offline eval

```bash
python eval_demo.py samples/example_sequence.json
```

## Optional gloss→English LLM

```bash
export ASL_GLOSS_LLM_CMD='ollama run llama3.2'
```

No API keys are stored in the repo.
