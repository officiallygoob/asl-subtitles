# ASL Subtitles — Conversation Mode

Real conversations with a deaf friend: **live English captions from signing**, plus a **speech→text** panel so you can speak and they can read.

Inspired by privacy-preserving landmark architectures (MediaPipe Holistic → stream geometry only → continuous SLT), similar in *shape* to research systems like DeepMind SL2T / Uni-Sign — **we do not include or claim Google’s model**.

> **Honesty:** open-domain fluent ASL→English chat is **unsolved**. This app ships a **limited-domain continuous recognition** architecture + offline heuristics. The path to usefulness is continuous streaming + **friend adaptation** (record → Create ML / fine-tune), not marketing a toy phrase spotter as “fluent.”

## Architecture

```
┌───────────────────────── iPhone (iOS 17+) ─────────────────────────┐
│  Camera → Vision holistic landmarks (hands + body + face)           │
│       ↓ discard pixels                                              │
│  LandmarkFrame buffer (~36 frames) + utterance segmentation         │
│       ├── WebSocket stream ──► Recognition server (LAN)             │
│       └── offline fallback: Core ML (if present) / heuristics       │
│                                                                     │
│  Mic → SFSpeechRecognizer → "You said" transcript (reverse channel) │
│                                                                     │
│  Conversation Mode UI: Signing captions + history + You said        │
└─────────────────────────────────────────────────────────────────────┘
                              │ landmarks only (no video)
                              ▼
┌──────────────────── server/ (FastAPI) ──────────────────────────────┐
│  WS /v1/stream  ·  POST /v1/translate  ·  GET /health               │
│  normalize → PoseLSTM / Uni-Sign plug-in / demo decoder → gloss     │
│  gloss → English (rules + optional local LLM)                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Requirements

- Mac with **Xcode 15+**, iPhone/iPad **iOS 17+** (physical device recommended)
- Python 3.11+ **or** Docker for the recognition server
- Same Wi‑Fi / LAN when using the server from a real device

## Run the recognition server

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8765
```

Or: `cd server && docker compose up --build`

Check: `curl http://127.0.0.1:8765/health`

On a physical iPhone, set **Settings → Continuous recognition** to  
`ws://<your-mac-lan-ip>:8765/v1/stream` (not `127.0.0.1`).

## Run the iOS app

1. Open `ASLSubtitles.xcodeproj` in Xcode.
2. Select the **ASLSubtitles** scheme → your iPhone.
3. Signing: Team = your Personal Team (`com.officiallygoob.aslsubtitles`).
4. Build & Run (⌘R). Allow **Camera**, **Microphone**, and **Speech Recognition**.
5. Conversation Mode is the default screen.

### First conversation tips

- Point the rear camera at your friend; keep torso + hands in frame.
- Toggle **Mic** so your speech appears as “You said.”
- **Debug** draws holistic landmarks.
- Without a server, the app stays useful via **offline fallback** (heuristics / optional Core ML).

## What’s included

| Area | Implementation |
|----------------------|
| Conversation Mode | Signing captions + speech transcript + scroll history |
| Holistic landmarks | Vision hand pose + body pose + face landmarks → `LandmarkFrame` |
| Temporal buffer | ~36 frames (~1–2 s) + pause/rest utterance segmentation |
| Streaming client | WebSocket landmark protocol (`RecognitionClient`) |
| Speech → text | `SFSpeechRecognizer` reverse channel |
| Offline fallback | Heuristics + optional `CoreMLSignClassifier` |
| LandmarkRecorder | Export labeled JSON/JSONL for Create ML / fine-tunes |
| Server | FastAPI WS + REST, PoseLSTM scaffold, gloss→English, Docker |

## Friend adaptation (recommended next step)

1. Settings → **Landmark training** → record labeled glosses with your friend.
2. Export JSONL → train Create ML Hand Action (or fine-tune server weights).
3. Install `ASLSignClassifier.mlmodel` on device **or** drop `uni_sign.pt` into `server/models/`.

Details: [`MODELS.md`](MODELS.md) · server: [`server/README.md`](server/README.md)

## Limitations

- Not an interpreter. Limited gloss domain until you add weights / friend data.
- Ambiguous fingerspelling and facial grammar remain hard.
- Demo server decoder is for protocol + limited glosses — replace with Uni-Sign/PoseLSTM for real accuracy (see MODELS.md).
- Simulator has no real camera/hands; use a device.

## License / care

Personal / educational use. Treat errors carefully; never rely on this for safety-critical communication.
