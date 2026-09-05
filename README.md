# ASL Subtitles — Conversation Mode

Real conversations with a deaf friend: **live English captions from signing**, plus a **speech→text** panel so you can speak and they can read.

Inspired by privacy-preserving landmark architectures (MediaPipe Holistic → stream geometry only → continuous SLT), similar in *shape* to research systems like DeepMind SL2T / Uni-Sign — **we do not include or claim Google’s model**.

> **Honesty:** open-domain fluent ASL→English chat is **unsolved**. This app ships a **limited-domain continuous recognition** architecture + offline heuristics. The path to usefulness is continuous streaming + **friend adaptation** (record → Create ML / fine-tune), not marketing a toy phrase spotter as “fluent.”

## Architecture

```
┌───────────────────────── iPhone (iOS 27+) ─────────────────────────┐
│  Camera → Vision holistic landmarks (hands + body + face)           │
│       ↓ discard pixels                                              │
│  LandmarkFrame buffer (~36 frames) + utterance segmentation         │
│       ├── WebSocket stream ──► Recognition server (LAN)             │
│       └── offline fallback: Core ML (if present) / heuristics       │
│                                                                     │
│  Mic → SFSpeechRecognizer → "You said" transcript (reverse channel) │
│                                                                     │
│  Tabs: In Person · Call · Learn · History (+ Liquid Glass chrome)   │
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

- Mac with **Xcode 15+**, iPhone/iPad **iOS 27+** (Liquid Glass + Apple Intelligence APIs) (physical device recommended)
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

Shipping weights live in `server/models/sign_classifier.pt`. Rebuild with `pip install -r requirements-ml.txt && python scripts/train_poselstm.py` (see [`MODELS.md`](MODELS.md)).

On a physical iPhone, set **Settings → Continuous recognition** to  
`ws://<your-mac-lan-ip>:8765/v1/stream` (not `127.0.0.1`).

## Run the iOS app

1. Open `ASLSubtitles.xcodeproj` in Xcode.
2. Select the **ASLSubtitles** scheme → your iPhone.
3. Signing: Team = your Personal Team (`com.officiallygoob.aslsubtitles`).
4. Build & Run (⌘R). Allow **Camera**, **Microphone**, and **Speech Recognition**.
5. Tabs: **In Person** (camera conversation), **Call** (in-app video scaffold), **Learn** (flashcards), **History**.

### First conversation tips

- Point the rear camera at your friend; keep torso + hands in frame.
- Toggle **Mic** so your speech appears as “You said.”
- **Debug** draws holistic landmarks.
- Without a server, the app stays useful via **offline fallback** (heuristics / optional Core ML).

## What’s included

| Area | Implementation |
|----------------------|
| Conversation Mode | Persistent signing transcript + live word + speech + history |
| Holistic landmarks | Vision hand pose + body pose + face landmarks → `LandmarkFrame` |
| Temporal buffer | ~36 frames (~1–2 s) + pause/rest utterance segmentation |
| Streaming client | WebSocket landmark protocol (`RecognitionClient`) |
| Speech → text | `SFSpeechRecognizer` reverse channel |
| Offline fallback | Heuristics + NMM English + optional `CoreMLSignClassifier` |
| NMMs | Face/body cues drive questions / negation / emphasis |
| Liquid Glass | iOS 27 chrome for controls; accessibility-first captions |
| Siri / Apple Intelligence | App Intents, on-device polish/summary/suggestions, past chats |
| LandmarkRecorder | Export labeled JSON/JSONL for Create ML / fine-tunes |
| Server | FastAPI WS + REST, shipping PoseLSTM (~170+ conversational glosses), gloss→English, Docker |

## Vocabulary (honest split)

| Layer | What it covers |
|-------|----------------|
| **Offline heuristics** | Letters A–Z plus a **conversational subset** (greetings, courtesy, pronouns, questions, common verbs, numbers 1–5/10, food/feelings/time/places approximations). Tuned for clear single-hand / two-hand templates — not fluent ASL. |
| **Continuous PoseLSTM (server)** | Broader **friend-chat gloss list** (family, days, colors, places, verbs, social/tech, etc. — 170+ tokens). Trained on synthetic kinematics so it has a real decision surface over our feature layout; still weaker than real signer data. |
| **Friend adaptation** | `LandmarkRecorder` → Create ML / fine-tune remains the path for **their** dialect and the signs heuristics can’t approximate (“Needs ML” in the in-app vocabulary sheet). |

Empty detection gaps **no longer wipe** prior signing text: Conversation Mode keeps a persistent transcript until **Clear**.


## Non-manual markers (face + body)

ASL grammar leans on **face and body**, not hands alone. This app:

- Extracts richer Vision face landmarks (brows, lids, lips) + body pose
- Estimates **NMMState**: brow raise/furrow, eye widen/squint, mouth open/smile/frown, head shake/nod proxies, torso lean / shoulder tilt
- Conditions English offline and on the server: raised brows → questions; head shake / frown → negation; lean → emphasis
- Feature layout **v2** (`FEATURE_DIM=170`) adds face joints + 11 NMM channels (additive vs v1’s 139)

**Honesty:** phone Vision landmarks are approximate versus studio MoCap. Treat NMMs as soft cues. Enable **Debug** + **NMM badges** in Settings to visualize brow / shake / lean.

## Liquid Glass UI (iOS 27)

Conversation Mode uses Apple’s **Liquid Glass** for controls floating above content (`glassEffect`, `GlassEffectContainer`, morph IDs). Camera + signing transcript stay content-first for caption readability. Deployment target: **iOS 27**.



## Learn ASL

Flashcard practice using the same recognition stack:

1. Card shows a gloss / English word + hint (fingerspelling A–Z and everyday heuristic vocab).
2. Tap **I’m ready** → camera watches you sign.
3. Offline heuristics / Core ML / server score **correct** vs try again (confidence + hold).
4. **Again / Got it** queue with light spaced intervals; progress saved on-device.


## In-app Call tab

Primary remote path we’re building toward: **1:1 WebRTC / LiveKit** where ASL Subtitles owns both video feeds and runs recognition on the friend’s frames with persistent captions + speech→text.

**Now:** lobby with **username + share link/code** (no phone numbers / Contacts), local/remote layout, Liquid Glass caption chrome, mic/camera controls (signaling stubbed).  
**Next:** wire SDP/ICE or LiveKit room; pipe remote `CMSampleBuffer`s into `HolisticPoseDetector` → existing NMM + recognition.

FaceTime screen-capture remains **advanced** (Call ⋯ menu / Settings), not the hero.

## FaceTime & video calls

ASL Subtitles **cannot draw inside FaceTime’s own chrome** or tap a private FaceTime remote-video API. Call Mode uses the supported paths:

1. **ScreenCaptureKit (preferred, iOS 27)** — In-app **Use with FaceTime** captures the FaceTime window / display, runs holistic Vision on-device, and shows the same persistent signing transcript (Liquid Glass overlay in ASL Subtitles).
2. **ReplayKit Broadcast** — Control Center → Screen Broadcasting → **ASL Subtitles** while FaceTime is open. Extension scaffold lives in `ASLSubtitlesBroadcast/` (add the Xcode Broadcast Upload target once; enable App Group `group.com.officiallygoob.aslsubtitles`). See `ASLSubtitlesBroadcast/README.md`.
3. **SharePlay** — Start a custom `GroupActivity` so ASL Subtitles joins the FaceTime SharePlay sheet (shared session; optional caption sync).
4. **Picture in Picture** — Pop captions via PiP video-call style UI so the transcript can stay visible over FaceTime when the system allows.

### Steps

1. Start your FaceTime call.
2. Open ASL Subtitles → **Call** → **Use with FaceTime** (or start Broadcast from Control Center).
3. Keep ASL Subtitles for captions, or enable PiP.
4. Optional: **SharePlay on FaceTime** from the Call panel.

**Limits:** System permission prompts apply; some capture paths require you to leave FaceTime briefly to tap Start; broadcast extensions are CPU-limited (full NMM recognition is richest in-process via SCK). Always tell the other person if the call is being screen-captured.

## Siri & Apple Intelligence

| Feature | What it does |
|---------|----------------|
| **App Intents / App Shortcuts** | “Start ASL conversation…”, “Clear ASL captions…”, toggle mic, flip camera, summarize, open past chats — works with Siri, Shortcuts, and Action Button |
| **Foundation Models** (on-device) | Polish signing English; summarize history; suggest 2–3 short replies after a signing final |
| **Past transcripts** | Saved on-device (Application Support); clearable; exposed as App Entities for Spotlight / Siri when available |

Requires an **Apple Intelligence–capable device** / region for Foundation Models. If unavailable, template suggestions + extractive summaries still work. Video and transcripts stay on-device; landmark streams to your LAN server never include pixels.

## Friend adaptation (recommended next step)


1. Settings → **Landmark training** → record labeled glosses with your friend.
2. Export JSONL → train Create ML Hand Action (or fine-tune server weights).
3. Install `ASLSignClassifier.mlmodel` on device **or** drop `uni_sign.pt` into `server/models/`.

Details: [`MODELS.md`](MODELS.md) · server: [`server/README.md`](server/README.md)

## Limitations

- Not an interpreter. Heuristics = subset; server = larger limited domain; friend data still required for reliability.
- Ambiguous fingerspelling and facial grammar remain hard.
- Server ships `sign_classifier.pt` (PoseLSTM on synthetic kinematics over 170+ glosses) — better coverage than heuristics, still limited vs real signers; fine-tune on friend data (see MODELS.md).
- Uni-Sign `.pth` is research-only here (architecture mismatch); not Google SL2T.
- Simulator has no real camera/hands; use a device.

## License / care

Personal / educational use. Treat errors carefully; never rely on this for safety-critical communication.
