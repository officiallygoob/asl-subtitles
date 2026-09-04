# Models — what ships, what to plug in

## Honesty first

**True open-domain conversational ASL → English is unsolved.** Even strong research systems (Uni-Sign, SignSpeak, DeepMind SL2T demos) are limited by data, domain, and signer variation. This project ships:

1. A **continuous landmark-streaming architecture** (client → server) ready for better weights.
2. A **limited-domain** demo decoder + gloss→English rules so Conversation Mode works today.
3. Clear plug-in points for Uni-Sign / PoseLSTM / Create ML — **not** Google’s proprietary SL2T.

Do **not** claim fluent chat. The practical path is **limited-domain continuous recognition + friend adaptation** (record your friend’s signs, fine-tune).

## What runs out of the box

| Component | Behavior |
|-----------|----------|
| iOS offline fallback | Heuristics (~20 glosses + A–Z) via `SignRecognizer` |
| iOS Core ML plug-in | `CoreMLSignClassifier` loads `ASLSignClassifier.mlmodel(c)` if **you** add it |
| Server without weights | `demo-continuous-v1` sliding-window decoder (protocol test / limited glosses) |
| Server with weights | Loads `server/models/*.pt` or `model.onnx` into PoseLSTM scaffold |

## Uni-Sign (preferred research checkpoint)

- Paper: Uni-Sign (ICLR 2025)
- Community pose checkpoints / how-to often referenced via Hugging Face (`ZechengLi19/Uni-Sign` and related How2Sign / OpenASL releases)
- **Sign-Speak is enterprise-only** — do not block on it

### Plug Uni-Sign pose weights

### Download (pose-only)

```bash
cd server
./scripts/download_uni_sign.sh
# fetches wlasl_pose_only_islr.pth into server/models/
```

Upstream files of interest (CC-BY-NC-4.0):

| File | Task |
|------|------|
| `wlasl_pose_only_islr.pth` | Isolated sign recognition (WLASL) |
| `how2sign_pose_only_slt.pth` | Continuous SLT (How2Sign, large) |
| `openasl_pose_only_slt.pth` | OpenASL continuous SLT |

**Important:** these checkpoints use Uni-Sign’s native architecture. Our server detects them and reports via `/health`, but inference still uses the demo/PoseLSTM path until you finish `pipeline/uni_sign_adapter.py` (landmark layout remap + forward). Exporting a distilled PoseLSTM head to `sign_classifier.pt` is the fastest way to get auto-load today.

Code: https://github.com/ZechengLi19/Uni-Sign · Weights: https://huggingface.co/ZechengLi19/Uni-Sign


```bash
cd server/models
# Example — adjust to the actual file names published for the pose encoder you choose:
# huggingface-cli download ZechengLi19/Uni-Sign --local-dir ./uni-sign-src
#
# Then convert / export a pose-only classifier head to our scaffold:
#   uni_sign.pt  with keys: state_dict, num_classes, input_dim, labels
```

Expected torch checkpoint dict:

```python
{
  "state_dict": <PoseLSTMClassifier state>,
  "num_classes": 2000,          # or your vocab size
  "input_dim": 139,             # must match normalize.FEATURE_DIM (or adapt)
  "labels": ["HELLO", "THANKS", ...]
}
```

Place as `server/models/uni_sign.pt` (or `poselstm.pt` / `sign_classifier.pt`). Restart uvicorn; `/health` should show the file name as `model`.

If the public Uni-Sign release is a full video-conditioned model rather than pure pose, keep the **pose encoder** weights and map MediaPipe/Vision landmarks into that encoder’s expected layout (document your remapping next to the checkpoint).

## Create ML / friend adaptation (highest leverage on-device)

1. In the app **Settings → Landmark training**, record labeled clips of your friend.
2. Export JSON / JSONL from Documents/`LandmarkRecordings`.
3. Train a Create ML **Hand Action** classifier (or a small custom sequence model) on those clips.
4. Drop the model into the app bundle or Documents as `ASLSignClassifier.mlmodel`.
5. `CoreMLSignClassifier` picks it up automatically; heuristics only fill gaps.

This friend-adapted loop beats generic demo glosses for real conversations.

## Feature layout (server)

`FEATURE_DIM = 139` per frame:

- Left hand 21×2, right hand 21×2
- Body 17×2, face 10×2
- Activity scalar

Normalization: subtract neck/hand origin, scale by shoulder width (`pipeline/normalize.py`).

## Optional gloss → English LLM

```bash
export ASL_GLOSS_LLM_CMD='ollama run llama3.2'
```

Rule-based `gloss_to_english` always runs; LLM is an optional local hook (no cloud keys in-repo).

## Roadmap toward better accuracy

1. Load Uni-Sign / How2Sign pose checkpoint into `ContinuousDecoder`
2. CTC / autoregressive continuous decoding (not isolated gloss windows)
3. Friend-specific Create ML or LoRA adaptation from `LandmarkRecorder` data
4. Stronger facial grammar features (roles, questions, negation)
5. On-device Core ML export of the server model for offline continuous mode
