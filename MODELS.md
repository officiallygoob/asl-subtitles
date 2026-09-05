# Models — what ships, what to plug in

## Honesty first

**True open-domain conversational ASL → English is unsolved.** Even strong research systems (Uni-Sign, SignSpeak, DeepMind SL2T demos) are limited by data, domain, and signer variation. This project ships:

1. A **continuous landmark-streaming architecture** (client → server) ready for better weights.
2. A **runnable PoseLSTM** (`sign_classifier.pt`) over a ~170+-gloss conversational vocabulary with sliding-window continuous decoding + gloss→English.
3. Clear documentation of the **Uni-Sign gap** (architecture mismatch) — **not** Google’s proprietary SL2T.

Do **not** claim fluent chat. The practical path is **limited-domain continuous recognition + friend adaptation** (record your friend’s signs, fine-tune).

## What runs out of the box

| Component | Behavior |
|-----------|----------|
| iOS offline fallback | Heuristics (expanded conversational subset + A–Z) via `SignRecognizer` |
| iOS Core ML plug-in | `CoreMLSignClassifier` loads `ASLSignClassifier.mlmodel(c)` if **you** add it |
| Server with shipping weights | `sign_classifier.pt` PoseLSTM (~170+ conversational glosses), sliding-window continuous decode |
| Server without weights | `demo-continuous-v1` heuristic decoder (protocol test) |
| Uni-Sign `.pth` present | Detected on `/health` as `present-architecture-mismatch` — **not** used for inference |

## Shipping PoseLSTM (`sign_classifier.pt`)

- **Input:** 32-frame windows, `FEATURE_DIM=139` (Vision/MediaPipe layout after neck/shoulder normalize).
- **Arch:** bidirectional LSTM (hidden 192) + MLP head.
- **Vocab:** conversational glosses in `pipeline/vocab.py` (greetings, questions, everyday needs).
- **Training data:** synthetic kinematic templates (`scripts/synthesize_pose_dataset.py`) that match our feature layout — **not** real WLASL video. Val accuracy on that synthetic holdout is high; **real-signer accuracy will be much lower** until you fine-tune.
- **Continuous decode:** longer utterances use sliding windows (stride 10) with consecutive-gloss dedupe → multi-gloss phrases → rule-based English.

### Enable / rebuild weights

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt   # torch
python scripts/synthesize_pose_dataset.py
python scripts/train_poselstm.py
uvicorn main:app --host 0.0.0.0 --port 8765
curl http://127.0.0.1:8765/health   # model: sign_classifier.pt
```

Expected checkpoint dict:

```python
{
  "state_dict": <PoseLSTMClassifier state>,
  "num_classes": 173,
  "input_dim": 139,
  "hidden_dim": 192,
  "labels": ["HELLO", "HI", ...],
  "trained_on": "synthetic-kinematics-v1",
}
```

Place as `server/models/sign_classifier.pt` (or `poselstm.pt`). Restart uvicorn.

### Fine-tune on real data (recommended)

1. App **Settings → Landmark training** → record labeled clips of your friend.
2. Export JSONL from Documents/`LandmarkRecordings`.
3. Convert exports into the NPZ layout used by `train_poselstm.py` (or extend the trainer to read JSONL).
4. Retrain and replace `sign_classifier.pt`.

Friend-adapted weights beat both the shipping synthetic model and the demo heuristics for real conversations.

## Uni-Sign (research checkpoint — not the runtime path)

- Paper: Uni-Sign (ICLR 2025)
- Weights: https://huggingface.co/ZechengLi19/Uni-Sign (CC-BY-NC-4.0)
- Code: https://github.com/ZechengLi19/Uni-Sign

### Why it doesn’t auto-load

Uni-Sign pose-only checkpoints expect **69 RTMPose whole-body keypoints**, Spatial **GCN** pose encoders, temporal encoders, and an **LLM** text head. Our iOS client streams **Vision holistic landmarks** packed as **139-d** features into a **PoseLSTM**. Bridging that gap means re-implementing (or vendoring) Uni-Sign’s full stack + remapping keypoints — not a drop-in `.load_state_dict`.

`wlasl_pose_only_islr.pth` alone is ~1.1 GB.

### Download for research / future adapter work

```bash
cd server
./scripts/download_uni_sign.sh
# fetches wlasl_pose_only_islr.pth into server/models/
```

| File | Task |
|------|------|
| `wlasl_pose_only_islr.pth` | Isolated sign recognition (WLASL) |
| `how2sign_pose_only_slt.pth` | Continuous SLT (How2Sign, large) |
| `openasl_pose_only_slt.pth` | OpenASL continuous SLT |

`/health` reports `uni_sign.status = present-architecture-mismatch` when a `.pth` is on disk. Inference stays on PoseLSTM / demo.

**Sign-Speak is enterprise-only** — do not block on it. This is **not** Google SL2T.

## Create ML / friend adaptation (highest leverage on-device)

1. In the app **Settings → Landmark training**, record labeled clips of your friend.
2. Export JSON / JSONL from Documents/`LandmarkRecordings`.
3. Train a Create ML **Hand Action** classifier (or fine-tune the server PoseLSTM).
4. Drop the model into the app bundle or Documents as `ASLSignClassifier.mlmodel`.
5. `CoreMLSignClassifier` picks it up automatically; heuristics only fill gaps.

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

Rule-based `gloss_to_english` always runs (phrase table + stitching); LLM is an optional local hook (no cloud keys in-repo).

## Accuracy expectations (honest)

| Setup | Expectation |
|-------|-------------|
| Shipping PoseLSTM, unseen real signer | Limited-domain; many confusions; useful for protocol + coarse phrases |
| PoseLSTM fine-tuned on your friend | Best practical path for real conversation |
| Demo heuristics | Protocol test only |
| Uni-Sign `.pth` without adapter | Does not run here |
| Open-domain fluent chat | Unsolved |

## Roadmap toward better accuracy

1. Fine-tune PoseLSTM on LandmarkRecorder / public WLASL MediaPipe pose sets
2. Optional Uni-Sign runtime via vendored GCN+LLM + RTMPose remap (large effort)
3. CTC / autoregressive continuous decoding (not only sliding ISLR windows)
4. Stronger facial grammar features (roles, questions, negation)
5. On-device Core ML export of the server model for offline continuous mode
