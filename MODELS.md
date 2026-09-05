# Models — on-device Core ML first

## Privacy first

**Default product path never leaves the iPhone:**

```
Camera → Vision holistic landmarks → Core ML PoseLSTM → English subtitles
```

- No video, frames, or landmarks are uploaded by default.
- `preferRecognitionServer` defaults to **false**.
- Optional LAN WebSocket server is **dev-only** for experiments.

## Honesty first

**True open-domain conversational ASL → English is unsolved.** This project ships:

1. An **on-device** continuous landmark → Core ML classifier (`ASLSignClassifier.mlpackage`).
2. Training scripts that convert **public pose dumps** (WLASL100 COCO-135) into our `FEATURE_DIM=170` layout and export Core ML.
3. Optional friend Capture/Train for adaptation later — **not** the primary accuracy path.
4. Clear docs that this is **not** Google SL2T.

## What runs out of the box

| Component | Behavior |
|-----------|----------|
| **iOS Core ML (primary)** | Bundled `ASLSignClassifier.mlpackage` — PoseLSTM over hands+face+body+NMM |
| iOS heuristics | Fallback when Core ML low-confidence / missing |
| LAN server | Opt-in only; loads `sign_classifier.pt` |
| Uni-Sign `.pth` | Detected, not inferred (architecture mismatch) |

## Shipping on-device model

- **Input:** `poses` float32 `[1, 32, 170]` (FEATURE_DIM v2).
- **Arch:** bidirectional LSTM + **NMM-conditioned temporal attention** + gloss head (+ NMM aux during train).
- **Training data (offline):** WLASL100 pose HDF5 from [CristianLazoQuispe/pose-action-recognition](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition) (COCO-133+2, MIT packaging of landmarks). Underlying WLASL video rights remain with WLASL (research / C-UDA). We redistribute **converted landmarks + trained weights**, not videos.
- **Synth fill:** conversational glosses missing from WLASL100 are filled with kinematic templates so HELLO etc. remain in the label set.
- **Eval (held-out WLASL test + val):** see `server/models/eval_report.json` — roughly **~17–25% top-1** over the mixed 234-class head on real pose holdout (far above chance ~0.4%; far below studio SLR). **Friend-specific data still helps a lot.**

### Rebuild Core ML (on your Mac/CI with Python)

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
pip install h5py coremltools
# Download WLASL100 pose dumps into server/data/wlasl100/ (see scripts/convert_wlasl_hdf5.py docstring)
python scripts/convert_wlasl_hdf5.py --mix-synth
python scripts/train_ondevice_coreml.py
# writes ASLSubtitles/Models/ASLSignClassifier.mlpackage + server/models/sign_classifier.pt
python scripts/eval_classifier.py --split test
```

### Optional friend fine-tune

1. Settings → Train/Capture → record takes → Export JSONL.
2. `python scripts/finetune_from_recordings.py --recordings /path/to/LandmarkRecordings`
3. Re-export Core ML and replace the bundled package.

## Public data sources (licenses)

| Source | What we use | License notes |
|--------|-------------|---------------|
| WLASL100 pose HDF5 (WholeBodyPose packaging) | COCO-135 landmarks → our layout | MIT packaging; cite WholeBodyPose; underlying WLASL is research/C-UDA |
| Synth kinematics | Fill missing conversational glosses | In-repo |
| Uni-Sign checkpoints | Research only, not runtime | CC-BY-NC-4.0 |
| How2Sign / ASL Citizen pose | Documented next; larger downloads | Follow original dataset terms |

**Next data:** ASL Citizen 100/300 pose HDF5 and How2Sign MediaPipe landmark shards for continuous SLT pretrain — convert with the same COCO-135 mapper when disk/time allow.

## Feature layout

`FEATURE_DIM = 170`: left/right hands 42+42, body 34, face 40, activity 1, NMM 11.

## Accuracy expectations

| Setup | Expectation |
|-------|-------------|
| Bundled Core ML, unseen signer | Limited-domain; many confusions; better than heuristics on covered glosses |
| Core ML + friend fine-tune | Best practical path for *their* signing |
| Open-domain fluent chat | Unsolved |

## Roadmap

1. Larger public pose pretrain (ASL Citizen / How2Sign landmarks)
2. Stronger continuous decoding (CTC / transducer), not only sliding ISLR
3. On-device quantization / ANE tuning
4. Richer facial grammar beyond soft NMM proxies
