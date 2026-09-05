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
2. Training scripts that convert **public pose dumps** (WLASL / ASL Citizen COCO-135) into our `FEATURE_DIM=170` layout and export Core ML.
3. Optional friend Capture/Train for adaptation later — **not** the primary accuracy path.
4. Clear docs that this is **not** Google SL2T.

## What runs out of the box

| Component | Behavior |
|-----------|----------|
| **iOS Core ML (primary)** | Bundled `ASLSignClassifier.mlpackage` — PoseLSTM-v3-stable over hands+face+body+NMM |
| iOS heuristics | Fallback when Core ML low-confidence / missing |
| LAN server | Opt-in only; loads `sign_classifier.pt` |
| Uni-Sign `.pth` | Detected, not inferred (architecture mismatch) |

## Shipping on-device model

- **Input:** `poses` float32 `[1, 32, 170]` (FEATURE_DIM v2).
- **Arch:** deeper bidirectional LSTM (3×224) + **NMM-conditioned temporal attention** + gloss head (+ NMM aux during train).
- **Training data (offline):** public pose HDF5 from [CristianLazoQuispe/pose-action-recognition](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition) (MIT packaging of landmarks):
  - WLASL100 + overlapping WLASL300 clips (same English gloss strings)
  - Synth kinematic fill for conversational glosses missing from public pose
  - ASL Citizen 100 downloaded & convertible; **not in shipping head** — gloss IDs barely overlap WLASL without a lexicon map (documented blocker)
- **Augmentation:** speed resample, L/R mirror, Gaussian noise, temporal shift, joint dropout; class-balanced sampling.
- **Underlying video rights** remain with WLASL (research / C-UDA). We redistribute **converted landmarks + trained weights**, not videos.
- **Eval (held-out WLASL100, comparable to prior ship):** see `server/models/eval_report.json`

| Split | Previous top-1 | **Now top-1** | **Now top-5** |
|-------|----------------|---------------|---------------|
| Val (WLASL100) | 25.4% | **25.7%** | **50.9%** |
| Test (WLASL100) | 16.7% | **20.9%** | **43.0%** |

234-class head. Friend-specific data still helps a lot.

### Rebuild Core ML

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
pip install h5py coremltools
# Place HDF5 under server/data/{wlasl100,wlasl300,aslcitizen100}/
python scripts/convert_pose_hdf5.py --sources wlasl100,wlasl300,aslcitizen100 --mix-synth --focus-wlasl100
# Optional: drop Citizen rows if gloss aliases are noisy
python scripts/train_ondevice_coreml.py --data models/pose_features.npz
python scripts/eval_classifier.py --split test --source-filter wlasl100
```

### Optional friend fine-tune

1. Settings → Train/Capture → record takes → Export JSONL.
2. `python scripts/finetune_from_recordings.py --recordings /path/to/LandmarkRecordings`
3. Re-export Core ML and replace the bundled package.

## Public data sources (licenses)

| Source | What we use | License notes |
|--------|-------------|---------------|
| WLASL100 / WLASL300 pose HDF5 | COCO-135 → our layout; shipping uses W100 + W300 overlap | MIT packaging; cite WholeBodyPose; underlying WLASL research/C-UDA |
| ASL Citizen 100 pose HDF5 | Downloaded; convert OK; **not merged into shipping labels** | MIT packaging; Microsoft research terms for videos; gloss naming mismatch |
| Synth kinematics | Fill missing conversational glosses | In-repo |
| Uni-Sign checkpoints | Research only, not runtime | CC-BY-NC-4.0 |
| How2Sign pose | **Blocked this round** | Continuous SLT; large shards; needs CTC/transducer — next continuous path |

**Blockers:** How2Sign (task mismatch + size). ASL Citizen→WLASL lexicon (only ~19 glosses alias-match after stripping sense ids). ASL Citizen 300/2731 and WLASL2000 remain available on the same HF repo.

## Feature layout

`FEATURE_DIM = 170`: left/right hands 42+42, body 34, face 40, activity 1, NMM 11.

## Accuracy expectations

| Setup | Expectation |
|-------|-------------|
| Bundled Core ML, unseen signer | Limited-domain; many confusions; better than heuristics on covered glosses |
| Core ML + friend fine-tune | Best practical path for *their* signing |
| Open-domain fluent chat | Unsolved |

## Roadmap

1. Gloss lexicon alignment for ASL Citizen 100/300/2731 + WLASL2000
2. Stronger continuous decoding (CTC / transducer) for How2Sign-style data
3. On-device quantization / ANE tuning
4. Richer facial grammar beyond soft NMM proxies
