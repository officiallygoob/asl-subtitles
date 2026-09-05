# Models — on-device Core ML first

## Privacy first

**Default product path never leaves the iPhone:**

```
Camera → Vision holistic landmarks → Core ML PoseLSTM/TCN → English subtitles
```

- No video, frames, or landmarks are uploaded by default.
- `preferRecognitionServer` defaults to **false**.
- Optional LAN WebSocket server is **dev-only** for experiments.

## Honesty first

**True open-domain conversational ASL → English is unsolved.** This project ships:

1. An **on-device** continuous landmark → Core ML classifier (`ASLSignClassifier.mlpackage`).
2. Optional dual **daily vocab** head (`ASLSignClassifierDaily.mlpackage`) — smaller closed set.
3. Training scripts that convert **public pose dumps** (WLASL / ASL Citizen COCO-135) into our `FEATURE_DIM=170` layout and export Core ML.
4. Gloss sense/synonym map so ASL Citizen merges into the shipping head.
5. Clear docs that this is **not** Google SL2T.

## What runs out of the box

| Component | Behavior |
|-----------|----------|
| **iOS Core ML (primary)** | Bundled `ASLSignClassifier.mlpackage` — TCN-BiLSTM + NMM attention over hands+face+body |
| **Daily head (optional)** | `ASLSignClassifierDaily.mlpackage` — same arch, ~110–163 conversational glosses |
| iOS heuristics | Fallback when Core ML low-confidence / missing |
| LAN server | Opt-in only; loads `sign_classifier.pt` |
| Uni-Sign `.pth` | Detected, not inferred (architecture mismatch) |

## Shipping on-device model

- **Input:** `poses` float32 `[1, 32, 170]` (FEATURE_DIM v2).
- **Arch:** temporal conv front-end + bidirectional LSTM + **NMM-conditioned temporal attention** + gloss head (+ NMM aux during train).
- **Training data (offline):** public pose HDF5 from [CristianLazoQuispe/pose-action-recognition](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition) (MIT packaging of landmarks):
  - WLASL100 + overlapping WLASL300 clips
  - **ASL Citizen 100 merged via gloss map** (sense-suffix strip + synonym/lemma → shipping labels)
  - Synth kinematic fill for conversational glosses missing from public pose (full head)
- **Augmentation:** heavy stack — speed resample, L/R mirror, Gaussian + hand noise, temporal shift/mask, joint dropout, spatial scale; class-balanced sampling.
- **Underlying video rights** remain with WLASL (research / C-UDA) and ASL Citizen (Microsoft research). We redistribute **converted landmarks + trained weights**, not videos.
- **Eval (held-out WLASL100, comparable to prior ship):** see `server/models/eval_report.json`

| Split | Previous top-1 | **Now top-1** | **Now top-5** |
|-------|----------------|---------------|---------------|
| Val (WLASL100) | 25.7% | **31.7%** | **58.0%** |
| Test (WLASL100) | 20.9% | **27.9%** | **57.4%** |

234-class full head. **+7.0 pp test top-1** vs prior ship — real gain, still far from usable conversation.

### Dual daily vocab head

| Head | Classes | Metric | top-1 | top-5 |
|------|---------|--------|-------|-------|
| Full (ships as primary) | 234 | WLASL100 holdout | **27.9%** | **57.4%** |
| Daily + synth fill | 163 | overall test / WLASL100∩daily | 27.6% / 35.7% | 55.0% / 56.6% |
| Daily real-only (ships dual) | 110 | overall test / WLASL100∩daily | **30.1%** / **39.5%** | 57.6% / 59.7% |

Daily numbers on WLASL100∩daily are **not** comparable to the full 100-class holdout (easier closed set). Friend-specific data still helps a lot.

### Rebuild Core ML

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
pip install h5py coremltools
# Place HDF5 under server/data/{wlasl100,wlasl300,aslcitizen100}/
python scripts/convert_pose_hdf5.py --sources wlasl100,wlasl300,aslcitizen100 --mix-synth --focus-wlasl100
python scripts/train_ondevice_coreml.py --data models/pose_features_focus.npz --arch tcn-bilstm --heavy-aug --aug-copies 3
# Daily head
python scripts/convert_pose_hdf5.py --sources wlasl100,wlasl300,aslcitizen100 --mix-synth --daily-vocab \
  --out models/pose_features_daily.npz
python scripts/train_ondevice_coreml.py --data models/pose_features_daily.npz --arch tcn-bilstm --heavy-aug \
  --coreml ../ASLSubtitles/Models/ASLSignClassifierDaily.mlpackage --out models/sign_classifier_daily.pt \
  --report models/eval_report_daily.json
python scripts/eval_classifier.py --split test --source-filter wlasl100 --data models/pose_features_focus.npz
```

### Optional friend fine-tune

1. Settings → Train/Capture → record takes → Export JSONL.
2. `python scripts/finetune_from_recordings.py --recordings /path/to/LandmarkRecordings`
3. Re-export Core ML and replace the bundled package.

## Public data sources (licenses)

| Source | What we use | License notes |
|--------|-------------|---------------|
| WLASL100 / WLASL300 pose HDF5 | COCO-135 → our layout; shipping uses W100 + W300 overlap | MIT packaging; cite WholeBodyPose; underlying WLASL research/C-UDA |
| ASL Citizen 100 pose HDF5 | **Merged** via `pipeline/gloss_map.py` (sense strip + synonyms) | MIT packaging; Microsoft research terms for videos; landmarks only stored |
| Synth kinematics | Fill missing conversational glosses (full / daily+synth) | In-repo |
| Uni-Sign checkpoints | Research only, not runtime | CC-BY-NC-4.0 |
| How2Sign pose | **Blocked this round** | Continuous SLT; large shards; needs CTC/transducer — next continuous path |

**Gloss map:** sense IDs (`ABOUT1`→`ABOUT`) + conservative synonyms (`BATH`→`BATHROOM`, `CALLTTY`→`CALL`, `WHATFOR`→`WHY`, meals→`FOOD`, …). See `pipeline/gloss_map.py`.

**Blockers:** How2Sign (task mismatch + size). ASL Citizen 300/2731 and WLASL2000 remain available on the same HF repo.

## Feature layout

`FEATURE_DIM = 170`: left/right hands 42+42, body 34, face 40, activity 1, NMM 11.

## Accuracy expectations

| Setup | Expectation |
|-------|-------------|
| Bundled Core ML, unseen signer | Limited-domain; many confusions; better than prior PoseLSTM ship on covered glosses |
| Daily head | Higher accuracy on smaller conversational closed set; not open-domain |
| Core ML + friend fine-tune | Best practical path for *their* signing |
| Open-domain fluent chat | Unsolved |

## Roadmap

1. Larger Citizen/WLASL packs with the same gloss map
2. Stronger continuous decoding (CTC / transducer) for How2Sign-style data
3. On-device quantization / ANE tuning
4. Richer facial grammar beyond soft NMM proxies
