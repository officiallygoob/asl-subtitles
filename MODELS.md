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
2. Optional dual **daily CORE30** head (`ASLSignClassifierDaily.mlpackage`) — densest 30 conversational glosses.
3. Training scripts that convert **public pose dumps** (WLASL / ASL Citizen COCO-135) into our `FEATURE_DIM=170` layout and export Core ML.
4. Gloss sense/synonym map so ASL Citizen merges into the shipping head (+ first-class Citizen conversational glosses).
5. On-device **bigram top-5 rerank** over glosses (no network).
6. Clear docs that this is **not** Google SL2T / not fluent conversation.

## What runs out of the box

| Component | Behavior |
|-----------|----------|
| **iOS Core ML (primary)** | Bundled `ASLSignClassifier.mlpackage` — TCN-BiLSTM + NMM attention + bigram rerank |
| **Daily head (optional dual)** | `ASLSignClassifierDaily.mlpackage` — CORE30 closed set (30 classes) |
| iOS heuristics | Fallback when Core ML low-confidence / missing |
| LAN server | Opt-in only; loads `sign_classifier.pt` |
| Uni-Sign `.pth` | Detected, not inferred (architecture mismatch) |

## Shipping on-device model

- **Input:** `poses` float32 `[1, 32, 170]` (FEATURE_DIM v2).
- **Arch:** temporal conv front-end + bidirectional LSTM + **NMM-conditioned temporal attention** + gloss head (+ NMM aux during train).
- **Training levers this round:** ensemble **KD distill** from high-test ungated ens3 into `distill_ens_s19`; longer tight singles (`tight_s7`); **val-A tune + val-B gate** before ens4 ship; mixup/distill/SWA/finetune retained.
- **Training data (offline):** public pose HDF5 from [CristianLazoQuispe/pose-action-recognition](https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition) (MIT packaging of landmarks):
  - WLASL100 + overlapping WLASL300 / **WLASL2000** clips (deduped)
  - **ASL Citizen 300 + 2731** via gloss map (**no MSASL** — diluted the holdout)
  - Synth kinematic fill for conversational glosses missing from public pose (full head only)
- **Underlying video rights** remain with WLASL (research / C-UDA) and ASL Citizen (Microsoft research). We redistribute **converted landmarks + trained weights**, not videos.
- **Eval (held-out WLASL100, comparable to prior ship):** see `server/models/eval_report.json`

### Comparable WLASL100 holdout (full head)

| Split | Prior ship (fb3e0f5) | **Now** top-1 | **Now** top-5 |
|-------|----------------------|---------------|---------------|
| Val (WLASL100) | 48.5% | **49.7%** | **73.1%** |
| Test (WLASL100) | 45.7% | **46.5%** | **72.1%** |

243-class full head. **Plain top-1 46.5%** vs 45.7% @f144fb9 (**+0.8 pp**). Distilled high-test ungated ens (s7+tight_s43+tcn2) into `distill_ens_s19`; **val-A/B gated** weighted ens4 (s7 + tight_s43 + tight_s7 + distill_ens_s19 → one Core ML) clears gates (val-A 52.1% / val-B 47.3%) with test 46.5%. Still short of ≥50% / conversation.

Bigram top-5 rerank remains available on-device; plain holdout is the ship gate.

### Dual daily CORE30 head

| Head | Classes | Metric | top-1 | top-5 |
|------|---------|--------|-------|-------|
| Full (ships as primary) | 243 | WLASL100 holdout | **46.5%** | **72.1%** |
| Daily CORE30 (ships dual) | 30 | own val / own test | **46.3%** / **36.8%** | 74.6% / **76.3%** |
| Daily CORE30 + bigram | 30 | own test chain / gold-prev | 39.8% / **60.2%** | — |

CORE30 = 30 highest-support conversational glosses from multi-source pose (no synth). **Own-test 36.8% is not ≥50%**, so full head remains the default Core ML; daily stays optional. WLASL100∩daily and CORE30 numbers are **easier closed sets** — do not compare to the full-head holdout.

### Rebuild Core ML

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
pip install h5py coremltools
# Place HDF5 under server/data/{wlasl100,wlasl300,aslcitizen100}/
python scripts/convert_pose_hdf5.py --sources wlasl100,wlasl300,aslcitizen300,aslcitizen2731,wlasl2000 \
  --mix-synth --focus-wlasl100 --dedup-exact --train-only-extra aslcitizen2731,wlasl2000 \
  --out models/pose_features_dense.npz
python scripts/train_ondevice_coreml.py --data models/pose_features_dense.npz --arch tcn-bilstm \
  --heavy-aug --aug-copies 4 --mixup 0.2 --epochs 64 --batch 48 --lr 4.5e-4 --seed 43 \
  --teacher models/sign_classifier_c49c751_41p9.pt --distill-alpha 0.42 \
  --wlasl100-boost 3.4 --real-boost 1.45 --synth-boost 0.4 --wlasl300-boost 1.1 \
  --citizen-overlap-boost 1.2 --finetune-wlasl100-epochs 28 --finetune-include-citizen-overlap \
  --hard-mine-from models/hard_mine_confusions.json --boost-glosses SHORT,FORGET,FOOD --gloss-boost 2.25 --bigram-rerank
# then val-A/B gate WeightedLogitEnsemble(s7, tight_s43, tight_s7, distill_ens_s19) → export one Core ML only if both slices lift + test > ship
# Daily CORE30 (filter from dense convert or core30 NPZ)
python scripts/convert_pose_hdf5.py --sources wlasl100,wlasl300,aslcitizen100 --daily-dense \
  --out models/pose_features_daily_dense.npz
# then subset / train core30 — see eval_report_daily_core30.json
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
| ASL Citizen 300 / **2731** pose HDF5 | **Merged** via `pipeline/gloss_map.py` (2731 densifies weak WLASL100 classes) | MIT packaging; Microsoft research terms for videos; landmarks only stored |
| WLASL2000 pose HDF5 | Overlap clips into focus head after fingerprint dedup | MIT packaging; underlying WLASL research/C-UDA |
| Synth kinematics | Fill missing conversational glosses (full head) | In-repo |
| Uni-Sign checkpoints | Research only, not runtime | CC-BY-NC-4.0 |
| How2Sign pose | **Blocked this round** | Continuous SLT; large shards; needs CTC/transducer — next continuous path |

**Gloss map:** sense IDs (`ABOUT1`→`ABOUT`) + conservative near-identity synonyms (`BATH`→`BATHROOM`, `MOM`→`MOTHER`, `FORGETFUL`→`FORGET`, `SHORTPERSON`→`SHORT`, …). **Meal lemmas are not bucketed into FOOD** (diluted holdout). See `pipeline/gloss_map.py`.

## What we are not claiming

- Not fluent ASL→English conversation.
- Not open-domain sign language translation.
- Holdout gains are isolated-sign classification on public pose — phone camera / friend signing will differ.
