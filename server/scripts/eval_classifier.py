#!/usr/bin/env python3
"""Score top-1 / confusion on held-out NPZ or LandmarkRecorder JSON folder."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_ondevice_coreml import normalize_matrix  # noqa
from pipeline.normalize import FEATURE_DIM  # noqa


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "models" / "wlasl100_features.npz")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    import torch
    from pipeline.sequence_model import PoseLSTMClassifier

    blob = np.load(args.data, allow_pickle=True)
    X = blob["X"].astype(np.float32)
    y = blob["y"].astype(np.int64)
    labels = [str(x) for x in blob["labels"].tolist()]
    splits = blob["split"].tolist() if "split" in blob else ["test"] * len(y)
    mask = np.array([s == args.split for s in splits])
    if not mask.any():
        mask = np.ones(len(y), dtype=bool)
    Xn = np.stack([normalize_matrix(X[i]) for i in np.where(mask)[0]])
    yt = y[mask]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PoseLSTMClassifier(
        input_dim=int(ckpt.get("input_dim", FEATURE_DIM)),
        hidden_dim=int(ckpt.get("hidden_dim", 192)),
        num_classes=int(ckpt["num_classes"]),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xn)).argmax(-1).numpy()
    top1 = float((pred == yt).mean())
    confusions = {}
    for p, t in zip(pred, yt):
        if p != t:
            k = f"{labels[int(t)]}->{ckpt['labels'][int(p)] if int(p) < len(ckpt['labels']) else p}"
            confusions[k] = confusions.get(k, 0) + 1
    top = sorted(confusions.items(), key=lambda kv: -kv[1])[:20]
    report = {"split": args.split, "top1": top1, "n": int(mask.sum()), "top_confusions": top}
    print(json.dumps(report, indent=2))
    (ROOT / "models" / "eval_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
