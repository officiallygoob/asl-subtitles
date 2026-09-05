#!/usr/bin/env python3
"""Score top-1 / top-5 / confusion on held-out NPZ."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_ondevice_coreml import normalize_matrix, topk_acc  # noqa
from pipeline.normalize import FEATURE_DIM  # noqa


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "models" / "pose_features.npz")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--source-filter", default="", help="Optional source tag e.g. wlasl100")
    args = ap.parse_args()

    if not args.data.exists():
        legacy = ROOT / "models" / "wlasl100_features.npz"
        if legacy.exists():
            args.data = legacy

    import torch
    from pipeline.sequence_model import PoseLSTMClassifier

    blob = np.load(args.data, allow_pickle=True)
    X = blob["X"].astype(np.float32)
    y = blob["y"].astype(np.int64)
    labels = [str(x) for x in blob["labels"].tolist()]
    splits = blob["split"].tolist() if "split" in blob else ["test"] * len(y)
    sources = blob["source"].tolist() if "source" in blob else [""] * len(y)
    mask = np.array([s == args.split for s in splits])
    if args.source_filter:
        mask &= np.array([s == args.source_filter for s in sources])
    if not mask.any():
        mask = np.ones(len(y), dtype=bool)
    Xn = np.stack([normalize_matrix(X[i]) for i in np.where(mask)[0]])
    yt = y[mask]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PoseLSTMClassifier(
        input_dim=int(ckpt.get("input_dim", FEATURE_DIM)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_layers=int(ckpt.get("num_layers", 3)),
        num_classes=int(ckpt["num_classes"]),
        bidirectional=True,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xn))
        pred = logits.argmax(-1).numpy()
        top1 = float((pred == yt).mean())
        top5 = topk_acc(logits, torch.from_numpy(yt), 5)
    confusions = {}
    ckpt_labels = ckpt.get("labels", labels)
    for p, t in zip(pred, yt):
        if p != t:
            k = f"{labels[int(t)]}->{ckpt_labels[int(p)] if int(p) < len(ckpt_labels) else p}"
            confusions[k] = confusions.get(k, 0) + 1
    top = sorted(confusions.items(), key=lambda kv: -kv[1])[:20]
    report = {
        "split": args.split,
        "source_filter": args.source_filter or None,
        "top1": top1,
        "top5": top5,
        "n": int(mask.sum()),
        "n_classes": int(ckpt["num_classes"]),
        "top_confusions": top,
        "previous": ckpt.get("previous_baseline"),
        "wlasl100_subset": ckpt.get("wlasl100_subset"),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
