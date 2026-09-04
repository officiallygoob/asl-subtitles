#!/usr/bin/env python3
"""Train PoseLSTM on synthetic (or NPZ) pose data → models/sign_classifier.pt

Usage:
  python scripts/synthesize_pose_dataset.py
  python scripts/train_poselstm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Same centering/scaling as pipeline.normalize.normalize_frames for (T,D)."""
    if mat.shape[0] == 0:
        return mat
    body = mat[:, 84:118]
    neck = body[:, 2:4]
    r_sh = body[:, 4:6]
    l_sh = body[:, 10:12]
    scale = np.linalg.norm(r_sh - l_sh, axis=1, keepdims=True)
    scale = np.clip(scale, 0.05, None)
    origin = neck.copy()
    missing = np.abs(origin).sum(axis=1) < 1e-6
    if missing.any():
        lw = mat[:, 0:2]
        rw = mat[:, 42:44]
        origin[missing] = 0.5 * (lw[missing] + rw[missing])
    xy = mat[:, :-1].reshape(mat.shape[0], -1, 2)
    xy = (xy - origin[:, None, :]) / scale[:, None, :]
    return np.concatenate([xy.reshape(mat.shape[0], -1), mat[:, -1:]], axis=1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "models" / "synth_dataset.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not args.data.exists():
        from scripts.synthesize_pose_dataset import main as synth_main

        print("No dataset — synthesizing…")
        sys.argv = ["synthesize_pose_dataset.py"]
        synth_main()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from pipeline.normalize import FEATURE_DIM
    from pipeline.sequence_model import PoseLSTMClassifier
    from pipeline.vocab import GLOSS_VOCAB

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    blob = np.load(args.data, allow_pickle=True)
    X = blob["X"].astype(np.float32)
    y = blob["y"].astype(np.int64)
    labels = [str(x) for x in blob["labels"].tolist()] if "labels" in blob else list(GLOSS_VOCAB)

    # Normalize each sequence
    Xn = np.stack([normalize_matrix(X[i]) for i in range(len(X))], axis=0)

    # Train/val split
    idx = rng.permutation(len(Xn))
    n_val = max(1, len(Xn) // 10)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    Xtr, ytr = Xn[train_idx], y[train_idx]
    Xva, yva = Xn[val_idx], y[val_idx]

    device = torch.device("cpu")
    model = PoseLSTMClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden,
        num_layers=2,
        num_classes=len(labels),
        bidirectional=True,
        dropout=0.25,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=args.batch,
        shuffle=True,
    )
    Xva_t = torch.from_numpy(Xva)
    yva_t = torch.from_numpy(yva)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += float(loss) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(dim=-1)
            acc = float((pred == yva_t).float().mean())
        print(f"epoch {epoch:02d}  loss={total_loss/max(n,1):.4f}  val_acc={acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "num_classes": len(labels),
        "input_dim": FEATURE_DIM,
        "hidden_dim": args.hidden,
        "labels": labels,
        "val_acc": best_acc,
        "backend": "poselstm",
        "trained_on": "synthetic-kinematics-v1",
        "note": (
            "Trained on synthetic landmark templates matching FEATURE_DIM. "
            "Better than demo heuristics for protocol+limited domain; "
            "fine-tune on LandmarkRecorder / WLASL pose for real signer accuracy. "
            "Not Uni-Sign / not Google SL2T."
        ),
    }
    torch.save(ckpt, args.out)
    labels_path = args.out.parent / "labels.json"
    labels_path.write_text(json.dumps({"labels": labels, "val_acc": best_acc}, indent=2))
    print(f"saved {args.out} val_acc={best_acc:.3f} classes={len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
