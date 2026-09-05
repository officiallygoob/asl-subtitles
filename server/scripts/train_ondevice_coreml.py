#!/usr/bin/env python3
"""Train PoseLSTM on WLASL-converted (+ synth) landmarks and export Core ML.

Primary product path: on-device Core ML. Server .pt is a side artifact for
optional LAN debug — not required for captions.

Usage:
  python scripts/convert_wlasl_hdf5.py --mix-synth
  python scripts/train_ondevice_coreml.py
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
    from pipeline.normalize import ACTIVITY_IDX

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
    spatial_end = ACTIVITY_IDX if mat.shape[1] > ACTIVITY_IDX else mat.shape[1] - 1
    xy = mat[:, :spatial_end].reshape(mat.shape[0], -1, 2)
    xy = (xy - origin[:, None, :]) / scale[:, None, :]
    spatial = xy.reshape(mat.shape[0], -1)
    tail = mat[:, spatial_end:]
    return np.concatenate([spatial, tail], axis=1).astype(np.float32)


def nmm_aux_targets(labels: list[str], y: np.ndarray) -> np.ndarray:
    """Soft targets: [question, negation, emphasis] from gloss linguistics."""
    questions = {"WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW", "WHICH", "QUESTION"}
    negation = {"NO", "DONT-KNOW", "WRONG", "FALSE"}
    emphasis = {"IMPORTANT", "NEED", "HELP", "STOP", "LOVE", "ANGRY", "YES"}
    out = np.zeros((len(y), 3), dtype=np.float32)
    for i, yi in enumerate(y):
        g = labels[int(yi)]
        if g in questions:
            out[i, 0] = 1.0
        if g in negation:
            out[i, 1] = 1.0
        if g in emphasis:
            out[i, 2] = 1.0
    return out


def export_coreml(model, labels: list[str], input_dim: int, out_path: Path, window: int = 32) -> Path | None:
    """Export traced PoseLSTM to Core ML mlpackage (works on Linux without libcoremlpython runtime)."""
    import torch

    try:
        import coremltools as ct
    except ImportError:
        print("coremltools missing — skip Core ML export")
        return None

    model.eval()
    example = torch.zeros(1, window, input_dim)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)

    wrapped = Wrapper(model)
    traced = torch.jit.trace(wrapped, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="poses", shape=(1, window, input_dim), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )
    # Attach class labels as metadata for the app
    mlmodel.user_defined_metadata["labels"] = json.dumps(labels)
    mlmodel.user_defined_metadata["feature_dim"] = str(input_dim)
    mlmodel.user_defined_metadata["window"] = str(window)
    mlmodel.user_defined_metadata["layout"] = "asl-subtitles.landmark-sequence.v2"
    mlmodel.short_description = "On-device ASL PoseLSTM (hands+face+body+NMM). Privacy: runs locally."
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Also write sidecar labels next to the package
    mlmodel.save(str(out_path))
    labels_sidecar = out_path.with_suffix(".labels.json")
    if out_path.suffix == "":
        labels_sidecar = Path(str(out_path) + ".labels.json")
    # mlpackage is a directory — put labels beside it
    side = out_path.parent / (out_path.stem + ".labels.json")
    side.write_text(json.dumps({"labels": labels, "input": "poses", "shape": [1, window, input_dim]}, indent=2))
    print(f"Core ML saved → {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "models" / "wlasl100_features.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--coreml", type=Path, default=ROOT.parent / "ASLSubtitles" / "Models" / "ASLSignClassifier.mlpackage")
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--aux-weight", type=float, default=0.25)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"missing {args.data} — run convert_wlasl_hdf5.py --mix-synth first", file=sys.stderr)
        return 1

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from pipeline.normalize import FEATURE_DIM
    from pipeline.sequence_model import PoseLSTMClassifier

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    blob = np.load(args.data, allow_pickle=True)
    X = blob["X"].astype(np.float32)
    y = blob["y"].astype(np.int64)
    labels = [str(x) for x in blob["labels"].tolist()]
    splits = blob["split"].tolist() if "split" in blob else ["train"] * len(y)

    Xn = np.stack([normalize_matrix(X[i]) for i in range(len(X))], axis=0)
    aux = nmm_aux_targets(labels, y)

    # Prefer explicit val/test splits when present; else 10% holdout of real (non-synth).
    val_mask = np.array([s == "val" for s in splits])
    test_mask = np.array([s == "test" for s in splits])
    train_mask = np.array([s in ("train", "synth") for s in splits])
    if not val_mask.any():
        idx = rng.permutation(len(Xn))
        n_val = max(1, len(Xn) // 10)
        val_mask = np.zeros(len(Xn), dtype=bool)
        val_mask[idx[:n_val]] = True
        train_mask = ~val_mask

    Xtr, ytr, atr = Xn[train_mask], y[train_mask], aux[train_mask]
    Xva, yva = Xn[val_mask], y[val_mask]
    Xte, yte = (Xn[test_mask], y[test_mask]) if test_mask.any() else (Xva, yva)

    device = torch.device("cpu")
    model = PoseLSTMClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden,
        num_layers=2,
        num_classes=len(labels),
        bidirectional=True,
        dropout=0.3,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(atr)),
        batch_size=args.batch,
        shuffle=True,
    )
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    Xte_t, yte_t = torch.from_numpy(Xte), torch.from_numpy(yte)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb, ab in train_loader:
            opt.zero_grad()
            logits, aux_logits = model(xb, return_aux=True)
            loss = crit(logits, yb) + args.aux_weight * bce(aux_logits, ab)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += float(loss) * len(xb)
            n += len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(Xva_t).argmax(dim=-1)
            acc = float((pred == yva_t).float().mean())
            te_pred = model(Xte_t).argmax(dim=-1)
            te_acc = float((te_pred == yte_t).float().mean())
        print(f"epoch {epoch:02d}  loss={total_loss/max(n,1):.4f}  val_acc={acc:.3f}  test_acc={te_acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_acc = float((model(Xte_t).argmax(-1) == yte_t).float().mean())
        # Confusion top offenders
        pred = model(Xte_t).argmax(-1).numpy()
        confusions: dict[str, int] = {}
        for p, t in zip(pred, yte):
            if p != t:
                key = f"{labels[int(t)]}->{labels[int(p)]}"
                confusions[key] = confusions.get(key, 0) + 1
        top_conf = sorted(confusions.items(), key=lambda kv: -kv[1])[:15]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "num_classes": len(labels),
        "input_dim": FEATURE_DIM,
        "hidden_dim": args.hidden,
        "labels": labels,
        "val_acc": best_acc,
        "test_acc": te_acc,
        "backend": "poselstm-nmm-attn",
        "trained_on": "wlasl100-coco135+synth-fill",
        "note": (
            "Trained on converted WLASL100 pose (COCO-135→FEATURE_DIM v2) with synth fill "
            "for conversational glosses missing from WLASL. Primary deployment: Core ML on-device."
        ),
        "top_confusions": top_conf,
    }
    torch.save(ckpt, args.out)
    (args.out.parent / "labels.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "val_acc": best_acc,
                "test_acc": te_acc,
                "trained_on": ckpt["trained_on"],
                "top_confusions": top_conf,
            },
            indent=2,
        )
    )
    print(f"saved {args.out} val_acc={best_acc:.3f} test_acc={te_acc:.3f} classes={len(labels)}")

    # Eval report
    report = {
        "val_top1": best_acc,
        "test_top1": te_acc,
        "n_classes": len(labels),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()) if test_mask.any() else int(val_mask.sum()),
        "top_confusions": top_conf,
        "privacy": "Training uses public pose dumps offline; app runs Core ML on-device with no upload.",
    }
    (args.out.parent / "eval_report.json").write_text(json.dumps(report, indent=2))

    export_coreml(model, labels, FEATURE_DIM, args.coreml)
    # Also copy mlpackage path note for Documents drop-in
    print("On-device: bundle ASLSignClassifier.mlpackage or copy to Documents/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
