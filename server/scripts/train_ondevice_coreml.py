#!/usr/bin/env python3
"""Train on-device Core ML classifiers (PoseLSTM / TCN-BiLSTM / Transformer).

Primary product path: on-device Core ML. Server .pt is a side artifact for
optional LAN debug — not required for captions. preferRecognitionServer stays OFF.

Usage:
  python scripts/convert_pose_hdf5.py --sources wlasl100,aslcitizen100,wlasl300 --mix-synth --focus-wlasl100
  python scripts/train_ondevice_coreml.py --arch tcn-bilstm --heavy-aug --aug-copies 4
  python scripts/convert_pose_hdf5.py ... --daily-vocab --out models/pose_features_daily.npz
  python scripts/train_ondevice_coreml.py --data models/pose_features_daily.npz --arch tcn-bilstm \\
      --coreml ../ASLSubtitles/Models/ASLSignClassifierDaily.mlpackage --out models/sign_classifier_daily.pt
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


def topk_acc(logits, y, k: int = 5) -> float:
    import torch

    topk = logits.topk(k, dim=-1).indices
    hit = (topk == y.unsqueeze(-1)).any(dim=-1).float().mean()
    return float(hit)


def forward_logits(model, X, batch: int = 256):
    import torch

    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(torch.from_numpy(X[i : i + batch])))
    return torch.cat(outs, dim=0)


def export_coreml(model, labels: list[str], input_dim: int, out_path: Path, window: int, arch: str) -> Path | None:
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
    # Transformer uses dynamic control; trace usually OK with fixed window.
    traced = torch.jit.trace(wrapped, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="poses", shape=(1, window, input_dim), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.user_defined_metadata["labels"] = json.dumps(labels)
    mlmodel.user_defined_metadata["feature_dim"] = str(input_dim)
    mlmodel.user_defined_metadata["window"] = str(window)
    mlmodel.user_defined_metadata["layout"] = "asl-subtitles.landmark-sequence.v2"
    mlmodel.user_defined_metadata["arch"] = arch
    mlmodel.short_description = f"On-device ASL {arch} (hands+face+body+NMM). Privacy: runs locally."
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    side = out_path.parent / (out_path.stem + ".labels.json")
    side.write_text(
        json.dumps({"labels": labels, "input": "poses", "shape": [1, window, input_dim], "arch": arch}, indent=2)
    )
    print(f"Core ML saved → {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "models" / "pose_features.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--coreml", type=Path, default=ROOT.parent / "ASLSubtitles" / "Models" / "ASLSignClassifier.mlpackage")
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--aug-copies", type=int, default=3, help="Extra augmented copies of each real train sample")
    ap.add_argument("--heavy-aug", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.04)
    ap.add_argument("--arch", default="tcn-bilstm", help="poselstm | tcn-bilstm | transformer")
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--report", type=Path, default=None, help="Override eval_report path")
    args = ap.parse_args()

    if not args.data.exists():
        legacy = ROOT / "models" / "wlasl100_features.npz"
        if legacy.exists():
            args.data = legacy
        else:
            print(f"missing {args.data} — run convert_pose_hdf5.py --mix-synth first", file=sys.stderr)
            return 1

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from pipeline.augment import augment_sequence, augment_sequence_heavy
    from pipeline.normalize import FEATURE_DIM
    from pipeline.sequence_model import build_sequence_model

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    aug_fn = augment_sequence_heavy if args.heavy_aug else augment_sequence

    blob = np.load(args.data, allow_pickle=True)
    X = blob["X"].astype(np.float32)
    y = blob["y"].astype(np.int64)
    labels = [str(x) for x in blob["labels"].tolist()]
    splits = blob["split"].tolist() if "split" in blob else ["train"] * len(y)
    sources = blob["source"].tolist() if "source" in blob else ["unknown"] * len(y)
    window = int(X.shape[1])

    print(f"loaded X={X.shape} classes={len(labels)} arch={args.arch} unique_sources={sorted(set(sources))}")

    val_mask = np.array([s == "val" for s in splits])
    test_mask = np.array([s == "test" for s in splits])
    train_mask = np.array([s in ("train", "synth") for s in splits])
    if not val_mask.any():
        idx = rng.permutation(len(X))
        n_val = max(1, len(X) // 10)
        val_mask = np.zeros(len(X), dtype=bool)
        val_mask[idx[:n_val]] = True
        train_mask = ~val_mask

    print("normalizing…")
    Xn = np.stack([normalize_matrix(X[i]) for i in range(len(X))], axis=0)
    aux = nmm_aux_targets(labels, y)

    Xtr_base, ytr_base, atr_base = Xn[train_mask], y[train_mask], aux[train_mask]
    aug_X, aug_y, aug_a = [Xtr_base], [ytr_base], [atr_base]
    real_train_idx = [i for i, s in enumerate(np.array(splits)[train_mask]) if s == "train"]
    if args.aug_copies > 0 and real_train_idx:
        print(f"augmenting {len(real_train_idx)} real train × {args.aug_copies} ({'heavy' if args.heavy_aug else 'std'})…")
        for _ in range(args.aug_copies):
            xs, ys, as_ = [], [], []
            for j in real_train_idx:
                xs.append(aug_fn(Xtr_base[j], rng))
                ys.append(ytr_base[j])
                as_.append(atr_base[j])
            aug_X.append(np.stack(xs, axis=0))
            aug_y.append(np.asarray(ys, dtype=np.int64))
            aug_a.append(np.stack(as_, axis=0))
    Xtr = np.concatenate(aug_X, axis=0)
    ytr = np.concatenate(aug_y, axis=0)
    atr = np.concatenate(aug_a, axis=0)

    Xva, yva = Xn[val_mask], y[val_mask]
    Xte, yte = (Xn[test_mask], y[test_mask]) if test_mask.any() else (Xva, yva)

    src_arr = np.array(sources)
    w100_val = val_mask & (src_arr == "wlasl100")
    w100_test = test_mask & (src_arr == "wlasl100")

    counts = np.bincount(ytr, minlength=len(labels)).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    class_w = 1.0 / np.sqrt(counts)
    class_w = class_w / class_w.sum() * len(labels)
    sample_w = class_w[ytr]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )

    device = torch.device("cpu")
    model_kwargs = dict(
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=len(labels),
        bidirectional=True,
        dropout=0.3,
        attn_heads=args.attn_heads,
    )
    model = build_sequence_model(args.arch, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,} hidden={args.hidden} layers={args.layers} arch={args.arch}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    ce_w = torch.as_tensor(class_w, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=ce_w, label_smoothing=args.label_smoothing)
    bce = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(atr)),
        batch_size=args.batch,
        sampler=sampler,
        drop_last=False,
    )
    yva_t = torch.from_numpy(yva)
    yte_t = torch.from_numpy(yte)

    best_acc = -1.0
    best_state = None
    history = []
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
        sched.step()
        model.eval()
        va_logits = forward_logits(model, Xva)
        te_logits = forward_logits(model, Xte)
        acc = float((va_logits.argmax(-1) == yva_t).float().mean())
        te_acc = float((te_logits.argmax(-1) == yte_t).float().mean())
        va5 = topk_acc(va_logits, yva_t, 5)
        te5 = topk_acc(te_logits, yte_t, 5)
        history.append({"epoch": epoch, "loss": total_loss / max(n, 1), "val_top1": acc, "test_top1": te_acc})
        print(
            f"epoch {epoch:02d}  loss={total_loss/max(n,1):.4f}  "
            f"val@1={acc:.3f} val@5={va5:.3f}  test@1={te_acc:.3f} test@5={te5:.3f}"
        )
        score = acc
        if w100_val.any():
            with torch.no_grad():
                w_logits = forward_logits(model, Xn[w100_val])
                score = float((w_logits.argmax(-1) == torch.from_numpy(y[w100_val])).float().mean())
            print(f"         wlasl100_val@1={score:.3f}")
        if score >= best_acc:
            best_acc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    def score_split(Xm, ym):
        logits = forward_logits(model, Xm)
        yt = torch.from_numpy(ym)
        t1 = float((logits.argmax(-1) == yt).float().mean())
        t5 = topk_acc(logits, yt, 5)
        pred = logits.argmax(-1).numpy()
        return t1, t5, pred, logits

    te_acc, te5, pred, te_logits = score_split(Xte, yte)
    va_acc, va5, _, _ = score_split(Xva, yva)

    w100_metrics = {}
    if w100_val.any():
        t1, t5, _, _ = score_split(Xn[w100_val], y[w100_val])
        w100_metrics["val_top1"] = t1
        w100_metrics["val_top5"] = t5
        w100_metrics["n_val"] = int(w100_val.sum())
    if w100_test.any():
        t1, t5, _, _ = score_split(Xn[w100_test], y[w100_test])
        w100_metrics["test_top1"] = t1
        w100_metrics["test_top5"] = t5
        w100_metrics["n_test"] = int(w100_test.sum())

    confusions: dict[str, int] = {}
    for p, t in zip(pred, yte):
        if p != t:
            key = f"{labels[int(t)]}->{labels[int(p)]}"
            confusions[key] = confusions.get(key, 0) + 1
    top_conf = sorted(confusions.items(), key=lambda kv: -kv[1])[:15]

    trained_on = "+".join(sorted(set(sources)))
    arch_name = f"{args.arch}-nmm"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "num_classes": len(labels),
        "input_dim": FEATURE_DIM,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "labels": labels,
        "val_acc": va_acc,
        "val_top5": va5,
        "test_acc": te_acc,
        "test_top5": te5,
        "backend": arch_name,
        "arch": args.arch,
        "trained_on": trained_on,
        "note": (
            f"{args.arch} on multi-source public pose (COCO-135→FEATURE_DIM v2) with "
            "gloss_map (Citizen sense/synonym) + aug + class-balanced sampling. Primary: Core ML on-device."
        ),
        "top_confusions": top_conf,
        "wlasl100_subset": w100_metrics,
        "previous_baseline": {"val_top1": 0.2574, "test_top1": 0.2093, "n_classes": 234},
        "window": window,
    }
    torch.save(ckpt, args.out)
    (args.out.parent / "labels.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "val_acc": va_acc,
                "val_top5": va5,
                "test_acc": te_acc,
                "test_top5": te5,
                "trained_on": trained_on,
                "arch": args.arch,
                "top_confusions": top_conf,
                "wlasl100_subset": w100_metrics,
            },
            indent=2,
        )
    )
    print(
        f"saved {args.out} val@1={va_acc:.3f} val@5={va5:.3f} "
        f"test@1={te_acc:.3f} test@5={te5:.3f} classes={len(labels)}"
    )
    if w100_metrics:
        print(f"WLASL100-subset: {w100_metrics}")

    prev_test = 0.2093
    delta = None
    if w100_metrics.get("test_top1") is not None:
        delta = (w100_metrics["test_top1"] - prev_test) * 100.0

    report = {
        "val_top1": va_acc,
        "val_top5": va5,
        "test_top1": te_acc,
        "test_top5": te5,
        "n_classes": len(labels),
        "n_train": int(len(Xtr)),
        "n_train_raw": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()) if test_mask.any() else int(val_mask.sum()),
        "top_confusions": top_conf,
        "trained_on": trained_on,
        "arch": arch_name,
        "aug": f"{'heavy' if args.heavy_aug else 'std'} mirror/speed/noise/shift/dropout x{args.aug_copies}",
        "class_balancing": "WeightedRandomSampler + CE class weights",
        "wlasl100_subset": w100_metrics,
        "previous": {
            "val_top1": 0.2574,
            "test_top1": 0.2093,
            "n_classes": 234,
            "trained_on": "synth+wlasl100+wlasl300",
        },
        "delta_vs_previous_wlasl100_test_pp": delta,
        "privacy": "offline public pose train; on-device Core ML inference; preferServer=false",
        "history_tail": history[-5:],
        "window": window,
        "gloss_map": True,
    }
    report_path = args.report or (args.out.parent / "eval_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    export_coreml(model, labels, FEATURE_DIM, args.coreml, window, arch_name)
    print("On-device: bundle ASLSignClassifier.mlpackage or copy to Documents/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
