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
    ap.add_argument("--mixup", type=float, default=0.0, help="Mixup alpha (0=off); try 0.2")
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--teacher", type=Path, default=None, help="Teacher .pt for distillation")
    ap.add_argument("--distill-temp", type=float, default=2.0)
    ap.add_argument("--distill-alpha", type=float, default=0.6, help="Weight on KL distill vs CE")
    ap.add_argument("--train-teacher", action="store_true", help="First train larger teacher then distill to student")
    ap.add_argument("--teacher-hidden", type=int, default=320)
    ap.add_argument("--teacher-layers", type=int, default=3)
    ap.add_argument("--bigram-rerank", action="store_true", help="Report bigram top-5 rerank metrics")
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

    from pipeline.augment import augment_sequence, augment_sequence_heavy, mixup_pair
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
    if args.train_teacher and not (args.teacher and Path(args.teacher).exists()):
        # Train a larger teacher in-process, save, then distill into student below.
        print(f"=== training teacher hidden={args.teacher_hidden} layers={args.teacher_layers} ===")
        teacher_path = args.out.with_name(args.out.stem + "_teacher.pt")
        t_kwargs = dict(model_kwargs)
        t_kwargs["hidden_dim"] = args.teacher_hidden
        t_kwargs["num_layers"] = args.teacher_layers
        teacher_model = build_sequence_model(args.arch, **t_kwargs).to(device)
        t_opt = torch.optim.AdamW(teacher_model.parameters(), lr=args.lr, weight_decay=2e-4)
        t_ce = nn.CrossEntropyLoss(weight=torch.as_tensor(class_w, dtype=torch.float32), label_smoothing=args.label_smoothing)
        t_bce = nn.BCEWithLogitsLoss()
        t_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(atr)),
            batch_size=args.batch,
            sampler=WeightedRandomSampler(
                weights=torch.as_tensor(sample_w, dtype=torch.double),
                num_samples=len(sample_w),
                replacement=True,
            ),
        )
        t_best, t_state = -1.0, None
        t_epochs = max(12, args.epochs // 2)
        for epoch in range(1, t_epochs + 1):
            teacher_model.train()
            for xb, yb, ab in t_loader:
                t_opt.zero_grad()
                logits, aux_logits = teacher_model(xb, return_aux=True)
                loss = t_ce(logits, yb) + args.aux_weight * t_bce(aux_logits, ab)
                loss.backward()
                nn.utils.clip_grad_norm_(teacher_model.parameters(), 2.0)
                t_opt.step()
            teacher_model.eval()
            with torch.no_grad():
                if w100_val.any():
                    w_logits = forward_logits(teacher_model, Xn[w100_val])
                    score = float((w_logits.argmax(-1) == torch.from_numpy(y[w100_val])).float().mean())
                else:
                    va_logits = forward_logits(teacher_model, Xva)
                    score = float((va_logits.argmax(-1) == torch.from_numpy(yva)).float().mean())
            print(f"  teacher epoch {epoch:02d} score={score:.3f}")
            if score >= t_best:
                t_best = score
                t_state = {k: v.detach().cpu().clone() for k, v in teacher_model.state_dict().items()}
        assert t_state is not None
        torch.save({
            "state_dict": t_state,
            "hidden_dim": args.teacher_hidden,
            "num_layers": args.teacher_layers,
            "arch": args.arch,
            "num_classes": len(labels),
            "labels": labels,
        }, teacher_path)
        args.teacher = teacher_path
        print(f"teacher saved → {teacher_path} best={t_best:.3f}")

    model = build_sequence_model(args.arch, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params:,} hidden={args.hidden} layers={args.layers} arch={args.arch}")

    # Optional teacher for distillation
    teacher = None
    if args.teacher and args.teacher.exists():
        tckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
        t_hidden = int(tckpt.get("hidden_dim", args.teacher_hidden))
        t_layers = int(tckpt.get("num_layers", args.teacher_layers))
        t_arch = tckpt.get("arch", args.arch)
        teacher = build_sequence_model(
            t_arch,
            input_dim=FEATURE_DIM,
            hidden_dim=t_hidden,
            num_layers=t_layers,
            num_classes=len(labels),
            bidirectional=True,
            dropout=0.0,
            attn_heads=args.attn_heads,
        )
        teacher.load_state_dict(tckpt["state_dict"])
        teacher.eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        print(f"distill teacher loaded from {args.teacher} hidden={t_hidden} layers={t_layers}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)

    def lr_at(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            return args.lr * epoch / max(args.warmup_epochs, 1)
        # cosine after warmup
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        import math as _math
        return args.lr * 0.05 + 0.5 * (args.lr - args.lr * 0.05) * (1.0 + _math.cos(_math.pi * progress))

    ce_w = torch.as_tensor(class_w, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=ce_w, label_smoothing=args.label_smoothing)
    bce = nn.BCEWithLogitsLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
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
        # warmup + cosine LR
        lr_now = lr_at(epoch)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb, ab in train_loader:
            opt.zero_grad()
            # Online mixup on pose sequences
            if args.mixup and args.mixup > 0:
                perm = torch.randperm(xb.size(0))
                xb2, yb2, ab2 = xb[perm], yb[perm], ab[perm]
                lam = float(torch.distributions.Beta(args.mixup, args.mixup).sample())
                lam = max(lam, 1.0 - lam)  # prefer dominant label
                xb = lam * xb + (1.0 - lam) * xb2
                ab = lam * ab + (1.0 - lam) * ab2
                logits, aux_logits = model(xb, return_aux=True)
                loss_ce = lam * crit(logits, yb) + (1.0 - lam) * crit(logits, yb2)
            else:
                logits, aux_logits = model(xb, return_aux=True)
                loss_ce = crit(logits, yb)
            loss = loss_ce + args.aux_weight * bce(aux_logits, ab)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(xb)
                T = args.distill_temp
                log_p = torch.log_softmax(logits / T, dim=-1)
                q = torch.softmax(t_logits / T, dim=-1)
                loss_kd = kl(log_p, q) * (T * T)
                loss = args.distill_alpha * loss_kd + (1.0 - args.distill_alpha) * loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += float(loss) * len(xb)
            n += len(xb)
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
        "mixup": args.mixup,
        "distill": bool(teacher is not None),
    }
    if args.bigram_rerank or True:
        from pipeline.bigram_prior import build_bigram_counts, eval_with_bigram, save_prior
        prior = build_bigram_counts(labels)
        save_prior(args.out.parent / "gloss_bigram_prior.json", labels)
        te_logits_np = forward_logits(model, Xte).numpy()
        report["bigram_test"] = eval_with_bigram(te_logits_np, yte, labels, prior)
        if w100_test.any():
            w_logits_np = forward_logits(model, Xn[w100_test]).numpy()
            report["bigram_wlasl100_test"] = eval_with_bigram(w_logits_np, y[w100_test], labels, prior)
        print(f"bigram_test: {report.get('bigram_test')}")
        if "bigram_wlasl100_test" in report:
            print(f"bigram_wlasl100_test: {report['bigram_wlasl100_test']}")
    report_path = args.report or (args.out.parent / "eval_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    export_coreml(model, labels, FEATURE_DIM, args.coreml, window, arch_name)
    print("On-device: bundle ASLSignClassifier.mlpackage or copy to Documents/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
