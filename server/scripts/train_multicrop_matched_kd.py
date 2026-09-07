#!/usr/bin/env python3
"""Multicrop-matched KD: train students on native-length random crops.

Inference uses evenly spaced 32-frame crops over ≤96 context (eval_multicrop).
Convert historically used last-32 only — this script trains on random crops from
the same native HDF5 so train/inference crop distributions match.

Val-gates on multicrop WLASL100 (val-A/B). Online KD from a frozen teacher
(default: shipping ens2). preferServer stays off; no product features.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.augment import augment_sequence_heavy  # noqa: E402
from pipeline.coco135 import coco135_sequence_to_features, pad_or_trim  # noqa: E402
from pipeline.gloss_map import canonicalize_gloss  # noqa: E402
from pipeline.normalize import FEATURE_DIM  # noqa: E402
from pipeline.sequence_model import build_sequence_model, load_ship_checkpoint  # noqa: E402
from scripts.eval_multicrop import load_wlasl100, score, temporal_crops  # noqa: E402
from scripts.train_ondevice_coreml import normalize_matrix, topk_acc, nmm_aux_targets  # noqa: E402

SOURCE_SPECS = {
    "wlasl100": ("wlasl100", "WLASL100_135"),
    "wlasl300": ("wlasl300", "WLASL300_135"),
    "wlasl2000": ("wlasl2000", "WLASL2000_135"),
    "aslcitizen300": ("aslcitizen300", "ASLCitizen300_135"),
    "aslcitizen2731": ("aslcitizen2731", "ASLCitizen2731_135"),
}


def random_crop(feat: np.ndarray, window: int, max_t: int, rng: np.random.Generator) -> np.ndarray:
    """Match inference crop family: last max_t context, then a random 32-window."""
    if feat.shape[0] > max_t:
        feat = feat[-max_t:]
    t = feat.shape[0]
    if t <= window:
        return pad_or_trim(feat, window)
    start = int(rng.integers(0, t - window + 1))
    return feat[start : start + window].astype(np.float32)


def load_native_train(
    sources: list[str],
    labels: list[str],
    data_root: Path,
    max_per_source: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    lab_to_i = {g: i for i, g in enumerate(labels)}
    allowed = set(labels)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    srcs: list[str] = []
    for src in sources:
        if src not in SOURCE_SPECS:
            print(f"skip unknown source {src}")
            continue
        folder, prefix = SOURCE_SPECS[src]
        path = data_root / folder / f"{prefix}-Train.hdf5"
        if not path.exists():
            print(f"missing {path}")
            continue
        n_before = len(xs)
        with h5py.File(path, "r") as f:
            keys = sorted(f.keys(), key=lambda k: int(k) if k.isdigit() else k)
            if max_per_source is not None and len(keys) > max_per_source and src not in ("wlasl100", "wlasl300"):
                rng = np.random.default_rng(abs(hash(src)) % (2**31))
                keys = list(rng.choice(keys, size=max_per_source, replace=False))
            for key in keys:
                g = f[key]
                raw = g["data"][:].astype(np.float32)
                lab = g["label"][()]
                if isinstance(lab, bytes):
                    lab = lab.decode("utf-8")
                gloss = canonicalize_gloss(str(lab).strip(), allowed=allowed)
                if gloss not in lab_to_i or raw.shape[0] < 6:
                    continue
                xs.append(coco135_sequence_to_features(raw))
                ys.append(lab_to_i[gloss])
                srcs.append(src)
        print(f"native {src}: +{len(xs) - n_before} (total {len(xs)})")
    return xs, np.asarray(ys, dtype=np.int64), srcs


class NativeCropDataset(Dataset):
    def __init__(
        self,
        xs: list[np.ndarray],
        ys: np.ndarray,
        aux: np.ndarray,
        sources: list[str],
        window: int,
        max_t: int,
        seed: int,
        heavy_aug: bool = True,
        n_crops_train: int = 1,
    ):
        self.xs = xs
        self.ys = ys
        self.aux = aux
        self.sources = sources
        self.window = window
        self.max_t = max_t
        self.heavy_aug = heavy_aug
        self.n_crops_train = max(1, n_crops_train)
        self.base_seed = seed

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, idx: int):
        # unique RNG per index+epoch via torch worker seed is hard; use numpy with idx salt
        salt = int(getattr(self, "epoch_salt", 0))
        rng = np.random.default_rng((self.base_seed + idx * 10007 + salt * 131) & 0x7FFFFFFF)
        crops = []
        for _ in range(self.n_crops_train):
            c = random_crop(self.xs[idx], self.window, self.max_t, rng)
            c = normalize_matrix(c)
            if self.heavy_aug:
                c = augment_sequence_heavy(c, rng)
            crops.append(c)
        x = np.stack(crops, 0).astype(np.float32)  # (C, T, D)
        return (
            torch.from_numpy(x),
            int(self.ys[idx]),
            torch.from_numpy(self.aux[idx]),
        )


def val_ab_indices(n_val: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_val)
    cut = len(perm) // 2
    return perm[:cut], perm[cut:]


@torch.no_grad()
def multicrop_logits(model, xs: list[np.ndarray], n_crops: int, max_t: int) -> torch.Tensor:
    logits = []
    for feat in xs:
        crops = temporal_crops(feat, n_crops, max_t)
        xn = np.stack([normalize_matrix(c.astype(np.float32)) for c in crops], 0)
        logits.append(model(torch.as_tensor(xn)).mean(0))
    return torch.stack(logits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "models" / "pose_features_dense.npz")
    ap.add_argument("--teacher", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--init-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--sources", default="wlasl100,wlasl300,aslcitizen300")
    ap.add_argument("--max-per-source", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--max-context", type=int, default=96)
    ap.add_argument("--eval-crops", type=int, default=5)
    ap.add_argument("--train-crops", type=int, default=2, help="random crops averaged in train loss")
    ap.add_argument("--distill-alpha", type=float, default=0.62)
    ap.add_argument("--distill-temp", type=float, default=2.5)
    ap.add_argument("--mixup", type=float, default=0.12)
    ap.add_argument("--wlasl100-boost", type=float, default=3.5)
    ap.add_argument("--wlasl300-boost", type=float, default=1.2)
    ap.add_argument("--citizen-boost", type=float, default=1.15)
    ap.add_argument("--label-smoothing", type=float, default=0.03)
    ap.add_argument("--aux-weight", type=float, default=0.25)
    ap.add_argument("--dropout", type=float, default=0.28)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--ship-test", type=float, default=0.5930232405662537)
    ap.add_argument("--cache", type=Path, default=ROOT / "models" / "native_train_cache_mc.pt")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--no-heavy-aug", action="store_true", help="crop-only aug (no heavy pose aug)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    blob = np.load(args.data, allow_pickle=True)
    labels = [str(x) for x in blob["labels"].tolist()]
    print(f"labels={len(labels)} init={args.init_from.name} teacher={args.teacher.name}")

    if args.cache.exists() and not args.rebuild_cache:
        print(f"loading cache {args.cache}")
        cache = torch.load(args.cache, map_location="cpu", weights_only=False)
        xs, ys, srcs = cache["xs"], cache["ys"], cache["sources"]
        assert cache["labels"] == labels
    else:
        t0 = time.time()
        xs, ys, srcs = load_native_train(
            [s.strip() for s in args.sources.split(",") if s.strip()],
            labels,
            ROOT / "data",
            max_per_source=args.max_per_source,
        )
        # also fold dense NPZ synth + any last-32 train as short sequences
        split = np.array([str(s) for s in blob["split"].tolist()])
        source = np.array([str(s) for s in blob["source"].tolist()])
        y_all = blob["y"].astype(np.int64)
        X_all = blob["X"].astype(np.float32)
        synth_m = split == "synth"
        extra_x = [X_all[i] for i in np.where(synth_m)[0]]
        extra_y = y_all[synth_m]
        xs.extend(extra_x)
        ys = np.concatenate([ys, extra_y]) if len(extra_y) else ys
        srcs.extend(["synth"] * len(extra_x))
        torch.save({"xs": xs, "ys": ys, "sources": srcs, "labels": labels}, args.cache)
        print(f"cache saved n={len(xs)} in {time.time()-t0:.1f}s")

    aux = nmm_aux_targets(labels, ys)
    src_arr = np.array(srcs)
    counts = np.bincount(ys, minlength=len(labels)).astype(np.float64).clip(1.0, None)
    class_w = 1.0 / np.sqrt(counts)
    class_w = class_w / class_w.sum() * len(labels)
    sample_w = class_w[ys].astype(np.float64).copy()
    sample_w[src_arr == "wlasl100"] *= args.wlasl100_boost
    sample_w[src_arr == "wlasl300"] *= args.wlasl300_boost
    sample_w[np.char.startswith(src_arr, "aslcitizen")] *= args.citizen_boost
    sample_w[src_arr == "synth"] *= 0.35
    print(
        f"train n={len(ys)} mean_w={sample_w.mean():.3f} "
        f"w100={(src_arr=='wlasl100').sum()} w300={(src_arr=='wlasl300').sum()} "
        f"cit={np.char.startswith(src_arr,'aslcitizen').sum()} synth={(src_arr=='synth').sum()}"
    )

    # val/test native for gating
    va_x, va_y = load_wlasl100("Val", labels)
    te_x, te_y = load_wlasl100("Test", labels)
    a_local, b_local = val_ab_indices(len(va_y), seed=0)
    print(f"multicrop val n={len(va_y)} test n={len(te_y)} A={len(a_local)} B={len(b_local)}")

    teacher, _, _ = load_ship_checkpoint(args.teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    init_ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    model = build_sequence_model(
        "tcn-bilstm",
        input_dim=FEATURE_DIM,
        hidden_dim=int(init_ck.get("hidden_dim", 192)),
        num_layers=int(init_ck.get("num_layers", 2)),
        num_classes=len(labels),
        bidirectional=True,
        dropout=args.dropout,
    )
    # support plain student or pull member-0 if somehow ens
    sd = init_ck["state_dict"]
    if any(k.startswith("base.models.0.") for k in sd):
        sd = {k[len("base.models.0.") :]: v for k, v in sd.items() if k.startswith("base.models.0.")}
    elif any(k.startswith("models.0.") for k in sd):
        sd = {k[len("models.0.") :]: v for k, v in sd.items() if k.startswith("models.0.")}
    model.load_state_dict(sd, strict=False)
    print(f"warm-start from {args.init_from}")

    ds = NativeCropDataset(
        xs, ys, aux, srcs, args.window, args.max_context, args.seed,
        heavy_aug=not args.no_heavy_aug, n_crops_train=args.train_crops,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True,
        ),
        num_workers=0,
        drop_last=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    crit = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_w, dtype=torch.float32),
        label_smoothing=args.label_smoothing,
    )
    bce = nn.BCEWithLogitsLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    def lr_at(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            return args.lr * epoch / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return args.lr * 0.15 + 0.5 * (args.lr - args.lr * 0.15) * (1.0 + math.cos(math.pi * progress))

    @torch.no_grad()
    def score_mc(m):
        Lva = multicrop_logits(m, va_x, args.eval_crops, args.max_context)
        Lte = multicrop_logits(m, te_x, args.eval_crops, args.max_context)
        yva_t = torch.as_tensor(va_y)
        yte_t = torch.as_tensor(te_y)
        sa = float((Lva[a_local].argmax(-1) == yva_t[a_local]).float().mean())
        sb = float((Lva[b_local].argmax(-1) == yva_t[b_local]).float().mean())
        va1 = float(topk_acc(Lva, yva_t, 1))
        te1 = float(topk_acc(Lte, yte_t, 1))
        te5 = float(topk_acc(Lte, yte_t, 5))
        return {
            "val_A": sa,
            "val_B": sb,
            "val_top1": va1,
            "test_top1": te1,
            "test_top5": te5,
            "correct": int((Lte.argmax(-1) == yte_t).sum()),
            "n_test": int(len(te_y)),
        }

    model.eval()
    base = score_mc(model)
    print(
        f"baseline multicrop valA={base['val_A']:.3f} valB={base['val_B']:.3f} "
        f"val={base['val_top1']:.3f} te={base['test_top1']:.3f} "
        f"({base['correct']}/{base['n_test']}) ship={args.ship_test:.4f}"
    )

    best_acc, best_b = base["val_A"], base["val_B"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_metrics = base
    history = [dict(epoch=0, **base)]

    for epoch in range(1, args.epochs + 1):
        lr_now = lr_at(epoch)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        model.train()
        ds.epoch_salt = epoch
        total_loss = 0.0
        n = 0
        t0 = time.time()
        for xb, yb, ab in loader:
            # xb: (B, C, T, D) — treat each crop as an independent sample
            B, C, T, D = xb.shape
            xb_flat = xb.reshape(B * C, T, D)
            yb = yb.long()
            ab = ab.float()
            y_flat = yb.repeat_interleave(C)
            a_flat = ab.repeat_interleave(C, dim=0)
            opt.zero_grad()
            with torch.no_grad():
                t_logits = teacher(xb_flat)
            if args.mixup > 0:
                perm = torch.randperm(B * C)
                xb2 = xb_flat[perm]
                yb2 = y_flat[perm]
                ab2 = a_flat[perm]
                with torch.no_grad():
                    t2 = teacher(xb2)
                lam = float(torch.distributions.Beta(args.mixup, args.mixup).sample())
                lam = max(lam, 1.0 - lam)
                xb_m = lam * xb_flat + (1.0 - lam) * xb2
                ab_m = lam * a_flat + (1.0 - lam) * ab2
                tb_use = lam * t_logits + (1.0 - lam) * t2
                logits, aux_logits = model(xb_m, return_aux=True)
                loss_ce = lam * crit(logits, y_flat) + (1.0 - lam) * crit(logits, yb2)
                loss = loss_ce + args.aux_weight * bce(aux_logits, ab_m)
            else:
                tb_use = t_logits
                logits, aux_logits = model(xb_flat, return_aux=True)
                loss = crit(logits, y_flat) + args.aux_weight * bce(aux_logits, a_flat)
            Td = args.distill_temp
            loss_kd = kl(torch.log_softmax(logits / Td, dim=-1), torch.softmax(tb_use / Td, dim=-1)) * (Td * Td)
            loss = args.distill_alpha * loss_kd + (1.0 - args.distill_alpha) * loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += float(loss) * B
            n += B

        model.eval()
        metrics = score_mc(model)
        history.append({"epoch": epoch, "loss": total_loss / max(n, 1), **metrics})
        sa, sb = metrics["val_A"], metrics["val_B"]
        print(
            f"epoch {epoch:02d} loss={total_loss/max(n,1):.4f} "
            f"valA={sa:.3f} valB={sb:.3f} val={metrics['val_top1']:.3f} "
            f"te={metrics['test_top1']:.3f} ({metrics['correct']}/{metrics['n_test']}) "
            f"lr={lr_now:.2e} {time.time()-t0:.1f}s"
        )
        if (sa + sb >= best_acc + best_b - 1e-9) and (sa + 0.015 >= best_acc) and (sb + 0.015 >= best_b):
            best_acc, best_b = sa, sb
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            print(f"  * new best (A/B sum gate) A={sa:.3f} B={sb:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    final = score_mc(model)
    print(
        f"FINAL multicrop val={final['val_top1']:.4f} te={final['test_top1']:.4f} "
        f"({final['correct']}/{final['n_test']}) dpp={(final['test_top1']-args.ship_test)*100:+.3f}"
    )

    ckpt = {
        "state_dict": best_state,
        "num_classes": len(labels),
        "input_dim": FEATURE_DIM,
        "hidden_dim": int(init_ck.get("hidden_dim", 192)),
        "num_layers": int(init_ck.get("num_layers", 2)),
        "labels": labels,
        "arch": "tcn-bilstm",
        "backend": "pytorch",
        "window": args.window,
        "note": (
            f"multicrop-matched KD from {args.teacher.name}; init {args.init_from.name}; "
            f"train_crops={args.train_crops} max_context={args.max_context}"
        ),
        "wlasl100_subset": {
            "val_top1": final["val_top1"],
            "val_top5": float("nan"),
            "n_val": len(va_y),
            "test_top1": final["test_top1"],
            "test_top5": final["test_top5"],
            "n_test": final["n_test"],
            "method": "multicrop-matched KD; val-A/B gated",
            "val_A": final["val_A"],
            "val_B": final["val_B"],
            "multi_crop_count": args.eval_crops,
            "max_context_frames": args.max_context,
        },
        "comparable_wlasl100": {
            "val_top1": final["val_top1"],
            "test_top1": final["test_top1"],
            "test_top5": final["test_top5"],
            "n_val": len(va_y),
            "n_test": final["n_test"],
            "method": "multicrop nc=5 max96 plain",
        },
        "trained_on": "+".join(sorted(set(srcs))),
        "history": history,
        "best_metrics": best_metrics,
    }
    # fill val top5
    with torch.no_grad():
        Lva = multicrop_logits(model, va_x, args.eval_crops, args.max_context)
        ckpt["wlasl100_subset"]["val_top5"] = float(topk_acc(Lva, torch.as_tensor(va_y), 5))

    torch.save(ckpt, args.out)
    report = {
        "arch": "tcn-bilstm-multicrop-matched-kd",
        "init_from": str(args.init_from),
        "teacher": str(args.teacher),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_crops": args.train_crops,
        "n_train": len(ys),
        "wlasl100_subset": ckpt["wlasl100_subset"],
        "comparable_wlasl100": ckpt["comparable_wlasl100"],
        "previous_ship": {"test_top1": args.ship_test, "commit": "7a0f14b"},
        "delta_vs_7a0f14b_wlasl100_test_pp": (final["test_top1"] - args.ship_test) * 100.0,
        "privacy": "offline public pose; preferServer=false",
        "notes": ckpt["note"],
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f"saved {args.out} report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
