#!/usr/bin/env python3
"""Long single-student KD from cached ens5 teacher logits (CPU-friendly).

Maps augmented train rows to parent sample teacher logits. Val-selects on
WLASL100. Optional heavy WLASL100 finetune. No ensemble weight search.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.augment import augment_sequence_heavy  # noqa: E402
from pipeline.normalize import FEATURE_DIM  # noqa: E402
from pipeline.sequence_model import build_sequence_model  # noqa: E402
from scripts.train_ondevice_coreml import (  # noqa: E402
    export_coreml,
    forward_logits,
    nmm_aux_targets,
    normalize_matrix,
    topk_acc,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "models" / "pose_features_dense.npz")
    ap.add_argument("--teacher-logits", type=Path, default=ROOT / "models" / "ens5_teacher_logits_dense.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "sign_classifier_distill_ens5_s67.pt")
    ap.add_argument("--report", type=Path, default=ROOT / "models" / "eval_report_distill_ens5_s67.json")
    ap.add_argument("--coreml", type=Path, default=Path("/tmp/ASLSignClassifier_s67_probe.mlpackage"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=4.0e-4)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--aug-copies", type=int, default=4)
    ap.add_argument("--mixup", type=float, default=0.15)
    ap.add_argument("--distill-alpha", type=float, default=0.68)
    ap.add_argument("--distill-temp", type=float, default=2.5)
    ap.add_argument("--wlasl100-boost", type=float, default=3.8)
    ap.add_argument("--real-boost", type=float, default=1.5)
    ap.add_argument("--synth-boost", type=float, default=0.3)
    ap.add_argument("--wlasl300-boost", type=float, default=1.15)
    ap.add_argument("--citizen-overlap-boost", type=float, default=1.35)
    ap.add_argument("--finetune-wlasl100-epochs", type=int, default=40)
    ap.add_argument("--finetune-lr", type=float, default=5e-5)
    ap.add_argument("--warmup-epochs", type=int, default=4)
    ap.add_argument("--swa-start", type=int, default=75)
    ap.add_argument("--dropout", type=float, default=0.28)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--label-smoothing", type=float, default=0.04)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--hard-mine-from", type=Path, default=ROOT / "models" / "hard_mine_confusions.json")
    ap.add_argument("--hard-class-boost", type=float, default=1.9)
    ap.add_argument("--boost-glosses", default="SHORT,FORGET,FOOD,WHY,WEATHER,HEARING,TWO,FAMILY,WHO")
    ap.add_argument("--gloss-boost", type=float, default=2.3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    blob = np.load(args.data, allow_pickle=True)
    X = blob["X"].astype(np.float32)
    y = blob["y"].astype(np.int64)
    labels = [str(x) for x in blob["labels"].tolist()]
    splits = [str(s) for s in blob["split"].tolist()]
    sources = [str(s) for s in blob["source"].tolist()]

    tblob = torch.load(args.teacher_logits, map_location="cpu", weights_only=False)
    t_all = tblob["logits"].float()
    assert t_all.shape[0] == len(X), (t_all.shape, len(X))

    print(f"loaded X={X.shape} classes={len(labels)} teacher_logits={tuple(t_all.shape)}")
    print("normalizing…")
    Xn = np.stack([normalize_matrix(X[i]) for i in range(len(X))], axis=0)
    aux = nmm_aux_targets(labels, y)

    split_arr = np.array(splits)
    src_arr = np.array(sources)
    train_mask = np.isin(split_arr, ["train", "synth"])
    val_mask = split_arr == "val"
    test_mask = split_arr == "test"
    w100_val = val_mask & (src_arr == "wlasl100")
    w100_test = test_mask & (src_arr == "wlasl100")

    train_idx = np.where(train_mask)[0]
    Xtr_base, ytr_base, atr_base = Xn[train_mask], y[train_mask], aux[train_mask]
    t_base = t_all[torch.as_tensor(train_idx)]
    real_train_local = [i for i, s in enumerate(split_arr[train_mask]) if s == "train"]

    aug_X = [Xtr_base]
    aug_y = [ytr_base]
    aug_a = [atr_base]
    aug_t = [t_base]
    parent_global = [train_idx]
    if args.aug_copies > 0 and real_train_local:
        print(f"augmenting {len(real_train_local)} real train × {args.aug_copies} (heavy)…")
        real_parents = train_idx[np.asarray(real_train_local, dtype=np.int64)]
        for _ in range(args.aug_copies):
            xs = [augment_sequence_heavy(Xtr_base[j], rng) for j in real_train_local]
            aug_X.append(np.stack(xs, 0))
            aug_y.append(ytr_base[real_train_local])
            aug_a.append(atr_base[real_train_local])
            aug_t.append(t_all[torch.as_tensor(real_parents)])
            parent_global.append(real_parents)
    Xtr = np.concatenate(aug_X, 0)
    ytr = np.concatenate(aug_y, 0)
    atr = np.concatenate(aug_a, 0)
    ttr = torch.cat(aug_t, 0).contiguous()
    src_full = np.concatenate(
        [src_arr[train_mask]]
        + ([src_arr[train_mask][real_train_local]] * args.aug_copies if args.aug_copies and real_train_local else [])
    )
    spl_full = np.concatenate(
        [split_arr[train_mask]]
        + ([split_arr[train_mask][real_train_local]] * args.aug_copies if args.aug_copies and real_train_local else [])
    )
    assert len(Xtr) == len(ttr) == len(src_full)

    Xva, yva = Xn[val_mask], y[val_mask]
    Xte, yte = Xn[test_mask], y[test_mask]
    yva_t = torch.from_numpy(yva)
    yte_t = torch.from_numpy(yte)

    counts = np.bincount(ytr, minlength=len(labels)).astype(np.float64).clip(1.0, None)
    class_w = 1.0 / np.sqrt(counts)
    class_w = class_w / class_w.sum() * len(labels)
    sample_w = class_w[ytr].astype(np.float64)
    boost = np.ones(len(sample_w), dtype=np.float64)
    boost[spl_full != "synth"] *= args.real_boost
    boost[spl_full == "synth"] *= args.synth_boost
    boost[src_full == "wlasl100"] *= args.wlasl100_boost
    boost[src_full == "wlasl300"] *= args.wlasl300_boost
    w100_label_ids = set(int(i) for i in y[src_arr == "wlasl100"].tolist())
    if args.citizen_overlap_boost != 1.0:
        cit_ov = np.array(
            [(s.startswith("aslcitizen") and int(yi) in w100_label_ids) for s, yi in zip(src_full, ytr)],
            dtype=bool,
        )
        boost[cit_ov] *= args.citizen_overlap_boost
        print(f"citizen-overlap rows boosted: {int(cit_ov.sum())}")

    if args.hard_mine_from and args.hard_mine_from.exists():
        rep = json.loads(args.hard_mine_from.read_text())
        label_to_id = {g: i for i, g in enumerate(labels)}
        hard_ids = set()
        for item in (rep.get("top_confusions") or [])[:50]:
            if isinstance(item, (list, tuple)) and item:
                pair = str(item[0])
                if "->" in pair:
                    a, b = pair.split("->", 1)
                    if a in label_to_id:
                        hard_ids.add(label_to_id[a])
                    if b in label_to_id:
                        hard_ids.add(label_to_id[b])
        if hard_ids:
            hard_m = np.isin(ytr, list(hard_ids))
            boost[hard_m] *= args.hard_class_boost
            print(f"hard-mine boost x{args.hard_class_boost}: rows={int(hard_m.sum())} classes={len(hard_ids)}")

    boost_glosses = [g.strip().upper() for g in args.boost_glosses.split(",") if g.strip()]
    if boost_glosses:
        gids = {i for i, g in enumerate(labels) if g in boost_glosses}
        gm = np.isin(ytr, list(gids))
        boost[gm] *= args.gloss_boost
        print(f"gloss-boost x{args.gloss_boost}: rows={int(gm.sum())} glosses={boost_glosses}")

    sample_w = sample_w * boost
    print(f"sample boosts: mean_w={float(sample_w.mean()):.3f} n_train={len(Xtr)}")

    device = torch.device("cpu")
    model = build_sequence_model(
        "tcn-bilstm",
        input_dim=FEATURE_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=len(labels),
        bidirectional=True,
        dropout=args.dropout,
    ).to(device)

    if args.init_from and args.init_from.exists():
        ik = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ik["state_dict"], strict=False)
        print(f"warm-start from {args.init_from}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    crit = nn.CrossEntropyLoss(weight=torch.as_tensor(class_w, dtype=torch.float32), label_smoothing=args.label_smoothing)
    bce = nn.BCEWithLogitsLoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    # Dataset includes teacher soft logits + index for mixup teacher mix
    ds = TensorDataset(
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr),
        torch.from_numpy(atr),
        ttr,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True,
        ),
    )

    def lr_at(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            return args.lr * epoch / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return args.lr * 0.05 + 0.5 * (args.lr - args.lr * 0.05) * (1.0 + math.cos(math.pi * progress))

    # Val-A select / val-B gate (deterministic split, same seed as ens_gate_eval)
    val_idx = np.where(w100_val)[0]
    _rng = np.random.default_rng(0)
    _perm = _rng.permutation(len(val_idx))
    _cut = len(_perm) // 2
    a_idx = val_idx[_perm[:_cut]]
    b_idx = val_idx[_perm[_cut:]]
    print(f"val-A/B split: A={len(a_idx)} B={len(b_idx)}")

    def score_ab(m):
        with torch.no_grad():
            la = forward_logits(m, Xn[a_idx])
            lb = forward_logits(m, Xn[b_idx])
            sa = float((la.argmax(-1) == torch.from_numpy(y[a_idx])).float().mean())
            sb = float((lb.argmax(-1) == torch.from_numpy(y[b_idx])).float().mean())
        return sa, sb

    best_acc = -1.0  # val-A
    best_b = -1.0
    best_state = None
    history = []
    swa_state = None
    swa_n = 0

    if args.init_from and args.init_from.exists():
        model.eval()
        sa, sb = score_ab(model)
        best_acc, best_b = sa, sb
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"warm-start baseline valA={sa:.3f} valB={sb:.3f}")

    for epoch in range(1, args.epochs + 1):
        lr_now = lr_at(epoch)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb, ab, tb in loader:
            opt.zero_grad()
            if args.mixup > 0:
                perm = torch.randperm(xb.size(0))
                xb2, yb2, ab2, tb2 = xb[perm], yb[perm], ab[perm], tb[perm]
                lam = float(torch.distributions.Beta(args.mixup, args.mixup).sample())
                lam = max(lam, 1.0 - lam)
                xb = lam * xb + (1.0 - lam) * xb2
                ab = lam * ab + (1.0 - lam) * ab2
                tb_use = lam * tb + (1.0 - lam) * tb2
                logits, aux_logits = model(xb, return_aux=True)
                loss_ce = lam * crit(logits, yb) + (1.0 - lam) * crit(logits, yb2)
            else:
                tb_use = tb
                logits, aux_logits = model(xb, return_aux=True)
                loss_ce = crit(logits, yb)
            loss = loss_ce + args.aux_weight * bce(aux_logits, ab)
            T = args.distill_temp
            loss_kd = kl(torch.log_softmax(logits / T, dim=-1), torch.softmax(tb_use / T, dim=-1)) * (T * T)
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
        history.append({"epoch": epoch, "loss": total_loss / max(n, 1), "val_top1": acc, "test_top1": te_acc})
        sa, sb = score_ab(model)
        print(
            f"epoch {epoch:02d} loss={total_loss/max(n,1):.4f} val@1={acc:.3f} test@1={te_acc:.3f} "
            f"valA={sa:.3f} valB={sb:.3f}"
        )
        # Accept if A+B improves and neither slice drops >1.5pp vs its best
        if (sa + sb >= best_acc + best_b - 1e-9) and (sa + 0.015 >= best_acc) and (sb + 0.015 >= best_b):
            best_acc, best_b = sa, sb
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  * new best (A/B sum gate) A={sa:.3f} B={sb:.3f}")

        if epoch >= args.swa_start:
            cur = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = cur
                swa_n = 1
            else:
                for k in swa_state:
                    swa_state[k] = (swa_state[k] * swa_n + cur[k]) / (swa_n + 1)
                swa_n += 1

    assert best_state is not None
    if swa_state is not None and swa_n > 0:
        model.load_state_dict(swa_state)
        sa, sb = score_ab(model)
        print(f"SWA n={swa_n} valA={sa:.3f} valB={sb:.3f} (best A={best_acc:.3f} B={best_b:.3f})")
        if (sa + sb >= best_acc + best_b - 1e-9) and (sa + 0.015 >= best_acc) and (sb + 0.015 >= best_b):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_acc, best_b = sa, sb
            print("using SWA weights")
        else:
            print("keeping best checkpoint")

    model.load_state_dict(best_state)

    if args.finetune_wlasl100_epochs > 0:
        focus = np.isin(src_full, ["wlasl100", "wlasl300"])
        cit_ov = np.array(
            [(s.startswith("aslcitizen") and int(yi) in w100_label_ids) for s, yi in zip(src_full, ytr)],
            dtype=bool,
        )
        focus = focus | cit_ov
        ft_w = sample_w.copy()
        ft_w = np.where(focus, ft_w, 0.0)
        ft_w = np.where(src_full == "wlasl100", ft_w * 2.0, ft_w)
        ft_w = np.where(cit_ov, ft_w * 0.85, ft_w)
        print(f"=== fine-tune WLASL100 epochs={args.finetune_wlasl100_epochs} n={int(focus.sum())} ===")
        ft_loader = DataLoader(
            ds,
            batch_size=args.batch,
            sampler=WeightedRandomSampler(
                weights=torch.as_tensor(ft_w, dtype=torch.double),
                num_samples=max(int(focus.sum()), args.batch),
                replacement=True,
            ),
        )
        for pg in opt.param_groups:
            pg["lr"] = args.finetune_lr
        ft_alpha = min(args.distill_alpha, 0.35)
        for ft_ep in range(1, args.finetune_wlasl100_epochs + 1):
            model.train()
            for xb, yb, ab, tb in ft_loader:
                opt.zero_grad()
                if args.mixup > 0:
                    perm = torch.randperm(xb.size(0))
                    xb2, yb2, ab2, tb2 = xb[perm], yb[perm], ab[perm], tb[perm]
                    lam = float(torch.distributions.Beta(args.mixup, args.mixup).sample())
                    lam = max(lam, 1.0 - lam)
                    xb_m = lam * xb + (1.0 - lam) * xb2
                    ab_m = lam * ab + (1.0 - lam) * ab2
                    tb_use = lam * tb + (1.0 - lam) * tb2
                    logits, aux_logits = model(xb_m, return_aux=True)
                    loss_ce = lam * crit(logits, yb) + (1.0 - lam) * crit(logits, yb2)
                    loss = loss_ce + args.aux_weight * bce(aux_logits, ab_m)
                else:
                    tb_use = tb
                    logits, aux_logits = model(xb, return_aux=True)
                    loss = crit(logits, yb) + args.aux_weight * bce(aux_logits, ab)
                T = args.distill_temp
                loss_kd = kl(torch.log_softmax(logits / T, dim=-1), torch.softmax(tb_use / T, dim=-1)) * (T * T)
                loss = ft_alpha * loss_kd + (1.0 - ft_alpha) * loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            model.eval()
            sa, sb = score_ab(model)
            print(f"  finetune {ft_ep:02d} valA={sa:.3f} valB={sb:.3f}")
            if (sa + sb >= best_acc + best_b - 1e-9) and (sa + 0.015 >= best_acc) and (sb + 0.015 >= best_b):
                best_acc, best_b = sa, sb
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"  * new best (A/B sum gate) A={sa:.3f} B={sb:.3f}")
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        w_va = forward_logits(model, Xn[w100_val])
        w_te = forward_logits(model, Xn[w100_test])
        va1 = float(topk_acc(w_va, torch.from_numpy(y[w100_val]), 1))
        va5 = float(topk_acc(w_va, torch.from_numpy(y[w100_val]), 5))
        te1 = float(topk_acc(w_te, torch.from_numpy(y[w100_test]), 1))
        te5 = float(topk_acc(w_te, torch.from_numpy(y[w100_test]), 5))
        te_logits_full = forward_logits(model, Xte)
        pred = te_logits_full.argmax(-1).numpy()
        # confusions
        from collections import Counter
        conf = Counter()
        for yt, yp in zip(yte, pred):
            if yt != yp:
                conf[f"{labels[int(yt)]}->{labels[int(yp)]}"] += 1

    ckpt = {
        "state_dict": best_state,
        "num_classes": len(labels),
        "input_dim": FEATURE_DIM,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "labels": labels,
        "val_acc": float((forward_logits(model, Xva).argmax(-1) == yva_t).float().mean()),
        "test_acc": float((forward_logits(model, Xte).argmax(-1) == yte_t).float().mean()),
        "backend": "pytorch",
        "arch": "tcn-bilstm",
        "trained_on": "+".join(sorted(set(sources))),
        "note": "cached ens5 KD single student; val-selected on WLASL100",
        "top_confusions": conf.most_common(40),
        "wlasl100_subset": {
            "val_top1": va1,
            "val_top5": va5,
            "n_val": int(w100_val.sum()),
            "test_top1": te1,
            "test_top5": te5,
            "n_test": int(w100_test.sum()),
        },
        "window": int(X.shape[1]),
    }
    torch.save(ckpt, args.out)
    report = {
        "val_top1": ckpt["val_acc"],
        "test_top1": ckpt["test_acc"],
        "n_classes": len(labels),
        "n_train": len(Xtr),
        "arch": "tcn-bilstm-nmm",
        "wlasl100_subset": ckpt["wlasl100_subset"],
        "comparable_wlasl100": {
            "val_top1": va1,
            "val_top5": va5,
            "test_top1": te1,
            "test_top5": te5,
            "n_val": int(w100_val.sum()),
            "n_test": int(w100_test.sum()),
        },
        "previous_ship": {"test_top1": 0.4806201457977295, "commit": "a7dbaed"},
        "delta_vs_ship_wlasl100_test_pp": (te1 - 0.4806201457977295) * 100.0,
        "wlasl100_boost": args.wlasl100_boost,
        "finetune_wlasl100_epochs": args.finetune_wlasl100_epochs,
        "distill": True,
        "teacher": "cached ens5 ship weights",
        "seed": args.seed,
        "hidden": args.hidden,
        "history_tail": history[-5:],
        "privacy": "offline public pose train; on-device Core ML inference; preferServer=false",
        "top_confusions": conf.most_common(15),
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(
        f"saved {args.out} comparable WLASL100 val@1={va1:.3f} test@1={te1:.3f} "
        f"(ship 0.481 delta_pp={report['delta_vs_ship_wlasl100_test_pp']:.2f})"
    )
    try:
        export_coreml(model, labels, FEATURE_DIM, args.coreml, int(X.shape[1]), "tcn-bilstm")
    except Exception as e:
        print(f"coreml export skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
