#!/usr/bin/env python3
"""Val-AB gate weighted ens2 (+ optional bias) under multicrop; ship if test > prior."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from pipeline.normalize import FEATURE_DIM  # noqa: E402
from pipeline.sequence_model import (  # noqa: E402
    LogitsPlusBias,
    WeightedLogitEnsemble,
    build_sequence_model,
)
from scripts.eval_multicrop import load_wlasl100  # noqa: E402
from scripts.train_multicrop_matched_kd import multicrop_logits, val_ab_indices  # noqa: E402
from scripts.train_ondevice_coreml import export_coreml, topk_acc  # noqa: E402


def load_student(path: Path, labels: list[str]):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_sequence_model(
        "tcn-bilstm",
        input_dim=FEATURE_DIM,
        hidden_dim=int(ck.get("hidden_dim", 192)),
        num_layers=int(ck.get("num_layers", 2)),
        num_classes=len(labels),
        bidirectional=True,
        dropout=0.0,
    )
    sd = ck["state_dict"]
    if any(k.startswith("base.models.0.") for k in sd):
        sd = {k[len("base.models.0.") :]: v for k, v in sd.items() if k.startswith("base.models.0.")}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", type=Path, required=True)
    ap.add_argument("--m2", type=Path, required=True)
    ap.add_argument("--ship", type=float, default=0.5930232405662537)
    ap.add_argument("--crops", type=int, default=5)
    ap.add_argument("--max-context", type=int, default=96)
    ap.add_argument("--bake-bias", action="store_true")
    ap.add_argument("--force-ship", action="store_true")
    ap.add_argument("--tag", default="multicrop-matched ens2")
    args = ap.parse_args()

    ck1 = torch.load(args.m1, map_location="cpu", weights_only=False)
    labels = list(ck1["labels"])
    m1, _ = load_student(args.m1, labels)
    m2, _ = load_student(args.m2, labels)

    va_x, va_y = load_wlasl100("Val", labels)
    te_x, te_y = load_wlasl100("Test", labels)
    a_idx, b_idx = val_ab_indices(len(va_y), 0)
    yva = torch.as_tensor(va_y)
    yte = torch.as_tensor(te_y)

    L1va = multicrop_logits(m1, va_x, args.crops, args.max_context)
    L2va = multicrop_logits(m2, va_x, args.crops, args.max_context)
    L1te = multicrop_logits(m1, te_x, args.crops, args.max_context)
    L2te = multicrop_logits(m2, te_x, args.crops, args.max_context)

    best = None
    for a in np.linspace(0.35, 0.65, 31):
        Lva = a * L1va + (1 - a) * L2va
        sa = float((Lva[a_idx].argmax(-1) == yva[a_idx]).float().mean())
        sb = float((Lva[b_idx].argmax(-1) == yva[b_idx]).float().mean())
        row = {"alpha_m1": float(a), "val_A": sa, "val_B": sb, "val_sum": sa + sb}
        if best is None or row["val_sum"] > best["val_sum"] + 1e-12:
            best = row
            best["Lva"] = Lva
            best["Lte"] = a * L1te + (1 - a) * L2te
    print(f"mix gate alpha_m1={best['alpha_m1']:.3f} A={best['val_A']:.4f} B={best['val_B']:.4f}")

    Lva, Lte = best["Lva"], best["Lte"]
    La, Lb = Lva[a_idx], Lva[b_idx]
    ya, yb = yva[a_idx], yva[b_idx]
    base_a, base_b = best["val_A"], best["val_B"]
    best_bias = None
    best_ab = (base_a, base_b)

    if args.bake_bias:
        bias = torch.zeros(len(labels), requires_grad=True)
        opt = torch.optim.Adam([bias], lr=0.04)
        best_sum = base_a + base_b
        for step in range(800):
            opt.zero_grad()
            F.cross_entropy(La + bias, ya).backward()
            opt.step()
            with torch.no_grad():
                bias.clamp_(-1.25, 1.25)
                sa = float(((La + bias).argmax(-1) == ya).float().mean())
                sb = float(((Lb + bias).argmax(-1) == yb).float().mean())
                if sa + sb > best_sum + 1e-9 and sa + 0.012 >= base_a and sb + 0.012 >= base_b:
                    best_sum = sa + sb
                    best_bias = bias.detach().clone()
                    best_ab = (sa, sb)
        if best_bias is not None:
            Lva = Lva + best_bias
            Lte = Lte + best_bias
            print(f"bias selected A={best_ab[0]:.4f} B={best_ab[1]:.4f} l2={float(best_bias.norm()):.3f}")
        else:
            print("bias: no improvement under gate; skip")

    va1 = float(topk_acc(Lva, yva, 1))
    va5 = float(topk_acc(Lva, yva, 5))
    te1 = float(topk_acc(Lte, yte, 1))
    te5 = float(topk_acc(Lte, yte, 5))
    corr = int((Lte.argmax(-1) == yte).sum())
    dpp = (te1 - args.ship) * 100
    print(f"SELECTED val={va1:.4f}/{va5:.4f} te={te1:.4f}/{te5:.4f} ({corr}/258) dpp={dpp:+.3f}")

    report = {
        "alpha_m1": best["alpha_m1"],
        "val_A": best_ab[0],
        "val_B": best_ab[1],
        "val_top1": va1,
        "val_top5": va5,
        "test_top1": te1,
        "test_top5": te5,
        "correct": corr,
        "n_test": 258,
        "delta_pp": dpp,
        "ship": args.ship,
        "m1": str(args.m1),
        "m2": str(args.m2),
        "bias": best_bias is not None,
    }
    (ROOT / "models" / "ens2_multicrop_matched_report.json").write_text(json.dumps(report, indent=2))

    if te1 <= args.ship + 1e-12 and not args.force_ship:
        print("NO SHIP — comparable test did not beat prior")
        return 0

    members = []
    for path in (args.m1, args.m2):
        m, _ = load_student(path, labels)
        members.append(m)
    w = [best["alpha_m1"], 1.0 - best["alpha_m1"]]
    ens = WeightedLogitEnsemble(members, w)
    model = LogitsPlusBias(ens, best_bias) if best_bias is not None else ens
    model.eval()

    out_pt = ROOT / "models" / "sign_classifier.pt"
    arch = "tcn-bilstm-weighted-ens2-bias+multicrop5" if best_bias is not None else "tcn-bilstm-weighted-ens2+multicrop5"
    ckpt = {
        "state_dict": model.state_dict(),
        "num_classes": len(labels),
        "input_dim": FEATURE_DIM,
        "hidden_dim": 192,
        "num_layers": 2,
        "labels": labels,
        "arch": arch,
        "ensemble": True,
        "weighted": True,
        "n_members": 2,
        "member_weights": w,
        "members": [args.m1.name, args.m2.name],
        "class_bias": best_bias.tolist() if best_bias is not None else None,
        "window": 32,
        "note": f"{args.tag}; alpha={w[0]:.3f}; multicrop nc={args.crops} max{args.max_context}",
        "wlasl100_subset": {
            "val_top1": va1,
            "val_top5": va5,
            "n_val": len(va_y),
            "test_top1": te1,
            "test_top5": te5,
            "n_test": 258,
            "method": args.tag,
            "multi_crop_count": args.crops,
            "max_context_frames": args.max_context,
            "mix_alpha_m1": w[0],
        },
        "comparable_wlasl100": {
            "val_top1": va1,
            "val_top5": va5,
            "test_top1": te1,
            "test_top5": te5,
            "n_val": len(va_y),
            "n_test": 258,
            "method": "val-AB ens2+bias multicrop plain",
        },
        "inference": {"multi_crop": args.crops, "max_context": args.max_context, "crop_window": 32},
        "previous_ship_test": args.ship,
    }
    torch.save(ckpt, out_pt)

    eval_report = {
        "val_top1": None,
        "test_top1": None,
        "n_classes": len(labels),
        "arch": arch,
        "wlasl100_subset": ckpt["wlasl100_subset"],
        "comparable_wlasl100": ckpt["comparable_wlasl100"],
        "previous_ship": {"test_top1": args.ship, "commit": "7a0f14b"},
        "delta_vs_7a0f14b_wlasl100_test_pp": dpp,
        "privacy": "offline public pose train; on-device Core ML ens2 + multi-crop; preferServer=false",
        "notes": ckpt["note"],
        "seed": f"{args.m1.stem}+{args.m2.stem}",
        "ensemble_retired": False,
        "ensemble_n": 2,
        "inference": ckpt["inference"],
    }
    (ROOT / "models" / "eval_report.json").write_text(json.dumps(eval_report, indent=2))

    pkg = REPO / "ASLSubtitles" / "Models" / "ASLSignClassifier.mlpackage"
    export_coreml(model, labels, FEATURE_DIM, pkg, 32, "tcn-bilstm-ens2")
    labels_json = REPO / "ASLSubtitles" / "Models" / "ASLSignClassifier.labels.json"
    labels_json.write_text(json.dumps({"labels": labels}, indent=2) + "\n")
    print(f"SHIPPED te={te1:.4f} ({corr}/258) -> {out_pt} + {pkg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
