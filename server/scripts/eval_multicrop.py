#!/usr/bin/env python3
"""Comparable WLASL100 eval with val-gated temporal multi-crop (plain argmax).

Native HDF5 clips are often >>32 frames; prior ship used last-32 only.
Val selects crop count; default nc=5, max_context=96 (matches on-device).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.coco135 import coco135_sequence_to_features, pad_or_trim  # noqa: E402
from pipeline.gloss_map import canonicalize_gloss  # noqa: E402
from pipeline.normalize import FEATURE_DIM  # noqa: E402
from pipeline.sequence_model import build_sequence_model  # noqa: E402
from scripts.train_ondevice_coreml import normalize_matrix, topk_acc  # noqa: E402


def load_wlasl100(split_cap: str, labels: list[str]) -> tuple[list[np.ndarray], np.ndarray]:
    lab_to_i = {g: i for i, g in enumerate(labels)}
    path = ROOT / "data" / "wlasl100" / f"WLASL100_135-{split_cap}.hdf5"
    xs: list[np.ndarray] = []
    ys: list[int] = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys(), key=lambda k: int(k) if k.isdigit() else k):
            g = f[key]
            raw = g["data"][:].astype(np.float32)
            lab = g["label"][()]
            if isinstance(lab, bytes):
                lab = lab.decode("utf-8")
            gloss = canonicalize_gloss(str(lab).strip(), allowed=set(labels))
            if gloss not in lab_to_i or raw.shape[0] < 6:
                continue
            xs.append(coco135_sequence_to_features(raw))
            ys.append(lab_to_i[gloss])
    return xs, np.asarray(ys, dtype=np.int64)


def temporal_crops(feat: np.ndarray, n_crops: int, max_t: int, window: int = 32) -> list[np.ndarray]:
    if feat.shape[0] > max_t:
        feat = feat[-max_t:]
    t = feat.shape[0]
    if t <= window:
        return [pad_or_trim(feat, window)]
    max_start = t - window
    if n_crops <= 1:
        return [feat[-window:]]
    starts = sorted({int(round(i * max_start / (n_crops - 1))) for i in range(n_crops)})
    return [feat[s : s + window] for s in starts]


@torch.no_grad()
def score(model, xs, ys, n_crops: int, max_t: int):
    logits = []
    for feat in xs:
        crops = temporal_crops(feat, n_crops, max_t)
        xn = np.stack([normalize_matrix(c.astype(np.float32)) for c in crops], 0)
        logits.append(model(torch.as_tensor(xn)).mean(0))
    lt = torch.stack(logits)
    yt = torch.as_tensor(ys)
    return {
        "top1": float(topk_acc(lt, yt, 1)),
        "top5": float(topk_acc(lt, yt, 5)),
        "correct": int((lt.argmax(-1) == yt).sum()),
        "n": int(len(ys)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--crops", type=int, default=5)
    ap.add_argument("--max-context", type=int, default=96)
    ap.add_argument("--ship", type=float, default=0.4844961166381836)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    labels = list(ck["labels"])
    arch = ck.get("arch") or "tcn-bilstm"
    if isinstance(arch, str) and arch.endswith("-nmm"):
        arch = arch[: -len("-nmm")]
    model = build_sequence_model(
        arch,
        input_dim=int(ck.get("input_dim", FEATURE_DIM)),
        hidden_dim=int(ck.get("hidden_dim", 192)),
        num_layers=int(ck.get("num_layers", 2)),
        num_classes=len(labels),
        bidirectional=True,
        dropout=0.0,
    )
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()

    va_x, va_y = load_wlasl100("Val", labels)
    te_x, te_y = load_wlasl100("Test", labels)

    # Val-gate crop count around the default
    best = None
    for nc in sorted({1, 3, args.crops, 5, 7}):
        va = score(model, va_x, va_y, nc, args.max_context)
        te = score(model, te_x, te_y, nc, args.max_context)
        row = {"crops": nc, "val": va, "test": te, "dpp": (te["top1"] - args.ship) * 100}
        print(json.dumps(row))
        if best is None or va["top1"] > best["val"]["top1"]:
            best = row

    print("VAL_GATED", json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
