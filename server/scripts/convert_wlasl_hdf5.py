#!/usr/bin/env python3
"""Convert WholeBodyPose WLASL100 HDF5 (COCO-135) → NPZ in our FEATURE_DIM layout.

Source: https://huggingface.co/datasets/CristianLazoQuispe/pose-action-recognition
License: MIT for the pose packaging; underlying WLASL video rights remain with
WLASL (C-UDA / research). We redistribute only derived landmarks we convert locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.coco135 import coco135_sequence_to_features, pad_or_trim  # noqa: E402
from pipeline.vocab import GLOSS_VOCAB  # noqa: E402


def load_split(path: Path, frames: int) -> tuple[list[np.ndarray], list[str]]:
    xs: list[np.ndarray] = []
    labels: list[str] = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys(), key=lambda k: int(k) if k.isdigit() else k):
            g = f[key]
            raw = g["data"][:]  # (T, 2, 135)
            lab = g["label"][()]
            if isinstance(lab, bytes):
                lab = lab.decode("utf-8")
            gloss = str(lab).strip().upper().replace(" ", "-")
            if raw.shape[0] < 6:
                continue
            feat = coco135_sequence_to_features(raw.astype(np.float32))
            feat = pad_or_trim(feat, frames)
            xs.append(feat)
            labels.append(gloss)
    return xs, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "wlasl100")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "wlasl100_features.npz")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--mix-synth", action="store_true", help="Also emit glosses from GLOSS_VOCAB missing in WLASL via synth")
    ap.add_argument("--synth-per-class", type=int, default=24)
    args = ap.parse_args()

    splits = {
        "train": args.data_dir / "WLASL100_135-Train.hdf5",
        "val": args.data_dir / "WLASL100_135-Val.hdf5",
        "test": args.data_dir / "WLASL100_135-Test.hdf5",
    }
    for name, p in splits.items():
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    all_X: list[np.ndarray] = []
    all_y_gloss: list[str] = []
    split_ids: list[str] = []

    for split, path in splits.items():
        xs, labs = load_split(path, args.frames)
        print(f"{split}: {len(xs)} sequences from {path.name}")
        all_X.extend(xs)
        all_y_gloss.extend(labs)
        split_ids.extend([split] * len(xs))

    wlasl_glosses = sorted(set(all_y_gloss))
    # Union with conversational vocab so HELLO etc. remain learnable via synth fill.
    union = list(dict.fromkeys(list(GLOSS_VOCAB) + wlasl_glosses))

    if args.mix_synth:
        from scripts.synthesize_pose_dataset import synthesize_sequence

        rng = np.random.default_rng(11)
        missing = [g for g in GLOSS_VOCAB if g not in set(wlasl_glosses)]
        print(f"synth-fill {len(missing)} conversational glosses × {args.synth_per_class}")
        for g in missing:
            for _ in range(args.synth_per_class):
                T = int(rng.integers(24, 40))
                seq = synthesize_sequence(g, T, rng)
                all_X.append(pad_or_trim(seq, args.frames))
                all_y_gloss.append(g)
                split_ids.append("synth")

    gloss_to_id = {g: i for i, g in enumerate(union)}
    y = np.array([gloss_to_id[g] for g in all_y_gloss], dtype=np.int64)
    X = np.stack(all_X, axis=0).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        labels=np.array(union),
        gloss=np.array(all_y_gloss),
        split=np.array(split_ids),
        source="wlasl100-coco135+optional-synth",
    )
    meta = {
        "n_samples": int(len(y)),
        "n_classes": len(union),
        "frames": args.frames,
        "feature_dim": int(X.shape[-1]),
        "wlasl_glosses": wlasl_glosses,
        "source": "CristianLazoQuispe/pose-action-recognition WLASL100 (COCO-135, MIT packaging)",
        "underlying_video": "WLASL — research / C-UDA; we store converted landmarks only",
    }
    (args.out.parent / "wlasl100_features.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out} X={X.shape} classes={len(union)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
