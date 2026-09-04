"""Uni-Sign checkpoint adapter (scaffold).

Uni-Sign (ICLR 2025) pose-only weights from Hugging Face `ZechengLi19/Uni-Sign`
use a research encoder/decoder stack that does **not** match our PoseLSTM
scaffold 1:1. This module:

1. Detects downloaded `.pth` files in `server/models/`
2. Reports metadata so `/health` can show they are present
3. Leaves a clear TODO for tensor-layout remapping + forward pass

Until the adapter is completed, `ContinuousDecoder` keeps using the demo
decoder (or a fine-tuned PoseLSTM `.pt` you export yourself).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("asl.uni_sign")

UNI_SIGN_CANDIDATES = (
    "wlasl_pose_only_islr.pth",
    "how2sign_pose_only_slt.pth",
    "openasl_pose_only_slt.pth",
    "csl_daily_pose_only_slt.pth",
    "uni_sign.pt",
)


def find_uni_sign_checkpoint(models_dir: Path) -> Path | None:
    for name in UNI_SIGN_CANDIDATES:
        path = models_dir / name
        if path.exists():
            return path
    return None


def describe_checkpoint(path: Path) -> dict:
    info = {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size if path.exists() else 0,
        "loadable_in_poselstm": False,
        "status": "present-needs-adapter",
        "license": "CC-BY-NC-4.0 (Uni-Sign upstream)",
        "hint": (
            "Run scripts/download_uni_sign.sh then implement forward() mapping "
            "Vision/MediaPipe landmarks → Uni-Sign pose format. See MODELS.md."
        ),
    }
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            info["top_keys"] = sorted(list(ckpt.keys()))[:40]
            info["type"] = "state_dict_dict"
        else:
            info["type"] = type(ckpt).__name__
    except Exception as exc:
        info["inspect_error"] = str(exc)
    return info
