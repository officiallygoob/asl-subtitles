"""Uni-Sign checkpoint detection + honest status.

Uni-Sign (ICLR 2025) pose-only weights from Hugging Face `ZechengLi19/Uni-Sign`
use Spatial GCN pose encoders + temporal encoders + an LLM text head on 69
RTMPose keypoints. That stack does **not** match our PoseLSTM FEATURE_DIM=139
MediaPipe/Vision layout.

This module:
1. Detects downloaded `.pth` files in `server/models/`
2. Reports metadata on `/health`
3. Documents why we ship PoseLSTM (`sign_classifier.pt`) instead for runtime

CC-BY-NC-4.0 upstream — non-commercial only.
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
        "status": "present-architecture-mismatch",
        "license": "CC-BY-NC-4.0 (Uni-Sign upstream)",
        "runtime": "PoseLSTM sign_classifier.pt is the active inference path",
        "hint": (
            "Uni-Sign needs its native GCN+LLM code + RTMPose-69 layout. "
            "Our server uses MediaPipe/Vision 139-d features + PoseLSTM. "
            "Download with scripts/download_uni_sign.sh for research; "
            "do not expect auto-inference. See MODELS.md."
        ),
    }
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            info["top_keys"] = sorted(list(ckpt.keys()))[:40]
            info["type"] = "state_dict_dict"
            # Heuristic: Uni-Sign checkpoints often nest under model/module keys
            nested = []
            for k in list(ckpt.keys())[:20]:
                if isinstance(ckpt[k], dict):
                    nested.append(k)
            if nested:
                info["nested_dict_keys"] = nested[:10]
        else:
            info["type"] = type(ckpt).__name__
    except Exception as exc:
        info["inspect_error"] = str(exc)
    return info
