"""Landmark normalization for sequence models."""

from __future__ import annotations

from typing import Any

import numpy as np

HAND_JOINTS = [
    "wrist",
    "thumbCMC", "thumbMP", "thumbIP", "thumbTip",
    "indexMCP", "indexPIP", "indexDIP", "indexTip",
    "middleMCP", "middlePIP", "middleDIP", "middleTip",
    "ringMCP", "ringPIP", "ringDIP", "ringTip",
    "littleMCP", "littlePIP", "littleDIP", "littleTip",
]

BODY_JOINTS = [
    "nose", "neck",
    "rightShoulder", "rightElbow", "rightWrist",
    "leftShoulder", "leftElbow", "leftWrist",
    "rightHip", "rightKnee", "rightAnkle",
    "leftHip", "leftKnee", "leftAnkle",
    "root", "rightEar", "leftEar",
]

FACE_JOINTS = [
    "leftEye", "rightEye", "nose", "mouthLeft", "mouthRight",
    "leftEyebrowOuter", "rightEyebrowOuter",
    "chin", "forehead", "mouthCenter",
]

# left hand 42 + right hand 42 + body 34 + face 20 + activity 1 = 139
FEATURE_DIM = 21 * 2 * 2 + 17 * 2 + 10 * 2 + 1


def _pack_hand(hand: dict | None) -> np.ndarray:
    out = np.zeros(21 * 2, dtype=np.float32)
    if not hand:
        return out
    joints = hand.get("joints") or {}
    for i, name in enumerate(HAND_JOINTS):
        xy = joints.get(name)
        if xy and len(xy) >= 2:
            out[i * 2] = float(xy[0])
            out[i * 2 + 1] = float(xy[1])
    return out


def _pack_named(joints: list[dict], order: list[str]) -> np.ndarray:
    out = np.zeros(len(order) * 2, dtype=np.float32)
    by_name = {j.get("name"): j for j in joints or []}
    for i, name in enumerate(order):
        j = by_name.get(name)
        if j:
            out[i * 2] = float(j.get("x", 0))
            out[i * 2 + 1] = float(j.get("y", 0))
    return out


def frame_to_vector(frame: dict[str, Any]) -> np.ndarray:
    hands = frame.get("hands") or []
    left = next((h for h in hands if h.get("chirality") == "left"), None)
    right = next((h for h in hands if h.get("chirality") == "right"), None)
    if left is None and hands:
        left = hands[0]
    if right is None and len(hands) > 1:
        right = hands[1]

    parts = [
        _pack_hand(left),
        _pack_hand(right),
        _pack_named(frame.get("body") or [], BODY_JOINTS),
        _pack_named(frame.get("face") or [], FACE_JOINTS),
        np.array([float(frame.get("activity") or 0.0)], dtype=np.float32),
    ]
    return np.concatenate(parts, axis=0)


def normalize_frames(frames: list[dict[str, Any]]) -> np.ndarray:
    """Return (T, D) array centered on torso/hand midpoints, scaled by shoulder width."""
    if not frames:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    mat = np.stack([frame_to_vector(f) for f in frames], axis=0)

    # Indices: left hand 0:42, right 42:84, body 84:118
    # body neck=1 → offsets 84+2,84+3; shoulders at body idx 2/5
    body = mat[:, 84:118]
    neck = body[:, 2:4]  # neck x,y
    r_sh = body[:, 4:6]
    l_sh = body[:, 10:12]
    scale = np.linalg.norm(r_sh - l_sh, axis=1, keepdims=True)
    scale = np.clip(scale, 0.05, None)

    # Prefer neck as origin; fall back to mean of present hand wrists.
    origin = neck.copy()
    missing = (np.abs(origin).sum(axis=1) < 1e-6)
    if missing.any():
        lw = mat[:, 0:2]
        rw = mat[:, 42:44]
        origin[missing] = 0.5 * (lw[missing] + rw[missing])

    # Subtract origin from all xy pairs
    xy = mat[:, :-1].reshape(mat.shape[0], -1, 2)
    xy = (xy - origin[:, None, :]) / scale[:, None, :]
    mat = np.concatenate([xy.reshape(mat.shape[0], -1), mat[:, -1:]], axis=1)
    return mat.astype(np.float32)


def frames_to_tensor(frames: list[dict[str, Any]], window: int = 32):
    """Pad/truncate to fixed window; returns torch FloatTensor (1, T, D) when torch available."""
    arr = normalize_frames(frames)
    if arr.shape[0] == 0:
        arr = np.zeros((window, FEATURE_DIM), dtype=np.float32)
    elif arr.shape[0] < window:
        pad = np.zeros((window - arr.shape[0], FEATURE_DIM), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    else:
        arr = arr[-window:]

    try:
        import torch

        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return arr[None, ...]
