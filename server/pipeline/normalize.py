"""Landmark normalization for sequence models.

Feature layout v2 (FEATURE_DIM=170):
  left hand 42 + right hand 42 + body 34 + face 40 + activity 1 + NMM 11
Protocol v1 was 139 (face 20, no NMM). Additive face joints + NMM channels.
"""

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

# First 10 = protocol v1; remaining additive for NMM proxies.
FACE_JOINTS = [
    "leftEye", "rightEye", "nose", "mouthLeft", "mouthRight",
    "leftEyebrowOuter", "rightEyebrowOuter",
    "chin", "forehead", "mouthCenter",
    "leftEyebrowInner", "rightEyebrowInner",
    "leftEyeTop", "leftEyeBottom", "rightEyeTop", "rightEyeBottom",
    "outerLipTop", "outerLipBottom", "innerLipTop", "innerLipBottom",
]

NMM_CHANNELS = [
    "browRaise", "browFurrow", "eyeWiden", "squint",
    "mouthOpen", "smile", "frown",
    "headShake", "headNod",
    "torsoLean", "shoulderTilt",
]

FEATURE_LAYOUT_VERSION = 2
# left 42 + right 42 + body 34 + face 40 + activity 1 + nmm 11 = 170
FEATURE_DIM = 21 * 2 * 2 + 17 * 2 + 20 * 2 + 1 + len(NMM_CHANNELS)

# Slice indices
FACE_START = 84 + 34  # 118
FACE_END = FACE_START + len(FACE_JOINTS) * 2  # 158
ACTIVITY_IDX = FACE_END  # 158
NMM_START = ACTIVITY_IDX + 1  # 159


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


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def derive_nmm_from_frame(frame: dict[str, Any]) -> np.ndarray:
    """Best-effort NMM channels from face/body geometry (server-side)."""
    out = np.zeros(len(NMM_CHANNELS), dtype=np.float32)
    face = {j.get("name"): j for j in (frame.get("face") or []) if j.get("name")}
    body = {j.get("name"): j for j in (frame.get("body") or []) if j.get("name")}

    def fy(name: str) -> float | None:
        j = face.get(name)
        return float(j["y"]) if j and "y" in j else None

    def fx(name: str) -> float | None:
        j = face.get(name)
        return float(j["x"]) if j and "x" in j else None

    # Brows
    brow_raise = 0.0
    brow_furrow = 0.0
    for brow, eye in (("leftEyebrowOuter", "leftEye"), ("rightEyebrowOuter", "rightEye"),
                      ("leftEyebrowInner", "leftEye"), ("rightEyebrowInner", "rightEye")):
        by, ey = fy(brow), fy(eye)
        if by is not None and ey is not None:
            brow_raise = max(brow_raise, _clamp01((by - ey - 0.012) / 0.045))
            brow_furrow = max(brow_furrow, _clamp01((ey - by - 0.004) / 0.03))
    out[0] = brow_raise
    out[1] = brow_furrow

    # Eyes
    def aperture(top: str, bottom: str) -> float:
        t, b = fy(top), fy(bottom)
        if t is None or b is None:
            return 0.022
        return abs(t - b)

    eye_open = max(aperture("leftEyeTop", "leftEyeBottom"), aperture("rightEyeTop", "rightEyeBottom"))
    out[2] = _clamp01((eye_open - 0.028) / 0.04)
    out[3] = _clamp01((0.018 - eye_open) / 0.015)

    # Mouth
    top = fy("outerLipTop") or fy("innerLipTop")
    bot = fy("outerLipBottom") or fy("innerLipBottom")
    if top is not None and bot is not None:
        out[4] = _clamp01((abs(top - bot) - 0.02) / 0.06)
    ml, mr, mc = fx("mouthLeft"), fx("mouthRight"), fy("mouthCenter")
    mly, mry = fy("mouthLeft"), fy("mouthRight")
    if None not in (ml, mr, mc, mly, mry):
        width = abs(mr - ml)
        lift = mc - min(mly, mry)
        out[5] = _clamp01((width - 0.06) / 0.08) * _clamp01((lift + 0.01) / 0.03)
        drop = min(mly, mry) - mc
        out[6] = _clamp01((drop - 0.005) / 0.025) * (1.0 - out[5])

    # Torso lean / shoulder tilt from body
    ls, rs = body.get("leftShoulder"), body.get("rightShoulder")
    lh, rh = body.get("leftHip"), body.get("rightHip")
    if ls and rs and lh and rh:
        sw = max(abs(float(rs["x"]) - float(ls["x"])), 0.05)
        lean = (0.5 * (float(ls["x"]) + float(rs["x"])) - 0.5 * (float(lh["x"]) + float(rh["x"]))) / sw
        out[9] = float(max(-1.0, min(1.0, lean)))
        dy = abs(float(ls["y"]) - float(rs["y"]))
        dx = max(abs(float(rs["x"]) - float(ls["x"])), 0.05)
        out[10] = _clamp01(abs(np.arctan2(dy, dx)) / 0.45)

    # headShake / headNod need temporal context — leave 0 here; client may send them.
    return out


def nmm_summary_from_frames(frames: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate NMM channels over a window for gloss→English conditioning."""
    if not frames:
        return {k: 0.0 for k in NMM_CHANNELS}
    mats = []
    for f in frames:
        raw = f.get("nmm")
        if isinstance(raw, list) and len(raw) >= len(NMM_CHANNELS):
            mats.append(np.array(raw[: len(NMM_CHANNELS)], dtype=np.float32))
        else:
            mats.append(derive_nmm_from_frame(f))
    arr = np.stack(mats, axis=0)
    # Use high percentiles so brief brow raises still register
    summary = {}
    for i, name in enumerate(NMM_CHANNELS):
        col = arr[:, i]
        if name == "torsoLean":
            summary[name] = float(np.median(col))
        else:
            summary[name] = float(np.percentile(np.abs(col) if name == "torsoLean" else col, 75))
    summary["confidence"] = float(np.mean(np.any(arr > 0.05, axis=1)))
    return summary


def frame_to_vector(frame: dict[str, Any]) -> np.ndarray:
    hands = frame.get("hands") or []
    left = next((h for h in hands if h.get("chirality") == "left"), None)
    right = next((h for h in hands if h.get("chirality") == "right"), None)
    if left is None and hands:
        left = hands[0]
    if right is None and len(hands) > 1:
        right = hands[1]

    nmm_raw = frame.get("nmm")
    if isinstance(nmm_raw, list) and len(nmm_raw) >= len(NMM_CHANNELS):
        nmm = np.array([float(x) for x in nmm_raw[: len(NMM_CHANNELS)]], dtype=np.float32)
    else:
        nmm = derive_nmm_from_frame(frame)

    parts = [
        _pack_hand(left),
        _pack_hand(right),
        _pack_named(frame.get("body") or [], BODY_JOINTS),
        _pack_named(frame.get("face") or [], FACE_JOINTS),
        np.array([float(frame.get("activity") or 0.0)], dtype=np.float32),
        nmm,
    ]
    vec = np.concatenate(parts, axis=0)
    assert vec.shape[0] == FEATURE_DIM, f"got {vec.shape[0]} want {FEATURE_DIM}"
    return vec


def normalize_frames(frames: list[dict[str, Any]]) -> np.ndarray:
    """Return (T, D) array centered on torso/hand midpoints, scaled by shoulder width.

    NMM channels and activity are not spatially normalized (already 0…1 / signed lean).
    """
    if not frames:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    mat = np.stack([frame_to_vector(f) for f in frames], axis=0)

    # Spatial part: everything before activity (exclude activity + NMM)
    spatial_end = ACTIVITY_IDX
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

    xy = mat[:, :spatial_end].reshape(mat.shape[0], -1, 2)
    xy = (xy - origin[:, None, :]) / scale[:, None, :]
    spatial = xy.reshape(mat.shape[0], -1)
    tail = mat[:, spatial_end:]  # activity + NMM
    mat = np.concatenate([spatial, tail], axis=1)
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
