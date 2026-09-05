"""COCO-WholeBody 133 (+2 virtual) → our FEATURE_DIM=170 landmark layout.

WholeBodyPose / CristianLazoQuispe HDF5 pose dumps use:
  0–16   body (COCO-17)
  17–22  feet
  23–90  face (68)
  91–111 left hand (21)
  112–132 right hand (21)
  133–134 chest, mid-hip (virtual)  → 135 total

Coordinates are image-normalized [0,1] as (T, 2, 135) in the HDF5 files.
"""

from __future__ import annotations

import numpy as np

from .normalize import (
    ACTIVITY_IDX,
    BODY_JOINTS,
    FACE_JOINTS,
    FEATURE_DIM,
    HAND_JOINTS,
    NMM_CHANNELS,
    derive_nmm_from_frame,
)

# COCO-17 body indices
COCO_NOSE, COCO_L_EYE, COCO_R_EYE = 0, 1, 2
COCO_L_EAR, COCO_R_EAR = 3, 4
COCO_L_SHOULDER, COCO_R_SHOULDER = 5, 6
COCO_L_ELBOW, COCO_R_ELBOW = 7, 8
COCO_L_WRIST, COCO_R_WRIST = 9, 10
COCO_L_HIP, COCO_R_HIP = 11, 12
COCO_L_KNEE, COCO_R_KNEE = 13, 14
COCO_L_ANKLE, COCO_R_ANKLE = 15, 16

FACE_START = 23
FACE_END = 91  # exclusive
LH_START, RH_START = 91, 112
CHEST_IDX, MIDHIP_IDX = 133, 134

# COCO-WholeBody 68-face → our named face joints (approximate but stable).
# Contour / landmark indices follow the common COCO-WholeBody face topology.
FACE_NAME_TO_COCO68 = {
    "leftEye": 37,  # approx left eye center-ish in 68-pt
    "rightEye": 46,
    "nose": 30,
    "mouthLeft": 48,
    "mouthRight": 54,
    "leftEyebrowOuter": 17,
    "rightEyebrowOuter": 26,
    "chin": 8,
    "forehead": 27,  # bridge / upper nose as soft forehead proxy
    "mouthCenter": 51,
    "leftEyebrowInner": 21,
    "rightEyebrowInner": 22,
    "leftEyeTop": 38,
    "leftEyeBottom": 40,
    "rightEyeTop": 43,
    "rightEyeBottom": 47,
    "outerLipTop": 51,
    "outerLipBottom": 57,
    "innerLipTop": 62,
    "innerLipBottom": 66,
}


def _xy(frame: np.ndarray, idx: int) -> np.ndarray:
    """frame is (2, 135) → return (2,) xy."""
    if idx < 0 or idx >= frame.shape[1]:
        return np.zeros(2, dtype=np.float32)
    return frame[:, idx].astype(np.float32)


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if float(np.abs(a).sum()) < 1e-8:
        return b.copy()
    if float(np.abs(b).sum()) < 1e-8:
        return a.copy()
    return 0.5 * (a + b)


def coco135_frame_to_feature(frame_2x135: np.ndarray) -> np.ndarray:
    """Convert one (2, 135) COCO-WholeBody frame → FEATURE_DIM vector."""
    assert frame_2x135.shape[0] == 2 and frame_2x135.shape[1] >= 133
    f = frame_2x135

    # Hands — MediaPipe / COCO hand order matches HAND_JOINTS.
    left = np.zeros((21, 2), dtype=np.float32)
    right = np.zeros((21, 2), dtype=np.float32)
    for i in range(21):
        left[i] = _xy(f, LH_START + i)
        right[i] = _xy(f, RH_START + i)

    # Body in our order.
    nose = _xy(f, COCO_NOSE)
    l_sh, r_sh = _xy(f, COCO_L_SHOULDER), _xy(f, COCO_R_SHOULDER)
    l_el, r_el = _xy(f, COCO_L_ELBOW), _xy(f, COCO_R_ELBOW)
    l_wr, r_wr = _xy(f, COCO_L_WRIST), _xy(f, COCO_R_WRIST)
    l_hip, r_hip = _xy(f, COCO_L_HIP), _xy(f, COCO_R_HIP)
    l_knee, r_knee = _xy(f, COCO_L_KNEE), _xy(f, COCO_R_KNEE)
    l_ank, r_ank = _xy(f, COCO_L_ANKLE), _xy(f, COCO_R_ANKLE)
    l_ear, r_ear = _xy(f, COCO_L_EAR), _xy(f, COCO_R_EAR)
    neck = _mid(l_sh, r_sh)
    root = _xy(f, MIDHIP_IDX) if f.shape[1] > MIDHIP_IDX else _mid(l_hip, r_hip)
    if float(np.abs(root).sum()) < 1e-8:
        root = _mid(l_hip, r_hip)

    body_map = {
        "nose": nose,
        "neck": neck,
        "rightShoulder": r_sh,
        "rightElbow": r_el,
        "rightWrist": r_wr,
        "leftShoulder": l_sh,
        "leftElbow": l_el,
        "leftWrist": l_wr,
        "rightHip": r_hip,
        "rightKnee": r_knee,
        "rightAnkle": r_ank,
        "leftHip": l_hip,
        "leftKnee": l_knee,
        "leftAnkle": l_ank,
        "root": root,
        "rightEar": r_ear,
        "leftEar": l_ear,
    }
    body = np.stack([body_map[n] for n in BODY_JOINTS], axis=0)

    # Face from 68-pt block (+ COCO eyes as fallback).
    face = np.zeros((len(FACE_JOINTS), 2), dtype=np.float32)
    for i, name in enumerate(FACE_JOINTS):
        idx68 = FACE_NAME_TO_COCO68.get(name)
        if idx68 is not None:
            face[i] = _xy(f, FACE_START + idx68)
    # Fallbacks from body/COCO face-adjacent
    if float(np.abs(face[0]).sum()) < 1e-8:  # leftEye
        face[0] = _xy(f, COCO_L_EYE)
    if float(np.abs(face[1]).sum()) < 1e-8:
        face[1] = _xy(f, COCO_R_EYE)
    if float(np.abs(face[2]).sum()) < 1e-8:
        face[2] = nose

    # Activity from wrist motion proxy (filled temporally by caller; per-frame: hand presence).
    hand_energy = float(np.mean(np.abs(left)) + np.mean(np.abs(right)))
    activity = float(np.clip(hand_energy * 1.5, 0.0, 1.0))

    # Build a temporary frame dict for NMM derivation.
    frame_dict = {
        "face": [
            {"name": FACE_JOINTS[i], "x": float(face[i, 0]), "y": float(face[i, 1])}
            for i in range(len(FACE_JOINTS))
        ],
        "body": [
            {"name": BODY_JOINTS[i], "x": float(body[i, 0]), "y": float(body[i, 1])}
            for i in range(len(BODY_JOINTS))
        ],
    }
    nmm = derive_nmm_from_frame(frame_dict)

    vec = np.concatenate(
        [
            left.reshape(-1),
            right.reshape(-1),
            body.reshape(-1),
            face.reshape(-1),
            np.array([activity], dtype=np.float32),
            nmm.astype(np.float32),
        ]
    )
    assert vec.shape[0] == FEATURE_DIM
    return vec.astype(np.float32)


def coco135_sequence_to_features(seq_t2k: np.ndarray) -> np.ndarray:
    """(T, 2, 135) → (T, FEATURE_DIM) with temporal activity + head shake/nod proxies."""
    if seq_t2k.ndim != 3:
        raise ValueError(f"expected (T,2,K), got {seq_t2k.shape}")
    # HDF5 layout is (T, 2, 135)
    T = seq_t2k.shape[0]
    out = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    noses = []
    for t in range(T):
        out[t] = coco135_frame_to_feature(seq_t2k[t])
        noses.append(out[t][84:86].copy())  # body nose xy after pack (first body joint)

    # Temporal activity from wrist deltas
    rw = out[:, 42:44]  # right wrist in hand pack
    if T > 1:
        d = np.linalg.norm(np.diff(rw, axis=0), axis=1)
        d = np.concatenate([[d[0]], d])
        out[:, ACTIVITY_IDX] = np.clip(d * 8.0 + 0.05, 0, 1).astype(np.float32)

    # Head shake / nod from nose trajectory → NMM channels 7, 8
    noses_arr = np.stack(noses, axis=0)
    if T >= 5:
        xs, ys = noses_arr[:, 0], noses_arr[:, 1]
        x_sc = y_sc = 0
        for i in range(2, T):
            dx0, dx1 = xs[i - 1] - xs[i - 2], xs[i] - xs[i - 1]
            dy0, dy1 = ys[i - 1] - ys[i - 2], ys[i] - ys[i - 1]
            if dx0 * dx1 < 0 and abs(dx1) > 0.004:
                x_sc += 1
            if dy0 * dy1 < 0 and abs(dy1) > 0.004:
                y_sc += 1
        x_range = float(xs.max() - xs.min())
        y_range = float(ys.max() - ys.min())
        shake = min(1.0, (x_sc / 4.0) * min(1.0, x_range / 0.04))
        nod = min(1.0, (y_sc / 4.0) * min(1.0, y_range / 0.035))
        # Broadcast mild temporal envelope
        for t in range(T):
            out[t, ACTIVITY_IDX + 1 + 7] = max(out[t, ACTIVITY_IDX + 1 + 7], shake * 0.85)
            out[t, ACTIVITY_IDX + 1 + 8] = max(out[t, ACTIVITY_IDX + 1 + 8], nod * 0.85)

    return out


def pad_or_trim(seq: np.ndarray, length: int = 32) -> np.ndarray:
    if seq.shape[0] == 0:
        return np.zeros((length, FEATURE_DIM), dtype=np.float32)
    if seq.shape[0] < length:
        pad = np.repeat(seq[:1], length - seq.shape[0], axis=0)
        return np.concatenate([pad, seq], axis=0)
    return seq[-length:]
