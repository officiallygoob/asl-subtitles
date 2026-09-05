"""On-device-training pose augmentations (FEATURE_DIM=170 layout).

Mirrors iPhone-camera variability: speed, left/right flip, noise, temporal jitter.
Offline training only — never ships user video.
"""

from __future__ import annotations

import numpy as np

from .normalize import ACTIVITY_IDX, FEATURE_DIM, NMM_START

# Layout: L-hand 0:42, R-hand 42:84, body 84:118, face 118:158, act 158, nmm 159:170
LH, RH, BODY, FACE = slice(0, 42), slice(42, 84), slice(84, 118), slice(118, 158)

# Body joint order indices (x,y pairs) for left↔right swap on mirror
# BODY_JOINTS: nose,neck, rSh,rEl,rWr, lSh,lEl,lWr, rHip,rKnee,rAnk, lHip,lKnee,lAnk, root, rEar, lEar
_BODY_SWAP_PAIRS = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (15, 16)]
# FACE: leftEye,rightEye, nose, mouthLeft,mouthRight, lBrowO,rBrowO, chin, forehead, mouthCenter,
#       lBrowI,rBrowI, lEyeTop,lEyeBot, rEyeTop,rEyeBot, outerLipTop,outerLipBot, innerLipTop,innerLipBot
_FACE_SWAP_PAIRS = [(0, 1), (3, 4), (5, 6), (10, 11), (12, 14), (13, 15)]


def _swap_xy_pairs(block: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """block (T, J*2); swap joint pairs in-place copy."""
    out = block.copy()
    for a, b in pairs:
        sa, sb = a * 2, b * 2
        out[:, sa : sa + 2], out[:, sb : sb + 2] = block[:, sb : sb + 2].copy(), block[:, sa : sa + 2].copy()
    return out


def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Horizontal flip: negate x, swap L/R hands + paired body/face joints."""
    assert seq.ndim == 2 and seq.shape[1] == FEATURE_DIM
    out = seq.copy()
    # negate all x channels in spatial block
    spatial = out[:, :ACTIVITY_IDX].reshape(out.shape[0], -1, 2)
    spatial[:, :, 0] *= -1.0
    out[:, :ACTIVITY_IDX] = spatial.reshape(out.shape[0], -1)
    # swap hands
    lh, rh = out[:, LH].copy(), out[:, RH].copy()
    out[:, LH], out[:, RH] = rh, lh
    out[:, BODY] = _swap_xy_pairs(out[:, BODY], _BODY_SWAP_PAIRS)
    out[:, FACE] = _swap_xy_pairs(out[:, FACE], _FACE_SWAP_PAIRS)
    # NMM: headShake stays; shoulderTilt / torsoLean flip sign-ish via channel swap approximations
    # shoulderTilt is last NMM channel index 10 → negate
    out[:, NMM_START + 10] *= -1.0
    out[:, NMM_START + 9] *= -1.0  # torsoLean
    return out.astype(np.float32)


def speed_resample(seq: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    """Temporal speed change then pad/trim back to original length."""
    T, D = seq.shape
    if T < 4:
        return seq.copy()
    new_t = max(4, int(round(T / max(factor, 0.25))))
    # linear interpolate along time
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, new_t)
    # Vectorized linear resample along time for all feature dims.
    # new_idx into old frames → weights
    idx0 = np.floor(new_idx).astype(np.int64)
    idx1 = np.clip(idx0 + 1, 0, T - 1)
    idx0 = np.clip(idx0, 0, T - 1)
    w1 = (new_idx - idx0).astype(np.float32)[:, None]
    w0 = 1.0 - w1
    resampled = (seq[idx0] * w0 + seq[idx1] * w1).astype(np.float32)
    if new_t == T:
        return resampled
    if new_t > T:
        start = int(rng.integers(0, new_t - T + 1))
        return resampled[start : start + T]
    # pad by repeating edges
    pad_l = (T - new_t) // 2
    pad_r = T - new_t - pad_l
    return np.concatenate(
        [np.repeat(resampled[:1], pad_l, axis=0), resampled, np.repeat(resampled[-1:], pad_r, axis=0)],
        axis=0,
    ).astype(np.float32)


def add_noise(seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.015) -> np.ndarray:
    out = seq.copy()
    noise = rng.normal(0.0, sigma, size=out[:, :ACTIVITY_IDX].shape).astype(np.float32)
    out[:, :ACTIVITY_IDX] += noise
    # mild NMM jitter
    out[:, NMM_START:] += rng.normal(0.0, sigma * 0.5, size=out[:, NMM_START:].shape).astype(np.float32)
    out[:, NMM_START:] = np.clip(out[:, NMM_START:], 0.0, 1.0)
    return out


def temporal_shift(seq: np.ndarray, rng: np.random.Generator, max_shift: int = 3) -> np.ndarray:
    if seq.shape[0] < 8:
        return seq.copy()
    s = int(rng.integers(-max_shift, max_shift + 1))
    if s == 0:
        return seq.copy()
    return np.roll(seq, shift=s, axis=0).astype(np.float32)


def joint_dropout(seq: np.ndarray, rng: np.random.Generator, p: float = 0.08) -> np.ndarray:
    """Zero random hand/face joints (simulates Vision dropout)."""
    out = seq.copy()
    T = out.shape[0]
    # drop some hand joints
    for base in (0, 42):
        for j in range(21):
            if rng.random() < p:
                out[:, base + j * 2 : base + j * 2 + 2] = 0.0
    for j in range(20):
        if rng.random() < p * 0.5:
            out[:, 118 + j * 2 : 118 + j * 2 + 2] = 0.0
    return out.astype(np.float32)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random subset of augmentations."""
    out = seq.astype(np.float32, copy=True)
    if rng.random() < 0.5:
        out = mirror_sequence(out)
    if rng.random() < 0.7:
        factor = float(rng.uniform(0.7, 1.35))
        out = speed_resample(out, factor, rng)
    if rng.random() < 0.8:
        out = add_noise(out, rng, sigma=float(rng.uniform(0.008, 0.025)))
    if rng.random() < 0.5:
        out = temporal_shift(out, rng)
    if rng.random() < 0.35:
        out = joint_dropout(out, rng)
    return out.astype(np.float32)
