#!/usr/bin/env python3
"""Synthesize landmark sequences matching FEATURE_DIM for PoseLSTM training.

These are kinematic *templates* (not real signer footage). They give the
shipping classifier a real learned decision surface over our feature layout
so Conversation Mode is better than the hand-coded demo heuristics.

Fine-tune on LandmarkRecorder exports or public WLASL pose data for real accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.normalize import FEATURE_DIM, HAND_JOINTS, BODY_JOINTS, FACE_JOINTS  # noqa: E402
from pipeline.vocab import GLOSS_VOCAB  # noqa: E402

# Hand joint indices in the 21-joint MediaPipe/Vision order
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, LITTLE_TIP = 0, 4, 8, 12, 16, 20
INDEX_MCP, MIDDLE_MCP = 5, 9


def _hand_open(scale: float = 1.0) -> np.ndarray:
    """Return (21, 2) open-hand shape relative to wrist."""
    pts = np.zeros((21, 2), dtype=np.float32)
    # Approximate finger rays
    tips = {
        4: (-0.06 * scale, -0.10 * scale),
        8: (-0.02 * scale, -0.16 * scale),
        12: (0.02 * scale, -0.17 * scale),
        16: (0.05 * scale, -0.15 * scale),
        20: (0.08 * scale, -0.12 * scale),
    }
    for tip, (dx, dy) in tips.items():
        chain = {
            4: [1, 2, 3, 4],
            8: [5, 6, 7, 8],
            12: [9, 10, 11, 12],
            16: [13, 14, 15, 16],
            20: [17, 18, 19, 20],
        }[tip]
        for i, j in enumerate(chain):
            t = (i + 1) / len(chain)
            pts[j] = (dx * t, dy * t)
    return pts


def _hand_fist(scale: float = 1.0) -> np.ndarray:
    pts = np.zeros((21, 2), dtype=np.float32)
    for j in range(1, 21):
        pts[j] = (0.01 * scale * ((j % 5) - 2), -0.03 * scale)
    pts[4] = (-0.04 * scale, -0.02 * scale)  # thumb out a bit
    return pts


def _hand_point(scale: float = 1.0) -> np.ndarray:
    pts = _hand_fist(scale)
    pts[5] = (0.0, -0.04 * scale)
    pts[6] = (0.0, -0.08 * scale)
    pts[7] = (0.0, -0.12 * scale)
    pts[8] = (0.0, -0.16 * scale)
    return pts


def _hand_flat(scale: float = 1.0) -> np.ndarray:
    pts = _hand_open(scale * 0.9)
    # Flatten tips vertically similar height
    for tip in (8, 12, 16, 20):
        pts[tip, 1] = -0.14 * scale
    return pts


def _body_neutral() -> np.ndarray:
    """(17, 2) body joints in image-ish coords."""
    b = np.zeros((17, 2), dtype=np.float32)
    # nose, neck, Rshoulder, Relbow, Rwrist, Lshoulder, Lelbow, Lwrist,
    # Rhip, Rknee, Rankle, Lhip, Lknee, Lankle, root, rear, lear
    b[0] = (0.50, 0.22)  # nose
    b[1] = (0.50, 0.30)  # neck
    b[2] = (0.38, 0.32)  # R shoulder
    b[3] = (0.32, 0.48)
    b[4] = (0.30, 0.62)
    b[5] = (0.62, 0.32)  # L shoulder
    b[6] = (0.68, 0.48)
    b[7] = (0.70, 0.62)
    b[8] = (0.42, 0.58)
    b[9] = (0.42, 0.75)
    b[10] = (0.42, 0.90)
    b[11] = (0.58, 0.58)
    b[12] = (0.58, 0.75)
    b[13] = (0.58, 0.90)
    b[14] = (0.50, 0.55)
    b[15] = (0.44, 0.22)
    b[16] = (0.56, 0.22)
    return b


def _face_neutral() -> np.ndarray:
    f = np.zeros((10, 2), dtype=np.float32)
    f[0] = (0.46, 0.20)  # leftEye
    f[1] = (0.54, 0.20)
    f[2] = (0.50, 0.24)
    f[3] = (0.46, 0.28)
    f[4] = (0.54, 0.28)
    f[5] = (0.44, 0.17)
    f[6] = (0.56, 0.17)
    f[7] = (0.50, 0.32)
    f[8] = (0.50, 0.14)
    f[9] = (0.50, 0.28)
    return f


# Motion primitive: name -> generator(t in 0..1, rng) -> dict of modifiers
def _motion_spec(gloss: str):
    """Return (hand_shape, wrist_path_fn, which_hand, extras)."""
    g = gloss.upper()

    def path_wave(t, rng):
        return (0.08 * np.sin(t * 4 * np.pi + rng.uniform(0, 0.3)), -0.02 + 0.01 * np.sin(t * 2 * np.pi))

    def path_nod(t, rng):
        return (0.0, 0.06 * np.sin(t * 3 * np.pi))

    def path_out(t, rng):
        return (0.12 * t, -0.04 * t)

    def path_up(t, rng):
        return (0.0, -0.14 * t)

    def path_down(t, rng):
        return (0.0, 0.12 * t)

    def path_circle(t, rng):
        return (0.06 * np.cos(t * 2 * np.pi), 0.06 * np.sin(t * 2 * np.pi))

    def path_lateral(t, rng):
        return (0.14 * np.sin(t * 2 * np.pi), 0.0)

    def path_still(t, rng):
        return (0.01 * rng.normal(), 0.01 * rng.normal())

    def path_toward_chin(t, rng):
        return (0.02 * (1 - t), -0.10 * t)

    def path_from_chin(t, rng):
        return (0.10 * t, 0.08 * t)

    table = {
        "HELLO": ("open", path_wave, "right", {"y0": 0.28}),
        "HI": ("open", path_wave, "right", {"y0": 0.30}),
        "BYE": ("open", path_wave, "right", {"y0": 0.35, "amp": 1.2}),
        "SEE": ("point", path_out, "right", {"y0": 0.28}),
        "LATER": ("open", path_circle, "right", {"y0": 0.40}),
        "GOOD-MORNING": ("flat", path_up, "right", {"y0": 0.45}),
        "GOOD-NIGHT": ("flat", path_down, "right", {"y0": 0.30}),
        "THANKS": ("flat", path_from_chin, "right", {"y0": 0.26}),
        "PLEASE": ("flat", path_circle, "right", {"y0": 0.48}),
        "SORRY": ("fist", path_circle, "right", {"y0": 0.32}),
        "EXCUSE": ("flat", path_lateral, "right", {"y0": 0.45}),
        "ME": ("point", path_toward_chin, "right", {"y0": 0.40}),
        "YOU": ("point", path_out, "right", {"y0": 0.42}),
        "WE": ("point", path_lateral, "right", {"y0": 0.42}),
        "THEY": ("point", path_lateral, "right", {"y0": 0.40, "amp": 1.3}),
        "MY": ("flat", path_toward_chin, "right", {"y0": 0.45}),
        "YOUR": ("flat", path_out, "right", {"y0": 0.42}),
        "NAME": ("point", path_lateral, "right", {"y0": 0.30, "amp": 0.6}),
        "FRIEND": ("hook", path_lateral, "both", {"y0": 0.48}),
        "FAMILY": ("open", path_circle, "both", {"y0": 0.45}),
        "WHAT": ("open", path_lateral, "both", {"y0": 0.50}),
        "WHERE": ("point", path_wave, "right", {"y0": 0.40}),
        "WHEN": ("point", path_circle, "right", {"y0": 0.35}),
        "WHO": ("point", path_circle, "right", {"y0": 0.28}),
        "WHY": ("point", path_nod, "right", {"y0": 0.35}),
        "HOW": ("fist", path_twist := path_lateral, "both", {"y0": 0.48}),
        "WHICH": ("point", path_lateral, "right", {"y0": 0.45}),
        "YES": ("fist", path_nod, "right", {"y0": 0.35}),
        "NO": ("point", path_wave, "right", {"y0": 0.40, "amp": 1.4}),
        "OK": ("ok", path_out, "right", {"y0": 0.42}),
        "MAYBE": ("open", path_wave, "both", {"y0": 0.45}),
        "TRUE": ("point", path_out, "right", {"y0": 0.32}),
        "FALSE": ("point", path_lateral, "right", {"y0": 0.32}),
        "GOOD": ("flat", path_up, "right", {"y0": 0.45}),
        "BAD": ("flat", path_down, "right", {"y0": 0.40}),
        "FINE": ("open", path_still, "right", {"y0": 0.35}),
        "GREAT": ("open", path_up, "right", {"y0": 0.40}),
        "MORE": ("flat", path_toward_chin, "both", {"y0": 0.48}),
        "LESS": ("flat", path_down, "both", {"y0": 0.45}),
        "SAME": ("point", path_lateral, "both", {"y0": 0.45}),
        "DIFFERENT": ("point", path_out, "both", {"y0": 0.45}),
        "WANT": ("claw", path_toward_chin, "both", {"y0": 0.50}),
        "NEED": ("claw", path_down, "right", {"y0": 0.40}),
        "HELP": ("fist", path_up, "both", {"y0": 0.50}),
        "UNDERSTAND": ("point", path_toward_chin, "right", {"y0": 0.28}),
        "KNOW": ("flat", path_toward_chin, "right", {"y0": 0.26}),
        "DONT-KNOW": ("open", path_wave, "both", {"y0": 0.32}),
        "LIKE": ("open", path_toward_chin, "right", {"y0": 0.45}),
        "LOVE": ("cross", path_toward_chin, "both", {"y0": 0.48}),
        "GO": ("point", path_out, "right", {"y0": 0.48}),
        "COME": ("point", lambda t, rng: (-0.10 * t, 0.0), "right", {"y0": 0.48}),
        "STOP": ("flat", path_still, "right", {"y0": 0.42, "amp": 0.2}),
        "WAIT": ("open", path_wave, "both", {"y0": 0.45, "amp": 0.4}),
        "AGAIN": ("point", path_circle, "right", {"y0": 0.45}),
        "SLOW": ("flat", path_out, "right", {"y0": 0.45, "slow": True}),
        "FAST": ("flat", path_out, "right", {"y0": 0.45, "fast": True}),
        "EAT": ("pinch", path_toward_chin, "right", {"y0": 0.30}),
        "DRINK": ("c", path_up, "right", {"y0": 0.40}),
        "HOME": ("flat", path_toward_chin, "both", {"y0": 0.32}),
        "WORK": ("fist", path_nod, "both", {"y0": 0.48}),
        "SCHOOL": ("flat", path_clap := path_nod, "both", {"y0": 0.48}),
        "TIME": ("point", path_circle, "left", {"y0": 0.42}),
        "TODAY": ("point", path_down, "both", {"y0": 0.40}),
        "TOMORROW": ("point", path_out, "right", {"y0": 0.30}),
        "HUNGRY": ("claw", path_down, "right", {"y0": 0.45}),
        "TIRED": ("open", path_down, "both", {"y0": 0.28}),
        "HAPPY": ("flat", path_circle, "both", {"y0": 0.48}),
        "SAD": ("open", path_down, "both", {"y0": 0.28}),
        "HOT": ("claw", path_out, "right", {"y0": 0.30}),
        "COLD": ("fist", path_shake := path_wave, "both", {"y0": 0.48}),
        "SPELL": ("point", path_lateral, "right", {"y0": 0.50, "amp": 0.5}),
        "WRITE": ("pinch", path_lateral, "right", {"y0": 0.50}),
        "LOOK": ("point", path_out, "right", {"y0": 0.28}),
    }
    return table.get(g, ("open", path_wave, "right", {"y0": 0.40}))


def _shape(name: str, scale: float, rng: np.random.Generator) -> np.ndarray:
    if name == "open":
        return _hand_open(scale)
    if name == "fist":
        return _hand_fist(scale)
    if name == "point":
        return _hand_point(scale)
    if name == "flat":
        return _hand_flat(scale)
    if name == "ok":
        pts = _hand_open(scale)
        pts[4] = pts[8] * 0.5
        return pts
    if name == "claw":
        pts = _hand_open(scale * 0.7)
        for tip in (8, 12, 16, 20):
            pts[tip, 1] *= 0.6
        return pts
    if name == "pinch":
        pts = _hand_fist(scale)
        pts[4] = (-0.02, -0.06)
        pts[8] = (0.0, -0.06)
        return pts
    if name == "c":
        pts = _hand_open(scale * 0.8)
        pts[:, 0] *= 0.7
        return pts
    if name == "hook":
        pts = _hand_fist(scale)
        pts[8] = (0.0, -0.08)
        pts[12] = (0.02, -0.08)
        return pts
    if name == "cross":
        return _hand_fist(scale)
    return _hand_open(scale)


def synthesize_sequence(gloss: str, T: int, rng: np.random.Generator) -> np.ndarray:
    """Return (T, FEATURE_DIM) raw (unnormalized) feature vectors in image coords."""
    shape_name, path_fn, which, extras = _motion_spec(gloss)
    y0 = float(extras.get("y0", 0.40))
    amp = float(extras.get("amp", 1.0))
    slow = bool(extras.get("slow", False))
    fast = bool(extras.get("fast", False))

    # Signer scale / handedness noise
    scale = float(rng.uniform(0.85, 1.15))
    body = _body_neutral()
    # Jitter shoulders
    body += rng.normal(0, 0.008, body.shape).astype(np.float32)
    face = _face_neutral() + rng.normal(0, 0.005, (10, 2)).astype(np.float32)

    # Base wrist positions near shoulders
    r_base = np.array([0.62, y0], dtype=np.float32)
    l_base = np.array([0.38, y0], dtype=np.float32)

    frames = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    shape_r = _shape(shape_name, scale, rng)
    shape_l = _shape(shape_name, scale, rng)

    for t_i in range(T):
        t = t_i / max(T - 1, 1)
        if slow:
            t = t * 0.6
        if fast:
            t = min(1.0, t * 1.4)
        dx, dy = path_fn(t, rng)
        dx, dy = amp * dx * scale, amp * dy * scale

        left = np.zeros((21, 2), dtype=np.float32)
        right = np.zeros((21, 2), dtype=np.float32)

        if which in ("right", "both"):
            wrist = r_base + np.array([dx, dy], dtype=np.float32)
            right = shape_r + wrist
            body[4] = wrist  # right wrist body joint
        if which in ("left", "both"):
            # Mirror x motion for left
            wrist = l_base + np.array([-dx, dy], dtype=np.float32)
            left = shape_l.copy()
            left[:, 0] *= -1
            left = left + wrist
            body[7] = wrist

        # Idle opposite hand near hip if unused
        if which == "right":
            left = _hand_fist(scale * 0.8) + np.array([0.40, 0.62], dtype=np.float32)
        if which == "left":
            right = _hand_fist(scale * 0.8) + np.array([0.60, 0.62], dtype=np.float32)

        # Micro jitter
        left = left + rng.normal(0, 0.004, left.shape).astype(np.float32)
        right = right + rng.normal(0, 0.004, right.shape).astype(np.float32)

        activity = float(np.clip(np.linalg.norm([dx, dy]) * 4 + 0.05 + rng.normal(0, 0.01), 0, 1))

        vec = np.concatenate(
            [
                left.reshape(-1),
                right.reshape(-1),
                body.reshape(-1),
                face.reshape(-1),
                np.array([activity], dtype=np.float32),
            ]
        )
        frames[t_i] = vec
    return frames


def frames_array_to_landmark_dicts(arr: np.ndarray) -> list[dict]:
    """Convert (T, D) raw features to LandmarkFrame-like dicts for the server API."""
    out = []
    for t, row in enumerate(arr):
        left = row[0:42].reshape(21, 2)
        right = row[42:84].reshape(21, 2)
        body = row[84:118].reshape(17, 2)
        face = row[118:138].reshape(10, 2)
        activity = float(row[138])

        def hand(chirality, pts):
            joints = {HAND_JOINTS[i]: [float(pts[i, 0]), float(pts[i, 1])] for i in range(21)}
            return {"chirality": chirality, "confidence": 0.9, "joints": joints}

        body_list = [{"name": BODY_JOINTS[i], "x": float(body[i, 0]), "y": float(body[i, 1])} for i in range(17)]
        face_list = [{"name": FACE_JOINTS[i], "x": float(face[i, 0]), "y": float(face[i, 1])} for i in range(10)]
        out.append(
            {
                "timestamp": t / 15.0,
                "hands": [hand("left", left), hand("right", right)],
                "body": body_list,
                "face": face_list,
                "activity": activity,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "synth_dataset.npz")
    ap.add_argument("--per-class", type=int, default=48)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = []
    y = []
    for gi, gloss in enumerate(GLOSS_VOCAB):
        for _ in range(args.per_class):
            T = int(rng.integers(max(16, args.frames - 8), args.frames + 9))
            seq = synthesize_sequence(gloss, T, rng)
            # pad/truncate to args.frames for training tensor
            if seq.shape[0] < args.frames:
                pad = np.repeat(seq[:1], args.frames - seq.shape[0], axis=0)
                seq = np.concatenate([pad, seq], axis=0)
            else:
                seq = seq[-args.frames :]
            X.append(seq)
            y.append(gi)

    X_arr = np.stack(X, axis=0).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X_arr, y=y_arr, labels=np.array(GLOSS_VOCAB))
    print(f"wrote {args.out} X={X_arr.shape} classes={len(GLOSS_VOCAB)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
