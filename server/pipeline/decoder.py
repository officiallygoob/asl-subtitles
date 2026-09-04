"""Continuous landmark → gloss decoder.

Loads optional PoseLSTM / Transformer weights from models/*.pt or *.onnx.
Without weights, uses a clearly-marked demo continuous decoder that maps
motion/pose heuristics over the sliding window to a small gloss vocab —
enough to exercise the streaming protocol end-to-end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .normalize import FEATURE_DIM, frames_to_tensor, normalize_frames
from .uni_sign_adapter import find_uni_sign_checkpoint, describe_checkpoint

logger = logging.getLogger("asl.decoder")

# Small demo gloss inventory used when no research checkpoint is present.
DEMO_GLOSS_VOCAB = [
    "HELLO", "THANKS", "YES", "NO", "PLEASE", "HELP", "NAME", "FRIEND",
    "LOVE", "HOW", "YOU", "ME", "GOOD", "BAD", "MORE", "SORRY", "BYE",
    "WHAT", "WHERE", "OK", "STOP", "UNDERSTAND", "AGAIN", "SLOW",
]

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ContinuousDecoder:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backend = "demo"
        self.model_name = "demo-continuous-v1"
        self._torch_model = None
        self._onnx = None
        self._label_map: list[str] = list(DEMO_GLOSS_VOCAB)
        self.uni_sign_meta = None
        self._load()

    def _load(self) -> None:
        # Prefer explicit uni-sign / poselstm checkpoints if present.
        candidates = [
            self.models_dir / "uni_sign.pt",
            self.models_dir / "poselstm.pt",
            self.models_dir / "sign_classifier.pt",
            self.models_dir / "model.onnx",
        ]
        meta_path = self.models_dir / "labels.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                if isinstance(data, list):
                    self._label_map = [str(x) for x in data]
                elif isinstance(data, dict) and "labels" in data:
                    self._label_map = [str(x) for x in data["labels"]]
            except Exception as exc:
                logger.warning("Could not read labels.json: %s", exc)

        for path in candidates:
            if not path.exists():
                continue
            if path.suffix == ".onnx":
                if self._try_onnx(path):
                    return
            else:
                if self._try_torch(path):
                    return

        uni = find_uni_sign_checkpoint(self.models_dir)
        if uni is not None:
            meta = describe_checkpoint(uni)
            logger.warning(
                "Found Uni-Sign checkpoint %s (%s bytes) but adapter is not wired — "
                "using demo decoder. Details: %s",
                uni.name,
                meta.get("bytes"),
                meta.get("hint"),
            )
            self.backend = "demo"
            self.model_name = f"demo-continuous-v1+pending:{uni.name}"
            self.uni_sign_meta = meta
            return

        logger.info(
            "No research checkpoint found in %s — using demo continuous decoder. "
            "See MODELS.md to plug Uni-Sign / PoseLSTM weights.",
            self.models_dir,
        )
        self.backend = "demo"
        self.model_name = "demo-continuous-v1"
        self.uni_sign_meta = None

    def _try_torch(self, path: Path) -> bool:
        try:
            import torch
            from .sequence_model import PoseLSTMClassifier

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
                num_classes = ckpt.get("num_classes", len(self._label_map))
                input_dim = ckpt.get("input_dim", FEATURE_DIM)
                if "labels" in ckpt:
                    self._label_map = [str(x) for x in ckpt["labels"]]
            elif isinstance(ckpt, dict) and any(k.startswith("lstm") or k.startswith("fc") for k in ckpt):
                state = ckpt
                num_classes = len(self._label_map)
                input_dim = FEATURE_DIM
            else:
                # Unknown format — keep file noted but don't crash.
                logger.warning("Unrecognized checkpoint format at %s", path)
                return False

            model = PoseLSTMClassifier(input_dim=input_dim, num_classes=num_classes)
            model.load_state_dict(state, strict=False)
            model.eval()
            self._torch_model = model
            self.backend = "pytorch"
            self.model_name = path.name
            logger.info("Loaded PyTorch weights from %s (%d classes)", path, num_classes)
            return True
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return False

    def _try_onnx(self, path: Path) -> bool:
        try:
            import onnxruntime as ort  # optional

            self._onnx = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.backend = "onnx"
            self.model_name = path.name
            logger.info("Loaded ONNX model from %s", path)
            return True
        except Exception as exc:
            logger.warning("ONNX load failed for %s: %s", path, exc)
            return False

    # ---- inference -----------------------------------------------------

    def decode_window(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        if not frames:
            return {"gloss": [], "confidence": 0.0, "source": self.backend}

        if self._torch_model is not None:
            return self._decode_torch(frames)
        if self._onnx is not None:
            return self._decode_onnx(frames)
        return self._decode_demo(frames)

    def _decode_torch(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        tensor = frames_to_tensor(frames, window=32)
        with torch.no_grad():
            logits = self._torch_model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            conf, idx = torch.max(probs, dim=-1)
        label = self._label_map[int(idx)] if int(idx) < len(self._label_map) else "UNK"
        return {
            "gloss": [label],
            "confidence": float(conf),
            "source": f"pytorch:{self.model_name}",
        }

    def _decode_onnx(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        arr = normalize_frames(frames)
        window = 32
        if arr.shape[0] < window:
            pad = np.zeros((window - arr.shape[0], FEATURE_DIM), dtype=np.float32)
            arr = np.concatenate([pad, arr], axis=0)
        else:
            arr = arr[-window:]
        inp = arr[None, ...].astype(np.float32)
        input_name = self._onnx.get_inputs()[0].name
        out = self._onnx.run(None, {input_name: inp})[0]
        probs = _softmax(out[0])
        idx = int(np.argmax(probs))
        label = self._label_map[idx] if idx < len(self._label_map) else "UNK"
        return {
            "gloss": [label],
            "confidence": float(probs[idx]),
            "source": f"onnx:{self.model_name}",
        }

    def _decode_demo(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        """Demo continuous decoder — NOT a research SLT model.

        Uses window-level motion + hand shape proxies so the streaming
        conversation architecture is testable without Uni-Sign weights.
        """
        arr = normalize_frames(frames)
        if arr.shape[0] < 4:
            return {"gloss": [], "confidence": 0.0, "source": "demo"}

        activity = float(np.mean(np.abs(arr[:, -1])))
        # Hand openness proxy: mean distance of fingertip dims from wrist.
        left = arr[:, 0:42]
        right = arr[:, 42:84]
        primary = right if np.mean(np.abs(right)) > np.mean(np.abs(left)) else left
        tips = primary[:, [8, 9, 16, 17, 24, 25, 32, 33, 40, 41]]  # tip xy pairs flattened-ish
        openness = float(np.mean(np.linalg.norm(tips.reshape(tips.shape[0], -1, 2), axis=-1)))
        # Vertical motion of wrist
        wrist_y = primary[:, 1]
        dy = float(wrist_y[-1] - wrist_y[0]) if len(wrist_y) else 0.0
        # Lateral oscillation count
        wrist_x = primary[:, 0]
        dx = np.diff(wrist_x)
        osc = int(np.sum((dx[1:] * dx[:-1]) < 0)) if len(dx) > 2 else 0

        gloss: list[str] = []
        conf = 0.45

        if activity < 0.02:
            return {"gloss": [], "confidence": 0.0, "source": "demo"}

        if openness > 0.35 and abs(dy) < 0.05 and osc <= 1:
            gloss, conf = ["HELLO"], 0.55
        elif dy > 0.12 and openness < 0.25:
            gloss, conf = ["THANKS"], 0.52
        elif osc >= 3 and openness < 0.3:
            gloss, conf = ["NO"], 0.5
        elif dy < -0.1 and openness > 0.25:
            gloss, conf = ["YES"], 0.5
        elif openness > 0.4 and dy > 0.05:
            gloss, conf = ["GOOD"], 0.48
        elif openness < 0.15 and activity > 0.05:
            gloss, conf = ["PLEASE"], 0.46
        elif osc >= 2 and openness > 0.3:
            gloss, conf = ["HOW", "YOU"], 0.45
        elif activity > 0.15 and openness > 0.25:
            gloss, conf = ["WHAT"], 0.42
        else:
            gloss, conf = ["ME"], 0.4

        return {"gloss": gloss, "confidence": conf, "source": "demo"}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)
