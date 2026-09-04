"""Continuous landmark → gloss decoder.

Loads PoseLSTM weights from models/sign_classifier.pt (shipping default when
present). Falls back to a heuristic demo decoder. Uni-Sign .pth files are
detected and reported but require their native GCN+LLM stack (see MODELS.md).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .normalize import FEATURE_DIM, frames_to_tensor, normalize_frames
from .uni_sign_adapter import find_uni_sign_checkpoint, describe_checkpoint
from .vocab import GLOSS_VOCAB

logger = logging.getLogger("asl.decoder")

DEMO_GLOSS_VOCAB = list(GLOSS_VOCAB[:25])

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
        self.trained_on: str | None = None
        self._load()

    def _load(self) -> None:
        candidates = [
            self.models_dir / "sign_classifier.pt",
            self.models_dir / "poselstm.pt",
            self.models_dir / "uni_sign.pt",
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
                    self._note_uni_sign_pending()
                    return
            else:
                if self._try_torch(path):
                    self._note_uni_sign_pending()
                    return

        uni = find_uni_sign_checkpoint(self.models_dir)
        if uni is not None:
            meta = describe_checkpoint(uni)
            logger.warning(
                "Found Uni-Sign checkpoint %s (%s bytes) but native GCN+LLM "
                "stack is not embedded — using demo decoder. %s",
                uni.name,
                meta.get("bytes"),
                meta.get("hint"),
            )
            self.backend = "demo"
            self.model_name = f"demo-continuous-v1+pending:{uni.name}"
            self.uni_sign_meta = meta
            return

        logger.info(
            "No PoseLSTM checkpoint in %s — demo decoder. "
            "Run scripts/train_poselstm.py or see MODELS.md.",
            self.models_dir,
        )
        self.backend = "demo"
        self.model_name = "demo-continuous-v1"
        self.uni_sign_meta = None

    def _note_uni_sign_pending(self) -> None:
        uni = find_uni_sign_checkpoint(self.models_dir)
        if uni is not None and uni.suffix == ".pth":
            self.uni_sign_meta = describe_checkpoint(uni)

    def _try_torch(self, path: Path) -> bool:
        try:
            import torch
            from .sequence_model import PoseLSTMClassifier

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
                num_classes = int(ckpt.get("num_classes", len(self._label_map)))
                input_dim = int(ckpt.get("input_dim", FEATURE_DIM))
                hidden_dim = int(ckpt.get("hidden_dim", 256))
                if "labels" in ckpt:
                    self._label_map = [str(x) for x in ckpt["labels"]]
                self.trained_on = str(ckpt.get("trained_on") or "")
            elif isinstance(ckpt, dict) and any(
                k.startswith("lstm") or k.startswith("fc") for k in ckpt
            ):
                state = ckpt
                num_classes = len(self._label_map)
                input_dim = FEATURE_DIM
                hidden_dim = 256
            else:
                logger.warning("Unrecognized checkpoint format at %s", path)
                return False

            model = PoseLSTMClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
            )
            model.load_state_dict(state, strict=False)
            model.eval()
            self._torch_model = model
            self.backend = "pytorch"
            self.model_name = path.name
            logger.info(
                "Loaded PyTorch weights from %s (%d classes, hidden=%d)",
                path,
                num_classes,
                hidden_dim,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return False

    def _try_onnx(self, path: Path) -> bool:
        try:
            import onnxruntime as ort

            self._onnx = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
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
            return self._decode_torch_continuous(frames)
        if self._onnx is not None:
            return self._decode_onnx(frames)
        return self._decode_demo(frames)

    def _predict_torch_single(self, frames: list[dict[str, Any]]) -> tuple[str, float]:
        import torch

        tensor = frames_to_tensor(frames, window=32)
        with torch.no_grad():
            logits = self._torch_model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            conf, idx = torch.max(probs, dim=-1)
        label = (
            self._label_map[int(idx)]
            if int(idx) < len(self._label_map)
            else "UNK"
        )
        return label, float(conf)

    def _decode_torch_continuous(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        """Sliding-window ISLR → gloss sequence with consecutive dedupe.

        For short clips (<=40 frames) emit a single top gloss.
        For longer utterances, slide windows and build a gloss phrase.
        """
        n = len(frames)
        if n < 6:
            return {"gloss": [], "confidence": 0.0, "source": f"pytorch:{self.model_name}"}

        # Gate on activity so rest poses don't spam glosses.
        arr = normalize_frames(frames)
        activity = float(np.mean(np.abs(arr[:, -1]))) if arr.size else 0.0
        if activity < 0.02:
            return {"gloss": [], "confidence": 0.0, "source": f"pytorch:{self.model_name}"}

        window = 32
        stride = 10
        min_conf = 0.35

        if n <= window + 8:
            label, conf = self._predict_torch_single(frames)
            if conf < min_conf:
                return {
                    "gloss": [],
                    "confidence": conf,
                    "source": f"pytorch:{self.model_name}",
                }
            return {
                "gloss": [label],
                "confidence": conf,
                "source": f"pytorch:{self.model_name}",
            }

        glosses: list[str] = []
        confs: list[float] = []
        for start in range(0, max(1, n - window + 1), stride):
            chunk = frames[start : start + window]
            if len(chunk) < 12:
                continue
            label, conf = self._predict_torch_single(chunk)
            if conf < min_conf:
                continue
            if not glosses or glosses[-1] != label:
                glosses.append(label)
                confs.append(conf)
            else:
                confs[-1] = max(confs[-1], conf)

        # Cap phrase length for subtitle readability
        glosses = glosses[:6]
        confs = confs[:6]
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return {
            "gloss": glosses,
            "confidence": mean_conf,
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
        """Demo continuous decoder — NOT a research SLT model."""
        arr = normalize_frames(frames)
        if arr.shape[0] < 4:
            return {"gloss": [], "confidence": 0.0, "source": "demo"}

        activity = float(np.mean(np.abs(arr[:, -1])))
        left = arr[:, 0:42]
        right = arr[:, 42:84]
        primary = right if np.mean(np.abs(right)) > np.mean(np.abs(left)) else left
        tips = primary[:, [8, 9, 16, 17, 24, 25, 32, 33, 40, 41]]
        openness = float(
            np.mean(np.linalg.norm(tips.reshape(tips.shape[0], -1, 2), axis=-1))
        )
        wrist_y = primary[:, 1]
        dy = float(wrist_y[-1] - wrist_y[0]) if len(wrist_y) else 0.0
        wrist_x = primary[:, 0]
        dx = np.diff(wrist_x)
        osc = int(np.sum((dx[1:] * dx[:-1]) < 0)) if len(dx) > 2 else 0

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
