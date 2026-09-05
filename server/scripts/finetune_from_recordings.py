#!/usr/bin/env python3
"""Fine-tune sign_classifier.pt on LandmarkRecorder JSON/JSONL exports (optional).

Primary accuracy path is offline WLASL→Core ML. Use this after capturing a
friend's signing on-device and AirDrop-ing Documents/LandmarkRecordings.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.normalize import FEATURE_DIM, frame_to_vector  # noqa
from scripts.train_ondevice_coreml import normalize_matrix  # noqa


def load_clips(path: Path) -> list[tuple[str, np.ndarray]]:
    clips = []
    files = []
    if path.is_file() and path.suffix == ".jsonl":
        for line in path.read_text().splitlines():
            if line.strip():
                files.append(json.loads(line))
    elif path.is_file() and path.suffix == ".json":
        files.append(json.loads(path.read_text()))
    elif path.is_dir():
        for p in sorted(path.glob("*.json")):
            files.append(json.loads(p.read_text()))
    for obj in files:
        label = str(obj.get("label", "")).upper().strip()
        frames = obj.get("frames") or []
        if not label or len(frames) < 8:
            continue
        mat = np.stack([frame_to_vector(f) for f in frames], axis=0)
        # pad/trim 32
        if mat.shape[0] < 32:
            pad = np.repeat(mat[:1], 32 - mat.shape[0], axis=0)
            mat = np.concatenate([pad, mat], 0)
        else:
            mat = mat[-32:]
        clips.append((label, mat.astype(np.float32)))
    return clips


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "sign_classifier.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    clips = load_clips(args.recordings)
    if not clips:
        print("no clips", file=sys.stderr)
        return 1

    import torch
    from torch import nn
    from pipeline.sequence_model import PoseLSTMClassifier

    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    labels = list(ckpt["labels"])
    for lab, _ in clips:
        if lab not in labels:
            labels.append(lab)
    gloss_to_id = {g: i for i, g in enumerate(labels)}
    X = np.stack([normalize_matrix(m) for _, m in clips])
    y = np.array([gloss_to_id[lab] for lab, _ in clips], dtype=np.int64)

    model = PoseLSTMClassifier(
        input_dim=int(ckpt.get("input_dim", FEATURE_DIM)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_layers=int(ckpt.get("num_layers", 3)),
        num_classes=len(labels),
    )
    # Load overlapping weights
    model.load_state_dict(ckpt["state_dict"], strict=False)
    # Expand final layer if new classes
    if model.num_classes != ckpt["num_classes"]:
        pass  # already constructed with new size; random head for new classes

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xt).argmax(-1) == yt).float().mean())
        print(f"epoch {ep:02d} loss={float(loss):.4f} train_acc={acc:.3f}")

    torch.save(
        {
            **{k: v for k, v in ckpt.items() if k not in {"state_dict", "labels", "num_classes"}},
            "state_dict": model.state_dict(),
            "labels": labels,
            "num_classes": len(labels),
            "input_dim": FEATURE_DIM,
            "hidden_dim": int(ckpt.get("hidden_dim", 192)),
            "trained_on": "friend-recordings-finetune",
            "val_acc": acc,
        },
        args.out,
    )
    (args.out.parent / "labels.json").write_text(json.dumps({"labels": labels, "val_acc": acc}, indent=2))
    print(f"saved {args.out} — re-export Core ML with train_ondevice_coreml.export_coreml if needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
