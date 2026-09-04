#!/usr/bin/env python3
"""Offline eval/demo: run a recorded landmark JSON sequence through the decoder.

Usage:
  python eval_demo.py samples/example_sequence.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.decoder import ContinuousDecoder
from pipeline.gloss_english import gloss_to_english


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "samples" / "example_sequence.json")
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text())
    frames = data["frames"] if isinstance(data, dict) and "frames" in data else data
    decoder = ContinuousDecoder()
    result = decoder.decode_window(frames)
    english = gloss_to_english(result.get("gloss") or [])
    print(json.dumps({"result": result, "english": english, "model": decoder.model_name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
