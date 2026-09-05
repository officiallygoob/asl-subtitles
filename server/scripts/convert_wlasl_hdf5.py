#!/usr/bin/env python3
"""Legacy entrypoint — prefer convert_pose_hdf5.py for multi-source merges."""
from __future__ import annotations
import sys
from pathlib import Path

# Re-dispatch to multi-source converter with wlasl100 default
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # Map old CLI to new
    argv = sys.argv[1:]
    if "--mix-synth" in argv and "--sources" not in argv:
        argv = ["--sources", "wlasl100", "--out", str(ROOT / "models" / "wlasl100_features.npz")] + argv
    sys.argv = [sys.argv[0]] + argv
    from scripts.convert_pose_hdf5 import main
    raise SystemExit(main())
