#!/usr/bin/env bash
# Download Uni-Sign pose-only checkpoints (CC-BY-NC-4.0, non-commercial).
# These use Uni-Sign's GCN+LLM architecture — NOT auto-loaded by PoseLSTM.
# Runtime inference uses models/sign_classifier.pt (see MODELS.md).
set -euo pipefail
cd "$(dirname "$0")/../models"
echo "Downloading into $(pwd)"
echo "License: CC-BY-NC-4.0 — non-commercial use only. Not Google SL2T."
echo "Note: ~1.1GB+ each. Architecture mismatch with our MediaPipe FEATURE_DIM."

download() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo "exists: $file"
    return
  fi
  echo "fetching $file ..."
  curl -L --fail --retry 3 -o "$file" \
    "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main/$file"
}

# Isolated sign recognition (WLASL) — research reference only
download "wlasl_pose_only_islr.pth"
# Optional continuous SLT (How2Sign) — large
# download "how2sign_pose_only_slt.pth"

echo "Done. /health will report the file as present-architecture-mismatch."
echo "Active runtime model remains sign_classifier.pt (PoseLSTM)."
