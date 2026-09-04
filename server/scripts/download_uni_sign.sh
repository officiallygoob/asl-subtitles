#!/usr/bin/env bash
# Download Uni-Sign pose-only checkpoints (CC-BY-NC-4.0, non-commercial).
# These are full Uni-Sign architectures — see MODELS.md for adapter notes.
# They will NOT auto-load into the PoseLSTM scaffold without conversion.
set -euo pipefail
cd "$(dirname "$0")/../models"
echo "Downloading into $(pwd)"
echo "License: CC-BY-NC-4.0 — non-commercial use only. Not Google SL2T."

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

# Isolated sign recognition (WLASL) — usually the smallest useful pose-only weight
download "wlasl_pose_only_islr.pth"
# Optional continuous SLT (How2Sign) — large
# download "how2sign_pose_only_slt.pth"

echo "Done. Next: follow MODELS.md to map Uni-Sign pose tensors → ContinuousDecoder."
