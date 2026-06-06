#!/usr/bin/env bash
# persona-gen installer for Apple Silicon macOS. Run once: bash scripts/install.sh
set -euo pipefail
say(){ printf "\033[36m%s\033[0m\n" "$*"; }; ok(){ printf "  \033[32mok\033[0m %s\n" "$*"; }; bad(){ printf "  \033[31mx\033[0m %s\n" "$*"; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

say "-- persona-gen installer --"
[ "$(uname)" = "Darwin" ] || { bad "macOS only."; exit 1; }
[ "$(uname -m)" = "arm64" ] || { bad "Apple Silicon (M1 or newer) required."; exit 1; }
ok "macOS on Apple Silicon"

if ! xcode-select -p >/dev/null 2>&1; then
  say "-> installing Xcode Command Line Tools (click Install in the popup, then re-run this script)"
  xcode-select --install >/dev/null 2>&1 || true
  exit 1
fi
ok "Xcode Command Line Tools"

if ! command -v uv >/dev/null 2>&1; then
  say "-> installing uv (Python/dependency manager)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version | awk '{print $2}')"

say "-> creating .venv (Python 3.12) and installing dependencies (several GB, be patient)"
# Idempotent: reuse an existing venv so re-running the script (e.g. to download the model
# later) does not fail with "a virtual environment already exists" - but only if it is a
# sane Python 3.12 / arm64 venv; otherwise recreate it rather than trust a stale/wrong one.
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys,platform; raise SystemExit(0 if sys.version_info[:2]==(3,12) and platform.machine()=="arm64" else 1)' 2>/dev/null; then
  :  # existing venv is fine, reuse it
else
  rm -rf .venv && uv venv --python 3.12 .venv
fi
# --prerelease=allow is required: diffusers 0.38 depends on safetensors>=0.8.0rc0 (a
# pre-release), so resolution fails without it. All versions are exact-pinned, so the result
# is still deterministic. The pinned set is verified to import and report MPS available.
VIRTUAL_ENV="$REPO/.venv" uv pip install --prerelease=allow --python .venv/bin/python -r requirements.txt
# Verify the stack the app actually uses imports AND that Metal really works (not just a bare
# assert, which `python -O` would skip). Exits with a clear message on any failure.
.venv/bin/python - <<'PYCHECK'
import sys
import torch
from diffusers import ZImagePipeline  # the engine the app actually loads
import fastapi, uvicorn, compel, safetensors, PIL, numpy, yaml  # noqa: F401
if not torch.backends.mps.is_available():
    sys.exit("  Metal (MPS) is not available - this app needs an Apple Silicon Mac on a recent macOS.")
torch.empty(8, device="mps")          # prove an MPS allocation actually works
torch.Generator(device="mps")
print(f"  torch {torch.__version__}, MPS ok, ZImagePipeline import ok")
PYCHECK
ok "dependencies installed and verified"

# Z-Image-Turbo is Apache-2.0 and not gated -> downloads without any token or login.
ZDIR="$REPO/models/Z-Image-Turbo"
MARK="$ZDIR/.download-complete"
MODEL_OK=0
[ -x .venv/bin/hf ] || { bad "the 'hf' CLI is missing (huggingface_hub did not install) - cannot fetch the model"; exit 1; }
# Gate on a completion MARKER, not model_index.json: an interrupted download can leave
# model_index.json present while shards are missing. hf download resumes/verifies existing
# files, so re-running after an interruption completes it; the marker is written only on success.
if [ -f "$MARK" ]; then
  MODEL_OK=1
else
  printf "\n-> Download the Z-Image model now? It is required to generate and is free (Apache-2.0,\n   no account/token). About 16 GB. [Y/n] "
  read -r ANS || ANS=y
  case "${ANS:-y}" in
    [Nn]*) say "   skipped - the app will not generate until you download a model (see docs/MODELS.md)";;
    # explicit if so a failed/interrupted download stops the script (set -e does NOT trip on
    # the left side of '&&') instead of silently writing a 'ready' config and printing Done.
    *) if .venv/bin/hf download Tongyi-MAI/Z-Image-Turbo --local-dir "$ZDIR"; then
         : > "$MARK"; MODEL_OK=1; ok "Z-Image downloaded"
       else
         bad "model download failed - re-run the script to resume (see docs/MODELS.md)"; exit 1
       fi;;
  esac
fi

# Write a working config.yaml (Z-Image ready to go). The SDXL/curation engine is optional.
if [ ! -f config.yaml ]; then
  cat > config.yaml <<YAML
output_dir: ./output
lock_file: /tmp/persona-gen.lock

engines:
  zimage:
    type: zimage
    label: Z-Image
    model: ./models/Z-Image-Turbo
    steps: 9
    cfg: 0.0

# Optional: a curated SDXL engine (base -> realism refiner -> face/feet/hands refine).
# Bring your own SDXL checkpoint (diffusers format) and uncomment. See docs/MODELS.md.
#  sdxl:
#    type: sdxl
#    label: SDXL (curated)
#    model: ~/models/your-sdxl-checkpoint
#    steps: 28
#    cfg: 6.5
#    refiner: ~/models/your-refiner-sdxl
#    feet_lora: ~/models/loras/feet.safetensors
#    hand_lora: ~/models/loras/hands.safetensors
#    detectors:
#      yoloface:  ~/models/onnx-face/yoloface_8n.onnx
#      pose_task: ~/models/mediapipe/pose_landmarker_heavy.task
#      hand_task: ~/models/mediapipe/hand_landmarker.task
YAML
  if [ "$MODEL_OK" = 1 ]; then ok "wrote config.yaml (Z-Image ready)"; else ok "wrote config.yaml (download a model before running - see docs/MODELS.md)"; fi
fi

printf "\n\033[32mDone.\033[0m Launch the app:  bash scripts/run.sh   (opens http://localhost:7860)\n"
[ "$MODEL_OK" = 1 ] || printf "\033[33mNote:\033[0m no model yet - re-run this script (and accept the download) or see docs/MODELS.md first.\n"
printf "\n"
