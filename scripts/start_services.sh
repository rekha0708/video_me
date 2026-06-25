#!/usr/bin/env bash
# Start all Track D services for the video_me pipeline.
# Run this after every pod restart.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[0;32m  ✓ %s\033[0m\n' "$*"; }
warn(){ printf '\033[0;33m  ! %s\033[0m\n' "$*"; }

# ── Ollama ────────────────────────────────────────────────────────────────────
log "Starting Ollama (LLM + VLM critique, port 11434)"
if pgrep -x ollama >/dev/null 2>&1; then
  ok "Ollama already running"
else
  OLLAMA_MODELS="$WORKSPACE/ollama" setsid -f ollama serve >"$LOG_DIR/ollama.log" 2>&1 &
  ok "Ollama started (log: $LOG_DIR/ollama.log)"
fi

# ── AUTOMATIC1111 ─────────────────────────────────────────────────────────────
log "Starting AUTOMATIC1111 Stable Diffusion (port 7860)"
A1111_DIR="$WORKSPACE/stable-diffusion-webui"
if ! curl -sf http://localhost:7860/sdapi/v1/sd-models >/dev/null 2>&1; then
  cd "$A1111_DIR"
  A1111_TAMING_REPO="$A1111_DIR/repositories/taming-transformers"
  A1111_SITE_PACKAGES=""
  for candidate in "$A1111_DIR"/venv/lib/python*/site-packages; do
    if [[ -d "$candidate" ]]; then
      A1111_SITE_PACKAGES="$candidate"
      break
    fi
  done
  if [[ -n "$A1111_SITE_PACKAGES" && -d "$A1111_TAMING_REPO" ]]; then
    printf '%s\n' "$A1111_TAMING_REPO" > "$A1111_SITE_PACKAGES/taming_transformers_repo.pth"
  fi
  # Use --skip-install and direct python launch to avoid git-fetch failures
  # (Stability-AI/stablediffusion repo was deleted upstream; git fetch hangs startup).
  # Pre-checkout the expected commit hash so git_clone() returns early.
  (cd "$A1111_DIR/repositories/stable-diffusion-stability-ai" && \
    git checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf 2>/dev/null || true)
  nohup "$A1111_DIR/venv/bin/python" "$A1111_DIR/launch.py" \
    --skip-install \
    -f \
    --api \
    --listen \
    --port 7860 \
    --nowebui \
    --skip-torch-cuda-test \
    >"$LOG_DIR/a1111.log" 2>&1 &
  ok "AUTOMATIC1111 starting (log: $LOG_DIR/a1111.log) — takes ~60s to load"
else
  ok "AUTOMATIC1111 already responding"
fi
cd "$ROOT_DIR"

# ── Chatterbox TTS ────────────────────────────────────────────────────────────
log "Starting Chatterbox TTS (port 8020)"
if ! curl -sf http://localhost:8020/health >/dev/null 2>&1; then
  cd "$ROOT_DIR"
  # Use the dedicated chatterbox venv (inherits system CUDA/torch; avoids dep conflicts)
  if [[ -f "$WORKSPACE/.venv_chatterbox/bin/uvicorn" ]]; then
    UVICORN="$WORKSPACE/.venv_chatterbox/bin/uvicorn"
  elif [[ -f ".venv/bin/uvicorn" ]]; then
    UVICORN=".venv/bin/uvicorn"
  else
    UVICORN="uvicorn"
  fi
  setsid -f "$UVICORN" services.chatterbox_server:app \
    --host 0.0.0.0 --port 8020 >"$LOG_DIR/chatterbox.log" 2>&1 &
  ok "Chatterbox TTS starting (log: $LOG_DIR/chatterbox.log)"
else
  ok "Chatterbox TTS already responding"
fi

# ── Wan 2.2 ───────────────────────────────────────────────────────────────────
log "Starting Wan2.2 image-to-video (port 8030)"
if ! curl -sf http://localhost:8030/health >/dev/null 2>&1; then
  cd "$ROOT_DIR"
  # Use dedicated Wan venv (inherits system CUDA/torch; has decord + imageio)
  if [[ -f "$WORKSPACE/.venv_wan/bin/uvicorn" ]]; then
    UVICORN="$WORKSPACE/.venv_wan/bin/uvicorn"
  elif [[ -f ".venv/bin/uvicorn" ]]; then
    UVICORN=".venv/bin/uvicorn"
  else
    UVICORN="uvicorn"
  fi
  WAN_DIR="$WORKSPACE/Wan2.2" WAN_MODEL_DIR="$WORKSPACE/Wan2.2-I2V-A14B" \
  setsid -f "$UVICORN" services.wan_server:app \
    --host 0.0.0.0 --port 8030 >"$LOG_DIR/wan.log" 2>&1 &
  ok "Wan2.2 starting (log: $LOG_DIR/wan.log)"
else
  ok "Wan2.2 already responding"
fi

# ── MuseTalk ──────────────────────────────────────────────────────────────────
log "Starting MuseTalk lip-sync (port 8040)"
if ! curl -sf http://localhost:8040/health >/dev/null 2>&1; then
  cd "$ROOT_DIR"
  # Use the MuseTalk conda env if it exists (needed for Python 3.10 + CUDA 11.8)
  if conda env list 2>/dev/null | grep -q "MuseTalk"; then
    MUSETALK_UVICORN="$(conda run -n MuseTalk which uvicorn)"
  elif [[ -f ".venv/bin/uvicorn" ]]; then
    MUSETALK_UVICORN=".venv/bin/uvicorn"
  else
    MUSETALK_UVICORN="uvicorn"
  fi
  MUSETALK_DIR="$WORKSPACE/MuseTalk" \
  setsid -f "$MUSETALK_UVICORN" services.musetalk_server:app \
    --host 0.0.0.0 --port 8040 >"$LOG_DIR/musetalk.log" 2>&1 &
  ok "MuseTalk starting (log: $LOG_DIR/musetalk.log)"
else
  ok "MuseTalk already responding"
fi

# ── Health check ──────────────────────────────────────────────────────────────
printf '\nWaiting 20s for services to start...\n'
sleep 20

cd "$ROOT_DIR"
if [[ -f ".venv/bin/python" ]]; then
  .venv/bin/python -m scripts.check_runtime_readiness --allow-missing-services
else
  python -m scripts.check_runtime_readiness --allow-missing-services
fi
