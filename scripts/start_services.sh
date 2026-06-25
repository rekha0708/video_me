#!/usr/bin/env bash
# Start all Track D services for the video_me pipeline.
# Run this after every pod restart.
#
# Key fact: Ollama lives in base Linux (/usr/local/bin/ollama) which is wiped on
# RunPod pod restart. This script re-installs it if missing before starting it.
# All other services use /workspace/ which survives restarts on the network volume.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[0;32m  ✓ %s\033[0m\n' "$*"; }
warn(){ printf '\033[0;33m  ! %s\033[0m\n' "$*"; }
die() { printf '\033[0;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

wait_for() {
  local name="$1" url="$2" tries="${3:-15}"
  for i in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      ok "$name responding at $url"
      return 0
    fi
    printf '  waiting for %s (%d/%d)...\n' "$name" "$i" "$tries"
    sleep 4
  done
  warn "$name did not respond at $url after $((tries * 4))s"
  return 1
}

# ── Ollama ────────────────────────────────────────────────────────────────────
# Ollama is installed into base Linux which is WIPED on RunPod pod restart.
# Re-install if the binary is missing (models at /workspace/ollama persist fine).
log "Ollama (LLM + VLM critique, port 11434)"
if ! command -v ollama >/dev/null 2>&1; then
  warn "ollama binary missing (pod restarted) — reinstalling..."
  curl -fsSL https://ollama.com/install.sh | sh
  ok "Ollama reinstalled: $(ollama --version 2>&1 | head -1)"
else
  ok "Ollama binary present: $(ollama --version 2>&1 | head -1)"
fi

if pgrep -x ollama >/dev/null 2>&1; then
  ok "Ollama already running"
else
  OLLAMA_MODELS="$WORKSPACE/ollama" nohup ollama serve >"$LOG_DIR/ollama.log" 2>&1 &
  ok "Ollama started (models: $WORKSPACE/ollama, log: $LOG_DIR/ollama.log)"
fi

# ── AUTOMATIC1111 ─────────────────────────────────────────────────────────────
log "AUTOMATIC1111 Stable Diffusion (port 7860)"
A1111_DIR="$WORKSPACE/stable-diffusion-webui"
if [[ ! -d "$A1111_DIR" ]]; then
  die "AUTOMATIC1111 not found at $A1111_DIR — run setup_gpu.sh first"
fi

if curl -sf http://localhost:7860/sdapi/v1/sd-models >/dev/null 2>&1; then
  ok "AUTOMATIC1111 already responding"
else
  cd "$A1111_DIR"
  # Symlink taming-transformers so A1111 finds it without git-fetch
  A1111_SITE_PACKAGES=""
  for candidate in "$A1111_DIR"/venv/lib/python*/site-packages; do
    [[ -d "$candidate" ]] && A1111_SITE_PACKAGES="$candidate" && break
  done
  A1111_TAMING_REPO="$A1111_DIR/repositories/taming-transformers"
  if [[ -n "$A1111_SITE_PACKAGES" && -d "$A1111_TAMING_REPO" ]]; then
    printf '%s\n' "$A1111_TAMING_REPO" > "$A1111_SITE_PACKAGES/taming_transformers_repo.pth"
  fi
  # Pre-checkout pinned commit so git_clone() returns early (upstream repo deleted)
  (cd "$A1111_DIR/repositories/stable-diffusion-stability-ai" && \
    git checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf 2>/dev/null || true)
  nohup "$A1111_DIR/venv/bin/python" "$A1111_DIR/launch.py" \
    --skip-install -f --api --listen --port 7860 --nowebui --skip-torch-cuda-test \
    >"$LOG_DIR/a1111.log" 2>&1 &
  ok "AUTOMATIC1111 starting (log: $LOG_DIR/a1111.log) — takes ~60s to load"
  cd "$ROOT_DIR"
fi

# ── Chatterbox TTS ────────────────────────────────────────────────────────────
log "Chatterbox TTS (port 8020)"
if [[ ! -d "$WORKSPACE/.venv_chatterbox" ]]; then
  die ".venv_chatterbox not found — run setup_gpu.sh first"
fi

if curl -sf http://localhost:8020/health >/dev/null 2>&1; then
  ok "Chatterbox TTS already responding"
else
  cd "$ROOT_DIR"
  nohup "$WORKSPACE/.venv_chatterbox/bin/uvicorn" services.chatterbox_server:app \
    --host 0.0.0.0 --port 8020 >"$LOG_DIR/chatterbox.log" 2>&1 &
  ok "Chatterbox TTS starting (log: $LOG_DIR/chatterbox.log)"
fi

# ── Wan 2.2 ───────────────────────────────────────────────────────────────────
log "Wan2.2 image-to-video (port 8030)"
if [[ ! -d "$WORKSPACE/.venv_wan" ]]; then
  die ".venv_wan not found — run setup_gpu.sh first"
fi

if curl -sf http://localhost:8030/health >/dev/null 2>&1; then
  ok "Wan2.2 already responding"
else
  cd "$ROOT_DIR"
  WAN_DIR="$WORKSPACE/Wan2.2" WAN_MODEL_DIR="$WORKSPACE/Wan2.2-I2V-A14B" \
  nohup "$WORKSPACE/.venv_wan/bin/uvicorn" services.wan_server:app \
    --host 0.0.0.0 --port 8030 >"$LOG_DIR/wan.log" 2>&1 &
  ok "Wan2.2 starting (log: $LOG_DIR/wan.log)"
fi

# ── MuseTalk ──────────────────────────────────────────────────────────────────
log "MuseTalk lip-sync (port 8040)"
if [[ ! -d "$WORKSPACE/.venv_musetalk" ]]; then
  die ".venv_musetalk not found — run setup_gpu.sh first"
fi

if curl -sf http://localhost:8040/health >/dev/null 2>&1; then
  ok "MuseTalk already responding"
else
  cd "$ROOT_DIR"
  # PYTHONPATH required: scripts/inference.py lives under scripts/ but imports
  # musetalk package from the repo root — Python does not auto-add CWD to sys.path.
  MUSETALK_DIR="$WORKSPACE/MuseTalk" \
  PYTHONPATH="$WORKSPACE/MuseTalk${PYTHONPATH:+:$PYTHONPATH}" \
  nohup "$WORKSPACE/.venv_musetalk/bin/uvicorn" services.musetalk_server:app \
    --host 0.0.0.0 --port 8040 >"$LOG_DIR/musetalk.log" 2>&1 &
  ok "MuseTalk starting (log: $LOG_DIR/musetalk.log)"
fi

# ── Wait + verify all services ────────────────────────────────────────────────
printf '\nWaiting for all services to become ready...\n'

FAILED=0
wait_for "Ollama"        "http://localhost:11434/api/tags"        20 || FAILED=1
wait_for "AUTOMATIC1111" "http://localhost:7860/sdapi/v1/sd-models" 30 || FAILED=1
wait_for "Chatterbox"    "http://localhost:8020/health"           15 || FAILED=1
wait_for "Wan2.2"        "http://localhost:8030/health"           15 || FAILED=1
wait_for "MuseTalk"      "http://localhost:8040/health"           15 || FAILED=1

printf '\n'
cd "$ROOT_DIR"
if [[ -f ".venv/bin/python" ]]; then
  .venv/bin/python -m scripts.check_runtime_readiness --allow-missing-services
else
  python3 -m scripts.check_runtime_readiness --allow-missing-services
fi

if [[ "$FAILED" == "1" ]]; then
  warn "One or more services did not respond in time. Check logs in $LOG_DIR/"
  exit 1
fi
