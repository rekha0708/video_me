#!/usr/bin/env bash
# Start the Track D services selected by the current video_me adapter stack.
# Run this after every pod restart.
#
# Required services (default stack):
#   Ollama        (port 11434) — qwen3.6:35b for all LLM + VLM stages
#   Wan 2.2 S2V   (port 8031)  — audio-conditioned singing video generation
#   Fish Audio S2 (port 8025)  — TTS, EN + HI + 80 languages
#
# Optional/fallback services are started only when selected in .env:
#   AUTOMATIC1111 (port 7860)  — SD 1.5 render_character fallback (RENDER_ADAPTER=a1111)
#   ComfyUI       (port 8188)  — LTX-2.3 legacy video gen / Flux fallback
#   Wan 2.2 I2V   (port 8030)  — image-to-video fallback (VIDEO_ADAPTER=wan)
#   LatentSync    (port 8041)  — lip-sync repair for Wan I2V
#   MuseTalk      (port 8040)  — legacy lip-sync fallback
#   Chatterbox    (port 8020)  — EN-only TTS fallback (TTS_ADAPTER=chatterbox)
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

# ── Claude Code memory symlink ────────────────────────────────────────────────
# /root/ is wiped on pod restart. Recreate the symlink so Claude Code memory
# points at the persistent copy in /workspace/video_me/.claude/memory/.
CLAUDE_MEM_TARGET="$ROOT_DIR/.claude/memory"
CLAUDE_MEM_LINK="/root/.claude/projects/-workspace/memory"
if [[ ! -L "$CLAUDE_MEM_LINK" || "$(readlink "$CLAUDE_MEM_LINK")" != "$CLAUDE_MEM_TARGET" ]]; then
  mkdir -p /root/.claude/projects/-workspace
  rm -rf "$CLAUDE_MEM_LINK"
  ln -s "$CLAUDE_MEM_TARGET" "$CLAUDE_MEM_LINK"
  ok "Claude Code memory symlink restored → $CLAUDE_MEM_TARGET"
else
  ok "Claude Code memory symlink OK"
fi

wait_for() {
  local name="$1" url="$2" tries="${3:-15}" body_pattern="${4:-}"
  for i in $(seq 1 "$tries"); do
    local body
    if body="$(curl -sf "$url" 2>/dev/null)"; then
      # A bare 2xx only proves the HTTP server is up. Services with deferred
      # model loading (e.g. Fish S2's /health always returns 200 with a
      # "model_loaded" field — see services/fish_s2_server.py) need the body
      # checked too, or start_services.sh reports ready before the model
      # actually finishes loading.
      if [[ -z "$body_pattern" ]] || grep -q "$body_pattern" <<<"$body"; then
        ok "$name responding at $url"
        return 0
      fi
    fi
    printf '  waiting for %s (%d/%d)...\n' "$name" "$i" "$tries"
    sleep 4
  done
  warn "$name did not respond at $url after $((tries * 4))s"
  return 1
}

# ── System tools (ffmpeg, yt-dlp) ────────────────────────────────────────────
# These are installed into base Linux and wiped on RunPod pod restart.
# Reinstall if missing — fast (~10s) since apt cache is warm on fresh pods.
log "System tools (ffmpeg, yt-dlp)"
TOOLS_MISSING=0
if ! command -v ffmpeg >/dev/null 2>&1; then
  warn "ffmpeg missing — reinstalling via apt-get..."
  apt-get install -y --no-install-recommends ffmpeg curl 2>/dev/null || \
    sudo apt-get install -y --no-install-recommends ffmpeg curl
  ok "ffmpeg reinstalled: $(ffmpeg -version 2>&1 | head -1)"
  TOOLS_MISSING=1
else
  ok "ffmpeg present: $(ffmpeg -version 2>&1 | head -1)"
fi

if ! command -v yt-dlp >/dev/null 2>&1; then
  warn "yt-dlp missing — reinstalling..."
  curl -fsSL "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp" \
    -o /usr/local/bin/yt-dlp && chmod +x /usr/local/bin/yt-dlp
  ok "yt-dlp reinstalled: $(yt-dlp --version 2>&1 | head -1)"
  TOOLS_MISSING=1
else
  ok "yt-dlp present: $(yt-dlp --version 2>&1 | head -1)"
fi

# Launches a background service; if it crashes immediately (import-time errors
# surface in ~1-2s, long model loads don't), grep its own traceback for the
# missing module, install it with the given pip, and retry once. This is a
# generic safety net for individual packages that go missing from a venv or
# system Python between pod restarts — seen so far with ComfyUI/sqlalchemy and
# Fish S2 + Wan2.2/click, but the next one won't necessarily have the same name.
launch_with_self_heal() {
  local name="$1" log_file="$2"; shift 2
  local -a pip_cmd=()
  while [[ "$1" != "--" ]]; do pip_cmd+=("$1"); shift; done
  shift # drop the --

  "$@" >"$log_file" 2>&1 &
  local pid=$!
  sleep 3
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi

  local missing
  missing="$(grep -oP "ModuleNotFoundError: No module named '\K[^']+" "$log_file" 2>/dev/null | tail -1 || true)"
  if [[ -z "$missing" ]]; then
    return 0  # crashed for some other reason — let wait_for's timeout report it
  fi

  warn "$name crashed on missing module '$missing' — installing via '${pip_cmd[*]}' and retrying once"
  "${pip_cmd[@]}" install -q "$missing" || warn "pip install $missing failed"
  "$@" >"$log_file" 2>&1 &
  pid=$!
  sleep 3
  if kill -0 "$pid" 2>/dev/null; then
    ok "$name recovered after installing $missing"
  else
    warn "$name still failing after installing $missing — check $log_file"
  fi
}

# Load adapter selections from .env when present. The setup script writes plain
# shell-compatible KEY=value lines; if an operator points a value at a remote
# service URL, this script still starts the local default service process for
# the selected adapter stack.
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

RENDER_ADAPTER="${VIDEO_ME_RENDER_ADAPTER:-musubi_flux}"
VIDEO_ADAPTER="${VIDEO_ME_VIDEO_ADAPTER:-wan_s2v}"
TTS_ADAPTER="${VIDEO_ME_TTS_ADAPTER:-fish_s2}"
LIPSYNC_ADAPTER="${VIDEO_ME_LIPSYNC_ADAPTER:-latentsync}"

NEED_COMFYUI=0
NEED_A1111=0
NEED_FISH_S2=0
NEED_CHATTERBOX=0
NEED_WAN_S2V=0
NEED_WAN_I2V=0
NEED_LATENTSYNC=0
NEED_MUSETALK=0

case "$RENDER_ADAPTER" in
  comfyui_flux) NEED_COMFYUI=1 ;;
  a1111) NEED_A1111=1 ;;
esac

case "$VIDEO_ADAPTER" in
  ltx) NEED_COMFYUI=1 ;;
  wan_s2v) NEED_WAN_S2V=1 ;;
  wan)
    NEED_WAN_I2V=1
    if [[ "$LIPSYNC_ADAPTER" == "latentsync" ]]; then
      NEED_LATENTSYNC=1
    else
      NEED_MUSETALK=1
    fi
    ;;
esac

case "$TTS_ADAPTER" in
  fish_s2) NEED_FISH_S2=1 ;;
  chatterbox) NEED_CHATTERBOX=1 ;;
esac

log "Configured adapters: render=$RENDER_ADAPTER video=$VIDEO_ADAPTER tts=$TTS_ADAPTER lipsync=$LIPSYNC_ADAPTER"

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

# ── ComfyUI ───────────────────────────────────────────────────────────────────
# ComfyUI serves the legacy LTX video path and the comfyui_flux render fallback.
if [[ "$NEED_COMFYUI" == "1" ]]; then
  log "ComfyUI (legacy LTX + Flux fallback, port 8188)"
  COMFYUI_DIR="$WORKSPACE/ComfyUI"
  if [[ ! -d "$COMFYUI_DIR" ]]; then
    warn "ComfyUI not found at $COMFYUI_DIR — run setup_gpu.sh first or install manually"
  elif curl -sf http://localhost:8188/ >/dev/null 2>&1; then
    ok "ComfyUI already responding"
  else
    launch_with_self_heal "ComfyUI" "$LOG_DIR/comfyui.log" "$ROOT_DIR/.venv/bin/python3" -m pip -- \
      nohup "$ROOT_DIR/.venv/bin/python3" "$COMFYUI_DIR/main.py" --listen 0.0.0.0 --port 8188
    ok "ComfyUI starting (log: $LOG_DIR/comfyui.log) — takes ~30s to load"
  fi
else
  ok "ComfyUI not selected by current adapters — skipping"
fi

# ── musubi-tuner (Flux 2.0 image gen for render_character) ───────────────────
# Default image backend — no server, no port. musubi_flux_adapter.py invokes
# this venv's python directly for each render, so there's nothing to "start"
# here; just verify the venv can still import musubi_tuner and self-heal if a
# package went missing (same individual-package-vanishes issue we've hit with
# ComfyUI/sqlalchemy and Fish S2 + Wan2.2/click — happens even to /workspace
# venvs, not just system Python).
if [[ "$RENDER_ADAPTER" == "musubi_flux" ]]; then
  log "musubi-tuner (Flux 2.0 image gen, subprocess — no port)"
  MUSUBI_VENV="$WORKSPACE/.venv_musubi"
  MUSUBI_DIR="$WORKSPACE/musubi-tuner"
  if [[ ! -d "$MUSUBI_VENV" ]]; then
    warn "musubi-tuner venv not found at $MUSUBI_VENV — run setup_gpu.sh first"
  else
    if "$MUSUBI_VENV/bin/python" -c "from musubi_tuner.flux_2 import flux2_utils" >/dev/null 2>&1; then
      ok "musubi-tuner importable in $MUSUBI_VENV"
    else
      warn "musubi-tuner not importable in $MUSUBI_VENV — reinstalling"
      "$MUSUBI_VENV/bin/pip" install -e "$MUSUBI_DIR" -q || warn "musubi-tuner reinstall failed"
      if "$MUSUBI_VENV/bin/python" -c "from musubi_tuner.flux_2 import flux2_utils" >/dev/null 2>&1; then
        ok "musubi-tuner recovered after reinstall"
      else
        warn "musubi-tuner still not importable after reinstall — check manually"
      fi
    fi
  fi
else
  ok "musubi-tuner not selected by current render adapter — skipping"
fi

# ── Fish Audio S2 TTS ─────────────────────────────────────────────────────────
# Default TTS: EN + HI + 80 languages, voice cloning from reference WAV.
if [[ "$NEED_FISH_S2" == "1" ]]; then
  log "Fish Audio S2 TTS (port 8025)"
  FISH_DIR="$WORKSPACE/fish-speech"
  FISH_VENV="$WORKSPACE/.venv_fish_s2"
  if [[ ! -d "$FISH_VENV" ]]; then
    warn ".venv_fish_s2 not found at $FISH_VENV — run setup_gpu.sh first"
  elif curl -sf http://localhost:8025/health >/dev/null 2>&1; then
    ok "Fish Audio S2 already responding"
  else
    cd "$ROOT_DIR"
    FISH_SPEECH_DIR="$FISH_DIR" \
    launch_with_self_heal "Fish Audio S2" "$LOG_DIR/fish_s2.log" "$FISH_VENV/bin/pip" -- \
      nohup "$FISH_VENV/bin/uvicorn" services.fish_s2_server:app \
      --host 0.0.0.0 --port 8025
    ok "Fish Audio S2 starting (log: $LOG_DIR/fish_s2.log)"
  fi
else
  ok "Fish Audio S2 not selected by current TTS adapter — skipping"
fi

# ── AUTOMATIC1111 (fallback only — skip if not installed) ─────────────────────
if [[ "$NEED_A1111" == "1" ]]; then
  log "AUTOMATIC1111 Stable Diffusion (port 7860, fallback for RENDER_ADAPTER=a1111)"
  A1111_DIR="$WORKSPACE/stable-diffusion-webui"
  if [[ ! -d "$A1111_DIR" ]]; then
    warn "AUTOMATIC1111 not found — run setup_gpu.sh --with-a1111 first"
  elif curl -sf http://localhost:7860/sdapi/v1/sd-models >/dev/null 2>&1; then
    ok "AUTOMATIC1111 already responding"
  else
    cd "$A1111_DIR"
    A1111_SITE_PACKAGES=""
    for candidate in "$A1111_DIR"/venv/lib/python*/site-packages; do
      [[ -d "$candidate" ]] && A1111_SITE_PACKAGES="$candidate" && break
    done
    A1111_TAMING_REPO="$A1111_DIR/repositories/taming-transformers"
    if [[ -n "$A1111_SITE_PACKAGES" && -d "$A1111_TAMING_REPO" ]]; then
      printf '%s\n' "$A1111_TAMING_REPO" > "$A1111_SITE_PACKAGES/taming_transformers_repo.pth"
    fi
    (cd "$A1111_DIR/repositories/stable-diffusion-stability-ai" && \
      git checkout cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf 2>/dev/null || true)
    nohup "$A1111_DIR/venv/bin/python" "$A1111_DIR/launch.py" \
      --skip-install -f --api --listen --port 7860 --nowebui --skip-torch-cuda-test \
      >"$LOG_DIR/a1111.log" 2>&1 &
    ok "AUTOMATIC1111 starting (log: $LOG_DIR/a1111.log) — takes ~60s to load"
    cd "$ROOT_DIR"
  fi
else
  ok "AUTOMATIC1111 not selected by current render adapter — skipping"
fi

# ── Chatterbox TTS (fallback only — skip if not installed) ────────────────────
if [[ "$NEED_CHATTERBOX" == "1" ]]; then
  log "Chatterbox TTS (port 8020, fallback for TTS_ADAPTER=chatterbox)"
  if [[ ! -d "$WORKSPACE/.venv_chatterbox" ]]; then
    warn "Chatterbox venv not found — run setup_gpu.sh --with-chatterbox first"
  elif curl -sf http://localhost:8020/health >/dev/null 2>&1; then
    ok "Chatterbox TTS already responding"
  else
    cd "$ROOT_DIR"
    launch_with_self_heal "Chatterbox TTS" "$LOG_DIR/chatterbox.log" "$WORKSPACE/.venv_chatterbox/bin/pip" -- \
      nohup "$WORKSPACE/.venv_chatterbox/bin/uvicorn" services.chatterbox_server:app \
      --host 0.0.0.0 --port 8020
    ok "Chatterbox TTS starting (log: $LOG_DIR/chatterbox.log)"
  fi
else
  ok "Chatterbox TTS not selected by current TTS adapter — skipping"
fi

# ── Wan 2.2 S2V (audio-conditioned singing video) ─────────────────────────────
if [[ "$NEED_WAN_S2V" == "1" ]]; then
  log "Wan2.2 Speech-to-Video (port 8031, VIDEO_ADAPTER=wan_s2v)"
  if [[ ! -d "$WORKSPACE/.venv_wan" ]]; then
    warn "Wan2.2 venv not found — run setup_gpu.sh first"
  elif curl -sf http://localhost:8031/health >/dev/null 2>&1; then
    ok "Wan2.2 S2V already responding"
  else
    cd "$ROOT_DIR"
    WAN_DIR="$WORKSPACE/Wan2.2" WAN_S2V_MODEL_DIR="$WORKSPACE/Wan2.2-S2V-14B" \
    launch_with_self_heal "Wan2.2 S2V" "$LOG_DIR/wan_s2v.log" "$WORKSPACE/.venv_wan/bin/pip" -- \
      nohup "$WORKSPACE/.venv_wan/bin/uvicorn" services.wan_s2v_server:app \
      --host 0.0.0.0 --port 8031
    ok "Wan2.2 S2V starting (log: $LOG_DIR/wan_s2v.log)"
  fi
else
  ok "Wan2.2 S2V not selected by current video adapter — skipping"
fi

# ── Wan 2.2 I2V (fallback only — skip if not installed) ───────────────────────
if [[ "$NEED_WAN_I2V" == "1" ]]; then
  log "Wan2.2 image-to-video (port 8030, fallback for VIDEO_ADAPTER=wan)"
  if [[ ! -d "$WORKSPACE/.venv_wan" ]]; then
    warn "Wan2.2 venv not found — run setup_gpu.sh --with-wan-i2v first"
  elif curl -sf http://localhost:8030/health >/dev/null 2>&1; then
    ok "Wan2.2 already responding"
  else
    cd "$ROOT_DIR"
    # wan package is installed as an editable package in .venv_wan so WAN_DIR is
    # also used as an existence guard for the repo root.
    WAN_DIR="$WORKSPACE/Wan2.2" WAN_MODEL_DIR="$WORKSPACE/Wan2.2-I2V-A14B" \
    launch_with_self_heal "Wan2.2" "$LOG_DIR/wan.log" "$WORKSPACE/.venv_wan/bin/pip" -- \
      nohup "$WORKSPACE/.venv_wan/bin/uvicorn" services.wan_server:app \
      --host 0.0.0.0 --port 8030
    ok "Wan2.2 starting (log: $LOG_DIR/wan.log)"
  fi
else
  ok "Wan2.2 I2V not selected by current video adapter — skipping"
fi

# ── LatentSync (preferred repair for Wan I2V) ─────────────────────────────────
if [[ "$NEED_LATENTSYNC" == "1" ]]; then
  log "LatentSync lip-sync (port 8041, LIPSYNC_ADAPTER=latentsync)"
  if [[ ! -d "$WORKSPACE/.venv_latentsync" ]]; then
    warn "LatentSync venv not found — run setup_gpu.sh --with-latentsync first"
  elif curl -sf http://localhost:8041/health >/dev/null 2>&1; then
    ok "LatentSync already responding"
  else
    cd "$ROOT_DIR"
    LATENTSYNC_DIR="$WORKSPACE/LatentSync" \
    launch_with_self_heal "LatentSync" "$LOG_DIR/latentsync.log" "$WORKSPACE/.venv_latentsync/bin/pip" -- \
      nohup "$WORKSPACE/.venv_latentsync/bin/uvicorn" services.latentsync_server:app \
      --host 0.0.0.0 --port 8041
    ok "LatentSync starting (log: $LOG_DIR/latentsync.log)"
  fi
else
  ok "LatentSync not selected by current video/lip-sync adapters — skipping"
fi

# ── MuseTalk (legacy fallback only — skip if not installed) ───────────────────
if [[ "$NEED_MUSETALK" == "1" ]]; then
  log "MuseTalk lip-sync (port 8040, fallback for VIDEO_ADAPTER=wan)"
  if [[ ! -d "$WORKSPACE/.venv_musetalk" ]]; then
    warn "MuseTalk venv not found — run setup_gpu.sh --with-musetalk first"
  elif curl -sf http://localhost:8040/health >/dev/null 2>&1; then
    ok "MuseTalk already responding"
  else
    cd "$ROOT_DIR"
    MUSETALK_DIR="$WORKSPACE/MuseTalk" \
    PYTHONPATH="$WORKSPACE/MuseTalk${PYTHONPATH:+:$PYTHONPATH}" \
    launch_with_self_heal "MuseTalk" "$LOG_DIR/musetalk.log" "$WORKSPACE/.venv_musetalk/bin/pip" -- \
      nohup "$WORKSPACE/.venv_musetalk/bin/uvicorn" services.musetalk_server:app \
      --host 0.0.0.0 --port 8040
    ok "MuseTalk starting (log: $LOG_DIR/musetalk.log)"
  fi
else
  ok "MuseTalk not selected by current video/lip-sync adapters — skipping"
fi

# ── Wait + verify required services ──────────────────────────────────────────
printf '\nWaiting for required services to become ready...\n'

FAILED=0
wait_for "Ollama"         "http://localhost:11434/api/tags"          20 || FAILED=1
if [[ "$NEED_COMFYUI" == "1" ]]; then
  wait_for "ComfyUI"      "http://localhost:8188/"                   40 || FAILED=1
fi
if [[ "$NEED_A1111" == "1" ]]; then
  wait_for "AUTOMATIC1111" "http://localhost:7860/sdapi/v1/sd-models" 40 || FAILED=1
fi
if [[ "$NEED_WAN_S2V" == "1" ]]; then
  wait_for "Wan2.2 S2V"   "http://localhost:8031/health"             20 || FAILED=1
fi
if [[ "$NEED_WAN_I2V" == "1" ]]; then
  wait_for "Wan2.2 I2V"   "http://localhost:8030/health"             20 || FAILED=1
fi
if [[ "$NEED_LATENTSYNC" == "1" ]]; then
  wait_for "LatentSync"   "http://localhost:8041/health"             20 || FAILED=1
fi
if [[ "$NEED_MUSETALK" == "1" ]]; then
  wait_for "MuseTalk"     "http://localhost:8040/health"             20 || FAILED=1
fi
# 150 tries * 4s = 600s, matching core/config.py's fish_s2_load_timeout_sec —
# same real-world wait (fish_s2_server.py's lifespan() loads the model
# eagerly at startup, so /health can't even respond, let alone report
# model_loaded=true, until the full load finishes). 5 real loads timed this
# session ranged 63.8-122.6s; the previous 15-try/60s budget here was never
# enough and failed silently (this check is advisory, not job-blocking, so
# nobody noticed until the same under-budget assumption also broke real jobs).
if [[ "$NEED_FISH_S2" == "1" ]]; then
  wait_for "Fish Audio S2" "http://localhost:8025/health"            150 '"model_loaded":true' || FAILED=1
fi
if [[ "$NEED_CHATTERBOX" == "1" ]]; then
  wait_for "Chatterbox"   "http://localhost:8020/health"             20 || FAILED=1
fi

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
