#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: This script requires bash, not sh. Run: bash scripts/setup_gpu.sh" >&2
  exit 1
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────
DRY_RUN=0
SKIP_SYSTEM_DEPS=0
SKIP_PYTHON_DEPS=0
SKIP_SERVICES=0
SKIP_OLLAMA=0
SKIP_A1111=0
SKIP_CHATTERBOX=0
SKIP_WAN=0
SKIP_MUSETALK=0
SKIP_ENV_FILE=0
ALLOW_MISSING_SERVICES=0
CODE_TEST=0
NO_VENV=0
WITH_STORAGE=0
TIMEOUT="${VIDEO_ME_READINESS_TIMEOUT:-3.0}"
PYTHON_BIN="${PYTHON_BIN:-}"
WORKSPACE="${WORKSPACE:-/workspace}"
HF_TOKEN="${HF_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_gpu.sh [options]

Full GPU-machine setup for video_me on RunPod (or any Ubuntu+CUDA box):
  1. Verify CUDA / GPU
  2. Install system packages  (ffmpeg, yt-dlp, curl, git)
  3. Create Python venv + install runtime extras
  4. Install Ollama + pull qwen2.5:7b and llava:7b
  5. Clone + configure AUTOMATIC1111 + download SD 1.5 model
  6. Write .env with GPU-correct settings
  7. Write scripts/start_services.sh (start all Track D services in one command)
  8. Run runtime readiness check

Network volume (RunPod):
  Models and service repos are placed under WORKSPACE (default /workspace) so
  they survive pod restarts when a network volume is mounted there.

Options:
  --workspace PATH          Network volume / persistent root  [default: /workspace]
  --dry-run                 Print commands without executing them
  --skip-system-deps        Skip apt-get installs
  --skip-python-deps        Skip pip install
  --skip-ollama             Skip Ollama install + model pull
  --skip-a1111              Skip AUTOMATIC1111 setup
  --skip-chatterbox         Skip Chatterbox TTS install
  --skip-wan                Skip Wan2.2 install + model download
  --skip-musetalk           Skip MuseTalk install + weight download
  --skip-env-file           Do not write .env (keep existing)
  --skip-services           Skip service HTTP checks in final readiness
  --allow-missing-services  Treat missing services as warnings not failures
  --with-storage            Also install boto3 + psycopg (for S3 / Postgres storage)
  --code-test               Accept TEST-ONLY placeholder LoRAs for smoke tests
  --no-venv                 Use current Python instead of creating .venv
  --python PATH             Python executable to bootstrap with
  --timeout SECONDS         Readiness HTTP timeout  [default: 3.0]
  --hf-token TOKEN          HuggingFace token for model downloads (or set HF_TOKEN env)
  -h, --help                Show this help

Quick commands:
  # Full first-time setup on a fresh RunPod pod:
  bash scripts/setup_gpu.sh

  # Dry-run to preview all steps:
  bash scripts/setup_gpu.sh --dry-run

  # Code-only smoke test (no GPU, no services):
  bash scripts/setup_gpu.sh --code-test --skip-services --skip-ollama --skip-a1111

  # After pod restart (services not running, everything already installed):
  bash scripts/start_services.sh
  python -m scripts.check_runtime_readiness
EOF
}

# ── Helpers ─────────────────────────────────────────────────────────────────
log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[0;32m  ✓ %s\033[0m\n' "$*"; }
warn(){ printf '\033[0;33m  ! %s\033[0m\n' "$*"; }
die() { printf '\033[0;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

run() {
  printf '  \033[0;36m+'; printf ' %q' "$@"; printf '\033[0m\n'
  if [[ "$DRY_RUN" == "0" ]]; then "$@"; fi
}

need_cmd() { command -v "$1" >/dev/null 2>&1; }

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)          [[ $# -ge 2 ]] || die "--workspace requires a path"; WORKSPACE="$2"; shift 2 ;;
    --dry-run)            DRY_RUN=1; shift ;;
    --skip-system-deps)   SKIP_SYSTEM_DEPS=1; shift ;;
    --skip-python-deps)   SKIP_PYTHON_DEPS=1; shift ;;
    --skip-ollama)        SKIP_OLLAMA=1; shift ;;
    --skip-a1111)         SKIP_A1111=1; shift ;;
    --skip-chatterbox)    SKIP_CHATTERBOX=1; shift ;;
    --skip-wan)           SKIP_WAN=1; shift ;;
    --skip-musetalk)      SKIP_MUSETALK=1; shift ;;
    --skip-env-file)      SKIP_ENV_FILE=1; shift ;;
    --skip-services)      SKIP_SERVICES=1; shift ;;
    --allow-missing-services) ALLOW_MISSING_SERVICES=1; shift ;;
    --with-storage)       WITH_STORAGE=1; shift ;;
    --code-test)          CODE_TEST=1; shift ;;
    --no-venv)            NO_VENV=1; shift ;;
    --python)             [[ $# -ge 2 ]] || die "--python requires a path"; PYTHON_BIN="$2"; shift 2 ;;
    --timeout)            [[ $# -ge 2 ]] || die "--timeout requires seconds"; TIMEOUT="$2"; shift 2 ;;
    --hf-token)           [[ $# -ge 2 ]] || die "--hf-token requires a token"; HF_TOKEN="$2"; shift 2 ;;
    -h|--help)            usage; exit 0 ;;
    *)                    die "Unknown option: $1" ;;
  esac
done

# ── Step 1: GPU / CUDA check ─────────────────────────────────────────────────
check_cuda() {
  log "Checking GPU / CUDA"
  if need_cmd nvidia-smi; then
    if [[ "$DRY_RUN" == "0" ]]; then
      GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
      ok "NVIDIA GPU detected: $GPU_INFO"
    else
      ok "nvidia-smi available (dry run — skipping query)"
    fi
  else
    warn "nvidia-smi not found. Continuing — whisper will fall back to CPU."
    warn "If this is a GPU pod, check that the NVIDIA driver is installed."
  fi

  if need_cmd nvcc; then
    if [[ "$DRY_RUN" == "0" ]]; then
      CUDA_VER=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
      ok "CUDA toolkit: $CUDA_VER"
    fi
  else
    warn "nvcc not on PATH — CUDA toolkit may not be installed. Services that need it will fail."
  fi
}

# ── Step 2: System packages ──────────────────────────────────────────────────
install_system_deps() {
  log "Installing system packages (ffmpeg, yt-dlp, curl, git)"

  if need_cmd apt-get; then
    run sudo apt-get update -qq
    run sudo apt-get install -y --no-install-recommends ffmpeg curl git
    ok "apt packages installed"

    # yt-dlp system binary (in addition to the Python package)
    if ! need_cmd yt-dlp; then
      run sudo curl -fsSL "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp" \
          -o /usr/local/bin/yt-dlp
      run sudo chmod +x /usr/local/bin/yt-dlp
      ok "yt-dlp binary installed to /usr/local/bin/yt-dlp"
    else
      ok "yt-dlp already on PATH: $(command -v yt-dlp)"
    fi
    return
  fi

  if need_cmd brew; then
    run brew install ffmpeg yt-dlp
    return
  fi

  die "No supported package manager (apt-get or brew). Install ffmpeg and yt-dlp manually."
}

# ── Step 3: Python env ───────────────────────────────────────────────────────
select_python() {
  if [[ -n "$PYTHON_BIN" ]]; then printf '%s\n' "$PYTHON_BIN"; return; fi
  if need_cmd python3; then command -v python3; return; fi
  if need_cmd python;  then command -v python;  return; fi
  die "python3/python not found. Install Python 3.11+ first."
}

setup_python_env() {
  local bootstrap_python
  bootstrap_python="$(select_python)"

  if [[ "$NO_VENV" == "1" ]]; then
    PYTHON_BIN="$bootstrap_python"
    return
  fi

  if [[ ! -d ".venv" ]]; then
    log "Creating .venv"
    run "$bootstrap_python" -m venv .venv
  else
    ok ".venv already exists"
  fi

  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  export PATH="$ROOT_DIR/.venv/bin:$PATH"
}

install_python_deps() {
  log "Installing Python runtime dependencies"
  run "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

  local extras="ingest,transcribe,llm,render"
  if [[ "$WITH_STORAGE" == "1" ]]; then
    extras="$extras,services"
    ok "Including storage extras (boto3, psycopg) for S3/Postgres"
  fi

  run "$PYTHON_BIN" -m pip install -e ".[$extras]"
}

# ── Step 4: Ollama ───────────────────────────────────────────────────────────
setup_ollama() {
  log "Setting up Ollama"

  if ! need_cmd ollama; then
    run bash -c 'curl -fsSL https://ollama.ai/install.sh | sh'
  else
    ok "Ollama already installed: $(command -v ollama)"
  fi

  # Store models on the network volume so they persist across pod restarts
  export OLLAMA_MODELS="$WORKSPACE/ollama"
  if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$OLLAMA_MODELS"
  fi

  # Start Ollama server in background for model pulls, stop it after
  log "Pulling Ollama models (qwen2.5:7b + llava:7b)"
  if [[ "$DRY_RUN" == "0" ]]; then
    OLLAMA_MODELS="$WORKSPACE/ollama" ollama serve &>/tmp/ollama_setup.log &
    OLLAMA_PID=$!
    sleep 5  # give server time to start

    ollama pull qwen2.5:7b  || warn "qwen2.5:7b pull failed — retry manually: ollama pull qwen2.5:7b"
    ollama pull llava:7b     || warn "llava:7b pull failed — retry manually: ollama pull llava:7b"

    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    ok "Ollama models pulled to $WORKSPACE/ollama"
  else
    run ollama pull qwen2.5:7b
    run ollama pull llava:7b
  fi
}

# ── Step 5: AUTOMATIC1111 ────────────────────────────────────────────────────
setup_a1111() {
  log "Setting up AUTOMATIC1111 Stable Diffusion Web UI"

  local a1111_dir="$WORKSPACE/stable-diffusion-webui"

  if [[ ! -d "$a1111_dir" ]]; then
    run git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui "$a1111_dir"
  else
    ok "AUTOMATIC1111 already cloned at $a1111_dir"
    run git -C "$a1111_dir" pull --ff-only || warn "git pull failed — continuing with existing checkout"
  fi

  # Download SD 1.5 base model
  local model_dir="$a1111_dir/models/Stable-diffusion"
  local model_file="$model_dir/v1-5-pruned-emaonly.safetensors"

  if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$model_dir"; fi

  if [[ ! -f "$model_file" ]]; then
    log "Downloading SD v1.5 model (~4 GB) — this takes a few minutes"
    local hf_header=""
    if [[ -n "$HF_TOKEN" ]]; then
      hf_header="--header Authorization: Bearer $HF_TOKEN"
    fi
    # shellcheck disable=SC2086
    run curl -fL $hf_header \
      "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" \
      -o "$model_file"
    ok "SD 1.5 model downloaded to $model_file"
  else
    ok "SD 1.5 model already at $model_file"
  fi

  # Symlink LoRA directory into A1111's expected location
  local lora_link="$a1111_dir/models/Lora"
  local lora_src="$ROOT_DIR/loras"
  if [[ ! -e "$lora_link" ]]; then
    run ln -s "$lora_src" "$lora_link"
    ok "Symlinked $lora_src → $lora_link"
  else
    ok "LoRA symlink already exists at $lora_link"
  fi
}

# ── Step 6a: Chatterbox TTS ──────────────────────────────────────────────────
setup_chatterbox() {
  log "Setting up Chatterbox TTS (port 8020)"
  # Chatterbox is a pip package — installs into the main venv
  run "$PYTHON_BIN" -m pip install chatterbox-tts torchaudio fastapi uvicorn
  ok "Chatterbox TTS installed"
}

# ── Step 6b: Wan 2.2 ─────────────────────────────────────────────────────────
setup_wan() {
  log "Setting up Wan2.2 image-to-video (port 8030)"

  local wan_dir="$WORKSPACE/Wan2.2"
  local wan_model_dir="$WORKSPACE/Wan2.2-I2V-A14B"

  if [[ ! -d "$wan_dir" ]]; then
    run git clone https://github.com/Wan-Video/Wan2.2.git "$wan_dir"
  else
    ok "Wan2.2 already cloned at $wan_dir"
    run git -C "$wan_dir" pull --ff-only || warn "git pull failed — continuing"
  fi

  run "$PYTHON_BIN" -m pip install -r "$wan_dir/requirements.txt"
  run "$PYTHON_BIN" -m pip install "huggingface_hub[cli]" fastapi uvicorn

  if [[ ! -d "$wan_model_dir" ]]; then
    log "Downloading Wan2.2-I2V-A14B model (~30 GB) — this will take a while"
    local hf_header=""
    if [[ -n "$HF_TOKEN" ]]; then hf_header="--token $HF_TOKEN"; fi
    # shellcheck disable=SC2086
    run huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
        --local-dir "$wan_model_dir" $hf_header
    ok "Wan2.2-I2V-A14B downloaded to $wan_model_dir"
  else
    ok "Wan2.2 model already at $wan_model_dir"
  fi
}

# ── Step 6c: MuseTalk ─────────────────────────────────────────────────────────
setup_musetalk() {
  log "Setting up MuseTalk lip-sync (port 8040)"
  # MuseTalk requires Python 3.10 + CUDA 11.8. If the current env is Python 3.11+,
  # create a separate conda env and install there, then point start_services.sh at it.

  local musetalk_dir="$WORKSPACE/MuseTalk"

  if [[ ! -d "$musetalk_dir" ]]; then
    run git clone https://github.com/TMElyralab/MuseTalk.git "$musetalk_dir"
  else
    ok "MuseTalk already cloned at $musetalk_dir"
    run git -C "$musetalk_dir" pull --ff-only || warn "git pull failed — continuing"
  fi

  if need_cmd conda; then
    log "Creating dedicated MuseTalk conda env (Python 3.10, CUDA 11.8)"
    run conda create -n MuseTalk python=3.10 -y || true
    run conda run -n MuseTalk pip install \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118
    run conda run -n MuseTalk pip install -r "$musetalk_dir/requirements.txt"
    run conda run -n MuseTalk pip install --no-cache-dir -U openmim
    run conda run -n MuseTalk mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"
    run conda run -n MuseTalk pip install fastapi uvicorn
    ok "MuseTalk conda env ready"

    log "Downloading MuseTalk model weights"
    run conda run -n MuseTalk bash -c "cd $musetalk_dir && bash download_weights.sh" \
        || warn "Weight download failed — run manually: cd $musetalk_dir && bash download_weights.sh"
  else
    warn "conda not found — installing MuseTalk into the main venv (may conflict with other deps)"
    warn "Recommended: install conda/miniconda, then rerun with --skip-chatterbox --skip-wan"
    run "$PYTHON_BIN" -m pip install \
        torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118
    run "$PYTHON_BIN" -m pip install -r "$musetalk_dir/requirements.txt"
    run "$PYTHON_BIN" -m pip install --no-cache-dir -U openmim
    run "$PYTHON_BIN" -m python -m mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"
    run "$PYTHON_BIN" -m pip install fastapi uvicorn
  fi
}

# ── Step 7: .env file ────────────────────────────────────────────────────────
write_env_file() {
  log "Writing .env with GPU settings"

  local env_file="$ROOT_DIR/.env"
  if [[ -f "$env_file" ]]; then
    warn ".env already exists — backing up to .env.bak"
    if [[ "$DRY_RUN" == "0" ]]; then cp "$env_file" "$env_file.bak"; fi
  fi

  if [[ "$DRY_RUN" == "0" ]]; then
    cat > "$env_file" <<EOF
# video_me GPU runtime settings — generated by setup_gpu.sh
# Edit as needed; do not commit this file.

VIDEO_ME_WHISPER_DEVICE=cuda
VIDEO_ME_WHISPER_COMPUTE_TYPE=float16

VIDEO_ME_DATA_DIR=$WORKSPACE/video_me/data
VIDEO_ME_REVIEW_DIR=$WORKSPACE/video_me/review
VIDEO_ME_LORA_DIR=$ROOT_DIR/loras
VIDEO_ME_VOICE_DIR=$ROOT_DIR/voices

VIDEO_ME_LLM_MODEL=qwen2.5:7b
VIDEO_ME_LLM_BASE_URL=http://localhost:11434/v1
VIDEO_ME_CRITIQUE_MODEL=llava:7b
VIDEO_ME_CRITIQUE_BASE_URL=http://localhost:11434/v1

VIDEO_ME_SD_BASE_URL=http://localhost:7860
VIDEO_ME_TTS_BASE_URL=http://localhost:8020
VIDEO_ME_WAN_BASE_URL=http://localhost:8030
VIDEO_ME_LIPSYNC_BASE_URL=http://localhost:8040

# Uncomment for PostgreSQL job store + S3 artifact store:
# VIDEO_ME_JOB_STORE=postgres
# VIDEO_ME_DATABASE_URL=postgresql://user:pass@localhost/video_me
# VIDEO_ME_ARTIFACT_STORE=s3
# VIDEO_ME_S3_BUCKET=video-me-artifacts
EOF
    ok ".env written to $env_file"
  else
    ok "[dry run] would write .env to $env_file"
  fi
}

# ── Step 8: start_services.sh ────────────────────────────────────────────────
write_start_services() {
  log "Writing scripts/start_services.sh"
  local start_script="$ROOT_DIR/scripts/start_services.sh"

  if [[ "$DRY_RUN" == "0" ]]; then
    cat > "$start_script" <<STARTEOF
#!/usr/bin/env bash
# Start all Track D services for the video_me pipeline.
# Run this after every pod restart.
set -euo pipefail

ROOT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="\${WORKSPACE:-$WORKSPACE}"
LOG_DIR="\$WORKSPACE/logs"
mkdir -p "\$LOG_DIR"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "\$*"; }
ok()  { printf '\033[0;32m  ✓ %s\033[0m\n' "\$*"; }
warn(){ printf '\033[0;33m  ! %s\033[0m\n' "\$*"; }

# ── Ollama ────────────────────────────────────────────────────────────────────
log "Starting Ollama (LLM + VLM critique, port 11434)"
if pgrep -x ollama >/dev/null 2>&1; then
  ok "Ollama already running"
else
  OLLAMA_MODELS="\$WORKSPACE/ollama" ollama serve >"\$LOG_DIR/ollama.log" 2>&1 &
  ok "Ollama started (log: \$LOG_DIR/ollama.log)"
fi

# ── AUTOMATIC1111 ─────────────────────────────────────────────────────────────
log "Starting AUTOMATIC1111 Stable Diffusion (port 7860)"
A1111_DIR="\$WORKSPACE/stable-diffusion-webui"
if ! curl -sf http://localhost:7860/sdapi/v1/sd-models >/dev/null 2>&1; then
  cd "\$A1111_DIR"
  nohup bash webui.sh \
    --api \
    --listen \
    --port 7860 \
    --nowebui \
    --skip-torch-cuda-test \
    >"\$LOG_DIR/a1111.log" 2>&1 &
  ok "AUTOMATIC1111 starting (log: \$LOG_DIR/a1111.log) — takes ~60s to load"
else
  ok "AUTOMATIC1111 already responding"
fi
cd "\$ROOT_DIR"

# ── Chatterbox TTS ────────────────────────────────────────────────────────────
log "Starting Chatterbox TTS (port 8020)"
if ! curl -sf http://localhost:8020/health >/dev/null 2>&1; then
  cd "\$ROOT_DIR"
  if [[ -f ".venv/bin/python" ]]; then UVICORN=".venv/bin/uvicorn"; else UVICORN="uvicorn"; fi
  nohup "\$UVICORN" services.chatterbox_server:app \
    --host 0.0.0.0 --port 8020 >"\$LOG_DIR/chatterbox.log" 2>&1 &
  ok "Chatterbox TTS starting (log: \$LOG_DIR/chatterbox.log)"
else
  ok "Chatterbox TTS already responding"
fi

# ── Wan 2.2 ───────────────────────────────────────────────────────────────────
log "Starting Wan2.2 image-to-video (port 8030)"
if ! curl -sf http://localhost:8030/health >/dev/null 2>&1; then
  cd "\$ROOT_DIR"
  if [[ -f ".venv/bin/python" ]]; then UVICORN=".venv/bin/uvicorn"; else UVICORN="uvicorn"; fi
  WAN_DIR="\$WORKSPACE/Wan2.2" WAN_MODEL_DIR="\$WORKSPACE/Wan2.2-I2V-A14B" \
  nohup "\$UVICORN" services.wan_server:app \
    --host 0.0.0.0 --port 8030 >"\$LOG_DIR/wan.log" 2>&1 &
  ok "Wan2.2 starting (log: \$LOG_DIR/wan.log)"
else
  ok "Wan2.2 already responding"
fi

# ── MuseTalk ──────────────────────────────────────────────────────────────────
log "Starting MuseTalk lip-sync (port 8040)"
if ! curl -sf http://localhost:8040/health >/dev/null 2>&1; then
  cd "\$ROOT_DIR"
  # Use the MuseTalk conda env if it exists (needed for Python 3.10 + CUDA 11.8)
  if conda env list 2>/dev/null | grep -q "MuseTalk"; then
    MUSETALK_PYTHON="\$(conda run -n MuseTalk which python)"
    MUSETALK_UVICORN="\$(conda run -n MuseTalk which uvicorn)"
  elif [[ -f ".venv/bin/uvicorn" ]]; then
    MUSETALK_UVICORN=".venv/bin/uvicorn"
  else
    MUSETALK_UVICORN="uvicorn"
  fi
  MUSETALK_DIR="\$WORKSPACE/MuseTalk" \
  nohup "\$MUSETALK_UVICORN" services.musetalk_server:app \
    --host 0.0.0.0 --port 8040 >"\$LOG_DIR/musetalk.log" 2>&1 &
  ok "MuseTalk starting (log: \$LOG_DIR/musetalk.log)"
else
  ok "MuseTalk already responding"
fi

# ── Health check ──────────────────────────────────────────────────────────────
printf '\nWaiting 20s for services to start...\n'
sleep 20

cd "\$ROOT_DIR"
if [[ -f ".venv/bin/python" ]]; then
  .venv/bin/python -m scripts.check_runtime_readiness --allow-missing-services
else
  python -m scripts.check_runtime_readiness --allow-missing-services
fi
STARTEOF
    chmod +x "$start_script"
    ok "start_services.sh written and made executable"
  else
    ok "[dry run] would write scripts/start_services.sh"
  fi
}

# ── Step 9: Readiness check ──────────────────────────────────────────────────
run_readiness() {
  local args=(--timeout "$TIMEOUT")
  if [[ "$CODE_TEST" == "1" ]]; then
    export VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true
    args+=(--code-test)
  fi
  if [[ "$SKIP_SERVICES" == "1" ]]; then
    args+=(--skip-services)
  fi
  if [[ "$ALLOW_MISSING_SERVICES" == "1" ]]; then
    args+=(--allow-missing-services)
  fi

  log "Running runtime readiness check"
  run "$PYTHON_BIN" -m scripts.check_runtime_readiness "${args[@]}"
}

# ── Main ─────────────────────────────────────────────────────────────────────
log "video_me GPU setup  (workspace: $WORKSPACE)"

check_cuda

if [[ "$SKIP_SYSTEM_DEPS" == "0" ]]; then
  install_system_deps
fi

setup_python_env

if [[ "$SKIP_PYTHON_DEPS" == "0" ]]; then
  install_python_deps
fi

if [[ "$SKIP_OLLAMA" == "0" ]]; then
  setup_ollama
fi

if [[ "$SKIP_A1111" == "0" ]]; then
  setup_a1111
fi

if [[ "$SKIP_CHATTERBOX" == "0" ]]; then
  setup_chatterbox
fi

if [[ "$SKIP_WAN" == "0" ]]; then
  setup_wan
fi

if [[ "$SKIP_MUSETALK" == "0" ]]; then
  setup_musetalk
fi

if [[ "$SKIP_ENV_FILE" == "0" ]]; then
  write_env_file
fi

write_start_services

# Services aren't started by setup — run readiness with allow-missing so setup
# succeeds even though services haven't been started yet.
if [[ "$ALLOW_MISSING_SERVICES" == "0" && "$SKIP_SERVICES" == "0" ]]; then
  ALLOW_MISSING_SERVICES=1
fi
run_readiness

log "Setup complete"
printf '\n'
printf '  Next steps:\n'
printf '    1. bash scripts/start_services.sh     # start all services\n'
printf '    2. python -m scripts.check_runtime_readiness   # verify all PASS\n'
printf '    3. python -m scripts.check_track_b    # verify LoRA + voice files\n'
printf '    4. python -m pytest -q                # confirm 313 tests pass\n'
printf '\n'
printf '  Then run the pipeline:\n'
printf '    python -c "import asyncio; from core.config import load_app_config; from core.workflow import run_with_critique; config = load_app_config(); job = asyncio.run(run_with_critique(source_url='"'"'YOUR_URL'"'"', rights_cleared=True, app_config=config)); print(job.status)"\n'
printf '\n'
