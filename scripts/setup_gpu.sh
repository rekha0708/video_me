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
SKIP_COMFYUI=0
SKIP_LTX=1
SKIP_FISH_S2=0
SKIP_A1111=1
SKIP_CHATTERBOX=1
SKIP_WAN=0
SKIP_WAN_I2V=1
SKIP_LIGHTX2V=1
SKIP_LATENTSYNC=0
SKIP_MUSETALK=1
SKIP_DEMUCS=1
SKIP_ENV_FILE=0
ALLOW_MISSING_SERVICES=0
CODE_TEST=0
NO_VENV=0
WITH_STORAGE=0
TIMEOUT="${VIDEO_ME_READINESS_TIMEOUT:-3.0}"
PYTHON_BIN="${PYTHON_BIN:-}"
WORKSPACE="${WORKSPACE:-/workspace}"
HF_TOKEN="${HF_TOKEN:-}"
HF_HOME="${HF_HOME:-}"
WHISPER_MODEL_SIZE="${VIDEO_ME_WHISPER_MODEL_SIZE:-large-v3}"
WHISPER_PREFETCH_MODELS="${VIDEO_ME_WHISPER_PREFETCH_MODELS:-$WHISPER_MODEL_SIZE}"
WHISPER_PREFETCH_MODELS_SET=0
if [[ -n "${VIDEO_ME_WHISPER_PREFETCH_MODELS:-}" ]]; then
  WHISPER_PREFETCH_MODELS_SET=1
fi
WHISPER_DOWNLOAD_ROOT="${VIDEO_ME_WHISPER_DOWNLOAD_ROOT:-}"
WHISPER_MODEL_REVISION="${VIDEO_ME_WHISPER_MODEL_REVISION:-}"
SKIP_WHISPER_MODEL=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_gpu.sh [options]

Full GPU-machine setup for video_me on RunPod (or any Ubuntu+CUDA box):
  1. Verify CUDA / GPU
  2. Install system packages  (ffmpeg, yt-dlp, curl, git)
  3. Create Python venv + install runtime extras
  4. Prefetch faster-whisper large-v3 into the persistent HF cache
  5. Install Ollama + pull qwen3.6:35b (LLM + VLM for all stages)
  6. Clone + set up ComfyUI model dirs (Flux 2.0 Dev image weights)
  7. Clone + set up Fish Audio S2 TTS (EN + HI, port 8025)
  8. Clone + set up Wan2.2 S2V (singing video, port 8031)
  9. Write .env with GPU-correct settings
  10. Run runtime readiness check

  Optional/fallback services:
    --with-a1111        AUTOMATIC1111 + SD 1.5  (RENDER_ADAPTER=a1111)
    --with-chatterbox   Chatterbox TTS           (TTS_ADAPTER=chatterbox)
    --with-wan-i2v      Wan 2.2 I2V model        (VIDEO_ADAPTER=wan)
    --with-lightx2v     LightX2V Wan I2V fast path (VIDEO_ADAPTER=wan_lightx2v)
    --with-latentsync   LatentSync repair        (default install; explicit for clarity)
    --with-demucs       Demucs vocal isolation   (VIDEO_ME_WHISPER_ISOLATE_VOCALS=true)
    --with-musetalk     MuseTalk repair fallback (LIPSYNC_ADAPTER=musetalk)
    --with-ltx          Legacy LTX/ComfyUI video (VIDEO_ADAPTER=ltx)
    --with-wan          Back-compat: Wan I2V + LatentSync + MuseTalk

Network volume (RunPod):
  Models and service repos are placed under WORKSPACE (default /workspace) so
  they survive pod restarts when a network volume is mounted there.

Options:
  --workspace PATH          Network volume / persistent root  [default: /workspace]
  --dry-run                 Print commands without executing them
  --skip-system-deps        Skip apt-get installs
  --skip-python-deps        Skip pip install
  --skip-whisper-model      Skip faster-whisper model prefetch
  --whisper-model MODEL     Whisper model used by jobs [default: large-v3]
  --whisper-model-revision REV
                            Optional HF commit/tag to pin faster-whisper downloads
  --prefetch-whisper-models CSV
                            Download comma-separated Whisper models, e.g. large-v3,large-v2
  --skip-ollama             Skip Ollama install + model pull
  --skip-comfyui            Skip ComfyUI/model-dir install (Flux weights + legacy fallbacks)
  --skip-ltx                Skip legacy LTX weights/custom nodes [default]
  --skip-fish-s2            Skip Fish Audio S2 install (default TTS, port 8025)
  --skip-wan                Skip Wan2.2 S2V install (default video, port 8031)
  --skip-wan-i2v            Skip Wan2.2 I2V fallback weights
  --skip-lightx2v           Skip LightX2V Wan I2V fast-path setup
  --skip-latentsync         Skip LatentSync fallback setup
  --with-a1111              Also install AUTOMATIC1111 (SD 1.5 render fallback, port 7860)
  --with-chatterbox         Also install Chatterbox TTS (EN-only fallback, port 8020)
  --with-wan-i2v            Also install Wan2.2 I2V weights (VIDEO_ADAPTER=wan)
  --with-lightx2v           Also install LightX2V + Wan2.2 I2V distill LoRAs
  --with-latentsync         Install LatentSync repair (already default; preferred for Wan I2V)
  --with-musetalk           Also install MuseTalk repair (legacy Wan I2V fallback)
  --with-demucs             Also install Demucs vocal isolation (pre-Whisper stem separation, opt-in toggle for singing jobs)
  --with-ltx                Also install legacy LTX ComfyUI nodes + weights
  --with-wan                Back-compat alias for --with-wan-i2v --with-latentsync --with-musetalk
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
  # Full first-time setup on a fresh RunPod pod (default stack only):
  bash scripts/setup_gpu.sh

  # Full setup + fallback adapters:
  bash scripts/setup_gpu.sh --with-a1111 --with-chatterbox --with-wan-i2v --with-latentsync
  bash scripts/setup_gpu.sh --with-lightx2v --with-latentsync

  # Dry-run to preview all steps:
  bash scripts/setup_gpu.sh --dry-run

  # Code-only smoke test (no GPU, no services):
  bash scripts/setup_gpu.sh --code-test --skip-services --skip-ollama --skip-comfyui --skip-fish-s2 --skip-wan

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
    --skip-whisper-model) SKIP_WHISPER_MODEL=1; shift ;;
    --whisper-model)      [[ $# -ge 2 ]] || die "--whisper-model requires a model name"; WHISPER_MODEL_SIZE="$2"; if [[ "$WHISPER_PREFETCH_MODELS_SET" == "0" ]]; then WHISPER_PREFETCH_MODELS="$2"; fi; shift 2 ;;
    --whisper-model-revision) [[ $# -ge 2 ]] || die "--whisper-model-revision requires a revision"; WHISPER_MODEL_REVISION="$2"; shift 2 ;;
    --prefetch-whisper-models) [[ $# -ge 2 ]] || die "--prefetch-whisper-models requires a comma-separated list"; WHISPER_PREFETCH_MODELS="$2"; WHISPER_PREFETCH_MODELS_SET=1; shift 2 ;;
    --skip-ollama)        SKIP_OLLAMA=1; shift ;;
    --skip-comfyui)       SKIP_COMFYUI=1; shift ;;
    --skip-ltx)           SKIP_LTX=1; shift ;;
    --skip-fish-s2)       SKIP_FISH_S2=1; shift ;;
    --with-a1111)         SKIP_A1111=0; shift ;;
    --with-chatterbox)    SKIP_CHATTERBOX=0; shift ;;
    --with-wan-i2v)       SKIP_WAN=0; SKIP_WAN_I2V=0; shift ;;
    --with-lightx2v)      SKIP_WAN=0; SKIP_WAN_I2V=0; SKIP_LIGHTX2V=0; shift ;;
    --with-latentsync)    SKIP_LATENTSYNC=0; shift ;;
    --with-musetalk)      SKIP_MUSETALK=0; shift ;;
    --with-demucs)        SKIP_DEMUCS=0; shift ;;
    --with-ltx)           SKIP_LTX=0; shift ;;
    --with-wan)           SKIP_WAN=0; SKIP_WAN_I2V=0; SKIP_LATENTSYNC=0; SKIP_MUSETALK=0; shift ;;
    # Kept for back-compat with older setup commands.
    --skip-a1111)         SKIP_A1111=1; shift ;;
    --skip-wan)           SKIP_WAN=1; shift ;;
    --skip-wan-i2v)       SKIP_WAN_I2V=1; shift ;;
    --skip-lightx2v)      SKIP_LIGHTX2V=1; shift ;;
    --skip-latentsync)    SKIP_LATENTSYNC=1; shift ;;
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

HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
WHISPER_DOWNLOAD_ROOT="${WHISPER_DOWNLOAD_ROOT:-$HF_HOME/hub}"
export HF_HOME
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

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
  log "Installing system packages (ffmpeg, yt-dlp, curl, git, libgl1)"

  if need_cmd apt-get; then
    run sudo apt-get update -qq
    run sudo apt-get install -y --no-install-recommends ffmpeg curl git libgl1
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

  # hf_transfer enables fast multi-part HF downloads.
  # Must be installed in the same Python that runs hf CLI and training scripts.
  # HF_HUB_ENABLE_HF_TRANSFER=1 (set in .env) will error if this is missing.
  run pip3 install hf_transfer || warn "hf_transfer install failed — set HF_HUB_ENABLE_HF_TRANSFER=0 to skip"
}

setup_whisper_model() {
  if [[ "$SKIP_WHISPER_MODEL" == "1" || "$CODE_TEST" == "1" ]]; then
    ok "Skipping faster-whisper model prefetch"
    return
  fi

  local normalized_models="${WHISPER_PREFETCH_MODELS//,/ }"
  if [[ -z "${normalized_models// /}" || "$normalized_models" == "none" ]]; then
    ok "No faster-whisper models requested for prefetch"
    return
  fi

  log "Prefetching faster-whisper model(s)"
  if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$WHISPER_DOWNLOAD_ROOT"; fi
  ok "Whisper model cache: $WHISPER_DOWNLOAD_ROOT"

  local -a env_args=("env" "HF_HOME=$HF_HOME" "HF_HUB_DISABLE_XET=1" "HF_HUB_ENABLE_HF_TRANSFER=0")
  if [[ -n "$HF_TOKEN" ]]; then
    env_args+=("HF_TOKEN=$HF_TOKEN")
  fi

  local model
  for model in $normalized_models; do
    if [[ -d "$model" ]]; then
      ok "Whisper model is a local directory; no download needed: $model"
      continue
    fi
    ok "Checking faster-whisper model cache: $model"
    run "${env_args[@]}" "$PYTHON_BIN" -c '
import os
import sys
from faster_whisper.utils import download_model

model = sys.argv[1]
cache_dir = sys.argv[2]
revision = sys.argv[3] or None
token = os.environ.get("HF_TOKEN") or None

try:
    path = download_model(
        model,
        cache_dir=cache_dir,
        local_files_only=True,
        revision=revision,
        use_auth_token=token,
    )
    print(f"{model} already cached -> {path}", flush=True)
except Exception:
    print(f"{model} missing from cache; downloading once...", flush=True)
    path = download_model(
        model,
        cache_dir=cache_dir,
        revision=revision,
        use_auth_token=token,
    )
    print(f"{model} downloaded -> {path}", flush=True)
' "$model" "$WHISPER_DOWNLOAD_ROOT" "$WHISPER_MODEL_REVISION"
  done
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
  # qwen3.6:35b handles ALL stages: text LLM + image critique + video critique (one model).
  # ~30 GB VRAM. Wan S2V and Fish S2 are sequenced later by the workflow.
  log "Pulling Ollama models (qwen3.6:35b — all stages)"
  if [[ "$DRY_RUN" == "0" ]]; then
    OLLAMA_MODELS="$WORKSPACE/ollama" nohup ollama serve &>/tmp/ollama_setup.log &
    OLLAMA_PID=$!
    sleep 5  # give server time to start

    ollama pull qwen3.6:35b || warn "qwen3.6:35b pull failed — retry manually: ollama pull qwen3.6:35b"

    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    ok "Ollama models pulled to $WORKSPACE/ollama"
  else
    run ollama pull qwen3.6:35b
  fi
}

# ── Step 5: ComfyUI model dirs (Flux 2.0 Dev + optional LTX) ────────────────
setup_comfyui() {
  log "Setting up ComfyUI model dirs (Flux 2.0 Dev image weights; LTX optional)"

  local comfyui_dir="$WORKSPACE/ComfyUI"

  if [[ ! -d "$comfyui_dir" ]]; then
    run git clone https://github.com/comfyanonymous/ComfyUI.git "$comfyui_dir"
  else
    ok "ComfyUI already cloned at $comfyui_dir"
    run git -C "$comfyui_dir" pull --ff-only || warn "git pull failed — continuing with existing checkout"
  fi

  run pip3 install -r "$comfyui_dir/requirements.txt" || \
      warn "Some ComfyUI deps may have failed — check $comfyui_dir/requirements.txt"

  # Custom nodes for Flux + optional LTX
  local custom_nodes="$comfyui_dir/custom_nodes"
  if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$custom_nodes"; fi

  if [[ ! -d "$custom_nodes/ComfyUI-Manager" ]]; then
    run git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
        "$custom_nodes/ComfyUI-Manager"
  else
    ok "ComfyUI-Manager already installed"
  fi

  if [[ "$SKIP_LTX" == "0" ]]; then
    if [[ ! -d "$custom_nodes/ComfyUI-LTXVideo" ]]; then
      run git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git \
          "$custom_nodes/ComfyUI-LTXVideo"
      run pip3 install -r "$custom_nodes/ComfyUI-LTXVideo/requirements.txt" || \
          warn "Some LTX node deps may have failed"
      ok "LTX-Video custom nodes installed"
    else
      ok "LTX-Video custom nodes already installed"
    fi
  else
    ok "Skipping legacy LTX custom nodes (use --with-ltx to install)"
  fi

  # Model directories
  local models_dir="$comfyui_dir/models"
  if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$models_dir/diffusion_models" "$models_dir/checkpoints" \
             "$models_dir/loras" "$models_dir/text_encoders"
  fi

  # ── Flux 2.0 Dev DiT (~61 GB) ────────────────────────────────────────────
  # Flux 2.0 is a fundamentally different architecture from Flux 1.x:
  #   - 8 double blocks + 48 single blocks (vs 19+38 in Flux 1.x)
  #   - No bias tensors in attention projections
  #   - Global modulation streams (not per-block)
  #   - Mistral 3 text encoder (not T5+CLIP) — used for LoRA training
  # Requires HF token + accepting license at huggingface.co/black-forest-labs/FLUX.2-dev
  local flux_model="$models_dir/diffusion_models/flux2-dev.safetensors"
  if [[ ! -f "$flux_model" ]]; then
    if [[ -n "$HF_TOKEN" ]]; then
      log "Downloading Flux 2.0 Dev DiT (~61 GB) — this will take a while"
      run hf download black-forest-labs/FLUX.2-dev \
          flux2-dev.safetensors \
          --local-dir "$models_dir/diffusion_models"
      ok "Flux 2.0 Dev DiT downloaded to $flux_model"
    else
      warn "Flux 2.0 Dev not downloaded — HF_TOKEN required."
      warn "  1. Accept license at https://huggingface.co/black-forest-labs/FLUX.2-dev"
      warn "  2. Set HF_TOKEN and re-run, or:"
      warn "     HF_TOKEN=hf_... hf download black-forest-labs/FLUX.2-dev flux2-dev.safetensors \\"
      warn "       --local-dir $models_dir/diffusion_models"
    fi
  else
    ok "Flux 2.0 Dev DiT already at $flux_model"
  fi

  # ── Flux 2.0 VAE / AE (~335 MB) ──────────────────────────────────────────
  # Bundled in the FLUX.2-dev repo alongside the DiT. Stored in diffusion_models/
  # (not vae/) so both DiT and AE are co-located for training scripts.
  local ae_model="$models_dir/diffusion_models/ae.safetensors"
  if [[ ! -f "$ae_model" ]]; then
    if [[ -n "$HF_TOKEN" ]]; then
      log "Downloading Flux 2.0 VAE/AE (~335 MB)"
      run hf download black-forest-labs/FLUX.2-dev \
          ae.safetensors \
          --local-dir "$models_dir/diffusion_models"
      ok "Flux 2.0 AE downloaded to $ae_model"
    else
      warn "Flux 2.0 AE not downloaded — needs same HF_TOKEN as DiT above"
    fi
  else
    ok "Flux 2.0 AE already at $ae_model"
  fi

  # ── Text encoders for ComfyUI inference (T5 fp8 + CLIP-L) ────────────────
  # ComfyUI uses T5+CLIP for inference conditioning (same as Flux 1.x path).
  # These are NOT used for LoRA training — training uses Mistral 3 (see musubi-tuner setup).
  local t5_model="$models_dir/text_encoders/t5xxl_fp8_e4m3fn.safetensors"
  if [[ ! -f "$t5_model" ]]; then
    log "Downloading T5 XXL FP8 text encoder for ComfyUI inference (~4.5 GB)"
    if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$models_dir/text_encoders"; fi
    run curl -fL "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors" \
        -o "$t5_model"
    ok "T5 XXL FP8 downloaded to $t5_model"
  else
    ok "T5 XXL FP8 already at $t5_model"
  fi

  local clip_model="$models_dir/text_encoders/clip_l.safetensors"
  if [[ ! -f "$clip_model" ]]; then
    log "Downloading CLIP-L text encoder for ComfyUI inference (~1 GB)"
    run curl -fL "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
        -o "$clip_model"
    ok "CLIP-L downloaded to $clip_model"
  else
    ok "CLIP-L already at $clip_model"
  fi

  if [[ "$SKIP_LTX" == "0" ]]; then
    # ── LTX-2.3 22B distilled v1.1 (~42 GB) ───────────────────────────────
    local ltx_model="$models_dir/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"
    if [[ ! -f "$ltx_model" ]]; then
      log "Downloading LTX-2.3 22B distilled v1.1 (~42 GB) — this will take a while"
      run hf download Lightricks/LTX-2.3 \
          ltx-2.3-22b-distilled-1.1.safetensors \
          --local-dir "$models_dir/checkpoints" \
          ${HF_TOKEN:+--token "$HF_TOKEN"}
      ok "LTX-2.3 22B distilled v1.1 downloaded to $ltx_model"
    else
      ok "LTX-2.3 already at $ltx_model"
    fi

    # ── LTX-AV Gemma-3 text encoder (~8.8 GB) ─────────────────────────────
    # Required by the LTXAVTextEncoderLoader node (assets/comfyui_workflows/ltx_i2v.json).
    # This node is hardcoded to a Gemma-3 tokenizer regardless of which file is
    # passed in — any other text encoder (e.g. t5xxl) fails with "invalid tokenizer".
    local ltx_text_encoder="$models_dir/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
    if [[ ! -f "$ltx_text_encoder" ]]; then
      log "Downloading LTX-AV Gemma-3 text encoder (~8.8 GB)"
      run hf download Comfy-Org/ltx-2 \
          split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
          --local-dir "$models_dir/text_encoders" \
          ${HF_TOKEN:+--token "$HF_TOKEN"}
      # hf download preserves the split_files/text_encoders/ subpath — flatten it.
      if [[ -f "$models_dir/text_encoders/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" ]]; then
        run mv "$models_dir/text_encoders/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
            "$ltx_text_encoder"
        run rm -rf "$models_dir/text_encoders/split_files"
      fi
      ok "LTX-AV Gemma-3 text encoder downloaded to $ltx_text_encoder"
    else
      ok "LTX-AV Gemma-3 text encoder already at $ltx_text_encoder"
    fi
  else
    ok "Skipping legacy LTX weights (use --with-ltx to install)"
  fi

  # Symlink project LoRA dir into ComfyUI's loras folder
  local lora_link="$models_dir/loras/kids_duo"
  local lora_src="$ROOT_DIR/loras"
  if [[ ! -e "$lora_link" ]]; then
    run ln -s "$lora_src" "$lora_link"
    ok "Symlinked $lora_src → $lora_link"
  else
    ok "ComfyUI LoRA symlink already exists at $lora_link"
  fi

  ok "ComfyUI setup complete (dir: $comfyui_dir)"
}

# ── Step 5b: musubi-tuner (Flux 2.0 LoRA training) ───────────────────────────
setup_musubi_tuner() {
  log "Setting up musubi-tuner (Flux 2.0 LoRA training)"
  # musubi-tuner is kohya's training framework with native Flux 2.0 support.
  # sd-scripts cannot train Flux 2.0 — different architecture (8 double/48 single
  # blocks, no bias tensors, Mistral 3 text encoder instead of T5+CLIP).

  local musubi_dir="$WORKSPACE/musubi-tuner"
  local text_enc_dir="$WORKSPACE/FLUX2-text-encoder"
  local venv_dir="$WORKSPACE/.venv_musubi"

  # Clone / update musubi-tuner
  if [[ ! -d "$musubi_dir" ]]; then
    run git clone https://github.com/kohya-ss/musubi-tuner.git "$musubi_dir"
  else
    ok "musubi-tuner already cloned at $musubi_dir"
    run git -C "$musubi_dir" pull --ff-only || warn "git pull failed — continuing"
  fi

  # Dedicated venv (inherits system torch via --system-site-packages, same pattern
  # as .venv_wan/.venv_fish_s2/.venv_musetalk). musubi_flux_adapter.py invokes
  # this venv's python directly — NOT system pip3/python. A prior version of this
  # script installed musubi-tuner into system Python, which silently vanished on a
  # pod restart (same class of loss as Ollama's binary); an isolated venv under
  # /workspace survives restarts and isn't at the mercy of system-wide package churn.
  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for torch 2.8.0+cu128)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "musubi-tuner venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip
  run "$pip" install accelerate hf_transfer flash-attn --no-build-isolation || \
      warn "flash-attn build failed — training will be slower without it"
  run "$pip" install -e "$musubi_dir" || warn "musubi-tuner install failed"

  # Configure accelerate for single-GPU bf16
  if [[ "$DRY_RUN" == "0" ]]; then
    "$venv_dir/bin/accelerate" config default --mixed_precision bf16 2>/dev/null || true
  fi

  # ── Mistral 3 text encoder (~45 GB, 10 shards) ───────────────────────────
  # Used only for LoRA training (musubi-tuner). ComfyUI inference uses T5+CLIP.
  # Stored at $WORKSPACE/FLUX2-text-encoder/text_encoder/ and tokenizer/.
  local te_shard="$text_enc_dir/text_encoder/model-00001-of-00010.safetensors"
  if [[ ! -f "$te_shard" ]]; then
    if [[ -n "$HF_TOKEN" ]]; then
      log "Downloading Mistral 3 text encoder for Flux 2.0 training (~45 GB, 10 shards)"
      if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$text_enc_dir"; fi
      run env HF_HUB_ENABLE_HF_TRANSFER=0 HF_TOKEN="$HF_TOKEN" \
          hf download black-forest-labs/FLUX.2-dev \
          --include "text_encoder/*" "tokenizer/*" \
          --local-dir "$text_enc_dir"
      ok "Mistral 3 text encoder downloaded to $text_enc_dir"
    else
      warn "Mistral 3 text encoder not downloaded — HF_TOKEN required."
      warn "  Run: HF_TOKEN=hf_... hf download black-forest-labs/FLUX.2-dev \\"
      warn "         --include 'text_encoder/*' 'tokenizer/*' --local-dir $text_enc_dir"
    fi
  else
    ok "Mistral 3 text encoder already at $text_enc_dir"
  fi

  ok "musubi-tuner setup complete"
  ok "LoRA training commands (run from $ROOT_DIR):"
  ok "  # Pre-cache (once per dataset, fast):"
  ok "  $venv_dir/bin/python $musubi_dir/src/musubi_tuner/flux_2_cache_latents.py \\"
  ok "    --dataset_config assets/kids_duo/training/musubi_dataset_max.toml \\"
  ok "    --vae $WORKSPACE/ComfyUI/models/diffusion_models/ae.safetensors --model_version dev"
  ok "  $venv_dir/bin/python $musubi_dir/src/musubi_tuner/flux_2_cache_text_encoder_outputs.py \\"
  ok "    --dataset_config assets/kids_duo/training/musubi_dataset_max.toml \\"
  ok "    --text_encoder $te_shard --batch_size 4 --model_version dev"
  ok "  # Train Max LoRA:"
  ok "  $venv_dir/bin/accelerate launch $musubi_dir/src/musubi_tuner/flux_2_train_network.py \\"
  ok "    --model_version dev --dit $WORKSPACE/ComfyUI/models/diffusion_models/flux2-dev.safetensors \\"
  ok "    --vae $WORKSPACE/ComfyUI/models/diffusion_models/ae.safetensors \\"
  ok "    --text_encoder $te_shard \\"
  ok "    --dataset_config assets/kids_duo/training/musubi_dataset_max.toml \\"
  ok "    --flash_attn --mixed_precision bf16 --fp8_base --fp8_scaled \\"
  ok "    --optimizer_type adamw --learning_rate 1e-4 \\"
  ok "    --network_module networks.lora_flux_2 --network_dim 32 --network_alpha 16 \\"
  ok "    --max_train_epochs 25 --save_every_n_epochs 5 --seed 42 \\"
  ok "    --output_dir loras/ --output_name kids_duo_max"
}

# ── Step 5b: AUTOMATIC1111 (fallback — opt-in only) ──────────────────────────
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

# ── Step 6a: Fish Audio S2 Pro TTS ───────────────────────────────────────────
setup_fish_s2() {
  log "Setting up Fish Audio S2 Pro TTS (port 8025, EN + HI + 80 languages)"
  # Fish S2 Pro (fishaudio/s2-pro) is the public voice-cloning checkpoint.
  # Runs as a FastAPI wrapper (services/fish_s2_server.py) around fish-speech's
  # ModelManager. Dedicated venv with --system-site-packages to inherit system torch.
  local fish_dir="$WORKSPACE/fish-speech"
  local venv_dir="$WORKSPACE/.venv_fish_s2"
  local ckpt_dir="$fish_dir/checkpoints/s2-pro"

  if [[ ! -d "$fish_dir" ]]; then
    run git clone https://github.com/fishaudio/fish-speech.git "$fish_dir"
  else
    ok "Fish Speech already cloned at $fish_dir"
    run git -C "$fish_dir" pull --ff-only || warn "git pull failed — continuing"
  fi

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for torch 2.8.0+cu128)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "Fish S2 venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip
  run "$pip" install -r "$fish_dir/requirements.txt" || warn "Some Fish S2 deps may have failed"
  # Extra deps required by fish-speech 2.0 not included in requirements.txt
  run "$pip" install fastapi uvicorn python-multipart \
      natsort lightning hydra-core opencc-python-reimplemented "pydantic==2.9.2"

  # Download Fish S2 Pro model weights (~20 GB)
  if [[ ! -d "$ckpt_dir" ]]; then
    if [[ -n "$HF_TOKEN" ]]; then
      log "Downloading Fish S2 Pro weights (~20 GB)"
      if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$fish_dir/checkpoints"; fi
      run env HF_HUB_ENABLE_HF_TRANSFER=0 HF_TOKEN="$HF_TOKEN" \
          hf download fishaudio/s2-pro --local-dir "$ckpt_dir"
      ok "Fish S2 Pro downloaded to $ckpt_dir"
    else
      warn "Fish S2 Pro not downloaded — HF_TOKEN required."
      warn "  Run: HF_TOKEN=hf_... hf download fishaudio/s2-pro --local-dir $ckpt_dir"
    fi
  else
    ok "Fish S2 Pro already at $ckpt_dir"
  fi

  ok "Fish Audio S2 Pro installed (venv: $venv_dir, checkpoint: $ckpt_dir)"
}

# ── Step 6a-fallback: Chatterbox TTS (optional, EN-only fallback) ────────────
setup_chatterbox() {
  log "Setting up Chatterbox TTS (port 8020, EN-only fallback)"
  run "$PYTHON_BIN" -m pip install chatterbox-tts torchaudio fastapi uvicorn
  ok "Chatterbox TTS installed"
}

# ── Step 6b: Wan 2.2 ─────────────────────────────────────────────────────────
setup_wan() {
  log "Setting up Wan2.2 Speech-to-Video (port 8031)"
  # Wan2.2 S2V is the default singing video adapter (VIDEO_ADAPTER=wan_s2v).
  # The I2V model is optional for fallback comparison runs (VIDEO_ADAPTER=wan).
  #
  # Clone layout: git clone → $WORKSPACE/Wan2.2/ (repo root with generate.py + wan/ pkg).
  # Wan2.2 is a source checkout, not always a pip-installable package, so setup
  # exposes the repo to .venv_wan via a .pth file when setup.py/pyproject.toml
  # is absent. WAN_DIR in start_services.sh points to this repo root too.

  local wan_dir="$WORKSPACE/Wan2.2"   # repo root containing generate.py + wan/ package
  local wan_s2v_model_dir="$WORKSPACE/Wan2.2-S2V-14B"
  local wan_i2v_model_dir="$WORKSPACE/Wan2.2-I2V-A14B"
  local venv_dir="$WORKSPACE/.venv_wan"

  if [[ ! -d "$wan_dir" ]]; then
    run git clone https://github.com/Wan-Video/Wan2.2.git "$wan_dir"
  else
    ok "Wan2.2 already cloned at $wan_dir"
    run git -C "$wan_dir" pull --ff-only || warn "git pull failed — continuing"
  fi

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for torch 2.8.0+cu128)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "Wan2.2 venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip

  # Core deps — numpy must be ≥2.0 to satisfy system scipy (1.18+).
  # huggingface_hub must stay <1.0 — transformers 4.51.3/peft 0.19.x both
  # require it, and an unpinned install here silently inherits whatever
  # newer (incompatible) version other venvs/system Python have picked up.
  run "$pip" install \
      opencv-python "diffusers>=0.31.0" "transformers>=4.49.0,<=4.51.3" \
      "tokenizers>=0.20.3" "accelerate>=1.1.1" tqdm "imageio[ffmpeg]" \
      easydict ftfy "numpy>=2.0,<2.3" dashscope \
      decord librosa rotary-embedding-torch peft \
      fastapi uvicorn python-multipart \
      "huggingface_hub>=0.30.0,<1.0" hf_transfer || warn "Some Wan deps may have failed"

  # Some Wan2.2 checkouts do not ship setup.py/pyproject.toml, so `pip install
  # -e` fails with "does not appear to be a Python project". Prefer editable
  # install when available; otherwise add the source checkout to this venv via
  # a .pth file so `import wan` works for the optional I2V server.
  if [[ -f "$wan_dir/setup.py" || -f "$wan_dir/pyproject.toml" ]]; then
    run "$pip" install -e "$wan_dir" --no-deps
  else
    if [[ "$DRY_RUN" == "0" ]]; then
      local site_packages
      site_packages="$("$venv_dir/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
      printf '%s\n' "$wan_dir" > "$site_packages/wan2_2_repo.pth"
      ok "Wan2.2 source checkout exposed via $site_packages/wan2_2_repo.pth"
    else
      ok "[dry run] Wan2.2 source checkout would be exposed via .venv_wan .pth file"
    fi
  fi

  # flash_attn — compile from source using system CUDA headers
  if ! "$venv_dir/bin/python" -c "import flash_attn" 2>/dev/null; then
    log "Building flash_attn (~10-15 min, CUDA compilation)"
    run "$pip" install flash-attn --no-build-isolation || \
        warn "flash_attn build failed — Wan2.2 will run without flash attention (slower)"
  else
    ok "flash_attn already installed"
  fi

  if [[ ! -d "$wan_s2v_model_dir" ]] || [[ -z "$(ls -A "$wan_s2v_model_dir" 2>/dev/null)" ]]; then
    log "Downloading Wan2.2-S2V-14B model — this will take a while"
    if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$wan_s2v_model_dir"; fi
    run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
        hf download Wan-AI/Wan2.2-S2V-14B --local-dir "$wan_s2v_model_dir"
    ok "Wan2.2-S2V-14B downloaded to $wan_s2v_model_dir"
  else
    ok "Wan2.2-S2V model already at $wan_s2v_model_dir"
  fi

  if [[ "$SKIP_WAN_I2V" == "0" ]]; then
    if [[ ! -d "$wan_i2v_model_dir" ]] || [[ -z "$(ls -A "$wan_i2v_model_dir" 2>/dev/null)" ]]; then
      log "Downloading Wan2.2-I2V-A14B model (~30 GB) — this will take a while"
      if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$wan_i2v_model_dir"; fi
      run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
          hf download Wan-AI/Wan2.2-I2V-A14B --local-dir "$wan_i2v_model_dir"
      ok "Wan2.2-I2V-A14B downloaded to $wan_i2v_model_dir"
    else
      ok "Wan2.2-I2V model already at $wan_i2v_model_dir"
    fi
  fi
}

# ── Step 6c: LightX2V Wan 2.2 I2V acceleration ──────────────────────────────
setup_lightx2v() {
  log "Setting up LightX2V Wan2.2 I2V fast path (port 8032)"
  local lightx2v_dir="$WORKSPACE/LightX2V"
  local venv_dir="$WORKSPACE/.venv_lightx2v"
  local wan_i2v_model_dir="$WORKSPACE/Wan2.2-I2V-A14B"
  local lora_dir="$WORKSPACE/Wan2.2-Distill-Loras"
  local high_lora="wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
  local low_lora="wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"

  if [[ ! -d "$lightx2v_dir" ]]; then
    run git clone https://github.com/ModelTC/LightX2V.git "$lightx2v_dir"
  else
    ok "LightX2V already cloned at $lightx2v_dir"
    run git -C "$lightx2v_dir" pull --ff-only || warn "LightX2V git pull failed — continuing"
  fi

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for existing CUDA torch)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "LightX2V venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip
  run "$pip" install -v "$lightx2v_dir" fastapi uvicorn python-multipart \
      "huggingface_hub>=0.30.0,<1.0" || warn "Some LightX2V deps may have failed"

  # LightX2V's fastest public Wan2.2 I2V configs use SageAttention. This is an
  # acceleration dependency, not correctness-critical for setup, so keep going
  # and let LIGHTX2V_I2V_ATTN_MODE be changed if the build is unsupported.
  run "$pip" install sageattention || warn "sageattention install failed — set LIGHTX2V_I2V_ATTN_MODE to a supported installed attention backend"

  if [[ ! -d "$wan_i2v_model_dir" ]] || [[ -z "$(ls -A "$wan_i2v_model_dir" 2>/dev/null)" ]]; then
    log "Downloading Wan2.2-I2V-A14B base model for LightX2V LoRA inference (~30 GB)"
    if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$wan_i2v_model_dir"; fi
    run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
        hf download Wan-AI/Wan2.2-I2V-A14B --local-dir "$wan_i2v_model_dir"
    ok "Wan2.2-I2V-A14B downloaded to $wan_i2v_model_dir"
  else
    ok "Wan2.2-I2V base model already at $wan_i2v_model_dir"
  fi

  if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$lora_dir"; fi
  if [[ ! -f "$lora_dir/$high_lora" ]]; then
    log "Downloading LightX2V high-noise Wan2.2 I2V distill LoRA"
    run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
        hf download lightx2v/Wan2.2-Distill-Loras "$high_lora" --local-dir "$lora_dir"
  else
    ok "LightX2V high-noise LoRA already at $lora_dir/$high_lora"
  fi
  if [[ ! -f "$lora_dir/$low_lora" ]]; then
    log "Downloading LightX2V low-noise Wan2.2 I2V distill LoRA"
    run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
        hf download lightx2v/Wan2.2-Distill-Loras "$low_lora" --local-dir "$lora_dir"
  else
    ok "LightX2V low-noise LoRA already at $lora_dir/$low_lora"
  fi
}

# ── Step 6b2: Demucs vocal isolation ─────────────────────────────────────────
setup_demucs() {
  log "Setting up Demucs vocal isolation (pre-Whisper stem separation)"
  # Optional preprocessing for the transcribe stage: separates vocals from
  # background music before faster-whisper runs, improving lyric accuracy on
  # singing/lyric jobs. Applied only when a job sets audio_profile=singing AND
  # VIDEO_ME_WHISPER_ISOLATE_VOCALS=true (dashboard toggle) — see
  # adapters/transcribe/whisper_adapter.py:_isolate_vocals().

  local venv_dir="$WORKSPACE/.venv_demucs"

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for torch 2.8.0+cu128)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "Demucs venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip
  run "$pip" install demucs || warn "Demucs install failed — vocal isolation toggle will no-op"
  ok "Demucs installed (model weights download lazily to \$HF_HOME on first use)"
}

# ── Step 6c: LatentSync ───────────────────────────────────────────────────────
setup_latentsync() {
  log "Setting up LatentSync lip-sync repair (port 8041)"
  # LatentSync is the preferred repair stage for VIDEO_ADAPTER=wan. It is not
  # used by the default S2V path because Wan S2V receives the audio directly.

  local latentsync_dir="$WORKSPACE/LatentSync"
  local venv_dir="$WORKSPACE/.venv_latentsync"

  if [[ ! -d "$latentsync_dir" ]]; then
    run git clone https://github.com/bytedance/LatentSync.git "$latentsync_dir"
  else
    ok "LatentSync already cloned at $latentsync_dir"
    run git -C "$latentsync_dir" pull --ff-only || warn "git pull failed — continuing"
  fi

  local bootstrap_python="python3"
  if need_cmd python3.10; then
    bootstrap_python="python3.10"
  else
    warn "python3.10 not found — using python3; LatentSync upstream recommends Python 3.10"
  fi

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for torch/CUDA)"
    run "$bootstrap_python" -m venv --system-site-packages "$venv_dir"
  else
    ok "LatentSync venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip
  run "$pip" install -r "$latentsync_dir/requirements.txt" || warn "Some LatentSync deps may have failed"
  run "$pip" install fastapi uvicorn python-multipart "huggingface_hub[cli]"

  local ckpt_dir="$latentsync_dir/checkpoints"
  if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$ckpt_dir"; fi
  if [[ ! -f "$ckpt_dir/latentsync_unet.pt" ]]; then
    run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
      "$venv_dir/bin/huggingface-cli" download ByteDance/LatentSync-1.6 \
      latentsync_unet.pt --local-dir "$ckpt_dir"
  else
    ok "LatentSync U-Net checkpoint already exists"
  fi
  if [[ ! -f "$ckpt_dir/whisper/tiny.pt" ]]; then
    run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
      "$venv_dir/bin/huggingface-cli" download ByteDance/LatentSync-1.6 \
      whisper/tiny.pt --local-dir "$ckpt_dir"
  else
    ok "LatentSync Whisper checkpoint already exists"
  fi

  ok "LatentSync setup complete (venv: $venv_dir)"
}

# ── Step 6d: MuseTalk ─────────────────────────────────────────────────────────
setup_musetalk() {
  log "Setting up MuseTalk lip-sync (port 8040)"
  # Strategy: isolated venv at /workspace/.venv_musetalk with --system-site-packages
  # so it inherits system torch 2.8.0+cu128 without reinstalling it.
  # mmcv must be built from source (no cp312 wheels exist); takes ~15-20 min on A100.

  local musetalk_dir="$WORKSPACE/MuseTalk"
  local venv_dir="$WORKSPACE/.venv_musetalk"

  if [[ ! -d "$musetalk_dir" ]]; then
    run git clone https://github.com/TMElyralab/MuseTalk.git "$musetalk_dir"
  else
    ok "MuseTalk already cloned at $musetalk_dir"
  fi

  # Create isolated venv inheriting system torch
  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for torch 2.8.0+cu128)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "MuseTalk venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"

  # Core inference deps
  run "$pip" install \
    opencv-python omegaconf librosa einops soundfile "imageio[ffmpeg]" \
    tqdm Pillow gdown huggingface_hub transformers \
    "diffusers==0.30.2" "accelerate>=0.28.0" \
    face-alignment ffmpeg-python moviepy \
    Cython munkres \
    mmengine "mmdet>=3.0.0" \
    fastapi uvicorn python-multipart

  # xtcocotools' setup.py imports numpy directly at build time — pip's isolated
  # build sandbox doesn't inherit --system-site-packages, so numpy isn't visible
  # there even though it's visible at install time. --no-build-isolation lets it
  # use this venv's already-inherited numpy instead of failing to find one.
  run "$pip" install xtcocotools --no-build-isolation

  # chumpy's old-style setup.py shells out to pip internally, which isn't
  # available inside an isolated build sandbox either — same fix.
  run "$pip" install chumpy json-tricks --no-build-isolation

  # mmpose 1.3.2 supports mmcv <3.0.0 (compatible with 2.2.0)
  run "$pip" install "mmpose==1.3.2" --no-deps

  # mmcv must be built from source for Python 3.12 (no prebuilt cp312 wheels),
  # pinned to 2.1.0 — mmdet 3.3.0 requires mmcv<2.2.0; 2.2.0+ breaks import.
  # MAX_JOBS=8 parallelises CUDA compilation; still takes ~15-20 min.
  if ! "$venv_dir/bin/python" -c "import mmcv; assert mmcv.__version__ == '2.1.0'" 2>/dev/null; then
    log "Building mmcv==2.1.0 from source (CUDA extensions, ~15-20 min with MAX_JOBS=8)"
    run MAX_JOBS=8 "$pip" install mmcv==2.1.0 --no-build-isolation
  else
    ok "mmcv 2.1.0 already installed"
  fi

  # Download model weights to $musetalk_dir/models/
  log "Downloading MuseTalk model weights"
  run "$venv_dir/bin/python" - <<PYEOF
import os, sys
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.chdir('$musetalk_dir')
sys.path.insert(0, '$musetalk_dir')

from huggingface_hub import hf_hub_download
import urllib.request

downloads = [
    ('TMElyralab/MuseTalk', 'musetalkV15/musetalk.json', 'models'),
    ('TMElyralab/MuseTalk', 'musetalkV15/unet.pth',      'models'),
    ('stabilityai/sd-vae-ft-mse', 'config.json',                      'models/sd-vae'),
    ('stabilityai/sd-vae-ft-mse', 'diffusion_pytorch_model.bin',      'models/sd-vae'),
    ('openai/whisper-tiny', 'config.json',               'models/whisper'),
    ('openai/whisper-tiny', 'pytorch_model.bin',         'models/whisper'),
    ('openai/whisper-tiny', 'preprocessor_config.json',  'models/whisper'),
    ('yzd-v/DWPose', 'dw-ll_ucoco_384.pth',              'models/dwpose'),
]
for repo, filename, local_dir in downloads:
    os.makedirs(local_dir, exist_ok=True)
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f'SKIP {dest}')
        continue
    print(f'Downloading {repo}/{filename}...', flush=True)
    hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    print(f'  OK ({os.path.getsize(dest):,} bytes)', flush=True)

# face-parse-bisent from Google Drive + PyTorch CDN
os.makedirs('models/face-parse-bisent', exist_ok=True)
if not os.path.exists('models/face-parse-bisent/79999_iter.pth'):
    import subprocess
    subprocess.run(['$venv_dir/bin/python', '-m', 'gdown',
                    'https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812',
                    '-O', 'models/face-parse-bisent/79999_iter.pth'], check=True)
if not os.path.exists('models/face-parse-bisent/resnet18-5c106cde.pth'):
    url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    print(f'Downloading resnet18 from {url}...', flush=True)
    urllib.request.urlretrieve(url, 'models/face-parse-bisent/resnet18-5c106cde.pth')

print('All MuseTalk weights ready.')
PYEOF

  ok "MuseTalk setup complete (venv: $venv_dir)"
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
HF_HOME=$HF_HOME
VIDEO_ME_WHISPER_MODEL_SIZE=$WHISPER_MODEL_SIZE
VIDEO_ME_WHISPER_DOWNLOAD_ROOT=$WHISPER_DOWNLOAD_ROOT
VIDEO_ME_WHISPER_LOCAL_FILES_ONLY=true
VIDEO_ME_WHISPER_MODEL_REVISION=$WHISPER_MODEL_REVISION
VIDEO_ME_WHISPER_LANGUAGE=en
VIDEO_ME_WHISPER_VAD_FILTER=false
VIDEO_ME_TRANSCRIPT_MIN_COVERAGE_RATIO=0.2

VIDEO_ME_DATA_DIR=$WORKSPACE/video_me/data
VIDEO_ME_REVIEW_DIR=$WORKSPACE/video_me/review
VIDEO_ME_LORA_DIR=$ROOT_DIR/loras
VIDEO_ME_VOICE_DIR=$ROOT_DIR/voices

# Single model for all stages: text LLM + image critique + video critique
# qwen3.6:35b (MoE 35B, natively multimodal, ~30 GB VRAM)
VIDEO_ME_LLM_MODEL=qwen3.6:35b
VIDEO_ME_LLM_BASE_URL=http://localhost:11434/v1
VIDEO_ME_CRITIQUE_MODEL=qwen3.6:35b
VIDEO_ME_CRITIQUE_BASE_URL=http://localhost:11434/v1

# Render: musubi Flux 2.0 Dev (default) | a1111/comfyui_flux (fallback)
VIDEO_ME_RENDER_ADAPTER=musubi_flux
VIDEO_ME_COMFYUI_BASE_URL=http://localhost:8188

# Video: Wan2.2 S2V image+audio singing video (default) | wan_lightx2v fast I2V | wan I2V | ltx legacy
VIDEO_ME_VIDEO_ADAPTER=wan_s2v
VIDEO_ME_WAN_S2V_BASE_URL=http://localhost:8031
VIDEO_ME_WAN_LIGHTX2V_BASE_URL=http://localhost:8032
VIDEO_ME_WAN_S2V_FPS=16
WAN_S2V_OFFLOAD_MODEL=false
WAN_I2V_OFFLOAD_MODEL=false
LIGHTX2V_I2V_OFFLOAD_MODEL=false
LIGHTX2V_I2V_WIDTH=400
LIGHTX2V_I2V_HEIGHT=704
LIGHTX2V_I2V_STEPS=4
LIGHTX2V_I2V_ATTN_MODE=sage_attn2

# Video/lip-sync QA policy. Wan S2V is native audio-conditioned video, so the
# lip-sync repair policy applies only when VIDEO_ME_VIDEO_ADAPTER is non-native.
VIDEO_ME_MAX_SHOT_DURATION_SEC=8.0
VIDEO_ME_LIPSYNC_FAILURE_POLICY=fallback_raw
VIDEO_ME_LIPSYNC_MAX_RETRIES=0
VIDEO_ME_AV_SYNC_DURATION_TOLERANCE_SEC=0.35
VIDEO_ME_AV_SYNC_FAILURE_POLICY=warn

# TTS: Fish Audio S2 (default, EN + HI + 80 languages) | chatterbox (EN-only fallback)
VIDEO_ME_TTS_ADAPTER=fish_s2
VIDEO_ME_FISH_S2_BASE_URL=http://localhost:8025

# Language: en | hi | both (runs pipeline twice for both)
VIDEO_ME_TARGET_LANGUAGE=en

# Fallback service URLs (only needed when using non-default adapters)
VIDEO_ME_SD_BASE_URL=http://localhost:7860
VIDEO_ME_TTS_BASE_URL=http://localhost:8020
VIDEO_ME_WAN_BASE_URL=http://localhost:8030
VIDEO_ME_LIPSYNC_ADAPTER=latentsync
VIDEO_ME_LATENTSYNC_BASE_URL=http://localhost:8041
VIDEO_ME_MUSETALK_BASE_URL=http://localhost:8040
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
  log "Verifying scripts/start_services.sh"
  local start_script="$ROOT_DIR/scripts/start_services.sh"

  # start_services.sh is maintained in git — just ensure it is executable.
  if [[ "$DRY_RUN" == "0" ]]; then
    chmod +x "$start_script"
    ok "start_services.sh is executable: $start_script"
    return
  fi

  ok "[dry run] start_services.sh is maintained in git — would chmod +x $start_script"
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

setup_whisper_model

if [[ "$SKIP_OLLAMA" == "0" ]]; then
  setup_ollama
fi

# Default stack
if [[ "$SKIP_COMFYUI" == "0" ]]; then
  setup_comfyui
  # musubi-tuner is the LoRA training framework for Flux 2.0.
  # Installed alongside ComfyUI since it shares the same DiT + AE weights.
  setup_musubi_tuner
fi

if [[ "$SKIP_FISH_S2" == "0" ]]; then
  setup_fish_s2
fi

if [[ "$SKIP_WAN" == "0" ]]; then
  setup_wan
fi

if [[ "$SKIP_LIGHTX2V" == "0" ]]; then
  setup_lightx2v
fi

# Fallback adapters (opt-in via --with-*)
if [[ "$SKIP_A1111" == "0" ]]; then
  setup_a1111
fi

if [[ "$SKIP_CHATTERBOX" == "0" ]]; then
  setup_chatterbox
fi

if [[ "$SKIP_LATENTSYNC" == "0" ]]; then
  setup_latentsync
fi

if [[ "$SKIP_DEMUCS" == "0" ]]; then
  setup_demucs
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
printf '    4. python -m pytest -q                # confirm 312+ tests pass\n'
printf '\n'
printf '  Then run the pipeline:\n'
printf '    python -c "import asyncio; from core.config import load_app_config; from core.workflow import run_with_critique; config = load_app_config(); job = asyncio.run(run_with_critique(source_url='"'"'YOUR_URL'"'"', rights_cleared=True, app_config=config)); print(job.status)"\n'
printf '\n'
