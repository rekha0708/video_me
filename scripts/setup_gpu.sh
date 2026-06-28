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
SKIP_FISH_S2=0
SKIP_CHATTERBOX=1
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
  4. Install Ollama + pull qwen3.6:35b (LLM + VLM for all stages)
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
  --skip-fish-s2            Skip Fish Audio S2 install (default TTS, port 8025)
  --with-chatterbox         Also install Chatterbox TTS (EN-only fallback, port 8020)
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
  bash scripts/setup_gpu.sh --code-test --skip-services --skip-ollama --skip-a1111 --skip-fish-s2

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
    --skip-fish-s2)       SKIP_FISH_S2=1; shift ;;
    --with-chatterbox)    SKIP_CHATTERBOX=0; shift ;;
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
  # qwen3.6:35b handles ALL stages: text LLM + image critique + video critique (one model).
  # ~30 GB VRAM. G200 budget: qwen3.6:35b (~30 GB) + LTX (~44 GB) + Fish S2 (~20 GB) = ~94 GB peak.
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

# ── Step 6a: Fish Audio S2 TTS ───────────────────────────────────────────────
setup_fish_s2() {
  log "Setting up Fish Audio S2 TTS (port 8025, EN + HI + 80 languages)"
  # Fish Audio S2 runs as a self-hosted HTTP server with voice cloning.
  # Requires a dedicated venv with system-site-packages for torch 2.8.0+cu128.
  local fish_dir="$WORKSPACE/fish-speech"
  local venv_dir="$WORKSPACE/.venv_fish_s2"

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
  run "$pip" install fastapi uvicorn python-multipart

  ok "Fish Audio S2 installed (venv: $venv_dir)"
}

# ── Step 6a-fallback: Chatterbox TTS (optional, EN-only fallback) ────────────
setup_chatterbox() {
  log "Setting up Chatterbox TTS (port 8020, EN-only fallback)"
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

  # flash_attn requires compilation and often fails — install everything else first,
  # then attempt flash_attn via --no-build-isolation (uses already-built torch headers)
  log "Installing Wan2.2 requirements (excluding flash_attn)"
  grep -v -i "flash.attn\|flash_attn" "$wan_dir/requirements.txt" > /tmp/wan_req_noflash.txt
  run "$PYTHON_BIN" -m pip install -r /tmp/wan_req_noflash.txt

  log "Installing flash_attn (pre-built, no compilation)"
  run "$PYTHON_BIN" -m pip install flash-attn --no-build-isolation || \
      warn "flash_attn install failed — Wan2.2 will run without flash attention (slower but functional)"

  run "$PYTHON_BIN" -m pip install "huggingface_hub[cli]" hf_transfer fastapi uvicorn

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
    Cython xtcocotools \
    mmengine "mmdet>=3.0.0" \
    fastapi uvicorn python-multipart

  # mmpose 1.3.2 supports mmcv <3.0.0 (compatible with 2.2.0)
  run "$pip" install "mmpose==1.3.2" --no-deps

  # mmcv must be built from source for Python 3.12 (no prebuilt cp312 wheels).
  # MAX_JOBS=8 parallelises CUDA compilation; still takes ~15-20 min.
  if ! "$venv_dir/bin/python" -c "import mmcv" 2>/dev/null; then
    log "Building mmcv from source (CUDA extensions, ~15-20 min with MAX_JOBS=8)"
    run MAX_JOBS=8 "$pip" install mmcv --no-build-isolation
  else
    ok "mmcv already installed: $("$venv_dir/bin/python" -c 'import mmcv; print(mmcv.__version__)')"
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

# Render: ComfyUI + Flux.1-dev (default) | a1111 (fallback)
VIDEO_ME_RENDER_ADAPTER=comfyui_flux
VIDEO_ME_COMFYUI_BASE_URL=http://localhost:8188

# Video: LTX-Video 2.3 via ComfyUI (default) | wan (fallback)
VIDEO_ME_VIDEO_ADAPTER=ltx

# TTS: Fish Audio S2 (default, EN + HI + 80 languages) | chatterbox (EN-only fallback)
VIDEO_ME_TTS_ADAPTER=fish_s2
VIDEO_ME_FISH_S2_BASE_URL=http://localhost:8025

# Language: en | hi | both (runs pipeline twice for both)
VIDEO_ME_TARGET_LANGUAGE=en

# Fallback service URLs (only needed when using non-default adapters)
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

if [[ "$SKIP_OLLAMA" == "0" ]]; then
  setup_ollama
fi

if [[ "$SKIP_A1111" == "0" ]]; then
  setup_a1111
fi

if [[ "$SKIP_FISH_S2" == "0" ]]; then
  setup_fish_s2
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
