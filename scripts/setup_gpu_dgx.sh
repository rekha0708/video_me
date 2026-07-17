#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: This script requires bash, not sh. Run: bash scripts/setup_gpu_dgx.sh" >&2
  exit 1
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# This is the DGX Spark (NVIDIA GB10 "Grace Blackwell", aarch64) variant of
# setup_gpu.sh. It is a fork, not a generic parameterization: the Hopper
# (H100/H200, x86_64) script at scripts/setup_gpu.sh is left untouched and
# keeps working as-is. Differences from that script, and why, are called out
# inline at each divergence point below (search "DGX:").
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "WARNING: this script targets aarch64 DGX Spark (GB10) hosts; detected $(uname -m)." >&2
  echo "         Use scripts/setup_gpu.sh on x86_64 Hopper/Ampere pods instead." >&2
fi

# ── Defaults ────────────────────────────────────────────────────────────────
DRY_RUN=0
SKIP_SYSTEM_DEPS=0
SKIP_CUDA_LIBS=0
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
SKIP_WAN_ANIMATE=1
SKIP_WAN_ANIMATE_FLUX=1
SKIP_LIGHTX2V=1
SKIP_VIDEO_ENHANCE=0
# DGX: LatentSync's requirements.txt hard-pins torch==2.5.1+cu121, decord==0.6.0,
# and onnxruntime-gpu==1.21.0 -- none have aarch64/Blackwell wheels, and naively
# installing them would downgrade/corrupt the system torch 2.13.0+cu130 build
# every other service inherits via --system-site-packages. It also is not used
# by the default wan_s2v path (S2V does native audio-conditioned lip motion, no
# separate repair stage). Default this OFF; --with-latentsync forces it with a
# loud warning.
SKIP_LATENTSYNC=1
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
DOWNLOAD_RIFE_MODEL_ONLY=0
FLUX2_EDIT_ENABLED=false

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_gpu_dgx.sh [options]

DGX Spark (NVIDIA GB10 "Grace Blackwell", aarch64) setup for video_me. This is
a fork of scripts/setup_gpu.sh (which stays the Hopper/x86_64 path) because
this hardware differs in ways that need real code changes, not flags:
  - aarch64: several pip packages ship zero ARM wheels (decord, onnxruntime-gpu)
  - Blackwell (compute capability 12.x): FlashAttention-3's public package only
    builds Hopper (sm_90a) kernels; that build is skipped here instead of
    failing, and Wan is told not to require FA3
  - This box's CUDA 13.2 install is partial (cuBLAS/cuDNN/cuSPARSE/cuSOLVER/
    cuRAND are missing even though nvcc/cudart are present) -- installed here
    via apt before anything that dynamically links them (e.g. ctranslate2/
    faster-whisper) is exercised

  1. Verify CUDA / GPU (GB10, unified memory -- nvidia-smi's "Not Supported"
     memory readout is expected, not an error)
  2. Install system packages (ffmpeg, yt-dlp, curl, git) + missing CUDA libs
  3. Create Python venv + install runtime extras
  4. Prefetch faster-whisper large-v3 into the persistent HF cache
  5. Install Ollama + pull qwen3.6:35b (LLM + VLM for all stages)
  6. Clone + set up ComfyUI model dirs (Flux 2.0 Dev image weights)
  7. Clone + set up Fish Audio S2 TTS (EN + HI, port 8025)
  8. Clone + set up Wan2.2 S2V (singing video, port 8031)
  9. Clone + set up AI video enhancement backends (Real-ESRGAN + RIFE/FILM subprocesses)
  10. Write .env with GPU-correct settings (WAN_REQUIRE_FLASH_ATTN_3=false here)
  11. Run runtime readiness check

  Optional/fallback services:
    --with-a1111        AUTOMATIC1111 + SD 1.5  (RENDER_ADAPTER=a1111) -- untested on aarch64
    --with-chatterbox   Chatterbox TTS           (TTS_ADAPTER=chatterbox) -- untested on aarch64
    --with-wan-i2v      Wan 2.2 I2V model        (VIDEO_ADAPTER=wan)
    --with-lightx2v     LightX2V Wan I2V fast path (VIDEO_ADAPTER=wan_lightx2v)
    --with-wan-animate  Wan 2.2 Animate -- BLOCKED on aarch64 (onnxruntime-gpu has
                        no ARM wheel at any version); this script dies immediately
                        with guidance rather than failing deep into the install
    --with-latentsync   LatentSync repair -- opt-in here (upstream defaults it on)
                        because its requirements.txt hard-pins torch==2.5.1+cu121,
                        decord==0.6.0, onnxruntime-gpu==1.21.0, none of which have
                        Blackwell/aarch64 wheels; also unused by default wan_s2v
    --with-demucs       Demucs vocal isolation   (VIDEO_ME_WHISPER_ISOLATE_VOCALS=true)
    --with-musetalk     MuseTalk repair fallback (LIPSYNC_ADAPTER=musetalk)
    --with-ltx          Legacy LTX/ComfyUI video (VIDEO_ADAPTER=ltx)
    --with-wan          Back-compat: Wan I2V + LatentSync + MuseTalk

Network volume:
  Models and service repos are placed under WORKSPACE (default /workspace) so
  they survive pod restarts when a network volume is mounted there.

Options:
  --workspace PATH          Network volume / persistent root  [default: /workspace]
  --dry-run                 Print commands without executing them
  --skip-system-deps        Skip apt-get installs
  --skip-cuda-libs          Skip installing missing cuBLAS/cuDNN/cuSPARSE/cuSOLVER/cuRAND
  --skip-python-deps        Skip pip install
  --skip-whisper-model      Skip faster-whisper model prefetch
  --download-rife-model-only
                            Only download/extract RIFE pretrained HD model, then exit
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
  --skip-video-enhance      Skip AI video enhance setup (installed by default)
  --skip-latentsync         Skip LatentSync fallback setup
  --with-a1111              Also install AUTOMATIC1111 (SD 1.5 render fallback, port 7860)
  --with-chatterbox         Also install Chatterbox TTS (EN-only fallback, port 8020)
  --with-wan-i2v            Also install Wan2.2 I2V weights (VIDEO_ADAPTER=wan)
  --with-lightx2v           Also install LightX2V + Wan2.2 I2V distill LoRAs
  --with-wan-animate        Install Wan2.2-Animate-14B and GPU preprocessors
  --with-wan-animate-flux-retarget
                            Also download optional FLUX.1-Kontext-dev retarget model
  --with-video-enhance      Back-compat no-op; AI enhance installs by default
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
  # Full first-time setup on a fresh DGX Spark box (default stack only):
  bash scripts/setup_gpu_dgx.sh

  # Dry-run to preview all steps:
  bash scripts/setup_gpu_dgx.sh --dry-run

  # Retry only the RIFE model zip download/extract:
  bash scripts/setup_gpu_dgx.sh --download-rife-model-only

  # Code-only smoke test (no GPU, no services):
  bash scripts/setup_gpu_dgx.sh --code-test --skip-services --skip-ollama --skip-comfyui --skip-fish-s2 --skip-wan

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

# decord (https://github.com/dmlc/decord) publishes zero aarch64 wheels at any
# version, but wan/speech2video.py does a module-level `from decord import
# VideoReader` -- so `import wan` (needed by services/wan_s2v_server.py) hard
# fails without something importable named `decord`, even though our default
# S2V request path (image + audio, no separate pose-conditioning video) never
# actually calls the one method that uses it (read_last_n_frames, only reached
# from load_pose_cond when a pose_video is passed). Rather than fighting
# decord's build (old ffmpeg ABI, CUDA video codec SDK), drop in a small
# cv2-backed shim that implements the exact surface Wan2.2 touches
# (VideoReader(path), .get_avg_fps(), len(), .get_batch(indices).asnumpy()) so
# the import succeeds and the method would still work correctly if ever hit.
install_decord_shim() {
  local venv_dir="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    ok "[dry run] would install cv2-backed decord shim into $venv_dir"
    return
  fi
  local site_packages
  site_packages="$("$venv_dir/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  if "$venv_dir/bin/python" -c "import decord" 2>/dev/null; then
    ok "decord (or shim) already importable in $venv_dir"
    return
  fi
  log "Installing cv2-backed decord shim into $venv_dir (no aarch64 decord wheel exists)"
  cat > "$site_packages/decord.py" <<'PYEOF'
"""Minimal cv2-backed drop-in for the small slice of decord's API that
Wan2.2 (wan/speech2video.py, wan/animate.py) touches at import time and at
call time. Installed because decord ships no aarch64 wheel (any version).
Not a general decord replacement -- only VideoReader(path).get_avg_fps(),
len(vr), and vr.get_batch(indices).asnumpy() are implemented.
"""
import cv2
import numpy as np


class _BatchResult:
    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    def asnumpy(self):
        return self._arr


class VideoReader:
    def __init__(self, uri, ctx=None, width=-1, height=-1, **kwargs):
        self._cap = cv2.VideoCapture(str(uri))
        if not self._cap.isOpened():
            raise IOError(f"decord shim: cannot open video {uri!r}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._len = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def get_avg_fps(self):
        return self._fps

    def __len__(self):
        return self._len

    def get_batch(self, indices):
        frames = []
        for idx in indices:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok_read, frame = self._cap.read()
            if not ok_read:
                raise IndexError(f"decord shim: failed to read frame {idx}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return _BatchResult(np.stack(frames, axis=0))

    def __del__(self):
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()


def cpu(*_args, **_kwargs):
    return None


def gpu(*_args, **_kwargs):
    return None
PYEOF
  "$venv_dir/bin/python" -c "from decord import VideoReader" \
    && ok "decord shim importable" \
    || die "decord shim failed to import — check $site_packages/decord.py"
}

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)          [[ $# -ge 2 ]] || die "--workspace requires a path"; WORKSPACE="$2"; shift 2 ;;
    --dry-run)            DRY_RUN=1; shift ;;
    --skip-system-deps)   SKIP_SYSTEM_DEPS=1; shift ;;
    --skip-cuda-libs)     SKIP_CUDA_LIBS=1; shift ;;
    --skip-python-deps)   SKIP_PYTHON_DEPS=1; shift ;;
    --skip-whisper-model) SKIP_WHISPER_MODEL=1; shift ;;
    --download-rife-model-only) DOWNLOAD_RIFE_MODEL_ONLY=1; shift ;;
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
    --with-wan-animate)
      [[ "$(uname -m)" == "aarch64" ]] && die "Wan Animate is not supported on aarch64: onnxruntime-gpu (its pose2d/det ONNX CUDA preprocessing) publishes no ARM wheel at any version, and its requirements_animate.txt has other x86-only pins. Use scripts/setup_gpu.sh on an x86_64 Hopper/Ampere box for Wan Animate instead."
      SKIP_WAN=0; SKIP_WAN_ANIMATE=0; shift ;;
    --with-wan-animate-flux-retarget)
      [[ "$(uname -m)" == "aarch64" ]] && die "Wan Animate is not supported on aarch64 (see --with-wan-animate)."
      SKIP_WAN=0; SKIP_WAN_ANIMATE=0; SKIP_WAN_ANIMATE_FLUX=0; shift ;;
    --with-video-enhance) SKIP_VIDEO_ENHANCE=0; shift ;;
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
    --skip-wan-animate)   SKIP_WAN_ANIMATE=1; shift ;;
    --skip-video-enhance) SKIP_VIDEO_ENHANCE=1; shift ;;
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
# DGX: this box's huggingface_hub (1.23.0) refuses a plain-HTTP download for
# very large single files (e.g. the 61 GB flux2-dev.safetensors) and demands
# the Xet backend instead -- "Install `hf_xet` ... for xet-powered downloads".
# hf_xet has a working aarch64 wheel and is already installed in .venv, so
# leave Xet enabled here (the x86_64 setup_gpu.sh disables it; do not carry
# that default over on this hf_hub version or the Flux download hard-fails).
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"

# ── Step 1: GPU / CUDA check ─────────────────────────────────────────────────
check_cuda() {
  log "Checking GPU / CUDA (expect NVIDIA GB10 / Grace Blackwell, aarch64)"
  if need_cmd nvidia-smi; then
    if [[ "$DRY_RUN" == "0" ]]; then
      GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
      ok "NVIDIA GPU detected: $GPU_INFO"
      # GB10 has unified CPU/GPU (LPDDR5x) memory. nvidia-smi reports
      # "Not Supported" for memory.total/memory.used on this SKU -- that is
      # expected on Grace Blackwell, not a broken driver.
      if [[ "$GPU_INFO" == *"Not Supported"* ]]; then
        ok "memory.total reads 'Not Supported' -- expected for GB10's unified memory, not an error"
      fi
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

  if [[ "$DRY_RUN" == "0" ]] && need_cmd python3; then
    local capability
    capability=$(python3 -c "import torch; print('.'.join(str(v) for v in torch.cuda.get_device_capability()))" 2>/dev/null || echo "")
    if [[ -n "$capability" ]]; then
      ok "torch reports GPU compute capability: $capability"
      if [[ "$capability" != 9.0* ]]; then
        ok "Not Hopper (sm_90a) -- FlashAttention-3's public build is Hopper-only, so this script skips it and relies on FA2/SDPA (see setup_wan())"
      fi
    else
      warn "Could not probe torch.cuda.get_device_capability() (torch not importable yet from system python3) — will re-check once the venv exists"
    fi
  fi
}

# ── Step 2: System packages ──────────────────────────────────────────────────
install_system_deps() {
  log "Installing system packages (ffmpeg, yt-dlp, curl, git, libgl1, unzip, portaudio19-dev)"

  if need_cmd apt-get; then
    run sudo apt-get update -qq
    # portaudio19-dev: fish-speech's pyproject.toml depends on pyaudio, which
    # has no prebuilt aarch64 wheel and fails to build from source without
    # portaudio.h ("fatal error: portaudio.h: No such file or directory").
    # Because pip aborts the WHOLE batch install when any one package in it
    # fails to build, this silently took uvicorn/fastapi down with it too --
    # the Fish S2 server then failed to start with "uvicorn: No such file".
    run sudo apt-get install -y --no-install-recommends ffmpeg curl git libgl1 unzip portaudio19-dev
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

# ── Step 2b: Missing CUDA libraries (DGX-specific) ───────────────────────────
# This box's CUDA 13.2 install is deliberately partial: nvcc/cudart/nvrtc/nccl
# are present but cuBLAS/cuDNN/cuSPARSE/cuSOLVER/cuRAND are not (confirmed via
# vast-capabilities' cuda.components map). torch bundles its own copies of
# these so it is unaffected, but packages that dynamically link the *system*
# libs at runtime are not -- confirmed concretely: ctranslate2 (faster-whisper's
# backend) reports `get_cuda_device_count() == 0` on this box until these are
# installed. Install from the already-configured NVIDIA CUDA apt repo (13-2
# packages only, never the `cuda` metapackage / cuda-drivers -- see AGENTS.md
# §12 on never touching the host driver from apt).
install_missing_cuda_libs() {
  log "Installing missing CUDA libraries (cuBLAS/cuDNN/cuSPARSE/cuSOLVER/cuRAND)"

  if ! need_cmd apt-get; then
    warn "apt-get not found — skipping CUDA library install; ctranslate2/faster-whisper GPU may not work"
    return
  fi

  run sudo apt-get update -qq
  run sudo apt-get install -y --no-install-recommends \
      libcublas-13-2 libcublas-dev-13-2 \
      libcusparse-13-2 libcusparse-dev-13-2 \
      libcusolver-13-2 libcusolver-dev-13-2 \
      libcurand-13-2 libcurand-dev-13-2 \
      libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    || warn "Some CUDA library packages failed to install — check 'apt-cache policy libcublas-13-2' etc. for this box's exact CUDA point release"
  run sudo ldconfig

  if [[ "$DRY_RUN" == "0" ]] && need_cmd python3; then
    python3 -c "import ctranslate2; n = ctranslate2.get_cuda_device_count(); assert n > 0, 'ctranslate2 still sees 0 CUDA devices'" 2>/dev/null \
      && ok "ctranslate2 (faster-whisper backend) confirms CUDA device visible" \
      || warn "ctranslate2 CUDA check inconclusive (ctranslate2 not installed yet, or still 0 devices — re-run 'python -c \"import ctranslate2; print(ctranslate2.get_cuda_device_count())\"' after install_python_deps)"
  fi
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

  local -a env_args=("env" "HF_HOME=$HF_HOME" "HF_HUB_DISABLE_XET=0" "HF_HUB_ENABLE_HF_TRANSFER=0")
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
  # -e follows symlinks, so a *broken* symlink (stale path from a prior
  # setup attempt under a different root) reads as "doesn't exist" and this
  # would then hit `ln: File exists` trying to recreate it. Check -L too.
  if [[ ! -e "$lora_link" && ! -L "$lora_link" ]]; then
    run ln -s "$lora_src" "$lora_link"
    ok "Symlinked $lora_src → $lora_link"
  elif [[ -L "$lora_link" && ! -e "$lora_link" ]]; then
    warn "Removing stale/broken symlink at $lora_link"
    run rm -f "$lora_link"
    run ln -s "$lora_src" "$lora_link"
    ok "Re-symlinked $lora_src → $lora_link"
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
    log "Creating $venv_dir (system-site-packages, inherits the preinstalled torch build)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "musubi-tuner venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip
  run "$pip" install accelerate hf_transfer safetensors
  # aarch64: no prebuilt flash-attn wheel, so this always compiles from source.
  # MAX_JOBS caps parallel CUDA TU compiles. flash-attn's backward-pass hdim128
  # kernels are individually memory-hungry (each nvcc job can spike well past
  # 10-15 GB), and MAX_JOBS=8 got several of them OOM-killed concurrently on
  # this 121 GB unified-memory GB10 box even with ~115 GB free at the time —
  # the failure is a burst of simultaneous peaks, not a total-memory shortage.
  # MAX_JOBS=2 trades build time for headroom.
  run env MAX_JOBS=2 "$pip" install flash-attn --no-build-isolation || \
      die "Musubi FLUX.2 requires flash-attn, but its installation failed"
  run "$pip" install -e "$musubi_dir" || die "musubi-tuner install failed"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$venv_dir/bin/python" -c \
      'import flash_attn; from flash_attn import flash_attn_func; assert callable(flash_attn_func)' || \
      die "Musubi FLUX.2 requires an importable flash_attn runtime"
    ok "Musubi flash-attn import verified"
  fi

  # Animate Studio's garment/accessory path uses Musubi FLUX.2 multi-reference
  # image editing. The flag is assigned only after the interface and every
  # runtime model asset are checked below.
  local flux2_generate="$musubi_dir/src/musubi_tuner/flux_2_generate_image.py"

  # Configure accelerate for single-GPU bf16
  if [[ "$DRY_RUN" == "0" ]]; then
    "$venv_dir/bin/accelerate" config default --mixed_precision bf16 2>/dev/null || true
  fi

  # ── Mistral 3 text encoder (~45 GB, 10 shards) ───────────────────────────
  # Used only for LoRA training (musubi-tuner). ComfyUI inference uses T5+CLIP.
  # Stored at $WORKSPACE/FLUX2-text-encoder/text_encoder/ and tokenizer/.
  local te_shard="$text_enc_dir/text_encoder/model-00001-of-00010.safetensors"
  local text_encoder_complete=1
  local shard
  local required
  for shard in 01 02 03 04 05 06 07 08 09 10; do
    if [[ ! -s "$text_enc_dir/text_encoder/model-000${shard}-of-00010.safetensors" ]]; then
      text_encoder_complete=0
    fi
  done
  for required in \
      "$text_enc_dir/text_encoder/config.json" \
      "$text_enc_dir/text_encoder/model.safetensors.index.json" \
      "$text_enc_dir/tokenizer/tokenizer.json" \
      "$text_enc_dir/tokenizer/tokenizer_config.json"; do
    if [[ ! -s "$required" ]]; then
      text_encoder_complete=0
    fi
  done
  if [[ "$text_encoder_complete" == "0" ]]; then
    if [[ -n "$HF_TOKEN" ]]; then
      log "Downloading/resuming Mistral 3 text encoder for Flux 2.0 (~45 GB, 10 shards)"
      if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$text_enc_dir"; fi
      # DGX: this box's `hf` CLI (huggingface_hub 1.23.0) only takes one glob
      # per --include flag -- passing two space-separated patterns after a
      # single --include makes it parse the second as an explicit filename,
      # which then disables --include entirely ("Ignoring `--include` since
      # filenames have been explicitly set") and 404s on the literal path.
      # Repeat the flag instead of space-separating the patterns.
      run env HF_HUB_ENABLE_HF_TRANSFER=0 HF_TOKEN="$HF_TOKEN" \
          hf download black-forest-labs/FLUX.2-dev \
          --include "text_encoder/*" --include "tokenizer/*" \
          --local-dir "$text_enc_dir"
      ok "Mistral 3 text encoder downloaded to $text_enc_dir"
    else
      warn "Mistral 3 text encoder not downloaded — HF_TOKEN required."
      warn "  Run: HF_TOKEN=hf_... hf download black-forest-labs/FLUX.2-dev \\"
      warn "         --include 'text_encoder/*' --include 'tokenizer/*' --local-dir $text_enc_dir"
    fi
  else
    ok "Mistral 3 text encoder already at $text_enc_dir"
  fi

  # Do not advertise the dashboard edit path based solely on CLI syntax. A
  # single existing text-encoder shard was previously enough to enable a path
  # that then failed after loading tens of gigabytes. Verify the DiT, AE, all
  # ten Mistral shards/index/config, and tokenizer before writing the env flag.
  local flux2_assets_ready=1
  local -a flux2_required_assets=(
    "$WORKSPACE/ComfyUI/models/diffusion_models/flux2-dev.safetensors"
    "$WORKSPACE/ComfyUI/models/diffusion_models/ae.safetensors"
    "$text_enc_dir/text_encoder/config.json"
    "$text_enc_dir/text_encoder/model.safetensors.index.json"
    "$text_enc_dir/tokenizer/tokenizer.json"
    "$text_enc_dir/tokenizer/tokenizer_config.json"
  )
  for shard in 01 02 03 04 05 06 07 08 09 10; do
    flux2_required_assets+=(
      "$text_enc_dir/text_encoder/model-000${shard}-of-00010.safetensors"
    )
  done
  for required in "${flux2_required_assets[@]}"; do
    if [[ ! -s "$required" ]]; then
      warn "Musubi FLUX.2 runtime asset missing or empty: $required"
      flux2_assets_ready=0
    fi
  done
  if [[ "$DRY_RUN" == "0" && "$flux2_assets_ready" == "1" ]]; then
    if ! "$venv_dir/bin/python" -c \
      'import json, pathlib, sys
from safetensors import safe_open
for path in (pathlib.Path(value) for value in sys.argv[1:]):
    if path.suffix == ".json":
        json.loads(path.read_text())
    elif path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as handle:
            if not list(handle.keys()):
                raise ValueError(f"empty safetensors file: {path}")' \
      "${flux2_required_assets[@]}"; then
      warn "Musubi FLUX.2 model/header validation failed"
      flux2_assets_ready=0
    else
      ok "Musubi FLUX.2 model headers and JSON metadata verified"
    fi
  fi

  if [[ -f "$flux2_generate" ]] \
      && grep -q "control_image_path" "$flux2_generate" \
      && grep -q -- "--ci" "$flux2_generate" \
      && [[ "$flux2_assets_ready" == "1" ]]; then
    FLUX2_EDIT_ENABLED=true
    ok "Musubi FLUX.2 multi-reference edit runtime verified"
  else
    FLUX2_EDIT_ENABLED=false
    warn "Musubi FLUX.2 edit runtime is incomplete — dashboard image references stay disabled"
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
  if [[ ! -e "$lora_link" && ! -L "$lora_link" ]]; then
    run ln -s "$lora_src" "$lora_link"
    ok "Symlinked $lora_src → $lora_link"
  elif [[ -L "$lora_link" && ! -e "$lora_link" ]]; then
    warn "Removing stale/broken symlink at $lora_link"
    run rm -f "$lora_link"
    run ln -s "$lora_src" "$lora_link"
    ok "Re-symlinked $lora_src → $lora_link"
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
    log "Creating $venv_dir (system-site-packages, inherits the preinstalled torch build)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "Fish S2 venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip

  # DGX: upstream fish-speech dropped requirements.txt for pyproject.toml,
  # which hard-pins torch==2.8.0/torchaudio==2.8.0. A normal `pip install -e .`
  # (or the old `-r requirements.txt`, which now 404s) would pull that pin and
  # downgrade/replace this box's preinstalled torch 2.13.0+cu130 — the build
  # every other service here inherits via --system-site-packages, and the one
  # Blackwell (compute capability 12.x) actually needs (min_cuda_for_wheels is
  # 12.8; torch 2.8.0's default index build predates Blackwell kernel support).
  # Install with --no-deps and pull the rest of pyproject.toml's dependency
  # list explicitly instead, leaving system torch/torchaudio untouched.
  run "$pip" install -e "$fish_dir" --no-deps || warn "Fish S2 editable install failed"
  run "$pip" install \
      numpy transformers "datasets==2.18.0" "lightning>=2.1.0" "hydra-core>=1.3.2" \
      "tensorboard>=2.14.1" "natsort>=8.4.0" "einops>=0.7.0" "librosa>=0.10.1" \
      "rich>=13.5.3" "gradio>5.0.0" "wandb>=0.19.0" "grpcio>=1.58.0" "kui>=1.6.0" \
      "uvicorn>=0.30.0" "loguru>=0.6.0" "loralib>=0.1.2" "pyrootutils>=1.0.4" \
      "resampy>=0.4.3" "einx[torch]==0.2.2" "zstandard>=0.22.0" pydub pyaudio \
      "modelscope==1.17.1" "opencc-python-reimplemented==0.1.7" silero-vad \
      ormsgpack "tiktoken>=0.8.0" "pydantic==2.9.2" cachetools \
      descript-audio-codec safetensors \
      fastapi python-multipart \
    || warn "Some Fish S2 deps may have failed — check $venv_dir/bin/pip install output above"

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
    log "Creating $venv_dir (system-site-packages, inherits the preinstalled torch build)"
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
  # DGX: decord is intentionally NOT in this list (no aarch64 wheel exists at
  # any version) — install_decord_shim() below drops in a cv2-backed stand-in.
  run "$pip" install \
      opencv-python "diffusers>=0.31.0" "transformers>=4.49.0,<=4.51.3" \
      "tokenizers>=0.20.3" "accelerate>=1.1.1" tqdm "imageio[ffmpeg]" \
      easydict ftfy "numpy>=2.0,<2.3" dashscope \
      librosa rotary-embedding-torch peft \
      fastapi uvicorn python-multipart \
      "huggingface_hub>=0.30.0,<1.0" hf_transfer || warn "Some Wan deps may have failed"

  install_decord_shim "$venv_dir"

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

  # FlashAttention-2 fallback — compile from source using system CUDA headers.
  # On aarch64 there is no prebuilt wheel, so this compiles locally. MAX_JOBS=2
  # (not 8): flash-attn's hdim128 backward kernels individually spike well past
  # 10-15 GB per nvcc job, and 8 concurrent jobs got several OOM-killed on this
  # 121 GB unified-memory GB10 box even with ~115 GB free (see setup_musubi_tuner).
  if ! "$venv_dir/bin/python" -c "import flash_attn" 2>/dev/null; then
    log "Building FlashAttention-2 fallback (aarch64 source build — can take well over the ~10-15 min seen on x86_64)"
    run env MAX_JOBS=2 "$pip" install flash-attn --no-build-isolation || \
        warn "flash_attn build failed — Wan2.2 will run without flash attention (slower, falls back to SDPA)"
  else
    ok "FlashAttention-2 fallback already installed"
  fi

  # DGX: flash-attn-3's public package (the `hopper` subdirectory of
  # Dao-AILab/flash-attention) only implements Hopper (sm_90a) kernels. GB10 is
  # Blackwell (compute capability 12.x), so that build can never succeed here
  # — skip it instead of burning time on a build that's certain to fail, and
  # force WAN_REQUIRE_FLASH_ATTN_3=false in the generated .env so the Wan
  # services accept the FA2/SDPA fallback Wan's own attention.py already
  # supports (wan/modules/attention.py auto-detects flash_attn_interface,
  # falling back to flash_attn 2, then SDPA, with no code changes needed).
  local capability
  capability=$("$venv_dir/bin/python" -c "import torch; print('.'.join(str(v) for v in torch.cuda.get_device_capability()))" 2>/dev/null || echo "")
  if [[ "$capability" == "9.0" ]]; then
    local fa3_sm90_check
    fa3_sm90_check='import torch, flash_attn_interface as fa3; assert torch.cuda.get_device_capability() == (9, 0), f"expected Hopper sm_90a, got {torch.cuda.get_device_capability()}"; q = torch.randn((1, 128, 8, 64), device="cuda", dtype=torch.bfloat16); fa3.flash_attn_func(q, q, q)'
    if ! "$venv_dir/bin/python" -c "$fa3_sm90_check" 2>/dev/null; then
      log "Building FlashAttention-3 for Hopper sm_90a from Dao-AILab's official package"
      run env FLASHATTENTION_DISABLE_SM80=TRUE MAX_JOBS=8 "$pip" install --force-reinstall --no-deps --no-build-isolation \
          "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3.post1#subdirectory=hopper" || \
          warn "flash-attn-3 build failed — falling back to FA2/SDPA (WAN_REQUIRE_FLASH_ATTN_3 is false in this script's .env)"
    fi
    if "$venv_dir/bin/python" -c "$fa3_sm90_check" 2>/dev/null; then
      ok "FlashAttention-3 verified with a BF16 kernel on Hopper sm_90a"
    else
      warn "FlashAttention-3 sm_90a verification failed — will run on FA2/SDPA instead"
    fi
  else
    ok "GPU compute capability is $capability, not Hopper sm_90a — skipping flash-attn-3 (no Blackwell build exists upstream); Wan will use FA2/SDPA"
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

# ── Step 6c: Wan 2.2 Animate ────────────────────────────────────────────────
setup_wan_animate() {
  log "Setting up Wan2.2 Animate (port 8033)"
  local wan_dir="$WORKSPACE/Wan2.2"
  local model_dir="$WORKSPACE/Wan2.2-Animate-14B"
  local venv_dir="$WORKSPACE/.venv_wan_animate"

  if [[ ! -d "$wan_dir" ]]; then
    run git clone https://github.com/Wan-Video/Wan2.2.git "$wan_dir"
  fi

  # Upstream get_mask_boxes()/get_aug_mask() assume SAM2 always returns a
  # non-empty body mask. When the subject is briefly lost/occluded in the
  # driver clip, SAM2 can hand back an all-zero mask for that frame, and
  # np.nonzero(mask) then feeds .min()/.max() an empty array — crashing
  # preprocessing with "ValueError: zero-size array to reduction operation
  # minimum which has no identity" partway through a real job. Patch in a
  # None-bbox short-circuit (skip augmentation for that frame) idempotently.
  if [[ "$DRY_RUN" == "0" ]] && [[ -f "$wan_dir/wan/modules/animate/preprocess/utils.py" ]] \
      && ! grep -q "SAM2 produced no foreground pixels" "$wan_dir/wan/modules/animate/preprocess/utils.py"; then
    log "Patching Wan2.2 Animate preprocess utils.py for empty SAM2 masks"
    python3 - "$wan_dir/wan/modules/animate/preprocess/utils.py" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
old = '''    y_coords, x_coords = np.nonzero(mask)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(np.int32)
    return bbox


def get_aug_mask(body_mask, w_len=10, h_len=20):
    body_bbox = get_mask_boxes(body_mask)

    bbox_wh = body_bbox[2:4] - body_bbox[0:2]
    w_slice = np.int32(bbox_wh[0] / w_len)
    h_slice = np.int32(bbox_wh[1] / h_len)'''
new = '''    y_coords, x_coords = np.nonzero(mask)
    if x_coords.size == 0 or y_coords.size == 0:
        return None
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(np.int32)
    return bbox


def get_aug_mask(body_mask, w_len=10, h_len=20):
    body_bbox = get_mask_boxes(body_mask)
    if body_bbox is None:
        # SAM2 produced no foreground pixels for this frame (subject briefly
        # lost/occluded) -- nothing to augment, keep the mask as-is.
        return body_mask

    bbox_wh = body_bbox[2:4] - body_bbox[0:2]
    w_slice = max(np.int32(bbox_wh[0] / w_len), 1)
    h_slice = max(np.int32(bbox_wh[1] / h_len), 1)'''
assert old in text, "get_mask_boxes/get_aug_mask source has changed upstream — update the patch"
open(path, "w", encoding="utf-8").write(text.replace(old, new))
PYEOF
    ok "Wan2.2 Animate preprocess utils.py patched"
  fi

  if [[ ! -d "$venv_dir" ]]; then
    run python3 -m venv --system-site-packages "$venv_dir"
  fi
  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip
  # Upstream requirements_animate.txt names CPU-only onnxruntime even though
  # pose2d.py requests CUDAExecutionProvider. Install the GPU distribution and
  # explicitly remove the CPU distribution to prevent silent CPU fallback.
  run "$pip" uninstall -y onnxruntime || true
  # onnxruntime-gpu>=1.21 default PyPI wheels link against CUDA 13
  # (libcudart.so.13), which isn't present on this pod's CUDA 12.8 stack —
  # pin to the last CUDA-12-linked release (verified: CUDAExecutionProvider
  # loads cleanly against libcudart.so.12 here).
  # librosa is required even though Animate doesn't use audio: wan/__init__.py
  # unconditionally imports speech2video.py (shared with the S2V module),
  # which imports librosa at module load time.
  run "$pip" install \
      opencv-python "numpy>=1.24.4,<2" decord peft "diffusers>=0.31.0" \
      "transformers>=4.49.0,<=4.51.3" "tokenizers>=0.20.3" accelerate \
      easydict ftfy tqdm Pillow hydra-core iopath pandas matplotlib loguru \
      librosa \
      sentencepiece moviepy "imageio[ffmpeg]" imageio-ffmpeg dashscope einops \
      fastapi uvicorn python-multipart \
      "huggingface_hub>=0.30.0,<1.0" "onnxruntime-gpu==1.20.1"
  # SAM2 is installed without dependency resolution so it cannot replace the
  # pod's CUDA torch build.  Its official runtime dependencies are installed
  # explicitly above (numpy/tqdm/hydra-core/iopath/pillow).
  # --no-build-isolation is required too: SAM2's build backend otherwise
  # fetches its own torch wheel into an ephemeral build env to compile its
  # CUDA extension, which can land a different CUDA build (e.g. cu130) than
  # the pod's nvcc (12.8) / the venv's inherited system torch (cu128),
  # causing "detected CUDA version mismatches the version used to compile
  # PyTorch". --no-build-isolation makes it reuse the venv's own torch.
  run "$pip" install --no-deps --no-build-isolation \
      "git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266"

  if [[ "$DRY_RUN" == "0" ]]; then
    local site_packages
    site_packages="$("$venv_dir/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    printf '%s\n' "$wan_dir" > "$site_packages/wan2_2_repo.pth"

    # Upstream's `pip install git+...` ships an EMPTY sam2_configs package: the
    # Hydra YAML configs (sam2_hiera_l.yaml etc.) live in a top-level
    # sam2_configs/ directory in the source tree but no package_data rule
    # bundles them into site-packages. Without this, the first real Wan
    # Animate preprocessing run dies with hydra.errors.MissingConfigException:
    # "Cannot find primary config 'sam2_hiera_l.yaml'".
    local sam2_configs_dir sam2_src_dir
    sam2_configs_dir="$("$venv_dir/bin/python" -c 'import sam2_configs, os; print(os.path.dirname(sam2_configs.__file__))')"
    if [[ -z "$(find "$sam2_configs_dir" -maxdepth 1 -iname '*.yaml' -print -quit)" ]]; then
      log "Populating sam2_configs Hydra YAML files (upstream pip package ships them empty)"
      sam2_src_dir="$WORKSPACE/.cache/sam2_configs_src"
      if [[ ! -d "$sam2_src_dir" ]]; then
        run git clone https://github.com/facebookresearch/sam2.git "$sam2_src_dir"
      fi
      run git -C "$sam2_src_dir" checkout 0e78a118995e66bb27d78518c4bd9a3e95b4e266
      run cp "$sam2_src_dir/sam2_configs/"*.yaml "$sam2_configs_dir/"
      ok "sam2_configs Hydra YAML files installed"
    fi
  fi

  if [[ "$DRY_RUN" == "0" ]]; then
    "$venv_dir/bin/python" -c \
      'import cv2, decord, diffusers, easydict, ftfy, hydra, iopath, loguru, onnxruntime, peft, PIL, sam2, torch, transformers, wan; from wan.configs import WAN_CONFIGS; assert "animate-14B" in WAN_CONFIGS' || \
      die "Wan Animate Python import smoke failed"
    ok "Wan Animate runtime imports verified"
  else
    ok "[dry run] would verify Wan Animate runtime imports"
  fi

  # FlashAttention-2 fallback: wan/modules/animate/clip.py's CLIP visual
  # encoder hardcodes flash_attention(..., version=2), which asserts
  # FLASH_ATTN_2_AVAILABLE unconditionally — it never falls back to FA3 even
  # when FA3 is installed and working. Without flash-attn (v2), Wan Animate
  # generation crashes with a bare AssertionError deep in attention.py the
  # first time a shot actually renders.
  if ! "$venv_dir/bin/python" -c "import flash_attn" 2>/dev/null; then
    log "Building FlashAttention-2 fallback for Wan Animate (~10-15 min, CUDA compilation)"
    run "$pip" install flash-attn --no-build-isolation || \
        warn "flash_attn build failed — Wan Animate's CLIP visual encoder requires FA2 and will fail at generation time"
  else
    ok "FlashAttention-2 fallback already installed"
  fi

  local fa3_sm90_check
  fa3_sm90_check='import torch, flash_attn_interface as fa3; assert torch.cuda.get_device_capability() == (9, 0); q=torch.randn((1,128,8,64),device="cuda",dtype=torch.bfloat16); fa3.flash_attn_func(q,q,q)'
  if ! "$venv_dir/bin/python" -c "$fa3_sm90_check" 2>/dev/null; then
    run env FLASHATTENTION_DISABLE_SM80=TRUE "$pip" install --force-reinstall --no-deps --no-build-isolation \
        "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3.post1#subdirectory=hopper"
  fi
  if [[ "$DRY_RUN" == "0" ]]; then
    "$venv_dir/bin/python" -c "$fa3_sm90_check" || die "Wan Animate FA3 sm_90a verification failed"
  fi

  # `hf download` resumes .incomplete files and verifies repository metadata.
  # Invoke it on every setup run: a merely non-empty 72 GB directory may still
  # be a partial download from an interrupted pod.
  log "Verifying/resuming Wan2.2-Animate-14B download (~72.4 GB total)"
  if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$model_dir"; fi
  run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
      hf download Wan-AI/Wan2.2-Animate-14B --local-dir "$model_dir"

  local checkpoints=(
    "$model_dir/config.json"
    "$model_dir/diffusion_pytorch_model-00001-of-00004.safetensors"
    "$model_dir/diffusion_pytorch_model-00002-of-00004.safetensors"
    "$model_dir/diffusion_pytorch_model-00003-of-00004.safetensors"
    "$model_dir/diffusion_pytorch_model-00004-of-00004.safetensors"
    "$model_dir/diffusion_pytorch_model.safetensors.index.json"
    "$model_dir/models_t5_umt5-xxl-enc-bf16.pth"
    "$model_dir/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "$model_dir/Wan2.1_VAE.pth"
    "$model_dir/relighting_lora.ckpt"
    "$model_dir/google/umt5-xxl/spiece.model"
    "$model_dir/xlm-roberta-large/sentencepiece.bpe.model"
    "$model_dir/process_checkpoint/pose2d/vitpose_h_wholebody.onnx"
    "$model_dir/process_checkpoint/det/yolov10m.onnx"
    "$model_dir/process_checkpoint/sam2/sam2_hiera_large.pt"
  )
  if [[ "$DRY_RUN" == "0" ]]; then
    local checkpoint
    for checkpoint in "${checkpoints[@]}"; do
      [[ -s "$checkpoint" ]] || die "Wan Animate checkpoint missing or empty: $checkpoint"
    done
    "$venv_dir/bin/python" -c \
      'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); index=json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text()); shards=set(index["weight_map"].values()); missing=[name for name in shards if not (root / name).is_file() or (root / name).stat().st_size == 0]; assert not missing, f"missing indexed DiT shards: {missing}"' \
      "$model_dir" || die "Wan Animate DiT shard index verification failed"
    ok "Wan Animate core model and preprocessing checkpoints verified"

    # Provider-list checks are insufficient: the CPU distribution can coexist
    # and a broken CUDA provider may still be advertised. Construct both real
    # preprocessing sessions with CUDA first and require it to remain active.
    # vitpose_h_wholebody.onnx is shipped upstream as a directory of external-data
    # tensor blobs plus an end2end.onnx graph file (HF Xet storage layout, not a
    # corrupt download) — mirror the os.path.isdir() resolution that
    # wan/modules/animate/preprocess/pose2d.py:SimpleOnnxInference already does,
    # since raw ort.InferenceSession() cannot open a directory path directly.
    "$venv_dir/bin/python" -c \
      'import onnxruntime as ort, os, sys; expected="CUDAExecutionProvider"; assert expected in ort.get_available_providers(), ort.get_available_providers(); paths=[os.path.join(path, "end2end.onnx") if os.path.isdir(path) else path for path in sys.argv[1:]]; sessions=[ort.InferenceSession(path, providers=[expected, "CPUExecutionProvider"]) for path in paths]; assert all(s.get_providers() and s.get_providers()[0] == expected for s in sessions), [s.get_providers() for s in sessions]' \
      "$model_dir/process_checkpoint/det/yolov10m.onnx" \
      "$model_dir/process_checkpoint/pose2d/vitpose_h_wholebody.onnx" || \
      die "Wan Animate detector/pose ONNX CUDA session smoke failed"
    ok "Wan Animate detector and pose ONNX CUDA sessions verified"
  fi

  if [[ "$SKIP_WAN_ANIMATE_FLUX" == "0" ]]; then
    local flux_dir="$model_dir/process_checkpoint/FLUX.1-Kontext-dev"
    run env HF_HUB_ENABLE_HF_TRANSFER=0 ${HF_TOKEN:+HF_TOKEN="$HF_TOKEN"} \
        hf download black-forest-labs/FLUX.1-Kontext-dev --local-dir "$flux_dir"
    if [[ "$DRY_RUN" == "0" ]]; then
      [[ -s "$flux_dir/model_index.json" ]] || \
        die "Wan Animate optional FLUX retarget model is incomplete: $flux_dir"
    fi
  fi
  ok "Wan2.2 Animate environment ready"
}

# ── Step 6d: LightX2V Wan 2.2 I2V acceleration ──────────────────────────────
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

# ── Step 6c2: AI video enhancement (Real-ESRGAN + RIFE/FILM) ────────────────
setup_rife_model() {
  log "Setting up RIFE pretrained HD model"

  local venv_dir="${1:-$WORKSPACE/.venv_video_enhance}"
  local rife_dir="${2:-$WORKSPACE/ECCV2022-RIFE}"

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for CUDA torch)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "AI video enhance venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip setuptools wheel

  if [[ ! -d "$rife_dir" ]]; then
    run git clone https://github.com/hzwer/ECCV2022-RIFE.git "$rife_dir"
  else
    ok "RIFE already cloned at $rife_dir"
    if [[ -d "$rife_dir/.git" ]]; then
      run git -C "$rife_dir" pull --ff-only || warn "RIFE git pull failed — continuing"
    else
      warn "$rife_dir is not a git checkout — continuing with existing files"
    fi
  fi

  # RIFE's upstream requirements.txt pins numpy<=1.23.5. That version has no
  # Python 3.12 wheel and fails during source-build setup with pkgutil.ImpImporter
  # removed. Keep RIFE on the modern shared video deps instead; torch is inherited
  # from the CUDA base image through --system-site-packages.
  run "$pip" install gdown opencv-python tqdm "numpy>=1.26,<2.3" ffmpeg-python moviepy "imageio[ffmpeg]" scikit-video || \
      warn "Some RIFE runtime deps may have failed"

  if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$rife_dir/train_log"; fi
  if compgen -G "$rife_dir/train_log/*.pkl" >/dev/null; then
    ok "RIFE model weights already present in $rife_dir/train_log"
    return
  fi

  log "Downloading RIFE pretrained HD model zip"
  local rife_zip="$WORKSPACE/RIFE_trained_model_v3.6.zip"
  local rife_tmp="$WORKSPACE/rife_model_extract"
  # Newer gdown releases removed/changed the old `--id <file_id>` form.
  # Use a positional Google Drive URL so the same command works across gdown
  # versions.
  run "$venv_dir/bin/gdown" "https://drive.google.com/uc?id=1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_" -O "$rife_zip"
  if [[ "$DRY_RUN" == "0" ]]; then
    rm -rf "$rife_tmp"
    mkdir -p "$rife_tmp"
    run "$venv_dir/bin/python" -m zipfile -e "$rife_zip" "$rife_tmp"
    run find "$rife_tmp" -type f \( -name "*.pkl" -o -name "RIFE*.py" \) -exec cp {} "$rife_dir/train_log/" \;
    rm -rf "$rife_tmp"
  fi
  if [[ "$DRY_RUN" != "0" ]]; then
    ok "[dry run] RIFE model zip would be extracted into $rife_dir/train_log"
  elif ! compgen -G "$rife_dir/train_log/*.pkl" >/dev/null; then
    warn "RIFE zip extracted but no *.pkl found; download manually into $rife_dir/train_log"
  else
    ok "RIFE model weights ready in $rife_dir/train_log"
  fi
}

setup_video_enhance() {
  log "Setting up AI video enhancement backends (Real-ESRGAN + RIFE/FILM)"
  # These run as short-lived subprocesses from adapters/video_enhance/ai_adapter.py,
  # not as resident services. That keeps VRAM free except during the
  # video_enhance stage.

  local venv_dir="$WORKSPACE/.venv_video_enhance"
  local realesrgan_dir="$WORKSPACE/Real-ESRGAN"
  local rife_dir="$WORKSPACE/ECCV2022-RIFE"
  local film_dir="$WORKSPACE/frame-interpolation"
  local film_venv="$WORKSPACE/.venv_film"
  local film_model_root="$WORKSPACE/FILM"

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating $venv_dir (system-site-packages for CUDA torch)"
    run python3 -m venv --system-site-packages "$venv_dir"
  else
    ok "AI video enhance venv already exists at $venv_dir"
  fi

  local pip="$venv_dir/bin/pip"
  run "$pip" install --upgrade pip setuptools wheel
  run "$pip" install gdown opencv-python tqdm "numpy>=1.26,<2.3" ffmpeg-python moviepy "imageio[ffmpeg]" scikit-video || \
      warn "Some shared video enhance deps may have failed"

  if [[ ! -d "$realesrgan_dir" ]]; then
    run git clone https://github.com/xinntao/Real-ESRGAN.git "$realesrgan_dir"
  else
    ok "Real-ESRGAN already cloned at $realesrgan_dir"
    if [[ -d "$realesrgan_dir/.git" ]]; then
      run git -C "$realesrgan_dir" pull --ff-only || warn "Real-ESRGAN git pull failed — continuing"
    else
      warn "$realesrgan_dir is not a git checkout — continuing with existing files"
    fi
  fi
  run "$pip" install -r "$realesrgan_dir/requirements.txt" || warn "Some Real-ESRGAN deps may have failed"
  run "$pip" install -e "$realesrgan_dir" || warn "Real-ESRGAN editable install failed"

  local realesrgan_weights="$realesrgan_dir/weights"
  if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$realesrgan_weights"; fi
  download_if_missing() {
    local url="$1" dest="$2" label="$3"
    if [[ -f "$dest" ]]; then
      ok "$label already exists at $dest"
    else
      log "Downloading $label"
      run curl -fL "$url" -o "$dest"
    fi
  }
  download_if_missing \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth" \
    "$realesrgan_weights/realesr-general-x4v3.pth" \
    "Real-ESRGAN general x4v3"
  download_if_missing \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth" \
    "$realesrgan_weights/realesr-general-wdn-x4v3.pth" \
    "Real-ESRGAN general weak-denoise x4v3"
  download_if_missing \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth" \
    "$realesrgan_weights/realesr-animevideov3.pth" \
    "Real-ESRGAN anime video v3"

  setup_rife_model "$venv_dir" "$rife_dir"

  local film_bootstrap="python3"
  if need_cmd python3.10; then
    film_bootstrap="python3.10"
  else
    warn "python3.10 not found — using python3 for FILM; upstream tested TF2/CUDA stack may require Python 3.9/3.10"
  fi
  if [[ ! -d "$film_venv" ]]; then
    log "Creating $film_venv (separate TensorFlow/FILM environment)"
    run "$film_bootstrap" -m venv "$film_venv"
  else
    ok "FILM venv already exists at $film_venv"
  fi
  local film_pip="$film_venv/bin/pip"
  run "$film_pip" install --upgrade pip setuptools wheel

  if [[ ! -d "$film_dir" ]]; then
    run git clone https://github.com/google-research/frame-interpolation.git "$film_dir"
  else
    ok "FILM already cloned at $film_dir"
    if [[ -d "$film_dir/.git" ]]; then
      run git -C "$film_dir" pull --ff-only || warn "FILM git pull failed — continuing"
    else
      warn "$film_dir is not a git checkout — continuing with existing files"
    fi
  fi
  run "$film_pip" install -r "$film_dir/requirements.txt" || warn "Some FILM deps may have failed"
  run "$film_pip" install gdown || warn "gdown install failed in FILM venv"

  if [[ -d "$film_model_root/film_net/Style/saved_model" ]]; then
    ok "FILM Style SavedModel already present at $film_model_root/film_net/Style/saved_model"
  else
    log "Downloading FILM pretrained SavedModels"
    if [[ "$DRY_RUN" == "0" ]]; then mkdir -p "$film_model_root"; fi
    run "$film_venv/bin/gdown" --folder \
        "https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy?usp=sharing" \
        -O "$film_model_root"
    if [[ "$DRY_RUN" == "0" ]] && [[ ! -d "$film_model_root/film_net/Style/saved_model" ]]; then
      local found_film_net
      found_film_net="$(find "$film_model_root" -type d -path "*/film_net/Style/saved_model" -print -quit 2>/dev/null || true)"
      if [[ -n "$found_film_net" ]]; then
        local nested_film_net
        nested_film_net="$(cd "$found_film_net/../.." && pwd)"
        if [[ ! -e "$film_model_root/film_net" ]]; then
          run ln -s "$nested_film_net" "$film_model_root/film_net"
          ok "Symlinked FILM model folder to $film_model_root/film_net"
        fi
      fi
    fi
    if [[ "$DRY_RUN" == "0" ]] && [[ ! -d "$film_model_root/film_net/Style/saved_model" ]]; then
      warn "FILM model folder did not land at $film_model_root/film_net/Style/saved_model; inspect $film_model_root"
    fi
  fi

  ok "AI video enhance setup complete"
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
    log "Creating $venv_dir (system-site-packages, inherits the preinstalled torch build)"
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
  #
  # DGX: LatentSync's requirements.txt hard-pins torch==2.5.1+cu121,
  # decord==0.6.0, and onnxruntime-gpu==1.21.0. None have Blackwell/aarch64
  # wheels: torch==2.5.1 predates Blackwell kernel support (min_cuda_for_wheels
  # is 12.8 on this GPU) and its cu121 index has no aarch64 CUDA build at all,
  # decord has zero aarch64 wheels, and onnxruntime-gpu has zero aarch64
  # wheels at any version. A plain `pip install -r requirements.txt` would
  # either fail outright or silently replace this box's working torch
  # 2.13.0+cu130 with a CPU-only aarch64 build. This needs a real port (patch
  # requirements.txt, swap in the decord shim, fall back to CPU onnxruntime
  # for the still-image face detector), not attempted here since it's opt-in
  # and unused by the default wan_s2v pipeline. Fail fast with guidance
  # instead of burning time mid-install.
  if [[ "$(uname -m)" == "aarch64" ]]; then
    die "LatentSync is not ported to aarch64/Blackwell yet on this script (torch==2.5.1+cu121, decord==0.6.0, onnxruntime-gpu==1.21.0 in its requirements.txt have no ARM wheels). Not needed for the default wan_s2v path anyway. Remove --with-latentsync, or edit setup_gpu_dgx.sh's setup_latentsync() to patch those pins if you need it."
  fi

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
    log "Creating $venv_dir (system-site-packages, inherits the preinstalled torch build)"
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
os.environ['HF_HUB_DISABLE_XET'] = '0'
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
# video_me GPU runtime settings — generated by setup_gpu_dgx.sh (DGX Spark / GB10)
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
VIDEO_ME_FLUX2_EDIT_ENABLED=$FLUX2_EDIT_ENABLED
VIDEO_ME_FLUX2_EDIT_MAX_REFERENCES=4
VIDEO_ME_COMFYUI_BASE_URL=http://localhost:8188

# Video: Wan2.2 S2V image+audio singing video (default) | wan_lightx2v fast I2V | wan I2V | ltx legacy
VIDEO_ME_VIDEO_ADAPTER=wan_s2v
VIDEO_ME_WAN_S2V_BASE_URL=http://localhost:8031
VIDEO_ME_WAN_LIGHTX2V_BASE_URL=http://localhost:8032
VIDEO_ME_WAN_ANIMATE_BASE_URL=http://localhost:8033
VIDEO_ME_WAN_ANIMATE_PYTHON=$WORKSPACE/.venv_wan_animate/bin/python
VIDEO_ME_WAN_ANIMATE_REPO_DIR=$WORKSPACE/Wan2.2
VIDEO_ME_WAN_ANIMATE_MODEL_DIR=$WORKSPACE/Wan2.2-Animate-14B
VIDEO_ME_WAN_ANIMATE_DATA_ROOT=$WORKSPACE/video_me/data
VIDEO_ME_WAN_S2V_FPS=16
WAN_S2V_OFFLOAD_MODEL=false
WAN_I2V_OFFLOAD_MODEL=false
WAN_ANIMATE_OFFLOAD_MODEL=true
# DGX: GB10 is Blackwell (compute capability 12.x); flash-attn-3's public
# package only builds Hopper (sm_90a) kernels, so it is never available here.
# Wan's own attention.py already falls back to flash-attn 2 / SDPA with no
# code changes -- this just tells the server wrapper not to refuse to load.
WAN_REQUIRE_FLASH_ATTN_3=false
LIGHTX2V_I2V_OFFLOAD_MODEL=false
LIGHTX2V_I2V_WIDTH=400
LIGHTX2V_I2V_HEIGHT=704
LIGHTX2V_I2V_STEPS=4
# DGX: sageattention support on Blackwell/aarch64 is unverified (LightX2V is
# opt-in and not part of the default stack); fall back to torch SDPA if the
# sageattention build fails or the LightX2V server errors on this attn mode.
LIGHTX2V_I2V_ATTN_MODE=sage_attn2
VIDEO_ME_VIDEO_UPSCALE_ENABLED=false
VIDEO_ME_VIDEO_UPSCALE_TARGET_FPS=48
VIDEO_ME_VIDEO_UPSCALE_INTERPOLATION=minterpolate
VIDEO_ME_VIDEO_ENHANCE_ENABLED=false
VIDEO_ME_VIDEO_ENHANCE_ADAPTER=ffmpeg
VIDEO_ME_VIDEO_ENHANCE_TARGET_FPS=48
VIDEO_ME_VIDEO_ENHANCE_INTERPOLATION=minterpolate
VIDEO_ME_VIDEO_ENHANCE_REALESRGAN_PYTHON=$WORKSPACE/.venv_video_enhance/bin/python
VIDEO_ME_VIDEO_ENHANCE_REALESRGAN_DIR=$WORKSPACE/Real-ESRGAN
VIDEO_ME_VIDEO_ENHANCE_REALESRGAN_MODEL=realesr-general-x4v3
VIDEO_ME_VIDEO_ENHANCE_REALESRGAN_OUTSCALE=2.0
VIDEO_ME_VIDEO_ENHANCE_REALESRGAN_TILE=256
VIDEO_ME_VIDEO_ENHANCE_RIFE_PYTHON=$WORKSPACE/.venv_video_enhance/bin/python
VIDEO_ME_VIDEO_ENHANCE_RIFE_DIR=$WORKSPACE/ECCV2022-RIFE
VIDEO_ME_VIDEO_ENHANCE_RIFE_MODEL_DIR=$WORKSPACE/ECCV2022-RIFE/train_log
VIDEO_ME_VIDEO_ENHANCE_RIFE_EXP=1
VIDEO_ME_VIDEO_ENHANCE_RIFE_SCALE=1.0
VIDEO_ME_VIDEO_ENHANCE_RIFE_FP16=true
VIDEO_ME_VIDEO_ENHANCE_FILM_PYTHON=$WORKSPACE/.venv_film/bin/python
VIDEO_ME_VIDEO_ENHANCE_FILM_DIR=$WORKSPACE/frame-interpolation
VIDEO_ME_VIDEO_ENHANCE_FILM_MODEL_PATH=$WORKSPACE/FILM/film_net/Style/saved_model
VIDEO_ME_VIDEO_ENHANCE_FILM_TIMES_TO_INTERPOLATE=2
VIDEO_ME_VIDEO_ENHANCE_FILM_BLOCK_HEIGHT=1
VIDEO_ME_VIDEO_ENHANCE_FILM_BLOCK_WIDTH=1
VIDEO_ME_VIDEO_ENHANCE_LATENT_WORKFLOW=assets/comfyui_workflows/video_enhance_latent.json

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

if [[ "$DOWNLOAD_RIFE_MODEL_ONLY" == "1" ]]; then
  setup_rife_model
  log "RIFE model-only setup complete"
  exit 0
fi

check_cuda

if [[ "$SKIP_SYSTEM_DEPS" == "0" ]]; then
  install_system_deps
fi

if [[ "$SKIP_CUDA_LIBS" == "0" ]]; then
  install_missing_cuda_libs
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

if [[ "$SKIP_WAN_ANIMATE" == "0" ]]; then
  setup_wan_animate
fi

if [[ "$SKIP_LIGHTX2V" == "0" ]]; then
  setup_lightx2v
fi

if [[ "$SKIP_VIDEO_ENHANCE" == "0" ]]; then
  setup_video_enhance
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
