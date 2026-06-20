#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=0
SKIP_SYSTEM_DEPS=0
SKIP_PYTHON_DEPS=0
SKIP_SERVICES=0
ALLOW_MISSING_SERVICES=0
CODE_TEST=0
NO_VENV=0
TIMEOUT="${VIDEO_ME_READINESS_TIMEOUT:-3.0}"
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_gpu.sh [options]

Installs/validates the dependencies needed for a video_me GPU run:
  - Python runtime extras: openai, httpx, faster-whisper, yt-dlp, storage deps
  - System ffmpeg package, which provides ffmpeg and ffprobe
  - Final runtime readiness check

Options:
  --dry-run                 Print commands without executing them
  --skip-system-deps        Do not install ffmpeg/ffprobe
  --skip-python-deps        Do not install Python runtime extras
  --skip-services           Skip model-service HTTP checks in readiness
  --allow-missing-services  Treat missing model services as warnings
  --code-test               Accept TEST-ONLY placeholder LoRAs for smoke tests
  --no-venv                 Use current Python instead of creating/using .venv
  --python PATH             Python executable to use/bootstrap with
  --timeout SECONDS         Readiness HTTP timeout, default 3.0
  -h, --help                Show this help

Common commands:
  bash scripts/setup_gpu.sh --dry-run
  bash scripts/setup_gpu.sh
  bash scripts/setup_gpu.sh --code-test --skip-services
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-system-deps)
      SKIP_SYSTEM_DEPS=1
      shift
      ;;
    --skip-python-deps)
      SKIP_PYTHON_DEPS=1
      shift
      ;;
    --skip-services)
      SKIP_SERVICES=1
      shift
      ;;
    --allow-missing-services)
      ALLOW_MISSING_SERVICES=1
      shift
      ;;
    --code-test)
      CODE_TEST=1
      shift
      ;;
    --no-venv)
      NO_VENV=1
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a path"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || die "--timeout requires seconds"
      TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

select_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  if need_cmd python3; then
    command -v python3
    return
  fi
  if need_cmd python; then
    command -v python
    return
  fi
  die "python3/python not found. Install Python 3.11+ first."
}

install_system_deps() {
  if need_cmd ffmpeg && need_cmd ffprobe; then
    log "ffmpeg/ffprobe already available"
    return
  fi

  log "Installing ffmpeg system package"
  if need_cmd apt-get; then
    run sudo apt-get update
    run sudo apt-get install -y ffmpeg
    return
  fi
  if need_cmd brew; then
    run brew install ffmpeg
    return
  fi

  die "No supported package manager found. Install ffmpeg manually, then rerun this script."
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
  fi

  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  export PATH="$ROOT_DIR/.venv/bin:$PATH"
}

install_python_deps() {
  log "Installing Python runtime dependencies"
  run "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  run "$PYTHON_BIN" -m pip install -e '.[services,ingest,transcribe,llm,render]'
}

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

log "video_me GPU setup"

if [[ "$SKIP_SYSTEM_DEPS" == "0" ]]; then
  install_system_deps
fi

setup_python_env

if [[ "$SKIP_PYTHON_DEPS" == "0" ]]; then
  install_python_deps
fi

run_readiness

log "Done"
