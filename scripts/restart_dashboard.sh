#!/usr/bin/env bash
# Restart the dashboard API server + worker — run this after every `git pull`
# that touches core/workflow.py, services/dashboard_*.py, or adapters/.
#
# Neither process auto-reloads: the API server used to run with --reload, but
# on this pod that reloader repeatedly hung at "Waiting for connections to
# close" whenever a browser tab held an open SSE stream (/api/jobs/*/stream)
# — every edit under core/ or services/ silently froze the whole dashboard,
# broke Cancel/Approve buttons, and made in-progress jobs look stuck with no
# way to tell they weren't. Both the API and the worker now need this script
# rerun after every code change that touches them.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
LOG_DIR="$WORKSPACE/logs"
DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
mkdir -p "$LOG_DIR"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[0;32m  ✓ %s\033[0m\n' "$*"; }
warn(){ printf '\033[0;33m  ! %s\033[0m\n' "$*"; }

stop_matching() {
  local label="$1" pattern="$2"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -z "$pids" ]]; then
    ok "$label already stopped"
    return
  fi
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 10); do
    pgrep -f "$pattern" >/dev/null 2>&1 || { ok "$label stopped"; return; }
    sleep 1
  done
  warn "$label did not stop within 10s — force killing"
  kill -9 $pids 2>/dev/null || true
}

# Belt-and-suspenders for stop_matching(): a stuck/hung process (or a stray
# leftover from a past run) can keep the listening socket even if its cmdline
# no longer matches the pgrep pattern above — pgrep -f then never finds it,
# and it silently keeps serving stale code forever on every future restart.
# Kill by port too.
stop_port() {
  local label="$1" port="$2"
  local pids
  pids="$(ss -ltnp 2>/dev/null | awk -v p=":$port\$" '$4 ~ p {print $0}' | \
    { grep -oP 'pid=\K[0-9]+' || true; } | sort -u)"
  if [[ -z "$pids" ]]; then
    return
  fi
  warn "$label: killing stray process(es) still bound to port $port: $pids"
  kill -9 $pids 2>/dev/null || true
}

log "Stopping dashboard API + worker"
stop_matching "Dashboard API"    "uvicorn services.dashboard_api"
stop_matching "Dashboard worker" "python -m services.dashboard_worker"
stop_port "Dashboard API" "$DASHBOARD_PORT"

cd "$ROOT_DIR"

log "Starting Dashboard API (port $DASHBOARD_PORT)"
nohup .venv/bin/uvicorn services.dashboard_api:create_app --factory \
  --port "$DASHBOARD_PORT" --host 0.0.0.0 \
  >"$LOG_DIR/dashboard_api.log" 2>&1 &
disown
ok "Dashboard API starting (log: $LOG_DIR/dashboard_api.log)"

log "Starting Dashboard worker"
nohup .venv/bin/python -m services.dashboard_worker \
  >"$LOG_DIR/dashboard_worker.log" 2>&1 &
disown
ok "Dashboard worker starting (log: $LOG_DIR/dashboard_worker.log)"

printf '\nWaiting for Dashboard API to respond...\n'
for i in $(seq 1 15); do
  if curl -sf "http://localhost:$DASHBOARD_PORT/" >/dev/null 2>&1; then
    ok "Dashboard API responding at http://localhost:$DASHBOARD_PORT/"
    exit 0
  fi
  sleep 1
done
warn "Dashboard API did not respond after 15s — check $LOG_DIR/dashboard_api.log"
exit 1
