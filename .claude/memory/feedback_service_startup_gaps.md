---
name: feedback-service-startup-gaps
description: "start_services.sh can fail on ComfyUI/Fish/Wan due to a few missing pip packages, not a full wipe"
metadata:
  type: feedback
---

`bash scripts/start_services.sh` sometimes fails to bring up ComfyUI, Fish Audio S2, or Wan2.2 even though torch and ~140 other system packages are intact — it's usually just 1-2 missing pip packages, not a full environment wipe.

Seen on 2026-07-03: system Python was missing `sqlalchemy` (breaks ComfyUI's `app/assets` import chain), and both `.venv_fish_s2` and `.venv_wan` were missing `click` (breaks uvicorn's CLI entrypoint even though uvicorn itself was installed).

**Why:** Only the Ollama binary is documented as wiped on RunPod pod restart ([[project_video_me]]); in practice small individual pip packages can also go missing from system Python and from /workspace venvs between sessions, cause unclear.

**How to apply:** If `start_services.sh` reports ComfyUI/Fish/Wan not responding, check `/workspace/logs/{comfyui,fish_s2,wan}.log` for `ModuleNotFoundError` before assuming a bigger problem. Fix is a quick targeted `pip install <missing-pkg>` (system pip3 for ComfyUI, `.venv_fish_s2/bin/pip` or `.venv_wan/bin/pip` for those), then re-run start_services.sh — do NOT reinstall from scratch. Also: Fish Audio S2 takes ~60-90s to load the 20GB model and warm up before `/health` responds — a "did not respond after 60s" warning from the script doesn't always mean failure, check the log for a benign "loading weights" state before treating it as broken.
