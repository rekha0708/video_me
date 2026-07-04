---
name: project-next-steps
description: "Pipeline run status as of 2026-07-04 — Track B complete, wan-adapter OOM blocking render_character"
metadata: 
  node_type: memory
  type: project
  originSessionId: dea3ed26-190c-4d16-812a-c23f9f1ff2c9
---

**Track B is now READY** (as of 2026-07-04, via `python3 -m scripts.check_track_b`): both
`loras/kids_duo_max.safetensors` and `loras/kids_duo_zoe.safetensors` present, both voice WAVs
present. This supersedes the older "INCOMPLETE — Max LoRA missing" state from 2026-06-29 —
CLAUDE.md in the repo may still say INCOMPLETE if it hasn't been regenerated since training
finished; trust the live `check_track_b` output over the doc.

**Dashboard runs on port 8765, not 8080.** The `video_me_agent` skill file
(`.claude/skills/video_me_agent.md`) documents 8080 as the dashboard API port — that's stale/wrong
for this deployment. The actual running services (`services.dashboard_api` + `services.dashboard_worker`,
started via `scripts/restart_dashboard.sh`) bind to **8765**. Browser UI is also served at `http://localhost:8765/`.

**Latest run: job `20260704-020418-33o`, phase=all, source=`file:///workspace/downloads/learn_body_parts_with_rosie_fun_kids_act.mp4`.**
- transcribe → analyze_content → adapt_script → plan_shots: all completed normally (critique loop hit
  its 3-iteration cap on pacing/character_fit, proceeded with best storyboard — 15 shots).
  Human storyboard approval: approved via `POST /api/jobs/{id}/approve`.
- render_character: Max's 3 candidates rendered successfully (musubi-tuner Flux 2.0, ~4min/candidate —
  see [[feedback_musubi_render_perf]] for why). Then **job failed with CUDA OOM** starting Zoe's render.
  Root cause: [[project-wan22]] — using `video_adapter=wan` (explicit override for this run, since we
  wanted to test wan despite MuseTalk's confirmed lip-sync failure) keeps Wan permanently resident
  (~52 GiB) unlike the default `ltx` stack, and Ollama auto-reloads `qwen3.6:35b` for image critique
  between shots — together they left no VRAM headroom for Flux 2.0.
- **Status: unresolved, paused for a decision — parked as a TODO to resume 2026-07-05.** Root cause
  refined further: `services/wan_server.py` loads WanI2V into VRAM once at **process startup**
  (`lifespan` hook) and keeps it resident for the whole session, even though Wan is only actually
  used in Phase B (`generate_video`) — well after `render_character`/`critique_images`/image-approval
  (Phase A) finish for every shot. The ~52 GiB it holds is likely inflated by leftover CUDA
  caching-allocator memory from earlier Wan test runs (not something it actively needs at rest).
  Next-session plan: either (1) ops-level — stop Wan (and MuseTalk, confirmed non-functional anyway)
  before running render_character, restart Wan only right before generate_video; or (2) code-level —
  add stage-boundary start/stop hooks in `core/workflow.py` mirroring the existing pattern that
  already unloads `qwen3.6:35b` from Ollama before the shot loop. Also found a doc/code mismatch in
  `services/wan_server.py`: docstring line 15 says `offload_model=False`, but `_inference()` line 134
  actually passes `offload_model=True` — reconcile regardless of which option is picked.
  Job/queue/worker are otherwise healthy — this is purely a VRAM budgeting issue, not a code bug.

**Gotcha found this session:** the local file the user initially pointed at
(`/workspace/downloads/learn_body_parts_with_rosie_fun_kids_act.f399.mp4`) only existed in the
project checkout's own `/workspace/video_me/downloads/`, not the canonical `/workspace/downloads/`
that `core/config.py`'s `local_video_dir` and all prior jobs use. Not a code bug — just check which
`downloads/` you're `ls`-ing (shell cwd matters).

**Why:** [[project-video-me]], [[project-wan22]], [[project-lora-training]]
