# video_me — Web UI Plan

Future effort. Purpose: replace the CLI (`run_pipeline.py`) with a browser UI for pipeline
runs, parameter tweaking, and output review — without changing any pipeline code.

---

## Goals

| Goal | Detail |
|---|---|
| Run the pipeline | Upload a local file or paste a URL; set rights-cleared flag |
| Tweak parameters | Pick LLM model, shot duration range, words/sec, Whisper device |
| Monitor progress | Live stage-by-stage status (fetch → … → publish) with elapsed time |
| Review outputs | Browse generated images and videos per shot; download final video |
| Job history | List past jobs, their status, and output links |

---

## Architecture (proposed)

```
Browser
  │
  │  HTTP/WebSocket
  ▼
FastAPI UI server  (new: services/ui_server.py)
  │
  ├── POST /jobs          — start a new pipeline job
  ├── GET  /jobs          — list all jobs
  ├── GET  /jobs/{id}     — job detail + stage statuses
  ├── GET  /jobs/{id}/artifacts/{path} — serve a file artifact
  ├── GET  /jobs/{id}/download  — stream final MP4
  └── WS   /jobs/{id}/stream   — live log tail (SSE or WebSocket)
  │
  └── calls run_pipeline_job() in a background asyncio task
        (same core/workflow.py — no changes needed)
```

Frontend: single-page, no build step. Plain HTML + vanilla JS + a lightweight CSS framework
(e.g. Pico CSS). No React/Vue — keeps it zero-dependency for the ops environment.

---

## Pages / Views

### 1. New Run

- **Source input**: URL text box OR file upload (drag-and-drop `.mp4`)
- **Rights cleared**: checkbox (must tick before submit)
- **LLM model**: dropdown — pre-populated from `GET /api/ollama/models`
  - Default: `qwen3.6:35b`; options: any model present in Ollama
- **Whisper device**: radio — `cuda` / `cpu`
- **Shot duration**: two sliders — min (2–8s) and max (5–15s)
- **Words per second**: number input (default 2.0)
- **[Run Pipeline]** button → POST /jobs → redirect to job detail page

### 2. Job Detail / Live Monitor

- Header: job ID, source URL/filename, started-at, elapsed
- **Stage pipeline**: horizontal stepper showing each stage with status icon
  - ⏳ pending / 🔄 running / ✅ done / ❌ failed / 🚫 blocked
  - Click a stage chip to expand its log lines
- **Live log tail**: scrolling console panel via WebSocket / SSE
- **Shot grid** (appears after plan_shots completes):
  - One card per shot: shot ID, speaker, duration, camera angle
  - Each card shows thumbnails as they are generated:
    - Render image (PNG from render_character)
    - Video preview (MP4 from generate_video, autoplay muted loop)
    - Synced video (MP4 from lip_sync)
- **Download** button (appears after publish): streams final assembled MP4

### 3. Job History

- Table: job ID | source | status | started | duration | actions
- Actions: View detail | Download | Delete
- Filter by status; sort by date

### 4. Settings (optional, phase 2 of UI)

- Live service health panel (Ollama, A1111, Chatterbox, Wan, MuseTalk)
- Default parameter presets (save per-channel profile)

---

## API design

### POST /api/jobs
```json
{
  "source": "file:///workspace/downloads/video.mp4",  // or https://...
  "rights_cleared": true,
  "overrides": {
    "llm_model": "qwen3.6:35b",
    "whisper_device": "cuda",
    "min_shot_sec": 5.0,
    "max_shot_sec": 8.0,
    "words_per_sec": 2.0
  }
}
```
Returns `{"job_id": "20260625-200336-1js"}`. Starts `run_pipeline_job()` in a background task.

### GET /api/jobs/{id}
Returns job status, per-stage results, artifact paths.

### GET /api/jobs/{id}/artifacts/{path}
Serves files from `.local/jobs/{id}/` (renders, clips, synced clips).

### GET /api/jobs/{id}/download
Streams `review/<stem>/video.mp4` with `Content-Disposition: attachment`.

### WS /api/jobs/{id}/stream
Tails the pipeline log in real time; closes when job reaches terminal state.

---

## Parameter overrides (wiring into workflow)

The UI overrides are passed via environment or a per-job config dict. Cleanest approach:
extend `Settings` with per-job overrides applied in `_make_adapters()`:

```python
# core/config.py — add optional per-job overrides
class JobOverrides(BaseModel):
    llm_model: str | None = None
    whisper_device: str | None = None
    min_shot_sec: float | None = None
    max_shot_sec: float | None = None
    words_per_sec: float | None = None

# core/workflow.py — pass to run_pipeline_job()
async def run_pipeline_job(..., overrides: JobOverrides | None = None)
```

`plan_shots/llm_adapter.py` reads `min_shot_sec`/`max_shot_sec`/`words_per_sec` from the
adapter constructor — just pass them down from settings if overrides are present.

---

## Implementation phases

### Phase UI-1 — Minimal viable (start here)
- [ ] `services/ui_server.py` — FastAPI with `/api/jobs` CRUD + background task runner
- [ ] `static/index.html` — New Run form + job list (no live updates yet)
- [ ] `static/job.html` — Job detail with stage stepper (polls `/api/jobs/{id}` every 5s)
- [ ] Artifact serving endpoint (images + videos)
- [ ] Download endpoint for final MP4

**Effort estimate:** ~2–3 days. No pipeline changes needed.

### Phase UI-2 — Live updates + shot grid
- [ ] WebSocket / SSE log tail (`/api/jobs/{id}/stream`)
- [ ] Shot grid with per-shot thumbnails populated as shots complete
- [ ] Video preview cards (autoplay muted loop for render + synced clips)

**Effort estimate:** ~1–2 days on top of UI-1.

### Phase UI-3 — Parameter tweaking + model selector
- [ ] Ollama model list endpoint (`GET /api/models` proxying `GET /api/tags`)
- [ ] Sliders/inputs for shot duration and words/sec wired into `JobOverrides`
- [ ] `JobOverrides` propagated through `run_pipeline_job()` + `_make_adapters()`

**Effort estimate:** ~1 day on top of UI-2.

### Phase UI-4 — Job history + settings panel
- [ ] History table with filter/sort (backed by existing SQLite job store)
- [ ] Service health panel (proxy the 5 `/health` endpoints)
- [ ] Default preset save/load

**Effort estimate:** ~1 day on top of UI-3.

---

## Tech choices rationale

| Choice | Reason |
|---|---|
| FastAPI for UI server | Already used for Wan/Chatterbox/MuseTalk services; same pattern |
| Vanilla JS, no build step | Ops environment has no Node; zero toolchain friction |
| Pico CSS | ~10KB, semantic HTML styling, no class-soup |
| SSE over WebSocket for logs | Simpler server-side; one-directional log stream |
| Poll `/api/jobs/{id}` for stage status | Simple; stage updates are infrequent (minutes apart) |
| No separate DB for UI | Reuses existing SQLite job store via `core/storage.py` |

---

## Open questions (decide before UI-1)

1. **Port**: run UI server on 8080? Or consolidate with an existing service?
2. **Auth**: none (local use only) or simple token for remote access?
3. **File upload vs path input**: upload streams to `.local/uploads/`; path input requires the file
   to already be on the server. For RunPod use, path input is simpler.
4. **Video preview format**: browsers support MP4/H.264; confirm Wan output is H.264 not H.265.
