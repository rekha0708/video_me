# Plan: Dashboard Robustness + New Input Modes + OOM Fix

## Context

The video_me dashboard works well for step-by-step phase runs but has gaps when running end-to-end (`phase="all"`) — the phase stepper stays gray because the worker only records `"all"` as a completed phase, not the individual sub-phases. Additionally, the pipeline currently requires a video URL (for transcription) and LoRA-trained models (for image generation), which limits flexibility. The user wants two new input modes: paste a story directly (skip transcription) and upload reference images directly (skip LoRA rendering). Finally, running Flux 2.0 render + Wan 2.2 video causes OOM because models compete for GPU memory.

Five independently-shippable changes, ordered by priority:

---

## Change 1: Fix phase stepper for `phase="all"` (bug fix)

**Root cause:** Template checks `phase_key == job.phase` (line 68 of `job_detail.html`), but `job.phase` is `"all"`. Worker calls `append_completed_phase(job_id, "all")`, not individual phase names.

### Worker — record sub-phase completions

File: `services/dashboard_worker.py`

- Define a constant mapping at module level:
  ```
  STAGE_TO_PHASE = {
    "fetch_media": "transcribe", "transcribe": "transcribe", "analyze_content": "transcribe",
    "adapt_script": "script_plan", "plan_shots": "script_plan",
    "render_character": "render", "synthesize_voice": "render", "generate_video": "render",
    "assemble_video": "assemble", "publish": "assemble",
  }
  PHASE_FINAL_STAGE = {
    "transcribe": "analyze_content", "script_plan": "plan_shots",
    "render": "generate_video", "assemble": "publish",
  }
  ```
- In `_make_stage_hook` (line ~174): when `job_phase == "all"` and `event_type == "stage_completed"`, check if `stage_name` is the final stage for a phase via `PHASE_FINAL_STAGE`. If so, call `self.repo.append_completed_phase(job_id, parent_phase)` and emit a `phase_completed` event.
- At line ~265 (pipeline success for `phase="all"`): also call `append_completed_phase` for all four phases as a safety net.

### Template — derive active phase from current_stage

File: `services/templates/job_detail.html`

- Before the stepper loop (line ~63), add a Jinja2 block that maps `job.current_stage` → active phase when `job.phase == "all"`, using the same stage→phase mapping.
- Change the "active" condition at line 68 from `phase_key == job.phase` to `phase_key == active_phase`.

### Gap 1A: SSE live update for phase stepper

**Problem:** The phase stepper is server-rendered Jinja2. When a sub-phase completes during a running job, the browser doesn't know — it only updates stage timeline dots via `updateTimeline()` in `app.js:302`. There is no `updatePhaseStepper()` function. The page reloads only on terminal status (`done` SSE event at line 82).

**Solution:** Add a `updatePhaseStepper()` JS function in `app.js` that:
1. Listens for `phase_completed` events in the SSE `onmessage` handler (line 66-75).
2. Finds the `.phase-step` element matching the completed phase and flips it from pending/active → done.
3. Finds the *next* phase in order and flips it from pending → active.

```javascript
function updatePhaseStepper(stageName, eventType) {
  if (eventType !== 'phase_completed') return;
  // Find the phase step node by phase name (from stage_name field on the event)
  document.querySelectorAll('.phase-step').forEach(el => {
    const label = el.querySelector('.phase-label');
    // ... match phase name, update class/icon
  });
}
```

Alternatively (simpler): Since the SSE `done` event already triggers `location.reload()` at line 82 when a job reaches terminal status, and phase completions during a long run only matter mid-run, we can add a lighter approach: on each SSE `stage_completed` event, also check if the completed stage is a phase-final stage (using a JS copy of the `PHASE_FINAL_STAGE` map), and if so, update the corresponding phase dot to "done" and the next phase to "active". This is ~15 lines of JS.

**Files:** `services/static/app.js` — add `updatePhaseStepper()`, call it from `onmessage` handler.

**~55 lines total across 3 files. Test: extend `tests/test_dashboard_worker.py`.**

---

## Change 2: Model swap gating between render and video (OOM fix)

**Root cause:** Wan 2.2 server loads models at startup and holds them resident. When musubi-tuner subprocesses also load Flux 2.0 onto GPU, they compete for VRAM. Musubi-tuner exits naturally (freeing GPU), but there's no delay/check before Phase B starts.

### Config

File: `core/config.py` — add to `Settings`:
```python
model_swap_enabled: bool = False       # VIDEO_ME_MODEL_SWAP_ENABLED
model_swap_delay_sec: int = 30         # VIDEO_ME_MODEL_SWAP_DELAY_SEC
```

### Workflow gate

File: `core/workflow.py` — insert between line 905 (after `_run_image_approval_gate`) and line 907 (before Phase B loop):

Add `async def _model_swap_gate(config, adapters)` that:
1. Logs "waiting Ns for GPU memory to settle"
2. `await asyncio.sleep(delay)`
3. Calls `adapters.video.health()` — if down, retry once after another delay, then raise `RuntimeError`
4. Logs "video adapter ready"

Call it conditionally: `if config.settings.model_swap_enabled: await _model_swap_gate(...)`.

### Gap 2A: Wan server holds GPU memory even when idle

**Problem:** The Wan server loads the WanI2V pipeline at startup (`wan_server.py:58-85`) and keeps it resident for the entire process lifetime. Even with `t5_cpu=True` and `init_on_cpu=True`, there's a CUDA context + some GPU buffers. The delay-only approach assumes the Wan server's idle GPU footprint is small enough that musubi-tuner can coexist. In practice on 80 GB GPUs this may still OOM.

**Solution:** Add an `/unload` and `/reload` endpoint to the Wan server (`services/wan_server.py`):

```python
@app.post("/unload")
async def unload_model():
    """Evict the pipeline from memory to free GPU for other tasks."""
    global _pipeline
    _pipeline = None
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return {"status": "unloaded"}

@app.post("/reload")
async def reload_model():
    """Reload the pipeline after an unload."""
    global _pipeline
    if _pipeline is not None:
        return {"status": "already_loaded"}
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_pipeline)
    if _pipeline is None:
        raise HTTPException(500, detail=_pipeline_error or "reload failed")
    return {"status": "loaded"}
```

Then update `_model_swap_gate` in `core/workflow.py` to:
1. Before Phase A: if `video_adapter == "wan"`, call `POST wan_base_url/unload` to free GPU.
2. After Phase A: call `POST wan_base_url/reload`, then `health()` check.
3. If `video_adapter == "ltx"`: ComfyUI manages its own model lifecycle, so just do the delay + health-check (no explicit unload/reload).

The workflow should call unload *before* render starts (not after), so Flux 2.0 subprocesses have full GPU:

```python
if config.settings.model_swap_enabled:
    await _model_swap_pre_render(config, adapters)   # unload video model
# Phase A: render loop
...
# Phase B: 
if config.settings.model_swap_enabled:
    await _model_swap_post_render(config, adapters)  # reload + health-check
```

### Gap 2B: Reload takes 4-5 minutes for Wan

**Problem:** `_load_pipeline()` takes 4-5 min for Wan's first load (108 GB of DiT weights from disk → CPU RAM). The reload after unload will be similar.

**Solution:** This is an inherent cost — but it's better than OOM. Log the expected wait time clearly: `"Reloading Wan model — this takes ~4-5 min..."`. The `_model_swap_post_render` function should:
1. Log the expected duration
2. Call `/reload` (blocks until loaded)  
3. Call `health()` to confirm
4. Emit a `stage_hook("model_swap", "stage_completed")` event so the dashboard shows progress

For LTX: ComfyUI reloads models from disk on-demand per workflow node, so no explicit reload is needed — just the delay + health-check suffices.

**~60 lines across 3 files (config + workflow + wan_server). Test: mock `asyncio.sleep` + `health()` + unload/reload calls.**

---

## Change 3: Direct story/text input (skip transcription)

User chose: **paste text in UI** (textarea in the dashboard).

### Model changes

File: `core/models/dashboard.py`

- `DashboardSource.kind`: add `"text"` → `Literal["url", "upload", "file", "text"]`
- Add `text: str | None = None` field
- Replace `require_url` field_validator with a `model_validator(mode="after")`:
  - `kind == "text"` → require non-empty `text`, set `url = "text://direct-input"` if empty
  - Otherwise → require non-empty `url` (existing behavior)
- `DashboardJobRecord.source_kind`: add `"text"` to Literal

### Dashboard UI

File: `services/templates/jobs_list.html`

- Add third radio option "Paste Story" in the source section
- Add a `<textarea id="nj-text">` row (hidden by default, shown when "Paste Story" selected)
- Update `toggleSourceMode()` to handle three modes (url/file/text)
- Update `submitNewJob()` to pass `{kind: "text", url: "", text: storyText}`

### Worker + Workflow

File: `core/workflow.py`

- Add `source_text: str | None = None` to `_JobContext` dataclass
- In `run_pipeline_job`, accept `source_text` param, set it on context
- In `_run_to_assembled_video`, when `source_text` is set:
  - Skip `fetch_media` and `transcribe` stages
  - Construct synthetic `TranscribeResult(segments=[...], full_text=source_text)`
  - Save synthetic artifacts to `artifact_store` (so resume works)
  - Proceed to `analyze_content` normally

File: `services/dashboard_worker.py`

- In `_run_pipeline`, when `req.source.kind == "text"`, extract `req.source.text` and pass to `run_pipeline_job(source_text=...)`.

### Gap 3A: `analyze_content` derives duration from transcript segments

**Problem:** In `adapters/analyze_content/llm_adapter.py:167`, the adapter computes `duration = req.transcript.segments[-1].end`. For a pasted story, all segments have `start=0.0, end=0.0`, so duration will be `0.0`. The LLM prompt template at line 28 says `Transcript ({duration:.0f}s, language: {language})` — it will say "0s". This won't crash, but the LLM may give odd pacing analysis for a "0 second" transcript.

**Solution:** When constructing the synthetic `TranscribeResult`:
- Estimate duration from word count: `duration_est = max(60.0, len(source_text.split()) / 2.5)` (assuming ~2.5 words/sec for kids content). Use this as the `end` time of the last segment.
- Or split the text into multiple segments (e.g., one per paragraph/line) with estimated timestamps. This gives the LLM better structure to work with.

Recommended approach — split into paragraphs:
```python
paragraphs = [p.strip() for p in source_text.split('\n\n') if p.strip()]
if not paragraphs:
    paragraphs = [source_text.strip()]
segments = []
cursor = 0.0
for p in paragraphs:
    words = len(p.split())
    dur = max(2.0, words / 2.5)
    segments.append(TranscriptSegment(text=p, start=cursor, end=cursor + dur))
    cursor += dur
transcribe_result = TranscribeResult(
    segments=segments,
    language=opts.language or "en",
    full_text=source_text,
)
```

This gives the analyze_content adapter realistic timing data.

### Gap 3B: Transcript review gate fires for text sources (redundant)

**Problem:** When `kind="text"` and `phase="all"`, the worker calls `_run_transcript_review_gate` after the transcribe phase completes (`dashboard_worker.py:246-248`). But the user just pasted this text — reviewing it is redundant. The approval card would show the user's own text back to them.

**Solution:** Skip the transcript review gate when `source.kind == "text"`. In `dashboard_worker.py:246`:
```python
if phase == "transcribe" and req.source.kind != "text":
    await self._run_transcript_review_gate(req, job_id)
    return
```

When `kind == "text"` and `phase == "transcribe"`, fall through to the normal `COMPLETED` path.

For `phase == "all"` with `kind == "text"`: the transcribe sub-phase is synthetic (skipped in workflow), so `_make_stage_hook` won't fire `stage_completed` for `analyze_content` since the stage was never formally run via `run_stage`. To fix this:
- In the workflow, after constructing the synthetic transcript and running `analyze_content`, emit the stage hook for `fetch_media`, `transcribe`, and `analyze_content` as completed (so the stage timeline and phase stepper update).
- Call `opts.stage_hook("fetch_media", "stage_started")` and `opts.stage_hook("fetch_media", "stage_completed")` etc. for the skipped stages.

### Gap 3C: Stage timeline shows Fetch/Transcribe dots that never complete

**Problem:** When `phase="all"` with text source, the stage timeline shows all 10 stages including `Fetch` and `Transcribe`. These will show as gray dots forever if we silently skip them.

**Solution:** Two options:
1. **Auto-mark as skipped (recommended):** In the workflow, when `source_text` is set, fire the stage hook for `fetch_media` and `transcribe` as `stage_completed`. This marks them as done with green checkmarks. The user sees them complete instantly, which is accurate — those stages were "completed" by the user providing the text directly.
2. **Hide them from the timeline:** Add a `skipped_stages` list to the job context and filter them out in the template. More complex, less intuitive.

Option 1 is simpler and more honest — emit synthetic stage hooks:
```python
if source_text and opts.stage_hook:
    for stage in ("fetch_media", "transcribe"):
        opts.stage_hook(stage, "stage_started")
        opts.stage_hook(stage, "stage_completed")
```

**~100 lines total across 5 files. Test: model validator tests + workflow test with mocked text source.**

---

## Change 4: Direct image upload per character (skip LoRA render)

User chose: **upload 1-2 reference images per character** (used for all shots).

### New adapter

New file: `adapters/render_character/static_image_adapter.py`

- `StaticImageAdapter(RenderCharacter)` — accepts `image_dir: Path` pointing to uploaded images
- `health()`: always OK (no external service)
- `run(req)`: finds images matching `{member_id}*` in `image_dir`, copies to work_dir, returns `ImageSet`
- Raises `FileNotFoundError` if no images found for the character
- No LoRA check — skips `_check_lora` entirely

### Config + model changes

- `core/config.py` `Settings.render_adapter`: add `"static"` to Literal
- `core/models/dashboard.py` `DashboardJobOverrides.render_adapter`: add `"static"` to Literal

### Workflow

File: `core/workflow.py` — in `_make_render_adapter` (line 142), add branch:
```python
if s.render_adapter == "static":
    from adapters.render_character.static_image_adapter import StaticImageAdapter
    return StaticImageAdapter(
        work_dir=work_dir / "renders",
        image_dir=work_dir / "uploads" / "characters",
        num_images=s.image_candidates,
    )
```

### Upload endpoint

File: `services/dashboard_api.py` — add `POST /api/jobs/{job_id}/upload-images`:
- Accepts `member_id: str = Form(...)` and `images: list[UploadFile] = File(...)`
- Saves to `<data_dir>/jobs/<job_id>/uploads/characters/{member_id}_00.png` etc.
- Returns list of saved filenames

### Dashboard UI

File: `services/templates/jobs_list.html`

- Add "Render Adapter" dropdown with options: Default, Musubi Flux, ComfyUI Flux, Static Images
- When "Static Images" selected, show a hint: "Upload character images after creating the job"
- Include `render_adapter` in `overrides` on submit

File: `services/templates/job_detail.html`

- When job has `render_adapter=static` and render phase hasn't started: show an image upload section with file inputs per character (Max, Zoe) and an upload button that POSTs to `/api/jobs/{job_id}/upload-images`

### Gap 4A: Image critique with a single candidate is a no-op

**Problem:** `VlmImageCritiqueAdapter` is designed to compare N candidates and pick the best. If the static adapter returns only 1 image per character, the VLM critique sends 1 image, scores it, and "picks" it — wasting a VLM call. The approval grid shows "1 of 1 candidates selected" which is confusing.

**Solution:** Skip the image critique stage when `render_adapter == "static"`. In `_render_shot_candidates()` in `core/workflow.py`:
- If the render adapter is `StaticImageAdapter` (check via `isinstance` or a `skip_critique: bool` attribute on the adapter), skip the `image_critique` call.
- Return a synthetic `ImageCritiqueResult` with the single image as the winner:
  ```python
  if hasattr(adapters.render, 'skip_critique') and adapters.render.skip_critique:
      return ImageCritiqueResult(
          winner_uri=render_result.images[0],
          winner_index=0,
          candidate_uris=render_result.images,
          scores=[],
          reasoning="User-provided static image — critique skipped.",
      )
  ```
- Set `skip_critique = True` on `StaticImageAdapter`.

The image approval gate still runs — the operator can see and confirm the uploaded images. This is useful even with static images.

### Gap 4B: Race between job creation and image upload

**Problem:** When the user creates a job with `render_adapter=static`, the worker may pick up the job before images are uploaded. The static adapter would raise `FileNotFoundError("No uploaded reference images found...")`.

**Solution:** Two-phase approach:
1. Jobs with `render_adapter=static` start in `created` status (not auto-queued) until images are uploaded.
2. Add a `POST /api/jobs/{job_id}/start` endpoint (or modify the existing queue logic) that moves the job from `created` → `queued` only after confirming images exist.

Implementation:
- In `dashboard_api.py` `POST /api/jobs`: when `overrides.render_adapter == "static"`, create the job record but **don't** create the queue item. Set status to `created`.
- On the job detail page, show the image upload UI + a "Start Pipeline" button.
- The "Start Pipeline" button calls a new `POST /api/jobs/{job_id}/start` endpoint that:
  1. Checks that uploaded images exist for all cast members.
  2. Creates the queue item.
  3. Sets status to `queued`.

This avoids the race entirely — the worker only sees the job after images are in place.

### Gap 4C: User needs to know character IDs

**Problem:** The upload endpoint takes `member_id` (e.g., "max", "zoe"). The user may not know these IDs. The UI shouldn't force the user to type IDs.

**Solution:** The job detail upload UI should list character names from the cast config, not ask the user to type IDs. On the job detail page:
- Load the cast member list from the API (add `GET /api/config/cast` endpoint that returns `[{id: "max", name: "Max"}, {id: "zoe", name: "Zoe"}]`).
- Or simpler: hardcode the `kids_duo` cast members in the template since there's only one cast config. Render a section per character:
  ```html
  <div class="upload-section">
    <h4>Max</h4>
    <input type="file" accept="image/*" id="upload-max">
    <h4>Zoe</h4>
    <input type="file" accept="image/*" id="upload-zoe">
    <button onclick="uploadCharacterImages('{{ job.job_id }}')">Upload & Start</button>
  </div>
  ```
  
Recommended: Add `GET /api/config/cast` that reads `config/casts/kids_duo.yaml` and returns member IDs + names. This keeps it dynamic if the cast changes later. The upload UI uses this to render one file input per character.

### Gap 4D: No existing test infrastructure for file uploads in dashboard

**Problem:** `tests/test_dashboard_repository.py` only covers DB CRUD. There are no tests for the dashboard API's HTTP layer (file uploads, form data).

**Solution:** Add a new test file `tests/test_dashboard_api.py` using FastAPI's `TestClient`:
```python
from fastapi.testclient import TestClient
from services.dashboard_api import create_app

def test_upload_character_images():
    app = create_app(config=..., repo=mock_repo)
    client = TestClient(app)
    resp = client.post(
        f"/api/jobs/{job_id}/upload-images",
        data={"member_id": "max"},
        files=[("images", ("max.png", b"fake-png-bytes", "image/png"))],
    )
    assert resp.status_code == 200
    assert "max_00.png" in resp.json()["uploaded"]
```

For the `StaticImageAdapter` itself, add tests in `tests/test_render_character.py` following the existing pattern — create temp dirs with test images, assert `ImageSet` output.

**~150 lines total across 7 files (including new adapter + new test file). Test: `StaticImageAdapter` unit test + upload endpoint test + job-start-gate test.**

---

## Change 5: Dashboard cleanup (constraints, not standalone work)

Not a separate change — enforced by how Changes 3 and 4 are built:

- Story textarea: hidden by default, shown only when "Paste Story" radio selected (same progressive-disclosure pattern as the existing file browser)
- Image upload area: hidden unless `render_adapter=static` is selected
- No new pages needed — everything fits in existing New Job modal and job detail page
- Phase stepper fix (Change 1) removes the biggest source of visual confusion

---

## Cross-cutting gaps

### Gap X1: DB schema for `source_kind = "text"`

**Problem:** `DashboardJobRecord.source_kind` is `Literal["url", "upload", "file"]`. Adding `"text"` changes the Pydantic model but the SQLite column `source_kind TEXT` has no constraint — it accepts any string. However, if we ever migrate to Postgres with an enum column, this would need a migration.

**Solution:** No action needed now — SQLite is unconstrained. Add a comment in `dashboard_repository.py` near the schema definition noting that `source_kind` accepts `url | upload | file | text`. When the Postgres migration happens, ensure the enum includes `text`.

### Gap X2: CLAUDE.md update

**Problem:** After implementation, CLAUDE.md needs to document the new `static` render adapter, `text` source kind, `model_swap_*` settings, `/unload` `/reload` endpoints, and the new API endpoints.

**Solution:** Update CLAUDE.md at the end of all changes:
- Add `static` to the render adapter table in "Key file map"
- Add `model_swap_enabled`, `model_swap_delay_sec` to env var table
- Add `StaticImageAdapter` and upload endpoint to file map
- Update test count
- Note the text-source skip behavior

### Gap X3: `run_pipeline_job` signature change affects CLI caller

**Problem:** Adding `source_text` to `run_pipeline_job()` changes its signature. The CLI entry point in `scripts/` and the worker both call it.

**Solution:** Make `source_text` a keyword-only argument with default `None`:
```python
async def run_pipeline_job(
    source_url: str,
    rights_cleared: bool,
    app_config: AppConfig,
    *,
    source_text: str | None = None,  # new, optional
    options: RunOptions | None = None,
    ...
)
```

This is backward-compatible — existing callers don't need to change.

---

## Verification

After each change:
1. Run `python -m pytest -q` — must stay at 335+ passing
2. Start dashboard (`uvicorn ... --reload` + `python -m services.dashboard_worker`)
3. For Change 1: Create a `phase=all` noop/mock job, verify stepper dots transition correctly via SSE
4. For Change 2: Set `VIDEO_ME_MODEL_SWAP_ENABLED=true`, run render+video, verify unload/delay/reload logs
5. For Change 3: Select "Paste Story", paste text, verify pipeline skips fetch/transcribe, stage dots auto-complete, no transcript review gate
6. For Change 4: Select "Static Images", create job, upload images on detail page, click "Start Pipeline", verify images are used directly, critique is skipped, approval gate still shows

## Implementation order

```
Change 1 (Phase Stepper)     — independent, smallest, ship first
Change 2 (Model Swap + Wan unload/reload)  — independent, ship second
Change 3 (Story Input)       — independent, ship third
Change 4 (Image Upload)      — independent, ship fourth
Change 5 (Dashboard Cleanup) — enforced by how 3+4 are built
```

Each change is independently shippable. No change depends on another.
