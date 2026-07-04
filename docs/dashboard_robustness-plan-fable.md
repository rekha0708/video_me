# Plan: Wan VRAM sequencing, story/image input modes, dashboard fixes

> **Plan of record, approved 2026-07-04.** Supersedes the earlier draft in
> `docs/DASHBOARD_ROBUSTNESS_PLAN.md` (model-swap health gate, `text` source kind,
> `StaticImageAdapter`, modal-based UI). This version was approved after a gap review
> (§ Gap review) — implement with those amendments.
>
> **Multi-session rule:** before doing ANY work, read § Implementation status below and
> continue from the first unchecked item. After finishing an item, tick it, add the date
> and commit hash, and note deviations in the "Notes" column. Never re-implement a
> checked item without a note explaining why.

## Implementation status

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[-]` skipped/obsolete

### Milestone 1 — Wan VRAM sequencing (OOM fix)

| # | Item | Status | Date / commit | Notes |
|---|------|--------|---------------|-------|
| 1.1 | `services/wan_server.py`: lazy load, `POST /load`, `POST /unload` (409 while loading), `/health` with `model_loaded`, `/generate` auto-load safety net, docstring | `[x]` | 2026-07-04 | |
| 1.2 | `adapters/generate_video/wan_adapter.py`: `managed_vram=True`, health ok when unloaded, `load()`/`unload()`/`wait_until_loaded()` | `[x]` | 2026-07-04 | health() needed no change — any 200 was already ok |
| 1.3 | New `core/gpu_sequencer.py`: `unload_ollama_model` (moved), `ensure_video_model_unloaded`, `prepare_video_model` (30 s gap + poll + `video_model_load` notify events) | `[x]` | 2026-07-04 | `_is_managed` uses strict `is True` so MagicMock adapters in tests don't opt in |
| 1.4 | `core/config.py`: `wan_load_gap_sec=30`, `wan_load_timeout_sec=1800` | `[x]` | 2026-07-04 | |
| 1.5 | `core/workflow.py`: unload hook before Phase A, prepare hook between image approval and Phase B | `[x]` | 2026-07-04 | `_unload_ollama_model` kept as alias to moved fn |
| 1.6 | Tests: wan adapter load/unload/wait, `test_gpu_sequencer.py` ordering, workflow hook ordering | `[~]` | 2026-07-04 | written; pending green run |

### Milestone 2 — Job page fix for phase="all" + latent bugs

| # | Item | Status | Date / commit | Notes |
|---|------|--------|---------------|-------|
| 2.1 | `core/storage.py`: `ArtifactStore.has()` on protocol + both impls | `[ ]` | | |
| 2.2 | `dashboard_api.py`: `_artifact_flags`, `_STAGE_TO_MACRO` (incl. `video_model_load`→render), `_stepper_state`; pass into `ui_job_detail` | `[ ]` | | |
| 2.3 | `job_detail.html`: artifact cards read `artifact_flags`; stepper reads `stepper`; delete phase_order arithmetic (lines 217-222) | `[ ]` | | |
| 2.4 | `get_renders`: fall back to `user_images/` when no `renders/` | `[ ]` | | |
| 2.5 | Worker: phase="all" completion appends all four phase names to `completed_phases` | `[ ]` | | |
| 2.6 | Fix `_load/_save_transcript_artifact` to use `create_artifact_store` (transcript review gate self-skips today) | `[ ]` | | |
| 2.7 | Verify `app.js` `updateTimeline` no-ops on unknown stage names | `[ ]` | | |
| 2.8 | Tests: `test_dashboard_api_helpers.py` (`_artifact_flags`, `_stepper_state`, `has()`) | `[ ]` | | |

### Milestone 3 — Story input modes (backend)

| # | Item | Status | Date / commit | Notes |
|---|------|--------|---------------|-------|
| 3.1 | `core/models/dashboard.py`: `story`/`story_images` kinds, `story_text`, `character_images`, model_validator, `DashboardJobRecord.source_kind` | `[ ]` | | |
| 3.2 | New `adapters/story_ingest/`: `parse_structured_story`, `heuristic_segments`, `LlmStorySegmentAdapter` | `[ ]` | | |
| 3.3 | `core/workflow.py`: `RunOptions.user_images`, relaxed resume guard, `ImageCritiqueResult.origin`, `_build_user_image_critiques`, Phase A skip branch | `[ ]` | | |
| 3.4 | Worker `_seed_story_job`: fetch_media stub written FIRST, then transcribe; Mode B image copy to `user_images/`; resume dispatch (`resume_job_id` only when core Job row exists) | `[ ]` | | |
| 3.5 | Tests: `test_story_ingest.py`, model validators, worker seeding, workflow user_images skip | `[ ]` | | |

### Milestone 4 — Dashboard UI: /jobs/new page + uploads

| # | Item | Status | Date / commit | Notes |
|---|------|--------|---------------|-------|
| 4.1 | `GET /api/local-images?dir=` | `[ ]` | | |
| 4.2 | `POST /api/uploads/character-image` (multipart, validation, ≤10 MB) | `[ ]` | | |
| 4.3 | `POST /api/jobs`: story_images validation; reject story-kind creation with phase ∈ {script_plan, render, assemble} | `[ ]` | | |
| 4.4 | New `services/templates/job_new.html` (mode selector, story textarea, per-member image slots, phase select restricted for story modes, reworded rights checkbox) | `[ ]` | | |
| 4.5 | `jobs_list.html`: remove modal + JS, New Job → link, source-kind badge; null-guard modal refs in `app.js`; bump static `?v=` | `[ ]` | | |
| 4.6 | Image approval `origin="user"` labeling (adapter payload + `approval_images.html`) | `[ ]` | | |
| 4.7 | Tests: upload/local-images endpoints, job-creation validation | `[ ]` | | |

### Final

| # | Item | Status | Date / commit | Notes |
|---|------|--------|---------------|-------|
| F.1 | Full `pytest -q` green (baseline: 3 known-stale failures per CLAUDE.md) | `[ ]` | | |
| F.2 | Local dashboard smoke test (story job, artifact cards, /jobs/new) | `[ ]` | | |
| F.3 | GPU-box verification (wan load/unload sequence, event order) — needs GPU box | `[ ]` | | |
| F.4 | Update CLAUDE.md "Current state" + memory files | `[ ]` | | |

---

## Context

Four problems from the last runs:

1. **VRAM OOM on the wan path**: `services/wan_server.py` loads WanI2V eagerly at startup and keeps it resident all session. During `render_character`, Wan's VRAM + Ollama (critique VLM) + the Flux subprocess exceed the G200's 143 GB (observed 116 GB before Flux even allocated — commit 58ce9d8). Fix: Wan must not be loaded during the render phase; load it only after images are approved, with an unload → 30 s gap → load → readiness-poll sequence. (This is the deferred-Wan-loading plan noted in commit f3efc3b.)
2. **New input modes**: (A) user pastes a story with a timeline directly — skip yt-dlp/Whisper; (B) user provides story + per-character reference images — skip LoRA/Flux rendering entirely and use the images as i2v input.
3. **Job-page bug**: for `phase="all"` jobs, the Transcript / Script+Plan / Rendered Images / Final Video cards never appear (`job_detail.html:217-222` computes `phase_order.index('all')` → -1, and `completed_phases` contains `"all"`, not phase names).
4. **Dashboard robustness/cleanliness**: New Job modal is crowded; move creation to a dedicated page.

**User decisions (locked):** story input accepts both structured `start-end: text` lines and free text (LLM segments it); image mapping is **hybrid** (per-character reference defaults + optional per-shot override at the existing image-approval gate); image input via **both** browser upload and server-dir picker; New Job becomes a **dedicated page** `/jobs/new`.

### Flow: old vs new (render → video, wan path)

```
OLD  wan_server boots ──[loads model, resident forever]
     render loop [Flux subprocess + Ollama critique]  ← OOM: Wan already resident
     image approval → voice+video loop

NEW  wan_server boots ──[no model load; /health says model_loaded:false]
     workflow: POST /unload (idempotent) ─→ render loop [Flux + Ollama, full headroom]
     image approval
     workflow: unload Ollama → sleep 30s → POST /load → poll /health until model_loaded
     voice+video loop
```

```
NEW INPUT MODES (dashboard)
  story:        story text ──parse/LLM──▶ transcribe.json + fetch_media.json stub
                pre-seeded ▶ pipeline resumes from analyze_content (no yt-dlp/whisper)
  story_images: same + user_images/{member}.png ▶ Phase A (Flux render + VLM critique)
                fully skipped ▶ image-approval gate shows reference images
                (speaker pre-selected per shot, per-shot override) ▶ Phase B i2v
```

---

## Work item 1 — Wan VRAM sequencing

### 1.1 `services/wan_server.py` — lazy load + `/load` + `/unload`
- Remove eager `run_in_executor(None, _load_pipeline)` from `lifespan` (line 96); keep dir warnings + shutdown cleanup.
- Add `_load_lock = threading.Lock()` + `_loading` flag; `_load_pipeline()` resets `_pipeline_error`, returns early if already loaded.
- `GET /health` → always 200 when process is up: `{"status":"ok","model_loaded":bool,"loading":bool,"error":_pipeline_error}`.
- `POST /load`: already loaded → 200; loading → 202; else fire `_load_pipeline` in executor **without awaiting**, return 202. Idempotent.
- `POST /unload`: if `_loading` → **409** (see Gap 2); else acquire `_infer_lock` (waits out in-flight inference), `_pipeline = None`, `gc.collect()`, `torch.cuda.empty_cache()` (guarded import). Idempotent 200.
- `POST /generate` safety net: if unloaded, blocking-load in executor first (standalone use keeps working; first request pays the 4–5 min load).
- Update module docstring (API contract changed).

### 1.2 `adapters/generate_video/wan_adapter.py`
- Class attr `managed_vram = True` (marker; LTX lacks it → all sequencing is a no-op on the default stack).
- `health()` (lines 54-71): reachable 200 = ok even when `model_loaded:false`.
- New methods (lazy `import httpx`, same style as `_call_wan`):
  - `load()` — POST `/load`, timeout 30 s.
  - `unload() -> bool` — POST `/unload`, timeout 120 s. `ConnectError` → warn + return False (server down = nothing resident); HTTP error (incl. 409 load-in-progress) → raise `RuntimeError` (VRAM not freed — proceeding would OOM the render).
  - `wait_until_loaded(timeout_sec, poll_sec=10)` — poll `/health` until `model_loaded`; raise on `error` body; `TimeoutError` otherwise.

### 1.3 New `core/gpu_sequencer.py`
- Move `_unload_ollama_model` (workflow.py:559-570) here as `unload_ollama_model()`; keep alias import in workflow for existing tests.
- `ensure_video_model_unloaded(video_adapter, *, notify=None)`: no-op unless `getattr(adapter, "managed_vram", False)`; else `await adapter.unload()` + log_event.
- `prepare_video_model(video_adapter, settings, *, sleep=asyncio.sleep, notify=None)`: no-op unless managed_vram; else (1) unload Ollama (llm model + critique model if different), (2) `await sleep(settings.wan_load_gap_sec)`, (3) `await adapter.load()`, (4) `await adapter.wait_until_loaded(settings.wan_load_timeout_sec)`. `notify` is the stage_hook — emit synthetic `("video_model_load", "stage_started"/"stage_completed")` events so the dashboard events feed shows the multi-minute load (Gap 4).

### 1.4 `core/config.py`
- `wan_load_gap_sec: int = 30`, `wan_load_timeout_sec: int = 1800`.

### 1.5 `core/workflow.py` hooks (in `_run_to_assembled_video`)
- Next to the existing `_unload_ollama_model` call at line 886 (before the Phase A loop): `await ensure_video_model_unloaded(adapters.video)`.
- Between image approval (line 905) and the Phase B loop (line 908): `await prepare_video_model(adapters.video, config.settings, notify=opts.stage_hook)` — runs for both the normal and `user_images` paths.
- LTX path and `phase == "assemble"` branch untouched.

---

## Work item 2 — Story / Story+Images modes + `/jobs/new` page

### 2.1 Models — `core/models/dashboard.py`
- `DashboardSource.kind`: extend Literal with `"story"`, `"story_images"`. Replace the url field_validator with a model_validator: url required only for url/file kinds; story kinds default url to `"story://<job>"` (column NOT NULL; jobs list shows it).
- `CreateDashboardJobRequest`: add `story_text: str | None = None`, `character_images: dict[str, str] = {}` (member_id → server path). Validator: story kinds need story_text; `story_images` needs ≥1 image. Keep the phase field permissive (advance re-queues later phases — Gap 1). Extend `DashboardJobRecord.source_kind` Literal too.

### 2.2 Workflow — `core/workflow.py`, `core/models/capabilities.py`
- `RunOptions`: add `user_images: dict[str, str] | None = None`.
- Relax resume guard (line 1084) to `if lang_opts.resume and not (resume_job_id or job_id):` — lets a fresh story job resume from pre-seeded artifacts under its own job_id. Verify `_make_job_context`/`job_store.save_job` upserts cleanly on retry of a story job (row already exists).
- `ImageCritiqueResult`: add `origin: Literal["vlm","user"] = "vlm"`.
- New `_build_user_image_critiques(shots, user_images, cast)`: candidates = user images in `cast.members` order (stable grid); per shot `winner_index` = image of `shot.characters_on_screen[0]` (fallback 0); `origin="user"`.
- Render branch: `if opts.user_images:` build synthetic critiques and **skip Phase A entirely** (no Flux render, no VLM, Track-B LoRA gate never fires); else existing Phase A. The existing image-approval gate then doubles as the hybrid shot-image review (speaker image pre-selected; per-shot override = existing picks mechanism; Approve = accept defaults). `approved_uris` zip into Phase B unchanged (lines 910-914).

### 2.3 New `adapters/story_ingest/` (`parser.py`, `llm_adapter.py`)
- `parse_structured_story(text, language) -> TranscribeResult | None`: every non-blank line must match `^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)$` with monotonic times; else None.
- `heuristic_segments(text)`: sentence split at 2 words/sec, 5 s floor / 8 s cap (matches shot-duration convention).
- `LlmStorySegmentAdapter(base_url, model)` mirroring `adapters/transcript_refine/llm_adapter.py` (lazy httpx, `think:False`, json_repair). On unrecoverable output → fall back to `heuristic_segments` (seeding never hard-fails).

### 2.4 Worker seeding — `services/dashboard_worker.py` (`_run_pipeline`, line 209)
- If `req.source.kind in ("story","story_images")` and no `transcribe` artifact yet → `_seed_story_job(req, job_id)`:
  1. Structured parse, else LLM segment (record `story_segmented` event noting which path).
  2. Write `fetch_media` stub **first** (`FetchMediaResult(video_uri="story://{job_id}", audio_uri="story://{job_id}", duration_sec=<last segment end>, source_url=req.source.url)`), **then** the `transcribe` artifact — the seed-guard checks `transcribe`, so a mid-seed crash can never leave `transcribe` present without `fetch_media` (Gap 3). With resume=True both stages skip cleanly and the pipeline starts at analyze_content.
  3. Mode B: copy each character image into `{data_dir}/jobs/{job_id}/user_images/{member_id}{ext}` (must be under data_dir so `/img/{b64}` can serve the approval grid); return the map.
- Dispatch: `resume = is_story or phase in ("script_plan","render","assemble")`; pass `user_images` into `RunOptions`; fresh story job → `job_id=job_id, resume_job_id=None` (set `resume_job_id=job_id` only when a core Job row already exists, i.e. retry/advance).
- Story + `phase="transcribe"`: fetch/transcribe skip via seeds, analyze_content runs for real, then the existing transcript review gate shows the story segmentation for operator review — this is the supported "review my story's segmentation" path (Gap 1).

### 2.5 API — `services/dashboard_api.py`
- `GET /api/local-images?dir=` — clone of `list_local_videos` with image extensions (`.png .jpg .jpeg .webp`); include `path_b64` when under data_dir (for `/img` thumbnail).
- `POST /api/uploads/character-image` — multipart (`member_id`, `file`); validate ext + member_id ∈ cast + ≤10 MB; write `{data_dir}/uploads/{token}/{member_id}{ext}`; return `{path, path_b64}`.
- `POST /api/jobs`: for `story_images`, 400 unless every image path exists and every key is a cast member id. For story kinds, reject **initial creation** with phase ∈ {script_plan, render, assemble} (400 with hint "start at 'transcribe' or 'all'") — those phases need artifacts that only exist after analyze has run; the advance path is unaffected because it re-queues an existing job (Gap 1).
- `GET /jobs/new` (line 818, route exists): render new `job_new.html` with cast members.

### 2.6 Templates / static (self-hosted CSS only, no CDN)
- **New `services/templates/job_new.html`**: input-mode radios (Video URL / Local file / Story / Story + Images), only relevant fields per mode. URL + local-file groups moved verbatim from the modal. Story textarea with hint: `` `start-end: text` per line, or free text (auto-segmented) ``. Per-cast-member image slots: upload button (fetch → upload endpoint) **or** dir-picker (`/api/local-images`), thumbnail preview via `/img/`; every cast member gets a slot, ≥1 required, all recommended (shots whose speaker has no image fall back to the first image — fixable at the approval gate). Phase select for story modes shows only `transcribe` (relabelled "Analyze story — review segmentation") and `all`; hide `noop`; keep upstream-dependency hints per user preference. Rights checkbox reworded for story modes ("I confirm I own/have rights to this story and images") — still required. Submit → POST `/api/jobs` → redirect to `/jobs/{id}`.
- `jobs_list.html`: delete the modal (lines 15-115) + its JS; New Job button → `<a href="/jobs/new">`; add small source-kind badge in the Source cell. Null-guard any `app.js` references to modal elements (refresh-skip check must tolerate the modal being absent — Gap 5). Bump static `?v=` in base.html.
- `adapters/approval/dashboard_image_approval_adapter.py` + `approval_images.html`: pass `origin` through; label pre-selected card "★ default" (not "★ VLM") and subtitle "Pick the reference image for each shot, or Approve to accept defaults" when `origin=="user"`.

---

## Work item 3 — Fix job page for `phase="all"`

- `core/storage.py`: add `has(job_id, stage) -> bool` to `ArtifactStore` protocol + both impls (`path.exists()` / `head_object`).
- `services/dashboard_api.py` — module-level testable helpers:
  - `_artifact_flags(store, work_dir, job_id)`: transcript ⇐ `has("transcribe")`; script ⇐ `has("adapt_script") or has("plan_shots")`; renders ⇐ `has("plan_shots")` and (`renders/` or `user_images/` exists); video ⇐ `assembled/final.mp4` exists.
  - `_STAGE_TO_MACRO` map (fetch/transcribe/analyze → transcribe; adapt/plan → script_plan; render/voice/video/lipsync **and `video_model_load`** → render; assemble/publish → assemble) + `_stepper_state(job, flags)`: passthrough for phased jobs; for "all" derive done-phases from artifacts and active phase from `current_stage`.
- `ui_job_detail` passes `artifact_flags` + `stepper` into the template.
- `job_detail.html`: replace line 204 and **delete** the phase_order arithmetic (217-222) — all four card conditions read `artifact_flags`; stepper block (53-119) reads `stepper`.
- `get_renders` (line 359): when no `renders/` candidates but `user_images/` exists, return the shot speaker's user image as the single candidate (Mode B jobs don't show an empty card).
- `dashboard_worker`: when phase=="all" completes, append all four phase names to `completed_phases` (keeps completed_phases truthful).
- `app.js` `updateTimeline`: confirm it no-ops for unknown stage names (the synthetic `video_model_load` event must not break the timeline).

## Work item 4 — small fixes
- **Latent bug**: `dashboard_worker._load_transcript_artifact`/`_save_transcript_artifact` (~line 379) read/write `data_dir/{job_id}/transcribe.json`, which never matches `LocalArtifactStore`'s `artifact_dir/{job_id}/transcribe.json` — the transcript review gate silently self-skips today. Fix to use `create_artifact_store(settings)`.
- Source-kind badge on the jobs list (covered in 2.6). Nothing else.

---

## Gap review (post-approval; amendments folded into the sections above)

1. **Story job starting at `script_plan` would crash** — `workflow.py:789-796` requires fetch_media + transcribe + **analyze_content** artifacts, and seeding only provides the first two. Amendment: story modes offer only `transcribe` ("Analyze story") and `all` at creation (UI + POST /api/jobs check); the transcribe phase runs analyze_content for real and the transcript review gate doubles as story-segmentation review; advance chaining then works normally.
2. **`/unload` racing an in-flight `/load`** — the loader thread could set `_pipeline` *after* unload returned, leaving the model resident during render. Amendment: `/unload` returns 409 while `_loading`; adapter raises rather than proceeding into an OOM render.
3. **Partial seeding crash** — if `transcribe.json` were written before `fetch_media.json` and seeding crashed between them, retry would skip seeding (guard sees transcribe) and fetch_media would feed `story://` to yt-dlp. Amendment: write fetch_media first; guard on transcribe.
4. **Silent multi-minute gap in the dashboard during Wan load** — between image approval and the first voice stage nothing would appear for up to ~6 min. Amendment: gpu_sequencer emits synthetic `video_model_load` stage events through the existing stage_hook (`dashboard_worker._make_stage_hook` already records arbitrary stage names); mapped to the render macro-phase in the stepper.
5. **Modal removal can break `app.js`** — the jobs-list auto-refresh skips refresh while the New Job modal is open; after the modal is deleted that lookup must null-guard.

Known accepted behaviors (not gaps):
- Retrying a Mode B / render-phase job re-runs the image approval gate (pre-existing pattern; cheap since Phase A is skipped).
- Two-character shots default to the first character's image (override at the gate).
- `start_services.sh`'s `wait_for` on Wan health now passes immediately at boot (correct — server up, model deliberately unloaded).
- A shot whose speaker has no provided image falls back to the first image; fixable at the gate.

---

## Tests (mock httpx per project pattern; see CLAUDE.md)

- `tests/test_generate_video.py`: wan health ok with `model_loaded:false`; load/unload URLs; `wait_until_loaded` success/timeout/error; unload ConnectError → False; unload 409 → RuntimeError.
- New `tests/test_gpu_sequencer.py`: no-op without `managed_vram`; ordering unload-Ollama → sleep(gap) → load → wait (injected sleep + call log); notify emits `video_model_load` events.
- `tests/test_workflow.py`: with `managed_vram=True`, `video.unload` awaited before `render.run`; `load`/`wait_until_loaded` after `image_approval.run`, before first `video.run`. With `user_images` set: `render.run`/`image_critique.run` never called; approval receives synthetic critiques with correct per-shot winner; approved URIs reach `video.run`; resume accepted with job_id and no resume_job_id.
- New `tests/test_story_ingest.py`: structured happy path; mixed format → None; non-monotonic → None; LLM segmentation (mocked httpx) + json_repair + heuristic fallback.
- Dashboard: model validators (story needs text; story_images needs images; legacy payloads parse); worker seeding writes fetch_media-then-transcribe + passes RunOptions kwargs (patch `run_pipeline_job` with AsyncMock); API rejects story+script_plan creation; new `tests/test_dashboard_api_helpers.py` for `_artifact_flags`/`_stepper_state`.

## Implementation order
1. Milestone 1 (server → adapter → sequencer/config → workflow hooks → tests) — self-contained.
2. Milestone 2 (small; makes story jobs verifiable in UI).
3. Milestone 3 (models + story_ingest + workflow + worker seeding + tests).
4. Milestone 4 (API endpoints + `job_new.html` + jobs_list cleanup + approval labels).

## Verification
- `python -m pytest -q` (local baseline: 3 known-stale failures per CLAUDE.md).
- Run dashboard locally: `.venv/bin/uvicorn services.dashboard_api:create_app --factory --port 8080 --reload` + `.venv/bin/python -m services.dashboard_worker`.
- Story mode (no GPU needed): create Story job with structured `0-4: …` lines → `fetch_media.json` + `transcribe.json` appear in artifact dir; Transcript card visible; review gate shows the segmentation.
- Story+Images: two PNGs → after plan approval the gate shows reference images with speaker pre-selected; override one shot; Approve → Phase B gets picked URIs.
- phase="all" regression: open an existing all-phase job → artifact cards appear; stepper tracks macro phase.
- Wan (GPU box): `/health` returns `model_loaded:false` immediately at boot; `POST /load` → poll → true; `POST /unload` idempotent (and 409 during load); run a wan-override job and confirm event order (no Wan during render; Ollama unload → 30 s gap → load → `video_model_load` events in the feed → Phase B).

## Risks / edge cases
- **Wan load timeout** (default 1800 s) → job FAILED with existing Retry; `/load`+`/unload` idempotent so retries are safe.
- **Unload vs in-flight inference**: `/unload` blocks on `_infer_lock`; adapter's 120 s timeout raises clearly rather than OOMing the render.
- **Mode B re-plan**: synthetic critiques are built from the *final approved* storyboard (gate runs after plan critique/approval), so defaults can't go stale.
- **Story parse failures**: strict regex → LLM → json_repair → deterministic heuristic; never hard-fails; events record the path.
- **`target_language: both`**: seeding idempotent (guarded by transcribe-artifact existence).
- **Back-compat**: all new fields optional/defaulted (`origin="vlm"` keeps cached critique JSONs valid; old queue payloads still parse; `data_dir/uploads/` staging is not garbage-collected — acceptable, commented).
