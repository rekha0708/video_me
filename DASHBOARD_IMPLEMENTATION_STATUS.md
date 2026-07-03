# Dashboard Implementation Status

Last updated: 2026-07-02 (phase chaining + transcript review session)

---

## Status Legend

- `DONE` — implemented, tested, live
- `PENDING` — not started
- `DEFERRED` — intentionally later
- `BLOCKED` — needs a decision or dependency

---

## All Milestones

| Step | Status | Notes |
|---|---|---|
| **D0.1** Fix CLI rights default | DONE | `--rights-cleared` requires explicit flag |
| **D0.2** Fix target language default | DONE | `run_pipeline_job()` accepts `target_language=None` |
| **D0.3** Fix runtime readiness service list | DONE | musubi_flux + ltx + fish_s2 checked correctly |
| **D1.1** Dashboard models | DONE | `core/models/dashboard.py` — job, queue, event, approval, heartbeat Pydantic models |
| **D1.2** Dashboard repository | DONE | `services/dashboard_repository.py` — SQLite, 6 tables, full CRUD |
| **D2.1** FastAPI API skeleton | DONE | `services/dashboard_api.py` — factory, health, jobs, events, cancel |
| **D2.2** Package deps | DONE | `[dashboard]` optional deps in `pyproject.toml` |
| **D3.1** Worker loop | DONE | `services/dashboard_worker.py` — poll, claim, run, heartbeat |
| **D3.2** Cancel handling | DONE | `asyncio.wait(FIRST_COMPLETED)` — cancel_event races pipeline task |
| **D3.3** Worker tests | DONE | `tests/test_dashboard_worker.py` — 333 total passing |
| **D4.1** Browser UI — jobs list | DONE | `services/templates/jobs_list.html` — table + New Job modal |
| **D4.2** Browser UI — job detail | DONE | `services/templates/job_detail.html` — header + timeline + events feed |
| **D4.3** Browser UI — health page | DONE | `services/templates/health.html` — worker heartbeat + readiness checks |
| **D4.4** Self-hosted CSS | DONE | `services/static/app.css` — zero CDN |
| **D4.5** SSE live updates + polling fallback | DONE | `services/static/app.js` + `GET /api/jobs/{id}/stream` |
| **D4.6** Stall detection | DONE | Banner appears when `last_heartbeat_at` > 120s on running job |
| **D5.1** Plan approval adapter | DONE | `adapters/approval/dashboard_approval_adapter.py` |
| **D5.2** Image approval adapter | DONE | `adapters/approval/dashboard_image_approval_adapter.py` |
| **D5.3** Workflow wiring | DONE | `run_pipeline_job(..., approval_overrides=)` — CLI path unchanged |
| **D5.4** Approve/reject API endpoints | DONE | `POST /api/jobs/{id}/approve`, `POST /api/jobs/{id}/reject` |
| **D5.5** Approval UI — plan | DONE | `services/templates/approval_plan.html` — score bars + shot table |
| **D5.6** Approval UI — images | DONE | `services/templates/approval_images.html` — candidate grid |
| **Phase chaining** `transcribe` phase | DONE | Stops after analyze_content, saves artifacts |
| **Phase chaining** `script_plan` phase | DONE | Loads cached transcribe artifacts, runs adapt→plan→approval |
| **Phase chaining** resume wiring | DONE | Worker sets `resume=True` for `script_plan`, `render`, `assemble` |
| **D6.1** AJAX jobs-list refresh | DONE | Replaced `location.reload()` with per-cell AJAX patch — no full reload |
| **D6.2** Modal-safe refresh | DONE | Refresh skipped when New Job modal is open |
| **D6.3** Phase-aware stage timeline | DONE | Only stages for the selected phase are shown (e.g. `transcribe` = 3 dots) |
| **D6.4** Live stage timeline updates | DONE | Worker emits `stage_started`/`stage_completed`/`stage_failed` via `RunOptions.stage_hook`; JS updates dots live via SSE |
| **D6.5** Stage events in DB | DONE | `update_job_status(current_stage=)` called on each `stage_started` hook |
| **D6.6** Readable validation errors | DONE | FastAPI 422 array mapped to `field: message` in error box |
| **D6.7** Auto-refresh trigger completeness | DONE | All 8 non-terminal statuses trigger list refresh (was missing `created`, `cancel_requested`, `stalled`, `pending_final_review`) |
| **D7.1** Phase stepper widget | DONE | Macro-level pill row: Transcribe → Script+Plan → Render → Assemble with ✓/⟳/· state derived from `completed_phases` events |
| **D7.2** `completed_phases` tracking | DONE | `completed_phases_json` column in DB; `append_completed_phase()` called by worker after each phase succeeds; auto-migrates existing DBs |
| **D7.3** Transcript review gate | DONE | After `transcribe` phase: worker creates `TRANSCRIPT` approval, sets `PENDING_TRANSCRIPT_REVIEW`, polls; on reject+notes → Qwen3.6:35b refines transcript (up to 3 iterations) |
| **D7.4** LLM transcript refine adapter | DONE | `adapters/transcript_refine/llm_adapter.py` — sends transcript JSON + operator notes to Ollama, returns corrected transcript in-place |
| **D7.5** `POST /api/jobs/{id}/advance` | DONE | Re-queues same job for next phase with `resume=True`; validates COMPLETED status + known phase sequence |
| **D7.6** Continue button | DONE | "Continue to [next phase] →" button on job detail when `status=completed` and there is a next phase |
| **D7.7** Transcript approval UI | DONE | `services/templates/approval_transcript.html` — segment table with timestamps, correction notes textarea, approve/reject |

---

## Phase dropdown options

| Value | Pipeline stages | Chains from |
|-------|----------------|-------------|
| `noop` | Mock pipeline, no services | — |
| `transcribe` | fetch → transcribe → analyze_content → transcript review gate → **stop** | — |
| `script_plan` | adapt_script → plan_shots → plan approval → **stop** | same job (via Advance button) |
| `render` | per-shot: render images + voice + video → **stop** | same job (via Advance button) |
| `assemble` | assemble_video → publish | same job (via Advance button) |
| `all` | Full end-to-end pipeline | — |

### Phase chaining (same job_id)

All phases now advance on the **same job record**. The operator clicks "Continue to [next phase] →" after each phase completes. The API re-queues the same `job_id` with the next phase and `resume=True` so artifacts from the previous phase are reused automatically.

Phase state is tracked in `completed_phases` (DB column) and displayed in the phase stepper on the job detail page.

---

## Bugs fixed — original D3–D5 session

| Bug | Root cause | Fix |
|-----|-----------|-----|
| `POST /api/jobs` → 422 (body as query param) | `from __future__ import annotations` + lazy `Request` import → `get_type_hints()` returned unresolvable `ForwardRef` | Replaced `request: Request` dep with `authorization: str \| None = Header(default=None)` |
| Cancel fires after confirm dismissed | `<form>` had both inline `onsubmit` and `app.js` generic listener — app.js ran regardless of confirm result | Removed `<form>` wrapper; cancel is `<button onclick="cancelJob(...)">` only |
| Infinite reload on terminal job pages | SSE emits `done` immediately for terminal jobs → `location.reload()` → reconnect → `done` again | `_wasTerminal` flag: SSE skipped if job already terminal at page load |

## Bugs fixed — UX follow-up session (2026-07-02)

| Bug | Root cause | Fix |
|-----|-----------|-----|
| Jobs list refreshed entire page every 10s | `setInterval(() => location.reload(), 10000)` | Replaced with `refreshJobsList()` — AJAX patch of status badge + stage cell only |
| New Job form "disappeared" | 10s reload fired while modal was open | Reload skipped when `new-job-modal` is visible |
| Stage timeline showed all 10 stages for any phase | No phase filter | Phase-to-stages map in `job_detail.html`; each phase renders only its own dots |
| Stage timeline never updated live | Worker emitted only `job_started`/`job_completed`, no per-stage events | Added `stage_hook: Callable` to `RunOptions` + `run_stage(..., stage_hook=)`; worker wires hook to `record_event` + `update_job_status(current_stage=)` |
| `"transcribe"` phase rejected (API 422) | API server process cached old Pydantic model that predated `"transcribe"` in the Literal | Restart server; use `--reload` flag going forward |
| Error box showed raw JSON on 422 | JS: `data.detail?.message` undefined for array; fell back to `JSON.stringify` | JS now maps 422 array to `field: message` per error |
| Auto-refresh missed non-terminal active statuses | `selectattr` only listed 4 statuses; `cancel_requested`, `stalled`, `created`, `pending_final_review` missing | Added all 8 non-terminal statuses to trigger |

---

## Running the dashboard

```bash
# Install deps (first time only)
.venv/bin/pip install -e ".[dashboard]"

# Terminal 1 — API server  (--reload auto-restarts on code changes)
.venv/bin/uvicorn services.dashboard_api:create_app --factory --port 8080 --host 127.0.0.1 --reload

# Terminal 2 — Worker
.venv/bin/python -m services.dashboard_worker

# Open browser
open http://localhost:8080
```

> **Important**: if you see a phase validation error after editing code, the server has the old model cached — restart it. The `--reload` flag prevents this.

Neither the API server nor the worker is in `scripts/start_services.sh` (that script is for the GPU box). Run both manually in separate terminals on the local Mac.

Environment overrides:

```bash
VIDEO_ME_DASHBOARD_TOKEN=secret     # bearer auth on write endpoints
VIDEO_ME_AUTO_APPROVE_PLAN=true     # skip plan approval gate (CI)
VIDEO_ME_AUTO_APPROVE_IMAGES=true   # skip image approval gate (CI)
```

---

## Bugs fixed — phase chaining session (2026-07-02)

| Bug / Gap | Root cause | Fix |
| --------- | ---------- | --- |
| `script_plan` job had no transcript to read | Each "New Job" created a fresh `job_id`/work dir — no link to prior `transcribe` job | Replaced multi-job model with single-job phase advancement; all phases share one `job_id` and one work dir |
| No visual indication of phase progress | Stage timeline only showed micro-level dots for current phase | Added phase stepper (pill row) derived from `completed_phases` — shows done/active/pending state for all 4 phases |
| No transcript review step | Pipeline moved from transcribe → script_plan without operator seeing the transcript | Worker transcript review gate: approval card, operator notes, LLM refine loop (up to 3 iterations) |
| `advance` endpoint missing | No way to re-queue a job for the next phase | `POST /api/jobs/{id}/advance` added; validates COMPLETED + known phase; enqueues RESUME action |

---

## What's pending / blocked

| Item | Priority | Notes |
| ---- | -------- | ----- |
| Artifact viewer on job detail | Medium | Show script, storyboard, transcript as formatted readable cards (currently raw JSON on disk only) |
| `scripts/start_dashboard.sh` | Low | Convenience script to start API + worker locally in one command |
| Worker multi-instance stress test | Low | Claim locking not load-tested |
| Postgres + S3 support | Low | Dashboard repo is SQLite-only |
| Track B — Flux LoRAs | Blocked | `loras/kids_duo_max.safetensors` missing; `kids_duo_zoe.safetensors` TEST-ONLY placeholder. Needs GPU. |
| Track D — GPU services | Blocked | ComfyUI + Fish Audio S2 need manual start on GPU box |
| Track E — Compliance sign-off | Blocked | Operator has not signed off |
