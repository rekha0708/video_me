# Web Dashboard and Chatbot Expansion Plan

This document is the implementation plan for replacing the current CLI/operator flow with a
web-based dashboard, while deliberately keeping the door open for a future chatbot that uses the
same backend APIs.

The key decision: build an API-first dashboard first. The chatbot should become another client of
that API, not a separate orchestration path.

---

## 1. Executive Summary

`video_me` already has the right core engine:

- `run_pipeline.py` is the current operator CLI.
- `core/workflow.py` owns the actual pipeline DAG.
- `core/executor.py` wraps each stage with health checks, execution, artifact persistence, and job
  updates.
- `core/storage.py` persists jobs and artifacts locally or through Postgres/S3.
- Approval gates currently use temporary local web UIs in `adapters/approval/`.

The dashboard should not replace the pipeline engine. It should wrap it with:

- a durable job state model,
- a real API,
- a background worker,
- centralized approval pages,
- live events and heartbeats,
- debug surfaces,
- artifact browsing,
- and later, chatbot tools.

The important change is that long-running work should not happen inside HTTP requests, and approval
should not be handled by ephemeral blocking pages. The dashboard should submit jobs, track jobs, show
progress, pause at approval checkpoints, and resume jobs after the operator decides.

---

## 2. Goals

### Product Goals

- Start a new pipeline run from the browser.
- Resume an existing job without using the CLI.
- See all jobs, their current status, current stage, elapsed time, and final output.
- Handle storyboard approval in one persistent dashboard.
- Handle image candidate approval and overrides in one persistent dashboard.
- Preview generated images, per-shot videos, final assembled video, metadata, logs, and stage
  artifacts.
- Make failures actionable instead of silent or stale.
- Prepare a stable API that a chatbot can call later.

### Reliability Goals

- The web UI must never sit forever with a spinner and no explanation.
- The API must return immediately when a job is submitted.
- A worker must own long-running pipeline execution.
- Every running job must have a heartbeat.
- Every stage transition must be written as a durable event.
- The frontend must show stale/running/failed states clearly.
- Approval requests must survive browser refreshes and server restarts.
- Failed jobs must expose enough context to debug without SSH first.

### Future Chatbot Goals

- The chatbot can start, resume, inspect, approve, reject, and debug jobs using the same APIs.
- The chatbot can ask required human questions, especially source rights clearance.
- The chatbot can summarize failures and suggest next actions.
- The dashboard remains fully usable even if the chatbot LLM is down.

---

## 3. Non-Goals

- Do not rewrite `core/workflow.py` from scratch.
- Do not make the chatbot the only control surface.
- Do not allow fully automated publish for kids' content.
- Do not let an LLM bypass rights clearance or final human review.
- Do not expose raw filesystem paths directly without path validation.
- Do not depend on a JavaScript build tool for the first dashboard MVP unless the team explicitly
  chooses to.

---

## 4. Current State

### Current Entry Points

- `run_pipeline.py`
  - CLI for running all phases, specific phases, resume jobs, and one-shot reruns.
- `core.workflow.run_pipeline_job()`
  - Phase 1 single-pass pipeline.
- `core.workflow.run_with_critique()`
  - Phase 2 generate, critique, regenerate, publish-to-review flow.
- `scripts.check_track_b`
  - Verifies LoRA and voice references.
- `scripts.check_runtime_readiness`
  - Verifies Python packages, tools, assets, and services.

### Current Approval Behavior

The workflow currently blocks inside local approval adapters:

- `WebApprovalAdapter` for storyboard approval.
- `ImageApprovalAdapter` for image candidate approval.

This is okay for local MVP use, but it is not ideal for a dashboard because:

- the UI is temporary,
- approval state is file-based,
- the pipeline waits while the web page is open,
- browser refreshes and server restarts are fragile,
- pending approvals are not easy to list centrally,
- and the final approved storyboard/images are not modeled as first-class durable resources.

### Current Persistence

The project already persists:

- `jobs`
- `stage_results`
- JSON artifacts

But for a dashboard, the current store should be extended with:

- job events,
- stage attempts,
- approval requests,
- artifact metadata,
- worker heartbeats,
- explicit current stage,
- debug/error records,
- and later chat sessions/messages.

---

## 5. Proposed Architecture

```text
Browser Dashboard
  |
  | HTTP + SSE
  v
FastAPI Dashboard API
  |
  | reads/writes
  v
Dashboard Repository
  |
  | jobs, events, approvals, artifacts, queue
  v
SQLite for local dev / Postgres for production

Separate Worker Process
  |
  | claims queued jobs
  v
core.workflow / existing adapters
  |
  | writes stage artifacts and events
  v
Artifact Store: local filesystem / S3-compatible storage

Future Chatbot
  |
  | tool calls to the same Dashboard API
  v
FastAPI Dashboard API
```

### Recommended Processes

For local development:

```text
uvicorn services.dashboard_api:app --host 0.0.0.0 --port 8080
python -m services.dashboard_worker
```

For production/GPU machine:

```text
dashboard-api      - FastAPI server
dashboard-worker   - one GPU job at a time by default
ollama             - LLM/VLM
comfyui            - LTX video
fish-s2            - TTS
postgres           - durable job store
s3/minio           - durable artifact store
```

### Why Separate Worker Instead of FastAPI BackgroundTask

FastAPI `BackgroundTasks` are fine for tiny jobs, but this pipeline can run for minutes or hours.
If the API process restarts, in-process background tasks disappear. A separate worker process gives
us:

- recoverable queued jobs,
- job claiming/locking,
- heartbeats,
- controlled GPU concurrency,
- simpler restart behavior,
- better failure isolation,
- and cleaner debugging.

MVP can optionally support in-process jobs for developer convenience, but the production path should
be a separate worker.

---

## 6. Job State Model

The dashboard needs more precise statuses than the current `JobStatus` enum alone. Keep the existing
job status values, but add dashboard-specific fields.

### Job Statuses

```text
created
queued
running
pending_plan_approval
pending_image_approval
pending_final_review
completed
blocked
failed
cancel_requested
cancelled
stalled
```

The existing `JobStatus.PENDING_APPROVAL` can remain for compatibility, but the dashboard should
distinguish the approval kind in a separate field:

```text
approval_kind = null | plan | images | final_publish
```

### Stage Statuses

```text
pending
running
completed
failed
skipped
blocked
waiting_for_approval
cancelled
stalled
```

### Core Job Fields

Each job detail response should include:

```json
{
  "job_id": "20260629-235051-ydm",
  "source_url": "https://example.com/video",
  "status": "running",
  "phase": "all",
  "target_language": "en",
  "rights_cleared": true,
  "current_stage": "generate_video",
  "current_shot_id": "s03",
  "approval_kind": null,
  "created_at": "2026-06-29T23:50:51Z",
  "queued_at": "2026-06-29T23:50:52Z",
  "started_at": "2026-06-29T23:51:00Z",
  "updated_at": "2026-06-29T23:58:12Z",
  "last_heartbeat_at": "2026-06-29T23:58:10Z",
  "completed_at": null,
  "progress": {
    "stage_index": 8,
    "stage_count": 14,
    "message": "Generating LTX video for shot s03",
    "percent": null
  },
  "terminal_error": null
}
```

### State Transition Rules

- `POST /api/jobs` creates a job as `queued`.
- The worker claims it and sets `running`.
- The worker writes `current_stage` before each stage.
- If a stage needs approval, the worker creates an approval request, sets the job to
  `pending_plan_approval` or `pending_image_approval`, then stops cleanly.
- Approval endpoints persist the operator decision and enqueue the resume action.
- The worker resumes from the correct phase or stage.
- Terminal states are `completed`, `blocked`, `failed`, and `cancelled`.
- The dashboard derives `stalled` if a running job has no heartbeat after a configured threshold.

---

## 7. Database Additions

The existing `jobs` and `stage_results` tables can remain. Add dashboard tables rather than forcing
all UI needs into the existing opaque job payload.

### `dashboard_jobs`

Canonical list/detail table for the dashboard.

```sql
CREATE TABLE dashboard_jobs (
  job_id TEXT PRIMARY KEY,
  source_url TEXT NOT NULL,
  source_kind TEXT NOT NULL, -- url | upload | file
  status TEXT NOT NULL,
  phase TEXT NOT NULL,
  target_language TEXT NOT NULL,
  rights_cleared BOOLEAN NOT NULL,
  current_stage TEXT,
  current_shot_id TEXT,
  approval_kind TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  queued_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL,
  last_heartbeat_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  terminal_error_json TEXT,
  request_json TEXT NOT NULL
);
```

### `job_queue`

Durable queue for worker execution.

```sql
CREATE TABLE job_queue (
  queue_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  action TEXT NOT NULL, -- start | resume | retry_stage | rerun_shot
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL, -- queued | claimed | completed | failed | cancelled
  priority INTEGER NOT NULL DEFAULT 100,
  created_at TIMESTAMPTZ NOT NULL,
  claimed_at TIMESTAMPTZ,
  claimed_by TEXT,
  completed_at TIMESTAMPTZ,
  error_json TEXT
);
```

### `job_events`

Append-only event stream. This powers the timeline, SSE, and debug bundles.

```sql
CREATE TABLE job_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  level TEXT NOT NULL, -- debug | info | warning | error
  stage_name TEXT,
  shot_id TEXT,
  message TEXT NOT NULL,
  payload_json TEXT,
  created_at TIMESTAMPTZ NOT NULL
);
```

Examples:

```text
job_queued
worker_claimed
stage_started
stage_heartbeat
stage_completed
stage_failed
approval_requested
approval_submitted
artifact_created
job_completed
job_failed
service_health_failed
```

### `stage_runs`

Tracks attempts and timing per stage.

```sql
CREATE TABLE stage_runs (
  stage_run_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  stage_name TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  status TEXT NOT NULL,
  adapter_name TEXT,
  shot_id TEXT,
  started_at TIMESTAMPTZ,
  last_heartbeat_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  duration_sec REAL,
  error_json TEXT,
  artifact_id TEXT
);
```

### `artifacts`

Metadata index for local/S3 artifacts.

```sql
CREATE TABLE artifacts (
  artifact_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  stage_name TEXT,
  shot_id TEXT,
  kind TEXT NOT NULL, -- json | image | audio | video | log | sidecar | debug_bundle
  uri TEXT NOT NULL,
  mime_type TEXT,
  size_bytes INTEGER,
  sha256 TEXT,
  previewable BOOLEAN NOT NULL DEFAULT false,
  metadata_json TEXT,
  created_at TIMESTAMPTZ NOT NULL
);
```

### `approval_requests`

Approval state must be durable and queryable.

```sql
CREATE TABLE approval_requests (
  approval_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  kind TEXT NOT NULL, -- plan | images | final_publish
  status TEXT NOT NULL, -- pending | approved | rejected | expired | cancelled
  iteration INTEGER NOT NULL DEFAULT 1,
  request_json TEXT NOT NULL,
  response_json TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  decided_at TIMESTAMPTZ,
  reviewer TEXT
);
```

### `image_candidates`

Supports the image approval grid and chatbot image-choice explanation.

```sql
CREATE TABLE image_candidates (
  candidate_id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  shot_id TEXT NOT NULL,
  candidate_index INTEGER NOT NULL,
  image_uri TEXT NOT NULL,
  selected_by_vlm BOOLEAN NOT NULL DEFAULT false,
  selected_by_human BOOLEAN NOT NULL DEFAULT false,
  score_json TEXT,
  reasoning TEXT,
  created_at TIMESTAMPTZ NOT NULL
);
```

### `worker_heartbeats`

Used to show whether workers are alive.

```sql
CREATE TABLE worker_heartbeats (
  worker_id TEXT PRIMARY KEY,
  hostname TEXT,
  process_id INTEGER,
  version TEXT,
  current_job_id TEXT,
  started_at TIMESTAMPTZ NOT NULL,
  last_heartbeat_at TIMESTAMPTZ NOT NULL
);
```

### Future `chat_sessions` and `chat_messages`

Only add these when the chatbot starts.

```sql
CREATE TABLE chat_sessions (
  chat_session_id TEXT PRIMARY KEY,
  title TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE chat_messages (
  message_id TEXT PRIMARY KEY,
  chat_session_id TEXT NOT NULL,
  role TEXT NOT NULL, -- user | assistant | tool
  content TEXT NOT NULL,
  tool_call_json TEXT,
  created_at TIMESTAMPTZ NOT NULL
);
```

---

## 8. Backend Services

### `services/dashboard_api.py`

FastAPI app that serves:

- dashboard HTML/static files,
- REST API,
- SSE event streams,
- artifact preview/download endpoints,
- approval endpoints,
- readiness/debug endpoints,
- chatbot endpoints later.

### `services/dashboard_worker.py`

Worker that:

- claims queued actions,
- runs pipeline phases,
- emits job events,
- writes heartbeats,
- writes stage attempts,
- pauses at approvals,
- resumes after approvals,
- handles cancellation,
- records terminal errors.

### `services/dashboard_repository.py`

Repository layer for dashboard tables. It can reuse existing settings from `core.config.Settings`.

Recommended methods:

```python
create_dashboard_job(request) -> DashboardJob
enqueue_action(job_id, action, payload) -> QueueItem
claim_next_action(worker_id) -> QueueItem | None
mark_action_completed(queue_id) -> None
mark_action_failed(queue_id, error) -> None
update_job_status(job_id, status, **fields) -> None
record_event(job_id, event_type, message, **fields) -> JobEvent
upsert_stage_run(...) -> StageRun
record_artifact(...) -> ArtifactRecord
create_approval_request(...) -> ApprovalRequest
submit_approval(...) -> ApprovalRequest
heartbeat_worker(worker_id, current_job_id=None) -> None
heartbeat_job(job_id, stage_name=None, message=None) -> None
```

### `services/dashboard_pipeline.py`

Thin orchestration wrapper around existing pipeline code. This is where dashboard-specific pause and
resume behavior should live, so `core/workflow.py` does not become a web framework.

Responsibilities:

- convert API request into `AppConfig`, `RunOptions`, and workflow call,
- inject dashboard-aware approval behavior,
- write durable artifacts after critique and approval,
- emit events around long-running calls,
- translate exceptions into structured error records.

---

## 9. Approval Refactor Plan

The current approval adapters block the pipeline until a local page returns a decision. The dashboard
should instead pause the job and resume later.

### New Approval Flow

```text
worker runs plan stages
  |
  v
plan_shots + critique_plan complete
  |
  v
worker writes approval_requests(kind="plan", status="pending")
  |
  v
job status = pending_plan_approval
  |
  v
worker stops cleanly
  |
  v
operator approves/rejects in dashboard
  |
  v
approval endpoint records decision and enqueues resume
  |
  v
worker resumes render phase or replans
```

Image approval is the same pattern:

```text
worker renders all candidate images
  |
  v
worker runs image critique
  |
  v
worker writes image_candidates + approval_requests(kind="images")
  |
  v
job status = pending_image_approval
  |
  v
operator approves or overrides selections
  |
  v
worker resumes video generation with approved image URIs
```

### Required Durable Artifacts

Add these artifacts explicitly:

```text
storyboard_raw.json
plan_critique_final.json
storyboard_review.json
storyboard_approved.json
image_candidates.json
image_critique_results.json
images_approved.json
final_video.json
publish_metadata.json
```

The current `plan_shots.json` may not be the final approved storyboard after critique/replan. The
dashboard must persist the approved version separately.

### Adapter Strategy

Add a dashboard approval adapter, but do not make it serve HTML:

```python
class DashboardApprovalAdapter:
    async def request_approval(...):
        create approval request
        raise ApprovalPending(approval_id)
```

The worker catches `ApprovalPending`, marks the job as pending approval, and exits normally. After
approval, the worker resumes from the correct phase.

This avoids holding a Python coroutine open for hours.

---

## 10. API Design

All API responses should include:

```json
{
  "request_id": "req_...",
  "server_time": "2026-06-29T23:58:12Z"
}
```

Errors should use one consistent shape:

```json
{
  "request_id": "req_...",
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "ComfyUI is not responding at http://localhost:8188",
    "details": {
      "service": "comfyui",
      "url": "http://localhost:8188/system_stats"
    },
    "retryable": true
  }
}
```

### 10.1 System and Readiness APIs

#### `GET /api/health/live`

Checks only that the dashboard API process is alive.

Response:

```json
{
  "status": "ok",
  "server_time": "2026-06-29T23:58:12Z"
}
```

#### `GET /api/health/ready`

Checks API dependencies: DB reachable, artifact store reachable, worker heartbeat recent.

Response:

```json
{
  "status": "degraded",
  "checks": [
    {"name": "database", "status": "ok"},
    {"name": "artifact_store", "status": "ok"},
    {"name": "worker", "status": "warn", "message": "No worker heartbeat in 45s"}
  ]
}
```

#### `GET /api/runtime/readiness?strict=true`

Runs the same logical checks as `scripts.check_runtime_readiness`, but returns structured JSON.

Response:

```json
{
  "mode": "strict",
  "status": "fail",
  "checks": [
    {
      "name": "Python package: openai",
      "status": "pass",
      "detail": "installed"
    },
    {
      "name": "Service: ComfyUI",
      "status": "fail",
      "detail": "http://localhost:8188/system_stats connection refused"
    }
  ]
}
```

Use cases:

- dashboard settings page,
- preflight before starting a job,
- chatbot "what is broken?" answer.

#### `GET /api/runtime/services`

Returns service-specific health.

```json
{
  "services": [
    {
      "name": "ollama",
      "required": true,
      "url": "http://localhost:11434/api/tags",
      "status": "ok",
      "latency_ms": 21
    },
    {
      "name": "comfyui",
      "required": true,
      "url": "http://localhost:8188/system_stats",
      "status": "down",
      "error": "connection refused"
    }
  ]
}
```

#### `GET /api/models/llm`

Lists models available from Ollama/OpenAI-compatible endpoint.

Response:

```json
{
  "provider": "ollama",
  "base_url": "http://localhost:11434/v1",
  "default_model": "qwen3.6:35b",
  "models": [
    {"id": "qwen3.6:35b", "available": true}
  ]
}
```

If Ollama is down, return `200` with a degraded status so the UI can show the problem without
hanging:

```json
{
  "provider": "ollama",
  "status": "down",
  "models": [],
  "error": "connection refused"
}
```

#### `GET /api/config/defaults`

Returns dashboard-safe defaults for the new job form.

```json
{
  "render_adapter": "musubi_flux",
  "video_adapter": "ltx",
  "tts_adapter": "fish_s2",
  "target_language": "en",
  "image_candidates": 3,
  "approval_required": true,
  "max_gpu_jobs": 1
}
```

### 10.2 Upload APIs

#### `POST /api/uploads`

Upload a local video to the server.

Request:

```text
multipart/form-data
file=<video>
```

Response:

```json
{
  "upload_id": "upl_20260629_abc123",
  "filename": "source.mp4",
  "source_url": "file:///.../.local/uploads/upl_20260629_abc123/source.mp4",
  "size_bytes": 123456789,
  "mime_type": "video/mp4"
}
```

Rules:

- Store under `.local/uploads/`.
- Validate extension and MIME type.
- Do not allow arbitrary user-provided paths.
- Convert uploaded files to `file://` source URLs for `YtDlpAdapter`.

### 10.3 Job APIs

#### `POST /api/jobs`

Creates and queues a new job. Must return quickly.

Request:

```json
{
  "source": {
    "kind": "url",
    "url": "https://www.youtube.com/watch?v=EXAMPLE"
  },
  "rights_cleared": true,
  "target_language": "en",
  "mode": "standard",
  "phase": "all",
  "run_critique": false,
  "overrides": {
    "llm_model": "qwen3.6:35b",
    "whisper_device": "cuda",
    "whisper_compute_type": "float16",
    "render_adapter": "musubi_flux",
    "video_adapter": "ltx",
    "tts_adapter": "fish_s2",
    "image_candidates": 3,
    "auto_approve_plan": false,
    "auto_approve_images": false
  },
  "idempotency_key": "optional-client-generated-key"
}
```

Response:

```json
{
  "job_id": "20260629-235051-ydm",
  "status": "queued",
  "links": {
    "detail": "/api/jobs/20260629-235051-ydm",
    "events": "/api/jobs/20260629-235051-ydm/events",
    "stream": "/api/jobs/20260629-235051-ydm/stream"
  }
}
```

Validation:

- Reject if `rights_cleared` is false unless the request explicitly asks for a dry-run plan that
  stops before `adapt_script`. For normal video generation, rights must be explicit.
- Reject unsupported language values.
- Reject unknown adapter names.
- Run lightweight preflight checks before queueing:
  - job store writable,
  - artifact store writable,
  - source looks valid,
  - Track B assets present for real render runs.

Do not block on:

- Ollama model loading,
- ComfyUI readiness,
- TTS readiness,
- full runtime readiness.

Instead, queue the job and let the worker record a clear failure if a service is down. The UI can
show preflight warnings before submit.

#### `GET /api/jobs`

List jobs.

Query params:

```text
status=running,pending_plan_approval
limit=50
cursor=<opaque>
sort=-created_at
source_contains=counting
```

Response:

```json
{
  "items": [
    {
      "job_id": "20260629-235051-ydm",
      "source_url": "https://...",
      "status": "pending_image_approval",
      "current_stage": "approve_images",
      "target_language": "en",
      "created_at": "2026-06-29T23:50:51Z",
      "updated_at": "2026-06-29T23:58:12Z",
      "last_heartbeat_at": "2026-06-29T23:58:10Z",
      "output_available": false,
      "needs_attention": true
    }
  ],
  "next_cursor": null
}
```

#### `GET /api/jobs/{job_id}`

Job detail.

Response includes:

- job fields,
- current stage,
- stage run summaries,
- latest events,
- pending approval summary,
- output summary,
- service warnings relevant to the current stage.

Example:

```json
{
  "job": {
    "job_id": "20260629-235051-ydm",
    "status": "running",
    "current_stage": "generate_video",
    "current_shot_id": "s03",
    "last_heartbeat_at": "2026-06-29T23:58:10Z",
    "is_stale": false
  },
  "stages": [
    {
      "name": "fetch_media",
      "status": "completed",
      "duration_sec": 42.1,
      "artifact_id": "art_fetch_..."
    },
    {
      "name": "generate_video",
      "status": "running",
      "shot_id": "s03",
      "last_heartbeat_at": "2026-06-29T23:58:10Z",
      "message": "Waiting for ComfyUI prompt to complete"
    }
  ],
  "pending_approval": null,
  "output": null,
  "latest_events": []
}
```

#### `POST /api/jobs/{job_id}/resume`

Queues a resume action.

Request:

```json
{
  "phase": "render",
  "only_shot": null,
  "force": false
}
```

Response:

```json
{
  "job_id": "20260629-235051-ydm",
  "status": "queued",
  "queue_id": "queue_..."
}
```

Rules:

- Do not resume if already running unless `force=true` and no worker lock exists.
- Validate required artifacts for the phase.
- If pending approval, require approval decision first.

#### `POST /api/jobs/{job_id}/cancel`

Request cancellation.

Response:

```json
{
  "job_id": "20260629-235051-ydm",
  "status": "cancel_requested"
}
```

Worker behavior:

- check cancellation between stages,
- terminate subprocesses if safe,
- mark `cancelled`,
- leave artifacts intact.

#### `POST /api/jobs/{job_id}/retry-stage`

Retries a failed stage.

Request:

```json
{
  "stage_name": "generate_video",
  "shot_id": "s03",
  "clear_downstream_artifacts": false
}
```

Response:

```json
{
  "job_id": "20260629-235051-ydm",
  "queue_id": "queue_...",
  "status": "queued"
}
```

Rules:

- Default should preserve existing artifacts.
- If `clear_downstream_artifacts=true`, list what will be invalidated and require confirmation in
  the UI.

#### `POST /api/jobs/{job_id}/rerun-shot`

Regenerates one shot.

Request:

```json
{
  "shot_id": "s03",
  "rerun": {
    "render_candidates": true,
    "image_critique": true,
    "voice": false,
    "video": true,
    "assemble_after": true
  }
}
```

This maps to existing `RunOptions(phase="render", only_shot="s03", resume=True)` plus dashboard
artifact handling.

### 10.4 Event and Log APIs

#### `GET /api/jobs/{job_id}/events?after_event_id=123&limit=200`

Returns durable events.

Response:

```json
{
  "items": [
    {
      "event_id": 124,
      "event_type": "stage_heartbeat",
      "level": "info",
      "stage_name": "generate_video",
      "shot_id": "s03",
      "message": "ComfyUI prompt still running",
      "created_at": "2026-06-29T23:58:10Z"
    }
  ],
  "latest_event_id": 124
}
```

#### `GET /api/jobs/{job_id}/stream`

Server-Sent Events stream.

Events:

```text
event: snapshot
data: {...job detail summary...}

event: job_event
data: {...event...}

event: heartbeat
data: {"server_time":"...","latest_event_id":124}

event: terminal
data: {"status":"completed"}
```

SSE rules:

- Send heartbeat comments every 10 seconds even if nothing changes.
- Support `Last-Event-ID`.
- Client reconnects automatically.
- UI falls back to polling `/events` if SSE fails.

#### `GET /api/jobs/{job_id}/logs`

Query params:

```text
stage_name=generate_video
shot_id=s03
tail=500
level=warning,error
```

Response:

```json
{
  "items": [
    {
      "created_at": "...",
      "level": "error",
      "message": "ComfyUI prompt timed out after 600s",
      "payload": {"prompt_id": "..."}
    }
  ]
}
```

### 10.5 Approval APIs

#### `GET /api/jobs/{job_id}/plan-review`

Returns storyboard, script, critique scores, and approval status.

Response:

```json
{
  "approval": {
    "approval_id": "appr_plan_...",
    "kind": "plan",
    "status": "pending",
    "iteration": 1,
    "created_at": "..."
  },
  "script": {
    "learning_objective": {
      "concept": "counting",
      "success_phrase": "I can count to five."
    },
    "scenes": []
  },
  "storyboard": {
    "shots": [
      {
        "shot_id": "s01",
        "scene_ref": "scene-1",
        "characters_on_screen": ["max"],
        "setting": "cozy classroom",
        "camera": "medium",
        "action": "Max points at number cards",
        "duration_sec": 5.0,
        "dialogue": "Let's count to five!"
      }
    ]
  },
  "critique": {
    "verdict": "pass",
    "scores": {
      "character_fit": 0.91,
      "scene_achievability": 0.84,
      "pacing": 0.88,
      "kids_safety": 0.98,
      "visual_clarity": 0.86
    },
    "revision_notes": []
  }
}
```

#### `POST /api/jobs/{job_id}/plan-review/approve`

Approves the storyboard and queues resume.

Request:

```json
{
  "approval_id": "appr_plan_...",
  "notes": "Looks good.",
  "reviewer": "operator"
}
```

Response:

```json
{
  "job_id": "20260629-235051-ydm",
  "approval_status": "approved",
  "next_status": "queued",
  "queue_id": "queue_..."
}
```

#### `POST /api/jobs/{job_id}/plan-review/reject`

Rejects the storyboard and queues re-plan.

Request:

```json
{
  "approval_id": "appr_plan_...",
  "notes": "Shot s03 is too visually complex. Make it a table activity.",
  "reviewer": "operator"
}
```

Response:

```json
{
  "job_id": "20260629-235051-ydm",
  "approval_status": "rejected",
  "next_status": "queued",
  "queue_id": "queue_..."
}
```

Rules:

- Rejecting plan approval should not mark the job failed immediately.
- The worker should re-plan with human notes.
- Enforce a configurable rejection budget. After the second rejection, mark failed or require a
  manual "start a new plan" action.

#### `GET /api/jobs/{job_id}/image-review`

Returns image candidate grid data.

Response:

```json
{
  "approval": {
    "approval_id": "appr_images_...",
    "kind": "images",
    "status": "pending"
  },
  "shots": [
    {
      "shot_id": "s01",
      "prompt": "Max in cozy classroom; action: points at number cards",
      "vlm_winner_index": 1,
      "selected_index": 1,
      "candidates": [
        {
          "candidate_index": 0,
          "artifact_id": "art_img_0",
          "preview_url": "/api/artifacts/art_img_0/preview",
          "scores": {"character_consistency": 0.72},
          "reasoning": "Good expression, shirt color slightly off."
        },
        {
          "candidate_index": 1,
          "artifact_id": "art_img_1",
          "preview_url": "/api/artifacts/art_img_1/preview",
          "scores": {"character_consistency": 0.92},
          "reasoning": "Best character consistency."
        }
      ]
    }
  ]
}
```

#### `POST /api/jobs/{job_id}/image-review/approve`

Approves image selections and queues resume.

Request:

```json
{
  "approval_id": "appr_images_...",
  "selections": [
    {"shot_id": "s01", "candidate_index": 1},
    {"shot_id": "s02", "candidate_index": 0}
  ],
  "notes": "Use candidate 0 for shot s02 because Zoe's face is clearer.",
  "reviewer": "operator"
}
```

Response:

```json
{
  "job_id": "20260629-235051-ydm",
  "approval_status": "approved",
  "next_status": "queued",
  "queue_id": "queue_..."
}
```

Rules:

- Store overrides in `approval_requests.response_json`.
- Update `image_candidates.selected_by_human`.
- Append feedback to `assets/kids_duo/critique_feedback.jsonl` for the self-learning loop.
- Write `images_approved.json` as a durable artifact.

### 10.6 Artifact APIs

#### `GET /api/jobs/{job_id}/artifacts`

Lists artifacts.

Query params:

```text
kind=image,video
stage_name=render_character
shot_id=s03
```

Response:

```json
{
  "items": [
    {
      "artifact_id": "art_...",
      "kind": "image",
      "stage_name": "render_character",
      "shot_id": "s03",
      "mime_type": "image/png",
      "size_bytes": 1234567,
      "previewable": true,
      "preview_url": "/api/artifacts/art_.../preview",
      "download_url": "/api/artifacts/art_.../download",
      "created_at": "..."
    }
  ]
}
```

#### `GET /api/artifacts/{artifact_id}/preview`

Serves browser-safe previews for images, audio, and MP4.

Security:

- Resolve artifact by ID from DB.
- Do not accept raw path fragments.
- Verify local path is under an allowed artifact root.
- For S3, generate a short-lived signed URL or stream through the API.

#### `GET /api/artifacts/{artifact_id}/download`

Downloads the artifact.

#### `GET /api/jobs/{job_id}/output`

Returns final review video and metadata if available.

Response:

```json
{
  "status": "available",
  "video": {
    "artifact_id": "art_final_video",
    "preview_url": "/api/artifacts/art_final_video/preview",
    "download_url": "/api/artifacts/art_final_video/download"
  },
  "metadata": {
    "artifact_id": "art_metadata",
    "download_url": "/api/artifacts/art_metadata/download"
  }
}
```

### 10.7 Debug APIs

#### `GET /api/jobs/{job_id}/debug`

Returns a structured debug summary.

Response:

```json
{
  "job_id": "20260629-235051-ydm",
  "status": "failed",
  "failed_stage": "generate_video",
  "error": {
    "code": "COMFYUI_TIMEOUT",
    "message": "ComfyUI prompt timed out after 600s",
    "retryable": true
  },
  "last_successful_stage": "synthesize_voice",
  "recommended_actions": [
    {
      "label": "Check ComfyUI health",
      "api": "GET /api/runtime/services"
    },
    {
      "label": "Retry generate_video for shot s03",
      "api": "POST /api/jobs/{job_id}/retry-stage"
    }
  ],
  "recent_events": []
}
```

#### `GET /api/jobs/{job_id}/debug-bundle`

Creates and downloads a zip with:

- job detail JSON,
- request JSON,
- stage run JSON,
- job events,
- approval requests,
- artifact manifest,
- failed stage artifact JSON,
- stderr/stdout tails when captured,
- readiness results,
- service health results,
- sanitized config,
- file tree under the job work directory.

Never include secrets:

- `HF_TOKEN`,
- API keys,
- S3 secret,
- raw `.env`.

---

## 11. Worker Execution Model

### Claiming Work

Worker loop:

```text
while true:
  heartbeat worker
  claim next queued action
  if none: sleep
  run action
```

For Postgres:

```sql
SELECT * FROM job_queue
WHERE status = 'queued'
ORDER BY priority ASC, created_at ASC
FOR UPDATE SKIP LOCKED
LIMIT 1;
```

For SQLite:

- use an atomic `UPDATE ... WHERE queue_id = (...) AND status='queued'`,
- then reselect the claimed row.

### GPU Concurrency

Default:

```text
max_concurrent_gpu_jobs = 1
```

Reason:

- Ollama, Flux, LTX, and TTS can contend for VRAM.
- The pipeline already unloads Ollama before GPU-heavy stages.
- Parallel GPU jobs are a later optimization.

### Stage Heartbeats

Every stage wrapper should update:

- `dashboard_jobs.last_heartbeat_at`,
- `stage_runs.last_heartbeat_at`,
- `job_events(stage_heartbeat)`.

Recommended intervals:

```text
normal stages: every 10s
external service polling: every poll cycle
subprocess stages: every stdout/stderr chunk or every 10s
approval pending: no worker heartbeat required, job status is not running
```

### Timeouts

Use explicit timeouts per stage:

```text
fetch_media:          30 min
transcribe:           30 min
LLM stages:           10 min each
plan critique:        10 min
render_character:     20 min per candidate batch
image critique:       10 min
synthesize_voice:      5 min per shot
generate_video LTX:   15 min per shot
assemble_video:       10 min
critique video:       15 min
publish:               2 min
approval:              no worker wait; dashboard expiration can be 24h
```

The UI should show the timeout configured for the current stage.

### Cancellation

Cancellation should be cooperative first:

- mark `cancel_requested`,
- worker checks before each stage and during polling loops,
- terminate child subprocesses when safe,
- mark `cancelled`.

For subprocesses:

- send terminate,
- wait a short grace period,
- kill if still running,
- record the action in job events.

---

## 12. Preventing Stale UI and Silent Hangs

This is a core requirement.

### Backend Rules

- Every job has `updated_at`.
- Every running job has `last_heartbeat_at`.
- Every stage run has `last_heartbeat_at`.
- Worker sends stage heartbeats during long operations.
- API returns `server_time` in every response.
- SSE sends a heartbeat event every 10 seconds.
- The API can derive `is_stale` when `server_time - last_heartbeat_at` exceeds threshold.

Recommended stale thresholds:

```text
warning: 60s without heartbeat
stalled: 180s without heartbeat
failed worker suspicion: worker heartbeat missing for 120s
```

### Frontend Rules

The job detail page should always show:

- last server update time,
- last job heartbeat time,
- current stage start time,
- elapsed time in current stage,
- stale warning if no update,
- reconnect state for SSE,
- polling fallback state.

UI states:

```text
Live              - SSE connected, heartbeat recent.
Reconnecting      - SSE lost, retrying, polling fallback active.
Possibly stalled  - job running but heartbeat older than warning threshold.
Stalled           - heartbeat older than stalled threshold.
Waiting on you    - pending approval.
Failed            - terminal error available.
```

### No Infinite Spinner Rule

Every loading region must have:

- skeleton state for initial load,
- empty state,
- error state,
- stale state,
- retry button,
- last refreshed timestamp.

### Event Stream Fallback

Primary:

```text
SSE /api/jobs/{job_id}/stream
```

Fallback:

```text
poll GET /api/jobs/{job_id} every 5s
poll GET /api/jobs/{job_id}/events?after_event_id=N every 5s
```

### Snapshot plus Events

The UI should not reconstruct the world only from events. It should use:

- `GET /api/jobs/{id}` as canonical snapshot,
- SSE/events as fast updates.

On reconnect:

1. fetch snapshot,
2. resume events from last seen ID,
3. if event gap is too large, refetch snapshot again.

---

## 13. Failure Handling and Debugging

### Error Taxonomy

Use structured error codes.

```text
RIGHTS_NOT_CLEARED
TRACK_B_MISSING
SERVICE_UNAVAILABLE
SERVICE_TIMEOUT
LLM_INVALID_JSON
LLM_REFUSAL_OR_EMPTY
ARTIFACT_MISSING
FILE_NOT_FOUND
SUBPROCESS_FAILED
FFMPEG_FAILED
GPU_OOM
APPROVAL_REJECTED_TWICE
APPROVAL_EXPIRED
WORKER_LOST
QUEUE_CLAIM_FAILED
STORAGE_ERROR
UNKNOWN_ERROR
```

### Error Record Shape

```json
{
  "code": "SERVICE_TIMEOUT",
  "message": "ComfyUI prompt timed out after 600 seconds.",
  "stage_name": "generate_video",
  "shot_id": "s03",
  "adapter_name": "ltx",
  "retryable": true,
  "operator_hint": "Check ComfyUI, then retry the stage.",
  "details": {
    "service": "comfyui",
    "url": "http://localhost:8188",
    "timeout_sec": 600
  },
  "stderr_tail": null
}
```

### Stage-Specific Debug Guide

#### `fetch_media`

Common failures:

- `yt-dlp` missing,
- source URL unsupported,
- source platform blocks download,
- ffmpeg extraction fails.

Dashboard should show:

- command stderr tail,
- source URL,
- duration if detected,
- whether local file path exists,
- retry action.

#### `transcribe`

Common failures:

- `faster-whisper` missing,
- CUDA mismatch,
- missing audio file,
- model download problem.

Dashboard should show:

- audio artifact,
- Whisper device and compute type,
- error tail,
- suggestion to switch to CPU for debugging.

#### `analyze_content`, `adapt_script`, `plan_shots`

Common failures:

- Ollama down,
- model missing,
- invalid JSON,
- LLM returns empty content,
- stale prompt assumptions.

Dashboard should show:

- model name,
- base URL,
- redacted prompt preview,
- raw response preview if safe,
- repaired JSON attempt status,
- retry action.

#### `critique_plan`

Common failures:

- LLM down,
- invalid JSON,
- excessive revision loop,
- low scores.

Dashboard should show:

- critique scores,
- revision notes,
- attempt count,
- approve/reject controls if appropriate.

#### `render_character`

Common failures:

- missing LoRA,
- placeholder LoRA not allowed,
- missing musubi-tuner script,
- missing Flux model files,
- subprocess failure,
- GPU OOM.

Dashboard should show:

- member ID,
- LoRA path/name,
- prompt,
- generated candidate paths,
- subprocess output tail,
- Track B check result.

#### `critique_images`

Common failures:

- VLM unavailable,
- candidate image missing,
- invalid JSON,
- feedback log write failure.

Dashboard should show:

- all candidates,
- scores,
- VLM reasoning,
- feedback log status.

#### `synthesize_voice`

Common failures:

- missing voice reference,
- Fish S2 down,
- unsupported language,
- TTS timeout.

Dashboard should show:

- speaker,
- voice profile ref,
- text,
- language,
- generated audio preview if available.

#### `generate_video`

Common failures:

- ComfyUI down,
- workflow template error,
- image missing,
- audio missing,
- prompt timeout,
- output not found,
- GPU OOM.

Dashboard should show:

- shot ID,
- prompt,
- approved image preview,
- audio preview,
- ComfyUI prompt ID,
- polling history,
- timeout,
- retry shot/stage action.

#### `assemble_video`

Common failures:

- clip missing,
- audio missing,
- ffmpeg missing,
- ffmpeg codec error,
- caption escaping issue.

Dashboard should show:

- concat list,
- caption file,
- ffmpeg args with paths safe to view,
- stderr tail,
- which clip is missing.

#### `publish`

Common failures:

- review dir not writable,
- final video missing,
- rights flag false.

Dashboard should show:

- review path,
- metadata sidecar,
- rights and disclosure flags.

### Debug Bundle

Every failed job page should have "Download Debug Bundle".

Bundle contents:

```text
job.json
request.json
stage_runs.json
events.jsonl
approval_requests.json
artifacts_manifest.json
runtime_readiness.json
services_health.json
sanitized_settings.json
work_dir_tree.txt
failed_stage_payload.json
stderr_tail.txt
```

---

## 14. Frontend Dashboard Views

### 14.1 New Job View

Controls:

- source URL input,
- upload video button,
- rights-cleared checkbox,
- target language segmented control: English, Hindi, Both,
- run mode: standard, with critique,
- phase: all, plan only,
- adapters display with default values,
- advanced overrides collapsible panel,
- preflight panel.

Important UX:

- Submit button disabled until rights checkbox is checked for real runs.
- Show warning if runtime readiness has failures.
- Allow "queue anyway" only for failures that are not guaranteed blockers.
- After submit, redirect to job detail page immediately.

### 14.2 Jobs List

Columns:

- status,
- needs attention,
- job ID,
- source,
- language,
- current stage,
- created,
- last update,
- output,
- actions.

Filters:

- running,
- waiting for approval,
- failed,
- completed,
- blocked.

Actions:

- view,
- resume,
- cancel,
- debug,
- output.

### 14.3 Job Detail

Sections:

- summary header,
- live/stale indicator,
- stage timeline,
- current stage panel,
- shot grid,
- latest events,
- artifacts,
- debug recommendations.

Stage timeline should be dense and practical:

```text
fetch_media [done 42s]
transcribe [done 3m12s]
analyze_content [done 22s]
adapt_script [done 18s]
plan_shots [done 21s]
plan_approval [waiting on you]
```

### 14.4 Plan Approval View

Show:

- learning objective,
- script by scene,
- storyboard shots,
- character list,
- critique scores,
- critique revision notes,
- approve button,
- reject with notes.

Do not hide low critique scores. Highlight them.

### 14.5 Image Approval View

Show:

- one row/card per shot,
- all image candidates,
- selected winner,
- VLM scores/reasoning,
- human override selection,
- approve all button,
- per-shot notes.

The UI should allow batch approval but make overrides obvious.

### 14.6 Output View

Show:

- final video preview,
- metadata JSON,
- review folder output,
- download links,
- debug bundle link,
- final human review status.

### 14.7 System Health View

Show:

- dashboard API health,
- worker heartbeat,
- runtime readiness,
- Track B assets,
- Ollama model list,
- ComfyUI health,
- Fish S2 health,
- disk space,
- queue length.

---

## 15. LLM Usage

### Dashboard

The dashboard itself does not need an LLM.

It should call normal APIs, display state, and let the existing pipeline use its current LLM/VLM
adapters:

- `qwen3.6:35b` via Ollama/OpenAI-compatible API for:
  - `analyze_content`,
  - `adapt_script`,
  - `plan_shots`,
  - `critique_plan`,
  - `critique_images`,
  - video/frame critique.

This is good because the dashboard still works when the LLM is down. It can show "Ollama down"
instead of becoming unusable.

### Future Chatbot

The chatbot is optional and should use the dashboard API as tools.

Recommended default model:

```text
VIDEO_ME_CHAT_MODEL=qwen3.6:35b
VIDEO_ME_CHAT_BASE_URL=http://localhost:11434/v1
```

Reasons:

- already part of the stack,
- local/self-hosted,
- same OpenAI-compatible client pattern,
- can inspect job state and summarize failures.

Optional cloud model:

```text
VIDEO_ME_CHAT_MODEL=<cloud-tool-calling-model>
VIDEO_ME_CHAT_BASE_URL=<provider-compatible-url>
```

Use a cloud model only if local tool calling is unreliable. The dashboard API should not care which
chat model is used.

### Chatbot Safety Rules

The chatbot may:

- create jobs,
- resume jobs,
- inspect status,
- summarize failures,
- suggest fixes,
- ask for approval,
- submit approval only after explicit user instruction,
- queue retries after explicit user instruction.

The chatbot may not:

- invent rights clearance,
- bypass approval gates,
- auto-publish live kids content,
- delete artifacts without explicit confirmation,
- run arbitrary shell commands,
- read secrets,
- expose raw local paths unnecessarily.

### Chatbot Tool List

The chatbot tools should map directly to APIs:

```text
check_readiness             -> GET /api/runtime/readiness
list_services               -> GET /api/runtime/services
list_models                 -> GET /api/models/llm
create_job                  -> POST /api/jobs
list_jobs                   -> GET /api/jobs
get_job                     -> GET /api/jobs/{job_id}
get_job_events              -> GET /api/jobs/{job_id}/events
resume_job                  -> POST /api/jobs/{job_id}/resume
cancel_job                  -> POST /api/jobs/{job_id}/cancel
get_plan_review             -> GET /api/jobs/{job_id}/plan-review
approve_plan                -> POST /api/jobs/{job_id}/plan-review/approve
reject_plan                 -> POST /api/jobs/{job_id}/plan-review/reject
get_image_review            -> GET /api/jobs/{job_id}/image-review
approve_images              -> POST /api/jobs/{job_id}/image-review/approve
list_artifacts              -> GET /api/jobs/{job_id}/artifacts
get_output                  -> GET /api/jobs/{job_id}/output
get_debug_summary           -> GET /api/jobs/{job_id}/debug
create_debug_bundle         -> GET /api/jobs/{job_id}/debug-bundle
```

### Chatbot Conversation Examples

User:

```text
Create a Hindi short from this URL: ...
```

Assistant:

```text
Do you confirm you have rights to transform this source into an original educational short?
```

User:

```text
Yes.
```

Assistant tool call:

```json
{
  "tool": "create_job",
  "arguments": {
    "source": {"kind": "url", "url": "..."},
    "rights_cleared": true,
    "target_language": "hi",
    "phase": "all"
  }
}
```

Assistant:

```text
Job 20260629-235051-ydm is queued. I will watch it and tell you when it needs approval.
```

Later:

```text
The storyboard is ready. All critique scores pass except visual clarity is borderline on shot s04.
I recommend rejecting with: "Make shot s04 a simple table activity." Should I send that rejection?
```

The user must approve the action before the bot calls `reject_plan`.

---

## 16. Security and Access Control

### Local MVP

If dashboard runs only on localhost or private RunPod access, use a simple operator token:

```text
VIDEO_ME_DASHBOARD_TOKEN=<random>
```

Require:

```text
Authorization: Bearer <token>
```

for mutating endpoints:

- create job,
- resume,
- cancel,
- retry,
- approve,
- reject,
- debug bundle.

### Remote Access

If exposed beyond localhost:

- use HTTPS,
- require auth,
- protect artifact downloads,
- avoid exposing service control endpoints,
- redact secrets in all debug views,
- add audit log for approvals and retries.

### Artifact Path Safety

Never serve arbitrary paths.

Safe pattern:

- client asks for `artifact_id`,
- API looks up URI,
- API validates path is under allowed roots,
- API streams file or redirects to signed URL.

Allowed local roots:

```text
.local/
review/
assets/kids_duo/
voices/
loras/ only for metadata, not raw download by default
```

---

## 17. Implementation Phases

### Phase D0 - Safety Fixes and Prep

Purpose: clean up obvious issues before the dashboard depends on them.

Tasks:

- Fix `run_pipeline.py` so `--rights-cleared` defaults to false.
- Ensure `target_language` respects config unless explicitly passed.
- Fix readiness logic for `musubi_flux` so it does not require A1111.
- Persist final approved storyboard separately from raw `plan_shots`.
- Persist approved image selections separately from raw critique results.

Acceptance:

- CLI still works.
- Track B check works.
- Readiness reports the correct default stack.
- Plan/image approval artifacts are durable.

### Phase D1 - Dashboard Repository and Event Model

Tasks:

- Add dashboard tables.
- Add `DashboardRepository`.
- Add migrations or idempotent schema init.
- Add event writing helpers.
- Add artifact indexing helper.

Acceptance:

- Can create/list dashboard jobs in SQLite.
- Can append/read job events.
- Can record worker heartbeat.
- Can record approval request.

### Phase D2 - API Skeleton

Tasks:

- Add `services/dashboard_api.py`.
- Add health/readiness endpoints.
- Add job create/list/detail endpoints.
- Add event polling endpoint.
- Add artifact listing endpoint.
- Add token auth for mutating endpoints.

Acceptance:

- API starts on port 8080.
- `GET /api/health/live` works.
- `POST /api/jobs` creates queued job and returns immediately.
- `GET /api/jobs` lists it.
- `GET /api/jobs/{id}` shows it.

### Phase D3 - Worker

Tasks:

- Add `services/dashboard_worker.py`.
- Worker claims queued jobs.
- Worker writes heartbeats.
- Worker runs no-op or plan-only job first.
- Worker records terminal success/failure.
- Worker handles cancellation between stages.

Acceptance:

- Create job from API.
- Worker picks it up.
- Dashboard status changes without CLI.
- Worker restart does not corrupt queued jobs.

### Phase D4 - Dashboard UI MVP

Tasks:

- Add static HTML/JS/CSS.
- New job form.
- Jobs list.
- Job detail page.
- Polling status updates.
- Stage timeline.
- Events panel.
- Runtime health page.

Acceptance:

- User can start a plan-only job from browser.
- User sees progress and failure state.
- No indefinite spinner.
- Refreshing browser preserves state.

### Phase D5 - Dashboard Approval Gates

Tasks:

- Add dashboard approval adapter or controller path.
- Plan approval page.
- Plan approve/reject endpoints.
- Image candidate indexing.
- Image approval page.
- Image approve endpoint.
- Resume job after approval.

Acceptance:

- Pipeline pauses at plan approval.
- User approves in dashboard.
- Worker resumes.
- Pipeline pauses at image approval.
- User overrides a candidate.
- Worker resumes with approved images.

### Phase D6 - Artifacts, Preview, and Debug

Tasks:

- Artifact preview/download endpoints.
- Final output page.
- Debug summary endpoint.
- Debug bundle endpoint.
- Stage-specific error display.
- Retry-stage and rerun-shot endpoints.

Acceptance:

- User can preview images/video/audio.
- Failed job shows concrete error and recommended action.
- Debug bundle downloads.
- Retry failed stage works for supported stages.

### Phase D7 - SSE Live Updates

Tasks:

- Add SSE endpoint.
- Add frontend SSE client.
- Add reconnect fallback to polling.
- Add stale indicators.

Acceptance:

- Job detail updates live.
- Lost SSE connection is visible.
- UI falls back to polling.
- Running job with missing heartbeat shows stale/stalled warning.

### Phase D8 - Chatbot

Tasks:

- Add chat config.
- Add chat session/message tables.
- Add `services/chatbot_orchestrator.py`.
- Add tool definitions for dashboard APIs.
- Add chat UI panel or separate chat page.
- Add explicit confirmation rules.

Acceptance:

- User can ask chatbot to check readiness.
- User can ask chatbot to start a job.
- Bot asks for rights confirmation before create job.
- Bot can summarize pending approvals.
- Bot can submit approval only after explicit user confirmation.
- Dashboard remains usable if chat model is down.

---

## 18. Open Design Decisions

### Frontend Stack

Recommended MVP:

```text
FastAPI + Jinja/HTML + vanilla JS + CSS
```

Reason:

- simple deployment,
- no Node build step,
- enough for ops dashboard.

Possible later:

```text
React/Vite or Next.js
```

Only move there if the UI becomes complex enough to justify it.

### Queue Backend

Recommended MVP:

```text
DB-backed queue in SQLite/Postgres
```

Possible later:

```text
Redis + RQ/Celery/Arq
```

Only add Redis when multiple workers or higher concurrency become necessary.

### Auth

Recommended MVP:

```text
single operator token
```

Possible later:

```text
user accounts + roles
```

### Chat Model

Recommended default:

```text
qwen3.6:35b through existing Ollama endpoint
```

Possible later:

```text
cloud tool-calling model
```

Only use a cloud model if local tool use is unreliable.

---

## 19. Recommended First Build Slice

The first slice should prove the architecture without touching GPU-heavy work:

1. Add dashboard DB tables and repository.
2. Add API server with:
   - `GET /api/health/live`
   - `GET /api/runtime/readiness`
   - `POST /api/jobs`
   - `GET /api/jobs`
   - `GET /api/jobs/{id}`
   - `GET /api/jobs/{id}/events`
3. Add worker that can run `run_noop_job` or `phase=plan` with mocked approvals.
4. Add a basic dashboard page:
   - submit job,
   - list jobs,
   - view job detail,
   - see events.
5. Add stale indicators and polling fallback from day one.

Do not start with the chatbot. Build the API and job state first. Once the dashboard can reliably
control jobs, the chatbot becomes a thin tool-calling layer.

---

## 20. Summary Recommendation

Build the dashboard as the primary operator surface.

Keep:

- `core/workflow.py` as the pipeline engine,
- `run_pipeline.py` as the dev/debug CLI,
- existing adapters as execution backends.

Add:

- dashboard API,
- dashboard worker,
- durable job events,
- durable approval requests,
- artifact index,
- debug bundle,
- frontend pages,
- then chatbot tools.

The chatbot should not be the orchestrator of record. The dashboard backend should be the
orchestrator of record. The chatbot should be a helpful operator assistant that calls the same APIs
the dashboard uses.
