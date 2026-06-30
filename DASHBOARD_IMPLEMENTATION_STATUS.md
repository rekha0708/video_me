# Dashboard Implementation Status

This file is the live checkpoint for the web dashboard and future chatbot implementation.
Update it whenever a step starts, completes, or is deferred.

## Current Scope

Implement the first foundation slice from `WEB_DASHBOARD_CHATBOT_PLAN.md`:

- D0 safety fixes and prep.
- D1 dashboard repository and event model.
- D2 FastAPI dashboard API skeleton.

GPU-heavy worker execution, dashboard approval replacement, browser UI, SSE, and chatbot tools are
later slices.

## Status Legend

- `DONE` - implemented and locally verified where practical.
- `IN_PROGRESS` - currently being edited.
- `PENDING` - not started.
- `DEFERRED` - intentionally left for a later slice.
- `BLOCKED` - cannot proceed without a decision or dependency.

## Checkpoint

| Step | Status | Notes |
|---|---|---|
| Create checkpoint file | DONE | Initial checkpoint created before code edits. |
| D0.1 Fix CLI rights default | DONE | `run_pipeline.py --rights-cleared` now requires explicit confirmation. |
| D0.2 Fix target language default behavior | DONE | `run_pipeline_job()` now accepts `target_language=None`; CLI only overrides config when `--target-language` is provided. |
| D0.3 Fix runtime readiness service list | DONE | Default `musubi_flux + ltx + fish_s2` no longer requires A1111 and now checks ComfyUI for LTX video. |
| D1.1 Add dashboard models | DONE | Added Pydantic models for job requests, events, queue items, artifacts, approvals, and worker heartbeat. |
| D1.2 Add dashboard repository | DONE | Added SQLite-first schema and repository methods for jobs, queue, events, approvals, artifacts, and worker heartbeats. |
| D2.1 Add dashboard API skeleton | DONE | Added factory-based FastAPI app with health, readiness, defaults, jobs, events, artifacts, and cancel endpoints. |
| D2.2 Add package dependency metadata | DONE | Added `dashboard` optional dependencies in `pyproject.toml`. |
| Tests and verification | DONE | Full suite passes: `325 passed`; compileall succeeded for touched packages. |

## Latest Notes

- Implementation started with the recommended defaults: port `8080`, simple bearer token when
  configured, local SQLite/local artifact store first, dashboard API served by FastAPI, and a
  database-backed queue foundation.
- D0 code edits are in place and covered by tests.
- D1 repository code is in place and covered by tests.
- D2 API skeleton is in place as a factory (`services.dashboard_api:create_app`) so importing the
  module does not require FastAPI unless the dashboard app is actually created.
- Added resilience for missing `json_repair`: invalid LLM JSON now raises the intended
  `RuntimeError` instead of leaking `ModuleNotFoundError`; declared `json-repair` under the `llm`
  optional dependencies.
- Verification complete: targeted suite passed, full test suite passed (`325 passed`), and
  `compileall` passed for touched packages.

## Next Pending Slice

| Step | Status | Notes |
|---|---|---|
| D3.1 Add dashboard worker loop | PENDING | Claim queued jobs, heartbeat, and run a safe `noop` action first. |
| D3.2 Add worker tests | PENDING | Queue claiming, heartbeat, success/failure events. |
| D4.1 Add minimal dashboard UI | PENDING | New job form, jobs list, job detail polling. |
| D5.1 Refactor approvals to durable dashboard approvals | PENDING | Replace blocking approval adapters for dashboard-run jobs. |
