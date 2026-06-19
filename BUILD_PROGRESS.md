# Build Progress — video_me

This file tracks what has been built, why each step was taken, and which project gates remain open.
It is the implementation journal for `Agent.md`, `project-flow-and-execution-plan.md`,
`orchestration-build-plan.md`, and `phase-1-spec-kids-educational.md`.

## Current Defaults In Use

- Workflow engine: Prefect-shaped MVP default. Rationale: documented safe default when the operator
  has not chosen Temporal/Dagster/custom yet; Phase 0 keeps the workflow isolated so the engine can
  be swapped before deeper integration.
- Target platform: generic manual-review publish adapter shape. Rationale: platform choice is still
  an operator gate; Phase 0 should not assume live publishing behavior.
- Source policy: own/licensed/public-domain/transformed only. Rationale: conservative default from
  the guardrails; no job may pass script adaptation without rights clearance.
- Compute: local-only scaffold. Rationale: paid rented GPU provisioning requires operator approval.

## Phase 0 — Skeleton

### 2026-06-18

- Created the initial repo structure from the orchestration plan:
  - `core/` for contracts, models, config, storage, observability, and workflow.
  - `adapters/` for one future adapter namespace per capability, including Phase 1 `plan_shots`.
  - `guardrails/`, `subsystems/`, `services/`, `config/`, and `tests/`.
- Added Phase 0 Python packaging metadata and dependency declarations.
  - Rationale: the plan calls for Python 3.11+, Pydantic models, YAML config, and test coverage.
- Added core capability ABCs, including the Phase 1 `PlanShots` capability.
  - Rationale: the pipeline must depend on contracts, not concrete model implementations.
- Added Pydantic data models for jobs, channel profiles, casts, learning objectives, scripts,
  storyboards, artifacts, health, cost, and critique results.
  - Rationale: the source-of-truth docs require shared schemas before adapters are implemented.
- Added config loading for YAML profiles and local settings.
  - Rationale: channel/cast/model choices must live in config, not hardcoded pipeline logic.
- Added local filesystem artifact storage and SQLite-backed job recording.
  - Rationale: Phase 0 needs a no-op job to flow through and be recorded locally before cloud
  storage/PostgreSQL are provisioned.
- Added structured logging helpers.
  - Rationale: every stage must emit structured logs with job, stage, adapter, and event context.
- Added a no-op workflow runner.
  - Rationale: Phase 0 acceptance requires an empty DAG/no-op job that records structured stage
  output.
- Added Docker Compose services for local PostgreSQL and MinIO.
  - Rationale: Track D requires DB and S3-compatible storage wiring; local services are the safe
  development baseline before paid infrastructure.
- Verified Python syntax with `python3 -m compileall core scripts tests`.
- Installed local dev dependencies into `.venv` and verified the Phase 0 test suite:
  - `5 passed`.
- Ran the Phase 0 no-op workflow:
  - Recorded a completed job in `.local/video_me.db`.
  - Wrote per-stage JSON artifacts under `.local/artifacts/`.
  - Emitted structured JSON logs for job and stage lifecycle events.

## Verification Notes

- `docker compose config` could not be run because Docker is not installed or not available in this
  shell (`docker: command not found`).
- Full cloud/GPU provisioning remains intentionally unstarted because paid resources require
  operator approval.

## Open Gates

- Operator decision #1: confirm workflow engine. Current default is Prefect for MVP.
- Operator decision #2: confirm target platform. Current code keeps publish behavior generic/manual.
- Operator decision #10: budget ceiling before any paid GPU/cloud provisioning.
- Track E: compliance posture needs operator sign-off.
