# Build Progress — video_me

This file tracks what has been built, why each step was taken, and which project gates remain open.
It is the implementation journal for `Agent.md`, `project-flow-and-execution-plan.md`,
`orchestration-build-plan.md`, and `phase-1-spec-kids-educational.md`.

## Current Defaults In Use

- Workflow engine: `asyncio` (custom stage runner in `core/executor.py`). Rationale: Prefect/Temporal
  require operator approval; asyncio is the working default for Phase 1.
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

### 2026-06-19

- Verified Docker and Colima outside the sandbox:
  - Colima is running with the Docker runtime.
  - Docker Compose can parse the project Compose file.
- Started local PostgreSQL and MinIO with Docker Compose:
  - PostgreSQL is healthy on `localhost:5432`.
  - MinIO is healthy on `localhost:9000` with console on `localhost:9001`.
- Worked around a local Docker credential-helper mismatch:
  - The global Docker config references `docker-credential-desktop`, which is not installed in the
    Colima setup.
  - A local ignored `.docker-config/` was used for Compose commands in this workspace.
- Added service-backed storage wiring:
  - `VIDEO_ME_JOB_STORE=postgres` stores jobs/stage results in PostgreSQL.
  - `VIDEO_ME_ARTIFACT_STORE=s3` stores JSON artifacts in MinIO/S3.
  - SQLite and local filesystem remain the defaults.
- Added optional service dependencies:
  - `psycopg[binary]` for PostgreSQL.
  - `boto3` for S3/MinIO.
- Verified local and service-backed workflows:
  - Local test suite: `5 passed`.
  - Service smoke run completed job `b042086f-59b0-4b8e-a491-a77d95e182ad`.
  - Postgres recorded the job as `completed`.
  - MinIO stored `create_job.json`, `noop_dag.json`, and `record_result.json`.

## Verification Notes

- Docker/Colima and the local PostgreSQL/MinIO services are verified and healthy.
- Full cloud/GPU provisioning remains intentionally unstarted because paid resources require
  operator approval.

## Open Gates

- Operator decision #1: confirm workflow engine. Default changed to `asyncio` (matches reality).
- Operator decision #2: confirm target platform. Current code keeps publish behavior generic/manual.
- Operator decision #10: budget ceiling before any paid GPU/cloud provisioning.
- Track E: compliance posture needs operator sign-off.

## Gotchas (found in Phase 0 audit)

- ~~**Capability generics are type-erased.**~~ **FIXED 2026-06-19.** Created
  `core/models/capabilities.py` with typed request/result models for all 14 capabilities.
  Updated `core/capabilities/base.py` to use them; each ABC now declares its exact
  `RequestT`/`ResultT` pair.

- ~~**`core/executor.py` is missing.**~~ **FIXED 2026-06-19.** Created `core/executor.py`
  with `run_stage()` (health-check → invoke → persist artifact → update job) and `check_rights()`
  pipeline gate. 4 new tests added and passing.

- ~~**Guardrails are Pydantic validators only — not pipeline gates.**~~ **FIXED 2026-06-19.**
  `check_rights()` in `core/executor.py` sets `job.status = BLOCKED` and raises `StageError`
  when `rights_cleared` is False. Must be called from the workflow before the adapt_script stage.

- **Registry is in-memory only.** `core/registry.py` uses a plain dict. Deferred to Phase 3
  (DB-backing is part of the Phase 3 registry/router refactor). Not urgent for Phase 1 since
  routing is hardcoded.

- ~~**Cast member Pippa (c1) has `gender: boy`.**~~ **FIXED 2026-06-19.** Changed to
  `gender: girl` in `config/casts/pig_kids_placeholder.yaml`. Name and gender now consistent.

- ~~**Workflow engine default mismatch.**~~ **FIXED 2026-06-19.** `Settings.workflow_engine`
  default changed from `"prefect"` to `"asyncio"` to match the actual implementation.
  Operator can confirm or override once the engine decision is made (Open Gate #1).

- ~~**All 12 adapter directories are empty.**~~ **PARTIALLY RESOLVED 2026-06-19.** 11 of 12
  adapters implemented and tested (A1.0–A1.11). A1.12 (workflow DAG wire-up) complete.

## Phase 1 — Next Steps (sequenced; top items unblock the rest)

### Operator decisions needed first (unblocks everything below)

- [ ] Confirm workflow engine (#1) — custom asyncio is acceptable for the MVP; just make it
      explicit so adapters are built against the right runner.
- [ ] Confirm target platform (#2) — determines the shape of the `publish` adapter and the
      made-for-kids / disclosure label mechanism.
- [ ] Set build budget ceiling (#10) — required before provisioning any rented GPU.

### Track D — rented compute

- [ ] D1 Provision rented GPU account (needs budget ceiling decision above).

### Track B — cast sign-off (needed before render_character / synthesize_voice adapters)

- [x] Select final cast config: `kids_duo` with Max and Zoe.
- [ ] Approve final Max and Zoe reference sheets.
- [ ] Train per-member character LoRAs once designs are approved.
- [ ] Design and place per-member synthetic child voice references.

### Track A — framework (critical path; build in this order)

- [x] A1.0 Build `core/executor.py` — stage runner: select adapter → run → catch errors →
      persist artifact → update job. All Phase 1 adapters plug into this.
- [x] A1.0b Fix capability ABCs — replaced `Capability[BaseModel, BaseModel]` with typed
      request/result models; created `core/models/capabilities.py`.
- [x] A1.0c Add pipeline-level rights gate — `check_rights()` in `core/executor.py` blocks
      with `JobStatus.BLOCKED` when `rights_cleared` is False.
- [x] A1.1 `adapters/fetch_media/ytdlp_adapter.py` — yt-dlp + ffmpeg stream split; record
      source URL and rights decision on the Job.
- [x] A1.2 `adapters/transcribe/whisper_adapter.py` — faster-whisper; sentence segments with
      per-segment word timestamps (start/end per word for lip-sync and captions).
- [x] A1.3 `adapters/analyze_content/llm_adapter.py` — OpenAI-compatible LLM (Ollama/vLLM)
      produces ContentMetadata + LearningObjective from transcript; language and length_sec
      derived directly from transcript for reliability.
- [x] A1.4 `adapters/adapt_script/llm_adapter.py` — LLM writes scenes only; adapter
      injects mode=transformed, source_rights (always cleared), learning_objective
      from metadata, and computed caption_text. Guards on missing learning_objective.
- [x] A1.5 `adapters/plan_shots/llm_adapter.py` — LLM contributes camera/action/characters;
      adapter derives shot_id, scene_ref, setting, dialogue_line_refs, duration (word-count).
      Speaker always first; trims to ≤2 chars; fills defaults when LLM returns fewer shots.
- [x] A1.6 `adapters/render_character/diffusion_adapter.py` — AUTOMATIC1111-compatible SD API;
      lora_name derived from member.lora_ref; _check_lora raises clear Track B error when file
      missing; prompt injects <lora:name:weight> tag + visual_descriptor + setting + expression;
      base64 PNG responses decoded and saved to work_dir/member_id/; httpx AsyncClient.
- [x] A1.7 `adapters/synthesize_voice/tts_adapter.py` — Chatterbox-compatible HTTP API;
      voice_profile_ref resolved to local reference WAV (Track B gate); expression keyword
      maps to exaggeration override; SHA-1 stem for deterministic output filenames;
      duration from WAV header with word-count fallback.
- [x] A1.8 `adapters/generate_video/wan_adapter.py` — Wan 2.7-compatible HTTP API; multipart
      PNG + action prompt; style prefix/suffix wrap LLM action; trusts req.duration_sec
      (computed by plan_shots word-count); raises clear error when image missing.
- [x] A1.9 `adapters/lip_sync/lip_sync_adapter.py` — Wav2Lip-compatible HTTP API; multipart
      MP4 + WAV; _check_inputs raises clear stage-ordering errors (generate_video /
      synthesize_voice); duration from WAV header (authoritative); output: synced.mp4.
- [x] A1.10 `adapters/assemble_video/ffmpeg_adapter.py` — ffmpeg concat demuxer; scale+pad
       to 1080×1920; drawtext caption (textfile= avoids shell-quoting); AI disclosure label
       burned at top when required; audio replaced from AudioTrack; -shortest; CRF 23.
- [x] A1.11 `adapters/publish/manual_adapter.py` — copies final MP4 to timestamped review
       subdir; writes metadata.json sidecar (8 required fields); refuses to run when
       rights_cleared=False (defense-in-depth after executor gate); raises clear error when
       video file missing (assemble_video must run first).
- [x] A1.12 Wire real workflow DAG — `run_pipeline_job()` in `core/workflow.py`; full 9-stage
       sequence (fetch→transcribe→analyze→rights-gate→adapt→plan→per-shot-loop→assemble→publish);
       `_run_shot()` helper (render→voice→video→lipsync per shot); `_concat_audio()` concatenates
       per-shot WAVs with ffmpeg; `_make_adapters()` instantiates all adapters from config; error
      handling sets BLOCKED (rights) or FAILED (all others); 22 new tests (264 total).

### Phase 1 work log — 2026-06-19

Built and tested all Track A adapters A1.0–A1.12. **264 tests passing** across 12 test files.

**Core framework added:**

| File | What it does |
| --- | --- |
| `core/executor.py` | Stage runner: health-check → invoke → persist artifact → update job; `check_rights()` pipeline gate |
| `core/models/capabilities.py` | Typed request/result Pydantic models for all 14 capabilities |

**Adapters implemented:**

| Step | Adapter | Mechanism | External service / tool |
| --- | --- | --- | --- |
| A1.1 | `fetch_media` | yt-dlp + ffmpeg subprocess | yt-dlp, ffmpeg (system) |
| A1.2 | `transcribe` | faster-whisper (CTranslate2) in executor | CPU/GPU local |
| A1.3 | `analyze_content` | OpenAI-compatible LLM | Ollama `http://localhost:11434` |
| A1.4 | `adapt_script` | OpenAI-compatible LLM | Ollama `http://localhost:11434` |
| A1.5 | `plan_shots` | OpenAI-compatible LLM | Ollama `http://localhost:11434` |
| A1.6 | `render_character` | AUTOMATIC1111-compatible SD API | `http://localhost:7860` |
| A1.7 | `synthesize_voice` | Chatterbox-compatible TTS API | `http://localhost:8020` |
| A1.8 | `generate_video` | Wan 2.7-compatible API | `http://localhost:8030` |
| A1.9 | `lip_sync` | Wav2Lip-compatible API | `http://localhost:8040` |
| A1.10 | `assemble_video` | ffmpeg subprocess | ffmpeg (system) |
| A1.11 | `publish` | Local file copy | None (review folder) |

**GPU services needed to run end-to-end (Track D):**

| Service | Default port | What to run |
| --- | --- | --- |
| LLM (Ollama) | 11434 | `ollama serve` + `ollama pull qwen2.5:7b` |
| Stable Diffusion | 7860 | AUTOMATIC1111 webui |
| TTS | 8020 | FastAPI wrapper around Chatterbox |
| Wan 2.7 | 8030 | FastAPI wrapper around Wan image-to-video |
| Wav2Lip | 8040 | FastAPI wrapper around Wav2Lip/MuseTalk |

**Guardrail enforcements built in:**

- `check_rights()` in `core/executor.py` — blocks at adapt_script if `rights_cleared=False`
- `LlmAdaptScriptAdapter.run()` — guard on missing `learning_objective`
- `DiffusionRenderAdapter._check_lora()` — Track B gate; refuses with clear message if LoRA missing
- `TtsAdapter._check_voice()` — Track B gate; refuses with clear message if voice file missing
- `ManualPublishAdapter.run()` — second-line rights check; refuses if `rights_cleared=False`
- Stage-ordering errors with named upstream stage in `generate_video`, `lip_sync`, `assemble_video`, `publish`

> Acceptance criteria (Phase 1): a real educational-kids reference link produces a watchable
> ~30–60s 9:16 short starring the configured original cast, with correct captions, distinct voices,
> per-shot lip sync, and a metadata sidecar containing all required flags. A job with a non-original
> cast, missing rights, or a failed age-appropriateness check is blocked — never written as output.

## Phase 1 Complete — 2026-06-19

**All 12 Track A items built and tested. 264 tests passing. Committed as `e84c146`.**

### What was built in Phase 1

| Item | Deliverable |
| --- | --- |
| A1.0 | `core/executor.py` — `run_stage()` + `check_rights()` pipeline gate |
| A1.0b | `core/models/capabilities.py` — typed request/result models for all 14 capabilities |
| A1.1 | `adapters/fetch_media/ytdlp_adapter.py` — yt-dlp + ffmpeg subprocess |
| A1.2 | `adapters/transcribe/whisper_adapter.py` — faster-whisper local inference |
| A1.3 | `adapters/analyze_content/llm_adapter.py` — LLM → ContentMetadata + LearningObjective |
| A1.4 | `adapters/adapt_script/llm_adapter.py` — LLM → Script; guardrail injection |
| A1.5 | `adapters/plan_shots/llm_adapter.py` — LLM → Storyboard; shot structure + timing |
| A1.6 | `adapters/render_character/diffusion_adapter.py` — AUTOMATIC1111 SD API; LoRA gate |
| A1.7 | `adapters/synthesize_voice/tts_adapter.py` — Chatterbox TTS HTTP API; voice gate |
| A1.8 | `adapters/generate_video/wan_adapter.py` — Wan 2.7 HTTP API; image-to-video |
| A1.9 | `adapters/lip_sync/lip_sync_adapter.py` — Wav2Lip HTTP API; dialogue sync |
| A1.10 | `adapters/assemble_video/ffmpeg_adapter.py` — ffmpeg concat + captions + disclosure |
| A1.11 | `adapters/publish/manual_adapter.py` — review folder + metadata.json sidecar |
| A1.12 | `core/workflow.py:run_pipeline_job()` — full 9-stage DAG with `_run_shot()` loop |
| Agents | `CLAUDE.md` + `.claude/agents/` (project-status, test-runner, track-b-setup, pipeline-runner) |

### What's blocking end-to-end execution

The code is complete. Two things must happen before the pipeline can run on real content:

**Track B (file blockers)**
- `loras/kids_duo_max.safetensors` and `loras/kids_duo_zoe.safetensors` — must be trained from approved character art
- `voices/kids_duo/max.wav` and `voices/kids_duo/zoe.wav` — must be recorded, matching each character's personality
- See `.claude/agents/track-b-setup.md` for exact setup steps

**Track D (service blockers)**
- Ollama + `qwen2.5:7b` must be running on port 11434
- AUTOMATIC1111 must be running on port 7860 (with LoRA weights loaded)
- Chatterbox TTS service on port 8020
- Wan 2.7 service on port 8030
- Wav2Lip service on port 8040
- Budget decision #10 needed before provisioning rented GPU
- See `.claude/agents/pipeline-runner.md` for pre-flight and startup guide

**Track E**
- Operator compliance sign-off (sourcing policy, COPPA, disclosure, age-appropriateness rubric)

## Phase 2 Plan — Critic Loop

Phase 2 adds an automated quality gate: generate → evaluate → regenerate if failing.

### What Phase 2 builds

| Item | Deliverable |
| --- | --- |
| A2.1 | `adapters/critique/vlm_adapter.py` — VLM rates the assembled video on age-appropriateness, clarity, engagement |
| A2.1b | `core/models/capabilities.py` additions — `CritiqueRequest`, `CritiqueResult` (already modeled) |
| A2.2 | Regeneration loop in `core/workflow.py` — `run_with_critique()` wraps `run_pipeline_job()` with up to `Settings.max_regenerations` retries |
| A2.3 | Critique storage — save all CritiqueResults as artifacts; log verdict + reasons per job |
| A2.4 | Candidate selection — if multiple candidates generated, choose highest-scoring |

### Phase 2 acceptance criteria

- A video that fails the VLM age-appropriateness check is auto-regenerated (up to `max_regenerations=3`)
- Only passing candidates proceed to publish
- All critique results are persisted and queryable
- **Milestone: with the real cast in place, first-pass output is good enough to judge quality**
  This is the "test the waters" gate — pause here and have the operator evaluate before any hardware purchase

### Phase 2 pre-requisites

- Phase 1 must be running end-to-end on real content (Track B + D complete)
- A VLM (e.g. LLaVA, Qwen-VL) must be available on Ollama or as a separate service
- The age-appropriateness rubric must be defined and operator-signed-off (Track E)

### Phase 2 key decisions

- Which VLM for critique? (LLaVA-1.5 via Ollama is the cheapest starting point)
- What's the age-appropriateness rubric? (operator defines pass/fail criteria)
- What's the max cost ceiling for regenerations? (each retry = full GPU cost)

## Phase 2 Complete — 2026-06-20

**Critic-loop code is built and tested. Current suite is 313 tests passing after GPU readiness prep.**

What was added:

| Item | Deliverable |
| --- | --- |
| A2.1 | `adapters/critique/vlm_adapter.py` — automated preflight + OpenAI-compatible VLM/LLM critique |
| A2.2 | `core/workflow.py:run_with_critique()` — candidate generation, critique, regenerate/reject/pass handling |
| A2.3 | Critique persistence via `run_stage()` using `critique_attempt_1`, `critique_attempt_2`, ... |
| A2.4 | Tests for pass, regenerate, reject, max-regeneration exhaustion, and rights blocking |
| A2.5 | Frame sampling inside the critique adapter; sampled frames are attached as multimodal image inputs and recorded in `CritiqueResult.sampled_frame_uris` |

Behavior:

- `pass` → publish to manual review folder and mark job `COMPLETED`.
- `regenerate` → rerun candidate generation up to `Settings.max_regenerations` retries.
- `reject` → mark job `BLOCKED` and do not publish.
- exhausted regeneration budget → mark job `FAILED` and do not publish.

Real-world caveat:

- The adapter defaults to `llava:7b` behind an OpenAI-compatible endpoint at
  `http://localhost:11434/v1`.
- Unit tests mock the VLM. Actual quality judgment still needs Track D service setup and an
  operator-approved age-appropriateness rubric.

### Phase 2 visual-input decision: helper step now, wrapper later

Decision: implement Option 2 first — sample frames inside the critique adapter before calling the
multimodal model.

Why this choice now:

- Fastest path to a real Phase 2 MVP without adding another service boundary.
- Sampled frames live in the job work directory and are recorded on the critique result, making
  verdicts easier to debug.
- The orchestration tests can mock ffprobe/ffmpeg and the VLM independently.
- The current system already depends on ffmpeg, so this adds little local operational complexity.

When to move to Option 1, a dedicated VLM wrapper service:

- Frame extraction or visual preprocessing becomes expensive enough to colocate with GPU inference.
- We need batching, caching, scene-aware sampling, or more advanced video understanding.
- Multiple critique adapters/backends need the same preprocessing.
- Phase 3 router/self-healing needs a cleaner service health boundary for critique.
- Remote object storage becomes the normal media path and local sampling becomes awkward.

## Open Gates (updated 2026-06-20)

- **#1** Workflow engine — asyncio is the working default; Prefect/Temporal can replace if needed in Phase 3
- **#2** Target platform — review folder is the default; real platform TBD by operator
- **#3** Max/Zoe reference sheets — provisionally accepted for current LoRA training; final operator sign-off still needed before production use
- **#10** Build budget ceiling — required before any paid GPU provisioning (Track D)
- **#E** Compliance posture — Track E sign-off required before first real video publish

## Track B Kickoff — 2026-06-20

`kids_duo` is now the final cast for the current pipeline. Track B has been
reframed around Max and Zoe instead of the earlier pig placeholder cast.

Added:

- `assets/kids_duo/README.md` — reference sheet, LoRA, and voice requirements.
- `assets/kids_duo/voice_scripts.md` — clean reference recording scripts for Max and Zoe.
- `assets/kids_duo/reference/max_reference_sheet_v1.png` — first-pass Max reference sheet for review.
- `assets/kids_duo/reference/zoe_reference_sheet_v1.png` — first-pass Zoe reference sheet for review.
- `loras/README.md` — exact LoRA drop paths.
- `voices/kids_duo/README.md` — exact voice reference drop paths.
- `voices/kids_duo/VOICE_SELECTION.md` — provisional local voice choices.
- `scripts/check_track_b.py` — repeatable preflight that checks the active cast config.

Current test status:

- `python -m scripts.check_track_b` now passes LoRA checks for Max and Zoe.
- First-pass Max and Zoe reference sheets are accepted for code testing.
- Reference voice WAVs are currently missing and remain the Track B blocker.

Still pending before real end-to-end rendering:

- Final operator approval of Max and Zoe reference sheets.
- Final designed voices:
  `voices/kids_duo/max.wav` and `voices/kids_duo/zoe.wav`.
- Final operator approval/sign-off for Max and Zoe production character assets.

## GPU Readiness Prep — 2026-06-20

Goal: reduce paid GPU experimentation time by making local mock checks and GPU-machine setup
repeatable before launch.

Added:

- `Settings` now carries runtime URLs/models/tool paths:
  `VIDEO_ME_LLM_*`, `VIDEO_ME_CRITIQUE_*`, `VIDEO_ME_SD_BASE_URL`,
  `VIDEO_ME_TTS_BASE_URL`, `VIDEO_ME_WAN_BASE_URL`, `VIDEO_ME_LIPSYNC_BASE_URL`,
  `VIDEO_ME_WHISPER_*`, `VIDEO_ME_FFMPEG_BIN`, and `VIDEO_ME_FFPROBE_BIN`.
- `_make_adapters()` passes those settings into every concrete adapter.
- Temporary placeholder-LoRA smoke mode:
  `VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true` accepts explicit `TEST-ONLY placeholder`
  files and omits their LoRA tags from SD prompts. Strict/default mode still fails them.
- `scripts/check_runtime_readiness.py` checks Python packages, system tools, Track B assets,
  and service health. Default mode is strict; `--code-test --skip-services` is for local/mock
  placeholder checks.
- `scripts/setup_gpu.sh` is the primary GPU setup entrypoint: it creates/uses `.venv`, installs
  runtime Python extras, installs/checks `ffmpeg`/`ffprobe`, keeps `yt-dlp` on PATH, and runs
  readiness.
- `scripts/setup_gpu.py` and `python setup.py gpu ...` remain reusable lower-level setup commands
  without embedding install side effects in package metadata.
- Tests added for placeholder render behavior, settings-to-adapter wiring, readiness checks,
  and setup command generation.

Current validation:

- `python -m pytest -q` returns `313 passed`.
- Local readiness now passes the LoRA file checks when the trained local weights are present;
  it still fails until voice WAVs and required runtime services/tools are available.

Still pending before a real GPU launch:

- Run `bash scripts/setup_gpu.sh` on the GPU box.
- Start and validate Ollama/VLM, AUTOMATIC1111, Chatterbox, Wan, and Wav2Lip services.
- Place final voice WAVs for strict Track B readiness.

## Track B LoRA Training Complete — 2026-06-24

Trained real SD 1.5 LoRA weights locally on the A100 using `sd-scripts` and the curated
20-image-per-character datasets under `assets/kids_duo/training/images/`.

Added:

- `assets/kids_duo/training/dataset_max.toml` — concrete sd-scripts dataset config for Max.
- `assets/kids_duo/training/dataset_zoe.toml` — concrete sd-scripts dataset config for Zoe.

Training details:

| Character | Output | Steps | Rank | Base model |
| --- | --- | ---: | ---: | --- |
| Max | `loras/kids_duo_max.safetensors` | 1000 | 32 | SD 1.5 `v1-5-pruned-emaonly.safetensors` |
| Zoe | `loras/kids_duo_zoe.safetensors` | 1000 | 32 | SD 1.5 `v1-5-pruned-emaonly.safetensors` |

Notes:

- The final LoRA `.safetensors` files are now tracked in git via Git LFS. Intermediate
  checkpoints (`*-step*.safetensors`) and the smoke-test file remain gitignored.
- Venv strategy (decided 2026-06-25): each GPU service gets an isolated venv that inherits
  system torch 2.8.0+cu128 via `python3 -m venv --system-site-packages`. This avoids
  cross-service dep conflicts.
  - `/workspace/venv` — sd-scripts LoRA training
  - `/workspace/.venv_chatterbox` — Chatterbox TTS server
  - `/workspace/video_me/.venv` — pipeline orchestration code only (no heavy ML deps)
- `voices/kids_duo/max.wav` and `voices/kids_duo/zoe.wav` were generated on 2026-06-25
  using gTTS as bootstrap reference voices. These are plain English voices; replace with
  child-voice recordings for brand-accurate results.
- `python -m scripts.check_track_b` reports **Track B: READY**.

**Track B is complete as of 2026-06-25.**
