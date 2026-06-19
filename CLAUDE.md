# CLAUDE.md — video_me Project Context

## What this project is

`video_me` is an orchestration pipeline that turns a reference video URL into an original animated
kids' educational short starring a 4-member pig cast (Pippa, Milo, Nia, Luma). Every model is an
interchangeable adapter behind a typed capability ABC. The pipeline is guardrail-enforced — jobs
with uncleared rights or unoriginal content are blocked, not silently passed.

---

## Current state (as of 2026-06-19)

**Phase 1 code is 100% complete. 264 tests pass. Blocked on Track B (files) + Track D (services).**

| Track / Phase | Status | Blocker |
|---|---|---|
| Phase 0 — Skeleton | ✅ COMPLETE | — |
| Phase 1 — Full pipeline A1.0–A1.12 | ✅ COMPLETE (code) | Track B + Track D files/services needed to run |
| Track B — LoRAs + voice files | ❌ MISSING | Character art not finalized; LoRAs not trained |
| Track D — GPU services | ❌ NOT RUNNING | Budget decision #10 pending; no GPU provisioned |
| Track E — Compliance sign-off | ❌ PENDING | Operator hasn't signed off |
| Phase 2 — Critic loop (A2.x) | ⏳ NOT STARTED | Depends on Phase 1 running end-to-end |

The pipeline will raise `RuntimeError("Complete Track B…")` before any HTTP call if `loras/` or
`voices/` files are absent. All other code paths are tested and working.

---

## Architecture

```
source URL
    │
    ▼
[fetch_media]        yt-dlp download + ffmpeg audio extraction
    │
    ▼
[transcribe]         faster-whisper → TranscribeResult (segments + timestamps)
    │
    ▼
[analyze_content]    LLM → ContentMetadata + LearningObjective
    │
    ▼
check_rights()  ◄─── BLOCKS job (status=BLOCKED) if rights_cleared=False
    │
    ▼
[adapt_script]       LLM → Script (scenes + lines, mode=transformed)
    │
    ▼
[plan_shots]         LLM → Storyboard (Shot list, ≤2 chars/shot)
    │
    ▼ (per shot)
    ├── [render_character]   AUTOMATIC1111 SD API → ImageSet (PNG)
    ├── [synthesize_voice]   Chatterbox TTS API → AudioTrack (WAV)
    ├── [generate_video]     Wan 2.7 API → VideoClip (MP4)
    └── [lip_sync]           Wav2Lip API → VideoClip (synced MP4)
    │
    ▼
[assemble_video]     ffmpeg concat + scale 1080×1920 + captions + disclosure
    │
    ▼
[publish]            copy to review/ folder + metadata.json sidecar
```

Every `[stage]` is a `Capability[Request, Result]` ABC. Concrete adapters live in `adapters/<stage>/`.
The stage runner is `core/executor.py:run_stage()`. The full DAG is `core/workflow.py:run_pipeline_job()`.

---

## Key file map

| Path | Purpose |
|---|---|
| `core/workflow.py` | `run_pipeline_job()` — full 9-stage DAG; `run_noop_job()` — Phase 0 compat |
| `core/executor.py` | `run_stage()` health-check→invoke→persist; `check_rights()` gate |
| `core/models/capabilities.py` | All typed request/result Pydantic models |
| `core/models/content.py` | Script, Scene, Line, Shot, Storyboard, LearningObjective |
| `core/models/profile.py` | ChannelProfile, CastMember, Cast |
| `core/models/guardrails.py` | SourceRights, SourceRightsKind |
| `core/config.py` | Settings (env/pydantic-settings) + AppConfig + load_app_config() |
| `core/storage.py` | SQLite/Postgres job store + local/S3 artifact store |
| `adapters/fetch_media/ytdlp_adapter.py` | yt-dlp + ffmpeg subprocess |
| `adapters/transcribe/whisper_adapter.py` | faster-whisper local inference |
| `adapters/analyze_content/llm_adapter.py` | Ollama/OpenAI-compat LLM |
| `adapters/adapt_script/llm_adapter.py` | Ollama/OpenAI-compat LLM + guardrail injection |
| `adapters/plan_shots/llm_adapter.py` | Ollama/OpenAI-compat LLM + shot structure derivation |
| `adapters/render_character/diffusion_adapter.py` | AUTOMATIC1111 SD API |
| `adapters/synthesize_voice/tts_adapter.py` | Chatterbox TTS HTTP API |
| `adapters/generate_video/wan_adapter.py` | Wan 2.7 HTTP API |
| `adapters/lip_sync/lip_sync_adapter.py` | Wav2Lip HTTP API |
| `adapters/assemble_video/ffmpeg_adapter.py` | ffmpeg subprocess |
| `adapters/publish/manual_adapter.py` | local file copy + metadata.json |
| `config/channels/education_kids.yaml` | Channel: 9:16, age 3-6, made_for_kids=true |
| `config/casts/pig_kids_placeholder.yaml` | 4-member pig cast with lora_ref + voice_profile_ref |
| `loras/` | LoRA weight files — **MUST EXIST** for render_character (Track B) |
| `voices/` | Reference WAV files — **MUST EXIST** for synthesize_voice (Track B) |
| `review/` | Output: `<timestamp>_<stem>/video.mp4` + `metadata.json` sidecar |
| `tests/` | 264 tests across 14 test files; no external services needed |
| `BUILD_PROGRESS.md` | Full implementation journal + decision log |
| `Agent.md` | Lead Designer agent charter |

---

## Track B — Files required before pipeline runs

`render_character` checks for LoRA files; `synthesize_voice` checks for voice files. Both raise
`RuntimeError("Complete Track B…")` before any HTTP call if files are absent.

### LoRA files (render_character)
`lora_ref` in the YAML is `loras/pig_kids_placeholder/c1`. The adapter derives the flat filename:
```
loras/
  pig_kids_placeholder_c1.safetensors   ← Pippa
  pig_kids_placeholder_c2.safetensors   ← Milo
  pig_kids_placeholder_c3.safetensors   ← Nia
  pig_kids_placeholder_c4.safetensors   ← Luma
```
Also accepts `.pt` or `.ckpt` extensions.

### Voice reference files (synthesize_voice)
`voice_profile_ref` is `voices/pig_kids_placeholder/c1`. Adapter checks nested path:
```
voices/
  pig_kids_placeholder/
    c1.wav    ← Pippa reference voice (~10s clear single-speaker speech)
    c2.wav    ← Milo
    c3.wav    ← Nia
    c4.wav    ← Luma
```
Also accepts `.mp3` or `.flac`.

---

## Track D — Services required before pipeline runs

All five services must be healthy before `run_pipeline_job()` is called. The executor calls
`capability.health()` before each stage.

| Service | Default URL | Purpose |
|---|---|---|
| Ollama | `http://localhost:11434` | LLM for analyze, adapt, plan stages |
| AUTOMATIC1111 | `http://localhost:7860` | Stable Diffusion for render_character |
| Chatterbox TTS | `http://localhost:8020` | TTS for synthesize_voice |
| Wan 2.7 | `http://localhost:8030` | Image-to-video for generate_video |
| Wav2Lip | `http://localhost:8040` | Lip sync for lip_sync stage |

Quick health check:
```bash
curl -s http://localhost:11434/api/tags | jq .
curl -s http://localhost:7860/sdapi/v1/sd-models | jq length
curl -s http://localhost:8020/health
curl -s http://localhost:8030/health
curl -s http://localhost:8040/health
```

LLM model needed: `qwen2.5:7b` (or override `VIDEO_ME_LLM_MODEL` env var — not yet in Settings).

---

## Running tests

Tests mock all HTTP calls and subprocesses — no external services needed.

```bash
# Full suite
python -m pytest -q

# One test file
python -m pytest tests/test_workflow.py -q
python -m pytest tests/test_plan_shots.py -v

# Specific test
python -m pytest tests/test_workflow.py::test_stage_call_order -v

# With coverage
python -m pytest --cov=core --cov=adapters --cov-report=term-missing -q
```

Test count by file:
- `test_workflow.py` — 22 (DAG orchestration, rights blocking, per-shot loop)
- `test_plan_shots.py` — 29
- `test_assemble_video.py` — 32
- `test_publish.py` — 26
- `test_adapt_script.py` — ~25
- `test_synthesize_voice.py` — 27
- `test_render_character.py` — 23
- `test_lip_sync.py` — 20
- `test_generate_video.py` — 18
- `test_transcribe.py`, `test_analyze_content.py`, `test_fetch_media.py` — ~10–15 each
- `test_executor.py`, `test_phase0_models.py`, `test_phase0_workflow.py` — Phase 0 tests

---

## Running the pipeline (when Track B + D are ready)

```python
import asyncio
from core.config import load_app_config
from core.workflow import run_pipeline_job

config = load_app_config()
job = asyncio.run(run_pipeline_job(
    source_url="https://www.youtube.com/watch?v=EXAMPLE",
    rights_cleared=True,   # operator confirms source is cleared for transformation
    app_config=config,
))
print(job.status)          # "completed"
# Output: review/<timestamp>_<stem>/video.mp4 + metadata.json
```

Environment overrides (via `.env` or shell):
```bash
VIDEO_ME_DATA_DIR=/data/video_me       # where job work dirs are created
VIDEO_ME_REVIEW_DIR=/data/review       # where publish output goes
VIDEO_ME_LORA_DIR=/models/loras        # where LoRA files are
VIDEO_ME_VOICE_DIR=/data/voices        # where reference WAV files are
VIDEO_ME_JOB_STORE=postgres            # use PostgreSQL instead of SQLite
VIDEO_ME_ARTIFACT_STORE=s3             # use MinIO/S3 instead of local filesystem
```

---

## Adding a new adapter (pattern reference)

1. Create `adapters/<stage>/<name>_adapter.py`
2. Subclass the ABC from `core/capabilities/<stage>.py`
3. Implement `health()`, `estimate_cost()`, `run()` — lazy-import heavy deps inside methods
4. **Track B gate**: call `_check_lora()` / `_check_voice()` BEFORE `import httpx`
5. **Stage-ordering errors**: raise `FileNotFoundError("upstream_stage must run before this_stage")`
6. Write `tests/test_<stage>.py` — mock httpx with:
```python
fake_httpx = MagicMock()
fake_httpx.AsyncClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
fake_httpx.AsyncClient.return_value.__aexit__ = AsyncMock(return_value=None)
with patch.dict(sys.modules, {"httpx": fake_httpx}):
    result = await adapter.run(request)
```

---

## Non-negotiable guardrails

These are enforced in code — pipeline blocks or raises, never silently skips.

1. **Original characters only** — cast must have `is_original_synthetic=True`; `design_constraints` forbid copying existing IP
2. **Transformative sourcing** — `rights_cleared=True` required before adapt_script; `Script.source_rights.rights_cleared` validated by Pydantic
3. **Children's safety** — human approval required before any real publish; `ManualPublishAdapter` writes to review folder only
4. **Made-for-kids + COPPA** — `ChannelProfile.made_for_kids=True`; no child-level data in any model
5. **AI disclosure** — `disclosure_label_required=True` burns label onto video via ffmpeg drawtext
6. **Phase gating** — do not advance past a phase until its acceptance criteria pass (see `orchestration-build-plan.md §9`)

---

## Open operator decisions

| # | Decision | Blocks | Current default |
|---|---|---|---|
| 1 | Confirm workflow engine | Phase 3 refactor | asyncio (core/executor.py) |
| 2 | Confirm target platform | Publish adapter upgrade | Manual review folder |
| 3 | Cast visual designs approved | Track B LoRA training | Placeholder designs |
| 10 | Build budget ceiling | Track D GPU | No GPU provisioned |
| E | Compliance posture sign-off | Track E | Unsigned |

---

## Sub-agents (invoke with /project:agent-name)

| Agent | When to use |
|---|---|
| `.claude/agents/project-status.md` | "Where are we? What's blocked? What's next?" |
| `.claude/agents/test-runner.md` | "Run tests, debug a failing test, add a new test" |
| `.claude/agents/track-b-setup.md` | "Help set up LoRAs and voice files for Track B" |
| `.claude/agents/pipeline-runner.md` | "Start services and run the pipeline end-to-end" |
