# CLAUDE.md — video_me Project Context

## What this project is

`video_me` is an orchestration pipeline that turns a reference video URL into an original animated
kids' educational short starring the final `kids_duo` cast: Max and Zoe. Every model is an
interchangeable adapter behind a typed capability ABC. The pipeline is guardrail-enforced — jobs
with uncleared rights or unoriginal content are blocked, not silently passed.

---

## Current state (as of 2026-06-25)

**Phase 2 code is complete. 313 tests pass ✅. Track B is READY. All 5 Track D GPU services are running. Pipeline running end-to-end through generate_video; MuseTalk fixes applied and first full run in progress.**

- **Fix applied**: `test_generate_video.py` now properly sets `status_code = 200` on mock responses (was missing, causing TypeError on comparison)
- **yt-dlp installed system-wide** at `/usr/local/bin/yt-dlp` (was missing from PATH — previously only in project `.venv`)
- **Chatterbox startup**: loads PerthNet model (~60s). The `wait_for` timeout in `start_services.sh` was tight; consider increasing Chatterbox's wait window if pods restart frequently.
- **LLM upgraded to qwen3.6:35b** (MoE 35B). Thinking mode disabled via `extra_body={"think": False}` + removed `response_format`. `max_tokens=16384`. `json_repair` fallback for malformed JSON.
- **VRAM unload**: `core/workflow.py` explicitly evicts qwen3.6:35b before the shot loop so Wan has full VRAM.
- **MuseTalk fixed**: mmcv rebuilt from source at 2.1.0; all 9 `torch.load` calls patched with `weights_only=False` for PyTorch 2.8 compatibility.
- **Shot duration**: 5–8s (was 2–5s); 2 words/sec, floor 5s, ceiling 8s.

| Track / Phase | Status | Blocker |
|---|---|---|
| Phase 0 — Skeleton | ✅ COMPLETE | — |
| Phase 1 — Full pipeline A1.0–A1.12 | ✅ COMPLETE (code) | — |
| Phase 2 — Critic loop A2.x | ✅ COMPLETE (code) | Real VLM service needed for real judgment |
| Track B — LoRAs + voice files | ✅ READY | Real LoRAs trained (1000 steps, rank 32, SD 1.5); voice refs generated |
| Track D — GPU services | ⚠️ Manual start required | Ollama ✅, A1111 ✅, Chatterbox ✅ (60s load time), Wan ✅, MuseTalk ✅ |
| Track E — Compliance sign-off | ❌ PENDING | Operator hasn't signed off |

Track B LoRAs are real trained weights (37 MB each, in git via LFS). Voice reference files are
gTTS bootstrap WAVs — acceptable for pipeline runs, replace with recorded child voices for
brand-accurate results.

Pipeline runs through all LLM stages (analyze → adapt → plan) without issue. MuseTalk is patched
and services confirmed healthy. First complete end-to-end run (including lip_sync) in progress.

**After every pod restart, run:**
```bash
bash scripts/start_services.sh
```
This script auto-reinstalls Ollama (base Linux binary is wiped on restart), then starts all 5 services and verifies each health endpoint.

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
[critique]           VLM/LLM rubric → pass | regenerate | reject (Phase 2 path)
                     samples frames locally with ffprobe/ffmpeg for visual input
    │
    ▼
[publish]            copy to review/ folder + metadata.json sidecar
```

Every `[stage]` is a `Capability[Request, Result]` ABC. Concrete adapters live in `adapters/<stage>/`.
The stage runner is `core/executor.py:run_stage()`. The Phase 1 DAG is
`core/workflow.py:run_pipeline_job()`; the Phase 2 critic loop is
`core/workflow.py:run_with_critique()`.

---

## Key file map

| Path | Purpose |
|---|---|
| `core/workflow.py` | `run_pipeline_job()` — Phase 1 DAG; `run_with_critique()` — Phase 2 loop; `run_noop_job()` — Phase 0 compat |
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
| `adapters/critique/vlm_adapter.py` | OpenAI-compatible VLM/LLM critique adapter |
| `adapters/publish/manual_adapter.py` | local file copy + metadata.json |
| `config/channels/education_kids.yaml` | Channel: 9:16, age 3-6, made_for_kids=true |
| `config/casts/kids_duo.yaml` | final Max/Zoe cast with lora_ref + voice_profile_ref |
| `assets/kids_duo/` | Track B reference plan, LoRA notes, and voice scripts |
| `loras/` | LoRA weight files — **MUST EXIST** for render_character (Track B) |
| `voices/` | Reference WAV files — **MUST EXIST** for synthesize_voice (Track B) |
| `review/` | Output: `<timestamp>_<stem>/video.mp4` + `metadata.json` sidecar |
| `scripts/check_track_b.py` | Track B asset placement check |
| `scripts/check_runtime_readiness.py` | runtime dependency/service/asset readiness check |
| `scripts/setup_gpu.sh` | one-command GPU-machine setup + validation |
| `scripts/setup_gpu.py` / `setup.py gpu` | lower-level GPU-machine setup helper |
| `tests/` | 313 tests across 17 test files; no external services needed (all passing as of 2026-06-25) |
| `BUILD_PROGRESS.md` | Full implementation journal + decision log |
| `Agent.md` | Lead Designer agent charter |

---

## Track B — Files required before pipeline runs

`render_character` checks for LoRA files; `synthesize_voice` checks for voice files. Both raise
`RuntimeError("Complete Track B…")` before any HTTP call if files are absent.

### LoRA files (render_character)
`lora_ref` in the YAML is `loras/kids_duo/max`. The adapter derives the flat filename:
```
loras/
  kids_duo_max.safetensors   ← Max
  kids_duo_zoe.safetensors   ← Zoe
```
Also accepts `.pt` or `.ckpt` extensions.

### Voice reference files (synthesize_voice)
`voice_profile_ref` is `voices/kids_duo/max`. Adapter checks nested path:
```
voices/
  kids_duo/
    max.wav    ← Max reference voice (~10–30s clear single-speaker speech)
    zoe.wav    ← Zoe
```
Also accepts `.mp3` or `.flac`.

Quick check:
```bash
python -m scripts.check_track_b
```
Current status: `Track B: READY` (real trained LoRAs + bootstrap voice refs in place).

Temporary placeholder-LoRA render smoke tests are opt-in:
```bash
export VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true
```
When this is true, explicit `TEST-ONLY placeholder` LoRA files are accepted and omitted from
the SD prompt. Keep it false for real runs; strict readiness fails placeholder LoRAs.

---

## Venv strategy (as of 2026-06-25)

Each GPU service uses an **isolated venv that inherits system torch 2.8.0+cu128** via
`python3 -m venv --system-site-packages`. This avoids cross-service dependency conflicts.

| Venv | Purpose | Key extra packages |
|---|---|---|
| `/workspace/video_me/.venv` | Pipeline orchestration + tests (no heavy ML) | httpx, faster-whisper, pydantic-settings |
| `/workspace/venv` | sd-scripts LoRA training | sd-scripts deps |
| `/workspace/.venv_chatterbox` | Chatterbox TTS server (port 8020) | chatterbox-tts, torchaudio==2.8.0+cu128, resemble-perth |
| `/workspace/.venv_wan` | Wan2.2 i2v server (port 8030) | decord, diffusers, transformers, accelerate, peft, librosa, moviepy, dashscope, rotary-embedding-torch, python-multipart |
| `/workspace/.venv_musetalk` | MuseTalk lip-sync server (port 8040) | opencv, librosa, einops, diffusers, mmengine, mmpose==1.3.2, mmcv==2.2.0 (built from source), face-alignment |
| AUTOMATIC1111 self-managed venv | SD rendering (port 7860) | leave untouched |

**Chatterbox fix note**: `resemble-perth` requires `pkg_resources` from setuptools<81.
Run `pip install "setuptools<81"` inside `.venv_chatterbox` if it fails on startup.
Do NOT install `perth` (wrong package on PyPI); it must be `resemble-perth`.

**MuseTalk notes**:
- mmcv **must be built from source at v2.1.0** (not 2.2.0): `MAX_JOBS=8 pip install mmcv==2.1.0 --no-build-isolation` (~20 min). mmdet 3.3.0 requires `mmcv<2.2.0`.
- mmpose 1.3.2 required (1.1.0 requires mmcv ≤2.1.0; 1.3.2 accepts <3.0.0).
- **PyTorch 2.8 `torch.load` fix**: all 9 checkpoint load calls patched with `weights_only=False` in mmengine/runner/checkpoint.py and 4 MuseTalk source files.
- musetalk package must be on PYTHONPATH since inference lives in `scripts/` not repo root.
- `start_services.sh` sets `PYTHONPATH=/workspace/MuseTalk` automatically.

**Ollama is in base Linux** (`/usr/local/bin/ollama`) and is WIPED on RunPod pod restart.
`start_services.sh` detects the missing binary and reinstalls via `curl | sh` before starting.
Models at `/workspace/ollama/` persist on the network volume (qwen3:14b + qwen2.5:7b + llava:7b).

`start_services.sh` uses the correct interpreter for each service. Never install
heavy ML packages into the project `.venv` — keep it lightweight for fast CI.

---

## Track D — Services required before pipeline runs

All five services must be healthy before `run_pipeline_job()` is called. The executor calls
`capability.health()` before each stage.

| Service | Default URL | Purpose |
|---|---|---|
| Ollama | `http://localhost:11434` | LLM for analyze, adapt, plan stages |
| Ollama / VLM | `http://localhost:11434` | Critique stage, e.g. LLaVA/Qwen-VL |
| AUTOMATIC1111 | `http://localhost:7860` | Stable Diffusion for render_character |
| Chatterbox TTS | `http://localhost:8020` | TTS for synthesize_voice |
| Wan 2.7 | `http://localhost:8030` | Image-to-video for generate_video |
| Wav2Lip | `http://localhost:8040` | Lip sync for lip_sync stage |

Quick health check:
```bash
python -m scripts.check_runtime_readiness
```

GPU-machine setup helper:
```bash
bash scripts/setup_gpu.sh
```

Local/mock placeholder check without services:
```bash
bash scripts/setup_gpu.sh --code-test --skip-services
```

LLM model needed: `qwen3.6:35b` (MoE 35B, 29–30 GB VRAM); critique defaults to `llava:7b`. Rollback: `VIDEO_ME_LLM_MODEL=qwen3:14b`. Phase 2 samples local video
frames in the adapter and sends them as multimodal `image_url` data URLs. This keeps the MVP
inspectable because sampled frames are saved under the job work directory and persisted on
`CritiqueResult.sampled_frame_uris`.

Future migration trigger: move frame extraction into a dedicated VLM wrapper service when critique
needs GPU-side batching/caching, scene-aware sampling, multiple VLM backends sharing preprocessing,
or cleaner separation for Phase 3 router/self-healing.

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
- `test_workflow.py` — 28 (DAG orchestration, settings wiring, rights blocking, critique loop)
- `test_critique.py` — 26 (VLM critique adapter, frame sampling, preflight, parsing)
- `test_plan_shots.py` — 29
- `test_assemble_video.py` — 32
- `test_publish.py` — 26
- `test_adapt_script.py` — ~25
- `test_synthesize_voice.py` — 27
- `test_render_character.py` — 29
- `test_lip_sync.py` — 20
- `test_generate_video.py` — 18
- `test_transcribe.py`, `test_analyze_content.py`, `test_fetch_media.py` — ~10–15 each
- `test_runtime_readiness.py` — 7
- `test_setup_gpu.py` — 4
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

Phase 2 critic path:
```python
import asyncio
from core.config import load_app_config
from core.workflow import run_with_critique

config = load_app_config()
job = asyncio.run(run_with_critique(
    source_url="https://www.youtube.com/watch?v=EXAMPLE",
    rights_cleared=True,
    app_config=config,
))
print(job.status)
```

Environment overrides (via `.env` or shell):
```bash
VIDEO_ME_DATA_DIR=/data/video_me       # where job work dirs are created
VIDEO_ME_REVIEW_DIR=/data/review       # where publish output goes
VIDEO_ME_LORA_DIR=/models/loras        # where LoRA files are
VIDEO_ME_VOICE_DIR=/data/voices        # where reference WAV files are
VIDEO_ME_LLM_MODEL=qwen3:14b
VIDEO_ME_LLM_BASE_URL=http://localhost:11434/v1
VIDEO_ME_CRITIQUE_MODEL=llava:7b
VIDEO_ME_CRITIQUE_BASE_URL=http://localhost:11434/v1
VIDEO_ME_SD_BASE_URL=http://localhost:7860
VIDEO_ME_TTS_BASE_URL=http://localhost:8020
VIDEO_ME_WAN_BASE_URL=http://localhost:8030
VIDEO_ME_LIPSYNC_BASE_URL=http://localhost:8040
VIDEO_ME_WHISPER_DEVICE=cpu            # use cuda on GPU
VIDEO_ME_WHISPER_COMPUTE_TYPE=int8     # use float16 on CUDA
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
| 3 | Final Max/Zoe reference sheets approved | Track B LoRA training | `kids_duo` config selected |
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
