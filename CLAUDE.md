# CLAUDE.md — video_me Project Context

## What this project is

`video_me` is an orchestration pipeline that turns a reference video URL into an original animated
kids' educational short. The default cast is `kids_duo` (Max and Zoe), but the pipeline is
**cast-agnostic** — each job selects its cast from a dropdown, and a different cast YAML
(even 1-character) works seamlessly. Every model is an interchangeable adapter behind a typed
capability ABC. The pipeline is guardrail-enforced — jobs with uncleared rights or unoriginal
content are blocked, not silently passed.

---

## Current state (as of 2026-07-10)

**Stack: Flux 2.0 Dev image via musubi-tuner (local subprocess) + Wan2.2-S2V video (audio-conditioned native lip-sync) + Fish Audio S2 (TTS). Plan critique loop + human approval gates + dashboard UI are in place.**

**Default adapter stack is code-enforced in `core/config.py`: `musubi_flux` (image) / `wan_s2v` (video) / `fish_s2` (TTS).** The image stage runs **musubi-tuner** as a subprocess — ComfyUI cannot load Flux 2.0 locally (no Mistral 3 encoder node; the `Flux2*` ComfyUI nodes are paid BFL cloud API), so `comfyui_flux` is a fallback. Wan S2V runs behind a thin local HTTP wrapper on port 8031 and receives the per-shot image + audio directly; the separate lip-sync repair stage is skipped on this default path.

**Test status:** full local suite verified 2026-07-11: **704 passed / 33 skipped**. Exact current count: `python -m pytest --collect-only -q`.

- **LLM**: qwen3.6:35b (MoE 35B). Thinking mode disabled via `extra_body={"think": False}` + no `response_format`. `max_tokens=16384`. `json_repair` fallback. Used for all LLM stages including plan critique.
- **Image generation**: Flux 2.0 Dev (32B, Nov 2025) + Flux LoRA, run **locally via musubi-tuner** (replaces A1111 + SD 1.5). Default adapter: `MusubiFluxAdapter` (subprocess, no server). `ComfyUIFluxAdapter` (port 8188) is a fallback but ComfyUI can't load Flux 2.0 locally — it needs the paid BFL cloud API / a custom Mistral 3 node.
- **Video generation**: Wan2.2-S2V 14B via `WanS2VAdapter` + `services/wan_s2v_server.py` (port 8031). It receives the approved still + the exact shot audio and derives `infer_frames` from shot duration using `VIDEO_ME_WAN_S2V_FPS` (default 16) with Wan-style `4n+1` frame counts. The service invokes Wan's `generate.py` as a subprocess per clip with `--offload_model True`; there is no long-resident S2V model endpoint.
- **Plan critique loop**: after `plan_shots`, `LlmPlanCritiqueAdapter` scores 5 dimensions (character_fit, scene_achievability, pacing, kids_safety, visual_clarity). All must be ≥ 0.75 to pass. Up to 3 re-plan iterations with specific fix notes injected.
- **Human approval gate (storyboard)**: after critique passes, web UI at `http://localhost:8765` shows shot table + score bars. Approve → render. Reject + notes → one more re-plan cycle. 2nd rejection → job FAILED. CI bypass: `VIDEO_ME_AUTO_APPROVE_PLAN=true`.
- **Image candidate generation**: render_character generates N images per shot (default 1 — operator decision 2026-07-07: Flux candidates are near-identical, so extra candidates waste GPU; raise via `VIDEO_ME_IMAGE_CANDIDATES` if variety is needed). `Shot.action` is included in the render prompt so each still shows the shot's pose/angle (LTX still animates the motion). Phase A batches all pending shots into one `run_many()` call per LoRA (one 64 GB model load instead of one per candidate) and dedups identically specified shots (same member/setting/camera/action → PNGs copied, not re-rendered). With a single candidate the VLM critique is skipped (auto-pick, `origin="single"`) — the human image gate stays the quality check. `VlmImageCritiqueAdapter` (qwen3.6:35b, natively multimodal) scores all candidates on 5 dimensions and picks the best. Self-learning: each pick + human override is appended to `assets/kids_duo/critique_feedback.jsonl`; last 5 entries are injected as few-shot context on the next run.
- **Human approval gate (images)**: after all shots are rendered and critiqued, web UI at `http://localhost:8765 (shared port)` shows a grid of winner images. Operator can override any pick, then clicks Approve. Overrides are written back to the feedback log. CI bypass: `VIDEO_ME_AUTO_APPROVE_IMAGES=true`.
- **Single VLM for everything**: qwen3.6:35b handles text LLM + image critique + video frame critique. Drops qwen2.5-vl:32b entirely. The workflow unloads Ollama before GPU-heavy render/video/voice phases that opt into managed VRAM.
- **TTS**: Fish Audio S2 (`FishS2TtsAdapter`, port 8025). Supports English and Hindi (80+ languages, voice cloning from reference WAV). Replaces Chatterbox TTS. Fallback: `VIDEO_ME_TTS_ADAPTER=chatterbox`.
- **Language selection**: `VIDEO_ME_TARGET_LANGUAGE=en|hi|both`. "both" runs the full pipeline twice (shared images, separate dialogue/audio). Script dialogue is translated by the LLM when language ≠ "en".
- **Whisper transcription**: faster-whisper defaults to `large-v3` on CUDA/float16 for better source-audio and lyric capture. `setup_gpu.sh` first checks the local `/workspace/.cache/huggingface/hub` cache and downloads only missing models; pass `--whisper-model large-v2` or `--prefetch-whisper-models large-v3,large-v2` for A/B testing. Generated GPU env uses `VIDEO_ME_WHISPER_LOCAL_FILES_ONLY=true` so runtime jobs never fetch surprise model snapshots. Pin a specific HF snapshot with `--whisper-model-revision <commit-or-tag>` / `VIDEO_ME_WHISPER_MODEL_REVISION`. The workflow unloads Whisper immediately after the transcribe stage so later Flux/Wan/Fish stages do not inherit its VRAM.
- **Source-audio chunking**: when a real source audio track is present, the workflow uses faster-whisper word timestamps and deterministic sentence/lyric boundary splitting to make many small shots instead of forcing every shot toward 8 seconds. No LLM is used for the boundary choice. `VIDEO_ME_WHISPER_VAD_FILTER=false` is the default so sung lyrics are not clipped by over-aggressive VAD; `VIDEO_ME_TRANSCRIPT_MIN_COVERAGE_RATIO` fails only catastrophic short transcripts.
- **Shot duration**: max planned shot duration defaults to 8s (`VIDEO_ME_MAX_SHOT_DURATION_SEC`), but source-audio jobs may split more finely at sentence/lyric boundaries.
- **AV/lip-sync QA**: non-native video paths (`VIDEO_ME_VIDEO_ADAPTER=wan`) keep raw video, every LatentSync/MuseTalk attempt, retry metadata, duration deltas, and selected/fallback reason in the dashboard's shot video card. Default failure policy is `fallback_raw`; set `VIDEO_ME_LIPSYNC_FAILURE_POLICY=fail` or `VIDEO_ME_AV_SYNC_FAILURE_POLICY=fail` to hard-fail.
- **Resume**: `--resume-job JOB_ID` skips completed stages/shots. Default Wan S2V completion marker: `clip.mp4`; Wan I2V + repair marker: `synced.mp4`.
- **Fallback adapters**: `VIDEO_ME_RENDER_ADAPTER=a1111` → A1111 + SD 1.5. `VIDEO_ME_VIDEO_ADAPTER=wan` → Wan 2.2 I2V + LatentSync by default (`VIDEO_ME_LIPSYNC_ADAPTER=latentsync`; MuseTalk remains fallback). `VIDEO_ME_VIDEO_ADAPTER=ltx` → legacy LTX via ComfyUI. `VIDEO_ME_TTS_ADAPTER=chatterbox` → Chatterbox TTS.
- **Track B LoRAs**: existing SD 1.5 weights won't work with Flux 2.0 — retrain with `flux_train_network.py` (kohya_ss config already updated).
- **Per-job cast selection**: each job picks its cast from a dropdown in `/jobs/new`. `GET /api/casts` scans `config/casts/*.yaml`. Worker loads the selected cast via `_config_for_job()`. Default cast: `kids_duo` (env: `VIDEO_ME_CAST_PATH`). `Cast.members` enforces min_length=1 — 0 members gives a clear `ValidationError`. Adapt-script scene guide adjusts for 1/2/3+ member casts.
- **Dashboard UI**: web UI at `http://localhost:8080` (uvicorn). Job list, detail, health, chat. Dedicated `/jobs/new` page with cast selector + 4 input modes (Video URL / Local file / Story / Story + Images). Per-job "Approval gates" checkboxes set `overrides.auto_approve_plan` / `auto_approve_images` / `auto_approve_transcript` so long unattended runs skip the human gates (dashboard approval adapters and the transcript review gate short-circuit and record an `approval_granted` event). Source kinds: `url`, `upload`, `file`, `story`, `story_images`. Story-kind jobs restricted to `transcribe` or `all` phases. Character image upload via `POST /api/uploads/character-image`. Optional `gpu_price_per_hour` on `/jobs/new` drives a detail-page cost summary derived from paired stage start/end events; approval waits and other idle gaps are not billed. Dashboard-created/job/event timestamps display in Pacific time (`America/Los_Angeles`) so server-rendered rows and live SSE rows match.
- **Story ingest**: `adapters/story_ingest/` — structured parser (`start-end: text`) + LLM segmenter fallback. `_seed_story_job` in worker creates fake TranscribeResult from story text. Story+images mode skips Phase A render; user images go through approval with `origin="user"` label.
- **GPU/model lifecycle**: `core/gpu_sequencer.py` coordinates only adapters that declare `managed_vram=True`. Wan I2V uses deferred `/load`/`/unload`; Fish S2 is loaded on demand and the dashboard worker kills the process after every job because its allocator retained VRAM across calls. Wan S2V and LatentSync are subprocess-per-request wrappers, so they release their model process after each clip/repair instead of using resident `/load` endpoints. The workflow unloads Fish before Wan S2V or lip-sync repair when the adapter marks `requires_voice_unloaded=True`.

| Track / Phase | Status | Blocker |
|---|---|---|
| Phase 0 — Skeleton | ✅ COMPLETE | — |
| Phase 1 — Full pipeline A1.0–A1.12 | ✅ COMPLETE (code) | — |
| Phase 2 — Critic loop A2.x | ✅ COMPLETE (code) | Real VLM service needed for real judgment |
| Plan critique + approval gate | ✅ COMPLETE (code) | — |
| Image candidate critique + approval | ✅ COMPLETE (code) | — |
| Track B — LoRAs + voice files | ❌ INCOMPLETE | `loras/kids_duo_max.safetensors` missing; `kids_duo_zoe.safetensors` is a TEST-ONLY placeholder. Voice WAVs present (bootstrap). Run `python -m scripts.check_track_b`. |
| Dashboard UI + story ingest | ✅ COMPLETE (code) | — |
| GPU sequencer (Wan VRAM) | ✅ COMPLETE (code) | — |
| Fish Audio S2 TTS (EN + HI) | ✅ COMPLETE (code) | Fish S2 server setup needed |
| Track D — GPU services | ⚠️ Manual start required | Run `scripts/setup_gpu.sh`, then `scripts/start_services.sh`; defaults require Ollama, Wan S2V, Fish S2 |
| Track E — Compliance sign-off | ❌ PENDING | Operator hasn't signed off |

Voice reference files are gTTS bootstrap WAVs — acceptable for pipeline runs; replace with recorded
child voices for brand-accurate results.

**After every pod restart, run:**
```bash
bash scripts/start_services.sh
```
This script auto-reinstalls Ollama (base Linux binary is wiped on restart), then starts services and verifies each health endpoint.

---

## Architecture

```
source URL                              story text (+ optional images)
    │                                       │
    ▼                                       ▼
[fetch_media]        yt-dlp download   [story_ingest]    structured/LLM parser
    │                + ffmpeg extract        │              → fake TranscribeResult
    ▼                                       │
[transcribe]         faster-whisper         │
    │                → TranscribeResult      │
    └──────────────┬────────────────────────┘
    │
    ▼
[analyze_content]    LLM → ContentMetadata + LearningObjective
    │
    ▼
[analyze_visuals]    VLM samples source-video frames per segment → VisualContext
    │                (settings/props); best-effort, empty for story jobs
    ▼
check_rights()  ◄─── BLOCKS job (status=BLOCKED) if rights_cleared=False
    │
    ▼
[adapt_script]       LLM → Script (scenes + lines, mode=transformed);
    │                scene settings grounded in VisualContext when present
    │
    ▼
[plan_shots]         LLM → Storyboard (Shot list, ≤2 chars/shot)
    │
    ▼
[critique_plan]      LLM loop (≤3×) → scores 5 dimensions, re-plans with fix notes if <0.75
    │
    ▼
[approval gate]      Web UI localhost:8765 → human approves/rejects; 2nd rejection = FAILED
    │
    ▼ (per shot — Phase A)
    ├── [render_character ×N]  musubi-tuner Flux 2.0 + LoRA → N candidate PNGs (default N=1, batched per LoRA)
    └── [critique_images]      qwen3.6:35b → picks best; logs to critique_feedback.jsonl (skipped when N=1)
    │
    ▼
[approve_images]     Web UI localhost:8765 (shared port) → image grid; operator confirms/overrides per shot
                     Overrides written back to feedback log (self-learning)
    │
    ▼ (per shot — Phase B, uses approved image)
    ├── [synthesize_voice]   Fish Audio S2 API → AudioTrack (WAV, EN or HI)
    ├── [generate_video]     Wan2.2-S2V → VideoClip (MP4, native audio-conditioned mouth motion) [default]
    └── [lip_sync]           LatentSync/MuseTalk repair — SKIPPED when VIDEO_ADAPTER=wan_s2v
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

## Where to find detail (read only what you need)

**Generated code map** — `docs/code_map/` is auto-generated from the AST by
`python -m scripts.generate_code_map` and kept in sync by `tests/test_code_map.py`
(the suite fails if the map is stale). Trust it over any prose file list.

- `docs/code_map/INDEX.md` — every module, one line each (incl. tests)
- `docs/code_map/adapters.md` — capability → adapter mapping + every adapter class/method signature
- `docs/code_map/core.md` — workflow, executor, storage, models, config APIs
- `docs/code_map/services.md` — dashboard API/worker/repository + GPU service servers
- `docs/code_map/scripts.md` — setup/check/utility scripts
- `docs/code_map/api.md` — all dashboard HTTP routes (method, path, handler, purpose)
- `docs/code_map/models.md` — every Pydantic model with fields and defaults
- `docs/code_map/env.md` — every `VIDEO_ME_*` env var with type and default
- `docs/code_map/dependencies.md` — module import graph
- `docs/code_map/LIMITATIONS.md` — **curated** per-stage weak points, coupling, fragility

**Key entry points** (stable; everything else is in the map):

| Path | Purpose |
|---|---|
| `core/workflow.py` | `run_pipeline_job()` — Phase 1 DAG; `run_with_critique()` — Phase 2 loop |
| `core/executor.py` | `run_stage()` health-check→invoke→persist; `check_rights()` gate |
| `core/capabilities/base.py` | ALL stage ABCs (single file, not per-stage) |
| `core/config.py` | `Settings` (env, prefix `VIDEO_ME_`) + `load_app_config()` |
| `services/dashboard_api.py` | Dashboard FastAPI app (port 8080); worker: `services/dashboard_worker.py` |
| `config/casts/kids_duo.yaml` + `config/casts/<cast>/params.py` | Cast definition + per-cast LoRA/voice/render params |
| `config/channels/education_kids.yaml` | Channel: 9:16, age 3-6, made_for_kids=true |
| `loras/`, `voices/` | Track B weights + reference WAVs — **MUST EXIST** before rendering/TTS |
| `review/` | Output: `<timestamp>_<stem>/video.mp4` + `metadata.json` sidecar |
| `docs/PIPELINE_STAGES_AND_VRAM.md` | per-stage model/service/VRAM reference |
| `docs/BUILD_PROGRESS.md` | Full implementation journal + decision log |

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
Current status: `Track B: INCOMPLETE`. `loras/kids_duo_zoe.safetensors` is a TEST-ONLY
placeholder and `loras/kids_duo_max.safetensors` is missing — train both Flux LoRAs.
Voice reference WAVs (`voices/kids_duo/{max,zoe}.wav`) are present (gTTS bootstrap).

Temporary placeholder-LoRA render smoke tests are opt-in:
```bash
export VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true
```
When this is true, explicit `TEST-ONLY placeholder` LoRA files are accepted and omitted from
the SD prompt. Keep it false for real runs; strict readiness fails placeholder LoRAs.

---

## Venv strategy (as of 2026-07-10)

Each GPU service uses an **isolated venv that inherits system torch 2.8.0+cu128** via
`python3 -m venv --system-site-packages`. This avoids cross-service dependency conflicts.

| Venv | Purpose | Key extra packages |
|---|---|---|
| `/workspace/video_me/.venv` | Pipeline orchestration + tests (no heavy ML) | httpx, faster-whisper, pydantic-settings |
| `/workspace/.venv_musubi` | musubi-tuner Flux 2.0 image subprocess | musubi-tuner + Flux deps |
| `/workspace/.venv_fish_s2` | Fish Audio S2 server (port 8025) | Fish Speech deps, fastapi, uvicorn |
| `/workspace/.venv_wan` | Wan2.2 S2V server (port 8031) and optional Wan I2V server (port 8030) | decord, diffusers, transformers, accelerate, peft, librosa, dashscope, rotary-embedding-torch, python-multipart |
| `/workspace/.venv_latentsync` | LatentSync lip-sync repair server (port 8041, opt-in fallback) | LatentSync requirements, fastapi, uvicorn |
| `/workspace/.venv_chatterbox` | Chatterbox TTS server (port 8020, opt-in fallback) | chatterbox-tts, torchaudio==2.8.0+cu128, resemble-perth |
| `/workspace/.venv_musetalk` | MuseTalk lip-sync server (port 8040) | opencv, librosa, einops, diffusers, mmengine, mmpose==1.3.2, mmcv==2.1.0 (built from source), face-alignment |
| `/workspace/venv` | sd-scripts LoRA training | sd-scripts deps |
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
Models at `/workspace/ollama/` persist on the network volume. Default model: **qwen3.6:35b**
(single model for LLM + VLM critique). Rollback: `VIDEO_ME_LLM_MODEL=qwen3:14b`.

`start_services.sh` uses the correct interpreter for each service. Never install
heavy ML packages into the project `.venv` — keep it lightweight for fast CI.

---

## Track D — Services required before pipeline runs

Required services are adapter-dependent. The executor calls `capability.health()` before each
stage, and `scripts/check_runtime_readiness.py` builds its service list from `core.config.Settings`.
`scripts/start_services.sh` also reads `.env` and starts/waits only for the selected adapter stack
(plus Ollama), so fallback services do not load just because their venv happens to exist.

| Service | Default URL | Purpose | Required? |
|---|---|---|---|
| Ollama | `http://localhost:11434` | LLM (analyze, adapt, plan, critique_plan) + VLM critique | ✅ Always |
| musubi-tuner | (subprocess, no port) | Flux 2.0 Dev image gen (render_character) | ✅ Default |
| Wan2.2 S2V | `http://localhost:8031` | Audio-conditioned video generation with native mouth motion | ✅ Default (`VIDEO_ME_VIDEO_ADAPTER=wan_s2v`) |
| Fish Audio S2 | `http://localhost:8025` | TTS (EN + HI) for synthesize_voice | ✅ Default |
| ComfyUI | `http://localhost:8188` | legacy LTX video gen / ComfyUI Flux fallback | ⚠️ `VIDEO_ME_VIDEO_ADAPTER=ltx` or `VIDEO_ME_RENDER_ADAPTER=comfyui_flux` |
| Chatterbox TTS | `http://localhost:8020` | TTS (EN only, fallback) | ⚠️ `VIDEO_ME_TTS_ADAPTER=chatterbox` only |
| AUTOMATIC1111 | `http://localhost:7860` | SD 1.5 render_character fallback | ⚠️ `VIDEO_ME_RENDER_ADAPTER=a1111` only |
| Wan 2.2 I2V | `http://localhost:8030` | Image-to-video fallback | ⚠️ `VIDEO_ME_VIDEO_ADAPTER=wan` only |
| LatentSync | `http://localhost:8041` | preferred lip-sync repair for Wan I2V | ⚠️ `VIDEO_ME_VIDEO_ADAPTER=wan` + `VIDEO_ME_LIPSYNC_ADAPTER=latentsync` |
| MuseTalk | `http://localhost:8040` | legacy lip-sync repair fallback | ⚠️ `VIDEO_ME_VIDEO_ADAPTER=wan` + `VIDEO_ME_LIPSYNC_ADAPTER=musetalk` |

Quick health check:
```bash
python -m scripts.check_runtime_readiness
```

GPU-machine setup helper:
```bash
bash scripts/setup_gpu.sh
```

Default `setup_gpu.sh` installs the default stack: musubi-tuner, Fish S2, Wan2.2 S2V, Ollama,
and the lightweight project venv. Wan I2V, LatentSync, MuseTalk, Chatterbox, and A1111 are opt-in
via `--with-wan-i2v`, `--with-latentsync`, `--with-musetalk`, `--with-chatterbox`, and
`--with-a1111` (or the back-compat `--with-wan` bundle).

Lifecycle policy:
- Managed resident adapters load only at their phase boundary: Wan I2V (`/load`/`/unload`) and Fish S2.
- After every dashboard job outcome, `services/dashboard_worker.py` kills Fish S2 and calls Wan I2V `/unload`.
- Before a Wan S2V or LatentSync/MuseTalk call, the workflow unloads Fish when the selected adapter declares `requires_voice_unloaded=True`.
- Wan S2V and LatentSync do not use resident `/load` endpoints; their wrappers spawn one subprocess per clip/repair, and that process exit is the release point.

Local/mock placeholder check without services:
```bash
bash scripts/setup_gpu.sh --code-test --skip-services
```

LLM+VLM model: `qwen3.6:35b` (MoE 35B, ~30 GB VRAM, natively multimodal) — used for ALL stages including image and video critique. No separate VLM model needed. Rollback: `VIDEO_ME_LLM_MODEL=qwen3:14b`. Phase 2 samples local video
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

For the per-file breakdown, run `python -m pytest --collect-only -q` or see the test entries in
`docs/code_map/INDEX.md` — do not maintain hardcoded per-file counts here (they rot).

`tests/test_code_map.py` guards `docs/code_map/` freshness: if it fails, run
`python -m scripts.generate_code_map` and commit the regenerated map.

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
VIDEO_ME_LLM_MODEL=qwen3.6:35b         # also VLM critique; rollback: qwen3:14b
VIDEO_ME_LLM_BASE_URL=http://localhost:11434/v1
VIDEO_ME_CRITIQUE_MODEL=qwen3.6:35b    # single multimodal model for all critique
VIDEO_ME_CRITIQUE_BASE_URL=http://localhost:11434/v1
# Per-job cast selection (process-level default; overridden per job):
VIDEO_ME_CAST_PATH=config/casts/kids_duo.yaml
VIDEO_ME_CHANNEL_PATH=config/channels/education_kids.yaml
# Default stack (musubi image + ComfyUI/LTX video + Fish S2 TTS):
VIDEO_ME_COMFYUI_BASE_URL=http://localhost:8188   # ComfyUI (LTX video; also comfyui_flux fallback)
VIDEO_ME_FISH_S2_BASE_URL=http://localhost:8025
# Legacy fallback URLs (only when the matching *_ADAPTER override is set):
# VIDEO_ME_SD_BASE_URL=http://localhost:7860       # a1111
# VIDEO_ME_TTS_BASE_URL=http://localhost:8020      # chatterbox
# VIDEO_ME_WAN_BASE_URL=http://localhost:8030      # wan
# VIDEO_ME_LIPSYNC_BASE_URL=http://localhost:8040  # musetalk
VIDEO_ME_WHISPER_DEVICE=cuda
VIDEO_ME_WHISPER_COMPUTE_TYPE=float16
VIDEO_ME_WHISPER_MODEL_SIZE=large-v3   # large-v2 also supported
VIDEO_ME_WHISPER_DOWNLOAD_ROOT=/workspace/.cache/huggingface/hub
VIDEO_ME_WHISPER_LOCAL_FILES_ONLY=true
# VIDEO_ME_WHISPER_MODEL_REVISION=<hf commit sha or tag>
VIDEO_ME_JOB_STORE=postgres            # use PostgreSQL instead of SQLite
VIDEO_ME_ARTIFACT_STORE=s3             # use MinIO/S3 instead of local filesystem
```

### Running the dashboard

```bash
bash scripts/restart_dashboard.sh
```

Starts (or restarts) both the API and the worker. Neither auto-reloads —
`--reload` was removed after it repeatedly hung at "Waiting for connections
to close" on this pod whenever a browser tab held an open SSE stream
(`/api/jobs/*/stream`), silently freezing the whole dashboard (Cancel/Approve
buttons stopped working, running jobs looked stuck). Rerun this script after
any change under `core/`, `adapters/`, or `services/dashboard_*.py`.

Navigate to `http://localhost:8080`. Key pages:

- `/` — job list with source-kind badges
- `/jobs/new` — create job (4 input modes: Video URL / Local file / Story / Story + Images)
- `/jobs/{id}` — job detail with stepper, artifacts, phase controls
- `/health` — service health checks
- `/api/docs` — OpenAPI docs

API endpoints for story ingest:

- `POST /api/jobs` — create job; `source.kind` = `url|upload|file|story|story_images`
- `POST /api/uploads/character-image` — multipart upload for story+images mode
- `GET /api/local-images?dir=` — list image files in a local directory

Story-kind jobs are restricted to `transcribe` or `all` phases. Later phases require upstream
artifacts — use the Advance button on an existing job to continue from a later phase.

---

## Adding a new adapter (pattern reference)

1. Create `adapters/<stage>/<name>_adapter.py`
2. Subclass the ABC from `core/capabilities/base.py` (all stage ABCs live in that one file)
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
6. **Phase gating** — do not advance past a phase until its acceptance criteria pass (see `docs/orchestration-build-plan.md §9`)

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
