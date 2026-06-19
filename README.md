# video_me

Orchestration pipeline that turns a reference video URL into an original animated kids'
educational short starring a swappable 4-member pig cast. Every model is an
interchangeable adapter behind a typed capability ABC.

## Status

**Phase 1 code-complete — 264 tests passing.**
Pipeline is blocked on Track B (LoRA + voice files) and Track D (GPU services).
See `BUILD_PROGRESS.md` for the full implementation journal and next steps.

```
Phase 0  Skeleton + storage          ✅ COMPLETE
Phase 1  Full pipeline A1.0–A1.12   ✅ COMPLETE (code) — blocked on Track B + D
Track B  LoRAs + voice files         ❌ Files missing — see setup guide below
Track D  GPU services                ❌ Not provisioned — budget decision pending
Phase 2  Critic loop                 ⏳ NOT STARTED
```

## Quick start (tests only — no services needed)

```bash
git clone https://github.com/rekha0708/video_me
cd video_me
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q      # 264 tests, all passing
```

## Running the Phase 0 no-op workflow

```bash
python -m scripts.run_noop_job
# Writes artifacts under .local/artifacts/ and records job in .local/video_me.db
```

With local PostgreSQL + MinIO:
```bash
docker compose up -d
VIDEO_ME_JOB_STORE=postgres VIDEO_ME_ARTIFACT_STORE=s3 python -m scripts.run_noop_job
```

## Running the Phase 1 pipeline (requires Track B + D)

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
print(job.status)   # "completed"
# Output: review/<timestamp>_<stem>/video.mp4 + metadata.json
```

## Track B — Files required before pipeline runs

Place LoRA weights and reference voice WAVs at these exact paths:

```
loras/
  pig_kids_placeholder_c1.safetensors   (Pippa)
  pig_kids_placeholder_c2.safetensors   (Milo)
  pig_kids_placeholder_c3.safetensors   (Nia)
  pig_kids_placeholder_c4.safetensors   (Luma)

voices/
  pig_kids_placeholder/
    c1.wav   (Pippa — ~10–30s reference speech)
    c2.wav   (Milo)
    c3.wav   (Nia)
    c4.wav   (Luma)
```

See `.claude/agents/track-b-setup.md` for the full setup guide.

## Track D — Services required before pipeline runs

| Service | Port | Purpose |
|---|---|---|
| Ollama | 11434 | LLM (analyze, adapt, plan stages) |
| AUTOMATIC1111 | 7860 | Stable Diffusion (render_character) |
| Chatterbox TTS | 8020 | TTS (synthesize_voice) |
| Wan 2.7 | 8030 | Image-to-video (generate_video) |
| Wav2Lip | 8040 | Lip sync (lip_sync) |

See `.claude/agents/pipeline-runner.md` for startup commands and pre-flight check.

## Architecture

```
URL → fetch_media → transcribe → analyze_content → [check_rights gate]
    → adapt_script → plan_shots
    → per shot: render_character + synthesize_voice + generate_video + lip_sync
    → assemble_video → publish → review/
```

Every stage is a `Capability[Request, Result]` ABC in `core/capabilities/`.
Concrete adapters live in `adapters/<stage>/`.
The full DAG is in `core/workflow.py:run_pipeline_job()`.

## Configuration

Channel and cast config live in `config/`:
- `config/channels/education_kids.yaml` — 9:16, age 3–6, `made_for_kids: true`
- `config/casts/pig_kids_placeholder.yaml` — 4-member pig cast

Environment variables (via `.env` or shell):
```bash
VIDEO_ME_DATA_DIR=/data/video_me
VIDEO_ME_REVIEW_DIR=/data/review
VIDEO_ME_LORA_DIR=/models/loras
VIDEO_ME_VOICE_DIR=/data/voices
VIDEO_ME_JOB_STORE=postgres           # default: sqlite
VIDEO_ME_ARTIFACT_STORE=s3            # default: local
```

## Local services (Docker)

```bash
docker compose up -d
```

Starts local PostgreSQL (`localhost:5432`) and MinIO (`localhost:9000`).
MinIO console: `localhost:9001` — credentials: `video_me` / `video_me_dev_password`.

## Non-negotiable guardrails

1. **Original characters only** — `is_original_synthetic=True` enforced in Pydantic
2. **Transformative sourcing** — `rights_cleared=True` required before adapt_script
3. **Children's safety** — human approval gate; publish writes to review folder only
4. **Made-for-kids + COPPA** — `made_for_kids=True` in channel profile; no child-level data
5. **AI disclosure** — label burned onto video via ffmpeg drawtext
6. **Phase gating** — do not advance past a phase until acceptance criteria pass

## Claude Code context

This project has a `CLAUDE.md` at the root that gives Claude Code full project context
every session. Sub-agents in `.claude/agents/` handle specific tasks:

```
/project:project-status   — current state report
/project:test-runner      — run and debug tests
/project:track-b-setup    — LoRA + voice file setup guide
/project:pipeline-runner  — end-to-end run guide
```
