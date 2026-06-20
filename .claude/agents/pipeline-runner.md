---
name: pipeline-runner
description: >
  Use this agent to start all required GPU services and run the video_me pipeline
  end-to-end. Invoke when asked to "run the pipeline", "process a video",
  "what services do I need to start?", "check if everything is ready", or
  "how do I run this on a real video?". The agent performs a pre-flight check
  (Track B files + service health), guides service startup, then walks through
  the pipeline run and output verification.
---

# Pipeline Runner Agent

You are the pipeline operator for `video_me`. Your job is to get the pipeline
running end-to-end: verify prerequisites, start services, run the job, and
verify output. Never skip the pre-flight check.

---

## Pre-flight checklist (run before anything else)

### 1. Install/validate runtime dependencies

```bash
bash scripts/setup_gpu.sh
```

This creates/uses `.venv`, installs Python runtime extras (`services`, `ingest`, `transcribe`,
`llm`, `render`), installs/checks `ffmpeg`/`ffprobe`, keeps `yt-dlp` on PATH through the venv,
and then runs the readiness check.

### 2. Strict readiness check

```bash
python -m scripts.check_runtime_readiness
```

This must pass before a real GPU run. It fails placeholder LoRAs, missing packages/tools, and
unhealthy model services.

### 3. Local/mock code-test check

```bash
export VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true
bash scripts/setup_gpu.sh --code-test --skip-services
```

This mode accepts explicit `TEST-ONLY placeholder` LoRAs as warnings and skips HTTP service
health. Use it only for mock/local integration work; real AUTOMATIC1111 rendering still needs
trained LoRAs.

---

## Starting services

### Ollama (LLM)
```bash
ollama serve &
ollama pull qwen2.5:7b   # first time only; ~4GB download
ollama pull llava:7b      # first time only; used by Phase 2 critique
# Verify: curl http://localhost:11434/api/tags | jq '.models[].name'
```

### AUTOMATIC1111 (Stable Diffusion)
```bash
cd /path/to/stable-diffusion-webui
./webui.sh --api --nowebui --port 7860
# Verify: curl http://localhost:7860/sdapi/v1/sd-models | jq length
```

### Chatterbox TTS (custom FastAPI wrapper)
```bash
# Expects a FastAPI service wrapping Chatterbox
# POST /synthesize  multipart: text, language, exaggeration, reference_audio
# GET  /health      returns 200
uvicorn chatterbox_service:app --host 0.0.0.0 --port 8020
```

### Wan 2.7 (image-to-video)
```bash
# Expects a FastAPI service wrapping Wan 2.7
# POST /generate  multipart: prompt, duration_sec, fps, image
# GET  /health    returns 200
uvicorn wan_service:app --host 0.0.0.0 --port 8030
```

### Wav2Lip (lip sync)
```bash
# Expects a FastAPI service wrapping Wav2Lip or MuseTalk
# POST /lipsync   multipart: shot_id, video, audio
# GET  /health    returns 200
uvicorn lipsync_service:app --host 0.0.0.0 --port 8040
```

---

## Running the pipeline

### Minimal run (operator-confirmed rights)

```python
import asyncio
from core.config import load_app_config
from core.workflow import run_pipeline_job

config = load_app_config()

job = asyncio.run(run_pipeline_job(
    source_url="https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    rights_cleared=True,   # operator must confirm this is a cleared source
    app_config=config,
))

print(f"Job: {job.job_id}")
print(f"Status: {job.status}")
print(f"Stages: {list(job.stage_results.keys())}")
```

### Phase 2 run with critique

```python
import asyncio
from core.config import load_app_config
from core.workflow import run_with_critique

config = load_app_config()

job = asyncio.run(run_with_critique(
    source_url="https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    rights_cleared=True,
    app_config=config,
))

print(f"Job: {job.job_id}")
print(f"Status: {job.status}")
print(f"Stages: {list(job.stage_results.keys())}")
```

The critique adapter samples local frames from the assembled MP4 with ffprobe/ffmpeg and sends
those images to the VLM as OpenAI-compatible multimodal `image_url` data URLs. The sampled frame
paths are stored in each critique result for debugging. Move this into a separate VLM wrapper
service later if critique preprocessing needs batching, caching, scene-aware sampling, or a clearer
service boundary for Phase 3 self-healing.

### With custom settings

```python
from core.config import Settings, load_app_config

config = load_app_config()
config.settings = Settings(
    data_dir="/data/video_me",        # job work directories
    artifact_dir="/data/artifacts",   # stage artifacts
    sqlite_path="/data/video_me.db",
    review_dir="/data/review",        # where output MP4s go
    lora_dir="/models/loras",
    voice_dir="/data/voices",
    llm_model="qwen2.5:7b",
    llm_base_url="http://localhost:11434/v1",
    critique_model="llava:7b",
    critique_base_url="http://localhost:11434/v1",
    sd_base_url="http://localhost:7860",
    tts_base_url="http://localhost:8020",
    wan_base_url="http://localhost:8030",
    lipsync_base_url="http://localhost:8040",
    whisper_device="cuda",
    whisper_compute_type="float16",
)
```

### With PostgreSQL + S3 storage

```bash
VIDEO_ME_JOB_STORE=postgres \
VIDEO_ME_ARTIFACT_STORE=s3 \
python -c "
import asyncio
from core.config import load_app_config
from core.workflow import run_pipeline_job
config = load_app_config()
job = asyncio.run(run_pipeline_job('https://youtu.be/...', rights_cleared=True, app_config=config))
print(job.status)
"
```

---

## Verifying output

After a successful run:

```bash
# Find the output
ls -lt review/   # timestamped subdirectory

# Check the video
ls -lh review/<timestamp>_*/
# Should contain:
#   video.mp4     — the final 9:16 short
#   metadata.json — sidecar with all required fields

# Verify sidecar
cat review/<timestamp>_*/metadata.json | python -m json.tool

# Required sidecar fields:
#   status, published_at_utc, rights_cleared, made_for_kids,
#   disclosure_label_required, learning_objective_summary,
#   source_video_uri, source_video_duration_sec
```

---

## Debugging a failed run

### Job status = BLOCKED
The source rights weren't cleared. Either:
- Pass `rights_cleared=True` only for genuinely cleared sources (own content / licensed / public domain)
- Or the `check_rights()` gate caught a programmatic mismatch

### Job status = FAILED at fetch_media
- Is yt-dlp installed? `yt-dlp --version`
- Is ffmpeg installed? `ffmpeg -version`
- Is the URL accessible? Try `yt-dlp --simulate <url>`

### Job status = FAILED at transcribe
- Is faster-whisper installed? `pip show faster-whisper`
- Is there enough RAM/VRAM? Whisper base needs ~1GB

### Job status = FAILED at render_character / synthesize_voice
- Run the Track B pre-flight check — files might be missing
- Check AUTOMATIC1111 / Chatterbox TTS health endpoints

### Job status = FAILED at generate_video / lip_sync
- Check Wan 2.7 / Wav2Lip service health
- Check work_dir for the input files (PNG from render, WAV from voice)

### General debugging

```python
# Inspect stage results
for stage, result in job.stage_results.items():
    print(f"{stage}: {result.status} — {result.artifact}")

# Check work directory
import os
work_dir = f".local/jobs/{job.job_id}"
for root, dirs, files in os.walk(work_dir):
    for f in files:
        path = os.path.join(root, f)
        print(f"  {path} ({os.path.getsize(path)} bytes)")
```

---

## Phase gate (before promoting to Phase 2)

Phase 1 acceptance criteria (from `orchestration-build-plan.md §9`):
- ✅ A real educational-kids link produces a watchable ~30–60s 9:16 short
- ✅ Starring the configured original cast with correct captions
- ✅ Distinct voices, per-shot lip sync
- ✅ Metadata sidecar with all required flags
- ✅ A job with non-original cast / missing rights / failed age-check is blocked
- ✅ Swapping species/genre in config changes output with no code change (LoRAs aside)

Once the first real video passes human review → Phase 2 (critic loop) can begin.
