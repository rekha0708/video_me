# Video Enhance Pipeline Plan

Status: experimental, wired. The stable production path still uses
`FfmpegAssembleAdapter` for final assembly, with optional Lanczos +
`minterpolate` controlled by `VIDEO_ME_VIDEO_UPSCALE_ENABLED`.

## Goal

Build a separate `video_enhance` pipeline that can mature independently and
later be plugged between shot generation/lip-sync and final assembly.

The target mature flow is:

1. Generate or lip-sync per-shot clips.
2. Enhance clips before captions, overlays, and disclosure text are burned in.
3. Assemble enhanced clips into the final 1080x1920 video.
4. Keep the existing FFmpeg assemble/upscale path as a fallback.

## Why Separate

The current upscale is safe because it is pure FFmpeg and runs in the final
assemble filter chain. AI upscalers/interpolators are riskier:

- They can distort captions, charts, disclosure labels, and any other burned-in
  text if run after final assembly.
- They may alter mouth shapes frame-to-frame, which matters for Wan S2V and
  LatentSync/MuseTalk outputs.
- They are GPU-heavy and need the same lifecycle discipline as Wan/LightX2V.
- They need per-shot artifacts for before/after review and retry debugging.

Keeping this as a separate capability lets us benchmark quality and throughput
without destabilizing completed jobs.

## Current Implementation

Code added/wired:

- `core.models.capabilities.VideoEnhanceRequest`
- `core.models.capabilities.VideoEnhanceResult`
- `core.capabilities.base.EnhanceVideo`
- `adapters/video_enhance/ffmpeg_adapter.py`
- `adapters/video_enhance/ai_adapter.py`
- `scripts/enhance_video.py` for manual/offline experiments against an
  existing MP4

The scaffold is wired into `core.workflow` behind `video_enhance_enabled`. It is
off by default and only runs when the dashboard/job override or env enables it.
The dashboard exposes a backend selector:

- `ffmpeg`
- `rife`
- `film`
- `realesrgan_rife`
- `realesrgan_film`
- `latent_rife`
- `latent_film`

`setup_gpu.sh` installs the Real-ESRGAN/RIFE/FILM backend assets by default in
the test GPU environment. These backends are still not resident services: they
run as short-lived subprocesses only during the `video_enhance` stage, so simply
installing them does not keep VRAM occupied. Use `--skip-video-enhance` only for
minimal/fast setup runs.

`realesrgan_rife` is the intended first AI throughput benchmark:

1. Real-ESRGAN video restoration.
2. RIFE target-FPS interpolation.
3. FFmpeg normalization to the requested resolution/FPS with original audio
   reattached.

`realesrgan_film` is a quality/comparison path:

1. Real-ESRGAN video restoration.
2. Extract frames.
3. FILM recursive midpoint interpolation.
4. Encode at the source-FPS multiplier, then normalize/drop frames to the
   requested target FPS while preserving duration.
5. Reattach original audio.

`latent_rife` and `latent_film` are backend contracts only until
`assets/comfyui_workflows/video_enhance_latent.json` exists and is validated.
The adapter health check reports down when the workflow file is missing.

## Candidate Backends

### FFmpeg Baseline

Purpose: contract validation and fallback.

Behavior:

- `minterpolate` or simple `fps`
- Lanczos scale/pad to target resolution
- optional audio stream preservation

This is not AI super-resolution and not AI frame interpolation.

### RIFE Interpolation

Purpose: first AI interpolation backend.

Why first:

- RIFE is explicitly a video frame interpolation method.
- The upstream project documents 2x interpolation from video and image
  sequences, plus target-FPS mode.
- The upstream README reports 30+ FPS for 2x 720p interpolation on a 2080Ti,
  which makes it a practical first benchmark for our throughput goal.

Integration shape:

- Calls upstream `inference_video.py` with `--video`, `--output`, `--model`,
  `--fps`, `--scale`, and `--exp`.
- Uses `--fps` for requested target FPS. Upstream notes audio is not merged in
  this mode, so the adapter always reattaches the original audio in the final
  FFmpeg normalization step.
- Health requires a local `train_log/*.pkl` model. Runtime jobs do not download
  RIFE weights.
- Setup does not install RIFE's upstream `requirements.txt` directly because it
  pins `numpy<=1.23.5`, which fails on Python 3.12. The setup script installs a
  compatible modern runtime dependency set instead and inherits CUDA torch from
  the base image.

### FILM Interpolation

Purpose: quality fallback for large-motion clips.

Gotcha:

- The reference implementation is TensorFlow 2 SavedModel based.
- That likely means a separate venv/container from the PyTorch-heavy Wan/Flux
  stack to avoid dependency collisions.

Implementation:

- Uses a separate `/workspace/.venv_film`.
- Extracts frames from each clip and calls upstream `python -m
  eval.interpolator_cli`.
- `times_to_interpolate` produces a power-of-two multiplier, so the adapter
  encodes the FILM output at `source_fps * 2^times` to preserve duration, then
  normalizes to the requested target FPS.

Use FILM when RIFE has visible failure cases worth comparing; it is expected to
be slower and more dependency-sensitive.

### AI Super-Resolution

Pragmatic near-term option:

- Real-ESRGAN or equivalent image/video restoration before interpolation.

Implementation:

- Uses upstream `inference_realesrgan_video.py`.
- Default model is `realesr-general-x4v3` with outscale `2.0`.
- Setup downloads the expected weights ahead of time. Health fails when weights
  are absent so runtime jobs do not silently fetch model files.

Gotcha:

- Frame-by-frame super-resolution can shimmer unless the model and settings are
  temporally stable.
- It may sharpen skin, teeth, captions, or chart lines in ways that look
  overprocessed.

### True Latent Two-Pass Upscale

Target mature option:

- A ComfyUI or service-backed workflow that upscales in latent/diffusion space
  with controlled denoise and temporal consistency.

Gotchas:

- This is not just "turn on an upscaler"; it needs chunking, overlap, seed
  control, and consistency checks to avoid flicker.
- Do not run it on final captioned video unless we explicitly want text altered.
- It may need a dedicated service and VRAM lifecycle, not reuse the render
  ComfyUI process casually.

Current repo status:

- The adapter can submit a ComfyUI workflow and fetch a video output.
- The repo intentionally does not ship a placeholder latent workflow graph,
  because an unvalidated graph would be misleading.
- A real latent workflow must be added at
  `assets/comfyui_workflows/video_enhance_latent.json` with placeholder node
  titles the adapter understands:
  - `__INPUT_VIDEO__`
  - `__TARGET_WIDTH__`
  - `__TARGET_HEIGHT__`
  - `__TARGET_FPS__`
  - `__OUTPUT_PREFIX__`
  - `__SEED__`

## Proposed Implementation Phases

### Phase 0 - Contract and Docs

Done in this scaffold.

- Add `video_enhance` request/result models.
- Add `EnhanceVideo` capability.
- Add standalone FFmpeg baseline adapter.
- Document backend choices and gotchas.

### Phase 1 - Manual/Offline Runner

Initial script exists:

```bash
python -m scripts.enhance_video path/to/clip.mp4 \
  --work-dir .local/video_enhance/manual \
  --output-name clip_enhanced.mp4 \
  --fps 48 \
  --interpolation minterpolate
```

Next: add a dashboard-only action that can enhance one existing artifact
without affecting normal jobs.

Inputs:

- existing clip/final video path
- mode: `ffmpeg`
- target FPS/resolution

Outputs:

- enhanced MP4 under the job work directory
- JSON metadata with command, duration, backend, and notes

### Phase 2 - RIFE Backend

Add `RifeVideoEnhanceAdapter` behind the same capability.

Status: implemented in `AiVideoEnhanceAdapter`.

Acceptance covered:

- no workflow integration by default
- health check detects missing binary/model clearly
- tests mock the command line
- GPU setup downloads/checks only if missing
- start-services is not required

### Phase 3 - Dashboard Experiment Panel

Expose enhancement attempts on the job detail page:

- source clip/final video
- backend
- output video
- elapsed time
- estimated cost
- rejection/failure reason

Status: partially done.

- New job dashboard can select enhancement backend.
- Retry UI can enable/disable enhancement and select backend.
- Stage artifacts record per-shot source/enhanced clip paths, adapter, and
  notes.
- A richer side-by-side experiment panel is still future work.

### Phase 4 - Clip-Level Pipeline Integration

Insert enhancement after `generate_video`/`lip_sync` and before
`assemble_video`.

Integration is present and disabled by default:

```env
VIDEO_ME_VIDEO_ENHANCE_ENABLED=false
VIDEO_ME_VIDEO_ENHANCE_ADAPTER=ffmpeg
VIDEO_ME_VIDEO_ENHANCE_TARGET_FPS=48
VIDEO_ME_VIDEO_ENHANCE_INTERPOLATION=minterpolate
```

Important:

- captions/overlays remain after enhancement
- audio timing remains authoritative
- failed enhancement falls back to original clip unless policy is `fail`

### Phase 5 - Latent Two-Pass Backend

Add a validated ComfyUI/service backend only after RIFE has baseline metrics.

Acceptance:

- tile/chunk plan is deterministic
- overlap avoids boundary flicker
- VRAM load/unload is explicit
- before/after artifacts are dashboard-visible
- default remains off
- workflow graph exists in repo and has been run against real per-shot clips
- output is compared for flicker, mouth distortion, identity drift, and text
  distortion before making it selectable as a default

## Gotchas Checklist

- Do not enhance burned-in captions unless the operator explicitly selects it.
- Do not run frame interpolation across hard cuts; process per-shot clips before
  assembly or split final video by shot boundaries.
- Do not run AI enhancement after captions/disclosure text have been burned in
  unless the operator explicitly wants text altered.
- RIFE/FILM may alter mouth motion slightly; use on S2V outputs only after a
  visual/lip-sync spot check.
- FILM target FPS is not arbitrary internally; it is a recursive power-of-two
  interpolation, then normalized.
- Real-ESRGAN can shimmer frame-to-frame. Benchmark against FFmpeg and RIFE-only
  before using it as a default.
- Latent workflows must be per-shot/chunked with overlap; final-video latent
  processing risks cut-boundary artifacts and text distortion.
- Do not trust generated duration metadata; probe output with ffprobe.
- Preserve original audio timing and reattach audio after visual processing.
- Do not let AI interpolation duplicate/shift audio frames; audio is separate.
- Track cost by backend and elapsed time, not just overall assembly time.
- Keep failed enhance attempts visible on the dashboard.
- Use local/pinned model files; setup should download only when missing.
- Keep RIFE/FILM/latent deps out of the main venv unless proven compatible.
- Add visual QA: black-frame check, output resolution/FPS probe, duration delta.
- Avoid caption/chart distortion by applying overlays after enhancement.
- Keep final workflow default as current FFmpeg path until real benchmarks pass.

## Initial Recommendation

Build in this order:

1. FFmpeg standalone adapter and offline runner.
2. RIFE CLI adapter.
3. Dashboard experiment panel.
4. Clip-level workflow insertion behind an off-by-default setting.
5. Latent two-pass service after RIFE metrics are known.
