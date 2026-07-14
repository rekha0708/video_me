# Wan 2.2 Animate Integration Plan

Status: implemented in code; real-model Hopper smoke validation pending  
Scope: both official Wan 2.2 Animate modes, selectable per job from the
dashboard:

- **Motion transfer** (`animate`; called **Move** in some ComfyUI workflows)
- **Character replacement** (`replace`; called **Mix** in some ComfyUI workflows)

This plan targets the official `Wan-AI/Wan2.2-Animate-14B` implementation and
preprocessing pipeline. ComfyUI documentation and community reports were used
to identify operational failure modes, but the proposed production backend is
not ComfyUI.

## Executive Decision

Add Wan Animate as a fifth `generate_video` backend with a dedicated native
service on port `8033`, a separate Python environment, and an explicit
preprocessing phase between image approval and Animate model loading.

The intended job flow is:

1. Obtain and approve one reference image per shot using the existing flow.
2. Resolve the driving-video timeline for every shot.
3. Slice and normalize every driving segment.
4. Run the official pose/face preprocessing for all shots; replacement mode
   additionally runs SAM2 and produces background/mask inputs.
5. Release all preprocessing models and verify GPU memory is free.
6. Load Wan Animate once, generate all shots, then unload it once.
7. Apply the selected lip-sync repair, if any, and assemble as today.

Do not load SAM2, optional Flux Kontext, and the 14B Animate diffusion model at
the same time. Batch the stages, not the models.

## Why Native Wan Instead of a ComfyUI Workflow

The native official path fits this repository's current architecture:

- Wan S2V, Wan I2V, and LightX2V already use managed adapters/services.
- The official preprocessing contract is explicit and can be version-pinned.
- It avoids runtime dependence on ComfyUI custom nodes such as KJNodes and
  `comfyui_controlnet_aux`, their workflow schema, and node-version drift.
- It gives the worker direct control over preprocessing artifacts, caching,
  GPU release, retries, and QA.

ComfyUI remains useful as a comparison/debugging tool, but it should not be the
first production implementation.

## What the Two Modes Actually Require

### Motion transfer (`animate`)

Input:

- one approved reference image
- one driving-video segment

Preprocessed artifacts:

- `src_ref.png`
- `src_pose.mp4`
- `src_face.mp4` (official preprocessing crops faces to 512x512)

Behavior: transfers body motion and facial motion to the reference character.
The result is not a background-preserving edit of the source video.

### Character replacement (`replace`)

Input:

- one approved reference image for the replacement character
- one driving-video segment containing the person to replace

Preprocessed artifacts:

- `src_ref.png`
- `src_pose.mp4`
- `src_face.mp4`
- `src_bg.mp4`
- `src_mask.mp4`

Behavior: segments a source person, replaces that person, and retains the
source background through the generated background/mask conditioning. The
official relighting LoRA is loaded for this mode.

Both modes require a driving video. A still image plus audio alone is not a Wan
Animate input; that use case remains Wan S2V.

## Dashboard Product Contract

Add `Wan 2.2 Animate — motion transfer / character replacement` to the **Video
Model** selector. Selecting it reveals a Wan Animate panel.

### Basic controls

- **Mode**
  - Motion transfer
  - Character replacement
- **Driving video**
  - Use this job's source video
  - Upload a separate driving video
  - Choose a server-local video
- **Timeline mapping**
  - Source timestamps
  - Sequential from start
- **Subject selection** (replacement mode)
  - Largest person (default)
  - Person nearest center
  - Pick subject from a preview frame (later Phase 2 enhancement)

### Advanced controls

- resolution-area preset: 480p-equivalent or 720p-equivalent (default 720p)
- target preprocessing FPS: fixed to 30 in the first release
- pose retargeting: off by default
- Flux-assisted pose retargeting: off and hidden unless installed
- temporal reference frames: 1 (default) or 5
- sampling steps: 20 default, bounded to a tested range
- replacement mask dilation iterations: 3 default
- replacement mask kernel: 7 default
- replacement mask grid `w_len` / `h_len`: 1 default

The UI should explain:

- Motion transfer generates a new scene based on the reference image.
- Character replacement aims to preserve the driving video's background.
- Wan Animate does not natively use job audio for lip sync. `LatentSync`,
  `MuseTalk`, or `none` remains a separate choice.
- One pass targets one primary person. Multi-person replacement is not
  supported in the first release.

### Source-mode behavior

| Job source | Default driver | Default timeline | Validation |
|---|---|---|---|
| URL/upload/file video + source audio | source video | source timestamps | every shot needs valid source start/end |
| URL/upload/file video + re-voice | source video | source timestamps | every shot needs valid source start/end |
| URL/upload/file video + full | source video | sequential | total planned duration must fit |
| Story/story + images | uploaded/local driver | sequential | driver is required before queueing |

Do not silently loop, reverse, freeze, or ping-pong a short driving video. Fail
before GPU work with the exact required and available durations. Looping can be
added later as an explicit creative option.

## Data Model

Add `"wan_animate"` to the video-adapter literals in `core/config.py` and
`core/models/dashboard.py`.

Prefer a nested job option object instead of adding many unrelated fields to
`DashboardJobOverrides`:

```python
class WanAnimateOptions(BaseModel):
    mode: Literal["animate", "replace"] = "animate"
    driver_source: Literal["job_source", "upload", "local"] = "job_source"
    driver_uri: str | None = None
    timeline: Literal["source_timestamps", "sequential"] = "source_timestamps"
    subject_selection: Literal["largest", "center"] = "largest"
    resolution_area: Literal["480p", "720p"] = "720p"
    fps: Literal[30] = 30
    retarget_pose: bool = False
    use_flux_retarget: bool = False
    refert_num: Literal[1, 5] = 1
    sampling_steps: int = Field(default=20, ge=10, le=40)
    mask_iterations: int = Field(default=3, ge=0, le=10)
    mask_kernel: int = Field(default=7, ge=1, le=31)
    mask_w_len: int = Field(default=1, ge=1, le=8)
    mask_h_len: int = Field(default=1, ge=1, le=8)
```

`use_flux_retarget=True` must require `retarget_pose=True`. Replacement mode
must reject both retarget flags because the official preprocessor supports
them only in animation mode.

Add an explicit driver contract to `core/models/capabilities.py`:

```python
class VideoDriver(BaseModel):
    uri: str
    start_sec: float
    end_sec: float
    mode: Literal["animate", "replace"]
    prepared_dir: str | None = None

class VideoRequest(BaseModel):
    # existing fields...
    driver: VideoDriver | None = None
```

Only `WanAnimateAdapter` accepts `driver`; existing adapters ignore `None` and
retain their current contract.

Add a batch-preparation method to the Animate adapter rather than running
preprocessing inside every `generate()` call:

```python
async def prepare_inputs(
    requests: list[VideoRequest],
) -> dict[str, PreparedWanAnimateInput]: ...
```

This keeps GPU lifecycle and caching visible to the workflow.

## Driver Ingestion and Normalization

### Accepted uploads

First release:

- containers: MP4, MOV, WebM, MKV
- source codecs accepted through FFmpeg normalization: H.264, H.265, VP9, AV1
- no GIF, animated WebP, or image-sequence input
- no remote driver URL separate from the job URL initially; upload or local
  file is safer and easier to validate

The upload endpoint must stream chunks to a job-scoped directory. It must not
read a full video into memory. Resolve all local paths and ensure they remain
under an allowed media root; the service must never accept an arbitrary client
filesystem path.

Recommended configurable intake limits:

- raw driver size: 2 GiB
- raw driver duration: 10 minutes
- minimum usable shot segment: 1 second
- initial hard per-shot maximum: 10 seconds
- recommended shot duration: 2 to 2.5 seconds
- warn above 77 frames at 30 FPS (about 2.57 seconds), because generation then
  crosses the model's native chunk boundary

The existing default maximum shot duration is 8 seconds, so the initial
10-second Animate cap does not reduce normal jobs.

### Preflight with FFprobe

Before queueing GPU work, validate:

- exactly one readable video stream
- finite, positive duration
- sane width/height (recommend at least 256 pixels on the shorter side)
- valid time base and average frame rate
- source segment boundaries lie inside the video duration
- total sequential timeline fits inside the driver
- rotation metadata, pixel aspect ratio, color transfer, and HDR flags are
  recorded

Normalize each shot slice to a deterministic intermediate:

- constant 30 FPS
- H.264, `yuv420p`
- square pixels
- no rotation metadata
- no audio stream
- aspect ratio preserved
- dimensions rounded to multiples of 16
- SDR Rec.709 output for HDR sources
- timestamps starting at zero

Variable-frame-rate video must be converted to constant frame rate before the
official Decord/MoviePy pipeline. This avoids frame-count/duration mismatches
already acknowledged in the upstream preprocessing code.

## Timeline Resolution

Create all per-shot `VideoRequest` objects after image approval, before loading
the Animate model.

For `source_timestamps`, use `Shot.source_start_sec` and
`Shot.source_end_sec`. This mode is legal only when all rendered shots have
those fields.

For `sequential`, allocate contiguous ranges in storyboard order from time
zero using each shot's planned duration. Record the resolved ranges as a JSON
artifact so resume and rerun use the same mapping.

A rerun of one shot must reuse its original time range unless the user
explicitly changes the driver mapping.

## Preprocessing Design

Use the official Wan preprocessing implementation, pinned to the same Wan repo
revision as generation.

### Batch lifecycle

1. Render/approve all reference images.
2. Slice and normalize all required driver segments with FFmpeg.
3. Start one short-lived preprocessing subprocess.
4. Load YOLOv10 and ViTPose once; load SAM2 once only for replacement jobs.
5. Process all pending shots in one batch.
6. Write manifests and preview artifacts.
7. Exit the subprocess and verify its CUDA memory has been released.
8. Only then load the Animate service/model.

The official preprocessor requests `CUDAExecutionProvider` for its ONNX pose
models, but its requirements file lists CPU-only `onnxruntime`. Our environment
must install `onnxruntime-gpu`, remove/conflict-check `onnxruntime`, and assert:

```python
"CUDAExecutionProvider" in onnxruntime.get_available_providers()
```

No silent CPU inference fallback is allowed. FFmpeg/Decord/OpenCV decoding,
resizing, and video encoding remain CPU work; the GPU-only promise applies to
neural model inference.

SAM2 runs on CUDA for replacement. Optional Flux Kontext runs BF16 on CUDA.
The upstream SAM2 preprocessing explicitly disables its flash-attention path,
so FA3 availability should not be treated as accelerating SAM2. Wan Animate
generation itself should use the repository's already verified Hopper FA3
setup where supported by the pinned Wan implementation.

### Preprocessing artifacts

Store under:

```text
<job>/wan_animate_preprocess/<shot_id>/
  driver_normalized.mp4
  src_ref.png
  src_pose.mp4
  src_face.mp4
  src_bg.mp4          # replace only
  src_mask.mp4        # replace only
  preview_pose.mp4
  preview_mask.mp4    # replace only
  contact_sheet.jpg
  manifest.json
```

The manifest contains input hashes, exact model revisions, options, detected
people/pose statistics, frame count, FPS, dimensions, elapsed time, and output
hashes.

### Cache key

Cache only when all of these match:

- driver content hash (or trusted local path + size + nanosecond mtime)
- exact slice start/end
- reference-image SHA-256
- mode
- preprocessing FPS and resolution area
- retarget and optional Flux settings
- mask settings and subject-selection rule
- Wan repo revision and preprocessing checkpoint revisions

Changing a reference image or mask option must invalidate that shot only.

## Model Service

Add `services/wan_animate_server.py` and
`adapters/generate_video/wan_animate_adapter.py`.

Recommended API:

- `GET /health`
  - service ready
  - model loaded/loading
  - model revision/path
  - CUDA device and capability
  - FA3 validation status
  - last load error
- `POST /load`
- `POST /unload`
- `POST /generate`
  - job-relative prepared-input identifier
  - mode
  - seed
  - sampling steps
  - `refert_num`
  - output path/name

The server must resolve prepared directories under the configured job-data
root. Do not expose an unrestricted server-side path API.

Mark the adapter `managed_vram=True`, `native_lipsync=False`. Load once before
the video loop, generate all shots, and unload/re-exec after the job if CUDA
allocator retention is observed, matching the proven S2V lifecycle pattern.

Generation defaults should follow upstream:

- clip length: 77 frames (`4n+1`, not user-editable initially)
- temporal reference frames: 1, optionally 5
- sampling steps: 20
- solver: DPM++
- guide scale: 1
- no custom positive prompt by default; upstream does not recommend it
- relighting LoRA enabled only for replacement

Note the FPS distinction: official native preprocessing defaults to 30 FPS,
so 77 frames is about 2.57 seconds. Many ComfyUI examples use 16 FPS and call
77 frames about 4.81 seconds. Do not transfer that 16 FPS duration assumption
into this native implementation.

## Setup and Dependency Isolation

Extend `scripts/setup_gpu.sh` with opt-in `--with-wan-animate` and an optional
`--with-wan-animate-flux-retarget`.

Use a separate `/workspace/.venv_wan_animate` rather than modifying the shared
S2V/I2V environment. Animate adds SAM2, PEFT, ONNX Runtime, Diffusers/Flux
dependencies, and a pinned Git dependency; isolating them avoids destabilizing
the working Wan backends.

Install and pin:

- the same Torch/CUDA ABI used by the GPU image
- FA3 built for `sm_90a` and validated with the existing Hopper check
- Wan repository at an explicit commit
- `decord`
- `peft`
- `onnxruntime-gpu` (not `onnxruntime`)
- `pandas`, `matplotlib`, `loguru`, `sentencepiece`
- SAM2 at upstream's currently specified commit
- MoviePy version compatible with both import layouts used upstream
- FFmpeg/FFprobe system packages

Download `Wan-AI/Wan2.2-Animate-14B` at a pinned revision. The current model
repository is approximately 72.4 GB and includes the 14B weights, VAE, T5,
CLIP, preprocessing checkpoints, and relighting LoRA. Budget additional cache
and temporary space; setup should require at least 160 GB free before download
and extraction/caching.

The required preprocessing checkpoint tree includes:

- `process_checkpoint/pose2d/vitpose_h_wholebody.onnx`
- `process_checkpoint/det/yolov10m.onnx`
- `process_checkpoint/sam2/sam2_hiera_large.pt`

Flux Kontext is optional, large, and not part of the base install. It should
have a separate setup flag, readiness result, disk estimate, and license/model
access check. Never auto-download it during a job.

Extend:

- `scripts/start_services.sh` with `NEED_WAN_ANIMATE` and port 8033
- `scripts/check_runtime_readiness.py` with Animate service, checkpoints,
  CUDA provider, FA3, FFmpeg, and disk checks
- `docs/PORTS.md`, deployment docs, `.env.example`, and code maps

## GPU/VRAM Sequencing

Target sequence for an Animate job:

```text
LLM/transcription -> release
image renderer -> approve references -> unload/release
Animate preprocessor (pose + optional SAM2/Flux) -> subprocess exits
Wan Animate 14B -> all shots -> unload/re-exec
voice/lip-sync services -> assemble
```

Do not eagerly load Wan Animate from `start_services.sh`; start the lightweight
HTTP process and defer model load. Preprocessing should not happen inside the
resident Animate server because a normal Python-level unload may not return all
SAM2/Flux allocator memory.

Before loading Animate, record `nvidia-smi` free memory and enforce a minimum
headroom threshold derived from a real H100 load benchmark. Do not guess the
final threshold in code; capture load/generation peaks during Phase 0 and use
the observed maximum plus margin.

## Audio and Lip Sync

Wan Animate consumes pose/face/background conditioning, not the job's audio.
The existing audio path remains authoritative:

- source-audio mode slices the source audio by the same shot timestamps
- re-voice/full synthesize TTS as today
- the final assembler receives separate `AudioTrack` objects

Default recommendation:

- movement-focused or non-speaking clips: `lipsync_adapter=none`
- talking/singing output: `latentsync`, with the existing fallback/fail policy

Applying lip-sync after Animate can change facial motion already transferred
from the driver. Surface that tradeoff in the UI and preserve both raw Animate
and repaired clips for review.

## Validation and Automatic Guardrails

### Before preprocessing

- driver metadata and timeline checks pass
- approved reference is readable RGB and has a detected primary person
- warn for multiple large people; replacement defaults to largest only
- detect hard cuts inside a shot and warn/fail by policy
- reject paths outside allowed roots

### After preprocessing

- source pose, face, mask, and background frame counts match
- frame count is nonzero and FPS/dimensions match manifest
- dimensions are multiples of 16
- pose is detected in a configurable fraction of frames
- face crop is valid and not empty/tiny for a configurable fraction of frames
- replacement mask is neither all-zero nor all-one
- mask coverage remains in a sane range and temporal jitter is reported
- produce contact sheets/previews for diagnosis

### After generation

- output can be decoded fully
- expected frame count/duration, FPS, and dimensions are within tolerance
- no NaN/black-frame run, frozen-frame run, or gross duplicate-frame run
- inspect chunk boundaries for sudden histogram/brightness changes
- detect resolution/aspect changes or zoom jumps
- replacement mode: compare outside-mask background preservation
- replacement mode: verify sufficient change occurred inside the mask so an
  unchanged original person is flagged

Quality failures should retain all inputs and previews, return a specific error
code, and be retryable without rerunning completed preprocessing.

## Known Corner Cases

### Driver/person detection

- no person, no visible face, tiny person, or person almost fully off-screen
- several similarly sized people; largest-person selection may switch targets
- subject enters/leaves frame or crosses another person
- heavy occlusion, back-facing/profile motion, fast motion blur
- animals, highly stylized figures, or non-human rigs that pose extraction
  cannot represent
- hands, hair, loose clothing, props, or accessories leaking outside masks
- hard camera cuts within a shot

### Reference image

- reference is not front-facing or does not show limbs required by the driver
- extreme reference/driver body-proportion mismatch
- crop/aspect mismatch, transparent image, grayscale image, or embedded color
  profile surprises
- multiple characters in a single approved reference
- different approved reference per shot causing identity drift in the final
  multi-shot video

Pose retargeting can help body-proportion/pose mismatch. Flux-assisted
retargeting is recommended upstream only when the reference or driving first
frame is not in a standard front-facing pose, but it adds major VRAM, latency,
disk, and another generative failure point.

### Replacement masks/background

- SAM2 chooses the wrong person or includes nearby objects
- mask is too tight and clips hair/hands/clothes
- mask is too broad and regenerates background
- mask dilation creates block/halo artifacts
- moving camera or rapid occlusion causes mask flicker
- original subject remains because the mask was ineffective
- lighting/color mismatch between replacement and retained background

### Video/timing

- VFR sources, bad timestamps, corrupt/truncated frames
- portrait rotation stored only in metadata
- HDR/10-bit input producing washed-out SDR output
- source duration shorter than the resolved storyboard
- source and output FPS assumptions mixed between native 30 FPS and ComfyUI
  16 FPS examples
- long shots crossing one or more 77-frame chunks
- seams, color/brightness shifts, or scale/zoom jumps at chunk boundaries
- rounding a requested duration to the model's legal `4n+1` frame structure

### Operations

- `onnxruntime-gpu` missing or CUDA provider unavailable
- out-of-memory during optional Flux retarget, SAM2, or Animate load
- preprocessor crashes and leaves a child process/GPU allocation
- job cancellation between preprocessing and model load
- service restart during a shot
- concurrent jobs attempting to claim the same GPU
- cache becomes stale after checkpoint/repo revision changes
- retry of one shot accidentally remaps its driver time range
- non-contiguous NumPy arrays passed into OpenCV/ONNX code (reported by
  community workflows); normalize with `np.ascontiguousarray` at boundaries

## Observability and Artifacts

Add dashboard stages/events:

- `wan_animate_driver_validate`
- `wan_animate_driver_slice`
- `wan_animate_preprocess`
- `wan_animate_preprocess_review` (optional gate in Phase 2)
- `wan_animate_model_load`
- `wan_animate_generate`
- `wan_animate_chunk_qa`
- existing lip-sync and assembly stages

Record per-stage wall time, peak GPU memory, peak host memory, input/output
frames, cache hit/miss, model revisions, and failure code. The job detail page
should expose normalized driver, pose preview, face contact sheet, replacement
mask/background previews, raw Animate output, and lip-synced output.

## File-Level Implementation Map

Core/contracts:

- `core/config.py`
- `core/models/dashboard.py`
- `core/models/capabilities.py`
- `core/capabilities/base.py` if a formal batch-prepare protocol is added
- `core/workflow.py`
- `core/gpu_sequencer.py`

Adapter/service:

- `adapters/generate_video/wan_animate_adapter.py`
- `services/wan_animate_server.py`
- `services/wan_animate_preprocess.py`

Dashboard:

- `services/dashboard_api.py` (streamed driver upload + request validation)
- `services/dashboard_worker.py` (settings override and final cleanup)
- `services/templates/job_new.html`
- `services/templates/job_detail.html`
- `services/templates/jobs_list.html`

Operations/docs:

- `scripts/setup_gpu.sh`
- `scripts/start_services.sh`
- `scripts/check_runtime_readiness.py`
- `.env.example`
- `docs/PORTS.md`
- `docs/DEPLOY.md`
- generated code maps after implementation

Tests:

- `tests/test_wan_animate_adapter.py`
- `tests/test_wan_animate_server.py`
- `tests/test_wan_animate_preprocess.py`
- dashboard model/API/template tests
- workflow integration and resume/rerun tests
- readiness/setup shell tests where present

## Delivery Phases

### Phase 0: pinned manual benchmark

- Build isolated environment and download pinned model/checkpoints.
- Validate CUDA ONNX provider and Hopper FA3.
- Run official preprocessing/generation manually for one short example of each
  mode at 480p and 720p.
- Record disk use, load time, preprocessing time, generation time, host RAM,
  and peak VRAM.
- Test 2.0s, 2.57s, 5s, and 8s inputs to measure chunk boundaries.
- Decide production VRAM headroom and timeouts from measurements.

Exit: reproducible command and benchmark report for both modes.

### Phase 1: backend and workflow MVP

- Add contracts, native service, adapter, setup, readiness, and port.
- Support source/local driver, largest-person selection, fixed 30 FPS,
  480p/720p, no Flux retarget.
- Batch preprocessing after image approval and generation after preprocessor
  exit.
- Support resume/cache and existing lip-sync paths.

Exit: one full dashboard job completes for each mode and leaves diagnosable
artifacts.

### Phase 2: dashboard upload and review UX

- Add streamed driver upload.
- Add dynamic Animate form and server-side cross-field validation.
- Show ffprobe metadata and resolved shot timeline before start.
- Add pose/mask preview artifacts and optional approval gate.
- Add center-person selection and eventually preview-point selection.

Exit: story jobs can supply a driver, and users can catch wrong pose/mask
selection before expensive generation.

### Phase 3: hardening and optional retargeting

- Add Flux-assisted retarget setup flag and readiness.
- Add automated mask/background/chunk QA.
- Add cancellation, crash recovery, stale-cache, service re-exec, and one-shot
  rerun coverage.
- Tune long-shot warning/failure thresholds from real benchmark data.

Exit: failure modes are specific, retryable, observable, and do not leak VRAM.

### Phase 4: controlled rollout

- Feature flag off by default in production.
- Canary on one Hopper worker.
- Track success rate and p50/p95 preprocessing/generation times separately by
  mode and duration bucket.
- Enable broadly only after each mode meets the agreed quality/success target.

## Test Matrix

At minimum, cover:

- animate and replace
- source timestamps and sequential mapping
- source video and separately uploaded/local driver
- portrait and landscape
- CFR and VFR source
- 2s, exactly 77 frames, just over 77 frames, 5s, and 8s
- one person, multiple people, no person, subject leaves frame
- reference with good pose and poor/profile pose
- valid mask, all-zero mask, all-one mask, mask flicker
- driver too short and out-of-range timestamp
- resume after preprocessing and rerun one shot
- cancel during preprocessing and during generation
- missing checkpoint, missing CUDA provider, failed model load, OOM
- lipsync none, LatentSync success, and lip-sync fallback

Unit tests mock heavyweight processes and HTTP. GPU integration tests are
explicitly marked and run only on the Hopper worker with local checkpoints.

## Acceptance Criteria

The feature is ready when:

1. Both modes are selectable and correctly validated from the dashboard.
2. Story jobs require an explicit driver; media jobs can use their source.
3. Neural preprocessing refuses silent CPU fallback.
4. Preprocessing and Animate generation never coexist in GPU memory.
5. The Animate model loads once per multi-shot job, not once per shot.
6. Every shot has a deterministic, visible driver time range.
7. Resume/rerun reuses valid preprocessing and invalidates it on relevant
   input/config/model changes.
8. Wrong-person/empty-mask/no-pose cases fail before Animate generation.
9. Raw and lip-synced outputs plus pose/mask previews remain reviewable.
10. Setup/readiness detects every required checkpoint/provider before a job.
11. Targeted unit/API/workflow tests pass, and both-mode Hopper smoke tests
    pass with no residual GPU allocation after cleanup.

## Community Findings to Treat as Risks, Not Specifications

Community reports consistently call out:

- color/brightness changes after the first long-video chunk
- seams improved by continuation/reference-frame handling but not eliminated
- zoom/scale drift caused by width/height wiring mistakes in nested workflows
- unchanged source people when subject/mask selection fails
- block/halo artifacts around replacement masks
- fragile behavior on longer clips and multi-person footage
- occasional non-C-contiguous NumPy/OpenCV errors

These are anecdotal field reports, so they should inform tests and diagnostics,
not override the official model contract.

## Research Sources

Official/primary:

- [Wan2.2 repository and Animate usage](https://github.com/Wan-Video/Wan2.2)
- [Official Wan Animate implementation](https://github.com/Wan-Video/Wan2.2/blob/main/wan/animate.py)
- [Official preprocessing CLI](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/preprocess_data.py)
- [Official preprocessing pipeline](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/process_pipepline.py)
- [Official pose ONNX CUDA-provider selection](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/pose2d.py)
- [Wan2.2-Animate-14B model repository](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B)
- [ComfyUI Wan2.2 Animate guide](https://docs.comfy.org/tutorials/video/wan/wan2-2-animate)
- [ComfyUI WanAnimateToVideo node](https://docs.comfy.org/built-in-nodes/WanAnimateToVideo)

Community/field reports:

- [Character-replacement workflow discussion](https://www.reddit.com/r/comfyui/comments/1of2up1/wan_22_animate_character_replacement_in_comfyui/)
- [Wan Animate success/failure discussion](https://www.reddit.com/r/StableDiffusion/comments/1sf5iv9/does_anyone_have_any_success_with_wan_22_animate/)
- [Longer-video experiments](https://www.reddit.com/r/StableDiffusion/comments/1ohhg5h/tried_longer_videos_with_wan_22_animate/)
- [Replacement workflow and mask reports](https://www.reddit.com/r/comfyui/comments/1nr3vzm/wan_animate_workflow_replace_your_character_in/)
