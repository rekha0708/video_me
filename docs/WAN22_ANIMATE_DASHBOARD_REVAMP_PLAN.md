# Wan 2.2 Animate Dashboard Revamp Plan

Status: approved for implementation; testing-only FLUX.2 Dev use confirmed  
Date: 2026-07-14  
Scope: a dedicated dashboard experience for both Wan 2.2 Animate modes:

- **Motion transfer** (`animate`)
- **Character replacement** (`replace`)

This is a follow-on to `docs/WAN22_ANIMATE_IMPLEMENTATION_PLAN.md`. The model
service and basic job overrides already exist. This plan replaces the crowded
generic form with a clear Animate-specific workflow and adds the missing
character-look, driver-video, audio, finishing, and safety contracts.

## Executive recommendation

Build **Animate Studio** as a dedicated, direct transformation job type at
`/animate/new`, while continuing to use the existing job queue, job detail
page, worker, artifact store, and `POST /api/jobs` entry point.

For the first release, one job should have:

- one driving video;
- one selected target cast member;
- one approved, canonical character look reused across the whole job;
- one audio policy;
- optional lip-sync repair; and
- either generated resolution/FPS or a clearly named enhanced export.

Do not duplicate the full Animate form inside the generic New Job form. When
`Wan 2.2 Animate` is selected there, show a short explanation and an **Open
Animate Studio** button.

## Confirmed product decisions

The operator confirmed these decisions on 2026-07-14:

1. **Direct transformation workflow:** the driving video supplies motion,
   timing, and scene content. Animate Studio does not run story adaptation or
   shot planning first.
2. **Audio semantics:** “Source audio” means audio extracted from the
   uploaded/linked driving video, not audio from a separate generic job source.
   “Cast audio” transcribes the driving-video speech and re-synthesizes the same
   words with the selected cast member's configured voice, fitted to the
   existing timing.
3. **One target person/character:** v1 rejects multi-target jobs. Generated
   looks and cast voice require an explicit cast member; the exact-image mode
   may instead use the uploaded character identity as-is. V1 requires one
   unambiguous primary person throughout the selected range; it does not expose
   a stable track-ID picker, so crossings/occlusion or people swapping
   prominence must be trimmed out rather than treated as safely selectable.
4. **Aspect-preserving export:** “Leave as generated” does not resize or
   interpolate. “Upscale” preserves aspect ratio and targets a configurable
   long edge. A 9:16 1080x1920 export is optional, not the default.

The plan also keeps these recommended working defaults:

- **Look lock:** one approved outfit/reference image is reused for every
  internal chunk. Per-shot wardrobe changes are deferred.
- **Both local inputs:** support upload from the operator's computer and an
  allowlisted server-file picker.

The operator confirmed this is a testing/evaluation workflow. The open
FLUX.2 Dev non-commercial license is therefore the selected path for this
implementation. Commercial/production rollout remains a separate future gate.

## Musubi versus FLUX.2: the current workflow

**Yes: after training, the current default workflow uses FLUX.2 Dev to generate
the cast images.** Musubi-tuner is the software toolkit that trains and runs the
model; it is not a competing image model.

The relationship is:

```text
Training images
  -> Musubi-tuner training scripts
  -> FLUX.2 Dev base model learns a small cast-specific LoRA
  -> loras/<cast>_<member>.safetensors

Shot/outfit prompt
  + FLUX.2 Dev base checkpoint
  + selected member's FLUX.2 LoRA
  -> Musubi-tuner inference script
  -> generated character still
  -> Wan Animate + driving video
  -> animated output
```

Repository evidence:

- `services/dashboard_worker.py` runs Musubi's
  `flux_2_cache_latents.py`, `flux_2_cache_text_encoder_outputs.py`, and
  `flux_2_train_network.py` with `--model_version dev`, the
  `flux2-dev.safetensors` checkpoint, and `networks.lora_flux_2`.
- `adapters/render_character/musubi_flux_adapter.py` runs Musubi's
  `flux_2_generate_image.py` with the same FLUX.2 Dev DiT, VAE, Mistral text
  encoder, and the selected cast LoRA.
- `core/config.py` selects `musubi_flux` as the default render adapter. A
  per-job/environment override can deliberately choose another renderer, but
  that is not the default workflow.

The earlier FLUX.2 caveat is narrower: the adapter currently sends **text
prompts plus a LoRA only**. It does not send garment, accessory, or character
reference images into FLUX.2's editing inputs. Therefore:

- auto-generated outfits can use today's FLUX.2 text-to-image path after we add
  the structured wardrobe prompt;
- exact garment/accessory guidance needs the FLUX.2 single/multi-reference edit
  path wired into a new or extended adapter; and
- exact uploaded character mode bypasses FLUX.2 generation and sends the
  normalized uploaded image directly to Wan Animate as its character
  reference.

This is an integration gap, not a decision to replace or avoid FLUX.2.

## Current baseline and gaps

Already implemented:

- both Wan Animate modes;
- browser driver upload and server-local path;
- streamed driver uploads up to 2 GiB and 10 minutes;
- FFprobe validation and 30 FPS normalization;
- `480p`/`720p`, timeline, subject selection, pose retarget, and temporal
  reference controls;
- source/cast-like audio behavior through the older global render modes;
- LatentSync/MuseTalk post-processing;
- final scaling/frame interpolation; and
- per-job selection of `wan_animate` in the generic dashboard form.

Missing or misleading today:

- Animate is buried in an 800+ line generic job form.
- A separate driving-video URL is unsupported.
- Raw local paths are exposed instead of safe media asset IDs.
- “Source audio” currently points to the generic job source. Audio from a
  separately uploaded Animate driver is discarded during normalization.
- Exact uploaded character images only work through `story_images`; ordinary
  Animate jobs ignore them.
- Cast LoRA rendering has no job-level wardrobe specification or consistency
  lock, so clothes can drift between shots.
- The current FLUX.2 adapter is text-to-image only at our boundary; it cannot
  condition on garment/accessory images.
- “Upscale off” still passes through fixed 1080x1920 assembly behavior, so it
  is not currently “leave as generated.”
- Retry/resume is based too heavily on output-file existence. Changing a
  driver, look, audio, lip-sync, or upscale option can reuse stale artifacts.
- `start_services.sh` starts the Animate service only when the global
  `VIDEO_ME_VIDEO_ADAPTER=wan_animate`. A dashboard-only job override can
  therefore select a service that was never started.
- Cancellation does not yet guarantee that every Animate preprocessing or
  server inference process releases its GPU allocation.

## Information architecture

Add **Animate** to the primary sidebar:

```text
Jobs
Animate
Health
GPU
API Docs
Chat
```

Use one focused page with progressive disclosure rather than multiple modal
forms:

```text
+-----------------------------------------------------------------------+
| Animate Studio                                  Readiness: Ready       |
| Transfer motion or replace one person in a driving video              |
+---------------------------------------------+-------------------------+
| 1. Mode                                     | Job summary             |
|    [ Motion transfer ] [ Character replace ]| Mode                    |
|                                             | Driver                  |
| 2. Driving video                            | Target member           |
|    [ URL | Upload | Server file ]           | Look source             |
|    preview + duration/resolution/audio       | Audio / lip-sync        |
|                                             | Output                  |
| 3. Character and look                       |                         |
|    Cast -> target member                    | Readiness warnings      |
|    [ Auto LoRA | Design outfit | Exact ]    |                         |
|                                             | [ Create Animate job ]  |
| 4. Audio and lip-sync                       |                         |
|                                             |                         |
| 5. Output                  [Advanced ...]    |                         |
+---------------------------------------------+-------------------------+
```

Desktop uses a two-column layout with a sticky summary. Mobile collapses to
one column and gets a small top navigation/menu instead of hiding navigation
entirely.

## Page behavior

### 1. Animate mode

Use two selectable cards, not a dropdown:

- **Motion transfer** — create a new scene/character animation using the
  driving person's body and facial motion.
- **Character replacement** — replace the selected person while attempting to
  retain the driving video's background.

Changing the mode updates the summary and reveals only relevant advanced
controls. Subject/mask controls appear only for replacement. Pose retargeting
appears only for motion transfer.

### 2. Driving video

Name this input **Driving video**, not “reference video,” because “reference”
is also used for character and clothing images.

Input tabs:

- **URL** — direct HTTP(S) video or a URL supported by the existing media
  fetcher;
- **Upload** — a video from the operator's computer;
- **Server file** — an advanced picker restricted to configured media roots.

After ingest/inspection, show:

- playable preview;
- filename or host;
- duration, dimensions, FPS, codec, and size;
- whether an audio stream exists;
- detected person count on representative frames;
- estimated internal chunk count and GPU-intensive warning; and
- a clear validation error before queueing.

The existing 2 GiB/10-minute intake limits remain storage guardrails for source
videos, but one queued Animate range is hard-capped at **30 seconds**. V1 shows
an explicit recommended duration of **3–10 seconds** and requires a second
confirmation above 10 seconds. A longer source must be trimmed with the range
control. Longer ranges cross Wan's roughly 77-frame generation boundaries
repeatedly and need seam/continuity QA.

Do not automatically loop, reverse, freeze, or ping-pong a driver that is too
short for a requested range.

### 3. Character and look

Require:

1. cast selection;
2. one target member from that cast; and
3. one of the following look strategies.

#### A. Auto-generate with cast LoRA — default

Behavior:

- apply the selected member's identity LoRA;
- use that member's configured default wardrobe when present;
- otherwise generate a structured wardrobe specification once from the visual
  style/context;
- render 2–4 full-body candidates with deterministic seeds;
- require a **Look approval** before Animate preprocessing; and
- store the selected image and wardrobe manifest as the canonical job look.

Important wording: the LoRA maintains identity; the wardrobe prompt/spec picks
the outfit. The UI must not imply that the LoRA alone knows which outfit the
operator wants.

#### B. Design a complete look

In this workflow, “garment” or the legacy API name `wardrobe` means the cast's
complete styled look, not clothing alone. It can include a dress or other
clothing, jewelry, bags, sandals/boots or other footwear, makeup/lipstick,
hair, and custom styling. The dashboard must let the operator identify every
category that may change so FLUX.2 can preserve untargeted parts of the
identity-base image during reference-guided edits. Text-only generation has no
identity-base edit pass, so unspecified styling uses coherent generated
defaults rather than claiming exact preservation.

Basic fields:

- requested change categories;
- clothing type;
- primary color;
- material/pattern;
- jewelry;
- bags;
- footwear;
- makeup/lipstick;
- hair;
- other accessories;
- additional details; and
- optional negative constraints, such as “no hat” or “no logos.”

Optional assets:

- one or more clothing/dress images;
- one or more styling-detail images for jewelry, bags, footwear, makeup, or
  other accessories; and
- later, an optional style-board image.

Delivery should be split into two capabilities:

1. **Text-directed complete look** — combine the structured styling prompt with the
   cast LoRA and generate candidates. This fits the current renderer after its
   request contract is extended.
2. **Image-directed complete look** — generate a neutral canonical cast image,
   then run FLUX.2 image editing with that identity image plus clothing/styling
   references. This requires a new multi-reference edit adapter and real-Hopper
   validation; the current adapter cannot do it.

FLUX.2 supports image editing and multiple reference images in the official
model family, but this should be presented as guided dressing, not guaranteed
virtual try-on. Exact logos, patterns, jewelry, garment geometry, and identity
can drift. Always put this mode behind Look approval. Official references:

- [FLUX.2 repository](https://github.com/black-forest-labs/flux2)
- [FLUX.2 image-editing guide](https://docs.bfl.ai/flux_2/flux2_image_editing)
- [Musubi FLUX.2 documentation](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md)

Before implementation, pin the Musubi revision and verify:

- the exact control/reference-image CLI contract;
- whether the installed cast LoRAs are compatible with the edit path;
- identity retention with 1–3 clothing references;
- peak VRAM and clean model unload on Hopper; and
- the selected model's deployment license. FLUX.2 `dev` is not an
  unrestricted commercial model, so production use needs an explicit license
  decision.

#### C. Use an exact character image

Behavior:

- cast selection is optional for identity, but a member is still required if
  **Cast voice** is selected;
- bypass FLUX.2/LoRA character generation;
- normalize orientation/color/size for Wan without creatively regenerating the
  image;
- label the choice **Use this image as the character reference**, not “use
  as-is,” because preprocessing still resizes/converts it; and
- validate that it contains one clear person with adequate face/body coverage.

The user's requested polished thumbnail card is P3. A minimal local preview is
recommended in P1 because it prevents queueing the wrong file; P3 adds the
premium crop, replace/remove controls, and metadata presentation.

### Look approval gate

All generated look modes should stop before expensive Animate preprocessing
and show:

- target cast/member;
- LoRA and model revision;
- structured outfit summary;
- clothing/accessory references;
- candidate images, seeds, and prompts;
- approve, regenerate, and edit-description actions.

Save the approved image's content SHA and reuse that exact normalized image for
every internal chunk. Do not regenerate per shot by default.

### 4. Audio and lip-sync

Use a segmented control with three explicit audio choices:

- **Driving-video audio** — preserve the driver's audio and timing;
- **Cast voice** — transcribe the driver, synthesize the same words with the
  selected member's voice, and fit each segment to the original timeline;
- **No audio** — export silent video.

Under **Cast voice**, later expose an advanced script choice:

- **Keep original words** — v1 default;
- **Use adapted script** — later, because changing words while holding the
  driver's fixed timing needs rewrite/fit controls.

Lip-sync is a separate switch:

- default **Off** for driving-video audio because Wan already transfers facial
  motion and a repair pass can damage identity;
- default **On / LatentSync** for cast voice because newly generated speech may
  not align with the transferred mouth motion;
- allow MuseTalk under Advanced; and
- show a warning for multi-speaker audio in the single-target v1 workflow.

Processing order must be: finish Animate -> unload Wan -> run lip-sync -> unload
lip-sync -> upscale/export. Never co-reside the 14B Animate model and the
lip-sync model merely to save a service call.

### 5. Output

Expose two first-class choices:

- **Leave as generated** — preserve generated dimensions, aspect ratio, and
  FPS; this is the default.
- **Upscale final video** — aspect-preserving enhanced export with the exact
  target and method stated in the UI.

Do not call the existing FFmpeg Lanczos scale plus interpolation “AI
upscaling.” Initial labels should be factual, for example:

- **Scale to 1080p long edge**;
- **Smooth to 48 FPS** (separate checkbox); and
- later **AI enhance with Real-ESRGAN + RIFE** when that backend is selected.

Generate-at `480p`/`720p` belongs under output quality. Sampling steps,
timeline mapping, subject selection, temporal reference, pose retargeting, and
mask parameters belong under an Advanced disclosure with safe bounds.

## Domain and API contract

Do not add wardrobe and asset fields to the existing flat
`DashboardJobOverrides`. Introduce a versioned, Animate-specific request:

```python
class CreateDashboardJobRequest(BaseModel):
    workflow_kind: Literal["pipeline", "wan_animate_direct"] = "pipeline"
    animate: WanAnimateJobOptions | None = None
    # existing pipeline fields remain for workflow_kind="pipeline"


class WanAnimateJobOptions(BaseModel):
    schema_version: Literal[1] = 1
    mode: Literal["animate", "replace"] = "animate"
    driver: AnimateDriverInput
    character: AnimateCharacterOptions
    audio: AnimateAudioOptions
    lipsync: AnimateLipSyncOptions
    output: AnimateOutputOptions
    advanced: AnimateAdvancedOptions = Field(default_factory=...)
```

Recommended nested fields:

```python
class AnimateDriverInput(BaseModel):
    asset_id: str
    target_confirmed: Literal[True]
    timeline: Literal["full_driver", "selected_range"] = "full_driver"
    start_sec: float | None = None
    end_sec: float | None = None
    subject_selection: Literal["largest", "center"] = "largest"


class AnimateCharacterOptions(BaseModel):
    look_source: Literal["auto_lora", "styled_lora", "exact_image"] = "auto_lora"
    cast_ref: str | None = None
    member_id: str | None = None
    exact_image_asset_id: str | None = None
    wardrobe: WardrobeSpec | None = None
    consistency: Literal["job"] = "job"


class WardrobeSpec(BaseModel):
    change_targets: list[Literal[
        "clothing", "jewelry", "bags", "footwear", "makeup", "hair", "other"
    ]] = []
    clothing_type: str = ""
    primary_color: str = ""
    material_pattern: str = ""
    jewelry: list[str] = []
    bags: list[str] = []
    footwear: str = ""
    makeup: str = ""
    hair: str = ""
    accessories: list[str] = []
    details: str = ""
    negative_constraints: str = ""
    garment_asset_ids: list[str] = []
    accessory_asset_ids: list[str] = []


class AnimateAudioOptions(BaseModel):
    mode: Literal["driver", "cast_voice", "none"] = "driver"
    voice_member_id: str | None = None
    script_policy: Literal["verbatim"] = "verbatim"
    timing: Literal["match_driver"] = "match_driver"


class AnimateLipSyncOptions(BaseModel):
    enabled: bool = False
    backend: Literal["latentsync", "musetalk"] = "latentsync"


class AnimateOutputOptions(BaseModel):
    generation_area: Literal["480p", "720p"] = "720p"
    export: Literal["generated", "scale_1080p", "vertical_1080x1920"] = "generated"
    preserve_aspect: Literal[True] = True
    target_fps: Literal["generated", 48] = "generated"
```

Server validation must enforce cross-field rules:

- `exact_image` requires `exact_image_asset_id`;
- generated look modes require cast and member with a valid LoRA;
- `cast_voice` requires a cast member with a configured voice;
- `replace` rejects pose-retarget options;
- `animate` hides/rejects replacement mask options;
- one target person/member only in v1;
- driver audio mode requires a usable audio stream;
- reference image and garment assets must be image asset types owned by the
  same operator/job staging scope; and
- no silent fallback to CPU, prompt-only garment transfer, another target
  person, or stale output.

## Media asset layer

All browser uploads, remote downloads, and server-file selections should
resolve to opaque `asset_id` values. The job request must never trust a raw
client-supplied absolute path.

Suggested endpoints:

```text
POST /api/assets/video/upload
POST /api/assets/video/from-url
GET  /api/assets/video/server-files
POST /api/assets/video/from-server-file
POST /api/assets/image/upload
GET  /api/assets/{asset_id}
GET  /api/assets/{asset_id}/media       # Range-enabled video/image response
GET  /api/assets/{asset_id}/thumbnail
DELETE /api/assets/{asset_id}           # staged/unclaimed assets only
```

Every write endpoint must use dashboard write authentication. Asset records
should include owner/session, kind, original name, content SHA-256, size,
normalized metadata, created/expiry timestamps, claimed job ID, and storage
path managed only by the server.

Security and reliability requirements:

- stream uploads; never read a whole video or large image into memory;
- verify file signatures and decode, not just extensions;
- cap image pixels to prevent decompression bombs;
- apply EXIF orientation, convert to RGB, and strip metadata from normalized
  images;
- allow only HTTP(S) remote URLs, limit redirects, block private/link-local/
  metadata IPs, and re-check every redirect to prevent SSRF;
- enforce byte, timeout, duration, and codec limits while downloading;
- restrict server files to configured roots and reject symlink escapes;
- atomically claim staged assets into a job;
- delete abandoned uploads after a TTL; and
- enforce free-disk and per-user staging quotas.

Browser `URL.createObjectURL()` provides immediate local previews. Persisted
video preview needs the authenticated, Range-enabled media endpoint. Social
page URLs that cannot play in a browser should show a server-generated
thumbnail and inspected metadata instead.

## Worker workflow and GPU lifecycle

Add a direct Animate workflow path rather than forcing these inputs through
transcribe -> adapt story -> plan shots -> render every shot.

Recommended stage sequence:

```text
validate request/readiness
  -> ingest and normalize driver; retain original audio
  -> inspect/select target person
  -> resolve or generate canonical character look
  -> LOOK APPROVAL
  -> split internal driver ranges and prepare pose/face/mask inputs
  -> optional MOTION/MASK PREVIEW APPROVAL
  -> transcribe/TTS and duration-fit cast audio, if selected
  -> unload image/voice/preprocess models and verify free VRAM
  -> load Wan Animate once
  -> generate all chunks
  -> unload/re-exec Wan Animate service
  -> optional lip-sync repair
  -> optional upscale/interpolation
  -> seam/output QA
  -> publish raw and final artifacts
```

For replacement mode, a preview gate should display the chosen person and mask
contact sheet before loading the 14B model. Reject empty, nearly full-frame,
leaking, or obviously wrong-person masks.

### Service readiness fix

Dashboard selection cannot depend on the global default adapter. Implement one
of these before exposing Animate Studio:

1. preferred: a small service supervisor/launcher that ensures port 8033 is
   running when an Animate job is claimed; or
2. simpler operational first release: start the lightweight deferred-loading
   Animate HTTP server whenever Animate is installed, regardless of the global
   default adapter.

The service must keep the heavy model unloaded until generation and must expose
model-loaded state and revision in `/health`.

### Caching, retry, and cancellation

Each stage writes a manifest containing input fingerprints, code/model
revisions, and output SHAs. A stage may be reused only if its manifest matches.

At minimum, include:

- driver content SHA and selected time range;
- normalization/FFmpeg version;
- mode and target track/person;
- canonical look SHA and wardrobe manifest SHA;
- Wan repository/model/checkpoint revision;
- preprocessing version and pose/mask settings;
- sampling/resolution/temporal-reference settings;
- audio transcript/voice/settings SHA;
- lip-sync backend/model/settings; and
- export/upscale settings.

Changing an option invalidates only that stage and its dependents. Retry APIs
must expose the nested Animate options rather than relying on the old flat
allowlist.

Cancellation must terminate process groups for preprocessing and FLUX.2
children, signal/cancel server inference, and force service re-exec if a model
cannot be cleanly interrupted. A cancelled job must not leave ONNX, SAM2,
FLUX.2, Wan, or lip-sync allocations resident.

## Premium polish without a frontend framework

Keep Jinja, vanilla JavaScript, and the existing self-hosted stylesheet. No
React/Vue, external font, animation library, or build system is required.

Lightweight visual changes:

- widen Animate content to about 1180 px;
- use 12–14 px cards, subtle borders, layered shadows, and a restrained header
  gradient;
- use radio cards and segmented controls for high-level choices;
- add small `Recommended`, `Uploaded`, `Unavailable`, and `GPU intensive`
  status pills;
- use consistent spacing and section numbers instead of dense field stacks;
- make the job summary sticky on wide screens;
- add upload progress for large driver files;
- show inline field-level errors and readiness reasons;
- use skeleton/disabled states during URL inspection and upload;
- provide `focus-visible`, keyboard navigation, semantic labels, sufficient
  contrast, and `prefers-reduced-motion`; and
- define the currently missing `--accent` and `--surface2` CSS variables.

Use a dedicated `animate.js` module. Do not add more Animate-specific inline
JavaScript to `job_new.html`. Increment the static asset cache version when the
page ships.

## Change map

Expected files/components:

| Area | Primary changes |
|---|---|
| Navigation/page | `services/templates/base.html`, new `services/templates/animate_new.html`, route in `services/dashboard_api.py` |
| Frontend behavior | new `services/static/animate.js`, focused additions to `services/static/app.css` |
| Generic form | `services/templates/job_new.html`: replace expanded Animate panel with link to Animate Studio |
| Request models | `core/models/dashboard.py`: versioned direct Animate contract and cross-field validation |
| Assets | new asset model/repository helpers and authenticated upload/URL/local/media routes |
| Worker | `services/dashboard_worker.py`: dispatch `wan_animate_direct` and preserve nested config snapshot |
| Workflow | new direct Animate orchestration module or a clearly separated path in `core/workflow.py` |
| Character look | extend text wardrobe requests; add canonical look plan/approval; generalize uploaded character image handling |
| FLUX.2 editing | new multi-reference edit adapter after GPU spike; pin Musubi revision |
| Audio | extract driver audio before normalization; explicit driver/cast/none path |
| Export | distinguish preserve-generated, scaling, interpolation, and AI enhancement |
| Runtime | ensure Animate service starts for dashboard jobs; expose readiness and model revision |
| Detail/history | show Animate badge, driver metadata, target/look, audio, lip-sync, output, and artifacts |
| Tests/docs | model/API/UI/security/workflow tests, `.env.example`, README, deployment/readiness docs |

## Delivery phases

### P0 — Contract and runtime foundations

- Confirm the product questions below.
- Add `workflow_kind="wan_animate_direct"` and the nested v1 schema.
- Add the opaque asset abstraction and staging/claim lifecycle.
- Fix Animate service startup/readiness for per-job dashboard selection.
- Add stage manifests/fingerprints and downstream invalidation rules.
- Make preprocessing/server cancellation release GPU resources.
- Pin the Wan, Musubi, and relevant model revisions.

Exit criteria: a mocked direct Animate job can validate, queue, resume, retry,
and cancel without passing through the generic story pipeline or using raw
client paths.

### P1 — Functional Animate Studio

- Add sidebar entry and `/animate/new` page.
- Add both mode cards.
- Add driver URL, browser upload, and allowlisted server-file inputs.
- Add cast and single target-member selection.
- Support **Auto LoRA** and **Exact character image**.
- Generate/approve one canonical look and reuse it byte-identically.
- Add driving-audio/cast-voice/no-audio controls.
- Add lip-sync toggle/backend and factual output controls.
- Add minimal previews, validation metadata, upload progress, sticky summary,
  and basic responsive/accessibility treatment.
- Add Animate metadata/artifacts to job history/detail.
- Keep the generic form as a link-only bridge.

Exit criteria: both modes work end-to-end with auto-LoRA or uploaded reference,
on a real Hopper GPU, without stale-cache, wrong-audio, model-co-residency, or
service-startup failures.

### P2 — Designed complete look

- Add structured complete-look fields, explicit change scope, and default-look
  generation.
- Add clothing/styling-detail image assets.
- Build and validate the FLUX.2 multi-reference edit adapter.
- Add edit/regenerate controls to Look approval.
- Record prompts, seeds, model/LoRA hashes, and source references.
- Add clothing, jewelry, bag, footwear, makeup, and overall-look fidelity QA
  warnings.

Exit criteria: text-only complete-look control is reliable; image-guided styling
is enabled only when GPU compatibility, identity retention, and licensing are
explicitly verified. No prompt-only fallback is labeled as reference transfer.

### P3 — Visual refinement and operator ergonomics

- Add polished uploaded-character/clothing thumbnails with replace/remove.
- Add target-person and replacement-mask contact-sheet picker.
- Complete premium spacing, status pills, empty/loading states, and mobile
  navigation.
- Add side-by-side raw Animate, lip-synced, and enhanced previews.
- Add saved Animate presets if operators repeat configurations.

Exit criteria: the page is visually consistent, keyboard-usable, responsive,
and makes every expensive or destructive choice visible before queueing.

### P4 — Long-video and advanced features

- Per-range review/editing and continuity controls.
- Optional per-shot outfit overrides.
- Multi-person tracking/picker improvements.
- Adapted cast script with duration-aware rewrite.
- Advanced AI enhancement choices and output presets.

These are not required to launch the clearer v1 dashboard.

## Test plan

### Unit and schema tests

- all three look strategies and required-field combinations;
- cast/member/LoRA/voice availability;
- mode-specific advanced controls;
- driver-audio availability;
- selected range and duration limits;
- single-target enforcement;
- output/aspect/FPS contract; and
- semantic stage invalidation when each option changes.

### API and security tests

- unauthenticated upload/URL/local-file writes are rejected;
- extension spoof, corrupt media, decompression bomb, and oversize cleanup;
- remote private-IP, redirect-to-private-IP, non-HTTP, timeout, and byte caps;
- local path traversal and symlink escape;
- cross-user/unclaimed/expired asset IDs;
- Range responses and media authorization; and
- prepared-directory/path escape at the Animate service boundary.

### Workflow tests

- canonical reference SHA is identical across 3+ chunks;
- driver audio is extracted before audio-stripping normalization;
- cast TTS stays within bounded timing tolerance;
- Wan unload completes before lip-sync starts;
- lip-sync unload completes before enhance/export;
- changing look/audio/output invalidates only downstream artifacts;
- retry from each approval/stage;
- cancellation kills child processes and clears service GPU state; and
- no silent fallback when a requested model/voice/reference is unavailable.

### Real Hopper smoke matrix

Run both `animate` and `replace` across:

- auto LoRA, exact image, structured text complete look, and—when ready—
  image-directed complete look;
- 480p first, then 720p;
- driving audio, cast voice, and no audio;
- lip-sync off/on;
- generated export and upscale;
- a short clip and a clip crossing multiple 77-frame boundaries;
- portrait, landscape, square, VFR, rotated MOV, HDR, silent, corrupt, and
  multi-person drivers; and
- cancel/retry during preprocess, Wan inference, lip-sync, and upscale.

Record runtime, peak VRAM, residual VRAM after unload, disk expansion, seam
quality, and failure reason.

### Human acceptance rubric

- correct target person;
- cast identity retention;
- clothing, jewelry, bag, footwear, makeup, and accessory fidelity;
- body and facial-motion fidelity;
- background retention in replacement mode;
- face/hands quality;
- cross-chunk wardrobe/identity/color continuity;
- lip-sync quality and AV timing; and
- no unexpected crop, stretch, resolution, or FPS change.

Automated/VLM scores can warn, but should not be the sole approval criterion.

## Launch gates

Animate Studio can leave beta only when:

1. per-job selection reliably starts or reaches the Animate service;
2. all three driver input paths use safe asset IDs;
3. both modes pass real-Hopper smoke tests;
4. selected character/look is frozen and traceable by SHA/model revision;
5. audio labels match the actual media used;
6. “leave as generated” truly preserves dimensions/FPS;
7. cancellation clears GPU work;
8. retry cannot reuse stale outputs;
9. unavailable FLUX.2 edit, LoRA, voice, lip-sync, or upscale options are
   disabled with a reason; and
10. no workflow silently substitutes another model or behavior.

## Confirmed decisions

Confirmed:

1. Animate Studio is a **direct transformation** flow; the driving video is the
   content and timeline.
2. **Source audio** is the driving video's audio. **Cast audio** re-voices the
   same transcript with the selected cast member.
3. v1 targets **one selected cast member / one person track** per job.
4. Upscale preserves aspect ratio; 9:16 1080x1920 is an optional preset.

5. FLUX.2 Dev remains the cast LoRA training, still-generation, and experimental
   garment-edit model for this testing-only workflow. Any later commercial or
   production use requires a separate licensing review before rollout.
