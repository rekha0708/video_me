# Synthetic Creator Pipeline — Orchestration Build Plan

> **Audience:** an autonomous AI coding agent (e.g. Claude Code, Gemini CLI, or similar).
> **Goal:** build the *orchestration layer* that turns a social-media video link into an
> original, character-performed short video — with every model behind a swappable
> interface, plus self-critic, self-healing, and engagement-driven learning.
> **Scope of THIS document:** the orchestration framework and its contracts. Individual
> models (ASR, LLM, TTS, video, etc.) are plugged in as adapters; this plan asks for one
> *reference adapter per capability* so the system runs end-to-end, and leaves room to
> swap in better models later.

---

## 0. How the agent should use this document

1. Read the whole document before writing code. Do not start with model integration —
   start with the framework (interfaces, registry, router, job state). Models come last.
2. Build in the phase order in Section 9. Each phase has acceptance criteria; do not
   advance until they pass.
3. Treat every capability contract in Section 4 as fixed. Adapters conform to the contract;
   the pipeline never imports a concrete model class directly.
4. Where this plan names a specific model, treat it as the *default reference adapter*, not
   a hard dependency. The point of the architecture is replaceability.
5. Obey the guardrails in Section 8 as hard requirements, not suggestions. They are wired in
   as pipeline steps and validation, not left to operator discipline.
6. Ask the human operator before: choosing the workflow engine (Section 3), provisioning any
   paid cloud resource, or publishing to a live social account for the first time.

---

## 1. System overview

The system is a **DAG of pipeline stages** coordinated by an orchestrator. Each stage calls a
**capability** (e.g. "transcribe", "generate_video"). Each capability has one or more
**adapters** (concrete model implementations) registered in a **model registry**. A **router**
selects which adapter runs for a given job. Two feedback loops wrap the pipeline: a
**critic loop** (regenerate failing output) and a **learning loop** (tune future choices from
real engagement).

Pipeline stages, in order:

| # | Stage | Capability used | Default reference adapter |
|---|-------|-----------------|---------------------------|
| 1 | Ingest | `fetch_media`, `extract_streams` | `yt-dlp` + `ffmpeg` |
| 2 | Understand | `transcribe`, `separate_audio`, `analyze_content` | Whisper-family + Demucs + open LLM/VLM |
| 3 | Adapt script | `adapt_script` | open LLM (Qwen / Llama / DeepSeek / Mistral class) |
| 4 | Generate assets | `render_character`, `synthesize_voice`, `build_motion` | image-diffusion + LoRA, open TTS, pose/driver |
| 5 | Synthesize video | `generate_video`, `lip_sync` | Wan 2.7 (+ dedicated lip-sync) |
| 6 | Produce audio | `synthesize_voice`, `mix_audio` | open TTS + `ffmpeg` |
| 7 | Assemble | `assemble_video`, `caption` | `ffmpeg` + timestamps from Stage 2 |
| 8 | Critic gate | `critique` | open VLM + automated checks |
| 9 | Publish | `publish` | platform API client |
| 10 | Learn | `collect_metrics`, `update_policy` | metrics store + bandit/optimizer |

---

## 2. Design principles (non-negotiable)

- **Contracts over implementations.** The pipeline depends on abstract capability interfaces,
  never on a concrete model. Swapping a model = writing one new adapter + a config change.
- **Idempotent, resumable stages.** Every stage is keyed by `job_id` + `stage_name` and writes
  its output to durable storage. Re-running a completed stage returns the cached artifact.
- **Everything is observable.** Every stage emits structured logs, timings, cost, and the
  adapter used. No silent failures.
- **Fail to a fallback, not to a crash.** If an adapter fails or degrades, the router selects
  the next eligible adapter for that capability (self-healing).
- **Generate candidates, let the critic pick.** Quality emerges from generate→evaluate→refine,
  not from trusting a single generation.
- **Config-driven.** Channel profile, character bible, model selection, and thresholds live in
  config/data, not in code.

---

## 3. Tech stack (recommended defaults — confirm with operator)

- **Language:** Python 3.11+ (ecosystem alignment with all ML models).
- **Orchestration engine:** choose one and confirm with operator:
  - *Temporal* — best for durable, long-running, retry-heavy workflows (recommended).
  - *Prefect* or *Dagster* — lighter, good DX, fine for early phases.
  - *Custom asyncio DAG* — only acceptable for the Phase 1 MVP; migrate later.
- **Queue / async work:** the engine's native queue, or Redis + a worker pool for GPU jobs.
- **Object storage:** S3-compatible (MinIO locally, any S3 provider in cloud) for media/artifacts.
- **Metadata DB:** PostgreSQL (jobs, stage results, metrics, model registry, learning state).
- **Model serving:** each GPU model runs as its own containerized service behind an HTTP/gRPC
  adapter, so it can be scaled, replaced, or moved to rented GPUs independently.
- **Config:** Pydantic settings + YAML profiles. Validate all config at startup.
- **Packaging:** one repo, `uv` or `poetry`, fully containerized (`docker compose` for local).

GPU note: build and validate Phases 1–3 on *rented* cloud GPUs. Do not buy hardware until the
pipeline produces acceptable output end-to-end.

---

## 4. Capability interfaces (the core contracts)

Define these as abstract base classes in `core/capabilities/`. Every adapter implements exactly
one capability. Inputs and outputs are Pydantic models so they validate and serialize cleanly.

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class Capability(ABC):
    name: str                      # e.g. "transcribe"
    version: str                   # adapter version
    async def health(self) -> "HealthStatus": ...   # for self-healing
    async def estimate_cost(self, req: BaseModel) -> "CostEstimate": ...

class Transcribe(Capability):
    @abstractmethod
    async def run(self, req: "TranscribeRequest") -> "TranscribeResult": ...
    # TranscribeRequest: audio_uri
    # TranscribeResult: segments[{text, start, end, speaker?}], language, full_text

class AnalyzeContent(Capability):
    @abstractmethod
    async def run(self, req: "AnalyzeRequest") -> "ContentMetadata": ...
    # ContentMetadata schema is defined in Section 5.

class AdaptScript(Capability):
    @abstractmethod
    async def run(self, req: "AdaptScriptRequest") -> "Script": ...
    # req carries transcript, ContentMetadata, ChannelProfile, CharacterBible, mode
    # mode ∈ {verbatim, adapted, transformed}

class RenderCharacter(Capability):   # consistent visual identity (uses character LoRA)
    @abstractmethod
    async def run(self, req: "RenderCharacterRequest") -> "ImageSet": ...

class SynthesizeVoice(Capability):   # designed synthetic voice ONLY (see guardrails)
    @abstractmethod
    async def run(self, req: "VoiceRequest") -> "AudioTrack": ...

class GenerateVideo(Capability):     # e.g. Wan 2.7 adapter
    @abstractmethod
    async def run(self, req: "VideoRequest") -> "VideoClip": ...

class LipSync(Capability):
    @abstractmethod
    async def run(self, req: "LipSyncRequest") -> "VideoClip": ...

class AssembleVideo(Capability):
    @abstractmethod
    async def run(self, req: "AssembleRequest") -> "FinalVideo": ...

class Critique(Capability):
    @abstractmethod
    async def run(self, req: "CritiqueRequest") -> "CritiqueResult": ...
    # CritiqueResult: scores{dimension: 0..1}, verdict ∈ {pass, regenerate, reject},
    #                 reasons[], suggested_param_overrides{}

class Publish(Capability):
    @abstractmethod
    async def run(self, req: "PublishRequest") -> "PublishResult": ...
```

Each adapter lives in `adapters/<capability>/<adapter_name>.py` and declares the capability it
satisfies, its version, its resource needs (CPU/GPU/VRAM), and its config schema.

---

## 5. Data models

Define in `core/models/`. These are the shared schemas the whole pipeline reads and writes.

**ChannelProfile** (per social account, reusable): `genre_content`, `genre_music`,
`target_audience`, `tone`, `format` (talking_head | animated_character | dance_lifestyle | other),
`aspect_ratio`, `target_length_sec`, `posting_cadence`, `disclosure_label_required` (bool, default true).

**CharacterBible** (per character, reusable): `name`, `visual_descriptor`, `lora_ref`,
`wardrobe`, `voice_profile_ref`, `signature_expressions[]`, `signature_moves[]`, `personality`.
Hard rule: `is_original_synthetic: bool` must be `true`; the system refuses real-person likeness/voice cloning.

**ContentMetadata** (output of Stage 2 — make this the explicit target for "genre and other details"):
`content_genre`, `music_genre`, `topic`, `tone`, `hook`, `structure[]` (segment beats),
`pacing`, `visual_style`, `length_sec`, `call_to_action`, `language`.

**Script**: `mode`, `lines[{text, start, end, expression?, motion?}]`, `caption_text`, `source_rights` (see guardrails).

**Job**: `job_id`, `source_url`, `channel_profile_ref`, `character_bible_ref`, `script_mode`,
`status`, `stage_results{stage: artifact_uri + metadata}`, `rights_cleared: bool`, `created_at`, `cost_total`.

---

## 6. Model registry & router

`core/registry.py` and `core/router.py`.

**Registry**: maps `capability_name -> [AdapterRecord]`. Each `AdapterRecord` holds the adapter
class, version, enabled flag, priority, resource profile, cost profile, and a rolling quality/health
score updated by the critic and learning loops. Registry is backed by the DB so adapters can be
enabled/disabled without redeploy.

**Router**: `select(capability, job, context) -> Adapter`. Selection logic, in order:
1. Filter to enabled adapters whose `health()` is OK and whose resources are available.
2. If config pins an adapter for this capability/profile, use it.
3. Otherwise rank by a score combining quality, cost, and latency (weights are config).
4. Return an ordered list; the executor tries them in order (this gives self-healing fallback).

The router must be pure/deterministic given its inputs so routing decisions are reproducible and
logged. Learning (Section 7c) updates the scores the router reads — it does not bypass the router.

---

## 7. The three "self-" subsystems

### 7a. Self-critic (build in Phase 2)
After Stage 7 produces a candidate, Stage 8 calls `Critique`. Implement as:
- **Automated checks** (cheap, deterministic): lip-sync confidence, duration vs target, resolution,
  caption presence, audio loudness, NSFW/policy classifier, blank/black-frame detection.
- **VLM judgment** (model-based): score visual quality, on-brand fit, on-genre fit, character
  consistency, against a written rubric stored in config.
- **Verdict logic**: `pass` → publish path; `regenerate` → loop back to the failing stage with
  `suggested_param_overrides`, up to `max_regenerations` (config, e.g. 3); `reject` → human queue.
- Persist every critique. These scores feed both the router (which adapter produced passes) and
  the learning loop.

### 7b. Self-healing (build in Phase 3, partially in Phase 1)
- Per-adapter `health()` checks on a schedule; unhealthy adapters are skipped by the router.
- Retries with exponential backoff on transient failures (config: attempts, base delay).
- Automatic fallback to the next adapter in the router's ordered list on hard failure.
- Circuit breaker per adapter: after N consecutive failures, disable and alert.
- Dead-letter queue for jobs that exhaust all options; surfaced to the operator.
- Every stage idempotent + resumable so a recovered worker re-enters cleanly.

### 7c. Self-learning (build in Phase 4) — scoped honestly
Do **not** attempt continuous retraining of large generative models. Learn at the decision layer:
- **Metrics ingestion**: pull engagement (views, watch_time, retention, likes, shares) per post on
  a schedule into the metrics store, keyed back to the `Job` and the choices it made.
- **Choice optimization**: model creative decisions (hook style, length, character variant, music,
  posting time, which adapter) as arms in a contextual multi-armed bandit; optimize against a
  defined reward (e.g. retention-weighted engagement). Run as periodic batch, not real-time.
- **A/B harness**: ability to publish controlled variants and compare.
- **Cheap-component refresh** (manual-trigger, human-reviewed): periodically update prompt templates,
  the script LLM's few-shot examples, and character LoRAs based on what wins. Gate behind operator approval.
- Feed bandit outputs into the router scores and the Stage 0/3 defaults. Keep a full audit trail.

---

## 8. Guardrails (hard requirements, wired into the pipeline)

These are validation steps, not policy docs. The pipeline must enforce them.

1. **Rights checkpoint.** A `Job` cannot pass Stage 3 unless `rights_cleared == true`. Sourcing
   must be content the operator owns, has licensed, that is public-domain, or is genuinely
   transformed. Default `script_mode` to `transformed`/`adapted`; `verbatim` requires an explicit
   rights record on the job. Store `source_rights` on every `Script`.
2. **Original synthetic characters only.** Reject any `CharacterBible` with `is_original_synthetic != true`.
   No real-person face or voice cloning. The voice adapter must use a *designed* synthetic voice.
3. **AI-content disclosure.** Stage 9 must apply the platform's AI/synthetic-media label when
   `disclosure_label_required` is true (default true). Publishing without it is a blocked action.
4. **Platform ToS.** Note in the ingest adapter that downloading from some platforms may conflict
   with their terms; make source policy a configurable, logged decision.
5. **Policy safety gate.** The critic's automated checks include a content-safety classifier; a
   `reject` verdict on safety always routes to a human, never auto-publishes.

If a guardrail blocks a job, the system records *why* and routes to the operator queue. It never
silently works around a guardrail.

---

## 9. Build phases (do these in order; gate on acceptance criteria)

### Phase 0 — Skeleton
- Repo, containerization, config loading, DB + object storage wired, logging/observability baseline.
- Capability ABCs (Section 4) and data models (Section 5) defined and validated.
- **Acceptance:** `docker compose up` runs; a no-op job flows through an empty DAG and is recorded
  in the DB with structured logs.

### Phase 1 — Linear MVP (one genre, one character, fixed models, manual publish)
- Implement one reference adapter per capability (Sections 1/4) with hardcoded routing.
- End-to-end: link → ingest → understand → adapt (transformed) → generate → assemble → file output.
- Publish is manual (writes a file the operator reviews/posts).
- Basic retries; no critic loop yet.
- **Acceptance:** a real link produces a watchable, captioned, correctly-formatted short video using
  the original character, with the rights checkpoint and disclosure flag enforced.

### Phase 2 — Critic loop
- Implement `Critique` (automated checks + VLM) and the generate→evaluate→regenerate loop.
- Candidate generation (N variants) + critic selection.
- **Acceptance:** failing candidates are caught and regenerated automatically; only passing output
  reaches the publish path; every critique is persisted.

### Phase 3 — Orchestration framework + self-healing
- Introduce the registry + router; move all adapters behind it; remove hardcoded routing.
- Add ≥2 adapters for at least one capability and prove hot-swap via config.
- Implement health checks, fallback, circuit breakers, dead-letter queue.
- **Acceptance:** disabling the primary video adapter mid-run causes automatic fallback with no job
  loss; swapping an adapter requires only a config change.

### Phase 4 — Learning loop
- Metrics ingestion, bandit-based choice optimization, A/B harness, router-score feedback.
- Operator-gated refresh of prompts/few-shots/LoRAs.
- **Acceptance:** the system measurably shifts choices toward higher-reward variants over a batch of
  posts, with a full audit trail; no large-model retraining is required for this to work.

### Phase 5 — Automation & scale
- Scheduling, automated (labeled) publishing after operator sign-off, multi-character/multi-channel,
  GPU autoscaling on rented capacity, cost dashboards.
- **Acceptance:** the system runs a posting cadence unattended within guardrails, with alerts and a
  kill switch.

---

## 10. Suggested repository layout

```
repo/
  core/
    capabilities/        # abstract base classes (the contracts)
    models/              # Pydantic data models (Section 5)
    registry.py
    router.py
    executor.py          # runs a stage: select adapter -> try ordered list -> persist
    workflow.py          # the DAG definition
    storage.py           # object storage + artifact caching
    observability.py
    config.py
  adapters/
    fetch_media/         # yt-dlp adapter, ...
    transcribe/          # whisper adapter, ...
    analyze_content/
    adapt_script/
    render_character/
    synthesize_voice/
    generate_video/      # wan_2_7 adapter, ...
    lip_sync/
    assemble_video/
    critique/
    publish/
  subsystems/
    critic/              # rubric, automated checks
    healing/             # health, retries, circuit breaker, DLQ
    learning/            # metrics, bandit, A/B
  guardrails/            # rights, character validation, disclosure, safety gate
  services/              # containerized model servers (one per GPU model)
  config/                # YAML profiles: channels, characters, model selection, thresholds
  tests/
  docker-compose.yml
  README.md
```

---

## 11. Testing & quality

- **Contract tests** per capability: any adapter must pass the same suite for its capability.
- **Golden-path integration test**: a fixture link → expected artifact shape at each stage.
- **Fallback test**: force primary adapter failure, assert clean fallback.
- **Guardrail tests**: assert jobs block correctly on missing rights, non-original characters,
  missing disclosure, and safety rejects.
- **Determinism test**: router returns the same ordered selection for the same inputs.
- Mock all external/model calls in unit tests; run real models only in the integration suite on GPU.

---

## 12. Open questions for the operator (resolve before/early in Phase 1)

1. **Video format/style** — talking_head, animated_character, dance_lifestyle, or other? This
   decides whether `lip_sync`, pose/driver, or pure text-to-video is the primary path in Stage 5.
2. **Meaning of "genre"** — content genre, music genre, or both? Decides which analyzers Stage 2 needs.
3. **Workflow engine choice** (Section 3) — Temporal vs Prefect/Dagster vs custom for the MVP.
4. **Target platform(s)** — sets the `publish` adapter and the disclosure-label mechanism.
5. **Reward definition** for the learning loop — what does "best" mean (retention? shares? follows?).

---

*End of plan. Build the framework first, plug models in last, gate every phase on its acceptance
criteria, and never route around a guardrail.*
