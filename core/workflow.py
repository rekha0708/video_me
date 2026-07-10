import asyncio
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Any, Literal

from core.cast_params import CastMemberParams, CastPairParams, load_cast_pair_params, load_cast_params
from core.config import AppConfig, load_app_config
from core.executor import StageError, check_rights, run_stage
from core.gpu_sequencer import (
    ensure_video_model_unloaded,
    free_comfyui,
    is_managed,
    prepare_video_model,
    prepare_voice_model,
    unload_ollama_model,
)
from core.models.capabilities import (
    AnalyzeRequest,
    AnalyzeVisualsRequest,
    AssembleRequest,
    AudioTrack,
    CritiqueRequest,
    CritiqueResult,
    FetchMediaRequest,
    FinalVideo,
    ImageApprovalRequest,
    ImageCritiqueRequest,
    ImageCritiqueResult,
    ImageSet,
    LipSyncRequest,
    PlanShotsRequest,
    PublishRequest,
    RenderCharacterRequest,
    TranscribeRequest,
    VideoClip,
    VideoRequest,
    VisualContext,
    VoiceRequest,
)
from core.models.capabilities import AdaptScriptRequest
from core.models.content import Script, Shot, Storyboard
from core.models.job import Job, JobStatus
from core.models.profile import Cast, CastMember
from core.observability import log_event
from core.storage import (
    ArtifactStore,
    JobRepository,
    completed_stage,
    create_artifact_store,
    create_job_store,
)

Phase = Literal["transcribe", "script_plan", "plan", "render", "assemble", "all"]


@dataclass
class RunOptions:
    """Controls which stages run and whether completed stages are skipped.

    phase:
      "transcribe" — fetch → transcribe → analyze_content, then stop (saves artifacts).
      "script_plan"— load cached transcribe artifacts, then adapt_script → plan_shots + approval.
      "plan"       — fetch → transcribe → analyze → adapt → plan_shots + approval, then stop.
      "render"     — shot loop only (render + voice + video + lip_sync for each shot).
      "assemble"   — assemble_video → publish only.
      "all"        — full pipeline (default).
    resume:      skip stages whose artifact JSON already exists on disk.
    only_shot:   in the shot loop, process only this shot ID (e.g. "s01").
    language:    BCP-47 language code for this run — "en" or "hi".
    user_images: member_id → image path. When set, render_character + VLM
                 critique are skipped entirely and each shot defaults to its
                 speaker's reference image (story_images input mode).
    stage_hook:  called as stage_hook(stage_name, event_type, shot_id=..., message=...).
                 shot_id/message are optional keyword extras (see
                 core/workflow.py's per-shot calls and
                 services/dashboard_worker.py's _make_stage_hook) used to
                 surface shot-level progress on the dashboard; the two
                 positional args alone are enough for ordinary stage events.
    render_mode: "full" (default) — normal pipeline.
                 "source_audio" — skip adapt_script + TTS; use source audio sliced per segment.
                 "re_voice" — TTS + adapt_script, but shot durations match source segment timing.
    """
    phase: Phase = "all"
    resume: bool = False
    only_shot: str | None = None
    language: str = "en"
    render_mode: str = "full"
    stage_hook: Callable[..., None] | None = None
    error_hook: Callable[[str, Exception], Any] | None = None
    user_images: dict[str, str] | None = None

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ Phase 0 compat

NOOP_STAGES = ("create_job", "noop_dag", "record_result")


async def run_noop_job(
    source_url: str = "noop://phase-0",
    app_config: AppConfig | None = None,
) -> Job:
    config = app_config or load_app_config()
    settings = config.settings
    artifacts = create_artifact_store(settings)
    jobs = create_job_store(settings)

    job = Job(
        source_url=source_url,
        channel_profile_ref=config.channel_profile.id,
        cast_ref=config.cast.id,
        rights_cleared=False,
    )
    job.status = JobStatus.RUNNING
    jobs.save_job(job)
    log_event(logger, "job_started", job_id=job.job_id, workflow_engine=settings.workflow_engine)

    for stage_name in NOOP_STAGES:
        log_event(logger, "stage_started", job_id=job.job_id, stage=stage_name, adapter="noop")
        artifact = artifacts.put_json(
            job.job_id,
            stage_name,
            {
                "job_id": job.job_id,
                "stage": stage_name,
                "status": "completed",
                "note": "Phase 0 no-op stage recorded successfully.",
            },
        )
        result = completed_stage(stage_name, artifact)
        job.stage_results[stage_name] = result
        jobs.save_stage_result(job.job_id, result)
        jobs.save_job(job)
        log_event(logger, "stage_completed", job_id=job.job_id, stage=stage_name, adapter="noop")

    job.status = JobStatus.COMPLETED
    jobs.save_job(job)
    log_event(logger, "job_completed", job_id=job.job_id)
    return job


# ------------------------------------------------------------------ Phase 1 pipeline

@dataclass
class _Adapters:
    """Holds one concrete adapter instance per capability for a single job run."""
    fetch_media: object
    transcribe: object
    analyze: object
    analyze_visuals: object
    adapt: object
    plan: object
    plan_critique: object
    overlays: object
    approval: object
    image_critique: object
    image_approval: object
    render: object
    voice: object
    video: object
    lipsync: object
    assemble: object
    critique: object
    publish: object
    ffmpeg_bin: str = field(default="ffmpeg")
    ffprobe_bin: str = field(default="ffprobe")


def _make_render_adapter(s, work_dir: Path):
    """Select render_character adapter based on VIDEO_ME_RENDER_ADAPTER env var."""
    if s.render_adapter == "musubi_flux":
        from adapters.render_character.musubi_flux_adapter import MusubiFluxAdapter
        return MusubiFluxAdapter(
            work_dir=work_dir / "renders",
            lora_dir=s.lora_dir,
            num_images=s.image_candidates,
            allow_placeholder_lora=s.render_allow_placeholder_lora,
        )
    if s.render_adapter == "comfyui_flux":
        from adapters.render_character.comfyui_flux_adapter import ComfyUIFluxAdapter
        return ComfyUIFluxAdapter(
            work_dir=work_dir / "renders",
            base_url=s.comfyui_base_url,
            lora_dir=s.lora_dir,
            num_images=s.image_candidates,
            allow_placeholder_lora=s.render_allow_placeholder_lora,
        )
    from adapters.render_character.diffusion_adapter import DiffusionRenderAdapter
    return DiffusionRenderAdapter(
        work_dir=work_dir / "renders",
        base_url=s.sd_base_url,
        lora_dir=s.lora_dir,
        allow_placeholder_lora=s.render_allow_placeholder_lora,
    )


def _make_video_adapter(s, work_dir: Path):
    """Select generate_video adapter based on VIDEO_ME_VIDEO_ADAPTER env var."""
    if s.video_adapter == "ltx":
        from adapters.generate_video.ltx_adapter import LtxAdapter
        return LtxAdapter(work_dir=work_dir / "video" / "ltx", base_url=s.ltx_base_url)
    if s.video_adapter == "wan_s2v":
        from adapters.generate_video.wan_s2v_adapter import WanS2VAdapter
        return WanS2VAdapter(work_dir=work_dir / "video" / "wan_s2v", base_url=s.wan_s2v_base_url)
    from adapters.generate_video.wan_adapter import WanAdapter
    return WanAdapter(work_dir=work_dir / "video" / "wan", base_url=s.wan_base_url)


def _make_lipsync_adapter(s, work_dir: Path):
    """Select lip_sync repair adapter for video backends without native sync."""
    namespace = f"{s.video_adapter}_{s.lipsync_adapter}"
    if s.lipsync_adapter == "latentsync":
        from adapters.lip_sync.latentsync_adapter import LatentSyncAdapter
        return LatentSyncAdapter(
            work_dir=work_dir / "synced" / namespace,
            base_url=s.latentsync_base_url,
            inference_steps=s.latentsync_inference_steps,
            guidance_scale=s.latentsync_guidance_scale,
        )
    from adapters.lip_sync.lip_sync_adapter import LipSyncAdapter
    base_url = getattr(s, "musetalk_base_url", None) or s.lipsync_base_url
    return LipSyncAdapter(work_dir=work_dir / "synced" / namespace, base_url=base_url)


def _make_tts_adapter(s, work_dir: Path):
    """Select synthesize_voice adapter based on VIDEO_ME_TTS_ADAPTER env var."""
    if s.tts_adapter == "fish_s2":
        from adapters.synthesize_voice.fish_s2_adapter import FishS2TtsAdapter
        return FishS2TtsAdapter(
            work_dir=work_dir / "audio",
            base_url=s.fish_s2_base_url,
            voice_dir=s.voice_dir,
        )
    from adapters.synthesize_voice.tts_adapter import TtsAdapter
    return TtsAdapter(
        work_dir=work_dir / "audio",
        base_url=s.tts_base_url,
        voice_dir=s.voice_dir,
    )


def _make_adapters(
    config: AppConfig,
    work_dir: Path,
    *,
    approval_overrides: dict | None = None,
) -> _Adapters:
    """Instantiate all Phase 1 adapters with job-scoped work directories."""
    from adapters.adapt_script.llm_adapter import LlmAdaptScriptAdapter
    from adapters.analyze_content.llm_adapter import LlmAnalyzeAdapter
    from adapters.analyze_visuals.vlm_adapter import VlmAnalyzeVisualsAdapter
    from adapters.assemble_video.ffmpeg_adapter import FfmpegAssembleAdapter
    from adapters.critique.vlm_adapter import VlmCritiqueAdapter
    from adapters.fetch_media.ytdlp_adapter import YtDlpAdapter
    from adapters.plan_shots.llm_adapter import LlmPlanShotsAdapter
    from adapters.publish.manual_adapter import ManualPublishAdapter
    from adapters.render_overlays.matplotlib_adapter import MatplotlibOverlayAdapter
    from adapters.synthesize_voice.tts_adapter import TtsAdapter  # Chatterbox fallback
    from adapters.transcribe.whisper_adapter import WhisperAdapter

    from adapters.approval.image_approval_adapter import ImageApprovalAdapter
    from adapters.approval.web_approval_adapter import WebApprovalAdapter
    from adapters.critique.image_critique_adapter import VlmImageCritiqueAdapter
    from adapters.critique.plan_critique_adapter import LlmPlanCritiqueAdapter

    s = config.settings
    return _Adapters(
        fetch_media=YtDlpAdapter(work_dir=work_dir / "fetch_media"),
        transcribe=WhisperAdapter(
            model_size=s.whisper_model_size,
            device=s.whisper_device,
            compute_type=s.whisper_compute_type,
        ),
        analyze=LlmAnalyzeAdapter(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
        ),
        analyze_visuals=VlmAnalyzeVisualsAdapter(
            model=s.analyze_visuals_model,
            base_url=s.analyze_visuals_base_url,
            api_key=s.analyze_visuals_api_key,
            work_dir=work_dir / "visuals",
            ffmpeg_bin=s.ffmpeg_bin,
            ffprobe_bin=s.ffprobe_bin,
            max_frames=s.visual_max_frames,
        ),
        adapt=LlmAdaptScriptAdapter(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
        ),
        plan=LlmPlanShotsAdapter(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            max_shot_duration_sec=s.max_shot_duration_sec,
        ),
        plan_critique=LlmPlanCritiqueAdapter(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
        ),
        overlays=MatplotlibOverlayAdapter(work_dir=work_dir / "overlays"),
        approval=(approval_overrides or {}).get("approval") or WebApprovalAdapter(
            work_dir=work_dir,
            port=s.approval_port,
            timeout_hours=s.approval_timeout_hours,
            auto_approve=s.auto_approve_plan,
        ),
        image_critique=VlmImageCritiqueAdapter(
            model=s.image_critique_model,
            base_url=s.image_critique_base_url,
            api_key=s.image_critique_api_key,
            feedback_log_dir=s.feedback_log_dir,
        ),
        image_approval=(approval_overrides or {}).get("image_approval") or ImageApprovalAdapter(
            work_dir=work_dir,
            feedback_log_path=Path(s.feedback_log_dir) / "critique_feedback.jsonl",
            port=s.approval_port,            # same port — gates run sequentially
            timeout_hours=s.approval_timeout_hours,
            auto_approve=s.auto_approve_images,
        ),
        render=_make_render_adapter(s, work_dir),
        voice=_make_tts_adapter(s, work_dir),
        video=_make_video_adapter(s, work_dir),
        lipsync=_make_lipsync_adapter(s, work_dir),
        assemble=FfmpegAssembleAdapter(
            work_dir=work_dir / "assembled",
            ffmpeg_bin=s.ffmpeg_bin,
        ),
        critique=VlmCritiqueAdapter(
            work_dir=work_dir / "critique",
            model=s.critique_model,
            base_url=s.critique_base_url,
            api_key=s.critique_api_key,
            ffmpeg_bin=s.ffmpeg_bin,
            ffprobe_bin=s.ffprobe_bin,
        ),
        publish=ManualPublishAdapter(review_dir=s.review_dir),
        ffmpeg_bin=s.ffmpeg_bin,
        ffprobe_bin=s.ffprobe_bin,
    )


@dataclass
class _JobContext:
    config: AppConfig
    artifact_store: ArtifactStore
    job_store: JobRepository
    job: Job
    work_dir: Path
    adapters: _Adapters


def _make_job_context(
    source_url: str,
    rights_cleared: bool,
    config: AppConfig,
    *,
    job_id: str | None = None,
    approval_overrides: dict | None = None,
) -> _JobContext:
    """Create stores, job record, work directory, and adapters for one run."""
    settings = config.settings
    artifact_store = create_artifact_store(settings)
    job_store = create_job_store(settings)

    job_kwargs: dict[str, Any] = dict(
        source_url=source_url,
        channel_profile_ref=config.channel_profile.id,
        cast_ref=config.cast.id,
        rights_cleared=rights_cleared,
    )
    if job_id:
        job_kwargs["job_id"] = job_id
    job = Job(**job_kwargs)
    job.status = JobStatus.RUNNING
    job_store.save_job(job)
    log_event(logger, "job_started", job_id=job.job_id, source_url=source_url)

    work_dir = settings.data_dir / "jobs" / job.job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    return _JobContext(
        config=config,
        artifact_store=artifact_store,
        job_store=job_store,
        job=job,
        work_dir=work_dir,
        adapters=_make_adapters(config, work_dir, approval_overrides=approval_overrides),
    )


def _restore_job_context(
    job_id: str,
    config: AppConfig,
    *,
    approval_overrides: dict | None = None,
) -> _JobContext:
    """Reconstruct a context for a previously started job (for resume/phase runs).

    Loads the job record from the store and points the work_dir at the existing
    job directory.  Raises ValueError if the job cannot be found.
    """
    settings = config.settings
    artifact_store = create_artifact_store(settings)
    job_store = create_job_store(settings)

    job = job_store.get_job(job_id)
    if job is None:
        raise ValueError(
            f"Job '{job_id}' not found in the job store. "
            "Check the job ID and make sure the same data_dir is configured."
        )

    job.status = JobStatus.RUNNING
    job_store.save_job(job)
    log_event(logger, "job_resumed", job_id=job.job_id)

    work_dir = settings.data_dir / "jobs" / job.job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    return _JobContext(
        config=config,
        artifact_store=artifact_store,
        job_store=job_store,
        job=job,
        work_dir=work_dir,
        adapters=_make_adapters(config, work_dir, approval_overrides=approval_overrides),
    )


def _resolve_line(ref: str, script: Script):
    """Parse a dialogue_line_ref and return the Line it points to.

    Format: "scene-{N}-line-{M}"  (N is 1-indexed, M is 0-indexed).
    """
    parts = ref.split("-")          # ["scene", "1", "line", "0"]
    scene_idx = int(parts[1]) - 1
    line_idx = int(parts[3])
    return script.scenes[scene_idx].lines[line_idx]


def _resolve_shot_characters(
    shot: Shot,
    cast: Cast,
) -> tuple[CastMember, list[CastMember], CastMemberParams | CastPairParams | None, bool]:
    """Resolve primary/secondary characters and LoRA params for a shot.

    Returns (primary_member, other_members, render_params, using_pair_lora).
    """
    member_map = {m.id: m for m in cast.members}
    char_ids = shot.characters_on_screen

    is_reaction = (shot.camera or "").strip().lower() == "reaction"
    if is_reaction and len(char_ids) >= 2:
        primary_id = char_ids[1]
        other_ids = [char_ids[0]]
    else:
        primary_id = char_ids[0]
        other_ids = list(char_ids[1:])

    primary = member_map[primary_id]
    others = [member_map[cid] for cid in other_ids if cid in member_map]

    pair_key = frozenset(char_ids)
    pair_map = load_cast_pair_params(cast.id)
    pair_params = pair_map.get(pair_key)
    if pair_params and pair_params.lora_file:
        return primary, [], pair_params, True

    member_params = load_cast_params(cast.id).get(primary_id)
    return primary, others, member_params, False


def _shot_render_request(
    shot: Shot,
    primary: CastMember,
    other_members: list[CastMember],
    params: "CastMemberParams | None",
) -> RenderCharacterRequest:
    """Build the render_character request for a shot — shared by the sequential
    and batched Phase A paths so both render identically."""
    return RenderCharacterRequest(
        member=primary,
        setting=shot.setting,
        expression=None,
        shot_id=shot.shot_id,
        camera=shot.camera,
        action=shot.action,
        other_members=other_members,
        lora_file=params.lora_file if params else "",
        lora_weight=params.lora_weight if params else None,
        steps=params.steps if params else None,
        guidance_scale=params.guidance_scale if params else None,
        trigger=params.trigger if params else "",
        style_suffix=params.style_suffix if params else "",
    )


async def _phase_a_prefetch_renders(
    shots: list[Shot],
    cast: Cast,
    adapters: _Adapters,
    work_dir: Path,
    options: RunOptions | None = None,
) -> dict[str, "ImageSet"]:
    """Batch-render candidates for every shot the resume cache can't serve.

    One adapters.render.run_many() call for all pending shots lets the adapter
    load the 64 GB Flux model once per LoRA instead of once per candidate (see
    MusubiFluxAdapter.run_many) — with 13 shots that's ~9 min of pure model
    loading saved per job. Returns shot_id → ImageSet, consumed by
    _render_shot_candidates as `prefetched`; fully cached shots are excluded
    and served by its own resume logic.
    """
    opts = options or RunOptions()
    num_candidates = getattr(adapters.render, "_num_images", 1)
    pending: list[tuple[Shot, RenderCharacterRequest]] = []
    for shot in shots:
        primary, other_members, params, _ = _resolve_shot_characters(shot, cast)
        render_dir = work_dir / "renders" / shot.shot_id / primary.id
        existing = sorted(render_dir.glob("render_??.png"))
        if opts.resume and not existing:
            # legacy member-keyed layout (pre shot-scoped renders)
            existing = sorted((work_dir / "renders" / primary.id).glob("render_??.png"))
        if opts.resume and len(existing) >= num_candidates:
            continue
        pending.append((shot, _shot_render_request(shot, primary, other_members, params)))

    if not pending:
        return {}

    # Dedup identically specified shots (same member/setting/camera/action/params
    # → byte-identical render prompt): render the first occurrence, copy its PNGs
    # into each duplicate's dir. With action in the prompt this only fires when
    # the plan itself repeats a shot spec, so it can't flatten visual variety.
    unique: dict[str, tuple[Shot, RenderCharacterRequest]] = {}
    duplicates: dict[str, list[Shot]] = {}
    for shot, req in pending:
        fingerprint = req.model_dump_json(exclude={"shot_id"})
        if fingerprint in unique:
            duplicates.setdefault(fingerprint, []).append(shot)
        else:
            unique[fingerprint] = (shot, req)

    log_event(logger, "phase_a_batch_render", shots=len(pending),
              unique_renders=len(unique),
              candidates=len(unique) * num_candidates)
    image_sets = await adapters.render.run_many(
        [req for _, req in unique.values()]
    )

    out: dict[str, ImageSet] = {}
    for (fingerprint, (rep_shot, rep_req)), image_set in zip(unique.items(), image_sets):
        out[rep_shot.shot_id] = image_set
        for dup_shot in duplicates.get(fingerprint, []):
            dup_dir = work_dir / "renders" / dup_shot.shot_id / rep_req.member.id
            dup_dir.mkdir(parents=True, exist_ok=True)
            copied: list[str] = []
            for src in image_set.images:
                dst = dup_dir / Path(src).name
                shutil.copyfile(src, dst)
                copied.append(str(dst))
            out[dup_shot.shot_id] = ImageSet(member_id=image_set.member_id, images=copied)
            log_event(logger, "render_deduplicated",
                      shot_id=dup_shot.shot_id, source_shot=rep_shot.shot_id)
    return out


async def _render_shot_candidates(
    shot: Shot,
    cast: Cast,
    adapters: _Adapters,
    work_dir: Path,
    options: RunOptions | None = None,
    prefetched: "ImageSet | None" = None,
) -> "ImageCritiqueResult":
    """Phase A: render N candidates for one shot, critique them, return the winner.

    Runs render_character (N images) then VLM image critique. `prefetched`
    carries an ImageSet already rendered by _phase_a_prefetch_renders (batched
    path) — the per-shot render call is skipped and critique runs on it.
    Resume: if all candidate PNGs and a cached critique result already exist,
    skips both render and critique (the VLM call is a real, non-trivial cost —
    ~20-40s per shot — that would otherwise redo on every retry/resume).
    """
    from core.models.capabilities import ImageCritiqueRequest, ImageCritiqueResult, ImageSet

    opts = options or RunOptions()
    primary, other_members, params, using_pair = _resolve_shot_characters(shot, cast)
    primary_id = primary.id

    expression = None
    render_dir = work_dir / "renders" / shot.shot_id / primary_id
    render_dir.mkdir(parents=True, exist_ok=True)

    num_candidates = getattr(adapters.render, "_num_images", 1)
    existing_pngs = sorted(render_dir.glob("render_??.png"))
    if opts.resume and not existing_pngs:
        legacy_dir = work_dir / "renders" / primary_id
        legacy_pngs = sorted(legacy_dir.glob("render_??.png"))
        if legacy_pngs:
            render_dir = legacy_dir
            existing_pngs = legacy_pngs
    critique_path = work_dir / "critique" / f"{shot.shot_id}.json"

    if opts.resume and len(existing_pngs) >= num_candidates and critique_path.exists():
        logger.info("Skipping render_character + image_critique for %s (cached)", shot.shot_id)
        return ImageCritiqueResult.model_validate_json(critique_path.read_text())

    if opts.resume and len(existing_pngs) >= num_candidates:
        logger.info("Skipping render_character for %s (all %d candidates exist)",
                    shot.shot_id, num_candidates)
        render_result = ImageSet(member_id=primary_id,
                                 images=[str(p) for p in existing_pngs[:num_candidates]])
    elif prefetched is not None:
        render_result = prefetched
    else:
        render_result = await adapters.render.run(
            _shot_render_request(shot, primary, other_members, params)
        )

    if len(render_result.images) == 1:
        # Nothing to rank — skip the ~30-40s VLM call and auto-pick the single
        # candidate. The human image-approval gate stays as the quality check.
        critique = ImageCritiqueResult(
            winner_index=0,
            winner_uri=render_result.images[0],
            candidate_uris=list(render_result.images),
            overall_reasoning="single candidate — VLM critique skipped",
            origin="single",
        )
        log_event(logger, "image_critique_skipped_single", shot_id=shot.shot_id)
        critique_path.parent.mkdir(parents=True, exist_ok=True)
        critique_path.write_text(critique.model_dump_json())
        return critique

    if other_members:
        other_names = ", ".join(m.name for m in other_members)
        shot_prompt = (
            f"{primary.name} ({primary.visual_descriptor}) "
            f"with {other_names} in {shot.setting}; action: {shot.action}"
        )
    else:
        shot_prompt = f"{primary.name} ({primary.visual_descriptor}) in {shot.setting}; action: {shot.action}"
    critique = await adapters.image_critique.run(
        ImageCritiqueRequest(
            shot_id=shot.shot_id,
            shot_prompt=shot_prompt,
            candidate_uris=render_result.images,
            cast_descriptor=primary.visual_descriptor,
            other_descriptors=[m.visual_descriptor for m in other_members],
        )
    )
    critique_path.parent.mkdir(parents=True, exist_ok=True)
    critique_path.write_text(critique.model_dump_json())
    return critique


def _build_user_image_critiques(
    shots: list[Shot],
    user_images: dict[str, str],
    cast: Cast,
) -> list["ImageCritiqueResult"]:
    """Synthesize critique results from user-provided reference images.

    story_images mode: render_character + VLM critique never run. Every shot's
    candidate list is the full set of reference images (in cast order, so the
    approval grid is stable across shots) with the shot speaker's image
    pre-selected. The operator can still override per shot at the approval gate.
    """
    from core.models.capabilities import ImageCritiqueResult

    ordered_members = [m.id for m in cast.members if m.id in user_images]
    candidate_uris = [user_images[member_id] for member_id in ordered_members]
    if not candidate_uris:
        raise ValueError("user_images is set but contains no known cast members")

    results: list[ImageCritiqueResult] = []
    for shot in shots:
        speaker_id = shot.characters_on_screen[0]
        winner_index = (
            ordered_members.index(speaker_id) if speaker_id in ordered_members else 0
        )
        results.append(
            ImageCritiqueResult(
                winner_index=winner_index,
                winner_uri=candidate_uris[winner_index],
                candidate_uris=candidate_uris,
                overall_reasoning=(
                    "User-provided reference images ("
                    + ", ".join(ordered_members)
                    + f"); default for this shot: {ordered_members[winner_index]}."
                ),
                origin="user",
            )
        )
    return results


async def _run_image_approval_gate(
    shots: list[Shot],
    critique_results: list["ImageCritiqueResult"],
    adapters: _Adapters,
    cast_id: str,
) -> list[str]:
    """Show the image approval grid UI; return approved URI per shot."""
    from core.models.capabilities import ImageApprovalRequest

    result = await adapters.image_approval.run(
        ImageApprovalRequest(
            shots=shots,
            critique_results=critique_results,
            cast_id=cast_id,
        )
    )
    return result.approved_uris


def _notify_shot(
    stage_hook: Callable[..., None] | None,
    stage_name: str,
    event_type: str,
    *,
    shot_id: str,
    label: str,
    detail: str,
) -> None:
    """Emit a stage_hook event carrying shot progress (e.g. "Shot s01 (1/14): synthesizing voice")."""
    if stage_hook:
        stage_hook(stage_name, event_type, shot_id=shot_id, message=f"{label}: {detail}")


def _existing_audio_for_shot(
    work_dir: Path,
    shot: Shot,
    speaker_id: str,
    render_mode: str,
) -> str | None:
    if render_mode == "source_audio":
        path = work_dir / "audio" / "source_slices" / f"{shot.shot_id}.wav"
        return str(path) if path.exists() else None
    if render_mode == "re_voice":
        path = work_dir / "audio" / "timed" / f"{shot.shot_id}.wav"
        return str(path) if path.exists() else None
    audio_files = sorted((work_dir / "audio" / speaker_id).glob("*.wav"))
    return str(audio_files[0]) if audio_files else None


async def _generate_shot_video(
    shot: Shot,
    script: Script,
    cast: Cast,
    adapters: _Adapters,
    work_dir: Path,
    image_uri: str,
    options: RunOptions | None = None,
    *,
    shot_index: int = 1,
    total_shots: int = 1,
    render_mode: str = "full",
    source_audio_uri: str | None = None,
) -> tuple[VideoClip, AudioTrack]:
    """Phase B: synthesize voice + generate video for one shot using the approved image.

    Skips stages whose output files already exist when opts.resume is True.
    ``shot_index``/``total_shots`` (1-based) are used only to label the
    stage_hook progress events emitted around each sub-stage below.
    """
    opts = options or RunOptions()
    member_map = {m.id: m for m in cast.members}
    speaker_id = shot.characters_on_screen[0]
    speaker = member_map[speaker_id]
    label = f"Shot {shot.shot_id} ({shot_index}/{total_shots})"

    first_line = (
        _resolve_line(shot.dialogue_line_refs[0], script)
        if shot.dialogue_line_refs
        else None
    )
    expression = first_line.expression if first_line else None
    lines = [_resolve_line(ref, script) for ref in shot.dialogue_line_refs]
    line_text = " ".join(line.text.strip() for line in lines if line.text.strip())

    native_lipsync = getattr(adapters.video, "native_lipsync", False)

    if native_lipsync:
        done_path = adapters.video.work_dir / shot.shot_id / "clip.mp4"
    else:
        done_path = adapters.lipsync.work_dir / shot.shot_id / "synced.mp4"

    if opts.resume and done_path.exists():
        logger.info("Skipping video generation for %s (%s exists)", shot.shot_id, done_path.name)
        audio_uri = (
            _existing_audio_for_shot(work_dir, shot, speaker_id, render_mode)
            or str(done_path)
        )
        return (
            VideoClip(uri=str(done_path), duration_sec=shot.duration_sec),
            AudioTrack(uri=audio_uri, duration_sec=shot.duration_sec),
        )

    log_event(logger, "shot_video_started", shot_id=shot.shot_id, speaker=speaker_id)

    # ── 1. synthesize_voice (or slice source audio) ────────────────────────────
    vparams = load_cast_params(cast.id).get(speaker_id)
    if render_mode == "source_audio" and source_audio_uri and shot.source_start_sec is not None and shot.source_end_sec is not None:
        audio_slice_path = work_dir / "audio" / "source_slices" / f"{shot.shot_id}.wav"
        if opts.resume and audio_slice_path.exists():
            logger.info("Skipping source audio slice for %s (exists)", shot.shot_id)
            audio_track = AudioTrack(uri=str(audio_slice_path), duration_sec=shot.duration_sec)
        else:
            _notify_shot(
                opts.stage_hook, "slice_source_audio", "stage_started",
                shot_id=shot.shot_id, label=label, detail="slicing source audio",
            )
            audio_track = await _slice_source_audio(
                source_audio_uri, shot.source_start_sec, shot.source_end_sec,
                audio_slice_path, adapters.ffmpeg_bin,
            )
            _notify_shot(
                opts.stage_hook, "slice_source_audio", "stage_completed",
                shot_id=shot.shot_id, label=label, detail="audio slice done",
            )
    else:
        existing_audio = _existing_audio_for_shot(work_dir, shot, speaker_id, render_mode)
        if opts.resume and existing_audio:
            logger.info("Skipping synthesize_voice for %s (audio exists)", shot.shot_id)
            audio_track = AudioTrack(uri=existing_audio, duration_sec=shot.duration_sec)
        else:
            _notify_shot(
                opts.stage_hook, "synthesize_voice", "stage_started",
                shot_id=shot.shot_id, label=label, detail="synthesizing voice",
            )
            voice_ref = (vparams.voice_file if vparams and vparams.voice_file
                         else speaker.voice_profile_ref)
            audio_track = await adapters.voice.run(
                VoiceRequest(
                    text=line_text,
                    voice_profile_ref=voice_ref,
                    speaker_id=speaker_id,
                    expression=expression,
                    language=opts.language if opts else "en",
                )
            )
            _notify_shot(
                opts.stage_hook, "synthesize_voice", "stage_completed",
                shot_id=shot.shot_id, label=label, detail="voice done",
            )
            if render_mode == "re_voice":
                ffprobe_bin = getattr(adapters, "ffprobe_bin", None)
                if not isinstance(ffprobe_bin, str):
                    ffprobe_bin = "ffprobe"
                _notify_shot(
                    opts.stage_hook, "fit_voice_audio", "stage_started",
                    shot_id=shot.shot_id, label=label, detail="matching source pacing",
                )
                audio_track = await _fit_audio_to_duration(
                    audio_track,
                    shot.duration_sec,
                    work_dir / "audio" / "timed" / f"{shot.shot_id}.wav",
                    adapters.ffmpeg_bin,
                    ffprobe_bin,
                )
                _notify_shot(
                    opts.stage_hook, "fit_voice_audio", "stage_completed",
                    shot_id=shot.shot_id, label=label, detail="voice timing matched",
                )

    if getattr(adapters.video, "requires_voice_unloaded", False) is True and is_managed(adapters.voice):
        _notify_shot(
            opts.stage_hook, "voice_model_unload", "stage_started",
            shot_id=shot.shot_id, label=label, detail="freeing voice model before video",
        )
        await adapters.voice.unload()
        _notify_shot(
            opts.stage_hook, "voice_model_unload", "stage_completed",
            shot_id=shot.shot_id, label=label, detail="voice model freed",
        )

    # ── 2. generate_video ─────────────────────────────────────────────────────
    video_action = shot.action
    if len(shot.characters_on_screen) >= 2:
        other_ids = [c for c in shot.characters_on_screen if c != speaker_id]
        other_name = member_map[other_ids[0]].name if other_ids else ""
        if other_name:
            video_action = (
                f"{speaker.name} is speaking, {other_name} is listening, "
                f"{shot.action}, medium to wide framing, two characters visible"
            )

    clip_path = adapters.video.work_dir / shot.shot_id / "clip.mp4"
    if opts.resume and clip_path.exists():
        logger.info("Skipping generate_video for %s (clip.mp4 exists)", shot.shot_id)
        clip = VideoClip(uri=str(clip_path), duration_sec=shot.duration_sec)
    else:
        _notify_shot(
            opts.stage_hook, "generate_video", "stage_started",
            shot_id=shot.shot_id, label=label, detail="generating video",
        )
        clip = await adapters.video.run(
            VideoRequest(
                image_uri=image_uri,
                action=video_action,
                duration_sec=shot.duration_sec,
                shot_id=shot.shot_id,
                setting=shot.setting,
                audio_uri=audio_track.uri if native_lipsync else None,
                style_suffix=vparams.style_suffix if vparams else "",
            )
        )
        _notify_shot(
            opts.stage_hook, "generate_video", "stage_completed",
            shot_id=shot.shot_id, label=label, detail="video done",
        )

    # ── 3. lip_sync (skipped when video adapter handles it natively) ──────────
    if native_lipsync:
        log_event(logger, "lip_sync_skipped", shot_id=shot.shot_id, reason="native_lipsync")
        synced = clip
    else:
        if getattr(adapters.lipsync, "requires_voice_unloaded", False) is True and is_managed(adapters.voice):
            _notify_shot(
                opts.stage_hook, "voice_model_unload", "stage_started",
                shot_id=shot.shot_id, label=label, detail="freeing voice model before lip sync",
            )
            await adapters.voice.unload()
            _notify_shot(
                opts.stage_hook, "voice_model_unload", "stage_completed",
                shot_id=shot.shot_id, label=label, detail="voice model freed",
            )
        _notify_shot(
            opts.stage_hook, "lip_sync", "stage_started",
            shot_id=shot.shot_id, label=label, detail="lip-syncing",
        )
        try:
            synced = await adapters.lipsync.run(
                LipSyncRequest(
                    video_uri=clip.uri,
                    audio_uri=audio_track.uri,
                    shot_id=shot.shot_id,
                )
            )
        except Exception:
            logger.warning(
                "lip_sync failed for %s — falling back to raw clip at %s",
                shot.shot_id, clip.uri, exc_info=True,
            )
            synced = clip
        _notify_shot(
            opts.stage_hook, "lip_sync", "stage_completed",
            shot_id=shot.shot_id, label=label, detail="lip sync done",
        )

    log_event(logger, "shot_completed", shot_id=shot.shot_id)
    _notify_shot(
        opts.stage_hook, "shot_complete", "stage_completed",
        shot_id=shot.shot_id, label=label, detail="shot complete",
    )
    return synced, audio_track


# Moved to core/gpu_sequencer.py; alias kept for existing callers/tests.
_unload_ollama_model = unload_ollama_model


async def _concat_audio(
    tracks: list[AudioTrack],
    work_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> AudioTrack:
    """Concatenate per-shot WAV files into one combined dialogue track."""
    if len(tracks) == 1:
        return tracks[0]

    concat_file = work_dir / "audio_concat.txt"
    lines = [f"file '{Path(t.uri).resolve()}'" for t in tracks]
    concat_file.write_text("\n".join(lines), encoding="utf-8")

    output = work_dir / "combined_audio.wav"
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin, "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-c:a", "copy",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-1000:]
        raise RuntimeError(f"Audio concat failed (exit {proc.returncode}):\n{tail}")

    return AudioTrack(
        uri=str(output),
        duration_sec=sum(t.duration_sec for t in tracks),
    )


def _build_verbatim_script(
    transcribe_result,
    cast: Cast,
    visual_context: "VisualContext | None" = None,
    rights_cleared: bool = True,
) -> Script:
    """Build a Script directly from transcript segments (source_audio mode).

    Skips adapt_script entirely — dialogue text is the original transcript,
    scenes are derived from visual_context segments (or a single default scene).
    """
    from core.models.content import LearningObjective, Line, Scene
    from core.models.guardrails import SourceRights

    segments = transcribe_result.segments
    if not segments:
        raise ValueError("Cannot build verbatim script: no transcript segments")

    primary_speaker = cast.members[0].id
    secondary_speaker = cast.members[1].id if len(cast.members) > 1 else primary_speaker

    vis_segments = getattr(visual_context, "segments", None) if visual_context else None
    if vis_segments:
        scenes: list[Scene] = []
        seg_idx = 0
        for vs in vis_segments:
            scene_lines: list[Line] = []
            while seg_idx < len(segments):
                seg = segments[seg_idx]
                if seg.start >= vs.end and vs != vis_segments[-1]:
                    break
                speaker = primary_speaker if seg_idx % 2 == 0 else secondary_speaker
                scene_lines.append(Line(
                    speaker=speaker,
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                ))
                seg_idx += 1
            if scene_lines:
                setting = getattr(vs, "setting", None) or getattr(vs, "description", "indoor scene")
                # adapt_script's LLM prompt weaves VisualSegment.props into the
                # scene text it writes (see _format_visual_block); this path
                # has no LLM call to do that, so fold them in directly or a
                # verbatim script silently loses every prop the VLM detected.
                props = getattr(vs, "props", None)
                if props:
                    setting = f"{setting} — props: {', '.join(props)}"
                scenes.append(Scene(
                    setting=setting,
                    characters_present=[primary_speaker],
                    lines=scene_lines,
                ))
        if seg_idx < len(segments):
            leftover: list[Line] = []
            while seg_idx < len(segments):
                seg = segments[seg_idx]
                speaker = primary_speaker if seg_idx % 2 == 0 else secondary_speaker
                leftover.append(Line(
                    speaker=speaker,
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                ))
                seg_idx += 1
            scenes.append(Scene(
                setting="indoor scene",
                characters_present=[primary_speaker],
                lines=leftover,
            ))
    else:
        lines = []
        for i, seg in enumerate(segments):
            speaker = primary_speaker if i % 2 == 0 else secondary_speaker
            lines.append(Line(
                speaker=speaker,
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
            ))
        scenes = [Scene(
            setting="indoor scene",
            characters_present=[primary_speaker],
            lines=lines,
        )]

    full_text = " ".join(seg.text.strip() for seg in segments)
    return Script(
        mode="verbatim",
        learning_objective=LearningObjective(
            concept="source content",
            age_range="3-6",
            success_phrase="understands the topic",
        ),
        scenes=scenes,
        caption_text=full_text[:200],
        source_rights=SourceRights(kind="transformed", rights_cleared=rights_cleared, notes="verbatim source_audio mode"),
    )


def _flatten_script_lines(script: Script) -> list[tuple[int, int, Any]]:
    return [
        (scene_idx, line_idx, line)
        for scene_idx, scene in enumerate(script.scenes)
        for line_idx, line in enumerate(scene.lines)
    ]


def _apply_source_timing_to_script(
    script: Script,
    transcribe_result,
    *,
    total_duration: float | None = None,
) -> Script:
    """Attach source-video timing to an adapted script for re_voice mode.

    When the adapted script has the same number of lines as the transcript, keep
    the transcript segment boundaries exactly. If the LLM wrote more/fewer
    lines, distribute the source duration across the adapted lines in order
    using their word counts as a rough pacing weight.
    """
    lines = _flatten_script_lines(script)
    segments = list(getattr(transcribe_result, "segments", []) or [])
    if not lines or not segments:
        return script

    if len(lines) == len(segments):
        for (_, _, line), segment in zip(lines, segments):
            line.start = float(segment.start)
            line.end = float(segment.end)
        return script

    source_start = min(float(segment.start) for segment in segments)
    source_end = (
        float(total_duration)
        if total_duration and total_duration > source_start
        else max(float(segment.end) for segment in segments)
    )
    if source_end <= source_start:
        return script

    weights = [max(1.0, len(line.text.split()) / 2.0) for _, _, line in lines]
    total_weight = sum(weights) or float(len(lines))
    cursor = source_start
    for idx, ((_, _, line), weight) in enumerate(zip(lines, weights)):
        if idx == len(lines) - 1:
            end = source_end
        else:
            end = cursor + (source_end - source_start) * (weight / total_weight)
        line.start = round(cursor, 3)
        line.end = round(max(end, cursor + 0.1), 3)
        cursor = end
    return script


def _apply_segment_timing_to_storyboard(
    storyboard: Storyboard,
    script: Script,
) -> Storyboard:
    """Override shot durations with source segment timing (source_audio/re_voice).

    Each shot maps to one dialogue line whose Line.start/end carry the original
    segment timestamps. The shot's duration_sec, source_start_sec, and
    source_end_sec are set from those timestamps.
    """
    for shot in storyboard.shots:
        if not shot.dialogue_line_refs:
            continue
        line = _resolve_line(shot.dialogue_line_refs[0], script)
        if line.start is not None and line.end is not None:
            shot.duration_sec = max(1.0, line.end - line.start)
            shot.source_start_sec = line.start
            shot.source_end_sec = line.end
    return storyboard


def _validate_timed_storyboard(storyboard: Storyboard, render_mode: str) -> None:
    """Fail fast when a timed mode is asked to render an untimed plan."""
    if render_mode not in {"source_audio", "re_voice"}:
        return
    missing = [
        shot.shot_id
        for shot in storyboard.shots
        if (
            shot.source_start_sec is None
            or shot.source_end_sec is None
            or shot.source_end_sec <= shot.source_start_sec
        )
    ]
    if missing:
        sample = ", ".join(missing[:5])
        more = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"{render_mode} mode requires source-timed storyboard shots. "
            f"Missing/invalid timing for: {sample}{more}. "
            f"Re-run the script_plan/plan phase with render_mode='{render_mode}' "
            "before rendering."
        )


def _storyboard_has_source_timing(storyboard: Storyboard) -> bool:
    return all(
        shot.source_start_sec is not None
        and shot.source_end_sec is not None
        and shot.source_end_sec > shot.source_start_sec
        for shot in storyboard.shots
    )


def _chunk_shot_durations(
    total_duration: float, max_shot_duration_sec: float
) -> list[tuple[float, float]]:
    """Greedy-fill the source video's total duration into ≤max_shot_duration_sec
    chunks: full-length chunks first, the last chunk gets whatever remains
    (e.g. 15s / 8s max → [0-8, 8-15], not two even 7.5s halves). Stitching the
    resulting shots back together exactly reconstructs the original length.
    """
    if total_duration <= 0 or max_shot_duration_sec <= 0:
        return []
    chunks: list[tuple[float, float]] = []
    start = 0.0
    remaining = total_duration
    while remaining > 1e-6:
        length = min(max_shot_duration_sec, remaining)
        chunks.append((start, start + length))
        start += length
        remaining -= length
    return chunks


def _rebuild_storyboard_by_duration(
    storyboard: Storyboard,
    script: Script,
    total_duration: float,
    max_shot_duration_sec: float,
) -> Storyboard:
    """Replace the LLM-planned shot count/boundaries with deterministic
    duration-based chunks (source_audio mode) so the assembled video's total
    length exactly matches the source, regardless of scene/segment count.

    Each chunk borrows camera/action/setting/characters_on_screen from
    whichever original (scene-based) plan_shots shot overlaps it the most —
    preserving that shot's creative direction — but gets its own
    dialogue_line_refs from whichever verbatim-script lines actually fall
    inside its time range, and its own source_start_sec/source_end_sec for
    audio slicing.
    """
    chunks = _chunk_shot_durations(total_duration, max_shot_duration_sec)
    if not chunks:
        return storyboard

    def _safe_line(ref: str):
        try:
            return _resolve_line(ref, script)
        except (IndexError, ValueError):
            return None

    def _shot_span(shot: Shot) -> tuple[float, float] | None:
        spans = [
            (line.start, line.end)
            for ref in shot.dialogue_line_refs
            for line in [_safe_line(ref)]
            if line is not None and line.start is not None and line.end is not None
        ]
        if not spans:
            return None
        return (min(s for s, _ in spans), max(e for _, e in spans))

    original_spans = [(shot, _shot_span(shot)) for shot in storyboard.shots]
    all_lines = [
        (f"scene-{i + 1}-line-{j}", line)
        for i, scene in enumerate(script.scenes)
        for j, line in enumerate(scene.lines)
        if line.start is not None and line.end is not None
    ]

    new_shots: list[Shot] = []
    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        best_shot = storyboard.shots[0]
        best_overlap = -1.0
        for shot, span in original_spans:
            if span is None:
                continue
            overlap = min(span[1], chunk_end) - max(span[0], chunk_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_shot = shot

        refs = [
            ref for ref, line in all_lines
            if chunk_start <= (line.start + line.end) / 2 < chunk_end
        ]
        if not refs and all_lines:
            chunk_mid = (chunk_start + chunk_end) / 2
            nearest_ref, _ = min(
                all_lines,
                key=lambda item: abs(((item[1].start + item[1].end) / 2) - chunk_mid),
            )
            refs = [nearest_ref]
        if not refs:
            refs = [ref for ref in best_shot.dialogue_line_refs if _safe_line(ref) is not None]

        new_shots.append(Shot(
            shot_id=f"s{idx:02d}",
            scene_ref=best_shot.scene_ref,
            characters_on_screen=best_shot.characters_on_screen,
            setting=best_shot.setting,
            camera=best_shot.camera,
            action=best_shot.action,
            dialogue_line_refs=refs,
            duration_sec=chunk_end - chunk_start,
            source_start_sec=chunk_start,
            source_end_sec=chunk_end,
        ))

    return Storyboard(shots=new_shots)


def _retime_source_audio_plan(
    storyboard: Storyboard,
    transcribe_result,
    fetch_result,
    cast: Cast,
    *,
    visual_context: "VisualContext | None" = None,
    rights_cleared: bool = True,
    max_shot_duration_sec: float = 8.0,
) -> tuple[Script, Storyboard]:
    """Build a verbatim script and source-timed storyboard for source_audio."""
    script = _build_verbatim_script(
        transcribe_result,
        cast,
        visual_context=visual_context,
        rights_cleared=rights_cleared,
    )
    timed = _rebuild_storyboard_by_duration(
        storyboard,
        script,
        total_duration=fetch_result.duration_sec,
        max_shot_duration_sec=max_shot_duration_sec,
    )
    return script, timed


async def _slice_source_audio(
    source_audio_uri: str,
    start: float,
    end: float,
    out_path: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> AudioTrack:
    """Extract an audio segment from the source audio file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin, "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(source_audio_uri),
        "-c", "copy",
        str(out_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"Audio slice failed (exit {proc.returncode}):\n{tail}")
    return AudioTrack(uri=str(out_path), duration_sec=end - start)


def _atempo_filter_chain(tempo: float) -> str:
    """Build an ffmpeg atempo chain whose product equals tempo."""
    if tempo <= 0:
        tempo = 1.0
    factors: list[float] = []
    remaining = tempo
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    if abs(remaining - 1.0) > 0.01 or not factors:
        factors.append(remaining)
    return ",".join(f"atempo={factor:.6g}" for factor in factors)


async def _fit_audio_to_duration(
    track: AudioTrack,
    target_duration_sec: float,
    out_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> AudioTrack:
    """Time-stretch, pad, and trim an audio track to a shot duration."""
    if target_duration_sec <= 0:
        return track
    source = Path(track.uri)
    if not source.exists():
        raise FileNotFoundError(f"Cannot fit missing audio track: {track.uri}")

    actual = await _probe_duration_sec(str(source), ffprobe_bin)
    actual = actual if actual and actual > 0 else track.duration_sec
    if actual and abs(actual - target_duration_sec) <= 0.05:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != out_path.resolve():
            shutil.copyfile(source, out_path)
        return AudioTrack(
            uri=str(out_path),
            duration_sec=target_duration_sec,
            speaker_id=track.speaker_id,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tempo = (actual / target_duration_sec) if actual and target_duration_sec else 1.0
    filters = (
        f"{_atempo_filter_chain(tempo)},"
        f"apad,atrim=0:{target_duration_sec:.3f},asetpts=N/SR/TB"
    )
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_bin, "-y",
        "-i", str(source),
        "-vn",
        "-af", filters,
        "-ar", "44100",
        "-ac", "1",
        str(out_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-500:]
        raise RuntimeError(f"Audio duration fit failed (exit {proc.returncode}):\n{tail}")
    return AudioTrack(
        uri=str(out_path),
        duration_sec=target_duration_sec,
        speaker_id=track.speaker_id,
    )


def _load_artifact(
    job_id: str,
    stage: str,
    model_cls: type,
    artifact_store: ArtifactStore,
) -> object | None:
    """Load a persisted stage artifact and deserialize it to model_cls. Returns None if absent."""
    data = artifact_store.get_json(job_id, stage)
    if data is None:
        return None
    try:
        return model_cls.model_validate(data)
    except Exception as exc:
        logger.warning("Could not deserialize artifact '%s' (%s): %s — will re-run stage", stage, model_cls.__name__, exc)
        return None


async def _critique_loop(
    storyboard: Storyboard,
    script: Script,
    ctx: "_JobContext",
    max_iterations: int,
    visual_context: "VisualContext | None" = None,
    storyboard_transform: Callable[[Storyboard], Storyboard] | None = None,
) -> tuple[Storyboard, list[str]]:
    """
    Run the plan critique loop: up to max_iterations re-plan attempts.
    Returns (final_storyboard, last_critique_notes).
    If critique never passes within the budget, returns the best storyboard
    with its notes (caller decides whether to fail or pass to approval UI).
    """
    from core.models.capabilities import PlanCritiqueRequest, PlanShotsRequest

    adapters = ctx.adapters
    config = ctx.config
    notes: list[str] = []

    for attempt in range(1, max_iterations + 1):
        if attempt > 1 and notes:
            log_event(logger, "plan_replan", attempt=attempt, notes=notes)
            storyboard = await adapters.plan.run(
                PlanShotsRequest(script=script, cast=config.cast, critique_notes=notes,
                                 visual_context=visual_context)
            )
        if storyboard_transform is not None:
            storyboard = storyboard_transform(storyboard)

        critique = await adapters.plan_critique.run(
            PlanCritiqueRequest(storyboard=storyboard, script=script, cast=config.cast)
        )
        log_event(logger, "critique_plan_result",
                  attempt=attempt, verdict=critique.verdict, scores=critique.scores)

        if critique.verdict == "pass":
            return storyboard, critique.revision_notes

        notes = critique.revision_notes
        logger.info("critique_plan: revise (attempt %d/%d) notes=%s", attempt, max_iterations, notes)

    logger.warning("critique_plan: max iterations reached — proceeding with best storyboard")
    return storyboard, notes


async def _render_plan_overlays(
    storyboard: Storyboard,
    ctx: "_JobContext",
    opts: "RunOptions",
) -> Storyboard:
    """Render each shot.overlay to a PNG panel and set overlay.png_uri.

    Best-effort: matplotlib missing, or any render failure, leaves the shot
    without a panel — never fails the job. Runs before the plan approval gate
    so the operator previews the panels; cheap CPU work, so re-plan iterations
    simply re-render.
    """
    from core.models.capabilities import RenderOverlaysRequest

    shots_with_overlays = [s for s in storyboard.shots if s.overlay is not None]
    if not shots_with_overlays:
        return storyboard

    adapters = ctx.adapters
    health = await adapters.overlays.health()
    if health.status != "ok":
        log_event(logger, "render_overlays_skipped", reason=health.reason)
        return storyboard

    if opts.stage_hook:
        opts.stage_hook("render_overlays", "stage_started")
    try:
        result = await adapters.overlays.run(
            RenderOverlaysRequest(shots=shots_with_overlays)
        )
    except Exception as exc:  # overlays are decorative — never fatal
        logger.warning("render_overlays failed (%s); continuing without panels", exc)
        return storyboard

    for shot in shots_with_overlays:
        shot.overlay.png_uri = result.images.get(shot.shot_id)

    ctx.artifact_store.put_json(
        ctx.job.job_id, "render_overlays",
        {"images": result.images, "skipped": result.skipped},
    )
    if opts.stage_hook:
        opts.stage_hook("render_overlays", "stage_completed")
    return storyboard


async def _run_plan_critique_and_approval(
    storyboard: Storyboard,
    script: Script,
    ctx: "_JobContext",
    opts: "RunOptions",
    visual_context: "VisualContext | None" = None,
    storyboard_transform: Callable[[Storyboard], Storyboard] | None = None,
) -> tuple[Storyboard, Script]:
    """
    Run the critique loop then the human approval gate.
    On rejection, runs one more critique loop with the human's notes, then
    shows the approval UI again. A second rejection fails the job.
    """
    from core.models.capabilities import PlanCritiqueRequest, PlanShotsRequest

    s = ctx.config.settings
    job = ctx.job
    job_store = ctx.job_store
    adapters = ctx.adapters

    def _persist_approved_plan(sb: Storyboard) -> None:
        # Re-plans call adapters.plan.run() directly (no run_stage), so the
        # persisted plan_shots artifact goes stale — and downstream phases
        # (render/assemble resume) plus overlay png_uris depend on the final
        # approved storyboard. Re-persist it after the gate.
        ctx.artifact_store.put_json(job.job_id, "plan_shots", sb.model_dump(mode="json"))

    # ── LLM critique loop ─────────────────────────────────────────────────────
    storyboard, last_notes = await _critique_loop(
        storyboard, script, ctx, s.max_plan_iterations, visual_context=visual_context,
        storyboard_transform=storyboard_transform,
    )

    # Reconstruct final critique result for the UI (re-run to get fresh scores)
    final_critique = await adapters.plan_critique.run(
        PlanCritiqueRequest(storyboard=storyboard, script=script, cast=ctx.config.cast)
    )

    # Render overlay panels so their previews ride the approval gate.
    storyboard = await _render_plan_overlays(storyboard, ctx, opts)

    # ── Human approval gate ───────────────────────────────────────────────────
    job.status = JobStatus.PENDING_APPROVAL
    job_store.save_job(job)

    approved, rejection_notes = await adapters.approval.request_approval(
        storyboard=storyboard,
        script=script,
        cast=ctx.config.cast,
        critique=final_critique,
        iteration=1,
    )

    if approved:
        job.status = JobStatus.RUNNING
        job_store.save_job(job)
        _persist_approved_plan(storyboard)
        return storyboard, script

    # ── Rejection path: one more re-plan with human notes ────────────────────
    log_event(logger, "plan_human_rejected", notes=rejection_notes)
    combined_notes = last_notes + ([rejection_notes] if rejection_notes else [])
    storyboard = await adapters.plan.run(
        PlanShotsRequest(script=script, cast=ctx.config.cast, critique_notes=combined_notes,
                         visual_context=visual_context)
    )
    if storyboard_transform is not None:
        storyboard = storyboard_transform(storyboard)
    storyboard, _ = await _critique_loop(
        storyboard, script, ctx, s.max_plan_iterations, visual_context=visual_context,
        storyboard_transform=storyboard_transform,
    )
    final_critique = await adapters.plan_critique.run(
        PlanCritiqueRequest(storyboard=storyboard, script=script, cast=ctx.config.cast)
    )

    storyboard = await _render_plan_overlays(storyboard, ctx, opts)

    # Second approval UI — if rejected again, fail the job
    approved, rejection_notes = await adapters.approval.request_approval(
        storyboard=storyboard,
        script=script,
        cast=ctx.config.cast,
        critique=final_critique,
        iteration=2,
    )

    if not approved:
        job.status = JobStatus.FAILED
        job_store.save_job(job)
        raise RuntimeError(
            f"Storyboard rejected twice by human reviewer. "
            f"Last notes: {rejection_notes!r}. "
            "Restart the job with --phase plan to generate a new storyboard."
        )

    job.status = JobStatus.RUNNING
    job_store.save_job(job)
    _persist_approved_plan(storyboard)
    return storyboard, script


async def _run_to_assembled_video(
    ctx: _JobContext,
    options: RunOptions | None = None,
) -> tuple[Script, FinalVideo]:
    """Run Phase 1 stages through assembled candidate video, but do not publish.

    When options.resume is True, stages whose artifact JSON already exists are
    loaded from disk and skipped.  options.phase controls which stage groups run:
      "plan"     — stop after plan_shots (returns early; FinalVideo will be None)
      "render"   — skip to per-shot loop (requires plan artifacts in artifact_store)
      "assemble" — skip to assemble_video (requires shot artifacts on disk)
      "all"      — full pipeline (default)
    """
    opts = options or RunOptions()
    config = ctx.config
    job = ctx.job
    adapters = ctx.adapters
    artifact_store = ctx.artifact_store
    job_store = ctx.job_store

    # ── Lazy import of model classes for artifact deserialization ─────────────
    from core.models.capabilities import (
        FetchMediaResult, TranscribeResult, ContentMetadata,
    )

    # ── Helper: run a stage or load its cached artifact ───────────────────────
    async def _stage(name, capability, request, result_cls):
        if opts.resume:
            cached = _load_artifact(job.job_id, name, result_cls, artifact_store)
            if cached is not None:
                logger.info("Resuming: skipping stage '%s' (artifact exists)", name)
                return cached
        return await run_stage(
            name, capability, request, job, artifact_store, job_store,
            stage_hook=opts.stage_hook,
            error_hook=opts.error_hook,
        )

    # ── Plan phase ────────────────────────────────────────────────────────────
    skip_plan = opts.phase == "render" or opts.phase == "assemble"
    # script_plan resumes from cached transcribe/analyze artifacts — force resume for those stages.
    resume_head = opts.phase == "script_plan"

    if not skip_plan:
        if resume_head:
            # script_plan: load artifacts saved by a previous "transcribe" phase run.
            fetch_result = _load_artifact(job.job_id, "fetch_media", FetchMediaResult, artifact_store)
            transcribe_result = _load_artifact(job.job_id, "transcribe", TranscribeResult, artifact_store)
            metadata = _load_artifact(job.job_id, "analyze_content", ContentMetadata, artifact_store)
            if fetch_result is None or transcribe_result is None or metadata is None:
                raise FileNotFoundError(
                    f"Phase 'script_plan' requires artifacts from a completed 'transcribe' phase "
                    f"for job '{job.job_id}'. Run the transcribe phase first."
                )
            # Optional — older jobs (pre-analyze_visuals) won't have it; ground with empty.
            visual_context = (
                _load_artifact(job.job_id, "analyze_visuals", VisualContext, artifact_store)
                or VisualContext()
            )
        else:
            # 1. fetch_media
            fetch_result = await _stage(
                "fetch_media", adapters.fetch_media,
                FetchMediaRequest(source_url=job.source_url),
                FetchMediaResult,
            )

            # 2. transcribe
            transcribe_result = await _stage(
                "transcribe", adapters.transcribe,
                TranscribeRequest(audio_uri=fetch_result.audio_uri),
                TranscribeResult,
            )

            # 3. analyze_content
            metadata = await _stage(
                "analyze_content", adapters.analyze,
                AnalyzeRequest(
                    transcript=transcribe_result,
                    channel_profile=config.channel_profile,
                ),
                ContentMetadata,
            )

            # 3b. analyze_visuals — describe the source video's settings so
            # adapt_script can ground scene backgrounds in the real footage.
            # Best-effort: empty for story jobs / remote URIs / failures.
            visual_context = await _stage(
                "analyze_visuals", adapters.analyze_visuals,
                AnalyzeVisualsRequest(
                    video_uri=fetch_result.video_uri,
                    segments=transcribe_result.segments,
                ),
                VisualContext,
            )

        # Stop here for "transcribe" phase — artifacts are saved, human can review.
        if opts.phase == "transcribe":
            logger.info("Phase 'transcribe' complete — stopping after analyze_content.")
            job.status = JobStatus.COMPLETED
            job_store.save_job(job)
            return None, None  # type: ignore[return-value]

        # 4. rights gate
        check_rights(job)

        if opts.render_mode == "source_audio":
            # ── source_audio path: skip adapt_script, build verbatim script ──
            script = _build_verbatim_script(
                transcribe_result, config.cast,
                visual_context=visual_context,
                rights_cleared=job.rights_cleared,
            )
            artifact_store.put_json(job.job_id, "adapt_script", script.model_dump(mode="json"))

            storyboard = await _stage(
                "plan_shots", adapters.plan,
                PlanShotsRequest(script=script, cast=config.cast, visual_context=visual_context),
                Storyboard,
            )
            # Deterministically re-chunk by max_shot_duration_sec so the
            # assembled video's total length exactly matches the source,
            # regardless of how many scenes/shots the LLM planned — each
            # chunk borrows its camera/action from the nearest original shot.
            storyboard = _rebuild_storyboard_by_duration(
                storyboard, script,
                total_duration=fetch_result.duration_sec,
                max_shot_duration_sec=config.settings.max_shot_duration_sec,
            )
            artifact_store.put_json(job.job_id, "plan_shots", storyboard.model_dump(mode="json"))

            storyboard = await _render_plan_overlays(storyboard, ctx, opts)
            log_event(logger, "source_audio_plan_ready",
                      shots=len(storyboard.shots), mode="source_audio")
        else:
            # 5. adapt_script
            script: Script = await _stage(
                "adapt_script", adapters.adapt,
                AdaptScriptRequest(
                    metadata=metadata,
                    cast=config.cast,
                    channel_profile=config.channel_profile,
                    language=opts.language,
                    visual_context=visual_context,
                ),
                Script,
            )
            if opts.render_mode == "re_voice":
                script = _apply_source_timing_to_script(
                    script,
                    transcribe_result,
                    total_duration=fetch_result.duration_sec,
                )
                artifact_store.put_json(
                    job.job_id, "adapt_script", script.model_dump(mode="json")
                )

            # 6. plan_shots
            storyboard: Storyboard = await _stage(
                "plan_shots", adapters.plan,
                PlanShotsRequest(script=script, cast=config.cast, visual_context=visual_context),
                Storyboard,
            )

            storyboard_transform = None
            if opts.render_mode == "re_voice":
                def storyboard_transform(sb: Storyboard) -> Storyboard:
                    timed = _apply_segment_timing_to_storyboard(sb, script)
                    return _rebuild_storyboard_by_duration(
                        timed, script,
                        total_duration=fetch_result.duration_sec,
                        max_shot_duration_sec=config.settings.max_shot_duration_sec,
                    )

            # 7. critique_plan loop + human approval gate
            storyboard, script = await _run_plan_critique_and_approval(
                storyboard=storyboard,
                script=script,
                ctx=ctx,
                opts=opts,
                visual_context=visual_context,
                storyboard_transform=storyboard_transform,
            )
            if opts.render_mode == "re_voice":
                storyboard = storyboard_transform(storyboard)
                artifact_store.put_json(
                    job.job_id, "plan_shots", storyboard.model_dump(mode="json")
                )

        if opts.phase in ("plan", "script_plan"):
            logger.info("Phase '%s' complete — stopping before render loop.", opts.phase)
            job.status = JobStatus.COMPLETED
            job_store.save_job(job)
            return script, None  # type: ignore[return-value]

    else:
        # Skip plan stages — load artifacts that the render/assemble phases need.
        script_data = artifact_store.get_json(job.job_id, "adapt_script")
        storyboard_data = artifact_store.get_json(job.job_id, "plan_shots")
        if script_data is None or storyboard_data is None:
            raise RuntimeError(
                f"Phase '{opts.phase}' requires completed plan artifacts for job "
                f"'{job.job_id}'.  Run --phase plan first."
            )
        script = Script.model_validate(script_data)
        storyboard = Storyboard.model_validate(storyboard_data)
        logger.info("Loaded plan artifacts for job %s (script + storyboard)", job.job_id)
        # fetch_result not available from plan block — set to None; loaded below if needed.
        fetch_result = None  # type: ignore[assignment]

    # ── Resolve source_audio_uri for source_audio mode ──────────────────────
    source_audio_uri: str | None = None
    if opts.render_mode == "source_audio":
        if fetch_result is None:
            fetch_result = _load_artifact(job.job_id, "fetch_media", FetchMediaResult, artifact_store)
        if fetch_result is not None:
            source_audio_uri = fetch_result.audio_uri
        if not source_audio_uri or source_audio_uri.startswith("story://"):
            raise ValueError(
                "source_audio mode requires a real audio file from fetch_media. "
                "Story-kind jobs do not have source audio."
            )
        if not _storyboard_has_source_timing(storyboard):
            transcribe_for_timing = _load_artifact(
                job.job_id, "transcribe", TranscribeResult, artifact_store
            )
            if transcribe_for_timing is None:
                raise ValueError(
                    "source_audio mode requires a transcribe artifact to calculate "
                    "source_start_sec/source_end_sec before rendering."
                )
            visual_for_timing = (
                _load_artifact(job.job_id, "analyze_visuals", VisualContext, artifact_store)
                or VisualContext()
            )
            script, storyboard = _retime_source_audio_plan(
                storyboard,
                transcribe_for_timing,
                fetch_result,
                config.cast,
                visual_context=visual_for_timing,
                rights_cleared=job.rights_cleared,
                max_shot_duration_sec=config.settings.max_shot_duration_sec,
            )
            artifact_store.put_json(
                job.job_id, "adapt_script", script.model_dump(mode="json")
            )
            artifact_store.put_json(
                job.job_id, "plan_shots", storyboard.model_dump(mode="json")
            )
            log_event(
                logger,
                "source_audio_plan_retimed",
                job_id=job.job_id,
                shots=len(storyboard.shots),
            )
    _validate_timed_storyboard(storyboard, opts.render_mode)

    # ── Render phase ──────────────────────────────────────────────────────────
    if opts.phase == "assemble":
        # Skip shot loop — reconstruct clip/audio lists from existing files.
        synced_clips, audio_tracks = _collect_existing_shot_artifacts(
            storyboard, script, config.cast, ctx.work_dir,
            video_adapter=config.settings.video_adapter,
            lipsync_adapter=config.settings.lipsync_adapter,
            render_mode=opts.render_mode,
        )
        shots_for_clips = storyboard.shots  # one clip per shot, same order
    else:
        # Release LLM from VRAM before the GPU-heavy shot loop, and make sure
        # VRAM-managed video (Wan) and voice (Fish S2) models are not resident
        # during render_character — neither is needed until Phase B. Also free
        # ComfyUI unconditionally: it's invisible to the two calls above unless
        # it's THIS job's own render/video adapter, so a resident model left
        # over from an earlier, different job (e.g. render_adapter=comfyui_flux
        # or video_adapter=ltx) would otherwise sit stranded through this job's
        # entire render phase.
        _unload_ollama_model(config.settings.llm_base_url, config.settings.llm_model)
        await ensure_video_model_unloaded(adapters.video)
        await ensure_video_model_unloaded(adapters.voice)
        await free_comfyui(config.settings.comfyui_base_url)

        shots_to_run = (
            [s for s in storyboard.shots if s.shot_id == opts.only_shot]
            if opts.only_shot
            else storyboard.shots
        )
        if opts.only_shot and not shots_to_run:
            raise ValueError(f"Shot '{opts.only_shot}' not found in storyboard.")

        total_shots = len(shots_to_run)
        if opts.stage_hook:
            opts.stage_hook(
                "render_loop", "stage_started",
                message=f"Rendering {total_shots} shot{'s' if total_shots != 1 else ''}",
            )

        # ── Phase A: render all candidates + VLM critique (sequential per shot) ──
        if opts.user_images:
            # story_images mode: user supplied reference images — skip Flux
            # rendering and VLM critique entirely (no LoRA/Track-B gate either).
            critique_results = _build_user_image_critiques(
                shots_to_run, opts.user_images, config.cast
            )
            log_event(logger, "phase_a_skipped_user_images",
                      members=sorted(opts.user_images.keys()))
        else:
            # Batch-capable adapters (musubi_flux) render every pending shot in
            # one subprocess per LoRA — one 64 GB model load instead of one per
            # candidate. `is True` so mock adapters' truthy attrs don't opt in.
            prefetched: dict[str, ImageSet] = {}
            if getattr(adapters.render, "supports_batch", False) is True:
                prefetched = await _phase_a_prefetch_renders(
                    shots_to_run, config.cast, adapters, ctx.work_dir, opts
                )
            critique_results = []
            for shot in shots_to_run:
                cr = await _render_shot_candidates(
                    shot, config.cast, adapters, ctx.work_dir, opts,
                    prefetched=prefetched.get(shot.shot_id),
                )
                critique_results.append(cr)

        # ── Image approval gate (single UI for all shots) ─────────────────────
        approved_uris = await _run_image_approval_gate(
            shots_to_run, critique_results, adapters, cast_id=config.cast.id
        )

        # Render phase is done — free the render adapter's VRAM if it's a
        # managed one (comfyui_flux fallback; musubi-tuner already self-released
        # as a one-shot subprocess), then bring up the VRAM-managed video (Wan)
        # and voice (Fish S2) models: unload Ollama, wait for VRAM to settle,
        # load, poll until ready.
        if is_managed(adapters.render):
            await adapters.render.unload()
        await prepare_video_model(
            adapters.video, config.settings, notify=opts.stage_hook
        )
        await prepare_voice_model(
            adapters.voice, config.settings, notify=opts.stage_hook
        )

        # ── Phase B: synthesize voice + generate video with approved image ─────
        if opts.stage_hook:
            opts.stage_hook(
                "video_loop", "stage_started",
                message=f"Generating video for {total_shots} shot{'s' if total_shots != 1 else ''}",
            )
        synced_clips = []
        audio_tracks = []
        for i, (shot, image_uri) in enumerate(zip(shots_to_run, approved_uris), start=1):
            synced, audio = await _generate_shot_video(
                shot, script, config.cast, adapters, ctx.work_dir, image_uri, opts,
                shot_index=i, total_shots=total_shots,
                render_mode=opts.render_mode, source_audio_uri=source_audio_uri,
            )
            synced_clips.append(synced)
            audio_tracks.append(audio)
        shots_for_clips = shots_to_run  # one clip per shot, same order

        if opts.phase == "render":
            logger.info("Phase 'render' complete — stopping before assemble.")
            job.status = JobStatus.COMPLETED
            job_store.save_job(job)
            return script, None  # type: ignore[return-value]

    combined_audio = await _concat_audio(audio_tracks, ctx.work_dir, adapters.ffmpeg_bin)

    # Chart/diagram overlay panels: absolute time windows from actual clip durations.
    overlay_windows = await _build_overlay_windows(
        shots_for_clips, synced_clips, config.settings.ffprobe_bin
    )

    # Video adapters (LTX/Wan) echo back the *requested* duration_sec, not the
    # actual rendered length — both round to their own frame-count constraints
    # (e.g. an 8.0s request often renders ~7.7s). assemble_video's crossfade
    # math schedules its xfade offset from clip.duration_sec, so an
    # over-stated duration pushes that offset past the clip's real end and
    # degrades/truncates the whole output (reproduced: a claimed ~15s 2-shot
    # assembly collapsing to the first clip's ~7.7s alone). Correct it here
    # with the same ffprobe probe _build_overlay_windows already trusts.
    synced_clips = await _probe_clip_durations(synced_clips, config.settings.ffprobe_bin)

    # ── Assemble phase ────────────────────────────────────────────────────────
    # 8. assemble_video
    final_video = await run_stage(
        "assemble_video", adapters.assemble,
        AssembleRequest(
            clips=synced_clips,
            audio=combined_audio,
            caption_text=script.caption_text,
            aspect_ratio=config.channel_profile.aspect_ratio,
            made_for_kids=config.channel_profile.made_for_kids,
            disclosure_label_required=config.channel_profile.disclosure_label_required,
            overlays=overlay_windows,
            audio_tracks=audio_tracks,
            preserve_timing=opts.render_mode in {"source_audio", "re_voice"},
        ),
        job, artifact_store, job_store,
        stage_hook=opts.stage_hook,
        error_hook=opts.error_hook,
    )

    return script, final_video


async def _probe_duration_sec(path: str, ffprobe_bin: str) -> float | None:
    """Actual media duration via ffprobe; None on any failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            ffprobe_bin, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        return float(stdout.decode().strip())
    except Exception:
        return None


async def _probe_clip_durations(
    clips: list[VideoClip], ffprobe_bin: str
) -> list[VideoClip]:
    """Replace each clip's duration_sec with its real ffprobe-measured length.

    Video adapters echo back the requested duration, not what actually got
    rendered (LTX/Wan both round to their own frame-count constraints), so
    assemble_video must not trust it for crossfade timing. Falls back to the
    original value if ffprobe fails (e.g. in tests with a fake file).
    """
    corrected = []
    for clip in clips:
        actual = await _probe_duration_sec(clip.uri, ffprobe_bin)
        corrected.append(
            clip if actual is None else clip.model_copy(update={"duration_sec": actual})
        )
    return corrected


async def _build_overlay_windows(
    shots: list[Shot],
    clips: list[VideoClip],
    ffprobe_bin: str,
) -> "list[OverlayWindow]":
    """Compute the absolute time window of each shot's overlay in the final video.

    Offsets use ffprobe-measured clip durations (fallback: the words/2 estimate)
    because native-lipsync clips follow the TTS audio length, not the estimate —
    per-shot drift would misplace later overlays.
    """
    from core.models.capabilities import OverlayWindow

    # Common case: no overlays at all — skip the per-clip ffprobe entirely.
    if not any(
        s.overlay is not None and s.overlay.png_uri and Path(s.overlay.png_uri).exists()
        for s in shots
    ):
        return []

    windows: list[OverlayWindow] = []
    offset = 0.0
    for shot, clip in zip(shots, clips):
        actual = await _probe_duration_sec(clip.uri, ffprobe_bin) or clip.duration_sec
        overlay = shot.overlay
        if overlay is not None and overlay.png_uri and Path(overlay.png_uri).exists():
            visible = min(overlay.duration_sec or actual, actual)
            start = offset + 0.25          # skip the very first frames of the cut
            end = offset + visible - 0.1
            if end > start:
                windows.append(OverlayWindow(
                    shot_id=shot.shot_id,
                    png_uri=overlay.png_uri,
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                ))
        offset += actual
    return windows


def _collect_existing_shot_artifacts(
    storyboard: Storyboard,
    script: Script,
    cast: Cast,
    work_dir: Path,
    video_adapter: str = "wan_s2v",
    lipsync_adapter: str = "latentsync",
    render_mode: str = "full",
) -> tuple[list[VideoClip], list[AudioTrack]]:
    """Reconstruct clip/audio lists from files written by a previous render phase.

    Checks adapter-namespaced paths first (``video/<adapter>/<shot_id>/clip.mp4``),
    then falls back to the legacy flat layout (``video/<shot_id>/clip.mp4``) for
    jobs that ran before the namespace migration.

    Raises RuntimeError for any shot where no completion marker is found.
    """
    member_map = {m.id: m for m in cast.members}
    clips: list[VideoClip] = []
    audios: list[AudioTrack] = []
    for shot in storyboard.shots:
        candidates = [
            work_dir / "synced" / f"{video_adapter}_{lipsync_adapter}" / shot.shot_id / "synced.mp4",
            work_dir / "synced" / video_adapter / shot.shot_id / "synced.mp4",
            work_dir / "video" / video_adapter / shot.shot_id / "clip.mp4",
            work_dir / "synced" / shot.shot_id / "synced.mp4",
            work_dir / "video" / shot.shot_id / "clip.mp4",
        ]
        video_path = next((p for p in candidates if p.exists()), None)
        if video_path is None:
            raise RuntimeError(
                f"Phase 'assemble' requires a completed video for all shots, "
                f"but no completion marker found for shot '{shot.shot_id}'. "
                f"Run --phase render first."
            )
        speaker_id = shot.characters_on_screen[0]
        audio_uri = (
            _existing_audio_for_shot(work_dir, shot, speaker_id, render_mode)
            or str(video_path)
        )
        clips.append(VideoClip(uri=str(video_path), duration_sec=shot.duration_sec))
        audios.append(AudioTrack(uri=audio_uri, duration_sec=shot.duration_sec))
    return clips, audios


async def _publish_candidate(
    ctx: _JobContext,
    script: Script,
    final_video: FinalVideo,
    *,
    language: str = "en",
    stage_hook: Callable[[str, str], None] | None = None,
) -> None:
    """Publish an assembled candidate to the manual review folder."""
    await run_stage(
        "publish", ctx.adapters.publish,
        PublishRequest(
            video=final_video,
            rights_cleared=ctx.job.rights_cleared,
            made_for_kids=ctx.config.channel_profile.made_for_kids,
            disclosure_label_required=ctx.config.channel_profile.disclosure_label_required,
            learning_objective_summary=script.learning_objective.success_phrase,
            language=language,
        ),
        ctx.job, ctx.artifact_store, ctx.job_store,
        stage_hook=stage_hook,
    )


async def _critique_candidate(
    ctx: _JobContext,
    script: Script,
    final_video: FinalVideo,
    attempt: int,
    *,
    stage_hook: Callable[[str, str], None] | None = None,
) -> CritiqueResult:
    """Run and persist one critique attempt."""
    return await run_stage(
        f"critique_attempt_{attempt}",
        ctx.adapters.critique,
        CritiqueRequest(
            video_uri=final_video.uri,
            script=script,
            channel_profile_id=ctx.config.channel_profile.id,
        ),
        ctx.job,
        ctx.artifact_store,
        ctx.job_store,
        stage_hook=stage_hook,
    )


def _critique_reason(result: CritiqueResult) -> str:
    reasons = "; ".join(result.reasons) if result.reasons else "no reason provided"
    return f"critique verdict={result.verdict}: {reasons}"


def _mark_job_failed(ctx: _JobContext, reason: str, *, blocked: bool = False) -> None:
    ctx.job.status = JobStatus.BLOCKED if blocked else JobStatus.FAILED
    log_event(logger, "job_failed", job_id=ctx.job.job_id, reason=reason)
    ctx.job_store.save_job(ctx.job)


async def run_pipeline_job(
    source_url: str,
    rights_cleared: bool = False,
    app_config: AppConfig | None = None,
    options: RunOptions | None = None,
    resume_job_id: str | None = None,
    job_id: str | None = None,
    target_language: str | None = None,
    approval_overrides: dict | None = None,
) -> Job:
    """
    Full Phase 1 pipeline: URL → review-folder MP4 + metadata sidecar.

    Stage sequence
    ~~~~~~~~~~~~~~
    1. fetch_media      — yt-dlp download + ffmpeg audio extraction
    2. transcribe       — faster-whisper speech-to-text
    3. analyze_content  — LLM extracts ContentMetadata + LearningObjective
    4. check_rights     — pipeline gate; blocks if rights_cleared=False
    5. adapt_script     — LLM writes original Script; adapter injects guardrails
    6. plan_shots       — LLM plans camera/action; adapter derives shot structure
    7. [per shot]       — render_character → synthesize_voice → generate_video → lip_sync
    8. assemble_video   — ffmpeg concat + scale + caption + disclosure label
    9. publish          — copy to review folder + metadata.json sidecar

    Args:
        source_url:     URL of the reference video to ingest.
        rights_cleared: Caller asserts the source is cleared for transformation.
                        If False, the job is blocked at stage 4.
        app_config:     Override the default YAML-loaded config (useful in tests).
    """
    config = app_config or load_app_config()

    # Resolve target language(s): "both" expands to ["en", "hi"]
    effective_lang = target_language or config.settings.target_language
    if effective_lang == "both":
        languages = ["en", "hi"]
    else:
        languages = [effective_lang]

    # For multi-language runs, execute sequentially and return the last job.
    last_job: Job | None = None
    for lang in languages:
        # Copy all fields from caller options (preserves stage_hook, error_hook, etc.)
        # and override language only.
        lang_opts = _dc_replace(options or RunOptions(), language=lang)
        # resume needs a job id to locate cached artifacts — either an existing
        # job (resume_job_id) or a fresh job created under a caller-chosen job_id
        # whose artifacts were pre-seeded (story input mode).
        if lang_opts.resume and not (resume_job_id or job_id):
            raise ValueError("resume=True requires resume_job_id or job_id to be set.")
        ctx = (
            _restore_job_context(resume_job_id, config, approval_overrides=approval_overrides)
            if resume_job_id
            else _make_job_context(
                source_url, rights_cleared, config,
                job_id=job_id,
                approval_overrides=approval_overrides,
            )
        )
        last_job = await _run_single_language_job(ctx, lang_opts)

    return last_job  # type: ignore[return-value]


async def _run_single_language_job(ctx: "_JobContext", opts: RunOptions) -> Job:
    """Run the pipeline for a single target language."""
    job = ctx.job

    try:
        script, final_video = await _run_to_assembled_video(ctx, opts)
        if final_video is None:
            # phase="plan" or phase="render" — no video to publish, already marked complete
            return job
        await _publish_candidate(ctx, script, final_video, language=opts.language, stage_hook=opts.stage_hook)

        job.status = JobStatus.COMPLETED
        ctx.job_store.save_job(job)
        log_event(logger, "job_completed", job_id=job.job_id)
        return job

    except StageError as exc:
        # check_rights sets BLOCKED itself; all other StageErrors → FAILED
        if job.status not in (JobStatus.BLOCKED,):
            job.status = JobStatus.FAILED
        log_event(
            logger, "job_failed",
            job_id=job.job_id, stage=exc.stage_name, reason=exc.reason,
        )
        ctx.job_store.save_job(job)
        raise

    except Exception as exc:
        job.status = JobStatus.FAILED
        log_event(logger, "job_failed", job_id=job.job_id, reason=str(exc))
        ctx.job_store.save_job(job)
        raise


async def run_with_critique(
    source_url: str,
    rights_cleared: bool = False,
    app_config: AppConfig | None = None,
) -> Job:
    """
    Phase 2 pipeline: generate candidate → critique → regenerate or publish.

    `Settings.max_regenerations` means retries after the first candidate. For
    example, max_regenerations=3 allows up to 4 assembled candidates.
    Critiques are persisted as `critique_attempt_1`, `critique_attempt_2`, ...
    """
    config = app_config or load_app_config()
    ctx = _make_job_context(source_url, rights_cleared, config)
    max_attempts = max(1, config.settings.max_regenerations + 1)

    try:
        for attempt in range(1, max_attempts + 1):
            log_event(
                logger,
                "candidate_attempt_started",
                job_id=ctx.job.job_id,
                attempt=attempt,
                max_attempts=max_attempts,
            )

            script, final_video = await _run_to_assembled_video(ctx)

            # Frame critique only needs the LLM/VLM — free whatever VRAM-managed
            # render/video/voice models Phase A/B left resident before it runs.
            if is_managed(ctx.adapters.render):
                await ctx.adapters.render.unload()
            await ensure_video_model_unloaded(ctx.adapters.video)
            await ensure_video_model_unloaded(ctx.adapters.voice)

            critique = await _critique_candidate(ctx, script, final_video, attempt)

            log_event(
                logger,
                "candidate_critiqued",
                job_id=ctx.job.job_id,
                attempt=attempt,
                verdict=critique.verdict,
                reasons=critique.reasons,
            )

            if critique.verdict == "pass":
                await _publish_candidate(ctx, script, final_video)
                ctx.job.status = JobStatus.COMPLETED
                ctx.job_store.save_job(ctx.job)
                log_event(logger, "job_completed", job_id=ctx.job.job_id)
                return ctx.job

            if critique.verdict == "reject":
                reason = _critique_reason(critique)
                _mark_job_failed(ctx, reason, blocked=True)
                raise StageError("critique", reason)

            if attempt < max_attempts:
                log_event(
                    logger,
                    "candidate_regeneration_requested",
                    job_id=ctx.job.job_id,
                    attempt=attempt,
                    reasons=critique.reasons,
                )
                continue

            reason = (
                f"max_regenerations exhausted after {max_attempts} attempts; "
                f"{_critique_reason(critique)}"
            )
            _mark_job_failed(ctx, reason)
            raise StageError("critique", reason)

        # Unreachable, but keeps type-checkers honest if loop bounds change.
        reason = "critique loop exited without a verdict"
        _mark_job_failed(ctx, reason)
        raise StageError("critique", reason)

    except StageError as exc:
        if ctx.job.status not in (JobStatus.BLOCKED, JobStatus.FAILED):
            ctx.job.status = JobStatus.FAILED
            ctx.job_store.save_job(ctx.job)
        log_event(
            logger, "job_failed",
            job_id=ctx.job.job_id, stage=exc.stage_name, reason=exc.reason,
        )
        raise

    except Exception as exc:
        ctx.job.status = JobStatus.FAILED
        log_event(logger, "job_failed", job_id=ctx.job.job_id, reason=str(exc))
        ctx.job_store.save_job(ctx.job)
        raise
