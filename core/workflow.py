import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Any, Literal

from core.config import AppConfig, load_app_config
from core.executor import StageError, check_rights, run_stage
from core.gpu_sequencer import (
    ensure_video_model_unloaded,
    prepare_video_model,
    unload_ollama_model,
)
from core.models.capabilities import (
    AnalyzeRequest,
    AssembleRequest,
    AudioTrack,
    CritiqueRequest,
    CritiqueResult,
    FetchMediaRequest,
    FinalVideo,
    ImageApprovalRequest,
    ImageCritiqueRequest,
    ImageCritiqueResult,
    LipSyncRequest,
    PlanShotsRequest,
    PublishRequest,
    RenderCharacterRequest,
    TranscribeRequest,
    VideoClip,
    VideoRequest,
    VoiceRequest,
)
from core.models.capabilities import AdaptScriptRequest
from core.models.content import Script, Shot, Storyboard
from core.models.job import Job, JobStatus
from core.models.profile import Cast
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
    """
    phase: Phase = "all"
    resume: bool = False
    only_shot: str | None = None
    language: str = "en"
    stage_hook: Callable[[str, str], None] | None = None
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
    adapt: object
    plan: object
    plan_critique: object
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
        return LtxAdapter(work_dir=work_dir / "video", base_url=s.ltx_base_url)
    from adapters.generate_video.wan_adapter import WanAdapter
    return WanAdapter(work_dir=work_dir / "video", base_url=s.wan_base_url)


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
    from adapters.assemble_video.ffmpeg_adapter import FfmpegAssembleAdapter
    from adapters.critique.vlm_adapter import VlmCritiqueAdapter
    from adapters.fetch_media.ytdlp_adapter import YtDlpAdapter
    from adapters.lip_sync.lip_sync_adapter import LipSyncAdapter
    from adapters.plan_shots.llm_adapter import LlmPlanShotsAdapter
    from adapters.publish.manual_adapter import ManualPublishAdapter
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
        adapt=LlmAdaptScriptAdapter(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
        ),
        plan=LlmPlanShotsAdapter(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
        ),
        plan_critique=LlmPlanCritiqueAdapter(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
        ),
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
        lipsync=LipSyncAdapter(work_dir=work_dir / "synced", base_url=s.lipsync_base_url),
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


async def _render_shot_candidates(
    shot: Shot,
    cast: Cast,
    adapters: _Adapters,
    work_dir: Path,
    options: RunOptions | None = None,
) -> "ImageCritiqueResult":
    """Phase A: render N candidates for one shot, critique them, return the winner.

    Runs render_character (N images) then VLM image critique.
    Resume: if all candidate PNGs and a cached critique result already exist,
    skips both render and critique (the VLM call is a real, non-trivial cost —
    ~20-40s per shot — that would otherwise redo on every retry/resume).
    """
    from core.models.capabilities import ImageCritiqueRequest, ImageCritiqueResult, ImageSet

    opts = options or RunOptions()
    member_map = {m.id: m for m in cast.members}
    speaker_id = shot.characters_on_screen[0]
    speaker = member_map[speaker_id]

    expression = None
    render_dir = work_dir / "renders" / speaker_id
    render_dir.mkdir(parents=True, exist_ok=True)

    # Count expected candidates from the adapter's num_images setting
    num_candidates = getattr(adapters.render, "_num_images", 1)
    existing_pngs = sorted(render_dir.glob("render_??.png"))
    critique_path = work_dir / "critique" / f"{shot.shot_id}.json"

    if opts.resume and len(existing_pngs) >= num_candidates and critique_path.exists():
        logger.info("Skipping render_character + image_critique for %s (cached)", shot.shot_id)
        return ImageCritiqueResult.model_validate_json(critique_path.read_text())

    if opts.resume and len(existing_pngs) >= num_candidates:
        logger.info("Skipping render_character for %s (all %d candidates exist)",
                    shot.shot_id, num_candidates)
        render_result = ImageSet(member_id=speaker_id,
                                 images=[str(p) for p in existing_pngs[:num_candidates]])
    else:
        render_result = await adapters.render.run(
            RenderCharacterRequest(
                member=speaker,
                setting=shot.setting,
                expression=expression,
            )
        )

    shot_prompt = f"{speaker.name} ({speaker.visual_descriptor}) in {shot.setting}; action: {shot.action}"
    critique = await adapters.image_critique.run(
        ImageCritiqueRequest(
            shot_id=shot.shot_id,
            shot_prompt=shot_prompt,
            candidate_uris=render_result.images,
            cast_descriptor=speaker.visual_descriptor,
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


async def _generate_shot_video(
    shot: Shot,
    script: Script,
    cast: Cast,
    adapters: _Adapters,
    work_dir: Path,
    image_uri: str,
    options: RunOptions | None = None,
) -> tuple[VideoClip, AudioTrack]:
    """Phase B: synthesize voice + generate video for one shot using the approved image.

    Skips stages whose output files already exist when opts.resume is True.
    """
    opts = options or RunOptions()
    member_map = {m.id: m for m in cast.members}
    speaker_id = shot.characters_on_screen[0]
    speaker = member_map[speaker_id]

    first_line = (
        _resolve_line(shot.dialogue_line_refs[0], script)
        if shot.dialogue_line_refs
        else None
    )
    expression = first_line.expression if first_line else None
    line_text = first_line.text if first_line else ""

    native_lipsync = getattr(adapters.video, "native_lipsync", False)

    if native_lipsync:
        done_path = work_dir / "video" / shot.shot_id / "clip.mp4"
    else:
        done_path = work_dir / "synced" / shot.shot_id / "synced.mp4"

    if opts.resume and done_path.exists():
        logger.info("Skipping video generation for %s (%s exists)", shot.shot_id, done_path.name)
        audio_files = sorted((work_dir / "audio" / speaker_id).glob("*.wav"))
        audio_uri = str(audio_files[0]) if audio_files else str(done_path)
        return (
            VideoClip(uri=str(done_path), duration_sec=shot.duration_sec),
            AudioTrack(uri=audio_uri, duration_sec=shot.duration_sec),
        )

    log_event(logger, "shot_video_started", shot_id=shot.shot_id, speaker=speaker_id)

    # ── 1. synthesize_voice ───────────────────────────────────────────────────
    audio_files = sorted((work_dir / "audio" / speaker_id).glob("*.wav"))
    if opts.resume and audio_files:
        logger.info("Skipping synthesize_voice for %s (audio exists)", shot.shot_id)
        audio_track = AudioTrack(uri=str(audio_files[0]), duration_sec=shot.duration_sec)
    else:
        audio_track = await adapters.voice.run(
            VoiceRequest(
                text=line_text,
                voice_profile_ref=speaker.voice_profile_ref,
                speaker_id=speaker_id,
                expression=expression,
                language=opts.language if opts else "en",
            )
        )

    # ── 2. generate_video ─────────────────────────────────────────────────────
    clip_path = work_dir / "video" / shot.shot_id / "clip.mp4"
    if opts.resume and clip_path.exists():
        logger.info("Skipping generate_video for %s (clip.mp4 exists)", shot.shot_id)
        clip = VideoClip(uri=str(clip_path), duration_sec=shot.duration_sec)
    else:
        clip = await adapters.video.run(
            VideoRequest(
                image_uri=image_uri,
                action=shot.action,
                duration_sec=shot.duration_sec,
                shot_id=shot.shot_id,
                audio_uri=audio_track.uri if native_lipsync else None,
            )
        )

    # ── 3. lip_sync (skipped when video adapter handles it natively) ──────────
    if native_lipsync:
        log_event(logger, "lip_sync_skipped", shot_id=shot.shot_id, reason="native_lipsync")
        synced = clip
    else:
        synced = await adapters.lipsync.run(
            LipSyncRequest(
                video_uri=clip.uri,
                audio_uri=audio_track.uri,
                shot_id=shot.shot_id,
            )
        )

    log_event(logger, "shot_completed", shot_id=shot.shot_id)
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
                PlanShotsRequest(script=script, cast=config.cast, critique_notes=notes)
            )

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


async def _run_plan_critique_and_approval(
    storyboard: Storyboard,
    script: Script,
    ctx: "_JobContext",
    opts: "RunOptions",
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

    # ── LLM critique loop ─────────────────────────────────────────────────────
    storyboard, last_notes = await _critique_loop(
        storyboard, script, ctx, s.max_plan_iterations
    )

    # Reconstruct final critique result for the UI (re-run to get fresh scores)
    final_critique = await adapters.plan_critique.run(
        PlanCritiqueRequest(storyboard=storyboard, script=script, cast=ctx.config.cast)
    )

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
        return storyboard, script

    # ── Rejection path: one more re-plan with human notes ────────────────────
    log_event(logger, "plan_human_rejected", notes=rejection_notes)
    combined_notes = last_notes + ([rejection_notes] if rejection_notes else [])
    storyboard = await adapters.plan.run(
        PlanShotsRequest(script=script, cast=ctx.config.cast, critique_notes=combined_notes)
    )
    storyboard, _ = await _critique_loop(storyboard, script, ctx, s.max_plan_iterations)
    final_critique = await adapters.plan_critique.run(
        PlanCritiqueRequest(storyboard=storyboard, script=script, cast=ctx.config.cast)
    )

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

        # Stop here for "transcribe" phase — artifacts are saved, human can review.
        if opts.phase == "transcribe":
            logger.info("Phase 'transcribe' complete — stopping after analyze_content.")
            job.status = JobStatus.COMPLETED
            job_store.save_job(job)
            return None, None  # type: ignore[return-value]

        # 4. rights gate
        check_rights(job)

        # 5. adapt_script
        script: Script = await _stage(
            "adapt_script", adapters.adapt,
            AdaptScriptRequest(
                metadata=metadata,
                cast=config.cast,
                channel_profile=config.channel_profile,
                language=opts.language,
            ),
            Script,
        )

        # 6. plan_shots
        storyboard: Storyboard = await _stage(
            "plan_shots", adapters.plan,
            PlanShotsRequest(script=script, cast=config.cast),
            Storyboard,
        )

        # 7. critique_plan loop + human approval gate
        storyboard, script = await _run_plan_critique_and_approval(
            storyboard=storyboard,
            script=script,
            ctx=ctx,
            opts=opts,
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

    # ── Render phase ──────────────────────────────────────────────────────────
    if opts.phase == "assemble":
        # Skip shot loop — reconstruct clip/audio lists from existing files.
        synced_clips, audio_tracks = _collect_existing_shot_artifacts(
            storyboard, script, config.cast, ctx.work_dir
        )
    else:
        # Release LLM from VRAM before the GPU-heavy shot loop, and make sure a
        # VRAM-managed video model (Wan) is not resident during render_character.
        _unload_ollama_model(config.settings.llm_base_url, config.settings.llm_model)
        await ensure_video_model_unloaded(adapters.video)

        shots_to_run = (
            [s for s in storyboard.shots if s.shot_id == opts.only_shot]
            if opts.only_shot
            else storyboard.shots
        )
        if opts.only_shot and not shots_to_run:
            raise ValueError(f"Shot '{opts.only_shot}' not found in storyboard.")

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
            critique_results = []
            for shot in shots_to_run:
                cr = await _render_shot_candidates(shot, config.cast, adapters, ctx.work_dir, opts)
                critique_results.append(cr)

        # ── Image approval gate (single UI for all shots) ─────────────────────
        approved_uris = await _run_image_approval_gate(
            shots_to_run, critique_results, adapters, cast_id=config.cast.id
        )

        # Render phase is done — bring the VRAM-managed video model (Wan) up:
        # unload Ollama, wait for VRAM to settle, load, poll until ready.
        await prepare_video_model(
            adapters.video, config.settings, notify=opts.stage_hook
        )

        # ── Phase B: synthesize voice + generate video with approved image ─────
        synced_clips = []
        audio_tracks = []
        for shot, image_uri in zip(shots_to_run, approved_uris):
            synced, audio = await _generate_shot_video(
                shot, script, config.cast, adapters, ctx.work_dir, image_uri, opts
            )
            synced_clips.append(synced)
            audio_tracks.append(audio)

        if opts.phase == "render":
            logger.info("Phase 'render' complete — stopping before assemble.")
            job.status = JobStatus.COMPLETED
            job_store.save_job(job)
            return script, None  # type: ignore[return-value]

    combined_audio = await _concat_audio(audio_tracks, ctx.work_dir, adapters.ffmpeg_bin)

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
        ),
        job, artifact_store, job_store,
        stage_hook=opts.stage_hook,
        error_hook=opts.error_hook,
    )

    return script, final_video


def _collect_existing_shot_artifacts(
    storyboard: Storyboard,
    script: Script,
    cast: Cast,
    work_dir: Path,
) -> tuple[list[VideoClip], list[AudioTrack]]:
    """Reconstruct clip/audio lists from files written by a previous render phase.

    Checks both possible completion markers, since which one exists depends on
    the video adapter: native-lipsync adapters (LTX) write video/<shot_id>/clip.mp4
    and skip lip_sync entirely; non-native adapters (Wan) go through a separate
    lip_sync stage and write synced/<shot_id>/synced.mp4.

    Raises RuntimeError for any shot where neither marker is present.
    """
    member_map = {m.id: m for m in cast.members}
    clips: list[VideoClip] = []
    audios: list[AudioTrack] = []
    for shot in storyboard.shots:
        native_path = work_dir / "video" / shot.shot_id / "clip.mp4"
        synced_path = work_dir / "synced" / shot.shot_id / "synced.mp4"
        video_path = native_path if native_path.exists() else synced_path
        if not video_path.exists():
            raise RuntimeError(
                f"Phase 'assemble' requires a completed video for all shots, "
                f"but neither '{native_path}' nor '{synced_path}' exists. "
                f"Run --phase render first."
            )
        speaker_id = shot.characters_on_screen[0]
        audio_files = sorted((work_dir / "audio" / speaker_id).glob("*.wav"))
        audio_uri = str(audio_files[0]) if audio_files else str(video_path)
        clips.append(VideoClip(uri=str(video_path), duration_sec=shot.duration_sec))
        audios.append(AudioTrack(uri=audio_uri, duration_sec=shot.duration_sec))
    return clips, audios


async def _publish_candidate(
    ctx: _JobContext,
    script: Script,
    final_video: FinalVideo,
    *,
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
        await _publish_candidate(ctx, script, final_video, stage_hook=opts.stage_hook)

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
