import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from core.config import AppConfig, load_app_config
from core.executor import StageError, check_rights, run_stage
from core.models.capabilities import (
    AnalyzeRequest,
    AssembleRequest,
    AudioTrack,
    CritiqueRequest,
    CritiqueResult,
    FetchMediaRequest,
    FinalVideo,
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
    render: object
    voice: object
    video: object
    lipsync: object
    assemble: object
    critique: object
    publish: object
    ffmpeg_bin: str = field(default="ffmpeg")


def _make_adapters(config: AppConfig, work_dir: Path) -> _Adapters:
    """Instantiate all Phase 1 adapters with job-scoped work directories."""
    from adapters.adapt_script.llm_adapter import LlmAdaptScriptAdapter
    from adapters.analyze_content.llm_adapter import LlmAnalyzeAdapter
    from adapters.assemble_video.ffmpeg_adapter import FfmpegAssembleAdapter
    from adapters.critique.vlm_adapter import VlmCritiqueAdapter
    from adapters.fetch_media.ytdlp_adapter import YtDlpAdapter
    from adapters.generate_video.wan_adapter import WanAdapter
    from adapters.lip_sync.lip_sync_adapter import LipSyncAdapter
    from adapters.plan_shots.llm_adapter import LlmPlanShotsAdapter
    from adapters.publish.manual_adapter import ManualPublishAdapter
    from adapters.render_character.diffusion_adapter import DiffusionRenderAdapter
    from adapters.synthesize_voice.tts_adapter import TtsAdapter
    from adapters.transcribe.whisper_adapter import WhisperAdapter

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
        render=DiffusionRenderAdapter(
            work_dir=work_dir / "renders",
            base_url=s.sd_base_url,
            lora_dir=s.lora_dir,
            allow_placeholder_lora=s.render_allow_placeholder_lora,
        ),
        voice=TtsAdapter(
            work_dir=work_dir / "audio",
            base_url=s.tts_base_url,
            voice_dir=s.voice_dir,
        ),
        video=WanAdapter(work_dir=work_dir / "video", base_url=s.wan_base_url),
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
) -> _JobContext:
    """Create stores, job record, work directory, and adapters for one run."""
    settings = config.settings
    artifact_store = create_artifact_store(settings)
    job_store = create_job_store(settings)

    job = Job(
        source_url=source_url,
        channel_profile_ref=config.channel_profile.id,
        cast_ref=config.cast.id,
        rights_cleared=rights_cleared,
    )
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
        adapters=_make_adapters(config, work_dir),
    )


def _resolve_line(ref: str, script: Script):
    """Parse a dialogue_line_ref and return the Line it points to.

    Format: "scene-{N}-line-{M}"  (N is 1-indexed, M is 0-indexed).
    """
    parts = ref.split("-")          # ["scene", "1", "line", "0"]
    scene_idx = int(parts[1]) - 1
    line_idx = int(parts[3])
    return script.scenes[scene_idx].lines[line_idx]


async def _run_shot(
    shot: Shot,
    script: Script,
    cast: Cast,
    adapters: _Adapters,
) -> tuple[VideoClip, AudioTrack]:
    """Run all four per-shot stages for one shot.

    Sequence: render_character → synthesize_voice → generate_video → lip_sync.
    Returns (synced VideoClip, AudioTrack) so the caller can collect both.
    """
    member_map = {m.id: m for m in cast.members}

    # Speaker is always characters_on_screen[0] (enforced by plan_shots trim_characters).
    speaker_id = shot.characters_on_screen[0]
    speaker = member_map[speaker_id]

    # Resolve first dialogue line (plan_shots produces one line per shot).
    first_line = (
        _resolve_line(shot.dialogue_line_refs[0], script)
        if shot.dialogue_line_refs
        else None
    )
    expression = first_line.expression if first_line else None
    line_text = first_line.text if first_line else ""

    log_event(
        logger, "shot_started",
        shot_id=shot.shot_id, speaker=speaker_id, text_chars=len(line_text),
    )

    # 1. render_character — still frame of the speaker in this setting
    render_result = await adapters.render.run(
        RenderCharacterRequest(
            member=speaker,
            setting=shot.setting,
            expression=expression,
        )
    )

    # 2. synthesize_voice — TTS for the dialogue line
    audio_track = await adapters.voice.run(
        VoiceRequest(
            text=line_text,
            voice_profile_ref=speaker.voice_profile_ref,
            speaker_id=speaker_id,
            expression=expression,
        )
    )

    # 3. generate_video — animate the still frame
    clip = await adapters.video.run(
        VideoRequest(
            image_uri=render_result.images[0],
            action=shot.action,
            duration_sec=shot.duration_sec,
            shot_id=shot.shot_id,
        )
    )

    # 4. lip_sync — align mouth to dialogue audio
    synced = await adapters.lipsync.run(
        LipSyncRequest(
            video_uri=clip.uri,
            audio_uri=audio_track.uri,
            shot_id=shot.shot_id,
        )
    )

    log_event(logger, "shot_completed", shot_id=shot.shot_id)
    return synced, audio_track


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


async def _run_to_assembled_video(ctx: _JobContext) -> tuple[Script, FinalVideo]:
    """Run Phase 1 stages through assembled candidate video, but do not publish."""
    config = ctx.config
    job = ctx.job
    adapters = ctx.adapters
    artifact_store = ctx.artifact_store
    job_store = ctx.job_store

    # 1. fetch_media
    fetch_result = await run_stage(
        "fetch_media", adapters.fetch_media,
        FetchMediaRequest(source_url=job.source_url),
        job, artifact_store, job_store,
    )

    # 2. transcribe
    transcribe_result = await run_stage(
        "transcribe", adapters.transcribe,
        TranscribeRequest(audio_uri=fetch_result.audio_uri),
        job, artifact_store, job_store,
    )

    # 3. analyze_content
    metadata = await run_stage(
        "analyze_content", adapters.analyze,
        AnalyzeRequest(
            transcript=transcribe_result,
            channel_profile=config.channel_profile,
        ),
        job, artifact_store, job_store,
    )

    # 4. rights gate (not a capability — runs synchronously)
    check_rights(job)

    # 5. adapt_script
    script: Script = await run_stage(
        "adapt_script", adapters.adapt,
        AdaptScriptRequest(
            metadata=metadata,
            cast=config.cast,
            channel_profile=config.channel_profile,
        ),
        job, artifact_store, job_store,
    )

    # 6. plan_shots
    storyboard: Storyboard = await run_stage(
        "plan_shots", adapters.plan,
        PlanShotsRequest(script=script, cast=config.cast),
        job, artifact_store, job_store,
    )

    # 7. per-shot loop
    synced_clips: list[VideoClip] = []
    audio_tracks: list[AudioTrack] = []
    for shot in storyboard.shots:
        synced, audio = await _run_shot(shot, script, config.cast, adapters)
        synced_clips.append(synced)
        audio_tracks.append(audio)

    combined_audio = await _concat_audio(audio_tracks, ctx.work_dir, adapters.ffmpeg_bin)

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
    )

    return script, final_video


async def _publish_candidate(
    ctx: _JobContext,
    script: Script,
    final_video: FinalVideo,
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
    )


async def _critique_candidate(
    ctx: _JobContext,
    script: Script,
    final_video: FinalVideo,
    attempt: int,
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
    ctx = _make_job_context(source_url, rights_cleared, config)
    job = ctx.job

    try:
        script, final_video = await _run_to_assembled_video(ctx)
        await _publish_candidate(ctx, script, final_video)

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
