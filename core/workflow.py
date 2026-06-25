import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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

Phase = Literal["plan", "render", "assemble", "all"]


@dataclass
class RunOptions:
    """Controls which stages run and whether completed stages are skipped.

    phase:
      "plan"     — fetch → transcribe → analyze → adapt → plan_shots, then stop.
      "render"   — shot loop only (render + voice + video + lip_sync for each shot).
      "assemble" — assemble_video → publish only.
      "all"      — full pipeline (default).
    resume:      skip stages whose artifact JSON already exists on disk.
    only_shot:   in the shot loop, process only this shot ID (e.g. "s01").
    """
    phase: Phase = "all"
    resume: bool = False
    only_shot: str | None = None

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


def _restore_job_context(job_id: str, config: AppConfig) -> _JobContext:
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
    work_dir: Path,
    options: RunOptions | None = None,
) -> tuple[VideoClip, AudioTrack]:
    """Run all four per-shot stages for one shot.

    Sequence: render_character → synthesize_voice → generate_video → lip_sync.
    Returns (synced VideoClip, AudioTrack) so the caller can collect both.
    When options.resume is True, stages whose output files already exist are skipped.
    """
    opts = options or RunOptions()
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

    # ── Resume shortcuts ──────────────────────────────────────────────────────
    # If synced.mp4 already exists the whole shot is done — return both artifacts.
    synced_path = work_dir / "synced" / shot.shot_id / "synced.mp4"
    if opts.resume and synced_path.exists():
        logger.info("Skipping shot %s (synced.mp4 exists)", shot.shot_id)
        clip_path = work_dir / "video" / shot.shot_id / "clip.mp4"
        audio_files = sorted((work_dir / "audio" / speaker_id).glob("*.wav"))
        audio_uri = str(audio_files[0]) if audio_files else str(synced_path)
        return (
            VideoClip(uri=str(synced_path), duration_sec=shot.duration_sec),
            AudioTrack(uri=audio_uri, duration_sec=shot.duration_sec),
        )

    # ── 1. render_character ───────────────────────────────────────────────────
    render_png = work_dir / "renders" / speaker_id / "render_00.png"
    if opts.resume and render_png.exists():
        logger.info("Skipping render_character for %s (render_00.png exists)", shot.shot_id)
        from core.models.capabilities import ImageSet
        render_result = ImageSet(member_id=speaker_id, images=[str(render_png)])
    else:
        render_result = await adapters.render.run(
            RenderCharacterRequest(
                member=speaker,
                setting=shot.setting,
                expression=expression,
            )
        )

    # ── 2. synthesize_voice ───────────────────────────────────────────────────
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
            )
        )

    # ── 3. generate_video ─────────────────────────────────────────────────────
    clip_path = work_dir / "video" / shot.shot_id / "clip.mp4"
    if opts.resume and clip_path.exists():
        logger.info("Skipping generate_video for %s (clip.mp4 exists)", shot.shot_id)
        clip = VideoClip(uri=str(clip_path), duration_sec=shot.duration_sec)
    else:
        clip = await adapters.video.run(
            VideoRequest(
                image_uri=render_result.images[0],
                action=shot.action,
                duration_sec=shot.duration_sec,
                shot_id=shot.shot_id,
            )
        )

    # ── 4. lip_sync ───────────────────────────────────────────────────────────
    synced = await adapters.lipsync.run(
        LipSyncRequest(
            video_uri=clip.uri,
            audio_uri=audio_track.uri,
            shot_id=shot.shot_id,
        )
    )

    log_event(logger, "shot_completed", shot_id=shot.shot_id)
    return synced, audio_track


def _unload_ollama_model(base_url: str, model: str) -> None:
    """Tell Ollama to evict the model from VRAM (keep_alive=0) before GPU-heavy stages."""
    import urllib.request, json as _json
    ollama_base = base_url.replace("/v1", "").rstrip("/")
    try:
        data = _json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{ollama_base}/api/generate", data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        logger.info("Unloaded %s from VRAM before shot loop", model)
    except Exception as exc:
        logger.warning("Could not unload Ollama model (non-fatal): %s", exc)


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
        return await run_stage(name, capability, request, job, artifact_store, job_store)

    # ── Plan phase ────────────────────────────────────────────────────────────
    skip_plan = opts.phase == "render" or opts.phase == "assemble"

    if not skip_plan:
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

        # 4. rights gate
        check_rights(job)

        # 5. adapt_script
        script: Script = await _stage(
            "adapt_script", adapters.adapt,
            AdaptScriptRequest(
                metadata=metadata,
                cast=config.cast,
                channel_profile=config.channel_profile,
            ),
            Script,
        )

        # 6. plan_shots
        storyboard: Storyboard = await _stage(
            "plan_shots", adapters.plan,
            PlanShotsRequest(script=script, cast=config.cast),
            Storyboard,
        )

        if opts.phase == "plan":
            logger.info("Phase 'plan' complete — stopping before render loop.")
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
        # Release LLM from VRAM before the GPU-heavy shot loop.
        _unload_ollama_model(config.settings.llm_base_url, config.settings.llm_model)

        # 7. per-shot loop
        synced_clips = []
        audio_tracks = []
        shots_to_run = (
            [s for s in storyboard.shots if s.shot_id == opts.only_shot]
            if opts.only_shot
            else storyboard.shots
        )
        if opts.only_shot and not shots_to_run:
            raise ValueError(f"Shot '{opts.only_shot}' not found in storyboard.")

        for shot in shots_to_run:
            synced, audio = await _run_shot(
                shot, script, config.cast, adapters, ctx.work_dir, opts
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
    )

    return script, final_video


def _collect_existing_shot_artifacts(
    storyboard: Storyboard,
    script: Script,
    cast: Cast,
    work_dir: Path,
) -> tuple[list[VideoClip], list[AudioTrack]]:
    """Reconstruct clip/audio lists from files written by a previous render phase.

    Raises RuntimeError for any shot whose synced.mp4 is missing.
    """
    member_map = {m.id: m for m in cast.members}
    clips: list[VideoClip] = []
    audios: list[AudioTrack] = []
    for shot in storyboard.shots:
        synced_path = work_dir / "synced" / shot.shot_id / "synced.mp4"
        if not synced_path.exists():
            raise RuntimeError(
                f"Phase 'assemble' requires synced.mp4 for all shots, "
                f"but '{synced_path}' is missing. Run --phase render first."
            )
        speaker_id = shot.characters_on_screen[0]
        audio_files = sorted((work_dir / "audio" / speaker_id).glob("*.wav"))
        audio_uri = str(audio_files[0]) if audio_files else str(synced_path)
        clips.append(VideoClip(uri=str(synced_path), duration_sec=shot.duration_sec))
        audios.append(AudioTrack(uri=audio_uri, duration_sec=shot.duration_sec))
    return clips, audios


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
    options: RunOptions | None = None,
    resume_job_id: str | None = None,
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
    opts = options or RunOptions()
    if opts.resume and not resume_job_id:
        raise ValueError("resume=True requires resume_job_id to be set.")
    ctx = (
        _restore_job_context(resume_job_id, config)
        if resume_job_id
        else _make_job_context(source_url, rights_cleared, config)
    )
    job = ctx.job

    try:
        script, final_video = await _run_to_assembled_video(ctx, opts)
        if final_video is None:
            # phase="plan" or phase="render" — no video to publish, already marked complete
            return job
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
