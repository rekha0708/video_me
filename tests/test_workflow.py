"""Tests for run_pipeline_job (A1.12) and its private helpers."""
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from core.config import Settings, load_app_config
from core.executor import StageError
from core.models.capabilities import (
    AudioTrack,
    FetchMediaResult,
    FinalVideo,
    PublishResult,
    TranscribeResult,
    VideoClip,
)
from core.models.content import (
    ContentMetadata,
    LearningObjective,
    Line,
    Scene,
    Script,
    Shot,
    Storyboard,
)
from core.models.guardrails import SourceRights
from core.models.job import JobStatus
from core.workflow import _concat_audio, _resolve_line, _run_shot, run_pipeline_job


# ------------------------------------------------------------------ shared fixtures


def _make_config(tmp_path):
    config = load_app_config()
    config.settings = Settings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "video_me.db",
    )
    return config


def _fetch_result() -> FetchMediaResult:
    return FetchMediaResult(
        video_uri="/tmp/video.mp4",
        audio_uri="/tmp/audio.wav",
        duration_sec=90.0,
        source_url="http://example.com/video",
    )


def _transcribe_result() -> TranscribeResult:
    return TranscribeResult(segments=[], language="en", full_text="Let's count!")


def _metadata() -> ContentMetadata:
    return ContentMetadata(
        content_genre="education",
        topic="counting",
        tone="playful",
        hook="Let's count!",
        pacing="medium",
        length_sec=90,
    )


def _script() -> Script:
    return Script(
        mode="transformed",
        learning_objective=LearningObjective(
            concept="counting",
            age_range="3-6",
            success_phrase="Children learn to count to five.",
        ),
        scenes=[
            Scene(
                setting="cozy classroom",
                characters_present=["max"],
                lines=[
                    Line(speaker="max", text="Let's count to five!", expression="excited"),
                ],
            )
        ],
        caption_text="Let's count to five!",
        source_rights=SourceRights(kind="transformed", rights_cleared=True, notes=""),
    )


def _storyboard() -> Storyboard:
    return Storyboard(
        shots=[
            Shot(
                shot_id="s01",
                scene_ref="scene-1",
                characters_on_screen=["max"],
                setting="cozy classroom",
                camera="medium shot",
                action="character points at numbers",
                dialogue_line_refs=["scene-1-line-0"],
                duration_sec=3.5,
            )
        ]
    )


def _two_shot_storyboard() -> Storyboard:
    return Storyboard(
        shots=[
            Shot(
                shot_id="s01",
                scene_ref="scene-1",
                characters_on_screen=["max"],
                setting="cozy classroom",
                camera="medium shot",
                action="character waves",
                dialogue_line_refs=["scene-1-line-0"],
                duration_sec=3.0,
            ),
            Shot(
                shot_id="s02",
                scene_ref="scene-1",
                characters_on_screen=["max"],
                setting="cozy classroom",
                camera="close-up",
                action="character counts fingers",
                dialogue_line_refs=["scene-1-line-0"],
                duration_sec=4.0,
            ),
        ]
    )


def _final_video() -> FinalVideo:
    return FinalVideo(uri="/tmp/final.mp4", duration_sec=3.5)


def _publish_result() -> PublishResult:
    return PublishResult(
        review_path="/review/video.mp4",
        metadata_path="/review/metadata.json",
        status="pending_review",
    )


def _synced_clip() -> VideoClip:
    return VideoClip(uri="/tmp/synced.mp4", duration_sec=3.5, shot_id="s01")


def _audio_track() -> AudioTrack:
    return AudioTrack(uri="/tmp/dialogue.wav", duration_sec=2.5, speaker_id="max")


def _stage_results():
    return {
        "fetch_media": _fetch_result(),
        "transcribe": _transcribe_result(),
        "analyze_content": _metadata(),
        "adapt_script": _script(),
        "plan_shots": _storyboard(),
        "assemble_video": _final_video(),
        "publish": _publish_result(),
    }


def _make_run_stage(results: dict):
    async def _run_stage(stage_name, capability, request, job, artifact_store, job_store):
        return results[stage_name]
    return _run_stage


# ------------------------------------------------------------------ run_pipeline_job


@pytest.mark.asyncio
async def test_run_pipeline_job_completes(tmp_path) -> None:
    config = _make_config(tmp_path)
    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._run_shot", new=AsyncMock(return_value=(_synced_clip(), _audio_track()))),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        job = await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert job.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_pipeline_job_job_is_running_when_stages_start(tmp_path) -> None:
    config = _make_config(tmp_path)
    observed_statuses: list[str] = []

    async def spy_run_stage(stage_name, capability, request, job, *args):
        observed_statuses.append(str(job.status))
        return _stage_results()[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=spy_run_stage),
        patch("core.workflow._run_shot", new=AsyncMock(return_value=(_synced_clip(), _audio_track()))),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert all(s == "running" for s in observed_statuses)


@pytest.mark.asyncio
async def test_run_pipeline_job_blocked_when_rights_not_cleared(tmp_path) -> None:
    config = _make_config(tmp_path)
    mock_job_store = MagicMock()

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow.create_job_store", return_value=mock_job_store),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with pytest.raises(StageError):
            await run_pipeline_job("http://example.com", rights_cleared=False, app_config=config)

    # The job saved after blocking must have status BLOCKED
    last_saved: object = mock_job_store.save_job.call_args_list[-1][0][0]
    assert last_saved.status == JobStatus.BLOCKED


@pytest.mark.asyncio
async def test_run_pipeline_job_stage_error_sets_failed(tmp_path) -> None:
    config = _make_config(tmp_path)
    mock_job_store = MagicMock()

    async def failing_run_stage(stage_name, capability, request, job, *args):
        if stage_name == "analyze_content":
            raise StageError("analyze_content", "LLM timeout")
        return _stage_results()[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=failing_run_stage),
        patch("core.workflow.create_job_store", return_value=mock_job_store),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with pytest.raises(StageError):
            await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    last_saved = mock_job_store.save_job.call_args_list[-1][0][0]
    assert last_saved.status == JobStatus.FAILED


@pytest.mark.asyncio
async def test_run_pipeline_job_generic_exception_sets_failed(tmp_path) -> None:
    config = _make_config(tmp_path)
    mock_job_store = MagicMock()

    async def exploding_run_stage(stage_name, capability, request, job, *args):
        if stage_name == "transcribe":
            raise ValueError("unexpected crash")
        return _stage_results()[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=exploding_run_stage),
        patch("core.workflow.create_job_store", return_value=mock_job_store),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with pytest.raises(ValueError):
            await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    last_saved = mock_job_store.save_job.call_args_list[-1][0][0]
    assert last_saved.status == JobStatus.FAILED


@pytest.mark.asyncio
async def test_stage_call_order(tmp_path) -> None:
    config = _make_config(tmp_path)
    call_order: list[str] = []

    async def recording_run_stage(stage_name, capability, request, job, *args):
        call_order.append(stage_name)
        return _stage_results()[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=recording_run_stage),
        patch("core.workflow._run_shot", new=AsyncMock(return_value=(_synced_clip(), _audio_track()))),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert call_order == [
        "fetch_media",
        "transcribe",
        "analyze_content",
        "adapt_script",
        "plan_shots",
        "assemble_video",
        "publish",
    ]


@pytest.mark.asyncio
async def test_per_shot_loop_runs_for_each_shot(tmp_path) -> None:
    config = _make_config(tmp_path)
    results = {**_stage_results(), "plan_shots": _two_shot_storyboard()}
    mock_run_shot = AsyncMock(return_value=(_synced_clip(), _audio_track()))

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=_make_run_stage(results)),
        patch("core.workflow._run_shot", new=mock_run_shot),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert mock_run_shot.call_count == 2


@pytest.mark.asyncio
async def test_assemble_receives_all_synced_clips(tmp_path) -> None:
    config = _make_config(tmp_path)
    results = {**_stage_results(), "plan_shots": _two_shot_storyboard()}
    assemble_request_captured = {}

    async def recording_run_stage(stage_name, capability, request, job, *args):
        if stage_name == "assemble_video":
            assemble_request_captured["request"] = request
        return results[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=recording_run_stage),
        patch("core.workflow._run_shot", new=AsyncMock(return_value=(_synced_clip(), _audio_track()))),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert len(assemble_request_captured["request"].clips) == 2


@pytest.mark.asyncio
async def test_publish_gets_script_learning_objective(tmp_path) -> None:
    config = _make_config(tmp_path)
    publish_request_captured = {}

    async def recording_run_stage(stage_name, capability, request, job, *args):
        if stage_name == "publish":
            publish_request_captured["request"] = request
        return _stage_results()[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=recording_run_stage),
        patch("core.workflow._run_shot", new=AsyncMock(return_value=(_synced_clip(), _audio_track()))),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    req = publish_request_captured["request"]
    assert req.learning_objective_summary == "Children learn to count to five."


@pytest.mark.asyncio
async def test_work_dir_created_under_data_dir(tmp_path) -> None:
    config = _make_config(tmp_path)

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._run_shot", new=AsyncMock(return_value=(_synced_clip(), _audio_track()))),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        job = await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    job_work_dir = tmp_path / "jobs" / job.job_id
    assert job_work_dir.is_dir()


# ------------------------------------------------------------------ _resolve_line


def test_resolve_line_first_scene_first_line() -> None:
    script = _script()
    line = _resolve_line("scene-1-line-0", script)
    assert line.text == "Let's count to five!"


def test_resolve_line_second_scene() -> None:
    script = Script(
        mode="transformed",
        learning_objective=LearningObjective(
            concept="colours", age_range="3-6", success_phrase="Learn colours."
        ),
        scenes=[
            Scene(
                setting="park",
                lines=[Line(speaker="c1", text="First scene.")],
            ),
            Scene(
                setting="home",
                lines=[
                    Line(speaker="c2", text="Second scene, first line."),
                    Line(speaker="c2", text="Second scene, second line."),
                ],
            ),
        ],
        caption_text="Colour lesson",
        source_rights=SourceRights(kind="transformed", rights_cleared=True, notes=""),
    )
    line = _resolve_line("scene-2-line-1", script)
    assert line.text == "Second scene, second line."


def test_resolve_line_maps_one_indexed_scene() -> None:
    script = _script()
    # scene-1 → index 0 (the only scene)
    line = _resolve_line("scene-1-line-0", script)
    assert line.speaker == "max"


# ------------------------------------------------------------------ _concat_audio


@pytest.mark.asyncio
async def test_concat_audio_single_track_returns_directly(tmp_path) -> None:
    track = _audio_track()
    result = await _concat_audio([track], tmp_path)
    assert result is track


@pytest.mark.asyncio
async def test_concat_audio_writes_concat_file(tmp_path) -> None:
    t1 = AudioTrack(uri=str(tmp_path / "a.wav"), duration_sec=2.0)
    t2 = AudioTrack(uri=str(tmp_path / "b.wav"), duration_sec=3.0)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
        await _concat_audio([t1, t2], tmp_path)

    concat_file = tmp_path / "audio_concat.txt"
    assert concat_file.exists()
    content = concat_file.read_text()
    assert "a.wav" in content
    assert "b.wav" in content


@pytest.mark.asyncio
async def test_concat_audio_total_duration_is_sum(tmp_path) -> None:
    t1 = AudioTrack(uri=str(tmp_path / "a.wav"), duration_sec=2.0)
    t2 = AudioTrack(uri=str(tmp_path / "b.wav"), duration_sec=3.0)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
        result = await _concat_audio([t1, t2], tmp_path)

    assert result.duration_sec == 5.0


@pytest.mark.asyncio
async def test_concat_audio_raises_on_nonzero_return(tmp_path) -> None:
    t1 = AudioTrack(uri="/a.wav", duration_sec=1.0)
    t2 = AudioTrack(uri="/b.wav", duration_sec=1.0)

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"ffmpeg error output"))

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
        with pytest.raises(RuntimeError, match="Audio concat failed"):
            await _concat_audio([t1, t2], tmp_path)


@pytest.mark.asyncio
async def test_concat_audio_output_uri_points_to_combined(tmp_path) -> None:
    t1 = AudioTrack(uri="/a.wav", duration_sec=1.0)
    t2 = AudioTrack(uri="/b.wav", duration_sec=1.0)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
        result = await _concat_audio([t1, t2], tmp_path)

    assert "combined_audio.wav" in result.uri


# ------------------------------------------------------------------ _run_shot


@pytest.mark.asyncio
async def test_run_shot_calls_adapters_in_sequence(tmp_path) -> None:
    """render → voice → video → lipsync must be called in order."""
    call_order: list[str] = []
    _ImageSet = type("ImageSet", (), {"images": ["/img.png"]})

    async def _render(req):
        call_order.append("render")
        return _ImageSet()

    async def _voice(req):
        call_order.append("voice")
        return AudioTrack(uri="/d.wav", duration_sec=2.0, speaker_id="max")

    async def _video(req):
        call_order.append("video")
        return VideoClip(uri="/c.mp4", duration_sec=3.0, shot_id="s01")

    async def _lipsync(req):
        call_order.append("lipsync")
        return VideoClip(uri="/s.mp4", duration_sec=3.0, shot_id="s01")

    adapters = MagicMock()
    adapters.render.run = _render
    adapters.voice.run = _voice
    adapters.video.run = _video
    adapters.lipsync.run = _lipsync

    config = _make_config(tmp_path)
    shot = _storyboard().shots[0]

    await _run_shot(shot, _script(), config.cast, adapters)

    assert call_order == ["render", "voice", "video", "lipsync"]


@pytest.mark.asyncio
async def test_run_shot_returns_synced_clip_and_audio_track(tmp_path) -> None:
    expected_synced = VideoClip(uri="/synced.mp4", duration_sec=3.5, shot_id="s01")
    expected_audio = AudioTrack(uri="/dlg.wav", duration_sec=2.0, speaker_id="max")

    adapters = MagicMock()
    adapters.render.run = AsyncMock(
        return_value=type("ImageSet", (), {"images": ["/img.png"]})()
    )
    adapters.voice.run = AsyncMock(return_value=expected_audio)
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/raw.mp4", duration_sec=3.5, shot_id="s01")
    )
    adapters.lipsync.run = AsyncMock(return_value=expected_synced)

    config = _make_config(tmp_path)
    shot = _storyboard().shots[0]

    synced, audio = await _run_shot(shot, _script(), config.cast, adapters)

    assert synced is expected_synced
    assert audio is expected_audio


@pytest.mark.asyncio
async def test_run_shot_passes_speaker_id_to_voice(tmp_path) -> None:
    adapters = MagicMock()
    adapters.render.run = AsyncMock(
        return_value=type("ImageSet", (), {"images": ["/img.png"]})()
    )
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/d.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.lipsync.run = AsyncMock(
        return_value=VideoClip(uri="/s.mp4", duration_sec=1.0, shot_id="s01")
    )

    config = _make_config(tmp_path)
    shot = _storyboard().shots[0]

    await _run_shot(shot, _script(), config.cast, adapters)

    voice_req = adapters.voice.run.call_args[0][0]
    assert voice_req.speaker_id == "max"


@pytest.mark.asyncio
async def test_run_shot_passes_shot_id_to_lipsync(tmp_path) -> None:
    adapters = MagicMock()
    adapters.render.run = AsyncMock(
        return_value=type("ImageSet", (), {"images": ["/img.png"]})()
    )
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/d.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.lipsync.run = AsyncMock(
        return_value=VideoClip(uri="/s.mp4", duration_sec=1.0, shot_id="s01")
    )

    config = _make_config(tmp_path)
    shot = _storyboard().shots[0]

    await _run_shot(shot, _script(), config.cast, adapters)

    lipsync_req = adapters.lipsync.run.call_args[0][0]
    assert lipsync_req.shot_id == "s01"
