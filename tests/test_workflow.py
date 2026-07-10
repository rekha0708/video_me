"""Tests for run_pipeline_job (A1.12) and its private helpers."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import Settings, load_app_config
from core.executor import StageError
from core.models.capabilities import (
    AudioTrack,
    CritiqueResult,
    FetchMediaResult,
    FinalVideo,
    ImageCritiqueResult,
    PublishResult,
    TranscribeResult,
    TranscriptSegment,
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
from core.models.job import Job, JobStatus
from core.workflow import (
    _apply_segment_timing_to_storyboard,
    _apply_source_timing_to_script,
    _build_verbatim_script,
    _chunk_shot_durations,
    _concat_audio,
    _fit_audio_to_duration,
    _generate_shot_video,
    _make_adapters,
    _rebuild_storyboard_by_duration,
    _retime_source_audio_plan,
    _resolve_line,
    _run_plan_critique_and_approval,
    _slice_source_audio,
    _storyboard_has_source_timing,
    _validate_timed_storyboard,
    run_pipeline_job,
    run_with_critique,
)


# ------------------------------------------------------------------ shared fixtures


@pytest.fixture(autouse=True)
def _no_real_comfyui_free(monkeypatch):
    """free_comfyui() talks to ComfyUI by raw URL, not through a mockable
    adapter instance — without this, every Phase-A test would fire a real
    HTTP call at whatever's listening on the default comfyui_base_url (which,
    on a real GPU pod, is a real ComfyUI serving real jobs). Autouse so no
    test can accidentally skip it; tests that care about this call still
    patch/assert on it explicitly and override this default."""
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr("core.workflow.free_comfyui", mock)
    return mock


@pytest.fixture(autouse=True)
def _no_real_fish_s2_process_spawn(monkeypatch):
    """prepare_voice_model() calls ensure_fish_s2_process_running() (defined
    in and called from within core.gpu_sequencer, not re-exported by
    core.workflow, hence patching the former) before loading a managed voice
    adapter. Without this, any test exercising a managed voice adapter with
    real default Settings() would attempt to actually spawn a Fish S2
    subprocess and poll it for up to fish_s2_load_timeout_sec (600s) — this
    bit a real test run: it silently no-op'd while the real Fish S2 happened
    to be up, then started genuinely timing out and failing once it was
    correctly down (per stop_fish_s2_process's own fix). Autouse so passing
    never again depends on incidental real-service state."""
    mock = AsyncMock(return_value=None)
    monkeypatch.setattr("core.gpu_sequencer.ensure_fish_s2_process_running", mock)
    return mock


def _make_config(tmp_path):
    config = load_app_config()
    config.settings = Settings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "video_me.db",
    )
    return config


def test_make_adapters_uses_runtime_settings(tmp_path) -> None:
    config = load_app_config()
    config.settings = Settings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "video_me.db",
        lora_dir=tmp_path / "loras",
        voice_dir=tmp_path / "voices",
        review_dir=tmp_path / "review",
        llm_model="llm-model",
        llm_base_url="http://llm.test/v1",
        llm_api_key="llm-key",
        critique_model="vlm-model",
        critique_base_url="http://vlm.test/v1",
        critique_api_key="vlm-key",
        sd_base_url="http://sd.test",
        tts_base_url="http://tts.test",
        fish_s2_base_url="http://fish.test",
        tts_adapter="fish_s2",
        render_adapter="a1111",
        video_adapter="wan",
        wan_base_url="http://wan.test",
        lipsync_base_url="http://lipsync.test",
        whisper_model_size="small",
        whisper_device="cuda",
        whisper_compute_type="float16",
        ffmpeg_bin="/opt/bin/ffmpeg",
        ffprobe_bin="/opt/bin/ffprobe",
        render_allow_placeholder_lora=True,
    )

    adapters = _make_adapters(config, tmp_path / "job")

    assert adapters.transcribe._model_size == "small"
    assert adapters.transcribe._device == "cuda"
    assert adapters.transcribe._compute_type == "float16"
    assert adapters.analyze._model == "llm-model"
    assert adapters.analyze._base_url == "http://llm.test/v1"
    assert adapters.adapt._api_key == "llm-key"
    assert adapters.plan._base_url == "http://llm.test/v1"
    assert adapters.render._base_url == "http://sd.test"
    assert adapters.render._allow_placeholder_lora is True
    assert adapters.voice._base_url == "http://fish.test"  # Fish S2 default
    assert adapters.video._base_url == "http://wan.test"
    assert adapters.lipsync._base_url == "http://lipsync.test"
    assert adapters.critique._model == "vlm-model"
    assert adapters.critique._base_url == "http://vlm.test/v1"
    assert adapters.critique._ffmpeg_bin == "/opt/bin/ffmpeg"
    assert adapters.critique._ffprobe_bin == "/opt/bin/ffprobe"
    assert adapters.assemble._ffmpeg_bin == "/opt/bin/ffmpeg"
    assert adapters.ffmpeg_bin == "/opt/bin/ffmpeg"


def test_make_adapters_chatterbox_fallback(tmp_path) -> None:
    config = load_app_config()
    config.settings = Settings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "video_me.db",
        tts_adapter="chatterbox",
        tts_base_url="http://chatterbox.test",
    )
    adapters = _make_adapters(config, tmp_path / "job")
    assert adapters.voice._base_url == "http://chatterbox.test"


# ------------------------------------------------------------------ shared data builders


def _fetch_result() -> FetchMediaResult:
    return FetchMediaResult(
        video_uri="/tmp/video.mp4",
        audio_uri="/tmp/audio.wav",
        duration_sec=90.0,
        source_url="http://example.com/video",
    )


def _transcribe_result() -> TranscribeResult:
    return TranscribeResult(segments=[], language="en", full_text="Let's count!")


def _visual_context() -> "VisualContext":
    from core.models.capabilities import VisualContext
    return VisualContext()


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
                setting="sunny garden",
                camera="wide shot",
                action="character points",
                dialogue_line_refs=["scene-1-line-0"],
                duration_sec=4.0,
            ),
        ]
    )


def _final_video() -> FinalVideo:
    return FinalVideo(uri="/tmp/final.mp4", duration_sec=10.0)


def _publish_result() -> PublishResult:
    return PublishResult(review_path="/review/video.mp4", metadata_path="/review/meta.json")


def _critique_result(verdict: str = "pass") -> CritiqueResult:
    return CritiqueResult(
        scores={"age_appropriateness": 0.9, "learning_clarity": 0.8},
        verdict=verdict,
        reasons=[f"verdict {verdict}"],
        suggested_param_overrides={},
    )


def _synced_clip() -> VideoClip:
    return VideoClip(uri="/tmp/synced.mp4", duration_sec=3.5, shot_id="s01")


def _audio_track() -> AudioTrack:
    return AudioTrack(uri="/tmp/dialogue.wav", duration_sec=2.5, speaker_id="max")


def _image_critique_result() -> ImageCritiqueResult:
    return ImageCritiqueResult(winner_index=0, winner_uri="/tmp/render_00.png")


def _stage_results():
    return {
        "fetch_media": _fetch_result(),
        "transcribe": _transcribe_result(),
        "analyze_content": _metadata(),
        "analyze_visuals": _visual_context(),
        "adapt_script": _script(),
        "plan_shots": _storyboard(),
        "assemble_video": _final_video(),
        "publish": _publish_result(),
    }


def _make_run_stage(results: dict):
    async def _run_stage(stage_name, capability, request, job, artifact_store, job_store, **_kw):
        return results[stage_name]
    return _run_stage


def _make_run_stage_with_critiques(verdicts: list[str], call_order: list[str] | None = None):
    critique_index = 0

    async def _run_stage(stage_name, capability, request, job, artifact_store, job_store, **_kw):
        nonlocal critique_index
        if call_order is not None:
            call_order.append(stage_name)
        if stage_name.startswith("critique_attempt_"):
            verdict = verdicts[critique_index]
            critique_index += 1
            return _critique_result(verdict)
        return _stage_results()[stage_name]

    return _run_stage


@contextmanager
def _mock_shot_patches():
    """Context manager that bypasses image critique/approval/shot loop."""
    with ExitStack() as stack:
        stack.enter_context(patch(
            "core.workflow._run_plan_critique_and_approval",
            new=AsyncMock(return_value=(_storyboard(), _script())),
        ))
        stack.enter_context(patch(
            "core.workflow._render_shot_candidates",
            new=AsyncMock(return_value=_image_critique_result()),
        ))
        stack.enter_context(patch(
            "core.workflow._run_image_approval_gate",
            new=AsyncMock(return_value=["/tmp/render_00.png"]),
        ))
        stack.enter_context(patch(
            "core.workflow._generate_shot_video",
            new=AsyncMock(return_value=(_synced_clip(), _audio_track())),
        ))
        yield


# ------------------------------------------------------------------ run_pipeline_job


@pytest.mark.asyncio
async def test_run_pipeline_job_completes(tmp_path) -> None:
    config = _make_config(tmp_path)
    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            job = await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert job.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_pipeline_job_announces_total_shot_count(tmp_path) -> None:
    """Dashboard should see an upfront shot-count message before Phase A and
    Phase B, so the timeline is clear before individual shots start."""
    from core.workflow import RunOptions

    config = _make_config(tmp_path)
    messages: list[str] = []

    def stage_hook(stage_name, event_type, **kw):
        if kw.get("message"):
            messages.append(kw["message"])

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            await run_pipeline_job(
                "http://example.com", rights_cleared=True, app_config=config,
                options=RunOptions(stage_hook=stage_hook),
            )

    # _storyboard() (used by _mock_shot_patches' plan_critique stub) has 1 shot.
    assert "Rendering 1 shot" in messages
    assert "Generating video for 1 shot" in messages


@pytest.mark.asyncio
async def test_run_pipeline_job_uses_config_target_language_when_not_overridden(tmp_path) -> None:
    config = _make_config(tmp_path)
    config.settings.target_language = "both"
    observed_languages: list[str] = []

    def fake_context(source_url, rights_cleared, app_config, **kwargs):
        return SimpleNamespace(
            job=Job(
                source_url=source_url,
                channel_profile_ref=app_config.channel_profile.id,
                cast_ref=app_config.cast.id,
                rights_cleared=rights_cleared,
            )
        )

    async def fake_single_language_job(ctx, opts):
        observed_languages.append(opts.language)
        return ctx.job

    with (
        patch("core.workflow._make_job_context", new=fake_context),
        patch("core.workflow._run_single_language_job", new=fake_single_language_job),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert observed_languages == ["en", "hi"]


@pytest.mark.asyncio
async def test_run_pipeline_job_job_is_running_when_stages_start(tmp_path) -> None:
    config = _make_config(tmp_path)
    observed_statuses: list[str] = []

    async def spy_run_stage(stage_name, capability, request, job, *args, **_kw):
        observed_statuses.append(str(job.status))
        return _stage_results()[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=spy_run_stage),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
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

    async def failing_run_stage(stage_name, capability, request, job, *args, **_kw):
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

    async def exploding_run_stage(stage_name, capability, request, job, *args, **_kw):
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

    async def recording_run_stage(stage_name, capability, request, job, *args, **_kw):
        call_order.append(stage_name)
        return _stage_results()[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=recording_run_stage),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert call_order == [
        "fetch_media",
        "transcribe",
        "analyze_content",
        "analyze_visuals",
        "adapt_script",
        "plan_shots",
        "assemble_video",
        "publish",
    ]


@pytest.mark.asyncio
async def test_vram_managed_video_adapter_sequencing(tmp_path) -> None:
    """A managed-VRAM video adapter (Wan) is unloaded before the render loop and
    loaded (unload Ollama → gap → load → readiness poll) only after image approval."""
    config = _make_config(tmp_path)
    config.settings.wan_load_gap_sec = 0  # don't sleep in tests
    call_order: list[str] = []

    adapters = MagicMock(ffmpeg_bin="ffmpeg")
    adapters.video.managed_vram = True
    adapters.video.unload = AsyncMock(side_effect=lambda: call_order.append("wan_unload") or True)
    adapters.video.load = AsyncMock(side_effect=lambda: call_order.append("wan_load"))
    adapters.video.wait_until_loaded = AsyncMock(
        side_effect=lambda *a, **k: call_order.append("wan_wait")
    )

    with (
        patch("core.workflow._make_adapters", return_value=adapters),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._unload_ollama_model"),
        patch("core.gpu_sequencer.unload_ollama_model"),
        patch(
            "core.workflow._run_plan_critique_and_approval",
            new=AsyncMock(return_value=(_storyboard(), _script())),
        ),
        patch(
            "core.workflow._render_shot_candidates",
            new=AsyncMock(
                side_effect=lambda *a, **k: call_order.append("render") or _image_critique_result()
            ),
        ),
        patch(
            "core.workflow._run_image_approval_gate",
            new=AsyncMock(
                side_effect=lambda *a, **k: call_order.append("approve") or ["/tmp/render_00.png"]
            ),
        ),
        patch(
            "core.workflow._generate_shot_video",
            new=AsyncMock(
                side_effect=lambda *a, **k: call_order.append("video")
                or (_synced_clip(), _audio_track())
            ),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert call_order == ["wan_unload", "render", "approve", "wan_load", "wan_wait", "video"]


@pytest.mark.asyncio
async def test_free_comfyui_called_before_render_loop_even_when_unused(
    tmp_path, _no_real_comfyui_free,
) -> None:
    """ComfyUI must get freed before Phase A even when this job's own render
    (musubi_flux) and video (wan) adapters are neither one ComfyUI — otherwise
    a model left resident by a *different* job's render_adapter=comfyui_flux
    or video_adapter=ltx choice sits stranded through this job's whole render
    phase (the actual cause of a production OOM)."""
    config = _make_config(tmp_path)
    config.settings.wan_load_gap_sec = 0
    config.settings.comfyui_base_url = "http://comfyui.test:8188"

    adapters = MagicMock(ffmpeg_bin="ffmpeg")
    adapters.render = MagicMock(spec=[])  # musubi_flux: not managed_vram
    adapters.video.managed_vram = True
    adapters.video.unload = AsyncMock(return_value=True)
    adapters.video.load = AsyncMock()
    adapters.video.wait_until_loaded = AsyncMock()

    with (
        patch("core.workflow._make_adapters", return_value=adapters),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._unload_ollama_model"),
        patch("core.gpu_sequencer.unload_ollama_model"),
        patch(
            "core.workflow._run_plan_critique_and_approval",
            new=AsyncMock(return_value=(_storyboard(), _script())),
        ),
        patch(
            "core.workflow._render_shot_candidates",
            new=AsyncMock(return_value=_image_critique_result()),
        ),
        patch(
            "core.workflow._run_image_approval_gate",
            new=AsyncMock(return_value=["/tmp/render_00.png"]),
        ),
        patch(
            "core.workflow._generate_shot_video",
            new=AsyncMock(return_value=(_synced_clip(), _audio_track())),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    _no_real_comfyui_free.assert_awaited_once_with("http://comfyui.test:8188")


@pytest.mark.asyncio
async def test_vram_managed_voice_and_render_adapters_sequencing(tmp_path) -> None:
    """Fish S2 (voice) and a managed render adapter (comfyui_flux fallback) get
    the same unload-before-Phase-A / unload-render-then-load-before-Phase-B
    treatment as Wan, alongside it."""
    config = _make_config(tmp_path)
    config.settings.wan_load_gap_sec = 0
    config.settings.fish_s2_load_gap_sec = 0
    call_order: list[str] = []

    adapters = MagicMock(ffmpeg_bin="ffmpeg")
    adapters.video.managed_vram = True
    adapters.video.unload = AsyncMock(side_effect=lambda: call_order.append("video_unload") or True)
    adapters.video.load = AsyncMock(side_effect=lambda: call_order.append("video_load"))
    adapters.video.wait_until_loaded = AsyncMock(
        side_effect=lambda *a, **k: call_order.append("video_wait")
    )
    adapters.voice.managed_vram = True
    adapters.voice.unload = AsyncMock(side_effect=lambda: call_order.append("voice_unload") or True)
    adapters.voice.load = AsyncMock(side_effect=lambda: call_order.append("voice_load"))
    adapters.voice.wait_until_loaded = AsyncMock(
        side_effect=lambda *a, **k: call_order.append("voice_wait")
    )
    adapters.render.managed_vram = True
    adapters.render.unload = AsyncMock(side_effect=lambda: call_order.append("render_unload") or True)

    with (
        patch("core.workflow._make_adapters", return_value=adapters),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._unload_ollama_model"),
        patch("core.gpu_sequencer.unload_ollama_model"),
        patch(
            "core.workflow._run_plan_critique_and_approval",
            new=AsyncMock(return_value=(_storyboard(), _script())),
        ),
        patch(
            "core.workflow._render_shot_candidates",
            new=AsyncMock(
                side_effect=lambda *a, **k: call_order.append("render") or _image_critique_result()
            ),
        ),
        patch(
            "core.workflow._run_image_approval_gate",
            new=AsyncMock(
                side_effect=lambda *a, **k: call_order.append("approve") or ["/tmp/render_00.png"]
            ),
        ),
        patch(
            "core.workflow._generate_shot_video",
            new=AsyncMock(
                side_effect=lambda *a, **k: call_order.append("video")
                or (_synced_clip(), _audio_track())
            ),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert call_order == [
        "video_unload", "voice_unload",
        "render", "approve",
        "render_unload", "video_load", "video_wait", "voice_load", "voice_wait",
        "video",
    ]


@pytest.mark.asyncio
async def test_unmanaged_video_adapter_skips_sequencing(tmp_path) -> None:
    """The default LTX adapter (no managed_vram attr) must not get load/unload calls."""
    config = _make_config(tmp_path)
    adapters = MagicMock(ffmpeg_bin="ffmpeg")
    adapters.video = MagicMock(spec=[])  # like LtxAdapter: no managed_vram / load / unload

    with (
        patch("core.workflow._make_adapters", return_value=adapters),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            job = await run_pipeline_job(
                "http://example.com", rights_cleared=True, app_config=config
            )

    assert job.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_per_shot_loop_runs_for_each_shot(tmp_path) -> None:
    config = _make_config(tmp_path)
    results = {**_stage_results(), "plan_shots": _two_shot_storyboard()}
    mock_generate = AsyncMock(return_value=(_synced_clip(), _audio_track()))

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=_make_run_stage(results)),
        patch(
            "core.workflow._run_plan_critique_and_approval",
            new=AsyncMock(return_value=(_two_shot_storyboard(), _script())),
        ),
        patch(
            "core.workflow._render_shot_candidates",
            new=AsyncMock(return_value=_image_critique_result()),
        ),
        patch(
            "core.workflow._run_image_approval_gate",
            new=AsyncMock(return_value=["/tmp/r0.png", "/tmp/r1.png"]),
        ),
        patch("core.workflow._generate_shot_video", new=mock_generate),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert mock_generate.call_count == 2


@pytest.mark.asyncio
async def test_render_plan_overlays_sets_png_uri_and_persists(tmp_path) -> None:
    from core.models.content import ShotOverlay
    from core.workflow import _render_plan_overlays, RunOptions

    sb = _storyboard()
    sb.shots[0].overlay = ShotOverlay(kind="callout", title="Count to five!")

    from core.models.capabilities import RenderOverlaysResult
    from core.models.common import HealthStatus

    ctx = SimpleNamespace(
        adapters=SimpleNamespace(overlays=SimpleNamespace(
            health=AsyncMock(return_value=HealthStatus(status="ok")),
            run=AsyncMock(return_value=RenderOverlaysResult(
                images={"s01": "/tmp/overlays/s01.png"})),
        )),
        artifact_store=MagicMock(),
        job=SimpleNamespace(job_id="job1"),
    )
    result = await _render_plan_overlays(sb, ctx, RunOptions())

    assert result.shots[0].overlay.png_uri == "/tmp/overlays/s01.png"
    stage_names = [c.args[1] for c in ctx.artifact_store.put_json.call_args_list]
    assert "render_overlays" in stage_names


@pytest.mark.asyncio
async def test_render_plan_overlays_best_effort_when_down(tmp_path) -> None:
    from core.models.common import HealthStatus
    from core.models.content import ShotOverlay
    from core.workflow import _render_plan_overlays, RunOptions

    sb = _storyboard()
    sb.shots[0].overlay = ShotOverlay(kind="callout", title="Count!")
    overlays_adapter = SimpleNamespace(
        health=AsyncMock(return_value=HealthStatus(status="down", reason="no matplotlib")),
        run=AsyncMock(),
    )
    ctx = SimpleNamespace(
        adapters=SimpleNamespace(overlays=overlays_adapter),
        artifact_store=MagicMock(),
        job=SimpleNamespace(job_id="job1"),
    )
    result = await _render_plan_overlays(sb, ctx, RunOptions())
    overlays_adapter.run.assert_not_called()
    assert result.shots[0].overlay.png_uri is None  # job continues without panels


@pytest.mark.asyncio
async def test_overlays_render_before_approval_and_plan_repersisted(tmp_path) -> None:
    """Previews must exist before the gate; the approved plan must be re-persisted."""
    from core.models.capabilities import PlanCritiqueResult, RenderOverlaysResult
    from core.models.common import HealthStatus
    from core.models.content import ShotOverlay
    from core.workflow import _run_plan_critique_and_approval, RunOptions

    config = _make_config(tmp_path)
    call_order: list[str] = []

    sb = _storyboard()
    sb.shots[0].overlay = ShotOverlay(kind="callout", title="Count!")

    async def overlays_run(req):
        call_order.append("overlays")
        return RenderOverlaysResult(images={"s01": "/tmp/overlays/s01.png"})

    async def approval_run(**kwargs):
        call_order.append("approval")
        # storyboard shown at the gate must already carry the preview path
        assert kwargs["storyboard"].shots[0].overlay.png_uri == "/tmp/overlays/s01.png"
        return (True, "")

    adapters = SimpleNamespace(
        plan=SimpleNamespace(run=AsyncMock(return_value=sb)),
        plan_critique=SimpleNamespace(run=AsyncMock(return_value=PlanCritiqueResult(verdict="pass"))),
        overlays=SimpleNamespace(
            health=AsyncMock(return_value=HealthStatus(status="ok")),
            run=overlays_run,
        ),
        approval=SimpleNamespace(request_approval=approval_run),
    )
    artifact_store = MagicMock()
    ctx = SimpleNamespace(
        config=config,
        job=Job(source_url="http://x", channel_profile_ref="p", cast_ref="c", rights_cleared=True),
        job_store=MagicMock(),
        adapters=adapters,
        artifact_store=artifact_store,
    )

    out_sb, _ = await _run_plan_critique_and_approval(sb, _script(), ctx, RunOptions())

    assert call_order == ["overlays", "approval"]
    persisted = [c for c in artifact_store.put_json.call_args_list if c.args[1] == "plan_shots"]
    assert persisted, "approved storyboard must be re-persisted to the plan_shots artifact"
    assert persisted[-1].args[2]["shots"][0]["overlay"]["png_uri"] == "/tmp/overlays/s01.png"


@pytest.mark.asyncio
async def test_build_overlay_windows_uses_probed_durations(tmp_path, monkeypatch) -> None:
    from core.models.content import ShotOverlay
    from core.workflow import _build_overlay_windows
    import core.workflow as wf

    sb = _two_shot_storyboard()
    png = tmp_path / "s02.png"
    png.write_bytes(b"png")
    sb.shots[1].overlay = ShotOverlay(kind="callout", title="Data!", png_uri=str(png))

    clips = [
        VideoClip(uri="/tmp/c1.mp4", duration_sec=3.0, shot_id="s01"),  # estimate 3.0
        VideoClip(uri="/tmp/c2.mp4", duration_sec=4.0, shot_id="s02"),
    ]
    # Actual durations differ from the estimates — offsets must use actuals.
    actuals = {"/tmp/c1.mp4": 6.5, "/tmp/c2.mp4": 5.0}

    async def fake_probe(path, ffprobe_bin):
        return actuals[str(path)]

    monkeypatch.setattr(wf, "_probe_duration_sec", fake_probe)
    windows = await _build_overlay_windows(sb.shots, clips, "ffprobe")

    assert len(windows) == 1
    w = windows[0]
    assert w.shot_id == "s02"
    assert w.start_sec == pytest.approx(6.5 + 0.25)   # offset from ACTUAL first clip
    assert w.end_sec == pytest.approx(6.5 + 5.0 - 0.1)


@pytest.mark.asyncio
async def test_build_overlay_windows_skips_missing_png(tmp_path, monkeypatch) -> None:
    from core.models.content import ShotOverlay
    from core.workflow import _build_overlay_windows
    import core.workflow as wf

    sb = _storyboard()
    sb.shots[0].overlay = ShotOverlay(kind="callout", title="Gone",
                                      png_uri=str(tmp_path / "missing.png"))
    clips = [VideoClip(uri="/tmp/c1.mp4", duration_sec=3.0, shot_id="s01")]
    monkeypatch.setattr(wf, "_probe_duration_sec", AsyncMock(return_value=None))
    windows = await _build_overlay_windows(sb.shots, clips, "ffprobe")
    assert windows == []


@pytest.mark.asyncio
async def test_probe_clip_durations_overrides_with_actual_length(monkeypatch) -> None:
    """Video adapters (LTX/Wan) echo back the requested duration, not what
    actually rendered — assemble_video's crossfade math must use the real,
    ffprobe-measured length or it can miscalculate the xfade offset and
    truncate the output (reproduced case: a claimed ~15s 2-shot assembly
    collapsing to just the first clip's real ~7.7s)."""
    from core.workflow import _probe_clip_durations
    import core.workflow as wf

    clips = [
        VideoClip(uri="/tmp/c1.mp4", duration_sec=8.0, shot_id="s01"),
        VideoClip(uri="/tmp/c2.mp4", duration_sec=7.23, shot_id="s02"),
    ]
    actuals = {"/tmp/c1.mp4": 7.708333, "/tmp/c2.mp4": 7.041667}

    async def fake_probe(path, ffprobe_bin):
        return actuals[str(path)]

    monkeypatch.setattr(wf, "_probe_duration_sec", fake_probe)
    corrected = await _probe_clip_durations(clips, "ffprobe")

    assert [c.duration_sec for c in corrected] == [7.708333, 7.041667]
    # uri/shot_id must be preserved — only duration_sec is corrected.
    assert [c.uri for c in corrected] == ["/tmp/c1.mp4", "/tmp/c2.mp4"]
    assert [c.shot_id for c in corrected] == ["s01", "s02"]


@pytest.mark.asyncio
async def test_probe_clip_durations_falls_back_when_probe_fails(monkeypatch) -> None:
    from core.workflow import _probe_clip_durations
    import core.workflow as wf

    clips = [VideoClip(uri="/tmp/c1.mp4", duration_sec=8.0, shot_id="s01")]
    monkeypatch.setattr(wf, "_probe_duration_sec", AsyncMock(return_value=None))

    corrected = await _probe_clip_durations(clips, "ffprobe")

    assert corrected[0].duration_sec == 8.0


@pytest.mark.asyncio
async def test_visual_context_flows_into_adapt_script(tmp_path) -> None:
    """analyze_visuals output must reach the AdaptScriptRequest for grounding."""
    from core.models.capabilities import VisualContext, VisualSegment

    config = _make_config(tmp_path)
    grounded = VisualContext(
        segments=[VisualSegment(start=0, end=5, setting="cozy kitchen", props=["apple"])]
    )
    results = {**_stage_results(), "analyze_visuals": grounded}
    captured = {}

    async def recording_run_stage(stage_name, capability, request, job, *args, **_kw):
        if stage_name == "adapt_script":
            captured["request"] = request
        return results[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=recording_run_stage),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    req = captured["request"]
    assert req.visual_context is not None
    assert req.visual_context.segments[0].setting == "cozy kitchen"


@pytest.mark.asyncio
async def test_assemble_receives_all_synced_clips(tmp_path) -> None:
    config = _make_config(tmp_path)
    results = {**_stage_results(), "plan_shots": _two_shot_storyboard()}
    assemble_request_captured = {}

    async def recording_run_stage(stage_name, capability, request, job, *args, **_kw):
        if stage_name == "assemble_video":
            assemble_request_captured["request"] = request
        return results[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=recording_run_stage),
        patch(
            "core.workflow._run_plan_critique_and_approval",
            new=AsyncMock(return_value=(_two_shot_storyboard(), _script())),
        ),
        patch(
            "core.workflow._render_shot_candidates",
            new=AsyncMock(return_value=_image_critique_result()),
        ),
        patch(
            "core.workflow._run_image_approval_gate",
            new=AsyncMock(return_value=["/tmp/r0.png", "/tmp/r1.png"]),
        ),
        patch(
            "core.workflow._generate_shot_video",
            new=AsyncMock(return_value=(_synced_clip(), _audio_track())),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    assert len(assemble_request_captured["request"].clips) == 2


@pytest.mark.asyncio
async def test_source_audio_assemble_request_preserves_timing(tmp_path) -> None:
    from core.workflow import RunOptions

    config = _make_config(tmp_path)
    fetch = FetchMediaResult(
        video_uri="/tmp/video.mp4",
        audio_uri="/tmp/audio.wav",
        duration_sec=15.0,
        source_url="file:///tmp/video.mp4",
    )
    transcript = TranscribeResult(
        segments=[
            TranscriptSegment(text="First part.", start=0.0, end=8.0),
            TranscriptSegment(text="Second part.", start=8.0, end=15.0),
        ],
        language="en",
        full_text="First part. Second part.",
    )
    results = {
        **_stage_results(),
        "fetch_media": fetch,
        "transcribe": transcript,
        "plan_shots": _two_shot_storyboard(),
    }
    assemble_request_captured = {}

    async def recording_run_stage(stage_name, capability, request, job, *args, **_kw):
        if stage_name == "assemble_video":
            assemble_request_captured["request"] = request
        return results[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=recording_run_stage),
        patch(
            "core.workflow._render_shot_candidates",
            new=AsyncMock(return_value=_image_critique_result()),
        ),
        patch(
            "core.workflow._run_image_approval_gate",
            new=AsyncMock(return_value=["/tmp/r0.png", "/tmp/r1.png"]),
        ),
        patch(
            "core.workflow._generate_shot_video",
            new=AsyncMock(return_value=(_synced_clip(), _audio_track())),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=AudioTrack(uri="/tmp/source.wav", duration_sec=15.0))),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        await run_pipeline_job(
            "file:///tmp/video.mp4",
            rights_cleared=True,
            app_config=config,
            options=RunOptions(render_mode="source_audio"),
        )

    req = assemble_request_captured["request"]
    assert req.preserve_timing is True
    assert len(req.audio_tracks) == 2
    assert [clip.duration_sec for clip in req.clips] == [3.5, 3.5]


@pytest.mark.asyncio
async def test_publish_gets_script_learning_objective(tmp_path) -> None:
    config = _make_config(tmp_path)
    publish_request_captured = {}

    async def recording_run_stage(stage_name, capability, request, job, *args, **_kw):
        if stage_name == "publish":
            publish_request_captured["request"] = request
        return _stage_results()[stage_name]

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=recording_run_stage),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    req = publish_request_captured["request"]
    assert req.learning_objective_summary == "Children learn to count to five."


@pytest.mark.asyncio
async def test_work_dir_created_under_data_dir(tmp_path) -> None:
    config = _make_config(tmp_path)

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch("core.workflow.run_stage", new=_make_run_stage(_stage_results())),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            job = await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    job_work_dir = tmp_path / "jobs" / job.job_id
    assert job_work_dir.is_dir()


# ------------------------------------------------------------------ run_with_critique


@pytest.mark.asyncio
async def test_run_with_critique_pass_publishes(tmp_path) -> None:
    config = _make_config(tmp_path)
    call_order: list[str] = []

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch(
            "core.workflow.run_stage",
            new=_make_run_stage_with_critiques(["pass"], call_order),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            job = await run_with_critique("http://example.com", rights_cleared=True, app_config=config)

    assert job.status == JobStatus.COMPLETED
    assert "critique_attempt_1" in call_order
    assert call_order[-1] == "publish"


@pytest.mark.asyncio
async def test_run_with_critique_unloads_managed_adapters_before_critique(tmp_path) -> None:
    """Frame critique only needs the LLM/VLM — any VRAM-managed render/video/
    voice models left resident from Phase A/B must be freed first."""
    config = _make_config(tmp_path)
    call_order: list[str] = []

    adapters = MagicMock(ffmpeg_bin="ffmpeg")
    adapters.video.managed_vram = True
    adapters.video.unload = AsyncMock(side_effect=lambda: call_order.append("video_unload") or True)
    adapters.video.load = AsyncMock()
    adapters.video.wait_until_loaded = AsyncMock()
    adapters.voice.managed_vram = True
    adapters.voice.unload = AsyncMock(side_effect=lambda: call_order.append("voice_unload") or True)
    adapters.voice.load = AsyncMock()
    adapters.voice.wait_until_loaded = AsyncMock()
    adapters.render.managed_vram = True
    adapters.render.unload = AsyncMock(side_effect=lambda: call_order.append("render_unload") or True)

    config.settings.wan_load_gap_sec = 0
    config.settings.fish_s2_load_gap_sec = 0

    with (
        patch("core.workflow._make_adapters", return_value=adapters),
        patch(
            "core.workflow.run_stage",
            new=_make_run_stage_with_critiques(["pass"], call_order),
        ),
        patch("core.workflow._unload_ollama_model"),
        patch("core.gpu_sequencer.unload_ollama_model"),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            job = await run_with_critique("http://example.com", rights_cleared=True, app_config=config)

    assert job.status == JobStatus.COMPLETED
    critique_index = call_order.index("critique_attempt_1")
    assert "render_unload" in call_order[:critique_index]
    assert "video_unload" in call_order[:critique_index]
    assert "voice_unload" in call_order[:critique_index]


@pytest.mark.asyncio
async def test_run_with_critique_regenerates_then_publishes(tmp_path) -> None:
    config = _make_config(tmp_path)
    call_order: list[str] = []

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch(
            "core.workflow.run_stage",
            new=_make_run_stage_with_critiques(["regenerate", "pass"], call_order),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=MagicMock()),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            job = await run_with_critique("http://example.com", rights_cleared=True, app_config=config)

    assert job.status == JobStatus.COMPLETED
    assert call_order.count("assemble_video") == 2
    assert "critique_attempt_1" in call_order
    assert "critique_attempt_2" in call_order
    assert call_order.count("publish") == 1


@pytest.mark.asyncio
async def test_run_with_critique_reject_blocks_without_publish(tmp_path) -> None:
    config = _make_config(tmp_path)
    mock_job_store = MagicMock()
    call_order: list[str] = []

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch(
            "core.workflow.run_stage",
            new=_make_run_stage_with_critiques(["reject"], call_order),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=mock_job_store),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            with pytest.raises(StageError, match="reject"):
                await run_with_critique("http://example.com", rights_cleared=True, app_config=config)

    last_saved = mock_job_store.save_job.call_args_list[-1][0][0]
    assert last_saved.status == JobStatus.BLOCKED
    assert "publish" not in call_order


@pytest.mark.asyncio
async def test_run_with_critique_max_regenerations_sets_failed(tmp_path) -> None:
    config = _make_config(tmp_path)
    config.settings.max_regenerations = 1
    mock_job_store = MagicMock()
    call_order: list[str] = []

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch(
            "core.workflow.run_stage",
            new=_make_run_stage_with_critiques(["regenerate", "regenerate"], call_order),
        ),
        patch("core.workflow._concat_audio", new=AsyncMock(return_value=_audio_track())),
        patch("core.workflow.create_job_store", return_value=mock_job_store),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with _mock_shot_patches():
            with pytest.raises(StageError, match="max_regenerations exhausted"):
                await run_with_critique("http://example.com", rights_cleared=True, app_config=config)

    last_saved = mock_job_store.save_job.call_args_list[-1][0][0]
    assert last_saved.status == JobStatus.FAILED
    assert call_order.count("assemble_video") == 2
    assert "critique_attempt_1" in call_order
    assert "critique_attempt_2" in call_order
    assert "publish" not in call_order


@pytest.mark.asyncio
async def test_run_with_critique_blocks_when_rights_not_cleared(tmp_path) -> None:
    config = _make_config(tmp_path)
    mock_job_store = MagicMock()
    call_order: list[str] = []

    with (
        patch("core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg")),
        patch(
            "core.workflow.run_stage",
            new=_make_run_stage_with_critiques(["pass"], call_order),
        ),
        patch("core.workflow.create_job_store", return_value=mock_job_store),
        patch("core.workflow.create_artifact_store", return_value=MagicMock()),
    ):
        with pytest.raises(StageError):
            await run_with_critique("http://example.com", rights_cleared=False, app_config=config)

    last_saved = mock_job_store.save_job.call_args_list[-1][0][0]
    assert last_saved.status == JobStatus.BLOCKED
    assert "critique_attempt_1" not in call_order
    assert "publish" not in call_order


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


# ------------------------------------------------------------------ _generate_shot_video


@pytest.mark.asyncio
async def test_generate_shot_video_calls_adapters_in_sequence(tmp_path) -> None:
    """voice → video → lipsync must be called in order for non-native-lipsync adapter."""
    call_order: list[str] = []

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
    adapters.voice.run = _voice
    adapters.video.run = _video
    adapters.lipsync.run = _lipsync
    adapters.video.native_lipsync = False

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)

    await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png"
    )

    assert call_order == ["voice", "video", "lipsync"]


@pytest.mark.asyncio
async def test_generate_shot_video_returns_synced_clip_and_audio(tmp_path) -> None:
    expected_synced = VideoClip(uri="/synced.mp4", duration_sec=3.5, shot_id="s01")
    expected_audio = AudioTrack(uri="/dlg.wav", duration_sec=2.0, speaker_id="max")

    adapters = MagicMock()
    adapters.voice.run = AsyncMock(return_value=expected_audio)
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/raw.mp4", duration_sec=3.5, shot_id="s01")
    )
    adapters.lipsync.run = AsyncMock(return_value=expected_synced)
    adapters.video.native_lipsync = False

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)

    synced, audio = await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png"
    )

    assert synced is expected_synced
    assert audio is expected_audio


@pytest.mark.asyncio
async def test_generate_shot_video_passes_speaker_id_to_voice(tmp_path) -> None:
    adapters = MagicMock()
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/d.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.lipsync.run = AsyncMock(
        return_value=VideoClip(uri="/s.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.video.native_lipsync = False

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)

    await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png"
    )

    voice_req = adapters.voice.run.call_args[0][0]
    assert voice_req.speaker_id == "max"


@pytest.mark.asyncio
async def test_generate_shot_video_passes_shot_id_to_lipsync(tmp_path) -> None:
    adapters = MagicMock()
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/d.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.lipsync.run = AsyncMock(
        return_value=VideoClip(uri="/s.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.video.native_lipsync = False

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)

    await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png"
    )

    lipsync_req = adapters.lipsync.run.call_args[0][0]
    assert lipsync_req.shot_id == "s01"


@pytest.mark.asyncio
async def test_generate_shot_video_skips_lipsync_for_native(tmp_path) -> None:
    """When video adapter has native_lipsync=True, lipsync adapter is not called."""
    adapters = MagicMock()
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/d.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.video.native_lipsync = True

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)

    # Create the clip.mp4 so resume check passes
    clip_dir = tmp_path / "video" / "s01"
    clip_dir.mkdir(parents=True)
    # native_lipsync path: video run returns the final clip directly
    synced, _ = await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png"
    )

    adapters.lipsync.run.assert_not_called()
    assert synced.uri == "/c.mp4"


@pytest.mark.asyncio
async def test_generate_shot_video_emits_shot_progress_events(tmp_path) -> None:
    """stage_hook should see start/complete events for voice, video, lip_sync,
    and a final shot_complete — each carrying shot_id and a labeled message."""
    from core.workflow import RunOptions

    adapters = MagicMock()
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/d.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.lipsync.run = AsyncMock(
        return_value=VideoClip(uri="/s.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.video.native_lipsync = False

    events: list[tuple] = []

    def stage_hook(stage_name, event_type, **kw):
        events.append((stage_name, event_type, kw.get("shot_id"), kw.get("message")))

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)

    await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png",
        RunOptions(stage_hook=stage_hook),
        shot_index=3, total_shots=14,
    )

    stage_names = [e[0] for e in events]
    assert stage_names == [
        "synthesize_voice", "synthesize_voice",
        "generate_video", "generate_video",
        "lip_sync", "lip_sync",
        "shot_complete",
    ]
    assert all(e[2] == "s01" for e in events)  # shot_id on every event
    assert events[0][3] == "Shot s01 (3/14): synthesizing voice"
    assert events[1][3] == "Shot s01 (3/14): voice done"
    assert events[-1][3] == "Shot s01 (3/14): shot complete"


@pytest.mark.asyncio
async def test_generate_shot_video_resume_uses_adapter_work_dir(tmp_path) -> None:
    """Resume checks the adapter's namespaced work_dir, not the flat video/ dir.

    This ensures switching video_adapter (e.g. ltx→wan) doesn't hit the old
    adapter's cached clip — each adapter writes to its own subdirectory.
    """
    from core.workflow import RunOptions

    # Simulate an LTX clip already on disk under the LTX namespace.
    ltx_dir = tmp_path / "video" / "ltx" / "s01"
    ltx_dir.mkdir(parents=True)
    (ltx_dir / "clip.mp4").write_bytes(b"ltx-fake")

    # Set up a Wan adapter whose work_dir points to the wan namespace —
    # no clip exists there, so generate_video must run.
    adapters = MagicMock()
    adapters.video.work_dir = tmp_path / "video" / "wan"
    adapters.video.native_lipsync = False
    adapters.lipsync.work_dir = tmp_path / "synced" / "wan"
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/d.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=1.0, shot_id="s01")
    )
    adapters.lipsync.run = AsyncMock(
        return_value=VideoClip(uri="/s.mp4", duration_sec=1.0, shot_id="s01")
    )

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)
    opts = RunOptions(resume=True)

    await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png",
        options=opts,
    )

    # Video adapter MUST have been called — the LTX clip must not poison the Wan cache.
    adapters.video.run.assert_called_once()


@pytest.mark.asyncio
async def test_generate_shot_video_resume_skips_when_clip_exists(tmp_path) -> None:
    """Resume skips video generation when clip exists under the adapter's own dir."""
    from core.workflow import RunOptions

    # Create clip under the adapter's namespaced dir.
    clip_dir = tmp_path / "video" / "ltx" / "s01"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.mp4").write_bytes(b"fake")
    # Audio file so the early return can reconstruct AudioTrack.
    audio_dir = tmp_path / "audio" / "max"
    audio_dir.mkdir(parents=True)
    (audio_dir / "dlg.wav").write_bytes(b"fake-audio")

    adapters = MagicMock()
    adapters.video.work_dir = tmp_path / "video" / "ltx"
    adapters.video.native_lipsync = True
    adapters.voice.run = AsyncMock()
    adapters.video.run = AsyncMock()

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)
    opts = RunOptions(resume=True)

    clip, audio = await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png",
        options=opts,
    )

    # Everything should be skipped — adapter NOT called.
    adapters.voice.run.assert_not_called()
    adapters.video.run.assert_not_called()
    assert "clip.mp4" in clip.uri


@pytest.mark.asyncio
async def test_generate_shot_video_lipsync_failure_falls_back_to_raw_clip(tmp_path) -> None:
    """When lip_sync fails, the pipeline should use the raw clip instead of failing."""
    raw_clip = VideoClip(uri="/raw.mp4", duration_sec=3.0, shot_id="s01")

    adapters = MagicMock()
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/d.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(return_value=raw_clip)
    adapters.video.native_lipsync = False
    adapters.video.work_dir = tmp_path / "video" / "wan"
    adapters.lipsync.work_dir = tmp_path / "synced" / "wan"
    adapters.lipsync.run = AsyncMock(side_effect=RuntimeError("MuseTalk crashed"))

    shot = _storyboard().shots[0]
    config = _make_config(tmp_path)

    synced, audio = await _generate_shot_video(
        shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png"
    )

    # Should fall back to the raw clip, not raise.
    assert synced is raw_clip
    adapters.lipsync.run.assert_called_once()


# ================================================================== source_audio helpers
# _build_verbatim_script, _apply_segment_timing_to_storyboard, _slice_source_audio


def _transcript_result():
    """Minimal TranscribeResult with timed segments for source_audio tests."""
    from core.models.capabilities import TranscriptSegment
    return TranscribeResult(
        segments=[
            TranscriptSegment(text="Hello everyone!", start=0.0, end=3.5),
            TranscriptSegment(text="Let's count together.", start=3.5, end=7.0),
        ],
        language="en",
        full_text="Hello everyone! Let's count together.",
    )


def _test_cast():
    """Cast fixture with all required CastMember fields for source_audio tests."""
    from core.models.profile import Cast, CastMember
    return Cast(
        id="kids_duo",
        species="human",
        is_original_synthetic=True,
        members=[
            CastMember(id="max", name="Max", visual_descriptor="a cartoon boy",
                       lora_ref="loras/kids_duo/max", voice_profile_ref="voices/kids_duo/max",
                       personality="curious"),
            CastMember(id="zoe", name="Zoe", visual_descriptor="a cartoon girl",
                       lora_ref="loras/kids_duo/zoe", voice_profile_ref="voices/kids_duo/zoe",
                       personality="cheerful"),
        ],
    )


def test_build_verbatim_script_creates_lines_from_segments():
    cast = _test_cast()
    transcript = _transcript_result()
    script = _build_verbatim_script(transcript, cast)

    assert script.mode == "verbatim"
    all_lines = [line for scene in script.scenes for line in scene.lines]
    assert len(all_lines) == 2
    assert all_lines[0].text == "Hello everyone!"
    assert all_lines[0].speaker == "max"
    assert all_lines[0].start == 0.0
    assert all_lines[0].end == 3.5
    assert all_lines[1].speaker == "zoe"
    assert all_lines[1].start == 3.5
    assert all_lines[1].end == 7.0


def test_build_verbatim_script_single_member_cast():
    from core.models.profile import Cast, CastMember

    cast = Cast(
        id="solo",
        species="human",
        is_original_synthetic=True,
        members=[CastMember(id="max", name="Max", visual_descriptor="a cartoon boy",
                            lora_ref="loras/solo/max", voice_profile_ref="voices/solo/max",
                            personality="curious")],
    )
    transcript = _transcript_result()
    script = _build_verbatim_script(transcript, cast)

    all_lines = [line for scene in script.scenes for line in scene.lines]
    assert all(line.speaker == "max" for line in all_lines)


def test_build_verbatim_script_folds_props_into_setting():
    """adapt_script's LLM prompt weaves VisualSegment.props into scene text
    (see _format_visual_block) — this deterministic path has no LLM call to
    do that, so it must fold props in directly or a verbatim script silently
    drops every prop the VLM detected (root cause of a real production bug:
    a job's mic/guitar never reached the script, so plan_shots invented an
    ungrounded "taps her thigh" action instead)."""
    from core.models.capabilities import VisualContext, VisualSegment

    cast = _test_cast()
    transcript = _transcript_result()
    vc = VisualContext(segments=[
        VisualSegment(
            start=0.0, end=7.0, setting="Stage with blurred background lights",
            props=["microphone on stand", "acoustic guitar"],
        ),
    ])

    script = _build_verbatim_script(transcript, cast, visual_context=vc)

    setting = script.scenes[0].setting
    assert "Stage with blurred background lights" in setting
    assert "microphone on stand" in setting
    assert "acoustic guitar" in setting


def test_build_verbatim_script_setting_unchanged_without_props():
    from core.models.capabilities import VisualContext, VisualSegment

    cast = _test_cast()
    transcript = _transcript_result()
    vc = VisualContext(segments=[VisualSegment(start=0.0, end=7.0, setting="cozy kitchen")])

    script = _build_verbatim_script(transcript, cast, visual_context=vc)

    assert script.scenes[0].setting == "cozy kitchen"


def test_apply_segment_timing_overrides_duration():
    script = Script(
        mode="verbatim",
        learning_objective=LearningObjective(
            concept="test", age_range="3-6", success_phrase="test",
        ),
        scenes=[Scene(
            setting="room",
            lines=[
                Line(speaker="max", text="Hello!", start=1.0, end=4.5),
                Line(speaker="zoe", text="Hi!", start=4.5, end=6.0),
            ],
        )],
        caption_text="test",
        source_rights=SourceRights(kind="transformed", rights_cleared=True, notes=""),
    )
    storyboard = Storyboard(shots=[
        Shot(
            shot_id="s01", scene_ref="scene-1",
            characters_on_screen=["max"], setting="room",
            camera="medium", action="waves",
            dialogue_line_refs=["scene-1-line-0"], duration_sec=5.0,
        ),
        Shot(
            shot_id="s02", scene_ref="scene-1",
            characters_on_screen=["zoe"], setting="room",
            camera="medium", action="smiles",
            dialogue_line_refs=["scene-1-line-1"], duration_sec=5.0,
        ),
    ])

    result = _apply_segment_timing_to_storyboard(storyboard, script)

    assert result.shots[0].duration_sec == 3.5
    assert result.shots[0].source_start_sec == 1.0
    assert result.shots[0].source_end_sec == 4.5
    assert result.shots[1].duration_sec == 1.5
    assert result.shots[1].source_start_sec == 4.5
    assert result.shots[1].source_end_sec == 6.0


def test_apply_source_timing_to_script_direct_segment_mapping():
    script = _script()
    script.scenes[0].lines.append(Line(speaker="max", text="Again!"))
    transcript = _transcript_result()

    result = _apply_source_timing_to_script(script, transcript, total_duration=7.0)

    assert result.scenes[0].lines[0].start == 0.0
    assert result.scenes[0].lines[0].end == 3.5
    assert result.scenes[0].lines[1].start == 3.5
    assert result.scenes[0].lines[1].end == 7.0


def test_apply_source_timing_to_script_distributes_when_line_count_differs():
    script = _script()
    script.scenes[0].lines.append(Line(speaker="max", text="One two three four five."))
    script.scenes[0].lines.append(Line(speaker="max", text="Done."))
    transcript = _transcript_result()

    result = _apply_source_timing_to_script(script, transcript, total_duration=9.0)
    lines = result.scenes[0].lines

    assert lines[0].start == 0.0
    assert lines[-1].end == 9.0
    assert all(line.start is not None and line.end is not None for line in lines)
    assert all(lines[i].end <= lines[i + 1].start for i in range(len(lines) - 1))


def test_validate_timed_storyboard_rejects_source_audio_without_source_bounds():
    with pytest.raises(ValueError, match="source_audio mode requires source-timed"):
        _validate_timed_storyboard(_storyboard(), "source_audio")


def test_validate_timed_storyboard_accepts_full_without_source_bounds():
    _validate_timed_storyboard(_storyboard(), "full")


# ------------------------------------------------------------------ _chunk_shot_durations

def test_chunk_shot_durations_even_split():
    """15s total / 5s max → three 5s chunks."""
    chunks = _chunk_shot_durations(15.0, 5.0)
    assert chunks == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


def test_chunk_shot_durations_greedy_remainder():
    """15s total / 8s max → one 8s chunk, one 7s chunk (not two 7.5s halves)."""
    chunks = _chunk_shot_durations(15.0, 8.0)
    assert chunks == [(0.0, 8.0), (8.0, 15.0)]
    durations = [end - start for start, end in chunks]
    assert durations == [8.0, 7.0]
    assert sum(durations) == 15.0


def test_chunk_shot_durations_single_chunk_when_shorter_than_max():
    chunks = _chunk_shot_durations(4.0, 8.0)
    assert chunks == [(0.0, 4.0)]


def test_chunk_shot_durations_exact_multiple_has_no_trailing_empty_chunk():
    chunks = _chunk_shot_durations(16.0, 8.0)
    assert chunks == [(0.0, 8.0), (8.0, 16.0)]


def test_chunk_shot_durations_empty_for_zero_duration():
    assert _chunk_shot_durations(0.0, 8.0) == []


# ------------------------------------------------------------------ _rebuild_storyboard_by_duration

def _duration_test_script() -> Script:
    return Script(
        mode="verbatim",
        learning_objective=LearningObjective(
            concept="test", age_range="3-6", success_phrase="test",
        ),
        scenes=[Scene(
            setting="room",
            lines=[
                Line(speaker="max", text="Hello there!", start=0.0, end=3.0),
                Line(speaker="max", text="Let's count to ten.", start=3.0, end=9.0),
                Line(speaker="zoe", text="One two three!", start=9.0, end=15.0),
            ],
        )],
        caption_text="test",
        source_rights=SourceRights(kind="transformed", rights_cleared=True, notes=""),
    )


def test_rebuild_storyboard_preserves_total_duration():
    """The chunked shots' durations must sum back to the original video length,
    regardless of how many shots the LLM originally planned."""
    script = _duration_test_script()
    storyboard = Storyboard(shots=[
        Shot(
            shot_id="s01", scene_ref="scene-1",
            characters_on_screen=["max"], setting="cozy room",
            camera="wide shot", action="greets the viewer excitedly",
            dialogue_line_refs=["scene-1-line-0", "scene-1-line-1"],
            duration_sec=9.0,
        ),
    ])

    result = _rebuild_storyboard_by_duration(
        storyboard, script, total_duration=15.0, max_shot_duration_sec=8.0,
    )

    assert [s.duration_sec for s in result.shots] == [8.0, 7.0]
    assert sum(s.duration_sec for s in result.shots) == 15.0
    assert result.shots[0].source_start_sec == 0.0
    assert result.shots[0].source_end_sec == 8.0
    assert result.shots[1].source_start_sec == 8.0
    assert result.shots[1].source_end_sec == 15.0
    # Shot IDs are freshly assigned, not inherited from the original plan.
    assert [s.shot_id for s in result.shots] == ["s01", "s02"]


def test_rebuild_storyboard_borrows_camera_action_from_best_overlapping_shot():
    """Each chunk should inherit creative direction from whichever originally
    planned shot overlaps its time range the most."""
    script = _duration_test_script()
    storyboard = Storyboard(shots=[
        Shot(
            shot_id="s01", scene_ref="scene-1",
            characters_on_screen=["max"], setting="cozy room",
            camera="close-up", action="greets the viewer excitedly",
            dialogue_line_refs=["scene-1-line-0"], duration_sec=3.0,
        ),
        Shot(
            shot_id="s02", scene_ref="scene-1",
            characters_on_screen=["zoe"], setting="playroom",
            camera="wide shot", action="counts on her fingers",
            dialogue_line_refs=["scene-1-line-1", "scene-1-line-2"], duration_sec=12.0,
        ),
    ])

    result = _rebuild_storyboard_by_duration(
        storyboard, script, total_duration=15.0, max_shot_duration_sec=5.0,
    )

    assert len(result.shots) == 3
    # Chunk 0-5s overlaps original shot s01 (0-3s) by 3s vs s02 (3-15s) by 2s.
    assert result.shots[0].camera == "close-up"
    assert result.shots[0].action == "greets the viewer excitedly"
    # Chunks 5-10s and 10-15s are fully inside original shot s02's 3-15s span.
    assert result.shots[1].camera == "wide shot"
    assert result.shots[2].camera == "wide shot"


def test_rebuild_storyboard_dialogue_refs_match_chunk_time_range():
    """Each chunk's dialogue_line_refs should be whichever verbatim lines'
    midpoints actually fall inside that chunk, not the borrowed shot's refs."""
    script = _duration_test_script()
    storyboard = Storyboard(shots=[
        Shot(
            shot_id="s01", scene_ref="scene-1",
            characters_on_screen=["max"], setting="room",
            camera="medium shot", action="talks to the camera",
            dialogue_line_refs=["scene-1-line-0", "scene-1-line-1", "scene-1-line-2"],
            duration_sec=15.0,
        ),
    ])

    result = _rebuild_storyboard_by_duration(
        storyboard, script, total_duration=15.0, max_shot_duration_sec=5.0,
    )

    # line-0 midpoint=1.5 (chunk0 0-5), line-1 midpoint=6.0 (chunk1 5-10),
    # line-2 midpoint=12.0 (chunk2 10-15).
    assert result.shots[0].dialogue_line_refs == ["scene-1-line-0"]
    assert result.shots[1].dialogue_line_refs == ["scene-1-line-1"]
    assert result.shots[2].dialogue_line_refs == ["scene-1-line-2"]


def test_rebuild_storyboard_returns_original_when_no_chunks():
    script = _duration_test_script()
    storyboard = Storyboard(shots=[
        Shot(
            shot_id="s01", scene_ref="scene-1",
            characters_on_screen=["max"], setting="room",
            camera="medium", action="waves",
            dialogue_line_refs=["scene-1-line-0"], duration_sec=3.0,
        ),
    ])

    result = _rebuild_storyboard_by_duration(
        storyboard, script, total_duration=0.0, max_shot_duration_sec=8.0,
    )

    assert result is storyboard


def test_rebuild_storyboard_by_duration_handles_invalid_old_refs():
    """Source-audio repair may receive a plan from a different script; invalid
    old dialogue refs should not prevent source timing from being rebuilt."""
    script = _duration_test_script()
    storyboard = Storyboard(shots=[
        Shot(
            shot_id="old01", scene_ref="scene-9",
            characters_on_screen=["max"], setting="room",
            camera="medium", action="speaks",
            dialogue_line_refs=["scene-9-line-9"], duration_sec=5.0,
        ),
    ])

    result = _rebuild_storyboard_by_duration(
        storyboard, script, total_duration=15.0, max_shot_duration_sec=8.0,
    )

    assert _storyboard_has_source_timing(result)
    assert result.shots[0].dialogue_line_refs == ["scene-1-line-0", "scene-1-line-1"]
    assert result.shots[1].dialogue_line_refs == ["scene-1-line-2"]
    # New refs are valid against the source-timed script.
    for shot in result.shots:
        for ref in shot.dialogue_line_refs:
            assert _resolve_line(ref, script).text


def test_retime_source_audio_plan_builds_verbatim_timed_artifacts():
    old_storyboard = Storyboard(shots=[
        Shot(
            shot_id="old01", scene_ref="scene-9",
            characters_on_screen=["max"], setting="studio",
            camera="wide", action="points",
            dialogue_line_refs=["scene-9-line-9"], duration_sec=5.0,
        ),
    ])
    fetch = FetchMediaResult(
        video_uri="/video.mp4",
        audio_uri="/audio.wav",
        duration_sec=7.0,
        source_url="file:///video.mp4",
    )

    script, storyboard = _retime_source_audio_plan(
        old_storyboard,
        _transcript_result(),
        fetch,
        _test_cast(),
        max_shot_duration_sec=4.0,
    )

    assert script.mode == "verbatim"
    assert _storyboard_has_source_timing(storyboard)
    assert [s.source_start_sec for s in storyboard.shots] == [0.0, 4.0]
    assert [s.source_end_sec for s in storyboard.shots] == [4.0, 7.0]
    for shot in storyboard.shots:
        for ref in shot.dialogue_line_refs:
            assert _resolve_line(ref, script).text


@pytest.mark.asyncio
async def test_slice_source_audio_calls_ffmpeg(tmp_path):
    src = tmp_path / "source.wav"
    src.write_bytes(b"fake-wav-data")
    out = tmp_path / "sliced" / "s01.wav"

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        track = await _slice_source_audio(str(src), 1.5, 4.0, out)

    assert track.duration_sec == 2.5
    assert track.uri == str(out)
    cmd = mock_exec.call_args[0]
    assert "-ss" in cmd
    assert "1.5" in cmd
    assert "-to" in cmd
    assert "4.0" in cmd


@pytest.mark.asyncio
async def test_fit_audio_to_duration_runs_ffmpeg_filter(tmp_path):
    src = tmp_path / "raw.wav"
    src.write_bytes(b"fake-wav-data")
    out = tmp_path / "timed" / "s01.wav"

    with (
        patch("core.workflow._probe_duration_sec", new=AsyncMock(return_value=1.5)),
        patch("asyncio.create_subprocess_exec") as mock_exec,
    ):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        track = await _fit_audio_to_duration(
            AudioTrack(uri=str(src), duration_sec=1.5, speaker_id="max"),
            3.0,
            out,
            "ffmpeg",
            "ffprobe",
        )

    assert track.uri == str(out)
    assert track.duration_sec == 3.0
    cmd = mock_exec.call_args[0]
    assert "-af" in cmd
    filter_arg = cmd[cmd.index("-af") + 1]
    assert "atempo=" in filter_arg
    assert "atrim=0:3.000" in filter_arg


@pytest.mark.asyncio
async def test_fit_audio_to_duration_copies_close_audio_to_timed_path(tmp_path):
    src = tmp_path / "raw.wav"
    src.write_bytes(b"already-close")
    out = tmp_path / "timed" / "s01.wav"

    with (
        patch("core.workflow._probe_duration_sec", new=AsyncMock(return_value=3.01)),
        patch("asyncio.create_subprocess_exec") as mock_exec,
    ):
        track = await _fit_audio_to_duration(
            AudioTrack(uri=str(src), duration_sec=3.01, speaker_id="max"),
            3.0,
            out,
            "ffmpeg",
            "ffprobe",
        )

    mock_exec.assert_not_called()
    assert track.uri == str(out)
    assert out.read_bytes() == b"already-close"


@pytest.mark.asyncio
async def test_generate_shot_video_source_audio_skips_tts(tmp_path):
    """In source_audio mode, TTS is skipped and source audio is sliced instead."""
    adapters = MagicMock()
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=3.0, shot_id="s01")
    )
    adapters.video.native_lipsync = True
    adapters.video.work_dir = tmp_path / "video" / "ltx"
    adapters.ffmpeg_bin = "ffmpeg"

    shot = Shot(
        shot_id="s01", scene_ref="scene-1",
        characters_on_screen=["max"], setting="cozy classroom",
        camera="medium", action="speaks",
        dialogue_line_refs=["scene-1-line-0"],
        duration_sec=3.5,
        source_start_sec=0.0,
        source_end_sec=3.5,
    )
    config = _make_config(tmp_path)

    with patch("core.workflow._slice_source_audio", new_callable=AsyncMock) as mock_slice:
        mock_slice.return_value = AudioTrack(uri="/sliced.wav", duration_sec=3.5)

        synced, audio = await _generate_shot_video(
            shot, _script(), config.cast, adapters, Path(tmp_path), "/img.png",
            render_mode="source_audio", source_audio_uri="/source.wav",
        )

    mock_slice.assert_called_once()
    adapters.voice.run.assert_not_called()
    assert audio.uri == "/sliced.wav"


@pytest.mark.asyncio
async def test_generate_shot_video_re_voice_fits_tts_and_uses_all_dialogue_refs(tmp_path):
    adapters = MagicMock()
    adapters.voice.run = AsyncMock(
        return_value=AudioTrack(uri="/raw.wav", duration_sec=1.0, speaker_id="max")
    )
    adapters.video.run = AsyncMock(
        return_value=VideoClip(uri="/c.mp4", duration_sec=4.0, shot_id="s01")
    )
    adapters.video.native_lipsync = True
    adapters.video.work_dir = tmp_path / "video" / "ltx"
    adapters.ffmpeg_bin = "ffmpeg"
    adapters.ffprobe_bin = "ffprobe"

    script = Script(
        mode="transformed",
        learning_objective=LearningObjective(
            concept="test", age_range="3-6", success_phrase="test",
        ),
        scenes=[Scene(
            setting="room",
            lines=[
                Line(speaker="max", text="First line.", start=0.0, end=2.0),
                Line(speaker="max", text="Second line.", start=2.0, end=4.0),
            ],
        )],
        caption_text="test",
        source_rights=SourceRights(kind="transformed", rights_cleared=True, notes=""),
    )
    shot = Shot(
        shot_id="s01", scene_ref="scene-1",
        characters_on_screen=["max"], setting="room",
        camera="medium", action="speaks",
        dialogue_line_refs=["scene-1-line-0", "scene-1-line-1"],
        duration_sec=4.0,
        source_start_sec=0.0,
        source_end_sec=4.0,
    )
    config = _make_config(tmp_path)

    with patch("core.workflow._fit_audio_to_duration", new_callable=AsyncMock) as mock_fit:
        mock_fit.return_value = AudioTrack(uri="/timed.wav", duration_sec=4.0, speaker_id="max")
        synced, audio = await _generate_shot_video(
            shot, script, config.cast, adapters, Path(tmp_path), "/img.png",
            render_mode="re_voice",
        )

    voice_req = adapters.voice.run.call_args.args[0]
    assert voice_req.text == "First line. Second line."
    mock_fit.assert_called_once()
    video_req = adapters.video.run.call_args.args[0]
    assert video_req.duration_sec == 4.0
    assert video_req.audio_uri == "/timed.wav"
    assert synced.uri == "/c.mp4"
    assert audio.uri == "/timed.wav"
