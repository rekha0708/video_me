"""End-to-end cast data-flow tests.

Verifies that a non-default cast (solo_fox) is propagated intact through every
layer: config loading → worker per-job config → pipeline stages → story/image
ingest → request serialization round-trip. Each test uses a 1-member fox cast
that is materially different from the default kids_duo to catch assumptions
about member count, IDs, LoRA paths, or voice paths.
"""
from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from core.models.capabilities import (
    AudioTrack,
    FetchMediaResult,
    FinalVideo,
    ImageCritiqueRequest,
    ImageCritiqueResult,
    PublishResult,
    RenderCharacterRequest,
    TranscribeResult,
    VideoClip,
    VideoRequest,
    VisualContext,
    VoiceRequest,
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
from core.models.dashboard import CreateDashboardJobRequest, DashboardSource
from core.models.guardrails import SourceRights
from core.models.profile import Cast, CastMember


# ====================================================================
# Fixtures — solo_fox: a 1-member fox cast, different from kids_duo
# ====================================================================

FOX_DESCRIPTOR = "cartoon red fox with a purple scarf"

def _fox_member() -> CastMember:
    return CastMember(
        id="fox",
        name="Roxy",
        gender="girl",
        visual_descriptor=FOX_DESCRIPTOR,
        lora_ref="loras/solo_fox/fox",
        voice_profile_ref="voices/solo_fox/fox",
        personality="adventurous and curious",
        signature_expressions=["tail wag", "ear perk"],
    )


def _fox_cast() -> Cast:
    return Cast(
        id="solo_fox",
        species="fox",
        is_original_synthetic=True,
        members=[_fox_member()],
    )


def _fox_yaml_dict() -> dict:
    cast = _fox_cast()
    return {
        "id": cast.id,
        "species": cast.species,
        "is_original_synthetic": cast.is_original_synthetic,
        "members": [m.model_dump() for m in cast.members],
    }


def _write_fox_yaml(directory: Path) -> Path:
    """Write solo_fox.yaml under the given directory and return the path."""
    path = directory / "solo_fox.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(_fox_yaml_dict()))
    return path


def _fox_config(tmp_path: Path):
    """Build an AppConfig using the solo_fox cast."""
    from core.config import Settings, AppConfig, load_yaml_model

    cast_path = _write_fox_yaml(tmp_path / "config" / "casts")
    cast = load_yaml_model(cast_path, Cast)

    channel_path = Path("config/channels/education_kids.yaml")
    from core.models.profile import ChannelProfile
    channel = load_yaml_model(channel_path, ChannelProfile)

    settings = Settings(
        data_dir=tmp_path / "data",
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "video_me.db",
        feedback_log_dir=Path(f"assets/{cast.id}"),
    )
    return AppConfig(settings=settings, channel_profile=channel, cast=cast)


# -- data builders (fox variants of test_workflow.py helpers) --

def _fox_script() -> Script:
    return Script(
        mode="transformed",
        learning_objective=LearningObjective(
            concept="exploring", age_range="3-6",
            success_phrase="I explored something new!",
        ),
        scenes=[
            Scene(
                setting="forest clearing",
                characters_present=["fox"],
                lines=[
                    Line(speaker="fox", text="What's behind that tree?", expression="curious"),
                ],
            )
        ],
        caption_text="What's behind that tree?",
        source_rights=SourceRights(kind="transformed", rights_cleared=True, notes=""),
    )


def _fox_storyboard() -> Storyboard:
    return Storyboard(
        shots=[
            Shot(
                shot_id="s01",
                scene_ref="scene-1",
                characters_on_screen=["fox"],
                setting="forest clearing",
                camera="medium shot",
                action="character peeks around a tree",
                dialogue_line_refs=["scene-1-line-0"],
                duration_sec=5.0,
            )
        ]
    )


def _fox_stage_results():
    return {
        "fetch_media": FetchMediaResult(
            video_uri="/tmp/video.mp4", audio_uri="/tmp/audio.wav",
            duration_sec=30.0, source_url="http://example.com/vid",
        ),
        "transcribe": TranscribeResult(segments=[], language="en", full_text="Let's explore!"),
        "analyze_content": ContentMetadata(
            content_genre="education", topic="exploring", tone="playful",
            hook="Let's explore!", pacing="medium", length_sec=30,
            learning_objective=LearningObjective(
                concept="exploring", age_range="3-6",
                success_phrase="I explored something new!",
            ),
        ),
        "analyze_visuals": VisualContext(),
        "adapt_script": _fox_script(),
        "plan_shots": _fox_storyboard(),
        "assemble_video": FinalVideo(uri="/tmp/final.mp4", duration_sec=5.0),
        "publish": PublishResult(review_path="/review/video.mp4", metadata_path="/review/meta.json"),
    }


# ====================================================================
# Layer 1: Config loading + Settings
# ====================================================================


def test_load_app_config_with_custom_cast_path(tmp_path: Path) -> None:
    """load_app_config(cast_path=...) loads the alternate cast and auto-resolves feedback_log_dir."""
    from core.config import load_app_config

    cast_path = _write_fox_yaml(tmp_path / "config" / "casts")
    config = load_app_config(cast_path=cast_path)

    assert config.cast.id == "solo_fox"
    assert config.cast.species == "fox"
    assert len(config.cast.members) == 1
    assert config.cast.members[0].id == "fox"
    assert config.cast.members[0].lora_ref == "loras/solo_fox/fox"
    assert config.cast.members[0].voice_profile_ref == "voices/solo_fox/fox"
    assert config.settings.feedback_log_dir == Path("assets/solo_fox")


def test_load_app_config_custom_feedback_log_dir_not_overridden(tmp_path: Path, monkeypatch) -> None:
    """When VIDEO_ME_FEEDBACK_LOG_DIR is set to a custom path, it should NOT be overridden."""
    from core.config import load_app_config

    custom_dir = str(tmp_path / "custom_feedback")
    monkeypatch.setenv("VIDEO_ME_FEEDBACK_LOG_DIR", custom_dir)

    cast_path = _write_fox_yaml(tmp_path / "config" / "casts")
    config = load_app_config(cast_path=cast_path)

    assert config.cast.id == "solo_fox"
    assert config.settings.feedback_log_dir == Path(custom_dir)


# ====================================================================
# Layer 2: Worker _config_for_job
# ====================================================================


def _make_worker(tmp_path: Path):
    from core.config import load_app_config
    from services.dashboard_repository import DashboardRepository
    from services.dashboard_worker import DashboardWorker

    config = load_app_config()
    config.settings.sqlite_path = tmp_path / "dashboard.db"
    repo = DashboardRepository(tmp_path / "dashboard.db")
    worker = DashboardWorker(repo, config, worker_id="test-worker-1")
    return worker, repo


def test_config_for_job_loads_alternate_cast(tmp_path: Path) -> None:
    """_config_for_job with cast_ref='solo_fox' loads that cast; base config unchanged."""
    _write_fox_yaml(Path("config/casts"))
    worker, _ = _make_worker(tmp_path)
    assert worker.config.cast.id == "kids_duo"

    req = CreateDashboardJobRequest(
        source=DashboardSource(url="http://example.com/v"),
        rights_cleared=True,
        cast_ref="solo_fox",
    )
    try:
        job_config = worker._config_for_job(req)

        assert job_config.cast.id == "solo_fox"
        assert job_config.cast.members[0].id == "fox"
        assert job_config.cast.members[0].lora_ref == "loras/solo_fox/fox"
        assert job_config.cast.members[0].voice_profile_ref == "voices/solo_fox/fox"
        assert job_config.settings.feedback_log_dir == Path("assets/solo_fox")
        # Base config must not be mutated
        assert worker.config.cast.id == "kids_duo"
        assert worker.config.settings.feedback_log_dir != Path("assets/solo_fox")
    finally:
        (Path("config/casts/solo_fox.yaml")).unlink(missing_ok=True)


def test_config_for_job_unknown_cast_raises(tmp_path: Path) -> None:
    worker, _ = _make_worker(tmp_path)
    req = CreateDashboardJobRequest(
        source=DashboardSource(url="http://example.com/v"),
        rights_cleared=True,
        cast_ref="nonexistent_cast_xyz",
    )
    with pytest.raises(ValueError, match="Unknown cast"):
        worker._config_for_job(req)


def test_config_for_job_default_cast_returns_base(tmp_path: Path) -> None:
    worker, _ = _make_worker(tmp_path)
    req = CreateDashboardJobRequest(
        source=DashboardSource(url="http://example.com/v"),
        rights_cleared=True,
        cast_ref="kids_duo",
    )
    result = worker._config_for_job(req)
    assert result is worker.config


def test_config_for_job_none_cast_ref_returns_base(tmp_path: Path) -> None:
    worker, _ = _make_worker(tmp_path)
    req = CreateDashboardJobRequest(
        source=DashboardSource(url="http://example.com/v"),
        rights_cleared=True,
        cast_ref=None,
    )
    result = worker._config_for_job(req)
    assert result is worker.config


# ====================================================================
# Layer 3: Pipeline stages receive correct cast
# ====================================================================


@pytest.mark.asyncio
async def test_pipeline_passes_cast_to_all_stages(tmp_path: Path) -> None:
    """run_pipeline_job with solo_fox config: adapt_script, plan_shots, render,
    approval, and video generation all receive the fox cast, not kids_duo."""
    config = _fox_config(tmp_path)

    captured_stage_requests: dict[str, object] = {}

    async def spy_run_stage(stage_name, capability, request, job, *args, **kw):
        captured_stage_requests[stage_name] = request
        return _fox_stage_results()[stage_name]

    captured_render_cast = {}
    captured_approval_cast_id = {}
    captured_video_cast = {}

    async def spy_render(shot, cast, *args, **kw):
        captured_render_cast["cast"] = cast
        return ImageCritiqueResult(winner_index=0, winner_uri="/tmp/render_00.png")

    async def spy_approval(shots, results, adapters, cast_id):
        captured_approval_cast_id["cast_id"] = cast_id
        return ["/tmp/render_00.png"]

    async def spy_video(shot, script, cast, adapters, work_dir, image_uri, options=None, **_kw):
        captured_video_cast["cast"] = cast
        return (
            VideoClip(uri="/tmp/clip.mp4", duration_sec=5.0, shot_id="s01"),
            AudioTrack(uri="/tmp/audio.wav", duration_sec=2.5, speaker_id="fox"),
        )

    with ExitStack() as stack:
        stack.enter_context(patch(
            "core.workflow._make_adapters", return_value=MagicMock(ffmpeg_bin="ffmpeg"),
        ))
        stack.enter_context(patch("core.workflow.run_stage", new=spy_run_stage))
        stack.enter_context(patch(
            "core.workflow._concat_audio",
            new=AsyncMock(return_value=AudioTrack(uri="/tmp/combined.wav", duration_sec=2.5)),
        ))
        stack.enter_context(patch("core.workflow.create_job_store", return_value=MagicMock()))
        stack.enter_context(patch("core.workflow.create_artifact_store", return_value=MagicMock()))
        stack.enter_context(patch(
            "core.workflow._run_plan_critique_and_approval",
            new=AsyncMock(return_value=(_fox_storyboard(), _fox_script())),
        ))
        stack.enter_context(patch("core.workflow._render_shot_candidates", new=spy_render))
        stack.enter_context(patch("core.workflow._run_image_approval_gate", new=spy_approval))
        stack.enter_context(patch(
            "core.workflow._prepare_shot_audio_tracks", new=AsyncMock(return_value={}),
        ))
        stack.enter_context(patch("core.workflow._generate_shot_video", new=spy_video))
        stack.enter_context(patch("core.workflow.prepare_video_model", new=AsyncMock()))

        from core.workflow import run_pipeline_job
        job = await run_pipeline_job("http://example.com", rights_cleared=True, app_config=config)

    from core.models.job import JobStatus
    assert job.status == JobStatus.COMPLETED
    assert job.cast_ref == "solo_fox"

    # adapt_script received the fox cast
    adapt_req = captured_stage_requests["adapt_script"]
    assert adapt_req.cast.id == "solo_fox"
    assert adapt_req.cast.members[0].id == "fox"
    assert adapt_req.cast.members[0].lora_ref == "loras/solo_fox/fox"

    # plan_shots received the fox cast
    plan_req = captured_stage_requests["plan_shots"]
    assert plan_req.cast.id == "solo_fox"
    assert plan_req.cast.members[0].voice_profile_ref == "voices/solo_fox/fox"

    # _render_shot_candidates received the fox cast
    assert captured_render_cast["cast"].id == "solo_fox"
    assert captured_render_cast["cast"].members[0].id == "fox"

    # _run_image_approval_gate received cast_id="solo_fox"
    assert captured_approval_cast_id["cast_id"] == "solo_fox"

    # _generate_shot_video received the fox cast
    assert captured_video_cast["cast"].id == "solo_fox"
    assert captured_video_cast["cast"].members[0].voice_profile_ref == "voices/solo_fox/fox"


@pytest.mark.asyncio
async def test_render_shot_candidates_uses_cast_member(tmp_path: Path) -> None:
    """_render_shot_candidates resolves the fox member for rendering and critique."""
    from core.models.capabilities import RenderCharacterRequest

    cast = _fox_cast()
    shot = _fox_storyboard().shots[0]

    captured_render_request = {}

    mock_render = AsyncMock()
    async def spy_render_run(request):
        captured_render_request["req"] = request
        from core.models.capabilities import ImageSet
        return ImageSet(member_id="fox", images=["/tmp/render_00.png"])
    mock_render.run = spy_render_run
    mock_render._num_images = 1

    mock_critique = AsyncMock()
    mock_critique.run = AsyncMock(return_value=ImageCritiqueResult(
        winner_index=0, winner_uri="/tmp/render_00.png",
        candidate_uris=["/tmp/render_00.png"],
    ))

    adapters = MagicMock()
    adapters.render = mock_render
    adapters.image_critique = mock_critique

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    from core.workflow import _render_shot_candidates
    result = await _render_shot_candidates(shot, cast, adapters, work_dir)

    req = captured_render_request["req"]
    assert isinstance(req, RenderCharacterRequest)
    assert req.member.id == "fox"
    assert req.member.lora_ref == "loras/solo_fox/fox"
    assert req.member.visual_descriptor == FOX_DESCRIPTOR
    assert result.winner_uri == "/tmp/render_00.png"


@pytest.mark.asyncio
async def test_generate_shot_video_uses_cast_voice_profile(tmp_path: Path) -> None:
    """_generate_shot_video sends the fox voice_profile_ref and speaker_id to synthesize_voice."""
    from core.models.capabilities import VoiceRequest

    cast = _fox_cast()
    script = _fox_script()
    shot = _fox_storyboard().shots[0]

    captured_voice_request = {}

    mock_voice = AsyncMock()
    async def spy_voice_run(request):
        captured_voice_request["req"] = request
        return AudioTrack(uri="/tmp/fox_audio.wav", duration_sec=2.5, speaker_id="fox")
    mock_voice.run = spy_voice_run

    mock_video = AsyncMock()
    mock_video.native_lipsync = True
    mock_video.run = AsyncMock(return_value=VideoClip(
        uri="/tmp/clip.mp4", duration_sec=5.0, shot_id="s01",
    ))

    mock_lipsync = AsyncMock()

    adapters = MagicMock()
    adapters.voice = mock_voice
    adapters.video = mock_video
    adapters.lipsync = mock_lipsync

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    from core.workflow import _generate_shot_video
    synced, audio = await _generate_shot_video(
        shot, script, cast, adapters, work_dir, "/tmp/render_00.png"
    )

    req = captured_voice_request["req"]
    assert isinstance(req, VoiceRequest)
    assert req.voice_profile_ref == "voices/solo_fox/fox"
    assert req.speaker_id == "fox"
    assert req.text == "What's behind that tree?"
    assert audio.speaker_id == "fox"


# ====================================================================
# Layer 4: Story + images flow
# ====================================================================


def test_build_user_image_critiques_solo_cast() -> None:
    """_build_user_image_critiques maps user images to shots using the fox cast."""
    from core.workflow import _build_user_image_critiques

    cast = _fox_cast()
    shots = _fox_storyboard().shots
    user_images = {"fox": "/tmp/fox_ref.png"}

    results = _build_user_image_critiques(shots, user_images, cast)

    assert len(results) == 1
    assert results[0].winner_uri == "/tmp/fox_ref.png"
    assert results[0].winner_index == 0
    assert results[0].origin == "user"
    assert results[0].candidate_uris == ["/tmp/fox_ref.png"]
    assert "fox" in results[0].overall_reasoning


@pytest.mark.asyncio
async def test_seed_story_job_copies_character_images(tmp_path: Path) -> None:
    """_seed_story_job copies user images and returns the mapped paths."""
    from services.dashboard_repository import DashboardRepository
    from services.dashboard_worker import DashboardWorker
    from core.config import load_app_config

    config = load_app_config()
    config.settings.data_dir = tmp_path / "data"
    config.settings.artifact_dir = tmp_path / "artifacts"
    config.settings.sqlite_path = tmp_path / "dashboard.db"

    repo = DashboardRepository(tmp_path / "dashboard.db")
    worker = DashboardWorker(repo, config, worker_id="test-w")

    # Create a fake source image
    src_image = tmp_path / "uploads" / "fox.png"
    src_image.parent.mkdir(parents=True, exist_ok=True)
    src_image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    req = CreateDashboardJobRequest(
        source=DashboardSource(kind="story_images", url=""),
        rights_cleared=True,
        story_text="0-4: Roxy explores the forest.",
        character_images={"fox": str(src_image)},
        cast_ref="solo_fox",
    )
    job, _ = repo.create_queued_job(req)

    user_images = await worker._seed_story_job(req, job.job_id)

    assert user_images is not None
    assert "fox" in user_images
    dest_path = Path(user_images["fox"])
    assert dest_path.exists()
    assert dest_path.name == "fox.png"
    assert "user_images" in str(dest_path)

    # fetch_media and transcribe artifacts should exist
    from core.storage import create_artifact_store
    store = create_artifact_store(config.settings)
    assert store.has(job.job_id, "fetch_media")
    assert store.has(job.job_id, "transcribe")


# ====================================================================
# Layer 5: Request serialization round-trip
# ====================================================================


def test_cast_ref_survives_model_dump_round_trip() -> None:
    """cast_ref, story_text, character_images survive Pydantic model_dump → reconstruct."""
    original = CreateDashboardJobRequest(
        source=DashboardSource(kind="story_images", url=""),
        rights_cleared=True,
        cast_ref="solo_fox",
        story_text="0-4: Roxy finds a treasure chest.",
        character_images={"fox": "/tmp/uploads/fox.png"},
        target_language="hi",
    )

    serialized = original.model_dump(mode="json")
    restored = CreateDashboardJobRequest(**serialized)

    assert restored.cast_ref == "solo_fox"
    assert restored.story_text == "0-4: Roxy finds a treasure chest."
    assert restored.character_images == {"fox": "/tmp/uploads/fox.png"}
    assert restored.source.kind == "story_images"
    assert restored.rights_cleared is True
    assert restored.target_language == "hi"


def test_queue_round_trip_stores_and_restores_cast_ref(tmp_path: Path) -> None:
    """cast_ref survives the DashboardRepository → SQLite → claim → reconstruct round-trip."""
    from services.dashboard_repository import DashboardRepository

    repo = DashboardRepository(tmp_path / "dashboard.db")

    original = CreateDashboardJobRequest(
        source=DashboardSource(kind="story_images", url=""),
        rights_cleared=True,
        cast_ref="solo_fox",
        story_text="0-4: Roxy finds a treasure chest.",
        character_images={"fox": "/tmp/uploads/fox.png"},
        target_language="hi",
        phase="all",
    )

    job, queue_item = repo.create_queued_job(original)
    assert job is not None

    # Claim the action (simulates worker picking it up)
    action = repo.claim_next_action("test-worker-1")
    assert action is not None

    # Reconstruct the request from the queue payload (how the worker does it)
    restored = CreateDashboardJobRequest(**action.payload)

    assert restored.cast_ref == "solo_fox"
    assert restored.story_text == "0-4: Roxy finds a treasure chest."
    assert restored.character_images == {"fox": "/tmp/uploads/fox.png"}
    assert restored.source.kind == "story_images"
    assert restored.rights_cleared is True
    assert restored.target_language == "hi"
    assert restored.phase == "all"


# ====================================================================
# Layer 5: Multi-character shot rendering + reaction flip
# ====================================================================

MAX_DESCRIPTOR = "cartoon boy with blue striped shirt and red sneakers"
ZOE_DESCRIPTOR = "cartoon girl with pink bow and yellow sundress"


def _duo_member_max() -> CastMember:
    return CastMember(
        id="max", name="Max", gender="boy",
        visual_descriptor=MAX_DESCRIPTOR,
        lora_ref="loras/kids_duo/max", voice_profile_ref="voices/kids_duo/max",
        personality="eager teacher", signature_expressions=["grin"],
    )


def _duo_member_zoe() -> CastMember:
    return CastMember(
        id="zoe", name="Zoe", gender="girl",
        visual_descriptor=ZOE_DESCRIPTOR,
        lora_ref="loras/kids_duo/zoe", voice_profile_ref="voices/kids_duo/zoe",
        personality="playful learner", signature_expressions=["giggle"],
    )


def _duo_cast() -> Cast:
    return Cast(
        id="test_duo",
        species="human",
        is_original_synthetic=True,
        members=[_duo_member_max(), _duo_member_zoe()],
    )


def _duo_script() -> Script:
    return Script(
        mode="transformed",
        learning_objective=LearningObjective(
            concept="counting", age_range="3-6",
            success_phrase="We can count!",
        ),
        scenes=[
            Scene(
                setting="playground",
                characters_present=["max", "zoe"],
                lines=[
                    Line(speaker="max", text="Let's count!", expression="excited"),
                    Line(speaker="zoe", text="One, two, three!", expression="happy"),
                ],
            )
        ],
        caption_text="Let's count! One, two, three!",
        source_rights=SourceRights(kind="transformed", rights_cleared=True, notes=""),
    )


def _duo_storyboard() -> Storyboard:
    return Storyboard(shots=[
        Shot(
            shot_id="s01", scene_ref="scene-1",
            characters_on_screen=["max", "zoe"],
            setting="playground", camera="medium",
            action="Max teaches Zoe to count",
            dialogue_line_refs=["scene-1-line-0"],
            duration_sec=6.0,
        ),
        Shot(
            shot_id="s02", scene_ref="scene-1",
            characters_on_screen=["zoe", "max"],
            setting="playground", camera="reaction",
            action="Zoe reacts to counting lesson",
            dialogue_line_refs=["scene-1-line-1"],
            duration_sec=5.0,
        ),
        Shot(
            shot_id="s03", scene_ref="scene-1",
            characters_on_screen=["max"],
            setting="playground", camera="close-up",
            action="Max celebrates",
            dialogue_line_refs=["scene-1-line-0"],
            duration_sec=5.0,
        ),
    ])


def _make_render_spy():
    """Return (mock_render, captured_requests) for render adapter spying."""
    captured = []

    mock_render = AsyncMock()
    async def spy_render_run(request):
        captured.append(request)
        from core.models.capabilities import ImageSet
        return ImageSet(member_id=request.member.id, images=["/tmp/render_00.png"])
    mock_render.run = spy_render_run
    mock_render._num_images = 1
    return mock_render, captured


def _make_critique_mock():
    return AsyncMock(return_value=ImageCritiqueResult(
        winner_index=0, winner_uri="/tmp/render_00.png",
        candidate_uris=["/tmp/render_00.png"],
    ))


def _make_adapters(render_mock, critique_mock):
    adapters = MagicMock()
    adapters.render = render_mock
    adapters.image_critique = MagicMock()
    adapters.image_critique.run = critique_mock
    return adapters


@pytest.mark.asyncio
async def test_render_multi_char_includes_other_member(tmp_path: Path) -> None:
    """Two-character shot passes the second member in other_members."""
    mock_render, captured = _make_render_spy()
    adapters = _make_adapters(mock_render, _make_critique_mock())

    cast = _duo_cast()
    shot = _duo_storyboard().shots[0]  # max+zoe, camera=medium

    from core.workflow import _render_shot_candidates
    await _render_shot_candidates(shot, cast, adapters, tmp_path)

    req = captured[0]
    assert isinstance(req, RenderCharacterRequest)
    assert req.member.id == "max"
    assert len(req.other_members) == 1
    assert req.other_members[0].id == "zoe"
    assert req.other_members[0].visual_descriptor == ZOE_DESCRIPTOR


@pytest.mark.asyncio
async def test_render_reaction_flips_primary(tmp_path: Path) -> None:
    """Reaction shot renders the listener (index 1) as primary."""
    mock_render, captured = _make_render_spy()
    adapters = _make_adapters(mock_render, _make_critique_mock())

    cast = _duo_cast()
    shot = _duo_storyboard().shots[1]  # zoe+max, camera=reaction

    from core.workflow import _render_shot_candidates
    await _render_shot_candidates(shot, cast, adapters, tmp_path)

    req = captured[0]
    assert req.member.id == "max"  # listener (index 1) becomes primary
    assert len(req.other_members) == 1
    assert req.other_members[0].id == "zoe"  # speaker goes to other_members


@pytest.mark.asyncio
async def test_render_single_char_no_other_members(tmp_path: Path) -> None:
    """Single-character shot has empty other_members (regression guard)."""
    mock_render, captured = _make_render_spy()
    adapters = _make_adapters(mock_render, _make_critique_mock())

    cast = _duo_cast()
    shot = _duo_storyboard().shots[2]  # max only, camera=close-up

    from core.workflow import _render_shot_candidates
    await _render_shot_candidates(shot, cast, adapters, tmp_path)

    req = captured[0]
    assert req.member.id == "max"
    assert req.other_members == []


@pytest.mark.asyncio
async def test_critique_request_has_other_descriptors(tmp_path: Path) -> None:
    """Two-character shot passes other_descriptors to image critique.

    Uses 2 candidates — with a single candidate the VLM critique is skipped
    entirely (auto-pick), so this flow only exists for N ≥ 2.
    """
    mock_render = AsyncMock()

    async def spy_render_run(request):
        from core.models.capabilities import ImageSet
        return ImageSet(member_id=request.member.id,
                        images=["/tmp/render_00.png", "/tmp/render_01.png"])

    mock_render.run = spy_render_run
    mock_render._num_images = 2
    captured_critique = []

    async def spy_critique(request):
        captured_critique.append(request)
        return ImageCritiqueResult(
            winner_index=0, winner_uri="/tmp/render_00.png",
            candidate_uris=["/tmp/render_00.png", "/tmp/render_01.png"],
        )

    adapters = MagicMock()
    adapters.render = mock_render
    adapters.image_critique = MagicMock()
    adapters.image_critique.run = spy_critique

    cast = _duo_cast()
    shot = _duo_storyboard().shots[0]  # max+zoe

    from core.workflow import _render_shot_candidates
    await _render_shot_candidates(shot, cast, adapters, tmp_path)

    creq = captured_critique[0]
    assert isinstance(creq, ImageCritiqueRequest)
    assert creq.cast_descriptor == MAX_DESCRIPTOR
    assert creq.other_descriptors == [ZOE_DESCRIPTOR]


@pytest.mark.asyncio
async def test_render_pair_lora_used_when_available(tmp_path: Path) -> None:
    """When a pair LoRA with non-empty lora_file exists, it's used and other_members is empty."""
    from core.cast_params import CastPairParams, _PAIR_CACHE

    mock_render, captured = _make_render_spy()
    adapters = _make_adapters(mock_render, _make_critique_mock())

    cast = _duo_cast()
    shot = _duo_storyboard().shots[0]

    pair_key = frozenset({"max", "zoe"})
    _PAIR_CACHE[(cast.id, "config/casts")] = {
        pair_key: CastPairParams(lora_file="duo_max_zoe.safetensors", lora_weight=0.85),
    }

    try:
        from core.workflow import _render_shot_candidates
        await _render_shot_candidates(shot, cast, adapters, tmp_path)

        req = captured[0]
        assert req.lora_file == "duo_max_zoe.safetensors"
        assert req.lora_weight == 0.85
        assert req.other_members == []
    finally:
        _PAIR_CACHE.clear()


@pytest.mark.asyncio
async def test_render_pair_lora_fallback_when_empty(tmp_path: Path) -> None:
    """Pair LoRA with empty lora_file falls back to individual LoRA + other_members."""
    from core.cast_params import CastPairParams, _PAIR_CACHE

    mock_render, captured = _make_render_spy()
    adapters = _make_adapters(mock_render, _make_critique_mock())

    cast = _duo_cast()
    shot = _duo_storyboard().shots[0]

    pair_key = frozenset({"max", "zoe"})
    _PAIR_CACHE[(cast.id, "config/casts")] = {
        pair_key: CastPairParams(lora_file=""),
    }

    try:
        from core.workflow import _render_shot_candidates
        await _render_shot_candidates(shot, cast, adapters, tmp_path)

        req = captured[0]
        assert req.lora_file != "duo_max_zoe.safetensors"
        assert len(req.other_members) == 1
        assert req.other_members[0].id == "zoe"
    finally:
        _PAIR_CACHE.clear()


@pytest.mark.asyncio
async def test_video_action_enriched_for_two_chars(tmp_path: Path) -> None:
    """Two-character shot enriches VideoRequest.action with speaker/listener roles."""
    cast = _duo_cast()
    script = _duo_script()
    shot = _duo_storyboard().shots[0]  # max+zoe

    captured_video = []

    mock_voice = AsyncMock()
    mock_voice.run = AsyncMock(return_value=AudioTrack(
        uri="/tmp/audio.wav", duration_sec=3.0, speaker_id="max",
    ))

    mock_video = AsyncMock()
    mock_video.native_lipsync = True
    async def spy_video(request):
        captured_video.append(request)
        return VideoClip(uri="/tmp/clip.mp4", duration_sec=6.0, shot_id="s01")
    mock_video.run = spy_video

    adapters = MagicMock()
    adapters.voice = mock_voice
    adapters.video = mock_video
    adapters.lipsync = AsyncMock()

    from core.workflow import _generate_shot_video
    await _generate_shot_video(shot, script, cast, adapters, tmp_path, "/tmp/img.png")

    vreq = captured_video[0]
    assert isinstance(vreq, VideoRequest)
    assert "Max is speaking" in vreq.action
    assert "Zoe is listening" in vreq.action
    assert "two characters visible" in vreq.action


@pytest.mark.asyncio
async def test_video_single_char_action_unchanged(tmp_path: Path) -> None:
    """Single-character shot leaves action string unchanged."""
    cast = _duo_cast()
    script = _duo_script()
    shot = _duo_storyboard().shots[2]  # max only

    captured_video = []

    mock_voice = AsyncMock()
    mock_voice.run = AsyncMock(return_value=AudioTrack(
        uri="/tmp/audio.wav", duration_sec=3.0, speaker_id="max",
    ))

    mock_video = AsyncMock()
    mock_video.native_lipsync = True
    async def spy_video(request):
        captured_video.append(request)
        return VideoClip(uri="/tmp/clip.mp4", duration_sec=5.0, shot_id="s03")
    mock_video.run = spy_video

    adapters = MagicMock()
    adapters.voice = mock_voice
    adapters.video = mock_video
    adapters.lipsync = AsyncMock()

    from core.workflow import _generate_shot_video
    await _generate_shot_video(shot, script, cast, adapters, tmp_path, "/tmp/img.png")

    vreq = captured_video[0]
    assert vreq.action == shot.action
    assert "is speaking" not in vreq.action


@pytest.mark.asyncio
async def test_video_voice_uses_speaker_not_listener(tmp_path: Path) -> None:
    """Reaction shot: voice synthesis still uses characters_on_screen[0] (speaker)."""
    cast = _duo_cast()
    script = _duo_script()
    shot = _duo_storyboard().shots[1]  # zoe+max, camera=reaction

    captured_voice = []

    mock_voice = AsyncMock()
    async def spy_voice(request):
        captured_voice.append(request)
        return AudioTrack(uri="/tmp/audio.wav", duration_sec=3.0, speaker_id="zoe")
    mock_voice.run = spy_voice

    mock_video = AsyncMock()
    mock_video.native_lipsync = True
    mock_video.run = AsyncMock(return_value=VideoClip(
        uri="/tmp/clip.mp4", duration_sec=5.0, shot_id="s02",
    ))

    adapters = MagicMock()
    adapters.voice = mock_voice
    adapters.video = mock_video
    adapters.lipsync = AsyncMock()

    from core.workflow import _generate_shot_video
    await _generate_shot_video(shot, script, cast, adapters, tmp_path, "/tmp/img.png")

    vreq = captured_voice[0]
    assert isinstance(vreq, VoiceRequest)
    assert vreq.speaker_id == "zoe"  # [0] is zoe in this shot
