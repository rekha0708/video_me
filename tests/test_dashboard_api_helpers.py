"""Unit tests for the module-level job-page helpers in services/dashboard_api.py."""
import importlib

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from core.storage import LocalArtifactStore
from services.dashboard_api import (
    _artifact_flags,
    _build_cost_summary,
    _format_dashboard_time,
    _lora_dataset_image_dir,
    _next_training_image_path,
    _stepper_state,
)
from adapters.approval.dashboard_image_approval_adapter import _build_request_payload

_has_fastapi = importlib.util.find_spec("fastapi") is not None


# ------------------------------------------------------------------ fixtures


def _store(tmp_path: Path, *stages: str) -> LocalArtifactStore:
    store = LocalArtifactStore(tmp_path / "artifacts")
    for stage in stages:
        store.put_json("job1", stage, {"ok": True})
    return store


def _work_dir(tmp_path: Path, *subpaths: str) -> Path:
    work_dir = tmp_path / "jobs" / "job1"
    for sub in subpaths:
        target = work_dir / sub
        if sub.endswith(".mp4"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
        else:
            target.mkdir(parents=True, exist_ok=True)
    return work_dir


def _job(phase: str = "all", status: str = "running", current_stage: str | None = None,
         completed_phases: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        phase=phase,
        status=status,
        current_stage=current_stage,
        completed_phases=completed_phases or [],
    )


_BASE_TIME = datetime(2026, 7, 11, 19, 0, 0, tzinfo=timezone.utc)


def _ev(
    event_id: int,
    event_type: str,
    stage_name: str | None,
    seconds: float,
    *,
    shot_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        event_id=event_id,
        event_type=event_type,
        stage_name=stage_name,
        shot_id=shot_id,
        created_at=_BASE_TIME + timedelta(seconds=seconds),
    )


_ALL_FLAGS = {"transcript": True, "visuals": True, "script": True, "renders": True, "video": True}
_NO_FLAGS = {
    "transcript": False,
    "visuals": False,
    "script": False,
    "renders": False,
    "shot_attempts": False,
    "video": False,
}


# ------------------------------------------------------------------ display time + cost summary


def test_dashboard_time_formats_in_pacific_time() -> None:
    dt = datetime(2026, 7, 11, 19, 5, 6, tzinfo=timezone.utc)

    assert _format_dashboard_time(dt) == "12:05:06"
    assert _format_dashboard_time(dt, "%m-%d %H:%M") == "07-11 12:05"


def test_cost_summary_disabled_without_price() -> None:
    events = [
        _ev(1, "stage_started", "transcribe", 0),
        _ev(2, "stage_completed", "transcribe", 30),
    ]

    summary = _build_cost_summary(events, {"gpu_price_per_hour": 0})

    assert summary["enabled"] is False
    assert summary["groups"] == []


def test_cost_summary_groups_stage_pairs_and_averages_per_shot() -> None:
    events = [
        _ev(1, "stage_started", "transcribe", 0),
        _ev(2, "stage_completed", "transcribe", 60),
        _ev(3, "stage_started", "render_character", 60, shot_id="s1"),
        _ev(4, "stage_completed", "render_character", 180, shot_id="s1"),
        _ev(5, "stage_started", "render_character", 180, shot_id="s2"),
        _ev(6, "stage_completed", "render_character", 300, shot_id="s2"),
        _ev(7, "stage_started", "generate_video", 300, shot_id="s1"),
        _ev(8, "stage_completed", "generate_video", 480, shot_id="s1"),
        _ev(9, "stage_started", "lip_sync", 480, shot_id="s1"),
        _ev(10, "stage_completed", "lip_sync", 540, shot_id="s1"),
        _ev(11, "stage_started", "generate_video", 540, shot_id="s2"),
        _ev(12, "stage_completed", "generate_video", 720, shot_id="s2"),
        _ev(13, "stage_started", "assemble_video", 720),
        _ev(14, "stage_completed", "assemble_video", 780),
    ]

    summary = _build_cost_summary(events, {"gpu_price_per_hour": 60})

    assert summary["enabled"] is True
    assert summary["total_seconds"] == 780
    assert summary["total_cost_display"] == "$13.00"
    assert summary["shot_count"] == 2
    groups = {group["key"]: group for group in summary["groups"]}
    assert groups["prep"]["cost_display"] == "$1.00"
    assert groups["render"]["seconds"] == 240
    assert groups["render"]["avg_cost_per_shot_display"] == "$2.00"
    assert groups["video"]["seconds"] == 420
    assert groups["video"]["avg_cost_per_shot_display"] == "$3.50"
    assert groups["assemble"]["cost_display"] == "$1.00"


def test_cost_summary_does_not_bill_approval_wait_gaps() -> None:
    events = [
        _ev(1, "stage_started", "transcribe", 0),
        _ev(2, "stage_completed", "transcribe", 10),
        _ev(3, "approval_requested", None, 10),
        _ev(4, "approval_granted", None, 1000),
        _ev(5, "stage_started", "plan_shots", 1000),
        _ev(6, "stage_completed", "plan_shots", 1010),
    ]

    summary = _build_cost_summary(events, {"gpu_price_per_hour": 360})

    assert summary["total_seconds"] == 20
    assert summary["total_cost_display"] == "$2.00"


def test_cost_summary_counts_failed_stage_time() -> None:
    events = [
        _ev(1, "stage_started", "generate_video", 0, shot_id="s1"),
        _ev(2, "stage_failed", "generate_video", 30, shot_id="s1"),
    ]

    summary = _build_cost_summary(events, {"gpu_price_per_hour": 120})

    assert summary["total_seconds"] == 30
    assert summary["total_cost_display"] == "$1.00"
    assert summary["groups"][0]["key"] == "video"


# ------------------------------------------------------------------ LocalArtifactStore.has


def test_has_true_when_artifact_exists(tmp_path: Path) -> None:
    store = _store(tmp_path, "transcribe")
    assert store.has("job1", "transcribe") is True


def test_has_false_when_artifact_missing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.has("job1", "transcribe") is False


# ------------------------------------------------------------------ _artifact_flags


def test_flags_all_false_for_fresh_job(tmp_path: Path) -> None:
    flags = _artifact_flags(_store(tmp_path), _work_dir(tmp_path), "job1")
    assert flags == _NO_FLAGS


def test_transcript_flag_follows_artifact(tmp_path: Path) -> None:
    flags = _artifact_flags(_store(tmp_path, "transcribe"), _work_dir(tmp_path), "job1")
    assert flags["transcript"] is True
    assert flags["script"] is False


def test_script_flag_from_adapt_script_or_plan(tmp_path: Path) -> None:
    flags = _artifact_flags(_store(tmp_path, "adapt_script"), _work_dir(tmp_path), "job1")
    assert flags["script"] is True
    flags = _artifact_flags(_store(tmp_path, "plan_shots"), _work_dir(tmp_path), "job1")
    assert flags["script"] is True


def test_renders_flag_requires_plan_and_render_dir(tmp_path: Path) -> None:
    store = _store(tmp_path, "plan_shots")
    assert _artifact_flags(store, _work_dir(tmp_path), "job1")["renders"] is False
    assert _artifact_flags(store, _work_dir(tmp_path, "renders"), "job1")["renders"] is True


def test_renders_flag_accepts_user_images_dir(tmp_path: Path) -> None:
    """Story+images jobs have no renders/ — user_images/ counts instead."""
    store = _store(tmp_path, "plan_shots")
    flags = _artifact_flags(store, _work_dir(tmp_path, "user_images"), "job1")
    assert flags["renders"] is True


def test_video_flag_requires_final_mp4(tmp_path: Path) -> None:
    store = _store(tmp_path)
    flags = _artifact_flags(store, _work_dir(tmp_path, "assembled/final.mp4"), "job1")
    assert flags["video"] is True


def test_shot_attempts_flag_requires_attempt_dir(tmp_path: Path) -> None:
    store = _store(tmp_path)
    flags = _artifact_flags(store, _work_dir(tmp_path, "shot_attempts"), "job1")
    assert flags["shot_attempts"] is True


def test_visuals_flag_true_when_segments_present(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    store.put_json("job1", "analyze_visuals", {"segments": [{"start": 0, "end": 5, "setting": "kitchen"}], "summary": "x"})
    flags = _artifact_flags(store, _work_dir(tmp_path), "job1")
    assert flags["visuals"] is True


def test_visuals_flag_false_when_empty(tmp_path: Path) -> None:
    """Story jobs persist an empty analyze_visuals artifact — card stays hidden."""
    store = LocalArtifactStore(tmp_path / "artifacts")
    store.put_json("job1", "analyze_visuals", {"segments": [], "summary": ""})
    flags = _artifact_flags(store, _work_dir(tmp_path), "job1")
    assert flags["visuals"] is False


def test_visuals_flag_false_when_missing(tmp_path: Path) -> None:
    flags = _artifact_flags(_store(tmp_path), _work_dir(tmp_path), "job1")
    assert flags["visuals"] is False


# ------------------------------------------------------------------ _stepper_state


def test_stepper_passthrough_for_phased_job(tmp_path: Path) -> None:
    job = _job(phase="script_plan", completed_phases=["transcribe"])
    state = _stepper_state(job, _NO_FLAGS)
    assert state == {"phase": "script_plan", "completed": ["transcribe"]}


def test_stepper_all_completed_job_shows_everything_done() -> None:
    state = _stepper_state(_job(status="completed"), _ALL_FLAGS)
    assert state["completed"] == ["transcribe", "script_plan", "render", "assemble"]


def test_stepper_all_running_at_generate_video_is_render_phase() -> None:
    state = _stepper_state(_job(current_stage="generate_video"), _NO_FLAGS)
    assert state["phase"] == "render"
    assert state["completed"] == ["transcribe", "script_plan"]


def test_stepper_all_video_model_load_maps_to_render() -> None:
    """The synthetic gpu_sequencer stage counts as the render macro phase."""
    state = _stepper_state(_job(current_stage="video_model_load"), _NO_FLAGS)
    assert state["phase"] == "render"


def test_stepper_all_failed_at_adapt_script_is_script_plan_phase() -> None:
    state = _stepper_state(_job(status="failed", current_stage="adapt_script"), _NO_FLAGS)
    assert state["phase"] == "script_plan"
    assert state["completed"] == ["transcribe"]


def test_stepper_all_queued_job_infers_from_artifacts() -> None:
    """No current_stage yet — fall back to artifact existence."""
    state = _stepper_state(_job(current_stage=None), _NO_FLAGS)
    assert state["phase"] == "transcribe"
    flags = {**_NO_FLAGS, "transcript": True}
    state = _stepper_state(_job(current_stage=None), flags)
    assert state["phase"] == "script_plan"
    assert state["completed"] == ["transcribe"]


@pytest.mark.parametrize(
    ("current_stage", "phase", "completed"),
    [
        ("canonical_look", "canonical_look", ["animate_validate"]),
        (
            "video_model_load",
            "animate_generate",
            [
                "animate_validate",
                "canonical_look",
                "animate_preprocess",
                "animate_audio",
            ],
        ),
        (
            "animate_lipsync",
            "animate_finish",
            [
                "animate_validate",
                "canonical_look",
                "animate_preprocess",
                "animate_audio",
                "animate_generate",
            ],
        ),
    ],
)
def test_stepper_direct_animate_maps_internal_stages_to_dashboard_macros(
    current_stage: str,
    phase: str,
    completed: list[str],
) -> None:
    job = SimpleNamespace(
        phase="all",
        status="running",
        current_stage=current_stage,
        completed_phases=[],
        request={"workflow_kind": "wan_animate_direct"},
    )

    state = _stepper_state(job, _NO_FLAGS)

    assert state == {"phase": phase, "completed": completed}


def test_stepper_completed_direct_animate_marks_every_macro_complete() -> None:
    job = SimpleNamespace(
        phase="all",
        status="completed",
        current_stage="animate_export",
        completed_phases=["wan_animate_direct"],
        request={"workflow_kind": "wan_animate_direct"},
    )

    state = _stepper_state(job, _NO_FLAGS)

    assert state == {
        "phase": "animate_finish",
        "completed": [
            "animate_validate",
            "canonical_look",
            "animate_preprocess",
            "animate_audio",
            "animate_generate",
            "animate_finish",
        ],
    }


# ---------------------------------------------------------- LoRA training utils


def test_lora_dataset_image_dir_maps_workspace_path_to_checkout(tmp_path: Path) -> None:
    config_path = tmp_path / "kohya_config_meera.toml"
    config_path.write_text(
        """
[[dataset.general]]
resolution = 1024

[[dataset.general.subsets]]
image_dir = "/workspace/video_me/assets/lady_model/training/images/meera"
caption_extension = ".txt"
"""
    )

    image_dir = _lora_dataset_image_dir(config_path, cwd=tmp_path)

    assert image_dir == tmp_path / "assets/lady_model/training/images/meera"


def test_lora_dataset_image_dir_reads_musubi_dataset_config(tmp_path: Path) -> None:
    config_path = tmp_path / "musubi_dataset_meera.toml"
    config_path.write_text(
        """
[general]
caption_extension = ".txt"

[[datasets]]
image_directory = "/workspace/video_me/assets/lady_model/training/images/meera"
cache_directory = "/workspace/video_me/assets/lady_model/training/cache/meera"
"""
    )

    image_dir = _lora_dataset_image_dir(config_path, cwd=tmp_path)

    assert image_dir == tmp_path / "assets/lady_model/training/images/meera"


def test_next_training_image_path_uses_next_member_index(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "meera_001.png").write_bytes(b"")
    (image_dir / "meera_009.webp").write_bytes(b"")
    (image_dir / "other_100.png").write_bytes(b"")

    next_path = _next_training_image_path(image_dir, "Meera", ".png")

    assert next_path == image_dir / "meera_010.png"


# -------------------------------------------------------- story phase restriction


@pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")
def test_story_job_rejected_with_invalid_phase(tmp_path: Path) -> None:
    """Story-kind jobs must start at 'transcribe' or 'all', not script_plan/render/assemble."""
    from fastapi.testclient import TestClient
    from services.dashboard_api import create_app
    from core.config import AppConfig, Settings
    from core.models.profile import ChannelProfile, Cast, CastMember

    settings = Settings(
        data_dir=str(tmp_path / "data"),
        artifact_dir=str(tmp_path / "art"),
        sqlite_path=str(tmp_path / "test.db"),
    )
    cfg = AppConfig(
        settings=settings,
        channel_profile=ChannelProfile(
            id="test", name="test", aspect_ratio="9:16",
            genre_content="education", tone="friendly",
            format="animated_character", made_for_kids=True,
        ),
        cast=Cast(id="kids_duo", species="human", is_original_synthetic=True, members=[
            CastMember(id="max", name="Max", visual_descriptor="boy", lora_ref="loras/max",
                       voice_profile_ref="voices/max", personality="friendly"),
        ]),
    )
    app = create_app(config_loader=lambda: cfg)
    client = TestClient(app, raise_server_exceptions=False)

    for bad_phase in ("script_plan", "render", "assemble"):
        resp = client.post("/api/jobs", json={
            "source": {"kind": "story", "url": ""},
            "story_text": "Once upon a time...",
            "phase": bad_phase,
            "rights_cleared": True,
        })
        assert resp.status_code == 400, f"phase={bad_phase} should be rejected"
        assert "INVALID_PHASE_FOR_STORY" in resp.text

    for ok_phase in ("transcribe", "all"):
        resp = client.post("/api/jobs", json={
            "source": {"kind": "story", "url": ""},
            "story_text": "Once upon a time...",
            "phase": ok_phase,
            "rights_cleared": True,
        })
        assert resp.status_code == 200, f"phase={ok_phase} should be accepted"


# -------------------------------------------------- image approval origin field


def test_build_request_payload_includes_origin() -> None:
    """_build_request_payload passes critique.origin through to the shot data."""
    from core.models.capabilities import ImageApprovalRequest, ImageCritiqueResult

    critique_vlm = ImageCritiqueResult(
        winner_uri="/img/a.png", winner_index=0, candidate_uris=["/img/a.png"],
        overall_reasoning="good", candidate_scores=[], origin="vlm",
    )
    critique_user = ImageCritiqueResult(
        winner_uri="/img/b.png", winner_index=0, candidate_uris=["/img/b.png"],
        overall_reasoning="user ref", candidate_scores=[], origin="user",
    )
    shot_a = SimpleNamespace(shot_id="s01", setting="park", action="walks")
    shot_b = SimpleNamespace(shot_id="s02", setting="room", action="sits")

    req = ImageApprovalRequest(
        cast_id="kids_duo",
        shots=[shot_a, shot_b],
        critique_results=[critique_vlm, critique_user],
    )
    payload = _build_request_payload(req)
    assert payload["shots"][0]["origin"] == "vlm"
    assert payload["shots"][1]["origin"] == "user"


def test_build_request_payload_defaults_origin_to_vlm() -> None:
    """Old critique results without origin field default to 'vlm'."""
    critique = SimpleNamespace(
        winner_uri="/img/c.png", winner_index=0,
        candidate_uris=["/img/c.png"], overall_reasoning="ok",
        candidate_scores=[],
    )
    shot = SimpleNamespace(shot_id="s01", setting="park", action="walks")
    req = SimpleNamespace(cast_id="kids_duo", shots=[shot], critique_results=[critique])
    payload = _build_request_payload(req)
    assert payload["shots"][0]["origin"] == "vlm"


# -------------------------------------------------- upload + local-images endpoints


def _make_test_client(tmp_path: Path):
    from fastapi.testclient import TestClient
    from services.dashboard_api import create_app
    from core.config import AppConfig, Settings
    from core.models.profile import ChannelProfile, Cast, CastMember

    settings = Settings(
        data_dir=str(tmp_path / "data"),
        artifact_dir=str(tmp_path / "art"),
        sqlite_path=str(tmp_path / "test.db"),
    )
    cfg = AppConfig(
        settings=settings,
        channel_profile=ChannelProfile(
            id="test", name="test", aspect_ratio="9:16",
            genre_content="education", tone="friendly",
            format="animated_character", made_for_kids=True,
        ),
        cast=Cast(id="kids_duo", species="human", is_original_synthetic=True, members=[
            CastMember(
                id="max", name="Max", visual_descriptor="boy",
                lora_ref="loras/kids_duo/max", voice_profile_ref="voices/kids_duo/max",
                personality="curious",
            ),
            CastMember(
                id="zoe", name="Zoe", visual_descriptor="girl",
                lora_ref="loras/kids_duo/zoe", voice_profile_ref="voices/kids_duo/zoe",
                personality="creative",
            ),
        ]),
    )
    return TestClient(create_app(config_loader=lambda: cfg), raise_server_exceptions=False)


@pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")
def test_upload_character_image_valid(tmp_path: Path) -> None:
    client = _make_test_client(tmp_path)
    resp = client.post(
        "/api/uploads/character-image",
        data={"member_id": "max"},
        files={"file": ("max.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["member_id"] == "max"
    assert data["filename"] == "max.png"
    assert Path(data["path"]).exists()


@pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")
def test_upload_character_image_invalid_member(tmp_path: Path) -> None:
    client = _make_test_client(tmp_path)
    resp = client.post(
        "/api/uploads/character-image",
        data={"member_id": "unknown"},
        files={"file": ("img.png", b"\x89PNG" + b"\x00" * 50, "image/png")},
    )
    assert resp.status_code == 400
    assert "INVALID_MEMBER" in resp.text


@pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")
def test_upload_character_image_invalid_format(tmp_path: Path) -> None:
    client = _make_test_client(tmp_path)
    resp = client.post(
        "/api/uploads/character-image",
        data={"member_id": "max"},
        files={"file": ("max.gif", b"GIF89a" + b"\x00" * 50, "image/gif")},
    )
    assert resp.status_code == 400
    assert "INVALID_FORMAT" in resp.text


@pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")
def test_local_images_endpoint(tmp_path: Path) -> None:
    client = _make_test_client(tmp_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "test.png").write_bytes(b"\x89PNG" + b"\x00" * 50)
    (img_dir / "test.txt").write_text("not an image")

    resp = client.get(f"/api/local-images?dir={img_dir}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["images"]) == 1
    assert data["images"][0]["name"] == "test.png"
