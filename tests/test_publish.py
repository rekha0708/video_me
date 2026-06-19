import json
from pathlib import Path

import pytest

from adapters.publish.manual_adapter import ManualPublishAdapter, _REQUIRED_SIDECAR_FIELDS
from core.models.capabilities import FinalVideo, PublishRequest, PublishResult


# ------------------------------------------------------------------ helpers

def _make_video(tmp_path: Path, duration: float = 10.5) -> FinalVideo:
    path = tmp_path / "final.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40)
    return FinalVideo(uri=str(path), duration_sec=duration)


def _request(tmp_path: Path, **kwargs) -> PublishRequest:
    return PublishRequest(
        video=kwargs.get("video") or _make_video(tmp_path),
        rights_cleared=kwargs.get("rights_cleared", True),
        made_for_kids=kwargs.get("made_for_kids", True),
        disclosure_label_required=kwargs.get("disclosure_label_required", True),
        learning_objective_summary=kwargs.get(
            "learning_objective_summary",
            "Children learn to count to five.",
        ),
    )


def _adapter(tmp_path: Path) -> ManualPublishAdapter:
    return ManualPublishAdapter(review_dir=tmp_path / "review")


# ------------------------------------------------------------------ _build_metadata

def test_build_metadata_contains_all_required_fields(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    video_dest = tmp_path / "video.mp4"
    metadata = adapter._build_metadata(req, video_dest)
    for field in _REQUIRED_SIDECAR_FIELDS:
        assert field in metadata, f"missing required sidecar field: {field}"


def test_build_metadata_reflects_request_flags(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, made_for_kids=True, disclosure_label_required=False)
    metadata = adapter._build_metadata(req, tmp_path / "video.mp4")
    assert metadata["made_for_kids"] is True
    assert metadata["disclosure_label_required"] is False


def test_build_metadata_includes_learning_objective(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, learning_objective_summary="Count to ten.")
    metadata = adapter._build_metadata(req, tmp_path / "video.mp4")
    assert metadata["learning_objective_summary"] == "Count to ten."


def test_build_metadata_includes_duration(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path, duration=8.75)
    req = _request(tmp_path, video=video)
    metadata = adapter._build_metadata(req, tmp_path / "video.mp4")
    assert metadata["source_video_duration_sec"] == 8.75


def test_build_metadata_status_is_pending_review(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    metadata = adapter._build_metadata(_request(tmp_path), tmp_path / "video.mp4")
    assert metadata["status"] == "pending_review"


# ------------------------------------------------------------------ _write_sidecar

def test_write_sidecar_creates_metadata_json(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    path = adapter._write_sidecar({"key": "value"}, out_dir)
    assert path.name == "metadata.json"
    assert path.exists()


def test_write_sidecar_valid_json(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    path = adapter._write_sidecar({"status": "pending_review", "made_for_kids": True}, out_dir)
    data = json.loads(path.read_text())
    assert data["status"] == "pending_review"
    assert data["made_for_kids"] is True


# ------------------------------------------------------------------ _copy_video

def test_copy_video_copies_to_review_dir(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path)
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    dest = adapter._copy_video(video.uri, out_dir)
    assert dest.exists()
    assert dest.name == "video.mp4"
    assert dest.read_bytes() == Path(video.uri).read_bytes()


def test_copy_video_raises_when_source_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        adapter._copy_video(str(tmp_path / "ghost.mp4"), out_dir)
    assert "assemble_video" in str(exc_info.value)


# ------------------------------------------------------------------ _make_output_dir

def test_make_output_dir_is_under_review_dir(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = adapter._make_output_dir(_request(tmp_path))
    assert str(out_dir).startswith(str(tmp_path / "review"))


def test_make_output_dir_contains_video_stem(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = adapter._make_output_dir(_request(tmp_path))
    assert "final" in out_dir.name


def test_make_output_dir_contains_timestamp(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = adapter._make_output_dir(_request(tmp_path))
    # Timestamps contain digits and 'T'
    assert any(c.isdigit() for c in out_dir.name)
    assert "T" in out_dir.name


# ------------------------------------------------------------------ health

async def test_health_ok_when_review_dir_writable(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    health = await adapter.health()
    assert health.status == "ok"


async def test_health_ok_creates_review_dir(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    assert not (tmp_path / "review").exists()
    await adapter.health()
    assert (tmp_path / "review").is_dir()


async def test_health_down_when_dir_not_creatable(tmp_path: Path) -> None:
    # Point at a path whose parent is a file (can't mkdir inside a file)
    fake_parent = tmp_path / "not_a_dir.txt"
    fake_parent.write_bytes(b"blocked")
    adapter = ManualPublishAdapter(review_dir=fake_parent / "review")
    health = await adapter.health()
    assert health.status == "down"
    assert "review_dir" in (health.reason or "")


# ------------------------------------------------------------------ run

async def test_run_returns_publish_result(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(_request(tmp_path))
    assert isinstance(result, PublishResult)
    assert result.status == "pending_review"


async def test_run_video_is_copied_to_review_dir(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    await adapter.run(_request(tmp_path))
    review_videos = list((tmp_path / "review").rglob("video.mp4"))
    assert len(review_videos) == 1


async def test_run_metadata_sidecar_written(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(_request(tmp_path))
    assert Path(result.metadata_path).exists()
    assert Path(result.metadata_path).name == "metadata.json"


async def test_run_sidecar_has_all_required_fields(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(_request(tmp_path))
    data = json.loads(Path(result.metadata_path).read_text())
    for field in _REQUIRED_SIDECAR_FIELDS:
        assert field in data, f"sidecar missing: {field}"


async def test_run_review_path_points_to_copied_mp4(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(_request(tmp_path))
    assert Path(result.review_path).exists()
    assert result.review_path.endswith("video.mp4")


async def test_run_raises_when_rights_not_cleared(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, rights_cleared=False)
    with pytest.raises(RuntimeError, match="rights_cleared is False"):
        await adapter.run(req)


async def test_run_error_message_mentions_executor(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, rights_cleared=False)
    with pytest.raises(RuntimeError) as exc_info:
        await adapter.run(req)
    assert "executor" in str(exc_info.value).lower()


async def test_run_raises_when_video_file_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    ghost = FinalVideo(uri=str(tmp_path / "ghost.mp4"), duration_sec=5.0)
    req = _request(tmp_path, video=ghost)
    with pytest.raises(FileNotFoundError, match="assemble_video"):
        await adapter.run(req)


async def test_run_sidecar_made_for_kids_matches_request(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(_request(tmp_path, made_for_kids=True))
    data = json.loads(Path(result.metadata_path).read_text())
    assert data["made_for_kids"] is True


async def test_run_sidecar_disclosure_flag_matches_request(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(_request(tmp_path, disclosure_label_required=False))
    data = json.loads(Path(result.metadata_path).read_text())
    assert data["disclosure_label_required"] is False


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero(tmp_path: Path) -> None:
    cost = await _adapter(tmp_path).estimate_cost(_request(tmp_path))
    assert cost.amount == 0.0
