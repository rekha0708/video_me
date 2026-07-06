"""Tests for VlmAnalyzeVisualsAdapter (source-video setting extraction)."""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from adapters.analyze_visuals.vlm_adapter import VlmAnalyzeVisualsAdapter
from core.models.capabilities import (
    AnalyzeVisualsRequest,
    TranscriptSegment,
    VisualContext,
)


def _segments(n: int = 2) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text=f"line {i}", start=float(i * 5), end=float(i * 5 + 5))
        for i in range(n)
    ]


def _adapter(tmp_path: Path, **kwargs) -> VlmAnalyzeVisualsAdapter:
    return VlmAnalyzeVisualsAdapter(work_dir=tmp_path / "visuals", **kwargs)


def _make_video(tmp_path: Path) -> Path:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40)
    return path


# ------------------------------------------------------------------ skip paths


async def test_remote_uri_returns_empty(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri="https://example.com/v.mp4", segments=_segments())
    )
    assert result == VisualContext()


async def test_story_uri_returns_empty(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri="story://job1", segments=_segments())
    )
    assert result.is_empty


async def test_missing_file_returns_empty(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(tmp_path / "nope.mp4"), segments=_segments())
    )
    assert result.is_empty


async def test_no_segments_returns_empty(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(_make_video(tmp_path)), segments=[])
    )
    assert result.is_empty


# ------------------------------------------------------------------ happy path


async def test_run_parses_visual_context(tmp_path: Path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path)

    async def fake_sample(video_path, buckets):
        return [tmp_path / "seg_01.jpg", tmp_path / "seg_02.jpg"]

    async def fake_call(buckets, frames):
        return (
            '{"segments": ['
            '{"start": 0, "end": 5, "setting": "cozy kitchen", "props": ["apple"]},'
            '{"start": 5, "end": 10, "setting": "sunny garden", "props": []}'
            '], "summary": "kitchen then garden"}'
        )

    monkeypatch.setattr(adapter, "_sample_frames", fake_sample)
    monkeypatch.setattr(adapter, "_call_vlm", fake_call)

    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(video), segments=_segments(2))
    )
    assert len(result.segments) == 2
    assert result.segments[0].setting == "cozy kitchen"
    assert result.segments[0].props == ["apple"]
    # Time range comes from our known buckets, not the model's echo.
    assert result.segments[1].start == 5.0
    assert result.summary == "kitchen then garden"


async def test_run_uses_bucket_times_over_model(tmp_path: Path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path)
    monkeypatch.setattr(adapter, "_sample_frames", AsyncMock(return_value=[Path("a.jpg")]))
    # Model returns wrong times; adapter should override with the segment's range.
    monkeypatch.setattr(
        adapter, "_call_vlm",
        AsyncMock(return_value='{"segments":[{"start":99,"end":99,"setting":"bar"}]}'),
    )
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(video), segments=[TranscriptSegment(text="x", start=0, end=5)])
    )
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 5.0


# ------------------------------------------------------------------ failure = empty


async def test_frame_sampling_failure_returns_empty(tmp_path: Path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path)
    monkeypatch.setattr(adapter, "_sample_frames", AsyncMock(side_effect=RuntimeError("ffmpeg")))
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(video), segments=_segments())
    )
    assert result.is_empty


async def test_vlm_failure_returns_empty(tmp_path: Path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path)
    monkeypatch.setattr(adapter, "_sample_frames", AsyncMock(return_value=[Path("a.jpg")]))
    monkeypatch.setattr(adapter, "_call_vlm", AsyncMock(side_effect=RuntimeError("api down")))
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(video), segments=_segments())
    )
    assert result.is_empty


async def test_bad_json_repaired(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("json_repair")  # repair only runs where json_repair is installed
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path)
    monkeypatch.setattr(adapter, "_sample_frames", AsyncMock(return_value=[Path("a.jpg")]))
    # Trailing comma — json_repair territory.
    monkeypatch.setattr(
        adapter, "_call_vlm",
        AsyncMock(return_value='{"segments":[{"start":0,"end":5,"setting":"pub",}],}'),
    )
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(video), segments=[TranscriptSegment(text="x", start=0, end=5)])
    )
    assert result.segments[0].setting == "pub"


# ------------------------------------------------------------------ frame cap / bucketing


def test_bucket_segments_caps_at_max_frames(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, max_frames=4)
    segments = [
        TranscriptSegment(text=f"s{i}", start=float(i), end=float(i + 1)) for i in range(10)
    ]
    buckets = adapter._bucket_segments(segments)
    assert len(buckets) <= 4
    # Coverage preserved: first bucket starts at 0, last ends at 10.
    assert buckets[0][0] == 0.0
    assert buckets[-1][1] == 10.0


def test_bucket_segments_passthrough_when_under_cap(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, max_frames=16)
    segments = _segments(3)
    buckets = adapter._bucket_segments(segments)
    assert len(buckets) == 3


# ------------------------------------------------------------------ chart detection


async def test_chart_field_parsed(tmp_path: Path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path)
    monkeypatch.setattr(adapter, "_sample_frames", AsyncMock(return_value=[Path("a.jpg")]))
    monkeypatch.setattr(
        adapter, "_call_vlm",
        AsyncMock(return_value='{"segments":[{"start":0,"end":5,"setting":"studio",'
                              '"chart":"bar chart comparing planets"}]}'),
    )
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(video), segments=[TranscriptSegment(text="x", start=0, end=5)])
    )
    assert result.segments[0].chart == "bar chart comparing planets"


async def test_chart_field_defaults_empty(tmp_path: Path, monkeypatch) -> None:
    adapter = _adapter(tmp_path)
    video = _make_video(tmp_path)
    monkeypatch.setattr(adapter, "_sample_frames", AsyncMock(return_value=[Path("a.jpg")]))
    monkeypatch.setattr(
        adapter, "_call_vlm",
        AsyncMock(return_value='{"segments":[{"start":0,"end":5,"setting":"studio"}]}'),
    )
    result = await adapter.run(
        AnalyzeVisualsRequest(video_uri=str(video), segments=[TranscriptSegment(text="x", start=0, end=5)])
    )
    assert result.segments[0].chart == ""
