import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.generate_video.wan_adapter import (
    WanAdapter,
    _DEFAULT_FPS,
    _PROMPT_PREFIX,
    _PROMPT_SUFFIX,
)
from core.models.capabilities import VideoClip, VideoRequest


# ------------------------------------------------------------------ helpers

_FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40  # minimal fake MP4 header


def _make_png(tmp_path: Path, name: str = "render_00.png") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    return path


def _request(tmp_path: Path, **kwargs) -> VideoRequest:
    image = kwargs.get("image_uri")
    if image is None:
        image = str(_make_png(tmp_path))
    return VideoRequest(
        image_uri=image,
        action=kwargs.get("action", "speaks warmly to the audience"),
        duration_sec=kwargs.get("duration_sec", 3.0),
        shot_id=kwargs.get("shot_id", "s01"),
    )


def _adapter(tmp_path: Path, **kwargs) -> WanAdapter:
    return WanAdapter(work_dir=tmp_path / "clips", **kwargs)


def _mock_httpx(
    mp4_bytes: bytes = _FAKE_MP4,
    *,
    get_error: Exception | None = None,
    post_error: Exception | None = None,
):
    mock_get_resp = MagicMock()
    mock_get_resp.raise_for_status = MagicMock()

    mock_post_resp = MagicMock()
    mock_post_resp.raise_for_status = MagicMock()
    mock_post_resp.content = mp4_bytes

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(
        side_effect=get_error if get_error else None,
        return_value=mock_get_resp,
    )
    mock_client.post = AsyncMock(
        side_effect=post_error if post_error else None,
        return_value=mock_post_resp,
    )

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return fake_httpx, mock_client


# ------------------------------------------------------------------ _build_prompt

def test_build_prompt_wraps_action(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt("dances with joy")
    assert "dances with joy" in prompt
    assert _PROMPT_PREFIX in prompt
    assert _PROMPT_SUFFIX in prompt


def test_build_prompt_prefix_before_action(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt("spins around")
    assert prompt.index(_PROMPT_PREFIX) < prompt.index("spins around")


def test_build_prompt_suffix_after_action(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt("waves goodbye")
    assert prompt.index("waves goodbye") < prompt.index(_PROMPT_SUFFIX)


# ------------------------------------------------------------------ _save_clip

def test_save_clip_writes_mp4(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = adapter._save_clip(_FAKE_MP4, out_dir)
    assert path.exists()
    assert path.name == "clip.mp4"
    assert path.read_bytes() == _FAKE_MP4


def test_save_clip_always_named_clip(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = adapter._save_clip(b"bytes", out_dir)
    assert path.name == "clip.mp4"


# ------------------------------------------------------------------ health

async def test_health_ok_when_service_reachable(tmp_path: Path) -> None:
    fake_httpx, _ = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await _adapter(tmp_path).health()
    assert health.status == "ok"


async def test_health_down_when_package_missing(tmp_path: Path) -> None:
    with patch.dict(sys.modules, {"httpx": None}):
        health = await _adapter(tmp_path).health()
    assert health.status == "down"
    assert "httpx" in (health.reason or "")


async def test_health_down_when_service_unreachable(tmp_path: Path) -> None:
    fake_httpx, _ = _mock_httpx(get_error=ConnectionError("refused"))
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await _adapter(tmp_path).health()
    assert health.status == "down"
    assert "unreachable" in (health.reason or "").lower()


# ------------------------------------------------------------------ run

async def test_run_returns_video_clip(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await adapter.run(req)

    assert isinstance(clip, VideoClip)
    assert clip.shot_id == "s01"
    assert clip.duration_sec == 3.0
    assert Path(clip.uri).exists()


async def test_run_creates_shot_subdir(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(_request(tmp_path, shot_id="s03"))

    assert (tmp_path / "clips" / "s03").is_dir()


async def test_run_clip_uri_is_under_shot_dir(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await adapter.run(_request(tmp_path, shot_id="s07"))

    assert "s07" in clip.uri
    assert clip.uri.endswith("clip.mp4")


async def test_run_raises_when_image_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, image_uri=str(tmp_path / "nonexistent.png"))
    with pytest.raises(FileNotFoundError, match="render_character"):
        await adapter.run(req)


async def test_run_posts_to_generate_endpoint(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(req)

    mock_client.post.assert_called_once()
    assert "/generate" in mock_client.post.call_args.args[0]


async def test_run_sends_fps_in_payload(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, fps=24)
    req = _request(tmp_path)
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(req)

    data = mock_client.post.call_args.kwargs.get("data") or {}
    assert data.get("fps") == "24"


async def test_run_sends_duration_in_payload(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, duration_sec=4.5)
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(req)

    data = mock_client.post.call_args.kwargs.get("data") or {}
    assert float(data.get("duration_sec", 0)) == 4.5


async def test_run_preserves_duration_sec_in_result(tmp_path: Path) -> None:
    """Adapter trusts req.duration_sec rather than parsing the MP4 header."""
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, duration_sec=2.5)
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await adapter.run(req)

    assert clip.duration_sec == 2.5


async def test_run_propagates_api_error(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    fake_httpx, _ = _mock_httpx(post_error=RuntimeError("Wan GPU OOM"))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(RuntimeError, match="Wan GPU OOM"):
            await adapter.run(req)


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero(tmp_path: Path) -> None:
    cost = await _adapter(tmp_path).estimate_cost(_request(tmp_path))
    assert cost.amount == 0.0
