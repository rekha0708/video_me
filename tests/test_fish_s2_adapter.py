import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.synthesize_voice.fish_s2_adapter import FishS2TtsAdapter


def _mock_httpx(
    *,
    get_error: Exception | None = None,
    post_error: Exception | None = None,
):
    mock_get_resp = MagicMock()
    mock_get_resp.raise_for_status = MagicMock()

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.raise_for_status = MagicMock()

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


def _adapter(tmp_path: Path) -> FishS2TtsAdapter:
    return FishS2TtsAdapter(work_dir=tmp_path / "audio", voice_dir=tmp_path / "voices")


# ------------------------------------------------------------------ VRAM lifecycle

def test_fish_s2_adapter_is_vram_managed() -> None:
    assert FishS2TtsAdapter.managed_vram is True


async def test_load_posts_to_load_endpoint(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await _adapter(tmp_path).load()
    assert "/load" in mock_client.post.call_args.args[0]


async def test_unload_posts_to_unload_endpoint(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await _adapter(tmp_path).unload()
    assert result is True
    assert "/unload" in mock_client.post.call_args.args[0]


async def test_unload_returns_false_when_service_down(tmp_path: Path) -> None:
    """Server unreachable ⇒ nothing resident ⇒ safe to continue."""
    class _FakeConnectError(Exception):
        pass

    fake_httpx, _ = _mock_httpx(post_error=_FakeConnectError("refused"))
    fake_httpx.ConnectError = _FakeConnectError
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await _adapter(tmp_path).unload()
    assert result is False


async def test_unload_raises_on_http_error(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    fake_httpx.ConnectError = ConnectionError
    mock_client.post.return_value.status_code = 409
    mock_client.post.return_value.text = "model load in progress"
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(RuntimeError, match="refused to unload"):
            await _adapter(tmp_path).unload()


async def test_wait_until_loaded_returns_when_ready(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    responses = [
        {"model_loaded": False, "loading": True, "error": None},
        {"model_loaded": True, "loading": False, "error": None},
    ]
    mock_client.get.return_value.json = MagicMock(side_effect=responses)
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await _adapter(tmp_path).wait_until_loaded(timeout_sec=30, poll_sec=0)


async def test_wait_until_loaded_raises_on_model_error(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    mock_client.get.return_value.json = MagicMock(
        return_value={"model_loaded": False, "loading": False, "error": "OOM during load"}
    )
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(RuntimeError, match="OOM during load"):
            await _adapter(tmp_path).wait_until_loaded(timeout_sec=30, poll_sec=0)


async def test_wait_until_loaded_times_out(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    mock_client.get.return_value.json = MagicMock(
        return_value={"model_loaded": False, "loading": True, "error": None}
    )
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(TimeoutError):
            await _adapter(tmp_path).wait_until_loaded(timeout_sec=0, poll_sec=0)


async def test_health_ok_when_model_not_loaded(tmp_path: Path) -> None:
    """Server reachable but model unloaded is still healthy — the sequencer loads it later."""
    fake_httpx, mock_client = _mock_httpx()
    mock_client.get.return_value.json = MagicMock(
        return_value={"status": "ok", "model_loaded": False, "loading": False, "error": None}
    )
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await _adapter(tmp_path).health()
    assert health.status == "ok"
