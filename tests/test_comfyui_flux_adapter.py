"""Unit tests for ComfyUIFluxAdapter's VRAM lifecycle (ComfyUIUnloadMixin).

The full unload()/error-path matrix is already covered via LtxAdapter in
tests/test_ltx_render.py — both adapters share the identical mixin
implementation. This file just confirms the mixin is correctly wired onto
this class too.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from adapters.render_character.comfyui_flux_adapter import ComfyUIFluxAdapter


def _adapter(tmp_path: Path) -> ComfyUIFluxAdapter:
    return ComfyUIFluxAdapter(work_dir=tmp_path / "renders", lora_dir=tmp_path / "loras")


def test_comfyui_flux_adapter_is_vram_managed() -> None:
    assert ComfyUIFluxAdapter.managed_vram is True


async def test_load_and_wait_are_noops(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    await adapter.load()
    await adapter.wait_until_loaded(timeout_sec=30)


async def test_unload_posts_to_free_endpoint(tmp_path: Path) -> None:
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_post_resp)

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=mock_client)

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await _adapter(tmp_path).unload()

    assert result is True
    assert "/free" in mock_client.post.call_args.args[0]
