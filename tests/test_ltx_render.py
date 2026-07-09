"""Unit tests for LtxAdapter prompt/workflow building (no ComfyUI calls)."""
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.generate_video.ltx_adapter import (
    LtxAdapter,
    _DEFAULT_STYLE_SUFFIX,
    _NEGATIVE_PROMPT,
    _STABILITY_SUFFIX,
    _WORKFLOW_TEMPLATE,
)


def _adapter(tmp_path: Path) -> LtxAdapter:
    return LtxAdapter(work_dir=tmp_path / "clips")


def _mock_httpx(*, post_error: Exception | None = None):
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(
        side_effect=post_error if post_error else None,
        return_value=mock_post_resp,
    )

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return fake_httpx, mock_client


# ------------------------------------------------------------------ _build_prompt

def test_build_prompt_defaults_to_cartoon_style_when_unset(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt("waves goodbye")
    assert prompt.startswith(_DEFAULT_STYLE_SUFFIX)


def test_build_prompt_uses_cast_style_suffix_when_set(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt("waves goodbye", style_suffix="photorealistic, cinematic lighting")
    assert prompt.startswith("photorealistic, cinematic lighting")
    assert _DEFAULT_STYLE_SUFFIX not in prompt


def test_build_prompt_includes_action_and_setting(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt("dances with joy", "cozy sunlit kitchen")
    assert "dances with joy" in prompt
    assert "cozy sunlit kitchen" in prompt
    assert prompt.index("dances with joy") < prompt.index("cozy sunlit kitchen")


def test_build_prompt_omits_empty_setting(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter._build_prompt("waves", "") == adapter._build_prompt("waves")


def test_build_prompt_includes_stability_suffix(tmp_path: Path) -> None:
    """Every prompt gets the anti-flicker/color-stability terms, regardless of cast style."""
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt("waves", style_suffix="photorealistic")
    assert _STABILITY_SUFFIX in prompt


# ------------------------------------------------------------------ negative prompt

def test_minimal_workflow_uses_stability_negative_prompt(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    workflow = adapter._minimal_ltx_workflow("img.png", "a prompt", 41, 123)
    negative_text = workflow["3"]["inputs"]["negative"]
    assert negative_text == _NEGATIVE_PROMPT
    assert "flicker" in negative_text
    assert "grey filter" in negative_text


def test_negative_prompt_marker_substituted_in_real_template() -> None:
    """The bundled ltx_i2v.json template's negative node gets the same
    stability negative prompt as the minimal fallback workflow."""
    adapter = LtxAdapter(work_dir=Path("/tmp/unused"))
    workflow = adapter._build_workflow(
        image_name="img.png", prompt_text="a prompt", num_frames=41, seed=1,
    )
    negative_node = next(
        n for n in workflow.values() if n.get("_meta", {}).get("title") == "__NEGATIVE__"
    )
    assert negative_node["inputs"]["text"] == _NEGATIVE_PROMPT


def test_workflow_template_negative_node_title_is_marker() -> None:
    """Guards against the JSON template's node title drifting back to a
    static (non-substituted) 'Negative Prompt' title."""
    workflow = json.loads(_WORKFLOW_TEMPLATE.read_text())
    titles = [n.get("_meta", {}).get("title") for n in workflow.values()]
    assert "__NEGATIVE__" in titles


# ------------------------------------------------------------------ ComfyUIUnloadMixin VRAM lifecycle

def test_ltx_adapter_is_vram_managed() -> None:
    assert LtxAdapter.managed_vram is True


async def test_load_is_a_noop(tmp_path: Path) -> None:
    """ComfyUI lazy-loads whatever the next /prompt needs — nothing to do here."""
    await _adapter(tmp_path).load()


async def test_wait_until_loaded_is_a_noop(tmp_path: Path) -> None:
    await _adapter(tmp_path).wait_until_loaded(timeout_sec=30)


async def test_unload_posts_to_free_endpoint(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await _adapter(tmp_path).unload()
    assert result is True
    assert "/free" in mock_client.post.call_args.args[0]
    assert mock_client.post.call_args.kwargs["json"] == {
        "unload_models": True,
        "free_memory": True,
    }


async def test_unload_returns_false_when_comfyui_unreachable(tmp_path: Path) -> None:
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
    mock_client.post.return_value.status_code = 500
    mock_client.post.return_value.text = "internal error"
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(RuntimeError, match="refused to free VRAM"):
            await _adapter(tmp_path).unload()
