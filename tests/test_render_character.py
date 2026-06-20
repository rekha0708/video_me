import base64
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.render_character.diffusion_adapter import (
    DiffusionRenderAdapter,
    _LORA_EXTENSIONS,
)
from core.models.capabilities import RenderCharacterRequest
from core.models.profile import CastMember


# ------------------------------------------------------------------ fixtures

def _member() -> CastMember:
    return CastMember(
        id="max",
        name="Max",
        gender="boy",
        visual_descriptor="soft cartoon 5-year-old boy with blue and white striped t-shirt",
        lora_ref="loras/kids_duo/max",
        voice_profile_ref="voices/kids_duo/max",
        personality="enthusiastic big-kid teacher who loves letters",
        signature_expressions=["wide-eyed teaching face", "proud big-kid grin"],
    )


def _request(**kwargs) -> RenderCharacterRequest:
    return RenderCharacterRequest(
        member=kwargs.get("member", _member()),
        setting=kwargs.get("setting", "sunny meadow"),
        expression=kwargs.get("expression", "wide-eyed wonder"),
    )


def _adapter(tmp_path: Path, **kwargs) -> DiffusionRenderAdapter:
    return DiffusionRenderAdapter(
        work_dir=tmp_path / "renders",
        lora_dir=kwargs.get("lora_dir", tmp_path / "loras"),
        **{k: v for k, v in kwargs.items() if k != "lora_dir"},
    )


def _fake_b64_png() -> str:
    # Minimal 1×1 PNG so base64.b64decode succeeds.
    tiny_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(tiny_png).decode()


def _mock_httpx(b64_image: str | None = None, *, get_error: Exception | None = None,
                post_error: Exception | None = None):
    """Return a fake httpx module whose AsyncClient is a working async context manager."""
    mock_get_resp = MagicMock()
    mock_get_resp.raise_for_status = MagicMock()

    mock_post_resp = MagicMock()
    mock_post_resp.raise_for_status = MagicMock()
    mock_post_resp.json.return_value = {"images": [b64_image or _fake_b64_png()]}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    if get_error:
        mock_client.get = AsyncMock(side_effect=get_error)
    else:
        mock_client.get = AsyncMock(return_value=mock_get_resp)

    if post_error:
        mock_client.post = AsyncMock(side_effect=post_error)
    else:
        mock_client.post = AsyncMock(return_value=mock_post_resp)

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return fake_httpx, mock_client


# ------------------------------------------------------------------ lora_name

def test_lora_name_strips_loras_prefix(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter.lora_name("loras/kids_duo/max") == "kids_duo_max"


def test_lora_name_no_prefix(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter.lora_name("kids_duo/max") == "kids_duo_max"


def test_lora_name_single_segment(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter.lora_name("loras/mycast") == "mycast"


# ------------------------------------------------------------------ _check_lora

def test_check_lora_finds_safetensors(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    lora_file = lora_dir / "kids_duo_max.safetensors"
    lora_file.write_bytes(b"fake lora")

    adapter = _adapter(tmp_path, lora_dir=lora_dir)
    path = adapter._check_lora(_member())
    assert path == lora_file


def test_check_lora_finds_pt_extension(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.pt").write_bytes(b"fake")

    adapter = _adapter(tmp_path, lora_dir=lora_dir)
    path = adapter._check_lora(_member())
    assert path.suffix == ".pt"


def test_check_lora_raises_with_track_b_message_when_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)  # lora_dir is empty tmp dir
    with pytest.raises(RuntimeError) as exc_info:
        adapter._check_lora(_member())
    msg = str(exc_info.value)
    assert "Track B" in msg
    assert "Max" in msg
    assert "max" in msg
    assert "kids_duo_max.safetensors" in msg


# ------------------------------------------------------------------ _build_prompt

def test_build_prompt_contains_lora_tag(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_request())
    assert "<lora:kids_duo_max:" in prompt


def test_build_prompt_contains_visual_descriptor(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_request())
    assert "striped t-shirt" in prompt


def test_build_prompt_contains_setting(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_request())
    assert "sunny meadow" in prompt


def test_build_prompt_contains_expression_when_set(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_request(expression="big grin"))
    assert "big grin" in prompt


def test_build_prompt_omits_expression_when_none(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_request(expression=None))
    assert "big grin" not in prompt
    assert "wide-eyed wonder" not in prompt


def test_build_prompt_uses_configured_lora_weight(tmp_path: Path) -> None:
    adapter = DiffusionRenderAdapter(
        work_dir=tmp_path / "r",
        lora_dir=tmp_path / "l",
        lora_weight=0.75,
    )
    prompt = adapter._build_prompt(_request())
    assert "0.75" in prompt


# ------------------------------------------------------------------ _save_images

def test_save_images_writes_png_files(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    b64 = _fake_b64_png()

    uris = adapter._save_images([b64, b64], out_dir)

    assert len(uris) == 2
    for uri in uris:
        assert Path(uri).exists()
        assert Path(uri).suffix == ".png"


def test_save_images_sequential_names(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    uris = adapter._save_images([_fake_b64_png(), _fake_b64_png()], out_dir)
    assert Path(uris[0]).name == "render_00.png"
    assert Path(uris[1]).name == "render_01.png"


# ------------------------------------------------------------------ health

async def test_health_ok_when_service_reachable(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    fake_httpx, _ = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await adapter.health()
    assert health.status == "ok"


async def test_health_down_when_package_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with patch.dict(sys.modules, {"httpx": None}):
        health = await adapter.health()
    assert health.status == "down"
    assert "httpx" in (health.reason or "")


async def test_health_down_when_service_unreachable(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    fake_httpx, _ = _mock_httpx(get_error=ConnectionError("refused"))
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await adapter.health()
    assert health.status == "down"
    assert "unreachable" in (health.reason or "").lower()


# ------------------------------------------------------------------ run

async def test_run_returns_image_set(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.safetensors").write_bytes(b"fake")

    adapter = _adapter(tmp_path, lora_dir=lora_dir)
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await adapter.run(_request())

    assert result.member_id == "max"
    assert len(result.images) == 1
    assert Path(result.images[0]).exists()


async def test_run_creates_member_subdir(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.safetensors").write_bytes(b"fake")

    adapter = _adapter(tmp_path, lora_dir=lora_dir)
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await adapter.run(_request())

    assert (tmp_path / "renders" / "max").is_dir()


async def test_run_raises_when_lora_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)  # empty lora_dir
    with pytest.raises(RuntimeError, match="Track B"):
        await adapter.run(_request())


async def test_run_posts_to_correct_endpoint(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.safetensors").write_bytes(b"fake")

    adapter = _adapter(tmp_path, lora_dir=lora_dir)
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(_request())

    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert "/sdapi/v1/txt2img" in call_args.args[0]


async def test_run_propagates_api_error(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.safetensors").write_bytes(b"fake")

    adapter = _adapter(tmp_path, lora_dir=lora_dir)
    fake_httpx, _ = _mock_httpx(post_error=RuntimeError("SD server error"))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(RuntimeError, match="SD server error"):
            await adapter.run(_request())


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero(tmp_path: Path) -> None:
    cost = await _adapter(tmp_path).estimate_cost(_request())
    assert cost.amount == 0.0
