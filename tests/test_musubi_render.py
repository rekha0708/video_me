"""Unit tests for MusubiFluxAdapter param-driven behavior (no subprocess)."""
from pathlib import Path

import pytest

from adapters.render_character.musubi_flux_adapter import MusubiFluxAdapter
from core.models.capabilities import RenderCharacterRequest
from core.models.profile import CastMember


def _member() -> CastMember:
    return CastMember(
        id="max", name="Max", gender="boy",
        visual_descriptor="cartoon boy in striped shirt",
        lora_ref="loras/kids_duo/max", voice_profile_ref="voices/kids_duo/max",
        personality="eager", signature_expressions=["grin"],
    )


def _req(tmp_path: Path, **kwargs) -> RenderCharacterRequest:
    return RenderCharacterRequest(
        member=kwargs.get("member", _member()),
        setting=kwargs.get("setting", "cozy kitchen"),
        shot_id=kwargs.get("shot_id", "s01"),
        camera=kwargs.get("camera", "close-up"),
        lora_file=kwargs.get("lora_file", ""),
        trigger=kwargs.get("trigger", ""),
    )


def _adapter(tmp_path: Path) -> MusubiFluxAdapter:
    return MusubiFluxAdapter(work_dir=tmp_path / "renders", lora_dir=tmp_path / "loras")


def test_build_prompt_includes_camera_setting_trigger(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_req(tmp_path, trigger="mxtok"), skip_lora=False)
    assert prompt.startswith("mxtok")               # trigger first
    assert "cozy kitchen" in prompt
    assert "close-up shot" in prompt


def test_build_prompt_without_trigger(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_req(tmp_path), skip_lora=False)
    assert prompt.startswith("cartoon boy")


def test_check_lora_prefers_params_lora_file(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.safetensors").write_bytes(b"weights")
    adapter = _adapter(tmp_path)
    path = adapter._check_lora(_req(tmp_path, lora_file="kids_duo_max.safetensors"))
    assert path.name == "kids_duo_max.safetensors"


def test_check_lora_params_file_missing_raises(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(RuntimeError, match="params.py"):
        adapter._check_lora(_req(tmp_path, lora_file="ghost.safetensors"))


def test_check_lora_falls_back_to_lora_ref(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.safetensors").write_bytes(b"weights")
    adapter = _adapter(tmp_path)
    # No lora_file on the request → derive from member.lora_ref.
    path = adapter._check_lora(_req(tmp_path, lora_file=""))
    assert path.name == "kids_duo_max.safetensors"


# ── Multi-character prompt tests ─────────────────────────────────────────────


def _other_member() -> CastMember:
    return CastMember(
        id="zoe", name="Zoe", gender="girl",
        visual_descriptor="cartoon girl with pink bow and yellow dress",
        lora_ref="loras/kids_duo/zoe", voice_profile_ref="voices/kids_duo/zoe",
        personality="playful", signature_expressions=["giggle"],
    )


def test_build_prompt_includes_other_member(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = RenderCharacterRequest(
        member=_member(),
        setting="cozy kitchen",
        shot_id="s01",
        camera="medium",
        other_members=[_other_member()],
    )
    prompt = adapter._build_prompt(req, skip_lora=False)
    assert "also present: cartoon girl with pink bow" in prompt
    assert "cartoon boy in striped shirt" in prompt


def test_build_prompt_no_other_members_unchanged(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req_with = RenderCharacterRequest(
        member=_member(), setting="park", shot_id="s01", camera="wide",
        other_members=[],
    )
    req_without = RenderCharacterRequest(
        member=_member(), setting="park", shot_id="s01", camera="wide",
    )
    assert adapter._build_prompt(req_with, skip_lora=False) == \
           adapter._build_prompt(req_without, skip_lora=False)
