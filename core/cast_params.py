"""Per-cast asset + render params, loaded from config/casts/<cast>/params.py.

Single source of truth for a cast's *trained assets* (LoRA safetensor + voice)
and per-member render tuning (weight / steps / guidance / trigger). The cast
YAML (config/casts/<cast>.yaml) still owns character design + personality; this
owns *how* the cast is rendered and voiced.

Optional and back-compatible: a cast with no params.py yields an empty map, and
the render/voice adapters fall back to the YAML `lora_ref` / `voice_profile_ref`
and the global render defaults exactly as before.

Example — config/casts/kids_duo/params.py:

    MEMBERS = {
        "max": {
            "lora_file": "kids_duo_max.safetensors",  # under settings.lora_dir
            "lora_weight": 0.9,
            "steps": 20,
            "guidance_scale": 3.5,
            "trigger": "",                            # optional prompt token(s)
            "voice_file": "voices/kids_duo/max",      # voice_profile_ref form
        },
        ...
    }
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CastMemberParams(BaseModel):
    lora_file: str = ""                    # filename under settings.lora_dir
    lora_weight: float | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    trigger: str = ""                      # optional prompt trigger token(s)
    voice_file: str = ""                   # voice_profile_ref-form override


# Loaded modules are cached per (cast_id, casts_dir) so a job doesn't re-exec
# the params module once per shot. Cleared in tests via _CACHE.clear().
_CACHE: dict[tuple[str, str], dict[str, CastMemberParams]] = {}


def load_cast_params(
    cast_id: str, casts_dir: str | Path = "config/casts"
) -> dict[str, CastMemberParams]:
    """Return {member_id: CastMemberParams} from config/casts/<cast_id>/params.py.

    Empty dict when the cast has no params.py (back-compat).
    """
    key = (cast_id, str(casts_dir))
    if key in _CACHE:
        return _CACHE[key]

    path = Path(casts_dir) / cast_id / "params.py"
    if not path.is_file():
        _CACHE[key] = {}
        return {}

    spec = importlib.util.spec_from_file_location(f"cast_params_{cast_id}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    raw: dict[str, Any] = getattr(module, "MEMBERS", {}) or {}
    result = {mid: CastMemberParams(**(vals or {})) for mid, vals in raw.items()}
    _CACHE[key] = result
    return result
