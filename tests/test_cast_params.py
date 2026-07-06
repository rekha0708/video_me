"""Tests for per-cast params loading (config/casts/<cast>/params.py)."""
from pathlib import Path

from core.cast_params import (
    CastMemberParams,
    CastPairParams,
    _CACHE,
    _PAIR_CACHE,
    load_cast_pair_params,
    load_cast_params,
)


def _write_params(tmp_path: Path, cast_id: str, body: str) -> Path:
    cast_dir = tmp_path / cast_id
    cast_dir.mkdir(parents=True, exist_ok=True)
    (cast_dir / "params.py").write_text(body, encoding="utf-8")
    return tmp_path


def setup_function() -> None:
    _CACHE.clear()
    _PAIR_CACHE.clear()


def test_loads_real_kids_duo_params() -> None:
    params = load_cast_params("kids_duo")
    assert set(params) == {"max", "zoe"}
    assert params["max"].lora_file == "kids_duo_max.safetensors"
    assert params["max"].lora_weight == 0.9
    assert params["max"].voice_file == "voices/kids_duo/max"


def test_missing_cast_returns_empty(tmp_path: Path) -> None:
    assert load_cast_params("nope", casts_dir=tmp_path) == {}


def test_loads_custom_params(tmp_path: Path) -> None:
    _write_params(tmp_path, "foo", 'MEMBERS = {"a": {"lora_file": "a.safetensors", "steps": 12}}')
    params = load_cast_params("foo", casts_dir=tmp_path)
    assert isinstance(params["a"], CastMemberParams)
    assert params["a"].lora_file == "a.safetensors"
    assert params["a"].steps == 12
    assert params["a"].lora_weight is None  # unset → None (adapter default)


def test_cache_returns_same_object(tmp_path: Path) -> None:
    _write_params(tmp_path, "bar", 'MEMBERS = {"a": {"lora_file": "a.safetensors"}}')
    first = load_cast_params("bar", casts_dir=tmp_path)
    second = load_cast_params("bar", casts_dir=tmp_path)
    assert first is second


def test_empty_members_ok(tmp_path: Path) -> None:
    _write_params(tmp_path, "baz", "MEMBERS = {}")
    assert load_cast_params("baz", casts_dir=tmp_path) == {}


# ── Pair params ──────────────────────────────────────────────────────────────


def test_load_pair_params_from_kids_duo() -> None:
    pairs = load_cast_pair_params("kids_duo")
    key = frozenset({"max", "zoe"})
    assert key in pairs
    assert isinstance(pairs[key], CastPairParams)
    assert pairs[key].lora_file == ""
    assert pairs[key].lora_weight == 0.9


def test_load_pair_params_missing_returns_empty(tmp_path: Path) -> None:
    assert load_cast_pair_params("nope", casts_dir=tmp_path) == {}


def test_load_pair_params_no_pairs_dict(tmp_path: Path) -> None:
    _write_params(tmp_path, "nopairs", 'MEMBERS = {"a": {"lora_file": "a.safetensors"}}')
    assert load_cast_pair_params("nopairs", casts_dir=tmp_path) == {}


def test_load_pair_params_custom(tmp_path: Path) -> None:
    body = (
        'MEMBERS = {"x": {}, "y": {}}\n'
        'PAIRS = {frozenset({"x", "y"}): {"lora_file": "xy.safetensors", "trigger": "xytok"}}\n'
    )
    _write_params(tmp_path, "duo", body)
    pairs = load_cast_pair_params("duo", casts_dir=tmp_path)
    key = frozenset({"x", "y"})
    assert pairs[key].lora_file == "xy.safetensors"
    assert pairs[key].trigger == "xytok"


def test_pair_cache_returns_same_object(tmp_path: Path) -> None:
    body = 'MEMBERS = {}\nPAIRS = {frozenset({"a", "b"}): {"lora_file": "ab.safetensors"}}\n'
    _write_params(tmp_path, "cached", body)
    first = load_cast_pair_params("cached", casts_dir=tmp_path)
    second = load_cast_pair_params("cached", casts_dir=tmp_path)
    assert first is second
