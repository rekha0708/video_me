"""High/low watermark tracking for the GPU/VRAM stats the dashboard already
polls every 5s (see services/gpu_status.py:collect_gpu_status and
services/templates/gpu_status.html's setInterval loop).

collect_gpu_status() is a pure, stateless snapshot — deliberately kept that
way for testability. This module holds the *only* mutable state: min/max
seen so far. The caller owns the store (a plain dict) and decides its
lifetime/reset scope; this module never reaches for a Python global itself.
"""

from __future__ import annotations

from typing import Any


def _update_metric(store: dict[str, Any], key: str, value: float | int | None) -> None:
    if value is None:
        return
    entry = store.get(key)
    if entry is None:
        store[key] = {"min": value, "max": value}
        return
    if value < entry["min"]:
        entry["min"] = value
    if value > entry["max"]:
        entry["max"] = value


def update_watermarks(store: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Fold one collect_gpu_status() sample into `store` (mutated in place) and return it.

    Tracks the same aggregates the dashboard's top metric cards already show
    (system cpu/memory percent, max GPU util across devices, aggregate VRAM
    percent/MB), plus a per-GPU breakdown keyed by UUID (falling back to
    index) for the devices table.
    """
    system = status.get("system") or {}
    memory = system.get("memory") or {}
    gpus = status.get("gpus") or []

    _update_metric(store, "cpu_percent", system.get("cpu_percent"))
    _update_metric(store, "memory_used_percent", memory.get("used_percent"))

    gpu_utils = [g.get("gpu_util_percent") for g in gpus if g.get("gpu_util_percent") is not None]
    if gpu_utils:
        _update_metric(store, "gpu_util_percent", max(gpu_utils))

    total_vram = sum(g.get("memory_total_mb") or 0 for g in gpus)
    if total_vram:
        used_vram = sum(g.get("memory_used_mb") or 0 for g in gpus)
        _update_metric(store, "vram_used_percent", round(used_vram / total_vram * 100, 1))
        _update_metric(store, "vram_used_mb", used_vram)

    per_gpu: dict[str, Any] = store.setdefault("per_gpu", {})
    for gpu in gpus:
        key = gpu.get("uuid") or (
            str(gpu["index"]) if gpu.get("index") is not None else None
        )
        if not key:
            continue
        gpu_store = per_gpu.setdefault(key, {})
        _update_metric(gpu_store, "gpu_util_percent", gpu.get("gpu_util_percent"))
        _update_metric(gpu_store, "vram_used_percent", gpu.get("vram_used_percent"))

    return store


def reset_watermarks(store: dict[str, Any]) -> None:
    store.clear()
