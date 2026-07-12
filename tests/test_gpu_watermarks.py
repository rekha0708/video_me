from services.gpu_watermarks import reset_watermarks, update_watermarks


def _status(cpu=None, mem_pct=None, gpus=None):
    return {
        "system": {"cpu_percent": cpu, "memory": {"used_percent": mem_pct}},
        "gpus": gpus or [],
    }


def test_first_sample_sets_min_and_max_equal() -> None:
    store: dict = {}
    update_watermarks(store, _status(cpu=42.0, mem_pct=30.0))

    assert store["cpu_percent"] == {"min": 42.0, "max": 42.0}
    assert store["memory_used_percent"] == {"min": 30.0, "max": 30.0}


def test_widens_range_across_samples() -> None:
    store: dict = {}
    update_watermarks(store, _status(cpu=50.0, mem_pct=50.0))
    update_watermarks(store, _status(cpu=10.0, mem_pct=90.0))
    update_watermarks(store, _status(cpu=70.0, mem_pct=20.0))

    assert store["cpu_percent"] == {"min": 10.0, "max": 70.0}
    assert store["memory_used_percent"] == {"min": 20.0, "max": 90.0}


def test_none_values_are_ignored() -> None:
    store: dict = {}
    update_watermarks(store, _status(cpu=40.0, mem_pct=None))
    update_watermarks(store, _status(cpu=None, mem_pct=None))

    assert store["cpu_percent"] == {"min": 40.0, "max": 40.0}
    assert "memory_used_percent" not in store


def test_tracks_max_gpu_util_across_devices() -> None:
    store: dict = {}
    gpus = [
        {"uuid": "gpu-0", "index": 0, "gpu_util_percent": 20, "vram_used_percent": 10.0,
         "memory_used_mb": 1000, "memory_total_mb": 10000},
        {"uuid": "gpu-1", "index": 1, "gpu_util_percent": 80, "vram_used_percent": 60.0,
         "memory_used_mb": 6000, "memory_total_mb": 10000},
    ]
    update_watermarks(store, _status(gpus=gpus))

    # aggregate: max util across devices, aggregate VRAM % across all devices combined
    assert store["gpu_util_percent"] == {"min": 80, "max": 80}
    assert store["vram_used_percent"] == {"min": 35.0, "max": 35.0}  # (1000+6000)/20000
    assert store["vram_used_mb"] == {"min": 7000, "max": 7000}


def test_per_gpu_breakdown_keyed_by_uuid() -> None:
    store: dict = {}
    update_watermarks(store, _status(gpus=[
        {"uuid": "gpu-0", "index": 0, "gpu_util_percent": 10, "vram_used_percent": 20.0,
         "memory_used_mb": 2000, "memory_total_mb": 10000},
    ]))
    update_watermarks(store, _status(gpus=[
        {"uuid": "gpu-0", "index": 0, "gpu_util_percent": 90, "vram_used_percent": 5.0,
         "memory_used_mb": 500, "memory_total_mb": 10000},
    ]))

    per_gpu = store["per_gpu"]["gpu-0"]
    assert per_gpu["gpu_util_percent"] == {"min": 10, "max": 90}
    assert per_gpu["vram_used_percent"] == {"min": 5.0, "max": 20.0}


def test_per_gpu_falls_back_to_index_when_uuid_missing() -> None:
    store: dict = {}
    update_watermarks(store, _status(gpus=[
        {"uuid": "", "index": 0, "gpu_util_percent": 15, "vram_used_percent": 25.0,
         "memory_used_mb": 2500, "memory_total_mb": 10000},
    ]))

    assert "0" in store["per_gpu"]


def test_no_gpus_does_not_set_gpu_keys() -> None:
    store: dict = {}
    update_watermarks(store, _status(cpu=10.0, gpus=[]))

    assert "gpu_util_percent" not in store
    assert "vram_used_percent" not in store
    assert store["per_gpu"] == {}


def test_update_watermarks_mutates_and_returns_same_store() -> None:
    store: dict = {}
    result = update_watermarks(store, _status(cpu=1.0))
    assert result is store


def test_reset_watermarks_clears_store() -> None:
    store: dict = {}
    update_watermarks(store, _status(cpu=1.0, gpus=[
        {"uuid": "gpu-0", "index": 0, "gpu_util_percent": 1, "vram_used_percent": 1.0,
         "memory_used_mb": 100, "memory_total_mb": 10000},
    ]))
    assert store

    reset_watermarks(store)

    assert store == {}
