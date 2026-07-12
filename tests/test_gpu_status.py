from pathlib import Path
from types import SimpleNamespace

import pytest

from services.gpu_status import (
    collect_log_tails,
    cpu_percent_from_samples,
    parse_nvidia_compute_apps_csv,
    parse_nvidia_gpu_csv,
    parse_proc_meminfo,
    parse_proc_stat_cpu_line,
)


def test_parse_nvidia_gpu_csv() -> None:
    text = "0, NVIDIA RTX 6000 Ada, GPU-abc, 62, 18, 98304, 45123, 53181, 65, 255.5, 300.0\n"

    gpus = parse_nvidia_gpu_csv(text)

    assert gpus == [
        {
            "index": 0,
            "name": "NVIDIA RTX 6000 Ada",
            "uuid": "GPU-abc",
            "gpu_util_percent": 62,
            "memory_util_percent": 18,
            "memory_total_mb": 98304,
            "memory_used_mb": 45123,
            "memory_free_mb": 53181,
            "vram_used_percent": 45.9,
            "temperature_c": 65,
            "power_draw_w": 255.5,
            "power_limit_w": 300.0,
        }
    ]


def test_parse_nvidia_gpu_csv_handles_na_values() -> None:
    text = "0, NVIDIA A100, GPU-def, 9, [N/A], 81920, 20480, 61440, [N/A], [N/A], [N/A]\n"

    gpu = parse_nvidia_gpu_csv(text)[0]

    assert gpu["memory_util_percent"] is None
    assert gpu["temperature_c"] is None
    assert gpu["power_draw_w"] is None
    assert gpu["vram_used_percent"] == 25.0


def test_parse_nvidia_compute_apps_csv() -> None:
    text = "GPU-abc, 1234, python, 24576\n"

    processes = parse_nvidia_compute_apps_csv(text)

    assert processes == [
        {
            "gpu_uuid": "GPU-abc",
            "pid": 1234,
            "process_name": "python",
            "process_name_resolved": True,
            "used_memory_mb": 24576,
        }
    ]


def test_parse_nvidia_compute_apps_csv_handles_unresolved_name() -> None:
    # Seen inside containers without --pid=host: NVML reports a real host PID,
    # but nvidia-smi can't resolve its name from the container's own /proc.
    text = "GPU-abc, 709032, [Not Found], 20106\n"

    processes = parse_nvidia_compute_apps_csv(text)

    assert processes == [
        {
            "gpu_uuid": "GPU-abc",
            "pid": 709032,
            "process_name": None,
            "process_name_resolved": False,
            "used_memory_mb": 20106,
        }
    ]


def test_parse_proc_meminfo() -> None:
    text = """
MemTotal:       131072000 kB
MemFree:          2048000 kB
MemAvailable:    65536000 kB
Buffers:          1024000 kB
"""

    memory = parse_proc_meminfo(text)

    assert memory == {
        "total_mb": 128000,
        "available_mb": 64000,
        "used_mb": 64000,
        "used_percent": 50.0,
    }


def test_cpu_percent_from_proc_stat_samples() -> None:
    start = parse_proc_stat_cpu_line("cpu  100 0 100 800 0 0 0 0 0 0")
    end = parse_proc_stat_cpu_line("cpu  150 0 150 850 0 0 0 0 0 0")

    assert cpu_percent_from_samples(start, end) == 66.7


def test_collect_log_tails_reads_existing_logs(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "dashboard_worker.log").write_text(
        "\n".join(f"line {idx}" for idx in range(8)),
        encoding="utf-8",
    )

    logs = collect_log_tails(tmp_path, log_lines=3)

    assert logs == [
        {
            "path": str(logs_dir / "dashboard_worker.log"),
            "name": "dashboard_worker",
            "exists": True,
            "lines": ["line 5", "line 6", "line 7"],
            "error": None,
            "size_bytes": (logs_dir / "dashboard_worker.log").stat().st_size,
        }
    ]


_has_fastapi = True
try:
    import fastapi  # noqa: F401
except ImportError:
    _has_fastapi = False


@pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")
def test_gpu_status_endpoint_clamps_log_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from core.config import AppConfig, Settings
    from core.models.profile import Cast, CastMember, ChannelProfile
    import services.dashboard_api as dashboard_api

    seen: dict[str, object] = {}

    def fake_collect_gpu_status(*, workspace: Path, log_lines: int):
        seen["workspace"] = workspace
        seen["log_lines"] = log_lines
        return {
            "collected_at": "2026-07-09T00:00:00+00:00",
            "workspace": str(workspace),
            "system": {"hostname": "gpu-host"},
            "nvidia_smi": {"available": True, "error": None},
            "gpus": [],
            "processes": [],
            "logs": [],
        }

    monkeypatch.setattr(dashboard_api, "collect_gpu_status", fake_collect_gpu_status)

    settings = Settings(
        data_dir=str(tmp_path / "data"),
        artifact_dir=str(tmp_path / "art"),
        sqlite_path=str(tmp_path / "test.db"),
    )
    cfg = AppConfig(
        settings=settings,
        channel_profile=ChannelProfile(
            id="test",
            name="test",
            aspect_ratio="9:16",
            genre_content="education",
            tone="friendly",
            format="animated_character",
            made_for_kids=True,
        ),
        cast=Cast(
            id="kids_duo",
            species="human",
            is_original_synthetic=True,
            members=[
                CastMember(
                    id="max",
                    name="Max",
                    visual_descriptor="boy",
                    lora_ref="loras/max",
                    voice_profile_ref="voices/max",
                    personality="friendly",
                ),
            ],
        ),
    )

    client = TestClient(dashboard_api.create_app(config_loader=lambda: cfg), raise_server_exceptions=False)
    resp = client.get("/api/runtime/gpu-status?log_lines=9999")

    assert resp.status_code == 200
    assert resp.json()["system"]["hostname"] == "gpu-host"
    assert seen["log_lines"] == 500
    assert isinstance(seen["workspace"], Path)


def _dashboard_test_app(tmp_path: Path):
    from core.config import AppConfig, Settings
    from core.models.profile import Cast, CastMember, ChannelProfile
    import services.dashboard_api as dashboard_api

    settings = Settings(
        data_dir=str(tmp_path / "data"),
        artifact_dir=str(tmp_path / "art"),
        sqlite_path=str(tmp_path / "test.db"),
    )
    cfg = AppConfig(
        settings=settings,
        channel_profile=ChannelProfile(
            id="test", name="test", aspect_ratio="9:16", genre_content="education",
            tone="friendly", format="animated_character", made_for_kids=True,
        ),
        cast=Cast(
            id="kids_duo", species="human", is_original_synthetic=True,
            members=[
                CastMember(
                    id="max", name="Max", visual_descriptor="boy",
                    lora_ref="loras/max", voice_profile_ref="voices/max", personality="friendly",
                ),
            ],
        ),
    )
    return dashboard_api, dashboard_api.create_app(config_loader=lambda: cfg)


@pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")
def test_gpu_status_endpoint_includes_and_widens_watermarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    dashboard_api, app = _dashboard_test_app(tmp_path)
    dashboard_api._gpu_watermarks.clear()  # isolate from other tests sharing this module global

    samples = iter([
        {
            "collected_at": "t1", "workspace": str(tmp_path),
            "system": {"hostname": "h", "cpu_percent": 10.0, "memory": {"used_percent": 20.0}},
            "nvidia_smi": {"available": True, "error": None}, "processes": [], "logs": [],
            "gpus": [{
                "uuid": "gpu-0", "index": 0, "gpu_util_percent": 30, "vram_used_percent": 40.0,
                "memory_used_mb": 4000, "memory_total_mb": 10000,
            }],
        },
        {
            "collected_at": "t2", "workspace": str(tmp_path),
            "system": {"hostname": "h", "cpu_percent": 90.0, "memory": {"used_percent": 5.0}},
            "nvidia_smi": {"available": True, "error": None}, "processes": [], "logs": [],
            "gpus": [{
                "uuid": "gpu-0", "index": 0, "gpu_util_percent": 5, "vram_used_percent": 10.0,
                "memory_used_mb": 1000, "memory_total_mb": 10000,
            }],
        },
    ])
    monkeypatch.setattr(dashboard_api, "collect_gpu_status", lambda **_kw: next(samples))

    client = TestClient(app, raise_server_exceptions=False)

    wm1 = client.get("/api/runtime/gpu-status").json()["watermarks"]
    assert wm1["cpu_percent"] == {"min": 10.0, "max": 10.0}

    wm2 = client.get("/api/runtime/gpu-status").json()["watermarks"]
    assert wm2["cpu_percent"] == {"min": 10.0, "max": 90.0}
    assert wm2["gpu_util_percent"] == {"min": 5, "max": 30}
    assert wm2["per_gpu"]["gpu-0"]["vram_used_percent"] == {"min": 10.0, "max": 40.0}

    dashboard_api._gpu_watermarks.clear()


@pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")
def test_gpu_watermarks_reset_endpoint_clears_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    dashboard_api, app = _dashboard_test_app(tmp_path)
    dashboard_api._gpu_watermarks.clear()

    monkeypatch.setattr(
        dashboard_api, "collect_gpu_status",
        lambda **_kw: {
            "collected_at": "t1", "workspace": str(tmp_path),
            "system": {"hostname": "h", "cpu_percent": 55.0, "memory": {"used_percent": 33.0}},
            "nvidia_smi": {"available": True, "error": None}, "processes": [], "logs": [], "gpus": [],
        },
    )

    client = TestClient(app, raise_server_exceptions=False)
    client.get("/api/runtime/gpu-status")
    assert dashboard_api._gpu_watermarks  # populated

    reset_resp = client.post("/api/runtime/gpu-watermarks/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["watermarks"] == {}
    assert dashboard_api._gpu_watermarks == {}

    dashboard_api._gpu_watermarks.clear()
