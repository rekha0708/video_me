"""Runtime GPU/server status collection for the dashboard."""

from __future__ import annotations

import csv
import os
import platform
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

GPU_QUERY_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
    "power.draw",
    "power.limit",
)

COMPUTE_APP_QUERY_FIELDS = (
    "gpu_uuid",
    "pid",
    "process_name",
    "used_memory",
)

DEFAULT_LOG_NAMES = (
    "dashboard_api",
    "dashboard_worker",
    "comfyui",
    "ollama",
    "fish_s2",
    "wan",
    "musetalk",
    "a1111",
    "chatterbox",
)


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_value(value: str) -> str:
    value = value.strip()
    if value in {"[N/A]", "N/A", "Not Supported", "[Not Supported]"}:
        return ""
    return value


def _to_int(value: str) -> int | None:
    value = _clean_value(value)
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: str) -> float | None:
    value = _clean_value(value)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _percent(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round((float(numerator) / float(denominator)) * 100.0, 1)


def _csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        rows.append([cell.strip() for cell in row])
    return rows


def parse_nvidia_gpu_csv(text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu` CSV output."""
    gpus: list[dict[str, Any]] = []
    for row in _csv_rows(text):
        if len(row) < len(GPU_QUERY_FIELDS):
            continue
        data = dict(zip(GPU_QUERY_FIELDS, row, strict=False))
        total_mb = _to_int(data["memory.total"])
        used_mb = _to_int(data["memory.used"])
        gpu = {
            "index": _to_int(data["index"]),
            "name": _clean_value(data["name"]),
            "gpu_util_percent": _to_int(data["utilization.gpu"]),
            "memory_util_percent": _to_int(data["utilization.memory"]),
            "memory_total_mb": total_mb,
            "memory_used_mb": used_mb,
            "memory_free_mb": _to_int(data["memory.free"]),
            "vram_used_percent": _percent(used_mb, total_mb),
            "temperature_c": _to_int(data["temperature.gpu"]),
            "power_draw_w": _to_float(data["power.draw"]),
            "power_limit_w": _to_float(data["power.limit"]),
        }
        gpus.append(gpu)
    return gpus


def parse_nvidia_compute_apps_csv(text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-compute-apps` CSV output."""
    if "No running processes found" in text:
        return []
    processes: list[dict[str, Any]] = []
    for row in _csv_rows(text):
        if len(row) < len(COMPUTE_APP_QUERY_FIELDS):
            continue
        data = dict(zip(COMPUTE_APP_QUERY_FIELDS, row, strict=False))
        processes.append(
            {
                "gpu_uuid": _clean_value(data["gpu_uuid"]),
                "pid": _to_int(data["pid"]),
                "process_name": _clean_value(data["process_name"]),
                "used_memory_mb": _to_int(data["used_memory"]),
            }
        )
    return processes


def parse_proc_meminfo(text: str) -> dict[str, Any] | None:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue

    total_kb = values.get("MemTotal")
    available_kb = values.get("MemAvailable", values.get("MemFree"))
    if total_kb is None or available_kb is None:
        return None

    used_kb = max(total_kb - available_kb, 0)
    return {
        "total_mb": round(total_kb / 1024),
        "available_mb": round(available_kb / 1024),
        "used_mb": round(used_kb / 1024),
        "used_percent": _percent(used_kb, total_kb),
    }


def parse_proc_stat_cpu_line(line: str) -> tuple[int, int] | None:
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError:
        return None
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def cpu_percent_from_samples(
    start: tuple[int, int] | None, end: tuple[int, int] | None
) -> float | None:
    if start is None or end is None:
        return None
    idle_delta = end[0] - start[0]
    total_delta = end[1] - start[1]
    if total_delta <= 0:
        return None
    busy_delta = max(total_delta - idle_delta, 0)
    return round((busy_delta / total_delta) * 100.0, 1)


def _read_proc_stat(path: Path = Path("/proc/stat")) -> tuple[int, int] | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return parse_proc_stat_cpu_line(fh.readline())
    except OSError:
        return None


def _read_proc_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, Any] | None:
    try:
        return parse_proc_meminfo(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _sample_cpu_percent(interval_sec: float = 0.1) -> float | None:
    start = _read_proc_stat()
    if start is None:
        return None
    time.sleep(max(interval_sec, 0.0))
    return cpu_percent_from_samples(start, _read_proc_stat())


def _default_runner(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _run_nvidia_smi(
    query_fields: Sequence[str],
    query_kind: str,
    *,
    runner: CommandRunner = _default_runner,
    timeout: float = 3.0,
) -> tuple[str, str | None]:
    args = [
        "nvidia-smi",
        f"--query-{query_kind}={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(args, timeout)
    except FileNotFoundError:
        return "", "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return "", f"nvidia-smi timed out after {timeout:g}s"
    except OSError as exc:
        return "", str(exc)

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        return "", message or f"nvidia-smi exited with {result.returncode}"
    return result.stdout.strip(), None


def _split_env_paths(value: str) -> list[str]:
    paths: list[str] = []
    for chunk in value.replace(",", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            paths.append(chunk)
    return paths


def _candidate_log_paths(workspace: Path) -> list[Path]:
    env_value = os.getenv("VIDEO_ME_GPU_LOG_FILES")
    if env_value:
        return [
            Path(value).expanduser() if Path(value).expanduser().is_absolute()
            else workspace / value
            for value in _split_env_paths(env_value)
        ]

    log_dir = workspace / "logs"
    paths: list[Path] = [log_dir / f"{name}.log" for name in DEFAULT_LOG_NAMES]
    if log_dir.is_dir():
        paths.extend(sorted(log_dir.glob("*.log")))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            unique.append(resolved)
    return unique


def tail_file(path: Path, *, max_lines: int = 120, max_bytes: int = 200_000) -> list[str]:
    with path.open("rb") as fh:
        try:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(size - max_bytes, 0), os.SEEK_SET)
        except OSError:
            fh.seek(0)
        data = fh.read().decode("utf-8", errors="replace")
    return data.splitlines()[-max_lines:]


def collect_log_tails(workspace: Path, *, log_lines: int = 120) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for path in _candidate_log_paths(workspace):
        record: dict[str, Any] = {
            "path": str(path),
            "name": path.stem,
            "exists": path.exists(),
            "lines": [],
            "error": None,
        }
        if not path.exists():
            record["error"] = "log file not found"
            logs.append(record)
            continue
        try:
            record["size_bytes"] = path.stat().st_size
            record["lines"] = tail_file(path, max_lines=log_lines)
        except OSError as exc:
            record["error"] = str(exc)
        logs.append(record)
    return logs


def _collect_system_status() -> dict[str, Any]:
    try:
        load_avg = list(os.getloadavg())
    except OSError:
        load_avg = None

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_percent": _sample_cpu_percent(),
        "load_avg": load_avg,
        "memory": _read_proc_meminfo(),
    }


def collect_gpu_status(
    *,
    workspace: Path | None = None,
    log_lines: int = 120,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Collect host, GPU, process, and service-log status for the dashboard."""
    workspace = (workspace or Path.cwd()).resolve()
    log_lines = max(10, min(int(log_lines), 500))

    gpu_output, gpu_error = _run_nvidia_smi(
        GPU_QUERY_FIELDS,
        "gpu",
        runner=runner,
    )
    gpus = parse_nvidia_gpu_csv(gpu_output) if gpu_output else []

    process_output, process_error = _run_nvidia_smi(
        COMPUTE_APP_QUERY_FIELDS,
        "compute-apps",
        runner=runner,
    )
    processes = parse_nvidia_compute_apps_csv(process_output) if process_output else []

    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_available = bool(gpus) or (nvidia_smi_path is not None and gpu_error is None)

    return {
        "collected_at": _utc_iso(),
        "workspace": str(workspace),
        "system": _collect_system_status(),
        "nvidia_smi": {
            "available": nvidia_available,
            "path": nvidia_smi_path,
            "error": gpu_error,
            "process_error": process_error,
        },
        "gpus": gpus,
        "processes": processes,
        "logs": collect_log_tails(workspace, log_lines=log_lines),
    }
