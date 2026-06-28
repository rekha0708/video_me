from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from adapters.render_character.diffusion_adapter import (
    DiffusionRenderAdapter,
    is_placeholder_lora,
)
from adapters.synthesize_voice.tts_adapter import TtsAdapter
from core.config import AppConfig, Settings, load_app_config

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_PYTHON_PACKAGES = (
    ("pydantic", "pydantic", "base"),
    ("pydantic_settings", "pydantic-settings", "base"),
    ("yaml", "pyyaml", "base"),
    ("openai", "openai", "llm/critique"),
    ("httpx", "httpx", "render/tts/video/lipsync"),
    ("faster_whisper", "faster-whisper", "transcribe"),
    ("yt_dlp", "yt-dlp", "ingest"),
)

_SYSTEM_TOOLS = (
    ("ffmpeg", "assemble/audio/critique"),
    ("ffprobe", "ingest/critique"),
    ("yt-dlp", "ingest"),
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def _module_exists(
    module_name: str,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def check_python_packages(
    settings: Settings,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> list[CheckResult]:
    packages = list(_PYTHON_PACKAGES)
    if settings.job_store == "postgres":
        packages.append(("psycopg", "psycopg[binary]", "postgres job store"))
    if settings.artifact_store == "s3":
        packages.append(("boto3", "boto3", "s3 artifact store"))

    results: list[CheckResult] = []
    for module_name, package_name, purpose in packages:
        if _module_exists(module_name, find_spec):
            results.append(
                CheckResult(
                    f"Python package: {package_name}",
                    PASS,
                    f"installed ({purpose})",
                )
            )
        else:
            results.append(
                CheckResult(
                    f"Python package: {package_name}",
                    FAIL,
                    f"missing ({purpose}); run python -m scripts.setup_gpu "
                    "--install-python-deps",
                )
            )
    return results


def check_system_tools(
    which: Callable[[str], str | None] = shutil.which,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for tool, purpose in _SYSTEM_TOOLS:
        path = which(tool)
        if path:
            results.append(CheckResult(f"System tool: {tool}", PASS, path))
        else:
            results.append(
                CheckResult(
                    f"System tool: {tool}",
                    FAIL,
                    f"missing ({purpose}); install ffmpeg and ensure yt-dlp is on PATH",
                )
            )
    return results


def check_track_b_assets(config: AppConfig, *, code_test: bool = False) -> list[CheckResult]:
    settings = config.settings
    work_dir = Path("/tmp/video_me_runtime_readiness")
    render = DiffusionRenderAdapter(
        work_dir=work_dir / "renders",
        lora_dir=settings.lora_dir,
    )
    voice = TtsAdapter(work_dir=work_dir / "voices", voice_dir=settings.voice_dir)

    results: list[CheckResult] = []
    for member in config.cast.members:
        try:
            path = render._check_lora(member)
        except RuntimeError as exc:
            results.append(CheckResult(f"LoRA: {member.name}", FAIL, str(exc)))
            continue

        if is_placeholder_lora(path):
            if code_test:
                detail = (
                    f"{path} is a TEST-ONLY placeholder; accepted for --code-test. "
                    "Set VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true before a "
                    "temporary render smoke test."
                )
                results.append(CheckResult(f"LoRA: {member.name}", WARN, detail))
            else:
                results.append(
                    CheckResult(
                        f"LoRA: {member.name}",
                        FAIL,
                        f"{path} is a TEST-ONLY placeholder; replace with trained weights "
                        "before renting GPU for a real run",
                    )
                )
        else:
            results.append(CheckResult(f"LoRA: {member.name}", PASS, str(path)))

    for member in config.cast.members:
        try:
            path = voice._check_voice(member.voice_profile_ref)
        except RuntimeError as exc:
            results.append(CheckResult(f"Voice: {member.name}", FAIL, str(exc)))
            continue
        results.append(CheckResult(f"Voice: {member.name}", PASS, str(path)))

    return results


def _join_url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"


def _service_urls(settings: Settings) -> list[tuple[str, str, bool]]:
    """Return (name, url, required) tuples. required=False means WARN on failure."""
    urls: list[tuple[str, str, bool]] = [
        # Required: always needed
        ("Ollama LLM/VLM API", _join_url(settings.llm_base_url, "models"), True),
    ]

    # Render backend
    if settings.render_adapter == "comfyui_flux":
        urls.append(("ComfyUI (Flux.1-dev + LTX)", settings.comfyui_base_url + "/", True))
    else:
        urls.append(("AUTOMATIC1111 (fallback)", _join_url(settings.sd_base_url, "sdapi/v1/sd-models"), True))

    # TTS backend
    if settings.tts_adapter == "fish_s2":
        urls.append(("Fish Audio S2 TTS", _join_url(settings.fish_s2_base_url, "health"), True))
    else:
        urls.append(("Chatterbox TTS (fallback)", _join_url(settings.tts_base_url, "health"), True))

    # Video backend (Wan + MuseTalk only needed when VIDEO_ADAPTER=wan)
    if settings.video_adapter == "wan":
        urls.append(("Wan image-to-video (fallback)", _join_url(settings.wan_base_url, "health"), True))
        urls.append(("MuseTalk lip-sync (fallback)", _join_url(settings.lipsync_base_url, "health"), True))

    return urls


def _url_ok(
    url: str,
    *,
    timeout: float,
    urlopen: Callable = urllib.request.urlopen,
) -> tuple[bool, str]:
    try:
        request = urllib.request.Request(url, method="GET")
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if 200 <= int(status) < 400:
                return True, f"HTTP {status}"
            return False, f"HTTP {status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def check_service_health(
    settings: Settings,
    *,
    timeout: float = 3.0,
    allow_missing_services: bool = False,
    urlopen: Callable = urllib.request.urlopen,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, url, required in _service_urls(settings):
        ok, detail = _url_ok(url, timeout=timeout, urlopen=urlopen)
        if ok:
            results.append(CheckResult(f"Service: {name}", PASS, f"{url} ({detail})"))
        else:
            # Non-required services always WARN; required services FAIL unless allow_missing
            status = WARN if (allow_missing_services or not required) else FAIL
            results.append(CheckResult(f"Service: {name}", status, f"{url} ({detail})"))
    return results


def collect_readiness_results(
    config: AppConfig,
    *,
    code_test: bool = False,
    skip_services: bool = False,
    allow_missing_services: bool = False,
    timeout: float = 3.0,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(check_python_packages(config.settings))
    results.extend(check_system_tools())
    results.extend(check_track_b_assets(config, code_test=code_test))
    if skip_services:
        results.append(
            CheckResult(
                "Service health checks",
                WARN,
                "skipped by --skip-services",
            )
        )
    else:
        results.extend(
            check_service_health(
                config.settings,
                timeout=timeout,
                allow_missing_services=allow_missing_services,
            )
        )
    return results


def exit_code_for_results(results: list[CheckResult]) -> int:
    return 1 if any(result.status == FAIL for result in results) else 0


def print_results(results: list[CheckResult], *, code_test: bool) -> None:
    mode = "CODE TEST" if code_test else "STRICT REAL RUN"
    print(f"video_me runtime readiness ({mode})")
    print()
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")

    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    print()
    print(
        f"Summary: {counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail"
    )
    if counts[FAIL]:
        print("Status: NOT READY")
    elif counts[WARN]:
        print("Status: READY WITH WARNINGS")
    else:
        print("Status: READY")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether video_me is ready for mock or real GPU runs."
    )
    parser.add_argument(
        "--code-test",
        action="store_true",
        help="Accept explicit TEST-ONLY LoRA placeholders as warnings.",
    )
    parser.add_argument(
        "--skip-services",
        action="store_true",
        help="Skip HTTP health checks for model services.",
    )
    parser.add_argument(
        "--allow-missing-services",
        action="store_true",
        help="Report service health failures as warnings instead of failures.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="HTTP health-check timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = load_app_config()
    except Exception as exc:
        print(f"[{FAIL}] Config load: {exc}")
        return 1

    results = collect_readiness_results(
        config,
        code_test=args.code_test,
        skip_services=args.skip_services,
        allow_missing_services=args.allow_missing_services,
        timeout=args.timeout,
    )
    print_results(results, code_test=args.code_test)
    return exit_code_for_results(results)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
