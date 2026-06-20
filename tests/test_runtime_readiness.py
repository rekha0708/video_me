from pathlib import Path

from adapters.render_character.diffusion_adapter import DiffusionRenderAdapter
from core.config import Settings, load_app_config
from scripts.check_runtime_readiness import (
    FAIL,
    PASS,
    WARN,
    CheckResult,
    check_python_packages,
    check_service_health,
    check_system_tools,
    check_track_b_assets,
    exit_code_for_results,
)


def _write_placeholder_lora(path: Path) -> None:
    path.write_text(
        "TEST-ONLY placeholder for local Track B file-gate checks.\n",
        encoding="utf-8",
    )


def _config_with_assets(tmp_path: Path, *, placeholder_loras: bool):
    config = load_app_config()
    lora_dir = tmp_path / "loras"
    voice_dir = tmp_path / "voices"
    lora_dir.mkdir()

    render = DiffusionRenderAdapter(work_dir=tmp_path / "renders", lora_dir=lora_dir)
    for member in config.cast.members:
        lora_path = lora_dir / f"{render.lora_name(member.lora_ref)}.safetensors"
        if placeholder_loras:
            _write_placeholder_lora(lora_path)
        else:
            lora_path.write_bytes(b"trained-ish weights")

        voice_path = voice_dir / "kids_duo" / f"{member.id}.wav"
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        voice_path.write_bytes(b"RIFF fake wav")

    config.settings = Settings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "video_me.db",
        lora_dir=lora_dir,
        voice_dir=voice_dir,
    )
    return config


def test_track_b_strict_fails_placeholder_loras(tmp_path: Path) -> None:
    config = _config_with_assets(tmp_path, placeholder_loras=True)

    results = check_track_b_assets(config, code_test=False)

    lora_results = [r for r in results if r.name.startswith("LoRA")]
    assert {r.status for r in lora_results} == {FAIL}
    assert all("TEST-ONLY placeholder" in r.detail for r in lora_results)


def test_track_b_code_test_warns_for_placeholder_loras(tmp_path: Path) -> None:
    config = _config_with_assets(tmp_path, placeholder_loras=True)

    results = check_track_b_assets(config, code_test=True)

    lora_results = [r for r in results if r.name.startswith("LoRA")]
    voice_results = [r for r in results if r.name.startswith("Voice")]
    assert {r.status for r in lora_results} == {WARN}
    assert {r.status for r in voice_results} == {PASS}


def test_track_b_real_loras_pass(tmp_path: Path) -> None:
    config = _config_with_assets(tmp_path, placeholder_loras=False)

    results = check_track_b_assets(config, code_test=False)

    assert all(r.status == PASS for r in results)


def test_python_package_check_reports_missing_package() -> None:
    def fake_find_spec(module_name: str):
        return None if module_name == "openai" else object()

    results = check_python_packages(Settings(), find_spec=fake_find_spec)

    openai_result = next(r for r in results if r.name == "Python package: openai")
    assert openai_result.status == FAIL
    assert "--install-python-deps" in openai_result.detail


def test_system_tool_check_reports_missing_tool() -> None:
    def fake_which(tool: str):
        return None if tool == "ffmpeg" else f"/usr/bin/{tool}"

    results = check_system_tools(which=fake_which)

    ffmpeg_result = next(r for r in results if r.name == "System tool: ffmpeg")
    assert ffmpeg_result.status == FAIL


def test_service_health_can_warn_for_missing_services() -> None:
    def fake_urlopen(request, timeout):
        raise OSError("connection refused")

    results = check_service_health(
        Settings(),
        allow_missing_services=True,
        urlopen=fake_urlopen,
    )

    assert results
    assert all(r.status == WARN for r in results)


def test_exit_code_for_results_fails_only_on_failures() -> None:
    assert exit_code_for_results([CheckResult("a", PASS, ""), CheckResult("b", WARN, "")]) == 0
    assert exit_code_for_results([CheckResult("a", FAIL, "")]) == 1
