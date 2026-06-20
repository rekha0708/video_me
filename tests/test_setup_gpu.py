import sys

from scripts.setup_gpu import python_deps_command, system_deps_commands


def test_python_deps_command_installs_runtime_extras() -> None:
    cmd = python_deps_command("python")

    assert cmd[:4] == ["python", "-m", "pip", "install"]
    assert ".[services,ingest,transcribe,llm,render]" in cmd


def test_system_deps_commands_use_brew_on_macos() -> None:
    commands = system_deps_commands(system="Darwin", which=lambda name: "/opt/homebrew/bin/brew")

    assert commands == [["brew", "install", "ffmpeg"]]


def test_system_deps_commands_use_apt_on_linux() -> None:
    def fake_which(name: str):
        return "/usr/bin/apt-get" if name == "apt-get" else None

    commands = system_deps_commands(system="Linux", which=fake_which)

    assert commands == [
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", "ffmpeg"],
    ]


def test_python_deps_command_defaults_to_current_interpreter() -> None:
    cmd = python_deps_command()

    assert cmd[0] == sys.executable
