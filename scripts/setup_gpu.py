from __future__ import annotations

import argparse
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from scripts import check_runtime_readiness

_EXTRAS = ("services", "ingest", "transcribe", "llm", "render")


def python_deps_command(python_bin: str = sys.executable) -> list[str]:
    extras = ",".join(_EXTRAS)
    return [python_bin, "-m", "pip", "install", "-e", f".[{extras}]"]


def system_deps_commands(
    *,
    system: str | None = None,
    which=shutil.which,
) -> list[list[str]]:
    system_name = system or platform.system()
    if system_name == "Darwin" and which("brew"):
        return [["brew", "install", "ffmpeg"]]
    if system_name == "Linux" and which("apt-get"):
        return [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", "ffmpeg"],
        ]
    return []


def _format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(f"$ {_format_command(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _print_plan(args: argparse.Namespace) -> None:
    print("video_me GPU setup")
    print()
    print("Python dependencies:")
    print(f"  {_format_command(python_deps_command())}")
    print()
    print("System dependencies:")
    commands = system_deps_commands()
    if commands:
        for cmd in commands:
            print(f"  {_format_command(cmd)}")
    else:
        print("  Install ffmpeg/ffprobe with your OS package manager.")
    print()
    print("Readiness check:")
    readiness_args = _readiness_args(args)
    print(
        "  python -m scripts.check_runtime_readiness "
        f"{' '.join(readiness_args)}".rstrip()
    )
    print()
    if args.code_test:
        print("Temporary code-test render mode:")
        print("  export VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true")
        print()


def _readiness_args(args: argparse.Namespace) -> list[str]:
    readiness_args: list[str] = []
    if args.code_test:
        readiness_args.append("--code-test")
    if args.skip_services:
        readiness_args.append("--skip-services")
    if args.allow_missing_services:
        readiness_args.append("--allow-missing-services")
    readiness_args.extend(["--timeout", str(args.timeout)])
    return readiness_args


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install and validate video_me dependencies on a GPU machine."
    )
    parser.add_argument(
        "--install-python-deps",
        action="store_true",
        help="Install project optional dependency groups needed for runtime.",
    )
    parser.add_argument(
        "--install-system-deps",
        action="store_true",
        help="Install ffmpeg/ffprobe when a supported package manager is available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print install commands without executing them.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Do not run the runtime readiness check after setup.",
    )
    parser.add_argument(
        "--code-test",
        action="store_true",
        help="Pass --code-test to readiness checks for placeholder LoRA smoke tests.",
    )
    parser.add_argument(
        "--skip-services",
        action="store_true",
        help="Pass --skip-services to readiness checks.",
    )
    parser.add_argument(
        "--allow-missing-services",
        action="store_true",
        help="Pass --allow-missing-services to readiness checks.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Readiness HTTP timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _print_plan(args)

    if args.install_system_deps:
        commands = system_deps_commands()
        if not commands:
            print("No supported system package manager detected for automatic ffmpeg install.")
            return 1
        for cmd in commands:
            _run(cmd, dry_run=args.dry_run)

    if args.install_python_deps:
        _run(python_deps_command(), dry_run=args.dry_run)

    if args.no_check:
        return 0

    if args.dry_run:
        print("Dry run complete; readiness check not executed.")
        return 0

    return check_runtime_readiness.main(_readiness_args(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
