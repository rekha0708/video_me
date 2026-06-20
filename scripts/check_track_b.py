from pathlib import Path

from adapters.render_character.diffusion_adapter import DiffusionRenderAdapter
from adapters.synthesize_voice.tts_adapter import TtsAdapter
from core.config import load_app_config


def _is_test_placeholder(path: Path) -> bool:
    try:
        return path.read_bytes()[:64].startswith(b"TEST-ONLY placeholder")
    except OSError:
        return False


def main() -> int:
    config = load_app_config()
    settings = config.settings
    work_dir = Path("/tmp/video_me_track_b_check")

    render = DiffusionRenderAdapter(work_dir=work_dir / "renders", lora_dir=settings.lora_dir)
    voice = TtsAdapter(work_dir=work_dir / "voices", voice_dir=settings.voice_dir)

    ok = True
    has_test_placeholders = False
    print(f"Track B preflight for cast: {config.cast.id}")
    print()

    print("LoRA checks:")
    for member in config.cast.members:
        try:
            path = render._check_lora(member)
            if _is_test_placeholder(path):
                has_test_placeholders = True
                print(f"  OK TEST {member.name}: {path} (placeholder, not a real LoRA)")
            else:
                print(f"  OK      {member.name}: {path}")
        except RuntimeError as exc:
            ok = False
            print(f"  MISSING {member.name}: {exc}")

    print()
    print("Voice checks:")
    for member in config.cast.members:
        try:
            path = voice._check_voice(member.voice_profile_ref)
            print(f"  OK      {member.name}: {path}")
        except RuntimeError as exc:
            ok = False
            print(f"  MISSING {member.name}: {exc}")

    print()
    if ok:
        if has_test_placeholders:
            print("Track B: READY_FOR_CODE_TESTS")
            print("Real render runs still need trained LoRA weights.")
        else:
            print("Track B: READY")
        return 0

    print("Track B: INCOMPLETE")
    print("Place the files listed above before running the full pipeline.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
