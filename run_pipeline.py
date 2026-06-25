#!/usr/bin/env python3
"""
video_me pipeline runner — turn a source video into a Max & Zoe kids' short.

Usage:
    python run_pipeline.py <URL>
    python run_pipeline.py <URL> --critique          # Phase 2: critique + regenerate loop
    python run_pipeline.py <URL> --rights-cleared    # confirm you hold rights to the source
    python run_pipeline.py <URL> --whisper-device cuda  # use GPU for transcription

Supported sources:  YouTube, Instagram, TikTok, and any URL yt-dlp handles.

Ideal source video length for a first trial:
  60–180 seconds (1–3 minutes).
  - Shorter than 60s: too little content for the LLM to build a rich adapted script.
  - Longer than 3 min: each extra minute adds ~2 shots, each shot runs SD + TTS +
    Wan video gen + lip sync — plan ~5–10 min GPU time per shot.
  Sweet spot: a punchy 90-second educational clip (counting, colours, animals, etc.)
  gives 4–6 shots and a full end-to-end run in under an hour.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Turn a reference video into a Max & Zoe kids' educational short.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("url", help="YouTube / Instagram / TikTok video URL")
    p.add_argument(
        "--critique",
        action="store_true",
        help="Run Phase 2 critic loop (generate → critique → regenerate up to max_regenerations)",
    )
    p.add_argument(
        "--rights-cleared",
        action="store_true",
        default=False,
        help=(
            "Assert that the source video is cleared for transformative use. "
            "Pipeline is BLOCKED without this flag."
        ),
    )
    p.add_argument(
        "--whisper-device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for faster-whisper transcription (default: from VIDEO_ME_WHISPER_DEVICE or 'cpu')",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="Ollama model for LLM stages (default: from VIDEO_ME_LLM_MODEL or 'qwen2.5:7b')",
    )
    p.add_argument(
        "--review-dir",
        default=None,
        type=Path,
        help="Output folder for the finished video (default: ./review/)",
    )
    return p.parse_args()


def print_banner(url: str, critique: bool, rights_cleared: bool) -> None:
    mode = "Phase 2 (critique loop)" if critique else "Phase 1 (single pass)"
    rights = "CLEARED" if rights_cleared else "NOT CLEARED — job will be BLOCKED at rights check"
    print()
    print("=" * 64)
    print("  video_me pipeline")
    print("=" * 64)
    print(f"  Source : {url}")
    print(f"  Mode   : {mode}")
    print(f"  Rights : {rights}")
    print("=" * 64)
    print()


def find_output(review_dir: Path) -> Path | None:
    """Return the most recently created video.mp4 under review_dir."""
    candidates = sorted(
        review_dir.glob("*/video.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


async def main() -> int:
    args = parse_args()

    if not args.rights_cleared:
        print(
            "\n[WARNING] --rights-cleared not set.\n"
            "The pipeline will be BLOCKED at the rights-check stage.\n"
            "Pass --rights-cleared to confirm the source is cleared for transformative use.\n"
        )

    # ── Config ────────────────────────────────────────────────────────────────
    # Import here so heavy deps don't slow --help
    from core.config import load_app_config

    config = load_app_config()

    if args.whisper_device:
        config.settings.whisper_device = args.whisper_device
        config.settings.whisper_compute_type = "float16" if args.whisper_device == "cuda" else "int8"

    if args.llm_model:
        config.settings.llm_model = args.llm_model

    if args.review_dir:
        config.settings.review_dir = args.review_dir

    print_banner(args.url, args.critique, args.rights_cleared)

    # ── Run ───────────────────────────────────────────────────────────────────
    start = time.perf_counter()

    try:
        if args.critique:
            from core.workflow import run_with_critique
            logger.info("Starting Phase 2 critique pipeline ...")
            job = await run_with_critique(
                source_url=args.url,
                rights_cleared=args.rights_cleared,
                app_config=config,
            )
        else:
            from core.workflow import run_pipeline_job
            logger.info("Starting Phase 1 pipeline ...")
            job = await run_pipeline_job(
                source_url=args.url,
                rights_cleared=args.rights_cleared,
                app_config=config,
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130

    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"\n[FAILED] after {elapsed:.1f}s: {exc}")
        logger.exception("Pipeline failed")
        return 1

    elapsed = time.perf_counter() - start
    minutes, seconds = divmod(int(elapsed), 60)

    print()
    print("=" * 64)
    print(f"  Status  : {job.status.value.upper()}")
    print(f"  Job ID  : {job.job_id}")
    print(f"  Elapsed : {minutes}m {seconds}s")

    if job.status.value == "completed":
        output = find_output(config.settings.review_dir)
        if output:
            print(f"  Output  : {output}")
            meta = output.parent / "metadata.json"
            if meta.exists():
                print(f"  Sidecar : {meta}")
        print("=" * 64)
        print()
        return 0

    if job.status.value == "blocked":
        print()
        print("  Pipeline was BLOCKED at rights check.")
        print("  Re-run with --rights-cleared to proceed.")

    print("=" * 64)
    print()
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
