#!/usr/bin/env python3
"""
video_me pipeline runner — turn a source video into a Max & Zoe kids' short.

Basic usage:
    python run_pipeline.py <URL|FILE>
    python run_pipeline.py <URL> --rights-cleared --whisper-device cuda

Phase control (run only one part of the pipeline):
    python run_pipeline.py <URL> --phase plan          # LLM only — fetch+analyze+plan
    python run_pipeline.py --resume-job <ID> --phase render    # GPU loop only
    python run_pipeline.py --resume-job <ID> --phase assemble  # ffmpeg concat only
    python run_pipeline.py --resume-job <ID> --only-shot s03   # one shot only

Resume a failed/partial run:
    python run_pipeline.py --resume-job <JOB_ID>       # skip already-done stages

Test a single stage against existing files (no full pipeline needed):
    python run_pipeline.py --resume-job <ID> --phase render --only-shot s01

Supported sources:  YouTube, Instagram, TikTok, and any URL yt-dlp handles.
"""

import argparse
import asyncio
import logging
import os
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
    p.add_argument(
        "url",
        nargs="?",
        help=(
            "YouTube / Instagram / TikTok video URL, "
            "OR a local file path.  Optional when --resume-job is given."
        ),
    )
    p.add_argument(
        "--resume-job",
        metavar="JOB_ID",
        default=None,
        help=(
            "Resume or continue an existing job by ID (e.g. 20260625-200336-1js). "
            "Skips stages whose artifact JSON already exists on disk."
        ),
    )
    p.add_argument(
        "--phase",
        choices=["plan", "render", "assemble", "all"],
        default="all",
        help=(
            "Run only one phase: "
            "'plan' = fetch+transcribe+analyze+adapt+plan_shots; "
            "'render' = per-shot loop (render+voice+video+lipsync); "
            "'assemble' = assemble_video+publish; "
            "'all' = full pipeline (default)."
        ),
    )
    p.add_argument(
        "--only-shot",
        metavar="SHOT_ID",
        default=None,
        help="In the render phase, process only this shot (e.g. s01, s03).",
    )
    p.add_argument(
        "--critique",
        action="store_true",
        help="Run Phase 2 critic loop (generate → critique → regenerate up to max_regenerations)",
    )
    p.add_argument(
        "--rights-cleared",
        action="store_true",
        help=(
            "Assert that the source video is cleared for transformative use. "
            "Pipeline is BLOCKED without this flag."
        ),
    )
    p.add_argument(
        "--target-language",
        choices=["en", "hi", "both"],
        default=None,
        help=(
            "Output language for this run. Defaults to VIDEO_ME_TARGET_LANGUAGE "
            "from config when omitted."
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
        help="Ollama model for LLM stages (default: from VIDEO_ME_LLM_MODEL or 'qwen3.5:35b')",
    )
    p.add_argument(
        "--review-dir",
        default=None,
        type=Path,
        help="Output folder for the finished video (default: ./review/)",
    )
    return p.parse_args()


def print_banner(url: str | None, critique: bool, rights_cleared: bool,
                 resume_job: str | None, phase: str, only_shot: str | None) -> None:
    if critique:
        mode = "Phase 2 (critique loop)"
    elif phase != "all":
        mode = f"Phase 1 — {phase} only"
    else:
        mode = "Phase 1 (single pass)"
    rights = "CLEARED" if rights_cleared else "NOT CLEARED — job will be BLOCKED at rights check"
    print()
    print("=" * 64)
    print("  video_me pipeline")
    print("=" * 64)
    if resume_job:
        print(f"  Resume : {resume_job}")
    if url:
        print(f"  Source : {url}")
    print(f"  Mode   : {mode}")
    if only_shot:
        print(f"  Shot   : {only_shot}")
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

    # Validate: need either a URL or a resume-job
    if not args.url and not args.resume_job:
        print("error: provide a source URL/file, or --resume-job <JOB_ID>")
        return 2

    # For render/assemble phases we must have a job to resume
    if args.phase in ("render", "assemble") and not args.resume_job:
        print(f"error: --phase {args.phase} requires --resume-job <JOB_ID>")
        return 2

    if not args.rights_cleared:
        print(
            "\n[WARNING] --rights-cleared not set.\n"
            "The pipeline will be BLOCKED at the rights-check stage.\n"
            "Pass --rights-cleared to confirm the source is cleared for transformative use.\n"
        )

    # ── Config ────────────────────────────────────────────────────────────────
    from core.config import load_app_config
    from core.workflow import RunOptions

    config = load_app_config()

    if args.whisper_device:
        config.settings.whisper_device = args.whisper_device
        config.settings.whisper_compute_type = "float16" if args.whisper_device == "cuda" else "int8"

    if args.llm_model:
        config.settings.llm_model = args.llm_model

    if args.review_dir:
        config.settings.review_dir = args.review_dir

    # Convert local file paths to file:// URI so fetch_media skips yt-dlp
    source = args.url
    if source and os.path.exists(source):
        source = Path(source).resolve().as_uri()
        logger.info("Local file detected — using file:// URI: %s", source)

    options = RunOptions(
        phase=args.phase,
        resume=bool(args.resume_job),
        only_shot=args.only_shot,
    )

    print_banner(source, args.critique, args.rights_cleared,
                 args.resume_job, args.phase, args.only_shot)

    # ── Run ───────────────────────────────────────────────────────────────────
    start = time.perf_counter()

    try:
        if args.critique:
            from core.workflow import run_with_critique
            logger.info("Starting Phase 2 critique pipeline ...")
            job = await run_with_critique(
                source_url=source or "",
                rights_cleared=args.rights_cleared,
                app_config=config,
            )
        else:
            from core.workflow import run_pipeline_job
            phase_label = f"phase={args.phase}" if args.phase != "all" else "full pipeline"
            resume_label = f" (resuming {args.resume_job})" if args.resume_job else ""
            logger.info("Starting Phase 1 pipeline — %s%s ...", phase_label, resume_label)
            job = await run_pipeline_job(
                source_url=source or "",
                rights_cleared=args.rights_cleared,
                app_config=config,
                options=options,
                resume_job_id=args.resume_job,
                target_language=args.target_language,
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
