#!/usr/bin/env python3
"""
Standalone video downloader for the video_me pipeline.

Downloads a video from YouTube, Instagram, TikTok, or any yt-dlp-supported
source to a local file. Run this first, then pass the local path to
run_pipeline.py — decouples network/auth from the pipeline itself.

Usage:
    python download_video.py <URL>
    python download_video.py <URL> --output-dir /workspace/downloads/
    python download_video.py <URL> --cookies-from-browser chrome
    python download_video.py <URL> --cookies /path/to/cookies.txt

Cookie options (for age-gated / login-required / geo-restricted videos):
    --cookies-from-browser chrome     export cookies live from Chrome
    --cookies-from-browser firefox    export cookies live from Firefox
    --cookies-from-browser edge       export cookies live from Edge
    --cookies /path/to/cookies.txt    Netscape-format cookies file

After download, run the pipeline:
    python run_pipeline.py /workspace/downloads/<title>.mp4 --rights-cleared
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("download_video")

DEFAULT_OUTPUT_DIR = Path("/workspace/downloads")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a video for the video_me pipeline.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("url", help="YouTube / Instagram / TikTok / any yt-dlp URL")
    p.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save the video (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--cookies-from-browser", metavar="BROWSER", help="Export cookies from a running browser (chrome, firefox, edge, safari)")
    p.add_argument("--cookies", metavar="FILE", type=Path, help="Netscape-format cookies.txt file (for login-required content)")
    p.add_argument("--format", default="bestvideo+bestaudio/best", help="yt-dlp format string (default: best quality)")
    p.add_argument("--info", action="store_true", help="Print video info only, do not download")
    return p.parse_args()


def _slug(title: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_len]


def _base_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["yt-dlp", "--js-runtimes", "node", "--no-playlist"]
    if args.cookies_from_browser:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    if args.cookies and args.cookies.exists():
        cmd += ["--cookies", str(args.cookies)]
    return cmd


def get_info(args: argparse.Namespace) -> dict:
    cmd = _base_cmd(args) + ["--dump-json", "--skip-download", args.url]
    logger.info("Fetching video info...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        logger.error("yt-dlp info failed:\n%s", result.stderr[-1000:])
        raise RuntimeError(f"Could not fetch video info: {result.stderr[-200:]}")
    return json.loads(result.stdout)


def download(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = get_info(args)
    title = info.get("title", "video")
    duration = info.get("duration", 0)
    logger.info("Title    : %s", title)
    logger.info("Uploader : %s", info.get("uploader", "unknown"))
    logger.info("Duration : %ds (%.1f min)", duration, duration / 60)
    if duration > 600:
        logger.warning("Video is %dm — ideal pipeline input is 1-3 min; longer videos take much more GPU time.", duration // 60)
    slug = _slug(title)
    output_template = str(args.output_dir / f"{slug}.%(ext)s")
    cmd = _base_cmd(args) + ["--format", args.format, "--merge-output-format", "mp4", "--output", output_template, "--no-overwrites", args.url]
    logger.info("Downloading to %s/", args.output_dir)
    result = subprocess.run(cmd, timeout=600)
    if result.returncode != 0:
        raise RuntimeError("yt-dlp download failed — see output above")
    mp4s = sorted(args.output_dir.glob(f"{slug}*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise RuntimeError(f"Download completed but no .mp4 found in {args.output_dir}")
    out = mp4s[0]
    logger.info("Saved    : %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def main() -> int:
    args = parse_args()
    if not shutil.which("yt-dlp"):
        logger.error("yt-dlp not found — install with: pip install yt-dlp")
        return 1
    if args.info:
        try:
            info = get_info(args)
            print(f"\nTitle    : {info.get('title')}")
            print(f"Uploader : {info.get('uploader')}")
            print(f"Duration : {info.get('duration', 0)}s ({info.get('duration', 0) // 60}m)")
            print(f"ID       : {info.get('id')}")
            return 0
        except Exception as e:
            logger.error("%s", e)
            return 1
    try:
        out = download(args)
        print()
        print("=" * 64)
        print("  Download complete!")
        print(f"  File : {out}")
        print()
        print("  Next — run the pipeline:")
        print(f"    python run_pipeline.py '{out}' --rights-cleared --whisper-device cuda")
        print("=" * 64)
        return 0
    except Exception as e:
        logger.error("Download failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
