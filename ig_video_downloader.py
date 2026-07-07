#!/usr/bin/env python3
"""
Authenticated Instagram video downloader for video_me.

Best option for login-required posts/reels:
    python ig_video_downloader.py "https://www.instagram.com/reel/..." \
      --cookies-from-browser chrome

If you use a non-default Chrome profile:
    python ig_video_downloader.py "https://www.instagram.com/reel/..." \
      --cookies-from-browser chrome --browser-profile "Profile 1"

Other auth options:
    python ig_video_downloader.py "https://www.instagram.com/reel/..." \
      --cookies /path/to/cookies.txt

Best quality may require ffmpeg because Instagram often serves video and audio
as separate streams:
    python ig_video_downloader.py "https://www.instagram.com/reel/..." \
      --cookies-from-browser chrome --best-quality

Instagram username/password login is not supported by yt-dlp. Use browser cookies
from an already-approved Instagram session instead.
Only download content you have rights or permission to use.
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ig_video_downloader")

DEFAULT_OUTPUT_DIR = Path("/Users/vryadav/git/video_me/downloads/instagram")
INSTAGRAM_HOST_RE = re.compile(r"^https?://(?:www\.)?instagram\.com/", re.IGNORECASE)
SINGLE_FILE_FORMAT = "best[ext=mp4]/best"
BEST_QUALITY_FORMAT = "bestvideo+bestaudio/best"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Instagram reels/posts/videos with optional login.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", help="Instagram reel/post/video URL")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the video (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Use cookies from an installed browser, e.g. chrome, firefox, edge, safari",
    )
    parser.add_argument(
        "--browser-profile",
        metavar="PROFILE",
        help='Browser profile name/path for cookie auth, e.g. "Default" or "Profile 1"',
    )
    parser.add_argument(
        "--cookies",
        metavar="FILE",
        type=Path,
        help="Netscape-format cookies.txt file exported from a logged-in browser session",
    )
    parser.add_argument(
        "--username",
        help="Deprecated for Instagram. Ignored when cookie auth is used.",
    )
    parser.add_argument("--password-env", help="Unsupported for Instagram; use cookies instead")
    parser.add_argument(
        "--ask-password",
        action="store_true",
        help="Unsupported for Instagram; use cookies instead",
    )
    parser.add_argument("--twofactor", help="Unsupported for Instagram; use cookies instead")
    parser.add_argument("--netrc", action="store_true", help="Let yt-dlp read credentials from ~/.netrc")
    parser.add_argument(
        "--format",
        default=None,
        help=f"yt-dlp format string (default: {SINGLE_FILE_FORMAT})",
    )
    parser.add_argument(
        "--best-quality",
        action="store_true",
        help="Download best video+audio and merge with ffmpeg. Requires ffmpeg.",
    )
    parser.add_argument("--info", action="store_true", help="Print video info only, do not download")
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose to yt-dlp for troubleshooting")
    return parser.parse_args()


def _slug(value: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^\w\s-]", "", value.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_len] or "instagram_video"


def _download_format(args: argparse.Namespace) -> str:
    if args.format:
        return args.format
    if args.best_quality:
        return BEST_QUALITY_FORMAT
    return SINGLE_FILE_FORMAT


def _format_needs_ffmpeg(format_selector: str) -> bool:
    return "+" in format_selector


def _check_ffmpeg(format_selector: str) -> None:
    if _format_needs_ffmpeg(format_selector) and not shutil.which("ffmpeg"):
        raise RuntimeError(
            "This format downloads separate video/audio streams and needs ffmpeg to merge them. "
            "Install ffmpeg, or rerun without --best-quality to prefer a single Instagram MP4 stream. "
            "On macOS: brew install ffmpeg"
        )


def _has_audio_stream(path: Path) -> bool | None:
    if not shutil.which("ffprobe"):
        logger.warning("ffprobe not found; skipping audio-stream verification.")
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        logger.warning("Could not verify audio stream with ffprobe: %s", result.stderr.strip())
        return None
    return "audio" in result.stdout


def _base_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["yt-dlp", "--no-playlist", "--no-progress"]

    if args.cookies_from_browser:
        browser = args.cookies_from_browser
        if args.browser_profile and ":" not in browser:
            browser = f"{browser}:{args.browser_profile}"
        cmd += ["--cookies-from-browser", browser]
    if args.cookies:
        if not args.cookies.exists():
            raise RuntimeError(f"Cookies file does not exist: {args.cookies}")
        cmd += ["--cookies", str(args.cookies)]
    if args.netrc:
        cmd.append("--netrc")
    if args.verbose:
        cmd.append("--verbose")

    return cmd


def _instagram_error_hint(stderr: str, args: argparse.Namespace) -> str:
    if "empty media response" not in stderr.lower():
        return ""

    lines = [
        "",
        "Instagram returned an empty media response for this session.",
        "Try these in order:",
        "  1. Open the same post in Chrome and confirm it plays while logged in.",
        "  2. Use the exact Chrome profile that is logged into Instagram:",
        '       python ig_video_downloader.py "<url>" --cookies-from-browser chrome --browser-profile "Default"',
        '       python ig_video_downloader.py "<url>" --cookies-from-browser chrome --browser-profile "Profile 1"',
        "  3. Close Chrome completely and retry, or export cookies.txt from the logged-in profile:",
        '       python ig_video_downloader.py "<url>" --cookies ~/Downloads/cookies.txt',
        "  4. Update yt-dlp inside this venv:",
        "       python -m pip install -U yt-dlp",
    ]
    if args.cookies_from_browser and not args.browser_profile:
        lines.insert(4, f"     Current browser auth was: --cookies-from-browser {args.cookies_from_browser}")
    return "\n".join(lines)


def get_info(args: argparse.Namespace) -> dict:
    cmd = _base_cmd(args) + ["--dump-json", "--skip-download", args.url]
    logger.info("Fetching Instagram video info...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        hint = _instagram_error_hint(result.stderr, args)
        raise RuntimeError(f"yt-dlp info failed:\n{result.stderr[-1200:]}{hint}")
    return json.loads(result.stdout)


def download(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = get_info(args)
    format_selector = _download_format(args)
    _check_ffmpeg(format_selector)

    title = info.get("title") or info.get("description") or info.get("id") or "instagram_video"
    video_id = info.get("id")
    slug = _slug(title)
    output_template = str(args.output_dir / f"{slug}.%(id)s.%(ext)s")
    expected_output = args.output_dir / f"{slug}.{video_id}.mp4" if video_id else None

    logger.info("Title    : %s", title)
    logger.info("Uploader : %s", info.get("uploader") or info.get("channel") or "unknown")
    if info.get("duration"):
        logger.info("Duration : %ss", info["duration"])

    cmd = _base_cmd(args) + [
        "--format",
        format_selector,
        "--merge-output-format",
        "mp4",
        "--output",
        output_template,
        "--force-overwrites",
        args.url,
    ]
    logger.info("Downloading to %s/", args.output_dir)
    result = subprocess.run(cmd, timeout=600)
    if result.returncode != 0:
        raise RuntimeError("yt-dlp download failed; see output above")

    fragment_mp4s = sorted(args.output_dir.glob(f"{slug}.*.fdash-*.mp4"))
    merged_candidates = []
    if expected_output and expected_output.exists():
        merged_candidates.append(expected_output)
    merged_candidates.extend(
        path
        for path in sorted(args.output_dir.glob(f"{slug}.*.mp4"), key=lambda item: item.stat().st_mtime, reverse=True)
        if path not in merged_candidates and ".fdash-" not in path.name
    )

    if not merged_candidates:
        raise RuntimeError(f"Download completed but no .mp4 found in {args.output_dir}")

    audio_parts = sorted(args.output_dir.glob(f"{slug}.*.m4a"), key=lambda path: path.stat().st_mtime, reverse=True)
    if audio_parts and _format_needs_ffmpeg(format_selector):
        video_part = fragment_mp4s[-1] if fragment_mp4s else None
        merge_hint = ""
        if video_part:
            merge_hint = (
                "\nManual merge command:\n"
                f"  ffmpeg -y -i '{video_part}' -i '{audio_parts[0]}' -c copy '{expected_output or merged_candidates[0]}'"
            )
        raise RuntimeError(
            "yt-dlp left separate video/audio files, which means merging did not complete. "
            "Install ffmpeg and rerun, or rerun without --best-quality for a single MP4 stream."
            f"{merge_hint}"
        )

    out = merged_candidates[0]
    has_audio = _has_audio_stream(out)
    if has_audio is False:
        raise RuntimeError(
            f"Downloaded MP4 has no audio stream: {out}\n"
            "If you used --best-quality, rerun after clearing old fragments or use the manual merge command shown by ffmpeg/yt-dlp."
        )
    return out


def validate_args(args: argparse.Namespace) -> None:
    if not INSTAGRAM_HOST_RE.match(args.url):
        raise RuntimeError("URL must start with https://www.instagram.com/ or https://instagram.com/")
    if args.password_env and args.ask_password:
        raise RuntimeError("Use either --password-env or --ask-password, not both")
    if args.password_env or args.ask_password or args.twofactor:
        raise RuntimeError("Instagram username/password login is not supported. Use --cookies-from-browser or --cookies.")
    if args.browser_profile and not args.cookies_from_browser:
        raise RuntimeError("--browser-profile requires --cookies-from-browser.")
    if args.username and not (args.cookies_from_browser or args.cookies):
        raise RuntimeError("Instagram username login is not supported. Use --cookies-from-browser or --cookies.")
    if args.username:
        logger.warning("Ignoring --username because Instagram auth must use cookies.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        if not shutil.which("yt-dlp"):
            raise RuntimeError("yt-dlp not found. Install it with: python -m pip install 'yt-dlp>=2024.1'")

        if args.info:
            info = get_info(args)
            print(f"Title    : {info.get('title')}")
            print(f"Uploader : {info.get('uploader') or info.get('channel')}")
            print(f"Duration : {info.get('duration', 0)}s")
            print(f"ID       : {info.get('id')}")
            return 0

        out = download(args)
        logger.info("Saved    : %s (%.1f MB)", out, out.stat().st_size / 1e6)
        print()
        print("=" * 64)
        print("  Instagram download complete")
        print(f"  File : {out}")
        print()
        print("  Next:")
        print(f"    python run_pipeline.py '{out}' --rights-cleared --whisper-device cuda")
        print("=" * 64)
        return 0
    except Exception as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
