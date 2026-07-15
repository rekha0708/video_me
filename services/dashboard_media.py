"""Safe media ingestion helpers for dashboard-owned opaque assets."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
from pathlib import Path
import socket
import stat
from typing import Any
from urllib.parse import ParseResult, urljoin, urlparse


class MediaIngestError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def stream_upload(upload: Any, destination: Path, *, max_bytes: int) -> int:
    """Stream an UploadFile-like object to disk with a hard byte limit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise MediaIngestError(
                        "FILE_TOO_LARGE",
                        f"Upload exceeds the {max_bytes / (1024 ** 2):.0f} MiB limit",
                    )
                handle.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if size == 0:
        destination.unlink(missing_ok=True)
        raise MediaIngestError("EMPTY_FILE", "Uploaded file is empty")
    return size


def copy_local_file_limited(source: Path, destination: Path, *, max_bytes: int) -> int:
    """Copy one validated server file through a no-follow descriptor with a hard cap."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, flags)
    size = 0
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise MediaIngestError("INVALID_SERVER_FILE", "Server video is not a regular file")
        if source_stat.st_size > max_bytes:
            raise MediaIngestError("FILE_TOO_LARGE", "Server video exceeds 2 GiB")
        with os.fdopen(source_fd, "rb", closefd=False) as incoming, destination.open("wb") as outgoing:
            while chunk := incoming.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise MediaIngestError("FILE_TOO_LARGE", "Server video exceeds 2 GiB")
                outgoing.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
    if size == 0:
        destination.unlink(missing_ok=True)
        raise MediaIngestError("EMPTY_FILE", "Server video is empty")
    return size


def _fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        try:
            return float(value)
        except ValueError:
            return 0.0


async def probe_video(
    path: Path,
    *,
    ffprobe_bin: str,
    max_duration_sec: float = 600.0,
    min_short_side: int = 256,
    max_long_side: int = 8192,
    max_fps: float = 240.0,
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """Decode/probe a driver and return normalized client-safe metadata."""
    process = await asyncio.create_subprocess_exec(
        ffprobe_bin,
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe,crypto,data",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate:"
        "format=duration,format_name",
        "-of",
        "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_sec
        )
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise MediaIngestError(
            "VIDEO_PROBE_TIMEOUT", "Video metadata inspection timed out"
        ) from exc
    if process.returncode != 0:
        raise MediaIngestError(
            "UNREADABLE_VIDEO",
            "Video could not be decoded: " + stderr.decode(errors="replace")[-500:],
        )
    try:
        body = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise MediaIngestError("UNREADABLE_VIDEO", "FFprobe returned invalid metadata") from exc
    streams = body.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise MediaIngestError("INVALID_VIDEO_STREAMS", "Video must contain exactly one video stream")
    video = video_streams[0]
    try:
        duration = float((body.get("format") or {}).get("duration") or 0.0)
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
    except (TypeError, ValueError) as exc:
        raise MediaIngestError("UNREADABLE_VIDEO", "Video metadata is incomplete") from exc
    if duration <= 0 or duration > max_duration_sec:
        raise MediaIngestError(
            "INVALID_DURATION",
            f"Video duration must be greater than 0 and at most {max_duration_sec / 60:.0f} minutes",
        )
    if min(width, height) < min_short_side:
        raise MediaIngestError(
            "VIDEO_TOO_SMALL", f"Video's shorter side must be at least {min_short_side} pixels"
        )
    if max(width, height) > max_long_side:
        raise MediaIngestError(
            "VIDEO_TOO_LARGE",
            f"Video dimensions must not exceed {max_long_side} pixels on either side",
        )
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    if fps <= 0 or fps > max_fps:
        raise MediaIngestError(
            "INVALID_FRAME_RATE",
            f"Video frame rate must be greater than 0 and at most {max_fps:g} FPS",
        )
    return {
        "duration_sec": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3) if fps else None,
        "codec": video.get("codec_name") or "unknown",
        "container": (body.get("format") or {}).get("format_name") or "unknown",
        "has_audio": bool(audio_streams),
        "audio_stream_count": len(audio_streams),
        "size_bytes": path.stat().st_size,
        "recommended_duration": duration <= 10.0,
        "estimated_77_frame_chunks": max(1, int((duration * 30 + 76) // 77)),
    }


def normalize_image(
    source: Path,
    destination: Path,
    *,
    max_pixels: int = 50_000_000,
    min_short_side: int = 128,
) -> dict[str, Any]:
    """Decode, orient, RGB-normalize, and strip metadata into a canonical PNG."""
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - dependency/readiness failure
        raise RuntimeError('Image uploads require Pillow; install the "dashboard" extra') from exc

    try:
        with Image.open(source) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise MediaIngestError("ANIMATED_IMAGE", "Animated images are not supported")
            image.verify()
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise MediaIngestError("IMAGE_TOO_LARGE", "Image dimensions exceed the pixel limit")
            if min(width, height) < min_short_side:
                raise MediaIngestError(
                    "IMAGE_TOO_SMALL",
                    f"Image's shorter side must be at least {min_short_side} pixels",
                )
            normalized = image.convert("RGB")
            destination.parent.mkdir(parents=True, exist_ok=True)
            normalized.save(destination, format="PNG", optimize=True)
    except MediaIngestError:
        destination.unlink(missing_ok=True)
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise MediaIngestError("UNREADABLE_IMAGE", "Image could not be decoded") from exc
    return {
        "width": width,
        "height": height,
        "format": "png",
        "size_bytes": destination.stat().st_size,
    }


async def _assert_public_http_url(url: str) -> tuple[ParseResult, list[str]]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MediaIngestError("INVALID_URL", "Only public HTTP(S) URLs are supported")
    if parsed.username or parsed.password:
        raise MediaIngestError("INVALID_URL", "URLs containing credentials are not supported")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise MediaIngestError("INVALID_URL", "Video URL contains an invalid port") from exc
    loop = asyncio.get_running_loop()
    try:
        addresses = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM),
        )
    except socket.gaierror as exc:
        raise MediaIngestError("URL_UNREACHABLE", "Video URL host could not be resolved") from exc
    if not addresses:
        raise MediaIngestError("URL_UNREACHABLE", "Video URL host has no addresses")
    public_addresses: set[str] = set()
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%", 1)[0])
        if not ip.is_global:
            raise MediaIngestError("URL_NOT_PUBLIC", "Private or local network URLs are blocked")
        public_addresses.add(str(ip))
    return parsed, sorted(
        public_addresses,
        key=lambda value: (ipaddress.ip_address(value).version, value),
    )


def _pinned_url(parsed: ParseResult, address: str) -> str:
    """Replace only the connection host while retaining path/query semantics."""

    host = f"[{address}]" if ipaddress.ip_address(address).version == 6 else address
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    return parsed._replace(netloc=netloc).geturl()


def _original_host_header(parsed: ParseResult) -> str:
    assert parsed.hostname is not None
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    default_port = 443 if parsed.scheme == "https" else 80
    return f"{host}:{parsed.port}" if parsed.port not in {None, default_port} else host


async def download_public_video_url(
    url: str,
    destination: Path,
    *,
    max_bytes: int = 2 * 1024 * 1024 * 1024,
    max_redirects: int = 5,
) -> tuple[int, str, str]:
    """Download a direct public video URL while re-checking every redirect."""
    import httpx

    current = url.strip()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": "video-me-dashboard/1.0"},
        ) as client:
            for redirect_count in range(max_redirects + 1):
                parsed, addresses = await _assert_public_http_url(current)
                # Pin the socket to the exact public address we validated. TLS
                # still verifies/SNI-routes against the original hostname and
                # HTTP receives the original Host header. This closes the DNS
                # rebinding window between validation and httpx connection.
                connect_url = _pinned_url(parsed, addresses[0])
                async with client.stream(
                    "GET",
                    connect_url,
                    headers={"Host": _original_host_header(parsed)},
                    extensions={"sni_hostname": parsed.hostname},
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or redirect_count >= max_redirects:
                            raise MediaIngestError("TOO_MANY_REDIRECTS", "Video URL redirected too many times")
                        current = urljoin(current, location)
                        continue
                    if response.status_code >= 400:
                        raise MediaIngestError(
                            "URL_DOWNLOAD_FAILED",
                            f"Video URL returned HTTP {response.status_code}",
                        )
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if content_type.startswith("text/") or content_type in {
                        "application/xhtml+xml",
                        "application/vnd.apple.mpegurl",
                        "application/x-mpegurl",
                    }:
                        raise MediaIngestError(
                            "NOT_DIRECT_VIDEO_URL",
                            "URL is not a direct video file; web pages and streaming playlists are unsupported",
                        )
                    declared = response.headers.get("content-length")
                    if declared:
                        try:
                            if int(declared) > max_bytes:
                                raise MediaIngestError("FILE_TOO_LARGE", "Remote video exceeds 2 GiB")
                        except ValueError:
                            pass
                    size = 0
                    with destination.open("wb") as handle:
                        async for chunk in response.aiter_bytes(8 * 1024 * 1024):
                            size += len(chunk)
                            if size > max_bytes:
                                raise MediaIngestError("FILE_TOO_LARGE", "Remote video exceeds 2 GiB")
                            handle.write(chunk)
                    if size == 0:
                        raise MediaIngestError("EMPTY_FILE", "Remote video is empty")
                    return size, content_type or "application/octet-stream", current
    except httpx.HTTPError as exc:
        destination.unlink(missing_ok=True)
        raise MediaIngestError(
            "URL_UNREACHABLE", "Video URL could not be downloaded"
        ) from exc
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    raise MediaIngestError("URL_DOWNLOAD_FAILED", "Video URL could not be downloaded")


__all__ = [
    "MediaIngestError",
    "copy_local_file_limited",
    "download_public_video_url",
    "normalize_image",
    "probe_video",
    "stream_upload",
]
