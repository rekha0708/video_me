from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from services.dashboard_media import (
    MediaIngestError,
    _assert_public_http_url,
    _original_host_header,
    _pinned_url,
)


@pytest.mark.asyncio
async def test_url_validation_rejects_any_private_dns_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def addresses(*_args, **_kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", addresses)

    with pytest.raises(MediaIngestError, match="Private or local"):
        await _assert_public_http_url("https://example.com/video.mp4")


@pytest.mark.asyncio
async def test_url_validation_returns_only_public_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def addresses(*_args, **_kwargs):
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1:248:1893:25c8:1946", 443, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", addresses)

    parsed, resolved = await _assert_public_http_url(
        "https://example.com:8443/path/video.mp4?token=x"
    )

    assert parsed.hostname == "example.com"
    assert resolved == ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]


def test_pinned_download_url_preserves_path_query_and_original_host_header() -> None:
    parsed = urlparse("https://video.example:8443/path/file.mp4?token=a%2Fb")

    assert _pinned_url(parsed, "93.184.216.34") == (
        "https://93.184.216.34:8443/path/file.mp4?token=a%2Fb"
    )
    assert _original_host_header(parsed) == "video.example:8443"
