"""Validation tests for the story/story_images additions to core/models/dashboard.py."""
import pytest
from pydantic import ValidationError

from core.models.dashboard import CreateDashboardJobRequest, DashboardSource


# ------------------------------------------------------------------ DashboardSource


def test_url_kind_requires_url() -> None:
    with pytest.raises(ValidationError, match="url is required"):
        DashboardSource(kind="url", url="  ")


def test_file_kind_requires_url() -> None:
    with pytest.raises(ValidationError, match="url is required"):
        DashboardSource(kind="file", url="")


def test_story_kind_defaults_placeholder_url() -> None:
    src = DashboardSource(kind="story")
    assert src.url == "story://direct-input"


def test_story_images_kind_defaults_placeholder_url() -> None:
    src = DashboardSource(kind="story_images", url="")
    assert src.url.startswith("story://")


# ------------------------------------------------------------------ CreateDashboardJobRequest


def _story_request(**kwargs) -> CreateDashboardJobRequest:
    defaults = dict(
        source=DashboardSource(kind="story"),
        rights_cleared=True,
        story_text="0-5: Max finds a seed.",
    )
    defaults.update(kwargs)
    return CreateDashboardJobRequest(**defaults)


def test_story_request_valid() -> None:
    req = _story_request()
    assert req.source.kind == "story"
    assert req.character_images == {}


def test_story_requires_story_text() -> None:
    with pytest.raises(ValidationError, match="story_text is required"):
        _story_request(story_text=None)
    with pytest.raises(ValidationError, match="story_text is required"):
        _story_request(story_text="   ")


def test_story_images_requires_images() -> None:
    with pytest.raises(ValidationError, match="at least one character image"):
        _story_request(source=DashboardSource(kind="story_images"))


def test_story_images_valid_with_images() -> None:
    req = _story_request(
        source=DashboardSource(kind="story_images"),
        character_images={"max": "/data/uploads/abc/max.png"},
    )
    assert req.character_images["max"].endswith("max.png")


def test_legacy_url_payload_still_parses() -> None:
    """Old queue payloads (pre story fields) must keep deserializing."""
    req = CreateDashboardJobRequest(
        **{
            "source": {"kind": "url", "url": "https://example.com/v"},
            "rights_cleared": True,
            "phase": "all",
        }
    )
    assert req.story_text is None
    assert req.character_images == {}
    assert req.audio_profile == "auto"


def test_audio_profile_accepts_operator_hint() -> None:
    req = CreateDashboardJobRequest(
        source=DashboardSource(kind="file", url="file:///tmp/source.mp4"),
        rights_cleared=True,
        audio_profile="singing",
    )
    assert req.audio_profile == "singing"


def test_gpu_price_per_hour_defaults_to_zero() -> None:
    req = CreateDashboardJobRequest(
        source=DashboardSource(kind="file", url="file:///tmp/source.mp4"),
        rights_cleared=True,
    )

    assert req.gpu_price_per_hour == 0.0


def test_gpu_price_per_hour_rejects_negative_value() -> None:
    with pytest.raises(ValidationError):
        CreateDashboardJobRequest(
            source=DashboardSource(kind="file", url="file:///tmp/source.mp4"),
            rights_cleared=True,
            gpu_price_per_hour=-1,
        )


def test_audio_profile_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        CreateDashboardJobRequest(
            source=DashboardSource(kind="file", url="file:///tmp/source.mp4"),
            rights_cleared=True,
            audio_profile="podcast",
        )


def test_url_request_ignores_story_validators() -> None:
    req = CreateDashboardJobRequest(
        source=DashboardSource(kind="url", url="https://example.com/v"),
        rights_cleared=True,
    )
    assert req.source.url == "https://example.com/v"


def test_wan_animate_story_requires_separate_driver() -> None:
    with pytest.raises(ValidationError, match="uploaded or local driver"):
        _story_request(overrides={"video_adapter": "wan_animate"})


def test_wan_animate_story_accepts_uploaded_driver() -> None:
    req = _story_request(overrides={
        "video_adapter": "wan_animate",
        "wan_animate_driver_source": "upload",
        "wan_animate_driver_uri": "/data/uploads/driver.mp4",
        "wan_animate_timeline": "sequential",
    })
    assert req.overrides.wan_animate_mode is None


def test_wan_animate_replace_rejects_retargeting() -> None:
    with pytest.raises(ValidationError, match="does not support pose retargeting"):
        CreateDashboardJobRequest(
            source=DashboardSource(kind="file", url="file:///tmp/source.mp4"),
            rights_cleared=True,
            overrides={
                "video_adapter": "wan_animate",
                "wan_animate_mode": "replace",
                "wan_animate_retarget_pose": True,
            },
        )
