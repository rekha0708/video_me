import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.generate_video.wan_animate_adapter import WanAnimateAdapter
from core.models.capabilities import PreparedWanAnimateInput, VideoDriver, VideoRequest


def _adapter(tmp_path: Path, **overrides) -> WanAnimateAdapter:
    kwargs = {
        "work_dir": tmp_path / "jobs" / "j1" / "video" / "wan_animate",
        "base_url": "http://localhost:8033",
        "python_bin": "/workspace/.venv_wan_animate/bin/python",
        "wan_dir": Path("/workspace/Wan2.2"),
        "model_dir": Path("/workspace/Wan2.2-Animate-14B"),
    }
    kwargs.update(overrides)
    return WanAnimateAdapter(**kwargs)


def _request(tmp_path: Path, *, end: float = 2.0) -> VideoRequest:
    image = tmp_path / "ref.png"
    image.write_bytes(b"png")
    driver = tmp_path / "driver.mp4"
    driver.write_bytes(b"mp4")
    return VideoRequest(
        image_uri=str(image), action="waves", duration_sec=end,
        shot_id="s01", setting="stage",
        driver=VideoDriver(uri=str(driver), start_sec=0, end_sec=end),
    )


def test_mode_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        _adapter(tmp_path, mode="bad")
    with pytest.raises(ValueError, match="requires pose"):
        _adapter(tmp_path, use_flux_retarget=True)
    with pytest.raises(ValueError, match="only in animate"):
        _adapter(tmp_path, mode="replace", retarget_pose=True)


async def test_prepare_rejects_driver_that_is_too_short(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._probe_duration = AsyncMock(return_value=1.0)
    with pytest.raises(ValueError, match="too short"):
        await adapter.prepare_inputs([_request(tmp_path, end=2.0)])


async def test_run_posts_prepared_directory(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, mode="replace", refert_num=5)
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    adapter._prepared["s01"] = PreparedWanAnimateInput(
        shot_id="s01", prepared_dir=str(prepared_dir), driver_uri=str(tmp_path / "driver.mp4"),
        start_sec=0, end_sec=2, frame_count=60, fps=30, width=720, height=1280,
    )
    response = MagicMock(status_code=200, content=b"mp4")
    response.raise_for_status = MagicMock()
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(return_value=response)
    fake_httpx = MagicMock(AsyncClient=MagicMock(return_value=client))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await adapter.run(_request(tmp_path))

    data = client.post.call_args.kwargs["data"]
    assert data["mode"] == "replace"
    assert data["refert_num"] == "5"
    assert data["prepared_dir"] == str(prepared_dir)
    assert Path(clip.uri).read_bytes() == b"mp4"


def test_cached_manifest_requires_all_replace_artifacts(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, mode="replace")
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared = PreparedWanAnimateInput(
        shot_id="s01", prepared_dir=str(prepared_dir), driver_uri="driver.mp4",
        start_sec=0, end_sec=2, frame_count=60, fps=30, width=720, height=1280,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cache_key": "key", "prepared": prepared.model_dump()}))
    assert adapter._read_cached(manifest, "key") is None
    for name in ("src_ref.png", "src_pose.mp4", "src_face.mp4", "src_bg.mp4", "src_mask.mp4"):
        (prepared_dir / name).write_bytes(b"x")
    assert adapter._read_cached(manifest, "key") is not None
