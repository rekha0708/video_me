from pathlib import Path

import pytest

from core.models.capabilities import FetchMediaResult, PreparedWanAnimateInput
from core.models.content import Shot
from core.workflow import _prepare_wan_animate_inputs


def _shot(shot_id: str, duration: float, start=None, end=None) -> Shot:
    return Shot(
        shot_id=shot_id, scene_ref="scene", characters_on_screen=["host"],
        setting="stage", camera="medium", action="waves", duration_sec=duration,
        source_start_sec=start, source_end_sec=end,
    )


class _FakeAnimateAdapter:
    driver_source = "job_source"
    driver_uri = ""
    timeline = "sequential"
    mode = "animate"

    async def prepare_inputs(self, requests):
        self.requests = requests
        return {
            request.shot_id: PreparedWanAnimateInput(
                shot_id=request.shot_id,
                prepared_dir=f"/data/{request.shot_id}",
                driver_uri=request.driver.uri,
                start_sec=request.driver.start_sec,
                end_sec=request.driver.end_sec,
                frame_count=60, fps=30, width=720, height=1280,
            )
            for request in requests
        }


async def test_sequential_timeline_is_stable_across_shots() -> None:
    adapter = _FakeAnimateAdapter()
    result = await _prepare_wan_animate_inputs(
        shots=[_shot("s01", 2.0), _shot("s02", 3.0)],
        image_uris=["/data/a.png", "/data/b.png"],
        adapter=adapter,
        fetch_result=FetchMediaResult(
            video_uri="/data/driver.mp4", audio_uri="/data/audio.wav",
            duration_sec=5.0, source_url="file:///data/driver.mp4",
        ),
        stage_hook=None,
    )
    assert (result["s01"].start_sec, result["s01"].end_sec) == (0.0, 2.0)
    assert (result["s02"].start_sec, result["s02"].end_sec) == (2.0, 5.0)
    assert result["s02"].prepared_dir == "/data/s02"


async def test_source_timeline_requires_shot_timestamps() -> None:
    adapter = _FakeAnimateAdapter()
    adapter.timeline = "source_timestamps"
    with pytest.raises(ValueError, match="no source timestamps"):
        await _prepare_wan_animate_inputs(
            shots=[_shot("s01", 2.0)], image_uris=["/data/a.png"], adapter=adapter,
            fetch_result=FetchMediaResult(
                video_uri="/data/driver.mp4", audio_uri="/data/audio.wav",
                duration_sec=2.0, source_url="file:///data/driver.mp4",
            ),
            stage_hook=None,
        )


async def test_story_driver_must_be_supplied() -> None:
    adapter = _FakeAnimateAdapter()
    with pytest.raises(ValueError, match="must upload or select"):
        await _prepare_wan_animate_inputs(
            shots=[_shot("s01", 2.0)], image_uris=["/data/a.png"], adapter=adapter,
            fetch_result=FetchMediaResult(
                video_uri="story://job", audio_uri="story://job",
                duration_sec=2.0, source_url="story://direct-input",
            ),
            stage_hook=None,
        )
