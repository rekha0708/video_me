"""Phase A cross-shot batch rendering (_phase_a_prefetch_renders + prefetched)."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.models.capabilities import ImageCritiqueResult, ImageSet
from core.models.content import Shot
from core.models.profile import Cast, CastMember
from core.workflow import RunOptions, _phase_a_prefetch_renders, _render_shot_candidates


def _member() -> CastMember:
    return CastMember(
        id="fox",
        name="Roxy",
        gender="girl",
        visual_descriptor="cartoon red fox with a purple scarf",
        lora_ref="loras/solo_fox/fox",
        voice_profile_ref="voices/solo_fox/fox",
        personality="adventurous and curious",
        signature_expressions=["tail wag"],
    )


def _cast() -> Cast:
    return Cast(id="solo_fox", species="fox", is_original_synthetic=True,
                members=[_member()])


def _shot(shot_id: str) -> Shot:
    return Shot(
        shot_id=shot_id,
        scene_ref="scene-1",
        characters_on_screen=["fox"],
        setting="forest clearing",
        camera="medium shot",
        action="character peeks around a tree",
        dialogue_line_refs=["scene-1-line-0"],
        duration_sec=5.0,
    )


def _adapters(num_images: int = 1) -> MagicMock:
    adapters = MagicMock()
    adapters.render = AsyncMock()
    adapters.render._num_images = num_images
    return adapters


@pytest.mark.asyncio
async def test_prefetch_batches_all_pending_shots_into_one_run_many(tmp_path: Path) -> None:
    adapters = _adapters()
    image_sets = [
        ImageSet(member_id="fox", images=[f"/tmp/{sid}/render_00.png"])
        for sid in ("s01", "s02")
    ]
    adapters.render.run_many = AsyncMock(return_value=image_sets)

    # distinct actions → no dedup; both shots render
    shots = [
        _shot("s01").model_copy(update={"action": "peeks around a tree"}),
        _shot("s02").model_copy(update={"action": "waves at the camera"}),
    ]
    result = await _phase_a_prefetch_renders(shots, _cast(), adapters, tmp_path)

    adapters.render.run_many.assert_awaited_once()
    requests = adapters.render.run_many.await_args.args[0]
    assert [r.shot_id for r in requests] == ["s01", "s02"]
    assert set(result) == {"s01", "s02"}
    assert result["s01"].images == ["/tmp/s01/render_00.png"]


@pytest.mark.asyncio
async def test_prefetch_skips_resume_cached_shots(tmp_path: Path) -> None:
    adapters = _adapters()
    # s01 fully cached on disk; s02 needs rendering.
    cached = tmp_path / "renders" / "s01" / "fox"
    cached.mkdir(parents=True)
    (cached / "render_00.png").write_bytes(b"png")
    adapters.render.run_many = AsyncMock(
        return_value=[ImageSet(member_id="fox", images=["/tmp/s02/render_00.png"])]
    )

    result = await _phase_a_prefetch_renders(
        [_shot("s01"), _shot("s02")], _cast(), adapters, tmp_path,
        RunOptions(resume=True),
    )

    requests = adapters.render.run_many.await_args.args[0]
    assert [r.shot_id for r in requests] == ["s02"]
    assert set(result) == {"s02"}


@pytest.mark.asyncio
async def test_prefetch_noop_when_everything_cached(tmp_path: Path) -> None:
    adapters = _adapters()
    cached = tmp_path / "renders" / "s01" / "fox"
    cached.mkdir(parents=True)
    (cached / "render_00.png").write_bytes(b"png")
    adapters.render.run_many = AsyncMock()

    result = await _phase_a_prefetch_renders(
        [_shot("s01")], _cast(), adapters, tmp_path, RunOptions(resume=True)
    )

    assert result == {}
    adapters.render.run_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_render_shot_candidates_uses_prefetched_and_skips_render(
    tmp_path: Path,
) -> None:
    adapters = _adapters(num_images=2)
    adapters.render.run = AsyncMock()  # must NOT be called
    uris = ["/tmp/pref/render_00.png", "/tmp/pref/render_01.png"]
    critique = ImageCritiqueResult(
        winner_index=0, winner_uri=uris[0], candidate_uris=uris,
    )
    adapters.image_critique.run = AsyncMock(return_value=critique)

    prefetched = ImageSet(member_id="fox", images=uris)
    result = await _render_shot_candidates(
        _shot("s01"), _cast(), adapters, tmp_path, prefetched=prefetched
    )

    adapters.render.run.assert_not_awaited()
    critique_req = adapters.image_critique.run.await_args.args[0]
    assert critique_req.candidate_uris == uris
    assert result.winner_uri == uris[0]


@pytest.mark.asyncio
async def test_single_candidate_skips_vlm_critique(tmp_path: Path) -> None:
    """N=1 → no VLM call; auto-picked winner with origin='single', json persisted."""
    adapters = _adapters(num_images=1)
    adapters.render.run = AsyncMock(
        return_value=ImageSet(member_id="fox", images=["/tmp/only/render_00.png"])
    )
    adapters.image_critique.run = AsyncMock()  # must NOT be called

    result = await _render_shot_candidates(_shot("s01"), _cast(), adapters, tmp_path)

    adapters.image_critique.run.assert_not_awaited()
    assert result.winner_uri == "/tmp/only/render_00.png"
    assert result.origin == "single"
    # persisted for resume
    cached = (tmp_path / "critique" / "s01.json").read_text()
    assert "single candidate" in cached


def _shot_with_action(shot_id: str, action: str) -> Shot:
    shot = _shot(shot_id)
    return shot.model_copy(update={"action": action})


@pytest.mark.asyncio
async def test_prefetch_dedups_identically_specified_shots(tmp_path: Path) -> None:
    """Same member+setting+camera+action → one render, PNGs copied to the dup."""
    adapters = _adapters()

    async def fake_run_many(requests):
        sets = []
        for req in requests:
            d = tmp_path / "renders" / req.shot_id / req.member.id
            d.mkdir(parents=True, exist_ok=True)
            p = d / "render_00.png"
            p.write_bytes(b"png")
            sets.append(ImageSet(member_id=req.member.id, images=[str(p)]))
        return sets

    adapters.render.run_many = AsyncMock(side_effect=fake_run_many)

    same = "waves at the camera"
    result = await _phase_a_prefetch_renders(
        [_shot_with_action("s01", same), _shot_with_action("s02", same)],
        _cast(), adapters, tmp_path,
    )

    requests = adapters.render.run_many.await_args.args[0]
    assert len(requests) == 1  # deduped
    assert set(result) == {"s01", "s02"}
    dup_png = tmp_path / "renders" / "s02" / "fox" / "render_00.png"
    assert dup_png.exists()
    assert result["s02"].images == [str(dup_png)]


@pytest.mark.asyncio
async def test_prefetch_does_not_dedup_different_actions(tmp_path: Path) -> None:
    adapters = _adapters()

    async def fake_run_many(requests):
        sets = []
        for req in requests:
            d = tmp_path / "renders" / req.shot_id / req.member.id
            d.mkdir(parents=True, exist_ok=True)
            p = d / "render_00.png"
            p.write_bytes(b"png")
            sets.append(ImageSet(member_id=req.member.id, images=[str(p)]))
        return sets

    adapters.render.run_many = AsyncMock(side_effect=fake_run_many)

    result = await _phase_a_prefetch_renders(
        [_shot_with_action("s01", "waves"), _shot_with_action("s02", "jumps")],
        _cast(), adapters, tmp_path,
    )

    requests = adapters.render.run_many.await_args.args[0]
    assert [r.shot_id for r in requests] == ["s01", "s02"]  # both rendered
    assert requests[0].action == "waves" and requests[1].action == "jumps"
    assert set(result) == {"s01", "s02"}
