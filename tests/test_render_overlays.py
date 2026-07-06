"""Tests for ShotOverlay model + MatplotlibOverlayAdapter (chart panel rendering)."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from adapters.render_overlays.matplotlib_adapter import MatplotlibOverlayAdapter
from core.models.capabilities import AssembleRequest, RenderOverlaysRequest
from core.models.content import Shot, ShotOverlay


# ------------------------------------------------------------------ ShotOverlay model


def test_bar_overlay_valid() -> None:
    ov = ShotOverlay(kind="bar", title="Apples vs Oranges", labels=["Apples", "Oranges"], values=[3, 5])
    assert ov.png_uri is None
    assert ov.duration_sec is None


def test_callout_needs_only_title() -> None:
    ov = ShotOverlay(kind="callout", title="5 a day!")
    assert ov.labels == []


def test_chart_label_value_mismatch_raises() -> None:
    with pytest.raises(ValidationError, match="matching values"):
        ShotOverlay(kind="bar", title="Bad", labels=["a"], values=[1, 2])


def test_chart_too_many_points_raises() -> None:
    with pytest.raises(ValidationError, match="2-6 labels"):
        ShotOverlay(kind="line", title="Bad", labels=[str(i) for i in range(7)], values=list(range(7)))


def test_pie_negative_values_raise() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        ShotOverlay(kind="pie", title="Bad", labels=["a", "b"], values=[1, -2])


def test_shot_without_overlay_back_compat() -> None:
    """Old plan_shots artifacts (no overlay key) must still validate."""
    shot = Shot.model_validate({
        "shot_id": "s01", "scene_ref": "scene-1", "characters_on_screen": ["max"],
        "setting": "kitchen", "camera": "medium", "action": "waves",
        "dialogue_line_refs": ["scene-1-line-0"], "duration_sec": 5.0,
    })
    assert shot.overlay is None


def test_assemble_request_overlays_default_empty() -> None:
    from core.models.capabilities import AudioTrack, VideoClip
    req = AssembleRequest(
        clips=[VideoClip(uri="/tmp/c.mp4", duration_sec=5.0)],
        audio=AudioTrack(uri="/tmp/a.wav", duration_sec=5.0),
        caption_text="hi",
    )
    assert req.overlays == []


# ------------------------------------------------------------------ adapter (matplotlib required)


def _shot(shot_id: str, overlay: ShotOverlay | None) -> SimpleNamespace:
    return SimpleNamespace(shot_id=shot_id, overlay=overlay)


def _adapter(tmp_path: Path) -> MatplotlibOverlayAdapter:
    return MatplotlibOverlayAdapter(work_dir=tmp_path / "overlays")


@pytest.mark.parametrize("kind,labels,values", [
    ("bar", ["A", "B", "C"], [1.0, 2.0, 3.0]),
    ("line", ["Mon", "Tue"], [4.0, 6.0]),
    ("pie", ["X", "Y"], [30.0, 70.0]),
    ("callout", [], []),
])
def test_render_one_produces_png(tmp_path: Path, kind: str, labels, values) -> None:
    pytest.importorskip("matplotlib")
    adapter = _adapter(tmp_path)
    adapter.work_dir.mkdir(parents=True, exist_ok=True)
    overlay = ShotOverlay(kind=kind, title="Test Title", labels=labels, values=values, caption="units")
    path = adapter._render_one("s01", overlay)
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 1000  # a real image, not an empty file


async def test_run_maps_shot_ids_and_skips_failures(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    adapter = _adapter(tmp_path)
    good = _shot("s01", ShotOverlay(kind="callout", title="Hello"))
    bad = _shot("s02", ShotOverlay(kind="callout", title="Boom"))
    none = _shot("s03", None)

    original = adapter._render_one

    def failing_render(shot_id, overlay):
        if shot_id == "s02":
            raise RuntimeError("draw failed")
        return original(shot_id, overlay)

    monkeypatch.setattr(adapter, "_render_one", failing_render)
    result = await adapter.run(RenderOverlaysRequest(shots=[good, bad, none]))
    assert "s01" in result.images
    assert result.skipped == {"s02": "draw failed"}
    assert "s03" not in result.images


async def test_health_down_without_matplotlib(tmp_path: Path, monkeypatch) -> None:
    import builtins
    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    health = await _adapter(tmp_path).health()
    assert health.status == "down"
    assert "overlays" in (health.reason or "")
