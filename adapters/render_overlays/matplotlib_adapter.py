"""render_overlays adapter: draw ShotOverlay specs to transparent PNG panels.

Deterministic matplotlib rendering (Agg backend, CPU-only, no network). Each
panel sits on a semi-opaque white rounded card so it stays readable over any
video content. Output: ``{work_dir}/{shot_id}.png`` — the workflow constructs
the adapter with ``work_dir={job_work_dir}/overlays`` (under data_dir so the
dashboard's /img route can serve previews at the plan approval gate).

matplotlib is an optional extra (``pip install -e '.[overlays]'``): ``health()``
reports down when it's missing and the workflow skips overlays best-effort.
"""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from core.capabilities.base import RenderOverlays
from core.models.capabilities import RenderOverlaysRequest, RenderOverlaysResult
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

# Bright, friendly palette — readable for kids content and neutral enough for
# explainer casts.
_PALETTE = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6C5CE7", "#51CF66", "#FF922B"]
_PANEL_FACE = "#FFFFFF"
_PANEL_ALPHA = 0.92
_TEXT_COLOR = "#2D3436"


class MatplotlibOverlayAdapter(RenderOverlays):
    """Draw chart/callout overlay panels with matplotlib."""

    version = "1.0.0"

    def __init__(
        self,
        work_dir: Path,
        width_px: int = 1000,
        height_px: int = 600,
        dpi: int = 100,
    ) -> None:
        self.work_dir = work_dir
        self._width_px = width_px
        self._height_px = height_px
        self._dpi = dpi

    async def health(self) -> HealthStatus:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            return HealthStatus(
                status="down",
                reason="matplotlib not installed. Run: pip install -e '.[overlays]'",
            )
        return HealthStatus(status="ok")

    async def estimate_cost(self, req: RenderOverlaysRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local matplotlib rendering; CPU only.")

    async def run(self, req: RenderOverlaysRequest) -> RenderOverlaysResult:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        result = RenderOverlaysResult()

        for shot in req.shots:
            overlay = getattr(shot, "overlay", None)
            shot_id = getattr(shot, "shot_id", "")
            if overlay is None or not shot_id:
                continue
            try:
                path = self._render_one(shot_id, overlay)
                result.images[shot_id] = str(path)
            except Exception as exc:  # one bad figure never blocks the others
                logger.warning("Overlay render failed for %s: %s", shot_id, exc)
                result.skipped[shot_id] = str(exc)

        log_event(
            logger,
            "render_overlays_completed",
            rendered=len(result.images),
            skipped=len(result.skipped),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers (unit-testable without asyncio)
    # ------------------------------------------------------------------

    def _render_one(self, shot_id: str, overlay) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        fig = plt.figure(
            figsize=(self._width_px / self._dpi, self._height_px / self._dpi),
            dpi=self._dpi,
        )
        try:
            # Semi-opaque rounded card behind everything (figure coords).
            card = FancyBboxPatch(
                (0.02, 0.03), 0.96, 0.94,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                transform=fig.transFigure,
                facecolor=_PANEL_FACE, alpha=_PANEL_ALPHA,
                edgecolor="#DDDDDD", linewidth=1.5, zorder=0,
            )
            fig.patches.append(card)

            if overlay.kind == "callout":
                self._draw_callout(fig, overlay)
            else:
                ax = fig.add_axes((0.12, 0.18, 0.8, 0.6), zorder=1)
                ax.patch.set_alpha(0)
                if overlay.kind == "bar":
                    self._draw_bar(ax, overlay)
                elif overlay.kind == "line":
                    self._draw_line(ax, overlay)
                elif overlay.kind == "pie":
                    self._draw_pie(ax, overlay)
                fig.suptitle(
                    overlay.title, fontsize=34, fontweight="bold",
                    color=_TEXT_COLOR, y=0.93,
                )
                if overlay.caption:
                    fig.text(0.95, 0.05, overlay.caption, fontsize=18,
                             style="italic", color=_TEXT_COLOR, ha="right")

            path = self.work_dir / f"{shot_id}.png"
            fig.savefig(path, transparent=True)
            return path
        finally:
            plt.close(fig)

    def _draw_bar(self, ax, overlay) -> None:
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(overlay.labels))]
        bars = ax.bar(overlay.labels, overlay.values, color=colors, width=0.6)
        for bar, value in zip(bars, overlay.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:g}", ha="center", va="bottom",
                    fontsize=22, fontweight="bold", color=_TEXT_COLOR)
        ax.tick_params(labelsize=22, colors=_TEXT_COLOR)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks([])

    def _draw_line(self, ax, overlay) -> None:
        ax.plot(overlay.labels, overlay.values, linewidth=5,
                color=_PALETTE[3], marker="o", markersize=14,
                markerfacecolor=_PALETTE[0])
        for x, y in zip(overlay.labels, overlay.values):
            ax.annotate(f"{y:g}", (x, y), textcoords="offset points",
                        xytext=(0, 12), ha="center",
                        fontsize=20, fontweight="bold", color=_TEXT_COLOR)
        ax.tick_params(labelsize=22, colors=_TEXT_COLOR)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks([])

    def _draw_pie(self, ax, overlay) -> None:
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(overlay.labels))]
        ax.pie(
            overlay.values, labels=overlay.labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 20, "color": _TEXT_COLOR},
        )

    def _draw_callout(self, fig, overlay) -> None:
        wrapped = textwrap.fill(overlay.title, width=24)
        fig.text(0.5, 0.55, wrapped, fontsize=44, fontweight="bold",
                 color=_TEXT_COLOR, ha="center", va="center")
        if overlay.caption:
            fig.text(0.5, 0.2, textwrap.fill(overlay.caption, width=40),
                     fontsize=22, color=_TEXT_COLOR, ha="center", va="center")
