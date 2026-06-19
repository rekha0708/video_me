import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from core.capabilities.base import Publish
from core.models.capabilities import PublishRequest, PublishResult
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

# Fields that must appear in every sidecar — checked by downstream review tooling.
_REQUIRED_SIDECAR_FIELDS = frozenset(
    {
        "status",
        "published_at_utc",
        "rights_cleared",
        "made_for_kids",
        "disclosure_label_required",
        "learning_objective_summary",
        "source_video_uri",
        "source_video_duration_sec",
    }
)


class ManualPublishAdapter(Publish):
    """
    publish adapter: writes the final MP4 + a metadata sidecar to a review
    folder for human sign-off before any live upload.

    No live publishing happens here.  Output structure::

        review_dir/
          {timestamp}_{video_stem}/
            video.mp4       ← copy of the assembled final
            metadata.json   ← sidecar with rights, flags, learning objective

    The adapter refuses to run if ``req.rights_cleared`` is False — this is
    a second line of defence after the pipeline gate in ``core/executor.py``.

    Args:
        review_dir:  Root directory for pending-review packages.
                     Default: ``Path("review")``.
    """

    version = "1.0.0"

    def __init__(self, review_dir: Path = Path("review")) -> None:
        self.review_dir = review_dir

    async def health(self) -> HealthStatus:
        try:
            self.review_dir.mkdir(parents=True, exist_ok=True)
            probe = self.review_dir / ".health"
            probe.touch()
            probe.unlink()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(
                status="down",
                reason=f"review_dir not writable ({self.review_dir}): {exc}",
            )

    async def estimate_cost(self, req: PublishRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local file copy; no cost.")

    async def run(self, req: PublishRequest) -> PublishResult:
        if not req.rights_cleared:
            raise RuntimeError(
                "Refusing to publish: rights_cleared is False. "
                "This job should have been blocked at the adapt_script stage "
                "by core/executor.py:check_rights()."
            )

        out_dir = self._make_output_dir(req)
        out_dir.mkdir(parents=True, exist_ok=True)

        video_dest = self._copy_video(req.video.uri, out_dir)
        metadata = self._build_metadata(req, video_dest)
        sidecar_path = self._write_sidecar(metadata, out_dir)

        log_event(
            logger,
            "publish_completed",
            review_path=str(video_dest),
            metadata_path=str(sidecar_path),
            made_for_kids=req.made_for_kids,
            disclosure_required=req.disclosure_label_required,
        )

        return PublishResult(
            review_path=str(video_dest),
            metadata_path=str(sidecar_path),
            status="pending_review",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_output_dir(self, req: PublishRequest) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = Path(req.video.uri).stem
        return self.review_dir / f"{timestamp}_{stem}"

    def _copy_video(self, source_uri: str, out_dir: Path) -> Path:
        """Copy the assembled MP4 to the review dir; return its new path."""
        source = Path(source_uri)
        if not source.exists():
            raise FileNotFoundError(
                f"Final video not found: {source}. "
                "assemble_video must complete before publish."
            )
        dest = out_dir / "video.mp4"
        shutil.copy2(source, dest)
        return dest

    def _build_metadata(self, req: PublishRequest, video_dest: Path) -> dict:
        return {
            "status": "pending_review",
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "rights_cleared": req.rights_cleared,
            "made_for_kids": req.made_for_kids,
            "disclosure_label_required": req.disclosure_label_required,
            "learning_objective_summary": req.learning_objective_summary,
            "source_video_uri": req.video.uri,
            "source_video_duration_sec": req.video.duration_sec,
            "review_video_path": str(video_dest),
        }

    def _write_sidecar(self, metadata: dict, out_dir: Path) -> Path:
        path = out_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return path
