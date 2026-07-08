"""Dashboard-backed image approval adapter.

Replaces ImageApprovalAdapter when a job runs through the dashboard worker.
Writes an approval request to the repo and polls until the operator confirms
or overrides the VLM-selected image candidates in the browser.

The workflow contract is identical to ImageApprovalAdapter:
    result = await adapter.run(ImageApprovalRequest(...))
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from core.models.capabilities import ImageApprovalRequest, ImageApprovalResult
from core.models.dashboard import (
    DashboardApprovalStatus,
    DashboardJobStatus,
)

if TYPE_CHECKING:
    from services.dashboard_repository import DashboardRepository

logger = logging.getLogger(__name__)

_IMAGE_APPROVAL_KIND = "images"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_request_payload(req: ImageApprovalRequest) -> dict[str, Any]:
    """Serialise candidate images + VLM picks for the approval UI."""
    shots_data = []
    for shot, critique in zip(req.shots, req.critique_results):
        shot_entry: dict[str, Any] = {
            "shot_id": getattr(shot, "shot_id", ""),
            "setting": getattr(shot, "setting", ""),
            "action": getattr(shot, "action", ""),
            "candidate_uris": critique.candidate_uris,
            "vlm_winner_index": critique.winner_index,
            "vlm_winner_uri": critique.winner_uri,
            "overall_reasoning": critique.overall_reasoning,
            "origin": getattr(critique, "origin", "vlm"),
            "candidate_scores": [
                {
                    "candidate_index": s.candidate_index,
                    "scores": s.scores,
                    "reasoning": s.reasoning,
                }
                for s in (critique.candidate_scores or [])
            ],
        }
        shots_data.append(shot_entry)

    return {"cast_id": req.cast_id, "shots": shots_data}


class DashboardImageApprovalAdapter:
    """Dashboard-backed drop-in for ImageApprovalAdapter."""

    def __init__(
        self,
        repo: "DashboardRepository",
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout_hours: float = 4.0,
        auto_approve: bool = False,
    ) -> None:
        self._repo = repo
        self._job_id = job_id
        self._poll_interval = poll_interval
        self._timeout_hours = timeout_hours
        self._auto_approve = auto_approve

    async def run(self, req: ImageApprovalRequest) -> ImageApprovalResult:
        """Write image approval request to repo; return operator-confirmed URIs."""
        from core.models.dashboard import DashboardApprovalKind

        if self._auto_approve:
            # Operator disabled this gate at job creation (unattended run) —
            # accept the VLM/single-candidate picks as-is.
            self._repo.record_event(
                self._job_id,
                "approval_granted",
                f"Images auto-approved for {len(req.shots)} shots "
                "(approval gate disabled for this job).",
            )
            logger.info(
                "Job %s: images auto-approved (gate disabled, %d shots)",
                self._job_id, len(req.shots),
            )
            return ImageApprovalResult(
                approved_uris=[r.winner_uri for r in req.critique_results],
                overrides={},
            )

        payload = _build_request_payload(req)
        approval = self._repo.create_approval_request(
            self._job_id,
            DashboardApprovalKind.IMAGES,
            request=payload,
        )
        self._repo.update_job_status(
            self._job_id,
            DashboardJobStatus.PENDING_IMAGE_APPROVAL,
            approval_kind=_IMAGE_APPROVAL_KIND,
        )
        self._repo.record_event(
            self._job_id,
            "approval_requested",
            f"Image approval requested for {len(req.shots)} shots. "
            "Open the dashboard to review candidates.",
            payload={"approval_id": approval.approval_id},
        )
        logger.info(
            "Job %s: image approval %s requested (%d shots)",
            self._job_id,
            approval.approval_id,
            len(req.shots),
        )

        # Build VLM-default approved URIs (fallback if operator doesn't override).
        default_uris = [r.winner_uri for r in req.critique_results]

        deadline = _utc_now() + timedelta(hours=self._timeout_hours)
        while _utc_now() < deadline:
            await asyncio.sleep(self._poll_interval)

            current = self._repo.get_approval(approval.approval_id)
            if current is None:
                continue

            if current.status == DashboardApprovalStatus.APPROVED:
                self._repo.update_job_status(self._job_id, DashboardJobStatus.RUNNING)
                self._repo.record_event(
                    self._job_id, "approval_granted", "Image candidates approved."
                )
                overrides: dict[str, int] = {}
                approved_uris = list(default_uris)

                if current.response:
                    raw_picks: dict[str, Any] = current.response.get("picks", {})
                    for shot_idx, (shot, critique) in enumerate(
                        zip(req.shots, req.critique_results)
                    ):
                        shot_id = getattr(shot, "shot_id", str(shot_idx))
                        if shot_id in raw_picks:
                            chosen_idx = int(raw_picks[shot_id])
                            uris = critique.candidate_uris or []
                            if 0 <= chosen_idx < len(uris):
                                approved_uris[shot_idx] = uris[chosen_idx]
                                if chosen_idx != critique.winner_index:
                                    overrides[shot_id] = chosen_idx

                return ImageApprovalResult(approved_uris=approved_uris, overrides=overrides)

            if current.status == DashboardApprovalStatus.REJECTED:
                # Image approval rejection is treated as a fatal failure — the
                # pipeline doesn't re-render from scratch on image rejection.
                notes = ""
                if current.response:
                    notes = current.response.get("notes", "")
                self._repo.update_job_status(
                    self._job_id,
                    DashboardJobStatus.FAILED,
                    terminal_error={"code": "IMAGE_APPROVAL_REJECTED", "message": notes},
                    completed=True,
                )
                self._repo.record_event(
                    self._job_id,
                    "approval_rejected",
                    f"Image candidates rejected: {notes}",
                )
                raise RuntimeError(f"Image approval rejected by operator: {notes}")

        # Timeout.
        self._repo.record_event(
            self._job_id,
            "approval_timeout",
            f"Image approval timed out after {self._timeout_hours}h.",
        )
        raise TimeoutError(
            f"Image approval timed out after {self._timeout_hours}h for job {self._job_id}"
        )
