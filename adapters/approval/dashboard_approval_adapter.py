"""Dashboard-backed plan approval adapter.

Replaces WebApprovalAdapter when a job runs through the dashboard worker.
Instead of starting an ephemeral local web server and polling a flag file,
this adapter writes an approval request to the dashboard repository and polls
it every few seconds until the operator approves or rejects in the browser.

The workflow contract is identical to WebApprovalAdapter:
    approved, rejection_notes = await adapter.request_approval(storyboard, ...)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from core.models.dashboard import (
    DashboardApprovalStatus,
    DashboardJobStatus,
)

if TYPE_CHECKING:
    from services.dashboard_repository import DashboardRepository

logger = logging.getLogger(__name__)

_PLAN_APPROVAL_KIND = "plan"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _storyboard_to_payload(storyboard: Any, critique: Any | None) -> dict[str, Any]:
    """Serialise the storyboard + critique scores for the approval UI."""
    shots = []
    for shot in storyboard.shots:
        entry: dict[str, Any] = {
            "shot_id": shot.shot_id,
            "setting": shot.setting,
            "action": shot.action,
            "character_ids": shot.characters_on_screen,
            "duration_sec": shot.duration_sec,
        }
        shots.append(entry)

    payload: dict[str, Any] = {"shots": shots}

    if critique is not None:
        try:
            payload["scores"] = {
                "character_fit": critique.character_fit,
                "scene_achievability": critique.scene_achievability,
                "pacing": critique.pacing,
                "kids_safety": critique.kids_safety,
                "visual_clarity": critique.visual_clarity,
                "overall": critique.overall,
            }
            payload["passed"] = critique.passed
            payload["fix_notes"] = critique.fix_notes
        except AttributeError:
            pass

    return payload


class DashboardPlanApprovalAdapter:
    """Dashboard-backed drop-in for WebApprovalAdapter."""

    def __init__(
        self,
        repo: "DashboardRepository",
        job_id: str,
        *,
        poll_interval: float = 5.0,
        timeout_hours: float = 4.0,
    ) -> None:
        self._repo = repo
        self._job_id = job_id
        self._poll_interval = poll_interval
        self._timeout_hours = timeout_hours

    async def request_approval(
        self,
        storyboard: Any,
        script: Any,
        cast: Any,
        critique: Any | None,
        iteration: int = 1,
    ) -> tuple[bool, str]:
        """Write an approval request to the repo and wait for the operator's decision."""
        from core.models.dashboard import DashboardApprovalKind

        payload = _storyboard_to_payload(storyboard, critique)
        payload["iteration"] = iteration

        approval = self._repo.create_approval_request(
            self._job_id,
            DashboardApprovalKind.PLAN,
            request=payload,
            iteration=iteration,
        )
        self._repo.update_job_status(
            self._job_id,
            DashboardJobStatus.PENDING_PLAN_APPROVAL,
            approval_kind=_PLAN_APPROVAL_KIND,
        )
        self._repo.record_event(
            self._job_id,
            "approval_requested",
            f"Plan approval requested (iteration {iteration}). "
            "Open the dashboard to review the storyboard.",
            payload={"approval_id": approval.approval_id, "iteration": iteration},
        )
        logger.info(
            "Job %s: plan approval %s requested (iteration %d)",
            self._job_id,
            approval.approval_id,
            iteration,
        )

        deadline = _utc_now() + timedelta(hours=self._timeout_hours)
        while _utc_now() < deadline:
            await asyncio.sleep(self._poll_interval)

            current = self._repo.get_approval(approval.approval_id)
            if current is None:
                # Approval was removed (shouldn't happen in normal flow).
                continue

            if current.status == DashboardApprovalStatus.APPROVED:
                self._repo.update_job_status(self._job_id, DashboardJobStatus.RUNNING)
                self._repo.record_event(
                    self._job_id,
                    "approval_granted",
                    f"Plan approved (iteration {iteration}).",
                )
                return (True, "")

            if current.status == DashboardApprovalStatus.REJECTED:
                notes: str = ""
                if current.response:
                    notes = current.response.get("notes", "")
                self._repo.update_job_status(self._job_id, DashboardJobStatus.RUNNING)
                self._repo.record_event(
                    self._job_id,
                    "approval_rejected",
                    f"Plan rejected (iteration {iteration}): {notes}",
                )
                return (False, notes)

        # Timeout — fail the job.
        self._repo.record_event(
            self._job_id,
            "approval_timeout",
            f"Plan approval timed out after {self._timeout_hours}h.",
            level="error",  # type: ignore[arg-type]
        )
        raise TimeoutError(
            f"Plan approval timed out after {self._timeout_hours}h for job {self._job_id}"
        )
