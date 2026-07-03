"""Dashboard worker — claims queued jobs, runs the pipeline, emits events.

Run as a separate process:
    python -m services.dashboard_worker

The worker polls the dashboard queue, claims one job at a time, runs the
appropriate pipeline phase, and writes events + heartbeats back to the
dashboard repository. It never touches the FastAPI server directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import traceback as _traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import AppConfig, load_app_config
from core.models.dashboard import (
    CreateDashboardJobRequest,
    DashboardApprovalKind,
    DashboardApprovalStatus,
    DashboardEventLevel,
    DashboardJobStatus,
    DashboardQueueItem,
)
from services.dashboard_repository import DashboardRepository

logger = logging.getLogger("dashboard_worker")

# Terminal statuses that indicate a job no longer needs a worker.
_TERMINAL = {
    DashboardJobStatus.COMPLETED,
    DashboardJobStatus.FAILED,
    DashboardJobStatus.BLOCKED,
    DashboardJobStatus.CANCELLED,
}


class DashboardWorker:
    HEARTBEAT_INTERVAL = 30  # seconds between worker heartbeats
    POLL_INTERVAL = 2        # seconds to wait when queue is empty
    STALL_THRESHOLD = 120    # seconds before a job is flagged stalled in the UI

    def __init__(
        self,
        repo: DashboardRepository,
        config: AppConfig,
        *,
        worker_id: str | None = None,
    ) -> None:
        self.repo = repo
        self.config = config
        self.worker_id = worker_id or f"worker-{socket.gethostname()}-{os.getpid()}"
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: register → poll → claim → execute → repeat."""
        logger.info("Worker %s starting", self.worker_id)
        self.repo.heartbeat_worker(self.worker_id)

        idle_heartbeat_task = asyncio.create_task(self._idle_heartbeat())
        try:
            while not self._stop_event.is_set():
                action = self.repo.claim_next_action(self.worker_id)
                if action is None:
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(self._stop_event.wait()),
                            timeout=self.POLL_INTERVAL,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue
                await self._run_action(action)
        finally:
            idle_heartbeat_task.cancel()
            await asyncio.gather(idle_heartbeat_task, return_exceptions=True)
            logger.info("Worker %s stopped", self.worker_id)

    def stop(self) -> None:
        """Signal the worker to stop after the current job finishes."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    async def _run_action(self, action: DashboardQueueItem) -> None:
        job_id = action.job_id
        logger.info("Worker %s claimed job %s", self.worker_id, job_id)

        self.repo.update_job_status(
            job_id,
            DashboardJobStatus.RUNNING,
            current_stage="starting",
        )
        self.repo.record_event(job_id, "job_started", f"Worker {self.worker_id} picked up job.")

        # cancel_event is set by the heartbeat loop when the operator requests cancel.
        cancel_event = asyncio.Event()

        pipeline_task = asyncio.create_task(self._execute_pipeline(action))
        heartbeat_task = asyncio.create_task(
            self._job_heartbeat_loop(job_id, cancel_event)
        )
        cancel_watch = asyncio.create_task(cancel_event.wait())

        done, _ = await asyncio.wait(
            {pipeline_task, cancel_watch},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Always stop the heartbeat.
        heartbeat_task.cancel()
        cancel_watch.cancel()
        await asyncio.gather(heartbeat_task, cancel_watch, return_exceptions=True)

        if cancel_watch in done:
            # Operator cancel requested — abort the pipeline.
            pipeline_task.cancel()
            await asyncio.gather(pipeline_task, return_exceptions=True)
            self._handle_cancel(job_id, action.queue_id)
            return

        # Pipeline finished — check for success or failure.
        exc = pipeline_task.exception() if not pipeline_task.cancelled() else None
        if pipeline_task.cancelled() or isinstance(exc, asyncio.CancelledError):
            self._handle_cancel(job_id, action.queue_id)
        elif exc:
            self._handle_failure(job_id, action.queue_id, exc)
        else:
            self.repo.complete_queue_action(action.queue_id)
            logger.info("Job %s completed successfully", job_id)

    async def _execute_pipeline(self, action: DashboardQueueItem) -> None:
        """Dispatch to the right workflow function based on job phase."""
        from core.models.dashboard import DashboardQueueAction as QA

        req = CreateDashboardJobRequest(**action.payload)
        job_id = action.job_id

        if req.phase == "noop":
            await self._run_noop(req, job_id)
        else:
            await self._run_pipeline(req, job_id)

    async def _run_noop(self, req: CreateDashboardJobRequest, job_id: str) -> None:
        from core.workflow import run_noop_job

        self.repo.record_event(job_id, "phase_started", "Running noop pipeline.", stage_name="noop")
        job = await run_noop_job(req.source.url, self.config)
        status = (
            DashboardJobStatus.COMPLETED
            if job.status.value == "completed"
            else DashboardJobStatus.FAILED
        )
        self.repo.update_job_status(job_id, status, completed=True)
        self.repo.record_event(
            job_id,
            "job_completed" if status == DashboardJobStatus.COMPLETED else "job_failed",
            f"Noop pipeline finished: {job.status.value}",
        )

    def _make_stage_hook(self, job_id: str):
        """Return a callback that writes per-stage events to the dashboard repo."""
        def hook(stage_name: str, event_type: str) -> None:
            if event_type == "stage_started":
                self.repo.update_job_status(
                    job_id, DashboardJobStatus.RUNNING, current_stage=stage_name,
                )
            level = (
                DashboardEventLevel.ERROR
                if event_type == "stage_failed"
                else DashboardEventLevel.INFO
            )
            self.repo.record_event(
                job_id,
                event_type,
                f"{stage_name}: {event_type.replace('_', ' ')}",
                level=level,
                stage_name=stage_name,
            )
        return hook

    async def _run_pipeline(self, req: CreateDashboardJobRequest, job_id: str) -> None:
        from core.workflow import RunOptions, run_pipeline_job

        # Build dashboard approval adapters so approval gates use the repo
        # instead of blocking flag-file servers.
        approval_overrides = self._make_approval_overrides(req, job_id)

        phase = req.phase if req.phase != "all" else "all"
        # script_plan/render/assemble resume from artifacts saved by the previous phase.
        resume = phase in ("script_plan", "render", "assemble")
        options = RunOptions(
            phase=phase,
            resume=resume,
            stage_hook=self._make_stage_hook(job_id),
            error_hook=self._make_error_hook(job_id),
        )

        self.repo.record_event(
            job_id, "phase_started", f"Starting pipeline phase={phase}.", stage_name=phase
        )

        job = await run_pipeline_job(
            source_url=req.source.url,
            rights_cleared=req.rights_cleared,
            app_config=self.config,
            options=options,
            approval_overrides=approval_overrides,
        )

        # Map core job status → dashboard status
        if job.status.value == "completed":
            # For the transcribe phase: run the transcript review gate before
            # marking the job COMPLETED. All other phases complete immediately.
            if phase == "transcribe":
                await self._run_transcript_review_gate(req, job_id)
                return  # gate sets final status itself
            dash_status = DashboardJobStatus.COMPLETED
            event_type = "job_completed"
            msg = f"Phase '{phase}' completed successfully."
        elif job.status.value == "blocked":
            dash_status = DashboardJobStatus.BLOCKED
            event_type = "job_blocked"
            msg = "Pipeline blocked: rights not cleared."
        else:
            dash_status = DashboardJobStatus.FAILED
            event_type = "job_failed"
            msg = f"Pipeline finished with status: {job.status.value}"

        self.repo.update_job_status(job_id, dash_status, completed=(dash_status == DashboardJobStatus.COMPLETED))
        self.repo.record_event(job_id, event_type, msg)

        # Record which phase just finished so the stepper can show it as done.
        if dash_status == DashboardJobStatus.COMPLETED:
            self.repo.append_completed_phase(job_id, phase)
            self.repo.record_event(
                job_id, "phase_completed", f"Phase '{phase}' done.", stage_name=phase
            )

    # ------------------------------------------------------------------
    # Transcript review gate
    # ------------------------------------------------------------------

    _TRANSCRIPT_REVIEW_TIMEOUT_HOURS: float = 4.0
    _TRANSCRIPT_REVIEW_MAX_ITERATIONS: int = 3
    _TRANSCRIPT_REVIEW_POLL_INTERVAL: float = 5.0

    async def _run_transcript_review_gate(
        self, req: CreateDashboardJobRequest, job_id: str
    ) -> None:
        """Pause after transcribe, let operator review/refine the transcript, then mark done."""
        transcript = self._load_transcript_artifact(job_id)
        if transcript is None:
            # Artifact missing — just complete without gating.
            logger.warning("Job %s: transcript artifact not found; skipping review gate", job_id)
            self.repo.update_job_status(job_id, DashboardJobStatus.COMPLETED, completed=True)
            self.repo.append_completed_phase(job_id, "transcribe")
            self.repo.record_event(job_id, "phase_completed", "Phase 'transcribe' done.", stage_name="transcribe")
            return

        iteration = 1
        while iteration <= self._TRANSCRIPT_REVIEW_MAX_ITERATIONS:
            approval = self.repo.create_approval_request(
                job_id,
                DashboardApprovalKind.TRANSCRIPT,
                request=self._transcript_to_payload(transcript, iteration),
                iteration=iteration,
            )
            self.repo.update_job_status(
                job_id,
                DashboardJobStatus.PENDING_TRANSCRIPT_REVIEW,
                approval_kind="transcript",
            )
            self.repo.record_event(
                job_id,
                "approval_requested",
                f"Transcript review requested (iteration {iteration}). Open the dashboard to review.",
                payload={"approval_id": approval.approval_id, "iteration": iteration},
            )
            logger.info("Job %s: transcript approval %s (iter %d)", job_id, approval.approval_id, iteration)

            decision = await self._poll_for_approval(job_id, approval.approval_id)
            if decision is None:
                # Timeout
                self.repo.update_job_status(job_id, DashboardJobStatus.FAILED, completed=True)
                self.repo.record_event(
                    job_id, "job_failed",
                    f"Transcript review timed out after {self._TRANSCRIPT_REVIEW_TIMEOUT_HOURS}h.",
                    level=DashboardEventLevel.ERROR,
                )
                return

            approved, notes = decision
            self.repo.update_job_status(job_id, DashboardJobStatus.RUNNING)

            if approved:
                # Mark phase done.
                self.repo.update_job_status(job_id, DashboardJobStatus.COMPLETED, completed=True)
                self.repo.append_completed_phase(job_id, "transcribe")
                self.repo.record_event(
                    job_id, "phase_completed",
                    f"Transcript approved (iteration {iteration}). Phase 'transcribe' done.",
                    stage_name="transcribe",
                )
                return

            # Rejected — refine transcript with LLM.
            if iteration >= self._TRANSCRIPT_REVIEW_MAX_ITERATIONS:
                self.repo.update_job_status(job_id, DashboardJobStatus.FAILED, completed=True)
                self.repo.record_event(
                    job_id, "job_failed",
                    "Transcript review exhausted (3 iterations). Restart the job.",
                    level=DashboardEventLevel.ERROR,
                )
                return

            self.repo.record_event(
                job_id, "transcript_refine_started",
                f"Refining transcript with LLM (iteration {iteration}): {notes}",
            )
            transcript = await self._refine_transcript(transcript, notes, job_id)
            self._save_transcript_artifact(job_id, transcript)
            iteration += 1

        # Should not reach here, but fail safely.
        self.repo.update_job_status(job_id, DashboardJobStatus.FAILED, completed=True)
        self.repo.record_event(job_id, "job_failed", "Transcript review loop exited unexpectedly.",
                               level=DashboardEventLevel.ERROR)

    async def _poll_for_approval(
        self, job_id: str, approval_id: str
    ) -> tuple[bool, str] | None:
        """Poll until operator approves/rejects. Returns (approved, notes) or None on timeout."""
        deadline = datetime.now(timezone.utc) + timedelta(hours=self._TRANSCRIPT_REVIEW_TIMEOUT_HOURS)
        while datetime.now(timezone.utc) < deadline:
            await asyncio.sleep(self._TRANSCRIPT_REVIEW_POLL_INTERVAL)
            current = self.repo.get_pending_approval(job_id)
            if current is None or current.approval_id != approval_id:
                continue
            if current.status == DashboardApprovalStatus.APPROVED:
                return (True, "")
            if current.status == DashboardApprovalStatus.REJECTED:
                notes = (current.response or {}).get("notes", "") if current.response else ""
                return (False, notes)
        return None

    def _load_transcript_artifact(self, job_id: str) -> dict | None:
        artifact_path = Path(self.config.settings.data_dir) / job_id / "transcribe.json"
        if not artifact_path.exists():
            return None
        try:
            return json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Job %s: failed to read transcript artifact: %s", job_id, exc)
            return None

    def _save_transcript_artifact(self, job_id: str, transcript: dict) -> None:
        artifact_path = Path(self.config.settings.data_dir) / job_id / "transcribe.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(transcript, indent=2, default=str), encoding="utf-8")

    def _transcript_to_payload(self, transcript: dict, iteration: int) -> dict:
        """Build the approval request payload shown in the UI."""
        segments = transcript.get("segments", [])
        return {
            "iteration": iteration,
            "language": transcript.get("language", ""),
            "duration": transcript.get("duration", 0),
            "segments": [
                {
                    "start": round(s.get("start", 0), 2),
                    "end": round(s.get("end", 0), 2),
                    "text": s.get("text", "").strip(),
                }
                for s in segments
            ],
            "full_text": " ".join(s.get("text", "").strip() for s in segments),
        }

    async def _refine_transcript(self, transcript: dict, notes: str, job_id: str) -> dict:
        from adapters.transcript_refine.llm_adapter import LlmTranscriptRefineAdapter

        s = self.config.settings
        refiner = LlmTranscriptRefineAdapter(
            base_url=s.llm_base_url,
            model=s.llm_model,
        )
        try:
            refined = await refiner.refine(transcript, notes)
            self.repo.record_event(job_id, "transcript_refine_completed", "Transcript refined by LLM.")
            return refined
        except Exception as exc:
            logger.exception("Job %s: transcript refinement failed: %s", job_id, exc)
            self.repo.record_event(
                job_id, "transcript_refine_failed",
                f"LLM refinement failed: {exc} — using previous version.",
                level=DashboardEventLevel.ERROR,
            )
            return transcript

    def _make_approval_overrides(
        self, req: CreateDashboardJobRequest, job_id: str
    ) -> dict[str, Any]:
        """Return approval adapter overrides that use the dashboard repo."""
        from adapters.approval.dashboard_approval_adapter import DashboardPlanApprovalAdapter
        from adapters.approval.dashboard_image_approval_adapter import (
            DashboardImageApprovalAdapter,
        )

        return {
            "approval": DashboardPlanApprovalAdapter(self.repo, job_id),
            "image_approval": DashboardImageApprovalAdapter(self.repo, job_id),
        }

    # ------------------------------------------------------------------
    # Heartbeat loops
    # ------------------------------------------------------------------

    async def _job_heartbeat_loop(
        self, job_id: str, cancel_event: asyncio.Event
    ) -> None:
        """Send heartbeats while a job runs; set cancel_event when cancel is requested."""
        while True:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            self.repo.heartbeat_worker(self.worker_id, current_job_id=job_id)
            self.repo.heartbeat_job(job_id)

            # Check for operator-requested cancellation.
            current = self.repo.get_job(job_id)
            if current and current.status == DashboardJobStatus.CANCEL_REQUESTED:
                logger.info("Cancel requested for job %s — signalling stop.", job_id)
                cancel_event.set()
                return

    async def _idle_heartbeat(self) -> None:
        """Send periodic heartbeats when no job is running."""
        while True:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            self.repo.heartbeat_worker(self.worker_id)

    # ------------------------------------------------------------------
    # Terminal state helpers
    # ------------------------------------------------------------------

    def _handle_cancel(self, job_id: str, queue_id: str) -> None:
        self.repo.update_job_status(job_id, DashboardJobStatus.CANCELLED, completed=True)
        self.repo.record_event(job_id, "job_cancelled", "Job cancelled by operator.")
        self.repo.fail_queue_action(queue_id, {"code": "CANCELLED", "message": "Operator cancelled."})
        logger.info("Job %s cancelled.", job_id)

    def _make_error_hook(self, job_id: str):
        """Return an async callback that records a stage_failed event with full traceback."""
        async def hook(stage_name: str, exc: Exception) -> None:
            self.repo.record_event(
                job_id,
                "stage_failed",
                f"{type(exc).__name__}: {exc}",
                level=DashboardEventLevel.ERROR,
                stage_name=stage_name,
                payload={
                    "code": type(exc).__name__,
                    "message": str(exc),
                    "traceback": _traceback.format_exc(),
                },
            )
        return hook

    def _handle_failure(self, job_id: str, queue_id: str, exc: Exception) -> None:
        error: dict[str, Any] = {
            "code": type(exc).__name__,
            "message": str(exc),
            "traceback": _traceback.format_exc(),
        }
        self.repo.update_job_status(
            job_id,
            DashboardJobStatus.FAILED,
            terminal_error=error,
            completed=True,
        )
        self.repo.record_event(
            job_id,
            "job_failed",
            f"{type(exc).__name__}: {exc}",
            level=DashboardEventLevel.ERROR,
            payload=error,
        )
        self.repo.fail_queue_action(queue_id, error)
        logger.exception("Job %s failed: %s", job_id, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _make_worker() -> tuple[DashboardWorker, DashboardRepository]:
    config = load_app_config()
    repo = DashboardRepository(Path(config.settings.sqlite_path))
    worker = DashboardWorker(repo, config)
    return worker, repo


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    worker, _ = _make_worker()
    logger.info("Dashboard worker starting — ID: %s", worker.worker_id)

    loop = asyncio.get_event_loop()

    def _on_signal() -> None:
        logger.info("Shutdown signal received, stopping after current job …")
        worker.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    await worker.run()


if __name__ == "__main__":
    asyncio.run(_main())
